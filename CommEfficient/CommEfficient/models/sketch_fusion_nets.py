"""
Sketch-based feature fusion models for federated multimodal learning.

All three models share the same per-modality extraction and refinement
pipeline from MultiModalNet (FeaExtractor → FeaRefiner), but replace
the **concatenation** fusion step with a sketch-based mechanism.
The sketch serves as both the feature fusion tool and (in the broader
FetchSGD pipeline) a communication-aware representation.

Model A — SketchFusionA  (Learnable Sketch)
    Each modality's refined features are projected into a shared space
    via *learned* linear mappings and fused by addition.  The learnable
    projections are the differentiable analogue of the random hash
    functions in a Count Sketch.

Model B — SketchFusionB  (Fixed Sketch + Nonlinear Head)
    Uses *fixed* random Count Sketch hash functions (non-learnable
    buffers) to map each modality's features into a shared table.
    The sketch operation is differentiable (via scatter_add), so
    gradients flow through it.  A nonlinear MLP head learns to decode
    the fused sketch table.

Model C — SketchFusionC  (Tensor Sketch)
    Approximates the outer product of the two modality representations
    via Tensor Sketch (Pham & Pagh, 2013): element-wise FFT product of
    per-modality Count Sketches.  Captures *multiplicative* cross-modal
    interactions, not just additive ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_net import FeaExtractor, FeaRefiner, CrossModalPredictor

__all__ = [
    "SketchFusionA",
    "SketchFusionB",
    "SketchFusionB4",
    "SketchFusionBLSTM",
    "SketchFusionC",
]

SKETCH_HASH_SEED = 2147483647


# =====================================================================
# Shared extraction / refinement / missing-modality base
# =====================================================================

class _SketchFusionBase(nn.Module):
    """Common front-end shared by all three sketch-fusion variants."""

    def _build_front_end(self, img_dim, txt_dim, feat_dim, dropout):
        self.img_extractor = FeaExtractor(img_dim, feat_dim, dropout)
        self.txt_extractor = FeaExtractor(txt_dim, feat_dim, dropout)
        self.img_refiner = FeaRefiner(feat_dim)
        self.txt_refiner = FeaRefiner(feat_dim)
        self.img_to_txt = CrossModalPredictor(feat_dim)
        self.txt_to_img = CrossModalPredictor(feat_dim)
        self._missing_loss = torch.tensor(0.0)

    def _extract_and_refine(self, img_feat, txt_feat, missing_prob=0.0):
        """Run extractors, cross-modal predictors, refiners."""
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)

        f_txt_hat = self.img_to_txt(f_img)
        f_img_hat = self.txt_to_img(f_txt)
        self._missing_loss = (
            F.mse_loss(f_txt_hat, f_txt.detach()) +
            F.mse_loss(f_img_hat, f_img.detach())
        )

        if self.training and missing_prob > 0:
            r = torch.rand(1).item()
            if r < missing_prob / 2:
                f_img = f_img_hat
            elif r < missing_prob:
                f_txt = f_txt_hat

        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        return f_img * c_img, f_txt * c_txt

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# =====================================================================
# Model A — Learnable Sketch Fusion
# =====================================================================

class SketchFusionA(_SketchFusionBase):
    """
    Learnable sketch fusion.

    Each modality's refined features (feat_dim) are projected into a
    shared sketch space of dimension ``sketch_r * sketch_c`` via
    separate *learned* linear layers, then fused by element-wise
    addition — the core additive property of a Count Sketch, but with
    learnable hash/sign functions.

    Pipeline (differences from MultiModalNet marked with ★):
      1. FeaExtractor + FeaRefiner  → f'_img, f'_txt   (identical)
      2. ★ Learned projection       → table_img, table_txt
      3. ★ Additive fusion          → table = table_img + table_txt
      4. Integrator MLP             → H
      5. Classifier                 → logits
    """

    def __init__(self, img_dim=4096, txt_dim=300, feat_dim=512,
                 num_classes=10, dropout=0.5,
                 sketch_r=4, sketch_c=128, **kwargs):
        super().__init__()
        self._build_front_end(img_dim, txt_dim, feat_dim, dropout)

        table_dim = sketch_r * sketch_c
        self.img_project = nn.Linear(feat_dim, table_dim, bias=False)
        self.txt_project = nn.Linear(feat_dim, table_dim, bias=False)

        self.integrator = nn.Sequential(
            nn.Linear(table_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _fuse(self, f_img_refined, f_txt_refined):
        return self.img_project(f_img_refined) + self.txt_project(f_txt_refined)

    def extract_fused(self, img_feat, txt_feat):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        fused = self._fuse(f_img * c_img, f_txt * c_txt)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img_r, f_txt_r = self._extract_and_refine(
            img_feat, txt_feat, missing_prob)
        fused = self._fuse(f_img_r, f_txt_r)
        H = self.integrator(fused)
        return self.classifier(H), H


# =====================================================================
# Model B — Fixed Count Sketch + Nonlinear Head
# =====================================================================

class SketchFusionB(_SketchFusionBase):
    """
    Fixed (random) Count Sketch fusion with a learned MLP head.

    The hash functions and signs are fixed random buffers (seeded
    deterministically).  The sketch operation is differentiable via
    ``scatter_add_``, so gradients flow through it to the extractors
    and refiners — only the hash structure itself is non-learnable.

    Pipeline (differences from MultiModalNet marked with ★):
      1. FeaExtractor + FeaRefiner  → f'_img, f'_txt   (identical)
      2. ★ Fixed Count Sketch       → table_img, table_txt
      3. ★ Additive fusion          → table = table_img + table_txt
      4. ★ Integrator MLP           → H  (decodes the sketch table)
      5. Classifier                 → logits
    """

    def __init__(self, img_dim=4096, txt_dim=300, feat_dim=512,
                 num_classes=10, dropout=0.5,
                 sketch_r=4, sketch_c=128, **kwargs):
        super().__init__()
        self._build_front_end(img_dim, txt_dim, feat_dim, dropout)

        self.sketch_r = sketch_r
        self.sketch_c = sketch_c
        table_dim = sketch_r * sketch_c

        rng = torch.Generator().manual_seed(SKETCH_HASH_SEED)
        self.register_buffer(
            "buckets",
            torch.randint(0, sketch_c, (sketch_r, feat_dim),
                          generator=rng),
        )
        self.register_buffer(
            "signs",
            (torch.randint(0, 2, (sketch_r, feat_dim),
                           generator=rng) * 2 - 1).float(),
        )

        self.integrator = nn.Sequential(
            nn.Linear(table_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _sketch_features(self, x):
        """Differentiable Count Sketch of a batch of feature vectors."""
        B = x.size(0)
        tables = []
        for r in range(self.sketch_r):
            signed = self.signs[r] * x                        # (B, feat_dim)
            row = torch.zeros(B, self.sketch_c, device=x.device)
            idx = self.buckets[r].unsqueeze(0).expand(B, -1)  # (B, feat_dim)
            row.scatter_add_(1, idx, signed)
            tables.append(row)
        return torch.cat(tables, dim=-1)                      # (B, r*c)

    def _fuse(self, f_img_refined, f_txt_refined):
        return (self._sketch_features(f_img_refined)
                + self._sketch_features(f_txt_refined))

    def extract_fused(self, img_feat, txt_feat):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        fused = self._fuse(f_img * c_img, f_txt * c_txt)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img_r, f_txt_r = self._extract_and_refine(
            img_feat, txt_feat, missing_prob)
        fused = self._fuse(f_img_r, f_txt_r)
        H = self.integrator(fused)
        return self.classifier(H), H


class SketchFusionB4(nn.Module):
    """
    SketchFusionB extended to **four** modalities (fixed Count Sketch + MLP head).

    Each modality has its own FeaExtractor + FeaRefiner; refined vectors are
    sketched with shared hash buffers and **summed** before the integrator.
    No cross-modal prediction loss (compatible with ``missing_loss_weight=0``).
    """

    def __init__(
        self,
        mod_dims,
        feat_dim=512,
        num_classes=10,
        dropout=0.5,
        sketch_r=4,
        sketch_c=128,
        **kwargs,
    ):
        super().__init__()
        if len(mod_dims) != 4:
            raise ValueError(f"SketchFusionB4 expects 4 mod_dims, got {len(mod_dims)}")
        self.mod_dims = tuple(int(d) for d in mod_dims)
        self.extractors = nn.ModuleList(
            [FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims]
        )
        self.refiners = nn.ModuleList([FeaRefiner(feat_dim) for _ in range(4)])
        self._missing_loss = torch.tensor(0.0)

        self.sketch_r = sketch_r
        self.sketch_c = sketch_c
        table_dim = sketch_r * sketch_c

        rng = torch.Generator().manual_seed(SKETCH_HASH_SEED)
        self.register_buffer(
            "buckets",
            torch.randint(0, sketch_c, (sketch_r, feat_dim),
                          generator=rng),
        )
        self.register_buffer(
            "signs",
            (torch.randint(0, 2, (sketch_r, feat_dim),
                           generator=rng) * 2 - 1).float(),
        )

        self.integrator = nn.Sequential(
            nn.Linear(table_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights_quad()

    def _init_weights_quad(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sketch_features(self, x):
        B = x.size(0)
        tables = []
        for r in range(self.sketch_r):
            signed = self.signs[r] * x
            row = torch.zeros(B, self.sketch_c, device=x.device)
            idx = self.buckets[r].unsqueeze(0).expand(B, -1)
            row.scatter_add_(1, idx, signed)
            tables.append(row)
        return torch.cat(tables, dim=-1)

    def _fuse_refined(self, refined):
        total = self._sketch_features(refined[0])
        for r in refined[1:]:
            total = total + self._sketch_features(r)
        return total

    def extract_fused(self, m0, m1, m2, m3):
        refined = []
        for i, x in enumerate((m0, m1, m2, m3)):
            f = self.extractors[i](x)
            c = self.refiners[i](f)
            refined.append(f * c)
        fused = self._fuse_refined(refined)
        return self.integrator(fused)

    def forward(self, m0, m1, m2, m3, missing_prob=0.0):
        del missing_prob  # reserved; no stochastic masking implemented
        refined = []
        for i, x in enumerate((m0, m1, m2, m3)):
            f = self.extractors[i](x)
            c = self.refiners[i](f)
            refined.append(f * c)
        fused = self._fuse_refined(refined)
        H = self.integrator(fused)
        return self.classifier(H), H


# =====================================================================
# SketchFusionB + MFedMC-style LSTM encoders (time × features per modality)
# =====================================================================

class ModalityLSTMEncoder(nn.Module):
    """
    Single-layer LSTM (hidden ``lstm_hidden``) + two-layer MLP, matching the
    spirit of MFedMC (2401.16685v2) per-modality encoders on UCI-style
    (time × features) inputs.

    Input:  ``(batch, time, in_features)``
    Output: ``(batch, feat_dim)`` — same role as ``FeaExtractor`` for fusion.
    """

    def __init__(self, in_features, feat_dim, lstm_hidden=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.net(h)


class SketchFusionBLSTM(_SketchFusionBase):
    """
    SketchFusionB with LSTM+FC modality encoders (MFedMC-style on UCI HAR).

    Same fusion as :class:`SketchFusionB` (fixed Count Sketch + integrator +
    classifier), but ``img_dim`` / ``txt_dim`` denote **per-timestep feature
    size** (e.g. 3 for acc xyz, 3 for gyro xyz), and inputs are
    ``(B, T, F)`` tensors.
    """

    def __init__(self, img_dim=3, txt_dim=3, feat_dim=512,
                 num_classes=10, dropout=0.5,
                 sketch_r=4, sketch_c=128, lstm_hidden=128, **kwargs):
        nn.Module.__init__(self)
        self.img_extractor = ModalityLSTMEncoder(
            img_dim, feat_dim, lstm_hidden=lstm_hidden, dropout=dropout,
        )
        self.txt_extractor = ModalityLSTMEncoder(
            txt_dim, feat_dim, lstm_hidden=lstm_hidden, dropout=dropout,
        )
        self.img_refiner = FeaRefiner(feat_dim)
        self.txt_refiner = FeaRefiner(feat_dim)
        self.img_to_txt = CrossModalPredictor(feat_dim)
        self.txt_to_img = CrossModalPredictor(feat_dim)
        self._missing_loss = torch.tensor(0.0)

        self.sketch_r = sketch_r
        self.sketch_c = sketch_c
        table_dim = sketch_r * sketch_c

        rng = torch.Generator().manual_seed(SKETCH_HASH_SEED)
        self.register_buffer(
            "buckets",
            torch.randint(0, sketch_c, (sketch_r, feat_dim),
                          generator=rng),
        )
        self.register_buffer(
            "signs",
            (torch.randint(0, 2, (sketch_r, feat_dim),
                           generator=rng) * 2 - 1).float(),
        )

        self.integrator = nn.Sequential(
            nn.Linear(table_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _sketch_features(self, x):
        B = x.size(0)
        tables = []
        for r in range(self.sketch_r):
            signed = self.signs[r] * x
            row = torch.zeros(B, self.sketch_c, device=x.device)
            idx = self.buckets[r].unsqueeze(0).expand(B, -1)
            row.scatter_add_(1, idx, signed)
            tables.append(row)
        return torch.cat(tables, dim=-1)

    def _fuse(self, f_img_refined, f_txt_refined):
        return (self._sketch_features(f_img_refined)
                + self._sketch_features(f_txt_refined))

    def extract_fused(self, img_feat, txt_feat):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        fused = self._fuse(f_img * c_img, f_txt * c_txt)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img_r, f_txt_r = self._extract_and_refine(
            img_feat, txt_feat, missing_prob)
        fused = self._fuse(f_img_r, f_txt_r)
        H = self.integrator(fused)
        return self.classifier(H), H


# =====================================================================
# Model C — Tensor Sketch Fusion
# =====================================================================

class SketchFusionC(_SketchFusionBase):
    """
    Tensor Sketch fusion (Pham & Pagh, 2013).

    Approximates the outer product ``f'_img ⊗ f'_txt`` in a compact
    sketch of dimension ``sketch_c``.  Unlike additive sketching
    (Models A/B), this captures *multiplicative* cross-modal feature
    interactions.

    Algorithm:
      p = CountSketch(f'_img)          # sketch_c-dim
      q = CountSketch(f'_txt)          # sketch_c-dim
      fused = IFFT( FFT(p) ⊙ FFT(q) ) # convolution = polynomial product

    Pipeline (differences from MultiModalNet marked with ★):
      1. FeaExtractor + FeaRefiner  → f'_img, f'_txt   (identical)
      2. ★ Tensor Sketch            → fused             (sketch_c-dim)
      3. Integrator MLP             → H
      4. Classifier                 → logits
    """

    def __init__(self, img_dim=4096, txt_dim=300, feat_dim=512,
                 num_classes=10, dropout=0.5,
                 sketch_c=512, **kwargs):
        super().__init__()
        self._build_front_end(img_dim, txt_dim, feat_dim, dropout)

        self.sketch_c = sketch_c

        rng_img = torch.Generator().manual_seed(SKETCH_HASH_SEED)
        rng_txt = torch.Generator().manual_seed(SKETCH_HASH_SEED + 1)

        self.register_buffer(
            "hash_img",
            torch.randint(0, sketch_c, (feat_dim,), generator=rng_img),
        )
        self.register_buffer(
            "sign_img",
            (torch.randint(0, 2, (feat_dim,), generator=rng_img) * 2 - 1).float(),
        )
        self.register_buffer(
            "hash_txt",
            torch.randint(0, sketch_c, (feat_dim,), generator=rng_txt),
        )
        self.register_buffer(
            "sign_txt",
            (torch.randint(0, 2, (feat_dim,), generator=rng_txt) * 2 - 1).float(),
        )

        self.integrator = nn.Sequential(
            nn.Linear(sketch_c, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _count_sketch(self, x, hash_idx, signs):
        """Differentiable Count Sketch → (batch, sketch_c)."""
        B = x.size(0)
        signed = signs * x                                     # (B, feat_dim)
        sk = torch.zeros(B, self.sketch_c, device=x.device)
        idx = hash_idx.unsqueeze(0).expand(B, -1)              # (B, feat_dim)
        sk.scatter_add_(1, idx, signed)
        return sk

    def _tensor_sketch(self, f_img, f_txt):
        """Tensor Sketch ≈ compact outer product via FFT convolution."""
        p = self._count_sketch(f_img, self.hash_img, self.sign_img)
        q = self._count_sketch(f_txt, self.hash_txt, self.sign_txt)
        P = torch.fft.rfft(p, dim=-1)
        Q = torch.fft.rfft(q, dim=-1)
        return torch.fft.irfft(P * Q, n=self.sketch_c, dim=-1)

    def _fuse(self, f_img_refined, f_txt_refined):
        return self._tensor_sketch(f_img_refined, f_txt_refined)

    def extract_fused(self, img_feat, txt_feat):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        fused = self._fuse(f_img * c_img, f_txt * c_txt)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img_r, f_txt_r = self._extract_and_refine(
            img_feat, txt_feat, missing_prob)
        fused = self._fuse(f_img_r, f_txt_r)
        H = self.integrator(fused)
        return self.classifier(H), H
