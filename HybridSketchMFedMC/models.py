"""
Multimodal fusion for an arbitrary number of modalities.

Each modality has FeaExtractor + FeaRefiner. Refined vectors are fused
by one of:

  fusion_mode="sketch"  — SketchFusionB: fixed Count Sketch then sum
  fusion_mode="sum"     — IndependentCompression: element-wise sum in
                          feat_dim (late / additive feature fusion)

The fused vector is decoded by the integrator MLP + classifier.
Gradient communication is separate (FetchSGD) in either case.

For two modalities the architecture matches CommEfficient SketchFusionB /
IndependentCompression (including unused-at-zero-weight cross-modal
predictors). For three or more modalities the layout matches
SketchFusionB4 (no pairwise cross-modal heads).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SKETCH_HASH_SEED = 2147483647


class FeaExtractor(nn.Module):
    def __init__(self, in_dim, feat_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class FeaRefiner(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalPredictor(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class SketchFusionBNet(nn.Module):
    """N-modality fusion net: Count Sketch (SketchFusionB) or summation."""

    def __init__(
        self,
        mod_dims,
        feat_dim=512,
        num_classes=6,
        dropout=0.3,
        sketch_r=4,
        sketch_c=128,
        fusion_mode="sketch",
    ):
        super().__init__()
        self.mod_dims = tuple(int(d) for d in mod_dims)
        self.n_mod = len(self.mod_dims)
        if self.n_mod < 2:
            raise ValueError("SketchFusionBNet needs at least 2 modalities")
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in ("sketch", "sum"):
            raise ValueError(
                f"fusion_mode must be 'sketch' or 'sum', got {fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode
        self.feat_dim = feat_dim
        self.sketch_r = sketch_r
        self.sketch_c = sketch_c
        fuse_dim = sketch_r * sketch_c if fusion_mode == "sketch" else feat_dim

        self.extractors = nn.ModuleList(
            [FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims]
        )
        self.refiners = nn.ModuleList(
            [FeaRefiner(feat_dim) for _ in self.mod_dims]
        )

        # Match two-modality SketchFusionB / IndependentCompression (MFM heads).
        self.img_to_txt = None
        self.txt_to_img = None
        if self.n_mod == 2:
            self.img_to_txt = CrossModalPredictor(feat_dim)
            self.txt_to_img = CrossModalPredictor(feat_dim)

        self._missing_loss = torch.tensor(0.0)

        if fusion_mode == "sketch":
            rng = torch.Generator().manual_seed(SKETCH_HASH_SEED)
            self.register_buffer(
                "buckets",
                torch.randint(0, sketch_c, (sketch_r, feat_dim), generator=rng),
            )
            self.register_buffer(
                "signs",
                (torch.randint(0, 2, (sketch_r, feat_dim), generator=rng) * 2 - 1).float(),
            )
        else:
            self.register_buffer("buckets", torch.empty(0, dtype=torch.long))
            self.register_buffer("signs", torch.empty(0))

        self.integrator = nn.Sequential(
            nn.Linear(fuse_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
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
            row = torch.zeros(B, self.sketch_c, device=x.device, dtype=x.dtype)
            idx = self.buckets[r].unsqueeze(0).expand(B, -1)
            row.scatter_add_(1, idx, signed)
            tables.append(row)
        return torch.cat(tables, dim=-1)

    def _refine_one(self, i, x):
        f = self.extractors[i](x)
        c = self.refiners[i](f)
        return f * c

    def _extract_and_refine(self, mods, missing_prob=0.0):
        feats = [self.extractors[i](mods[i]) for i in range(self.n_mod)]

        if self.n_mod == 2 and self.img_to_txt is not None:
            f_txt_hat = self.img_to_txt(feats[0])
            f_img_hat = self.txt_to_img(feats[1])
            self._missing_loss = (
                F.mse_loss(f_txt_hat, feats[1].detach())
                + F.mse_loss(f_img_hat, feats[0].detach())
            )
            if self.training and missing_prob > 0:
                r = torch.rand(1).item()
                if r < missing_prob / 2:
                    feats[0] = f_img_hat
                elif r < missing_prob:
                    feats[1] = f_txt_hat
        else:
            self._missing_loss = torch.zeros((), device=feats[0].device)

        return [feats[i] * self.refiners[i](feats[i]) for i in range(self.n_mod)]

    def _project(self, refined):
        if self.fusion_mode == "sketch":
            return self._sketch_features(refined)
        return refined

    def _fuse_refined(self, refined, modality_mask=None):
        total = None
        for i, r in enumerate(refined):
            if modality_mask is not None and not bool(modality_mask[i]):
                continue
            mapped = self._project(r)
            total = mapped if total is None else total + mapped
        if total is None:
            raise RuntimeError("No modalities selected for fusion")
        return total

    def forward(self, *mods, missing_prob=0.0, modality_mask=None):
        if len(mods) == 1 and isinstance(mods[0], (list, tuple)):
            mods = tuple(mods[0])
        if len(mods) != self.n_mod:
            raise ValueError(f"expected {self.n_mod} modalities, got {len(mods)}")
        refined = self._extract_and_refine(mods, missing_prob)
        fused = self._fuse_refined(refined, modality_mask)
        H = self.integrator(fused)
        return self.classifier(H), H

    def forward_single_modality(self, i, x):
        """Logits from modality i alone (for MFedMC-style SHAP)."""
        refined = self._refine_one(i, x)
        fused = self._project(refined)
        H = self.integrator(fused)
        return self.classifier(H)


def build_param_index_maps(model: SketchFusionBNet):
    """Flat-gradient index lists: one per modality, plus shared layers."""
    mod_idx = [[] for _ in range(model.n_mod)]
    shared = []
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        pos = list(range(offset, offset + n))
        assigned = False
        for i in range(model.n_mod):
            if name.startswith(f"extractors.{i}") or name.startswith(f"refiners.{i}"):
                mod_idx[i].extend(pos)
                assigned = True
                break
        if not assigned:
            if name.startswith("img_to_txt"):
                mod_idx[0].extend(pos)
            elif name.startswith("txt_to_img"):
                mod_idx[1].extend(pos)
            else:
                shared.extend(pos)
        offset += n
    return [torch.tensor(ix, dtype=torch.long) for ix in mod_idx], torch.tensor(
        shared, dtype=torch.long
    ), offset
