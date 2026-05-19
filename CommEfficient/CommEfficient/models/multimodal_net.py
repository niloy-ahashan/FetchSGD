import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["MultiModalNet"]


class FeaExtractor(nn.Module):
    """
    Maps raw modality features into a common k-dimensional latent space.
    Corresponds to Eq. 1 in PFMH:
        f* = FeaExtractor(* ; θ_1*) ∈ R^k
    """
    def __init__(self, in_dim, feat_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeaRefiner(nn.Module):
    """
    Learns modality-specific gating weights that suppress noise and
    emphasise discriminative features.
    Corresponds to Eq. 2 in PFMH:
        C* = FeaRefiner(f* ; θ_2*) ∈ R^k
    """
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
    """
    Predicts one modality's latent representation from another.
    Based on the MFM paper (Tsai et al., ICLR 2019) cross-modal
    encoders used for handling missing modalities.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalNet(nn.Module):
    """
    Multimodal fusion network following PFMH (Eqs. 1-4) up to the
    feature-fused matrix H, topped with a classification head rather
    than the prototype / hash encoder from the paper.

    Includes MFM-style cross-modal predictors for missing-modality
    robustness (Tsai et al., "Learning Factorized Multimodal
    Representations", ICLR 2019).

    Pipeline
    --------
    1. FeaExtractor per modality  →  f_img, f_txt           (Eq. 1)
    1b. CrossModalPredictor       →  f̂_txt from f_img       (MFM)
                                      f̂_img from f_txt       (MFM)
    2. FeaRefiner  per modality   →  C_img, C_txt           (Eq. 2)
    3. Refined features           →  f'_img = f_img ⊙ C_img (Eq. 3)
                                      f'_txt = f_txt ⊙ C_txt
    4. Integrate                  →  H = MLP(f'_img ⊕ f'_txt) (Eq. 4)
    5. Classify                   →  output = Linear(H)

    Parameters
    ----------
    img_dim    : int   – dimensionality of image features  (e.g. 4096)
    txt_dim    : int   – dimensionality of text features   (e.g. 1386)
    feat_dim   : int   – common latent dimension k          (default 512)
    num_classes: int   – number of output classes
    dropout    : float – dropout probability in extractors / integrator
    """

    def __init__(self, img_dim=4096, txt_dim=300, feat_dim=512,
                 num_classes=10, dropout=0.5, **kwargs):
        super().__init__()
        self.img_extractor = FeaExtractor(img_dim, feat_dim, dropout)
        self.txt_extractor = FeaExtractor(txt_dim, feat_dim, dropout)
        self.img_refiner = FeaRefiner(feat_dim)
        self.txt_refiner = FeaRefiner(feat_dim)

        # MFM-style cross-modal predictors for missing modality handling
        self.img_to_txt = CrossModalPredictor(feat_dim)
        self.txt_to_img = CrossModalPredictor(feat_dim)

        self.integrator = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

        self._missing_loss = torch.tensor(0.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_fused(self, img_feat, txt_feat):
        """Return the fused representation H (useful for downstream analysis)."""
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        f_img_refined = f_img * c_img
        f_txt_refined = f_txt * c_txt
        fused = torch.cat([f_img_refined, f_txt_refined], dim=-1)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)

        # Cross-modal predictions (MFM Sec. 3.3)
        f_txt_hat = self.img_to_txt(f_img)
        f_img_hat = self.txt_to_img(f_txt)

        # Missing loss: MSE between predicted and actual latent features.
        # Targets are detached so only the predictor is trained by this
        # loss — the encoders learn purely from the main task loss.
        self._missing_loss = (
            F.mse_loss(f_txt_hat, f_txt.detach()) +
            F.mse_loss(f_img_hat, f_img.detach())
        )

        # Random modality masking during training (MFM-style)
        if self.training and missing_prob > 0:
            r = torch.rand(1).item()
            if r < missing_prob / 2:
                f_img = f_img_hat          # image missing → predict from text
            elif r < missing_prob:
                f_txt = f_txt_hat          # text missing  → predict from image

        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        f_img_refined = f_img * c_img
        f_txt_refined = f_txt * c_txt
        fused = torch.cat([f_img_refined, f_txt_refined], dim=-1)
        H = self.integrator(fused)
        return self.classifier(H), H
