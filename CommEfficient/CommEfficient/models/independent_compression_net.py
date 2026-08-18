"""
Independent Compression — summation fusion + original FetchSGD.

This is *not* sketch-based multimodal fusion (SketchFusion A/B/C).
Modalities are fused by element-wise sum of refined feature vectors.
The fused representation is then classified locally; gradient
communication uses Rothchild et al. FetchSGD (one Count Sketch of the
full gradient), unchanged from the original paper.

Pipeline
--------
  1. FeaExtractor + FeaRefiner per modality  →  f'_img, f'_txt
  2. Summation fusion                        →  fused = f'_img + f'_txt
  3. Integrator MLP                          →  H
  4. Classifier                              →  logits

  Worker (FetchSGD, after local backward):
      S(∇θ L)  — single CSVecFed over the full parameter vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_net import FeaExtractor, FeaRefiner, CrossModalPredictor

__all__ = ["IndependentCompression"]


class IndependentCompression(nn.Module):
    """
    Two-branch multimodal net with **summation** fusion.

    Unlike MultiModalNet (concat) and SketchFusionB (per-modality
    Count Sketch then add), this variant adds the refined features
    in the shared ``feat_dim`` space.  Compression is *independent*
    of fusion: it happens later, on the flattened gradient, via
    original FetchSGD.
    """

    def __init__(
        self,
        img_dim=4096,
        txt_dim=300,
        feat_dim=512,
        num_classes=10,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.img_extractor = FeaExtractor(img_dim, feat_dim, dropout)
        self.txt_extractor = FeaExtractor(txt_dim, feat_dim, dropout)
        self.img_refiner = FeaRefiner(feat_dim)
        self.txt_refiner = FeaRefiner(feat_dim)

        self.img_to_txt = CrossModalPredictor(feat_dim)
        self.txt_to_img = CrossModalPredictor(feat_dim)

        self.integrator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
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

    def _extract_and_refine(self, img_feat, txt_feat, missing_prob=0.0):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)

        f_txt_hat = self.img_to_txt(f_img)
        f_img_hat = self.txt_to_img(f_txt)
        self._missing_loss = (
            F.mse_loss(f_txt_hat, f_txt.detach())
            + F.mse_loss(f_img_hat, f_img.detach())
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

    @staticmethod
    def _fuse(f_img_refined, f_txt_refined):
        """Summation fusion in the shared feature space."""
        return f_img_refined + f_txt_refined

    def extract_fused(self, img_feat, txt_feat):
        f_img = self.img_extractor(img_feat)
        f_txt = self.txt_extractor(txt_feat)
        c_img = self.img_refiner(f_img)
        c_txt = self.txt_refiner(f_txt)
        fused = self._fuse(f_img * c_img, f_txt * c_txt)
        return self.integrator(fused)

    def forward(self, img_feat, txt_feat, missing_prob=0.0):
        f_img_r, f_txt_r = self._extract_and_refine(
            img_feat, txt_feat, missing_prob
        )
        fused = self._fuse(f_img_r, f_txt_r)
        H = self.integrator(fused)
        return self.classifier(H), H
