"""
Independent Compression — summation fusion + original FetchSGD.

This is *not* sketch-based multimodal fusion (SketchFusion A/B/C).
Modalities are fused by element-wise sum of refined feature vectors.
The fused representation is then classified locally; gradient
communication uses Rothchild et al. FetchSGD (one Count Sketch of the
full gradient), unchanged from the original paper.

Pipeline
--------
  1. FeaExtractor + FeaRefiner per modality  →  f'_acc, f'_gyro
  2. Summation fusion                        →  fused = f'_acc + f'_gyro
  3. Integrator MLP                          →  H
  4. Classifier                              →  logits

  Worker (FetchSGD, after local backward):
      S(∇θ L)  — single CSVecFed over the full parameter vector
"""

import torch.nn as nn

from .multimodal_net import FeaExtractor, FeaRefiner

__all__ = ["IndependentCompression"]


class IndependentCompression(nn.Module):
    """
    Two-branch multimodal net with **summation** fusion.

    Unlike MultiModalNet (concat) and SketchFusionB (per-modality
    Count Sketch then add), this variant adds the refined features
    in the shared ``feat_dim`` space.  Compression is *independent*
    of fusion: it happens later, on the flattened gradient, via
    original FetchSGD.

    For UCI HAR the two branches are accelerometer (``acc_dim``) and
    gyroscope (``gyro_dim``).  ``img_dim`` / ``txt_dim`` are accepted
    as aliases so older call sites still construct the same net.
    """

    def __init__(
        self,
        acc_dim=None,
        gyro_dim=None,
        img_dim=None,
        txt_dim=None,
        feat_dim=512,
        num_classes=10,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        if acc_dim is None:
            acc_dim = 4096 if img_dim is None else img_dim
        if gyro_dim is None:
            gyro_dim = 300 if txt_dim is None else txt_dim

        self.acc_extractor = FeaExtractor(acc_dim, feat_dim, dropout)
        self.gyro_extractor = FeaExtractor(gyro_dim, feat_dim, dropout)
        self.acc_refiner = FeaRefiner(feat_dim)
        self.gyro_refiner = FeaRefiner(feat_dim)

        self.integrator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
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

    def _extract_and_refine(self, acc_feat, gyro_feat):
        f_acc = self.acc_extractor(acc_feat)
        f_gyro = self.gyro_extractor(gyro_feat)
        c_acc = self.acc_refiner(f_acc)
        c_gyro = self.gyro_refiner(f_gyro)
        return f_acc * c_acc, f_gyro * c_gyro

    @staticmethod
    def _fuse(f_acc_refined, f_gyro_refined):
        """Summation fusion in the shared feature space."""
        return f_acc_refined + f_gyro_refined

    def extract_fused(self, acc_feat, gyro_feat):
        f_acc_r, f_gyro_r = self._extract_and_refine(acc_feat, gyro_feat)
        fused = self._fuse(f_acc_r, f_gyro_r)
        return self.integrator(fused)

    def forward(self, acc_feat, gyro_feat):
        f_acc_r, f_gyro_r = self._extract_and_refine(acc_feat, gyro_feat)
        fused = self._fuse(f_acc_r, f_gyro_r)
        H = self.integrator(fused)
        return self.classifier(H), H
