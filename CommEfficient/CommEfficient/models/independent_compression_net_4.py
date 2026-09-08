"""
Independent Compression, extended to **four** modalities — summation fusion
+ original FetchSGD.

Same idea as ``independent_compression_net.py``'s 2-modality
``IndependentCompression``, generalized the way ``SketchFusionB4``
generalizes ``SketchFusionB``: one ``FeaExtractor``/``FeaRefiner`` pair per
modality (from a ``mod_dims`` list) instead of two named branches, refined
features are fused by plain element-wise **sum** (not sketched), then
decoded by an integrator MLP and a linear classifier. No cross-modal
prediction loss (compatible with ``missing_loss_weight=0``, matching
``SketchFusionB4``).

Pipeline
--------
  1. FeaExtractor + FeaRefiner per modality  →  f'_0, f'_1, f'_2, f'_3
  2. Summation fusion                        →  fused = sum(f'_i)
  3. Integrator MLP                          →  H
  4. Classifier                              →  logits
"""

import torch
import torch.nn as nn

from .multimodal_net import FeaExtractor, FeaRefiner

__all__ = ["IndependentCompression4"]


class IndependentCompression4(nn.Module):
    """
    Four-branch multimodal net with **summation** fusion.

    Unlike ``SketchFusionB4`` (per-modality Count Sketch then add), this
    variant sums the refined features directly in the shared ``feat_dim``
    space. Compression is independent of fusion: it happens later, on the
    flattened gradient, via original FetchSGD.
    """

    def __init__(
        self,
        mod_dims,
        feat_dim=512,
        num_classes=10,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        if len(mod_dims) != 4:
            raise ValueError(
                f"IndependentCompression4 expects 4 mod_dims, got {len(mod_dims)}"
            )
        self.mod_dims = tuple(int(d) for d in mod_dims)
        self.extractors = nn.ModuleList(
            [FeaExtractor(d, feat_dim, dropout) for d in self.mod_dims]
        )
        self.refiners = nn.ModuleList([FeaRefiner(feat_dim) for _ in range(4)])
        self._missing_loss = torch.tensor(0.0)

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

    @staticmethod
    def _fuse_refined(refined):
        """Summation fusion in the shared feature space."""
        total = refined[0]
        for r in refined[1:]:
            total = total + r
        return total

    def _refine_all(self, m0, m1, m2, m3):
        refined = []
        for i, x in enumerate((m0, m1, m2, m3)):
            f = self.extractors[i](x)
            c = self.refiners[i](f)
            refined.append(f * c)
        return refined

    def extract_fused(self, m0, m1, m2, m3):
        refined = self._refine_all(m0, m1, m2, m3)
        fused = self._fuse_refined(refined)
        return self.integrator(fused)

    def forward(self, m0, m1, m2, m3, missing_prob=0.0):
        del missing_prob  # reserved; no stochastic masking implemented
        refined = self._refine_all(m0, m1, m2, m3)
        fused = self._fuse_refined(refined)
        H = self.integrator(fused)
        return self.classifier(H), H
