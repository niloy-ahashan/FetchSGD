"""PyTorch port of the FLASH multimodal network (Salehi et al., INFOCOM'22).

Layer-for-layer from ``FLASH/code/federated/models.py`` (``coord_mlp`` with
``fusion=True``) and the fusion block in ``federated_learning.py:326-352``.

UCI HAR / mHealth modalities are 1-D feature vectors, so every modality gets
its own copy of FLASH's GPS ("coord") branch -- the only FLASH branch built
for 1-D input. The fusion head is unchanged except its input width, which is
64 * M (FLASH: 64 coord + 256 image + 512 lidar = 832).

Keras details kept: glorot-uniform kernels with zero biases, TF 'SAME'
padding / pooling, BatchNormalization(eps=1e-3, momentum=0.99) with biased
batch variance, and the fusion layers' kernel/bias regularizers (returned by
``reg_loss`` and added to the training and validation loss, as Keras does).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCH_OUT = 64       # coord_output: Dense(64, tanh)
BRANCH_DIMS = (1024, 512, 256)       # coord_dense1 / coord_dense2 / coord_dense22
FUSION_DIMS = (1024, 512, 256, 128)  # first_fusion / dense{1,2,3}_fusion
FUSION_REG = 1e-3     # l2(reg_val) on first_fusion / dense{1,2,3}_fusion
OUT_L1, OUT_L2 = 1e-5, 1e-4  # l1_l2 on fusion_output kernel


class KerasBatchNorm1d(nn.Module):
    """Keras BatchNormalization for 2-D input.

    Unlike ``nn.BatchNorm1d`` it accepts a batch of one sample (variance 0),
    which Keras does, and updates the moving variance with the biased batch
    variance.
    """

    def __init__(self, num_features, eps=1e-3, momentum=0.99):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mean)
                self.running_var.mul_(self.momentum).add_((1 - self.momentum) * var)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def _same(x):
    return F.pad(x, (0, 1))


class CoordBranch(nn.Module):
    """FLASH ``coord_mlp`` (fusion=True) for a length-``in_len`` vector.

    ``dims`` = widths of the three hidden Dense layers (FLASH: 1024, 512, 256).
    """

    def __init__(self, in_len, dims=BRANCH_DIMS):
        super().__init__()
        d1, d2, d3 = (int(d) for d in dims)
        # Conv1D(20, 2, padding='SAME'): TF pads an even kernel on the right only.
        self.conv1 = nn.Conv1d(1, 20, 2)
        self.conv2 = nn.Conv1d(20, 20, 2)
        self.conv3 = nn.Conv1d(20, 20, 2)
        self.conv4 = nn.Conv1d(20, 20, 2)
        # MaxPooling1D(2, padding='same') -> ceil(L / 2)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        flat_len = -(-(-(-in_len // 2)) // 2)
        self.dense1 = nn.Linear(20 * flat_len, d1)
        self.dense2 = nn.Linear(d1, d2)
        self.dense22 = nn.Linear(d2, d3)
        self.out = nn.Linear(d3, BRANCH_OUT)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        x = F.relu(self.conv2(_same(F.relu(self.conv1(_same(x))))))
        x = self.pool(x)
        x = F.relu(self.conv4(_same(F.relu(self.conv3(_same(x))))))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(F.relu(self.dense1(x)))
        x = self.drop(F.relu(self.dense2(x)))
        x = self.drop(F.relu(self.dense22(x)))
        return torch.tanh(self.out(x))


class FusionHead(nn.Module):
    """FLASH integration network: four Dense+ReLU+BN layers, softmax output.

    ``dims`` = widths of the four hidden layers (FLASH: 1024, 512, 256, 128).
    """

    def __init__(self, in_dim, num_classes, dims=FUSION_DIMS):
        super().__init__()
        dims = [int(d) for d in dims]
        if len(dims) != 4:
            raise ValueError(f"fusion dims must have 4 widths, got {dims}")
        widths = [in_dim, *dims]
        self.hidden = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])]
        )
        self.bns = nn.ModuleList([KerasBatchNorm1d(w) for w in widths[1:]])
        self.out = nn.Linear(widths[-1], num_classes)  # softmax folded into the CE loss

    def forward(self, z):
        for dense, bn in zip(self.hidden, self.bns):
            z = bn(F.relu(dense(z)))
        return self.out(z)

    def reg_loss(self):
        reg = 0.0
        for dense in self.hidden:
            reg = reg + FUSION_REG * (dense.weight.pow(2).sum() + dense.bias.pow(2).sum())
        w = self.out.weight
        return reg + OUT_L1 * w.abs().sum() + OUT_L2 * w.pow(2).sum()


class SharedBranch(nn.Module):
    """Hybrid v2 / IC per-modality encoder (``SketchFusionBNet._refine_one``).

    The refiner is a sigmoid gate on the extracted features:
    ``f * refiner(f)`` with ``f = extractor(x)`` (feat_dim out).
    """

    def __init__(self, in_len, feat_dim, dropout):
        super().__init__()
        from models import FeaExtractor, FeaRefiner  # HybridSketchMFedMC_v2/models.py

        self.extractor = FeaExtractor(in_len, feat_dim, dropout)
        self.refiner = FeaRefiner(feat_dim)

    def forward(self, x):
        f = self.extractor(x)
        return f * self.refiner(f)


def shared_branch(in_len, feat_dim, dropout):
    return SharedBranch(in_len, feat_dim, dropout)


class FlashNet(nn.Module):
    """One branch per modality -> concat -> FusionHead.

    ``branch="gps"``: FLASH's CoordBranch (64-d tanh out), as published.
    ``branch="shared"``: the Hybrid v2 / IC encoder (feat_dim out), so every
    method uses the same per-modality encoder; FLASH's fusion head and
    federated loop are unchanged apart from the head's input width.

    ``branch_dims`` (gps only) and ``fusion_dims`` narrow the hidden layers
    without changing the layer structure (cf. PMFH --hidden_dim/--common_dim).
    """

    def __init__(self, mod_dims, num_classes, branch="gps", feat_dim=512, dropout=0.3,
                 branch_dims=BRANCH_DIMS, fusion_dims=FUSION_DIMS):
        super().__init__()
        self.mod_dims = [int(d) for d in mod_dims]
        self.branch = branch
        self.branch_dims = tuple(int(d) for d in branch_dims)
        self.fusion_dims = tuple(int(d) for d in fusion_dims)
        if len(self.branch_dims) != 3:
            raise ValueError(f"branch dims must have 3 widths, got {list(self.branch_dims)}")
        if branch == "gps":
            self.branch_out = BRANCH_OUT
            self.branches = nn.ModuleList(
                [CoordBranch(d, self.branch_dims) for d in self.mod_dims]
            )
        elif branch == "shared":
            if self.branch_dims != BRANCH_DIMS:
                raise ValueError("--branch_dims applies to --branch gps only "
                                 "(the shared encoder's width is --feat_dim)")
            self.branch_out = int(feat_dim)
            self.branches = nn.ModuleList(
                [shared_branch(d, feat_dim, dropout) for d in self.mod_dims]
            )
        else:
            raise ValueError(f"branch must be 'gps' or 'shared', got {branch!r}")
        self.head = FusionHead(self.branch_out * len(self.mod_dims), num_classes, self.fusion_dims)
        self._keras_init()

    def _keras_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *xs):
        z = torch.cat([b(x) for b, x in zip(self.branches, xs)], dim=1)
        # (logits, aux) to match the evaluators shared with Hybrid v2.
        return self.head(z), None

    def reg_loss(self):
        return self.head.reg_loss()

    def branch_groups(self):
        """state_dict keys per FLASH branch: ``mod_0..mod_{M-1}`` then ``fusion``."""
        keys = list(self.state_dict().keys())
        groups = {
            f"mod_{i}": [k for k in keys if k.startswith(f"branches.{i}.")]
            for i in range(len(self.branches))
        }
        groups["fusion"] = [k for k in keys if k.startswith("head.")]
        return groups
