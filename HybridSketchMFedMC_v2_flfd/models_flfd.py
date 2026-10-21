"""FL-FD's low-parameter CNN in the model interface HybridSketchMFedMC_v2's loop expects.

HybridSketchMFedMC_v2/federated.py needs from the model:
  * ``model(*mods) -> (logits, H)`` and ``model.forward_single_modality(i, x) -> logits``;
  * ``model.n_mod`` and per-modality parameters named ``extractors.<i>`` (or ``refiners.<i>``):
    ``models.build_param_index_maps`` turns those names into the per-modality gradient coordinates
    that modality selection masks; every other parameter is "shared" and always uploaded.

Two models, both built from FL-FD's own layers (FL-FD/fixed/main.py ``FLFDNet``):

``FLFDBranchNet`` (--model branch, default)
    per modality: GAF image [1, 70, 70] -> FL-FD's 3 conv blocks (first conv 1 -> 3 channels) -> 588
    -> Hybrid's fixed Count-Sketch fusion (SketchFusionB, sketch_r x sketch_c, no parameters) or a sum
    -> FL-FD's single Linear head.  UCI HAR, 2 x 128 sketch: 900 + 900 + 1,542 = 3,342 params.

``FLFDInputFusionNet`` (--model flfd)
    FL-FD's network unchanged: [acc GAF, gyro GAF] stacked -> Conv2d(2, 3, 3) -> ... -> Linear(588, 6),
    4,461 params.  Its first conv is held as one Conv2d(1, 3, 3) per input channel (``extractors.<i>``,
    no bias) plus one shared bias, which is the same function and the same 57 numbers as Conv2d(2, 3, 3);
    that is the only part of FL-FD's model that belongs to one modality.
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn as nn

from models import SKETCH_HASH_SEED, SketchFusionBNet  # HybridSketchMFedMC_v2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_flfd_fixed():
    spec = importlib.util.spec_from_file_location(
        "flfd_fixed_main_for_hybrid", os.path.join(ROOT, "FL-FD", "fixed", "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FLFD = _load_flfd_fixed()


def flfd_features(in_channels, size):
    """FL-FD's conv stack (3 x conv3x3 -> Softsign -> maxpool2 -> Dropout2d(0.2) -> BN) and its output size."""
    net = _FLFD.FLFDNet(num_classes=1, size=size)
    feats = net.features
    old = feats[0]
    feats[0] = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride)
    return feats, net.fc1.in_features  # 12 * 7 * 7 = 588 at 70 x 70


class FLFDBranchNet(nn.Module):
    """One FL-FD conv branch per modality -> Hybrid's sketch (or sum) fusion -> FL-FD's Linear head."""

    def __init__(self, n_mod, num_classes, size=70, fusion_mode="sketch", sketch_r=2, sketch_c=128):
        super().__init__()
        if n_mod < 2:
            raise ValueError("needs at least 2 modalities")
        self.n_mod = int(n_mod)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in ("sketch", "sum"):
            raise ValueError(f"fusion_mode must be 'sketch' or 'sum', got {fusion_mode!r}")
        self.sketch_r, self.sketch_c = int(sketch_r), int(sketch_c)
        branches = [flfd_features(1, size) for _ in range(self.n_mod)]
        self.extractors = nn.ModuleList([b[0] for b in branches])
        self.feat_dim = branches[0][1]
        fuse_dim = self.sketch_r * self.sketch_c if self.fusion_mode == "sketch" else self.feat_dim
        if self.fusion_mode == "sketch":
            # same fixed hash as SketchFusionBNet (HybridSketchMFedMC_v2/models.py), over the 588 features
            rng = torch.Generator().manual_seed(SKETCH_HASH_SEED)
            self.register_buffer("buckets", torch.randint(0, self.sketch_c, (self.sketch_r, self.feat_dim), generator=rng))
            self.register_buffer("signs", (torch.randint(0, 2, (self.sketch_r, self.feat_dim), generator=rng) * 2 - 1).float())
        self.head = nn.Linear(fuse_dim, num_classes)  # FL-FD's single fc1, PyTorch default init

    _sketch_features = SketchFusionBNet._sketch_features  # Hybrid's Count-Sketch projection, reused

    def _encode(self, i, x):
        return torch.flatten(self.extractors[i](x), 1)

    def _project(self, f):
        return self._sketch_features(f) if self.fusion_mode == "sketch" else f

    def forward(self, *mods, modality_mask=None):
        if len(mods) == 1 and isinstance(mods[0], (list, tuple)):
            mods = tuple(mods[0])
        if len(mods) != self.n_mod:
            raise ValueError(f"expected {self.n_mod} modalities, got {len(mods)}")
        fused = None
        for i, x in enumerate(mods):
            if modality_mask is not None and not bool(modality_mask[i]):
                continue
            z = self._project(self._encode(i, x))
            fused = z if fused is None else fused + z
        return self.head(fused), fused

    def forward_single_modality(self, i, x):
        return self.head(self._project(self._encode(i, x)))


class FLFDInputFusionNet(nn.Module):
    """FL-FD's own network (input-level fusion), first conv split by input channel for Hybrid's masks."""

    def __init__(self, n_mod, num_classes, size=70):
        super().__init__()
        self.n_mod = int(n_mod)
        feats, flat = flfd_features(self.n_mod, size)
        conv = feats[0]
        self.extractors = nn.ModuleList(
            [nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, bias=False) for _ in range(self.n_mod)]
        )
        with torch.no_grad():  # same initial weights as FL-FD's Conv2d(n_mod, 3, 3), one input channel each
            for i, c in enumerate(self.extractors):
                c.weight.copy_(conv.weight[:, i:i + 1])
        self.conv1_bias = nn.Parameter(conv.bias.detach().clone())
        self.rest = feats[1:]  # Softsign, pool, dropout, BN, blocks 2 and 3 (FL-FD's layers)
        self.fc1 = nn.Linear(flat, num_classes)

    def _first(self, mods):
        z = None
        for i, x in enumerate(mods):
            y = self.extractors[i](x)
            z = y if z is None else z + y
        return z + self.conv1_bias.view(1, -1, 1, 1)

    def _tail(self, z):
        return self.fc1(torch.flatten(self.rest(z), 1))

    def forward(self, *mods, modality_mask=None):
        if len(mods) == 1 and isinstance(mods[0], (list, tuple)):
            mods = tuple(mods[0])
        if modality_mask is not None:
            mods = [x if bool(modality_mask[i]) else torch.zeros_like(x) for i, x in enumerate(mods)]
        z = self._first(mods)
        return self._tail(z), z

    def forward_single_modality(self, i, x):
        """The other channel left at zero, as if only modality i's image were stacked."""
        z = self.extractors[i](x) + self.conv1_bias.view(1, -1, 1, 1)
        return self._tail(z)
