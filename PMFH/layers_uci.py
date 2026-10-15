"""PMFH ``Layers`` with the baselines' gate and a trained classifier head (PMFH-v2).

Same network as upstream ``layers.Layers`` except:

* per-modality gate = IC / MFedMC v2 / Hybrid v2 ``FeaRefiner`` shape at PMFH's
  ``common_dim``: Linear(c, c) -> ReLU -> Linear(c, c) -> Sigmoid (upstream:
  c -> 2c -> c), still applied as ``feat * gate(feat)``;
* upstream ``classify`` (Linear(bit -> classes) on the hash code, never trained)
  is removed; ``head`` = Linear(common_dim -> classes) on ``nec_vec``, the neck
  output, mirroring the baselines' ``classifier(integrator(fused))``.

``forward`` still returns ``(nec_vec, code, logits)`` so upstream code that
unpacks three values keeps working.
"""

import torch
import torch.nn as nn

from layers import Layers


def _baseline_gate(dim):
    """Same layers as CommEfficient / MFedMC v2 / Hybrid v2 ``FeaRefiner``."""
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim),
        nn.Sigmoid(),
    )


class LayersUCI(Layers):
    def __init__(self, args):
        super(LayersUCI, self).__init__(args)
        self.ifeat_gate = _baseline_gate(self.common_dim)
        self.tfeat_gate = _baseline_gate(self.common_dim)
        del self.classify
        self.head = nn.Linear(self.common_dim, self.classes)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)
        textH = self.textMLP(text)
        image_feat = self.ifeat_gate(imageH) * imageH
        text_feat = self.tfeat_gate(textH) * textH
        fused_fine = self.fusionnn(image_feat, text_feat)
        cfeat_concat = self.activation(self.fusion_layer(fused_fine))
        nec_vec = self.neck(cfeat_concat)
        code = self.hash_output(nec_vec)
        return nec_vec, code, self.head(nec_vec)


class LayersFedHead(Layers):
    """Official PMFH ``Layers`` plus a federated classifier head (option B).

    Everything upstream is kept as is (gate, ``classify``, all sizes). Added:
    ``head`` = Linear(common_dim -> classes) on ``nec_vec.detach()``, so the
    head is trained, uploaded and FedAvg'd like the baselines' classifier, but
    its loss never reaches PMFH's own parameters. The head is created inside
    ``fork_rng`` so torch's RNG (init, shuffling, dropout) is the same as for
    official PMFH. ``forward`` returns ``(nec_vec, code, head logits)``.

    With ``args.head_grad == "joint"`` (option C) the head reads ``nec_vec``
    without ``detach``, so its CE gradient also trains PMFH's network, like the
    baselines' jointly trained classifier.
    """

    def __init__(self, args):
        super(LayersFedHead, self).__init__(args)
        # Only the CPU generator is used for init; fork_rng restores it afterwards
        # (torch.manual_seed would also reseed CUDA, which fork_rng(devices=[]) can't restore).
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed + 1)
            self.head = nn.Linear(self.common_dim, self.classes)
        self.joint_head = getattr(args, "head_grad", "detached") == "joint"

    def forward(self, image, text, tgt=None):
        nec_vec, code, _ = super(LayersFedHead, self).forward(image, text, tgt)
        head_in = nec_vec if self.joint_head else nec_vec.detach()
        return nec_vec, code, self.head(head_in)
