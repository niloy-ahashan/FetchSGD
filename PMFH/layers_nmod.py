"""PMFH ``Layers`` generalized from two inputs to N modalities (mHealth: 4).

Upstream ``layers.Layers`` encodes "image" and "text" with one ``MLP`` + gate
each and hashes their sum (``Fusion(x + y)``). All of PMFH's losses act on the
fused code / ``nec_vec``, never on per-modality pairs, so the N-modality version
only needs one ``MLP`` + gate per modality and ``Fusion`` over the sum of all N
gated features. Every other layer (and the creation order) is upstream's; with
N = 2 the network is exactly ``Layers``.

So upstream training code (``main.LocalUpdate`` and the loaders, which pass
``(image, text)``) runs unchanged, ``forward(image, text)`` takes all modalities
concatenated in ``image`` ([m0 | m1 | ...], split by ``args.mod_dims``) and
ignores ``text`` (a (n, 0) placeholder). It returns ``(nec_vec, code, logits)``
like ``Layers``.
"""

import torch
import torch.nn as nn

from layers import MLP, Fusion


def _upstream_gate(dim):
    """Same layers as upstream ``ifeat_gate`` / ``tfeat_gate``."""
    return nn.Sequential(
        nn.Linear(dim, dim * 2),
        nn.ReLU(),
        nn.Linear(dim * 2, dim),
        nn.Sigmoid())


class LayersNMod(nn.Module):
    def __init__(self, args):
        super(LayersNMod, self).__init__()
        self.mod_dims = [int(d) for d in args.mod_dims]
        self.common_dim = int(args.common_dim)
        self.nbit = int(args.bit)
        self.classes = args.classes
        self.batch_size = args.batch_size

        # Same creation order as upstream Layers: fusion, encoders, gates, then the rest.
        self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)
        self.mlps = nn.ModuleList([
            MLP(hidden_dim=[d, args.hidden_dim, self.common_dim], act=nn.Tanh(), dropout=args.mlpdrop)
            for d in self.mod_dims])
        self.gates = nn.ModuleList([_upstream_gate(self.common_dim) for _ in self.mod_dims])
        self.activation = nn.ReLU()
        self.neck = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(self.common_dim * 4, self.common_dim)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.nbit, self.common_dim),
            nn.ReLU()
        )
        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.classify = nn.Linear(self.nbit, self.classes)

    def forward(self, image, text=None, tgt=None):
        self.batch_size = len(image)
        mods = torch.split(image.float(), self.mod_dims, dim=1)
        feats = []
        for x, mlp, gate in zip(mods, self.mlps, self.gates):
            h = mlp(x)
            feats.append(gate(h) * h)
        # Fusion.forward(x, y) hashes x + y; with N inputs hash the sum of all of them.
        fused_fine = self.fusionnn.hash(torch.stack(feats, dim=0).sum(dim=0))
        cfeat_concat = self.activation(self.fusion_layer(fused_fine))
        nec_vec = self.neck(cfeat_concat)
        code = self.hash_output(nec_vec)
        return nec_vec, code, self.classify(code)


class LayersNModFedHead(LayersNMod):
    """``LayersNMod`` plus the federated classifier head of ``layers_uci.LayersFedHead``:
    Linear(common_dim -> classes) on ``nec_vec.detach()`` (``nec_vec`` with
    ``args.head_grad == "joint"``), created inside ``fork_rng`` so PMFH's RNG stream
    is unchanged.
    """

    def __init__(self, args):
        super(LayersNModFedHead, self).__init__(args)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed + 1)
            self.head = nn.Linear(self.common_dim, self.classes)
        self.joint_head = getattr(args, "head_grad", "detached") == "joint"

    def forward(self, image, text=None, tgt=None):
        nec_vec, code, _ = super(LayersNModFedHead, self).forward(image, text, tgt)
        head_in = nec_vec if self.joint_head else nec_vec.detach()
        return nec_vec, code, self.head(head_in)
