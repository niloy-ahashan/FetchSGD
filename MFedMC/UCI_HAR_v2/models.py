import torch.nn as nn
import torch.nn.functional as F


class FeaExtractor(nn.Module):
    """in_dim -> feat_dim -> feat_dim. Same shape as CommEfficient's
    FeaExtractor (CommEfficient/CommEfficient/models/multimodal_net.py),
    duplicated here so this package stays self-contained (MFedMC does not
    import from CommEfficient/, and has its own separate .venv)."""

    def __init__(self, in_dim, feat_dim, dropout=0.3):
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
    """Sigmoid-gated refinement, same shape as CommEfficient's FeaRefiner."""

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


class VectorMLP(nn.Module):
    """Per-modality encoder for UCI HAR feature vectors — v2.

    raw features -> Extractor (IC-style trunk: in_dim->feat_dim->feat_dim)
                 -> [optional Refiner gate, same as IC's FeaRefiner]
                 -> small classifier (log-probs over num_classes)

    Keeps v1 VectorMLP's exact forward interface (one modality's raw
    feature tensor in, log_softmax over num_classes out) so federated.py's
    NLLLoss training, SHAP/RandomForest fusion, and evaluate_global_test_acc
    are unmodified drop-in consumers. No shared/joint fusion is added here —
    each modality remains its own independent classifier, exactly as
    MFedMC requires; only the encoder trunk widens/deepens to match IC.
    """

    def __init__(self, in_dim, num_classes=6, feat_dim=512, dropout=0.3,
                 use_refiner=True):
        super().__init__()
        self.in_dim = in_dim
        self.use_refiner = use_refiner
        self.extractor = FeaExtractor(in_dim, feat_dim, dropout)
        if use_refiner:
            self.refiner = FeaRefiner(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        f = self.extractor(x)
        if self.use_refiner:
            c = self.refiner(f)
            f = f * c
        return F.log_softmax(self.classifier(f), dim=1)
