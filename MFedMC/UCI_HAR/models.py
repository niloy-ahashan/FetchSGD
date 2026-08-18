import torch.nn as nn
import torch.nn.functional as F


class VectorMLP(nn.Module):
    """Per-modality encoder for UCI HAR engineered feature vectors.

    SketchFusionB uses 348-D accelerometer and 213-D gyroscope vectors.
    This MLP is the MFedMC analogue of the ActionSense LSTM encoder:
    hidden 128 → class logits + LogSoftmax (NLLLoss).
    """

    def __init__(self, in_dim, num_classes=6, hidden=128, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc(x), dim=1)
