import torch.nn as nn
import torch.nn.functional as F

class Acc_MLP(nn.Module):
    """Accelerometer encoder for the SketchFusionB UCI HAR split (348-D vectors)."""
    def __init__(self):
        super(Acc_MLP, self).__init__()
        self.fc1 = nn.Linear(348, 128)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return nn.LogSoftmax(dim=1)(x)

class Gyro_MLP(nn.Module):
    """Gyroscope encoder for the SketchFusionB UCI HAR split (213-D vectors)."""
    def __init__(self):
        super(Gyro_MLP, self).__init__()
        self.fc1 = nn.Linear(213, 128)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return nn.LogSoftmax(dim=1)(x)
