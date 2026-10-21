import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """Per-modality encoder matching the paper's Sec 4.2 description: "a
    consistent long short-term memory (LSTM) network structure, consisting of
    a single LSTM layer with 128 hidden units, followed by a fully connected
    layer." No dropout anywhere (the paper's encoder has none).

    input_size=3 for both Acc and Gyro here (3 raw sensor axes each; the two
    modalities differ only in which raw signal feeds them, not in encoder
    shape, matching upstream's per-modality LSTM classes).
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 128, num_classes: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)          # (batch, 128, hidden_size)
        last = out[:, -1, :]           # last timestep's hidden state
        return F.log_softmax(self.fc(last), dim=1)
