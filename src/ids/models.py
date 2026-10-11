"""Neural architectures reviewed in the paper (Table 2), sized for tabular flow features.

Sequence models (CNN, LSTM, CNN-LSTM) read the feature vector as a 1-channel sequence of
length n_features, the common practice in flow-based IDS literature.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, n_out))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, n_in, n_out, channels=(32, 64), dropout=0.2):
        super().__init__()
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, 3, padding=1), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(), nn.AdaptiveMaxPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2 * 8, 64),
                                  nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.head(torch.cat([h[-2], h[-1]], 1))


class CNNLSTM(nn.Module):
    """Hybrid: convolution extracts local feature patterns, LSTM models their ordering."""
    def __init__(self, n_in, n_out, channels=32, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.BatchNorm1d(channels),
                                  nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1])


class Autoencoder(nn.Module):
    """Trained on Normal traffic only; reconstruction error is the anomaly score."""
    def __init__(self, n_in, bottleneck=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))

    def score(self, x):
        return ((self(x) - x) ** 2).mean(1)


CLASSIFIERS = {"MLP": MLP, "CNN-1D": CNN1D, "BiLSTM": LSTMNet, "CNN-LSTM": CNNLSTM}
