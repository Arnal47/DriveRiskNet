import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, sequence_length=32, num_features=7, hidden_size=64, dropout=0.2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(sequence_length * num_features, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x): return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, num_features=7, hidden_size=48, num_layers=1, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def build_model(name, sequence_length=32, num_features=7, hidden_size=48, num_layers=1, dropout=0.2):
    if name == "mlp":
        return MLPClassifier(sequence_length, num_features, hidden_size, dropout)
    if name == "lstm":
        return LSTMClassifier(num_features, hidden_size, num_layers, dropout)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
