# models/GRU.py
import torch.nn as nn


class MicroTrendGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, h = self.gru(x)  # h: (num_layers, batch, hidden_dim)
        last_h = h[-1]  # (batch, hidden_dim)
        logits = self.fc(last_h)  # (batch, num_classes)
        return logits, last_h
