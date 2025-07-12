# models/GRU.py

import torch.nn as nn


class MicroTrendGRU(nn.Module):
    """
    A GRU-based classifier for microtrend prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int, default=128
        Dimension of GRU hidden state.
    num_layers : int, default=1
        Number of stacked GRU layers.
    num_classes : int, default=3
        Number of output classes.
    dropout : float, default=0.1
        Dropout rate applied to GRU outputs before the final layer.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 1,
            num_classes: int = 3,
            dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_h = h_n[-1]
        out = self.dropout(last_h)
        logits = self.fc(out)
        return logits, last_h
