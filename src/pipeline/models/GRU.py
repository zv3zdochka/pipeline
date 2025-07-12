# src/pipeline/models/GRU.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroTrendGRU(nn.Module):
    """
    Двуслойный bidirectional-GRU + mean-pool + классификатор.
    Выход: logits и сам hidden-вектор (B, D*2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d_out = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
            nn.Linear(d_out, num_classes),
        )

    def forward(self, x):                              # x: (B, L, F)
        seq_out, _ = self.gru(x)                       # (B, L, D*dir)
        emb = seq_out.mean(1)                          # mean-pool
        logits = self.head(emb)                        # (B, C)
        return logits, emb                             # emb пригодится как feature
