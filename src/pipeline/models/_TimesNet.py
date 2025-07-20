from __future__ import annotations

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.w0 = nn.Linear(1, 1, bias=True)
        self.wp = nn.Linear(1, out_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, L, 1) fractional position in the window in [0, 1]
        Returns:
            (B, L, 1 + out_dim) positional encoding
        """
        v0 = self.w0(t)
        vp = torch.sin(self.wp(t))
        return torch.cat([v0, vp], dim=-1)


class TimesBlock(nn.Module):
    """
    Stack of dilated Conv1d → GLU gating → Dropout → Residual + LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        kernels: tuple[int, ...] = (3, 5),
        dilations: tuple[int, ...] = (1, 2),
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(
                d_model,
                2 * d_model,
                kernel_size=k,
                padding=d * (k - 1) // 2,
                dilation=d,
            )
            for k in kernels
            for d in dilations
        )
        self.proj = nn.Conv1d(len(self.convs) * d_model, d_model, 1)
        self.dropout = nn.Dropout(p_drop)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        y = x.transpose(1, 2)
        outs = []
        for conv in self.convs:
            z = conv(y)
            z, g = z.chunk(2, dim=1)
            z = z * torch.sigmoid(g)
            outs.append(z)
        y = torch.cat(outs, dim=1)
        y = self.proj(y).transpose(1, 2)
        y = self.dropout(y) * self.scale
        return self.norm(x + y)


class TimesNetModel(nn.Module):
    """
    Multivariate time-series classifier (3 classes).
    Input:  (B, seq_len, n_features)
    Output: logits (B, 3) and embedding (B, d_model)
    Components:
        * Time2Vec positional features
        * Multiple TimesBlocks
        * Learnable CLS token + Multi-Head Self-Attention
        * Linear classification head
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_blocks: int = 4,
        num_classes: int = 3,
        pos_dim: int = 8,
        n_heads: int = 8,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.time2vec = Time2Vec(pos_dim)
        in_dim = n_features + pos_dim + 1
        self.proj_in = nn.Linear(in_dim, d_model)

        self.blocks = nn.Sequential(
            *[TimesBlock(d_model, p_drop=p_drop) for _ in range(n_blocks)]
        )

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        self.head = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_tok, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, F)
        Returns:
            logits: (B, num_classes)
            emb: (B, d_model)
        """
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError("input length must equal seq_len")

        device = x.device
        t_frac = (
            torch.linspace(0, 1, steps=L, device=device)
            .view(1, L, 1)
            .repeat(B, 1, 1)
        )
        x = torch.cat([x, self.time2vec(t_frac)], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)

        cls = self.cls_tok.expand(B, -1, -1)
        seq = torch.cat([cls, x], dim=1)
        seq, _ = self.attn(seq, seq, seq)
        emb = seq[:, 0]
        logits = self.head(emb)
        return logits, emb
