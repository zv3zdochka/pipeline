"""
Лёгкая, самодостаточная реализация TimesNet-подобной архитектуры
для многомерного тайм-ряда (классификация на 3 класса).

* Time2Vec — learnable позиционные признаки
* несколько TimesBlock (dilated Conv1d + GLU + LayerNorm)
* CLS-токен + Multi-Head Self-Attention
* выход: logits (B, 3) и embedding (B, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# -------------------------------------------------------------------- #
#                           СЛУЖЕБНЫЕ БЛОКИ                            #
# -------------------------------------------------------------------- #
class Time2Vec(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.w0 = nn.Linear(1, 1, bias=True)
        self.wp = nn.Linear(1, out_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (B, L, 1) — доля от 0 до 1 внутри окна
        """
        v0 = self.w0(t)
        vp = torch.sin(self.wp(t))
        return torch.cat([v0, vp], dim=-1)          # (B, L, 1+out_dim)


class TimesBlock(nn.Module):
    """
    Набор dilated Conv1d → GLU → Dropout → Residual + LayerNorm
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
        x : (B, L, D)
        """
        y = x.transpose(1, 2)                      # (B, D, L)
        outs = []
        for conv in self.convs:
            z = conv(y)                            # (B, 2*D, L)
            z, g = z.chunk(2, dim=1)
            z = z * torch.sigmoid(g)
            outs.append(z)
        y = torch.cat(outs, dim=1)                 # (B, N*D, L)
        y = self.proj(y).transpose(1, 2)           # (B, L, D)
        y = self.dropout(y) * self.scale
        return self.norm(x + y)


# -------------------------------------------------------------------- #
#                             TimesNetModel                            #
# -------------------------------------------------------------------- #
class TimesNetModel(nn.Module):
    """
    Multivariate time-series classifier (3 класса):
        вход  — (B, seq_len, n_features)
        выход — logits (B, 3) и embedding (B, d_model)
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

        # позиционные признаки
        self.time2vec = Time2Vec(pos_dim)
        in_dim = n_features + pos_dim + 1           # +1 за v0 в Time2Vec
        self.proj_in = nn.Linear(in_dim, d_model)

        # стек TimesBlock-ов
        self.blocks = nn.Sequential(
            *[TimesBlock(d_model, p_drop=p_drop) for _ in range(n_blocks)]
        )

        # CLS-токен + MH-Attention
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        # классификатор
        self.head = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    # -------------------------------------------
    def _reset_parameters(self):
        nn.init.normal_(self.cls_tok, std=0.02)

    # -------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : (B, L, F)
        returns:
            logits : (B, 3)
            emb    : (B, D)
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
        x = torch.cat([x, self.time2vec(t_frac)], dim=-1)   # добавляем позицию
        x = self.proj_in(x)
        x = self.blocks(x)

        cls = self.cls_tok.expand(B, -1, -1)                # (B,1,D)
        seq = torch.cat([cls, x], dim=1)                    # (B,L+1,D)
        seq, _ = self.attn(seq, seq, seq)
        emb = seq[:, 0]                                     # CLS-эмбед
        logits = self.head(emb)
        return logits, emb
