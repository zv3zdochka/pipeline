from __future__ import annotations
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.w0 = nn.Linear(1, 1, bias=True)
        self.wp = nn.Linear(1, out_dim, bias=True)

    def forward(self, t):
        v0 = self.w0(t)
        vp = torch.sin(self.wp(t))
        return torch.cat([v0, vp], dim=-1)


class TimesBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernels=(3, 5),
        dilations=(1, 2),
        p_drop: float = 0.05,
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
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
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
    TimesNet-Mini: compact architecture for 3-class classification of a multivariate time series.
    Main differences from the full version:
      * d_model = 64 instead of 128
      * only 3 TimesBlocks
      * no Multi-Head Attention (replaced by global average pooling)
      * ~6M parameters and significantly lower VRAM usage.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 64,
        n_blocks: int = 3,
        num_classes: int = 3,
        pos_dim: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.time2vec = Time2Vec(pos_dim)
        in_dim = n_features + pos_dim + 1
        self.proj_in = nn.Linear(in_dim, d_model)

        self.blocks = nn.Sequential(
            *[TimesBlock(d_model, p_drop=0.05) for _ in range(n_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, F)
        Returns:
            logits: (B, num_classes)
            emb: (B, d_model)
        """
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError("input length mismatch")

        device = x.device
        t_frac = (
            torch.linspace(0, 1, steps=L, device=device)
            .view(1, L, 1)
            .repeat(B, 1, 1)
        )

        x = torch.cat([x, self.time2vec(t_frac)], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)

        emb = self.pool(x.transpose(1, 2)).squeeze(-1)
        logits = self.head(emb)
        return logits, emb
