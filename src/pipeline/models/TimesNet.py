import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Multiple dilated Conv1d → GLU → Dropout → Scaled Residual + LayerNorm"""

    def __init__(self, d_model: int, kernels=(3, 5), dilations=(1, 2)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(
                d_model, 2 * d_model, k,
                padding=d * (k - 1) // 2, dilation=d
            )
            for k in kernels for d in dilations
        )
        self.proj = nn.Conv1d(len(self.convs) * d_model, d_model, 1)
        self.dropout = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
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
    """Multivariate time series → embedding → 3-class classification"""

    def __init__(
            self,
            seq_len: int = 288,
            n_features: int = 32,
            d_model: int = 128,
            n_blocks: int = 4,
            num_classes: int = 3,
            pos_dim: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.time2vec = Time2Vec(pos_dim)
        in_dim = n_features + pos_dim + 1
        self.proj_in = nn.Linear(in_dim, d_model)
        self.blocks = nn.Sequential(*[TimesBlock(d_model) for _ in range(n_blocks)])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.head = nn.Linear(d_model, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x : (B, L, F) — normalized numerical features over sequence length
        returns logits: (B, 3), emb: (B, D)
        """
        B, L, _ = x.shape
        assert L == self.seq_len, "input length must equal seq_len"
        device = x.device
        t_frac = torch.linspace(0, 1, steps=L, device=device).view(1, L, 1).repeat(B, 1, 1)
        x = torch.cat([x, self.time2vec(t_frac)], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        cls_tok = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls_tok, x], dim=1)
        seq, _ = self.attn(seq, seq, seq)
        emb = seq[:, 0]
        logits = self.head(emb)
        return logits, emb
