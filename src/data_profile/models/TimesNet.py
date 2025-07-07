# src/data_profile/models/TimesNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimesBlock(nn.Module):
    """Один блок TimesNet (усечённая авторская идея):
    1) Conv1d (групповые, несколько dilation-ов)
    2) GLU-гейт
    3) Residual + LayerNorm
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dilations=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    2 * d_model,
                    kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                    groups=1,
                )
                for dilation in dilations
            ]
        )
        self.proj = nn.Conv1d(len(dilations) * d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        y = x.transpose(1, 2)  # (B, D, L)
        outs = []
        for conv in self.convs:
            z = conv(y)  # (B, 2D, L)
            z, g = z.chunk(2, dim=1)
            z = z * torch.sigmoid(g)  # GLU
            outs.append(z)
        y = torch.cat(outs, dim=1)          # (B, k*D, L)
        y = self.proj(y)                    # (B, D, L)
        y = y.transpose(1, 2)               # (B, L, D)
        return self.norm(x + y)             # residual


class TimesNetModel(nn.Module):
    """
    Полноценный TimesNet-подобный стэк для многомерного ряда.
    Задача: бинарная классификация (рост / падение).
    """

    def __init__(
        self,
        seq_len: int = 288,
        n_features: int = 32,
        d_model: int = 128,
        n_blocks: int = 3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.proj_in = nn.Linear(n_features, d_model)
        self.blocks = nn.Sequential(*[TimesBlock(d_model) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)  # глобальный по времени
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, L, F)
        x = self.proj_in(x)           # (B, L, D)
        x = self.blocks(x)            # (B, L, D)
        emb = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        logits = self.head(emb)       # (B, 2)
        return logits, emb            # logits для BCE, emb для сохранения
