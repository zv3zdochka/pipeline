# models/wave_ccn.py
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs,
        )
        self._left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x, (self._left_pad, 0))
        return super().forward(x)


class WaveCNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            emb_dim: int = 128,
            num_classes: int = 3,
            n_layers: int = 6,
            kernel_size: int = 3,
            dilation_base: int = 2,
    ):
        super().__init__()

        layers = []
        channels = in_channels
        for i in range(n_layers):
            dilation = dilation_base ** i
            layers += [
                CausalConv1d(
                    channels,
                    emb_dim,
                    kernel_size,
                    dilation=dilation,
                ),
                nn.ReLU(inplace=True),
            ]
            channels = emb_dim

        self.conv_stack = nn.Sequential(*layers)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        h = self.conv_stack(x)  # (B, C, T)
        emb = h[:, :, -1]  # last time step
        logits = self.classifier(emb)
        return logits, emb
