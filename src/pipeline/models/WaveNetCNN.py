import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution with left padding to enforce temporal causality.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x).contiguous()


class WaveletBlock(nn.Module):
    """
    Single residual block with gated activation (WaveNet-style),
    weight normalization and dropout for regularization.
    """

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv_filter = weight_norm(
            CausalConv1d(channels, channels, kernel_size, dilation)
        )
        self.conv_gate = weight_norm(
            CausalConv1d(channels, channels, kernel_size, dilation)
        )
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        z = f * g
        z = self.dropout(z)
        res = self.res_conv(z)
        return x + res, z


class WaveCNN(nn.Module):
    """
    WaveCNN composed of multiple WaveletBlocks, batch normalization,
    global pooling, and a classification head with dropout.
    Number of blocks is chosen to cover the input window.
    """

    def __init__(
        self,
        in_channels,
        emb_dim=256,
        num_classes=3,
        window_size=24,
        kernel_size=3,
        dilation_base=2,
    ):
        super().__init__()
        import math

        num_blocks = math.ceil(math.log2(window_size + 1)) - 1
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.blocks = nn.ModuleList()
        channels = in_channels
        for i in range(num_blocks):
            dilation = dilation_base**i
            self.blocks.append(WaveletBlock(channels, kernel_size, dilation))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(emb_dim, num_classes),
        )

    def forward(self, x):
        x = self.input_bn(x)
        for block in self.blocks:
            x, _ = block(x)
        h = self.global_pool(x).squeeze(-1)
        logits = self.classifier(h)
        return logits, h
