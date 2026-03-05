# models/quartznet_blocks.py

import torch
import torch.nn as nn


class TimeChannelSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        # Depthwise
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )

        # Pointwise
        self.pointwise = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class QuartzBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, R):
        super().__init__()

        layers = []
        for i in range(R):
            layers.append(
                TimeChannelSeparableConv(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size
                )
            )

        self.block = nn.Sequential(*layers)

        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.block(x)
        return x + res