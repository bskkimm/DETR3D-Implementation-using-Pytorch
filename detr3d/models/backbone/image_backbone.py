"""Multi-view image backbone modules."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + residual)


class TinyResNetBackbone(nn.Module):
    """A compact ResNet-style backbone for multi-view DETR3D experiments."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        self.stage5 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualBlock(base_channels * 8, base_channels * 8),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        return {
            "stage3": stage3,
            "stage4": stage4,
            "stage5": stage5,
        }


class MultiViewImageBackbone(nn.Module):
    """Apply the same image backbone to all camera views.

    Input:
        images: [B, N, 3, H, W]
    Output:
        dict[str, Tensor]: [B, N, C, H_l, W_l] per stage
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.backbone = TinyResNetBackbone(in_channels=in_channels, base_channels=base_channels)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_cams, channels, height, width = images.shape
        flat = images.reshape(batch * num_cams, channels, height, width)
        features = self.backbone(flat)

        multi_view = {}
        for name, feat in features.items():
            _, out_channels, out_height, out_width = feat.shape
            multi_view[name] = feat.reshape(batch, num_cams, out_channels, out_height, out_width)
        return multi_view
