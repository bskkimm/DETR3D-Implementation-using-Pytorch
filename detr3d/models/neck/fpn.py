"""Feature pyramid network for multi-view image features."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.lateral(x))


class ImageFPN(nn.Module):
    """Build a top-down feature pyramid from multi-view backbone features."""

    def __init__(
        self,
        in_channels: Iterable[int] = (128, 256, 512),
        out_channels: int = 256,
        out_names: Iterable[str] = ("p3", "p4", "p5"),
    ):
        super().__init__()
        self.out_names = list(out_names)
        self.blocks = nn.ModuleList([FPNBlock(ch, out_channels) for ch in in_channels])

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stage_names = sorted(features.keys())
        pyramid: Dict[str, torch.Tensor] = {}
        top_down = None

        for name, block, out_name in zip(reversed(stage_names), reversed(self.blocks), reversed(self.out_names)):
            feat = features[name]
            batch, num_cams, channels, height, width = feat.shape
            flat = feat.reshape(batch * num_cams, channels, height, width)
            lateral = block.lateral(flat)
            if top_down is not None:
                lateral = lateral + F.interpolate(top_down, size=lateral.shape[-2:], mode="nearest")
            out = block.output(lateral)
            top_down = lateral
            pyramid[out_name] = out.reshape(batch, num_cams, out.shape[1], out.shape[2], out.shape[3])

        return {name: pyramid[name] for name in self.out_names}
