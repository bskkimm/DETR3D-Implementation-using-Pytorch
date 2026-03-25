"""Feature pyramid network for multi-view image features."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageFPN(nn.Module):
    """Build a top-down feature pyramid from multi-view backbone features."""

    def __init__(
        self,
        in_channels: Iterable[int] = (512, 1024, 2048),
        out_channels: int = 256,
        out_names: Iterable[str] = ("p3", "p4", "p5", "p6"),
    ):
        super().__init__()
        self.out_names = list(out_names)
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )
        self.extra_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stage_names = sorted(features.keys())
        flat_feats = []
        batch, num_cams = next(iter(features.values())).shape[:2]
        for name in stage_names:
            feat = features[name]
            # FPN is defined over 2D feature maps, so flatten cameras into the batch axis:
            # [B, N_cam, C, H, W] -> [B * N_cam, C, H, W].
            flat_feats.append(feat.reshape(batch * num_cams, feat.shape[2], feat.shape[3], feat.shape[4]))

        # Lateral 1x1 convolutions unify channel width, e.g. 512/1024/2048 -> 256.
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, flat_feats)]
        for idx in range(len(laterals) - 1, 0, -1):
            # Standard top-down FPN recursion:
            # P_n = lateral(C_n) + upsample(P_{n+1})
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(laterals[idx], size=laterals[idx - 1].shape[-2:], mode="nearest")

        # The 3x3 output convolutions smooth the fused maps after upsampling/addition.
        outputs = [conv(feat) for conv, feat in zip(self.output_convs, laterals)]
        # P6 is an extra coarser level built by stride-2 downsampling of P5.
        outputs.append(self.extra_conv(outputs[-1]))

        pyramid: Dict[str, torch.Tensor] = {}
        for out_name, feat in zip(self.out_names, outputs):
            # Restore [B, N_cam, C, H_l, W_l] layout for downstream multi-view sampling.
            pyramid[out_name] = feat.reshape(batch, num_cams, feat.shape[1], feat.shape[2], feat.shape[3])
        return pyramid
