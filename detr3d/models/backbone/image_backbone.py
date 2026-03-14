"""Backbone wrapper for multi-view images."""

from typing import Dict

import torch
import torch.nn as nn


class MultiViewImageBackbone(nn.Module):
    """Minimal multi-view backbone interface.

    Input:
        images: [B, N, 3, H, W]
    Output:
        dict[str, Tensor]: feature maps keyed by stage name
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_cams, channels, height, width = images.shape
        flat = images.view(batch * num_cams, channels, height, width)
        features = self.backbone(flat)
        if isinstance(features, torch.Tensor):
            features = {"stage4": features}
        return features
