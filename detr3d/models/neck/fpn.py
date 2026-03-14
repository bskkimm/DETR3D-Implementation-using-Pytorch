"""FPN placeholder."""

from typing import Dict

import torch
import torch.nn as nn


class ImageFPN(nn.Module):
    """Keeps the FPN contract explicit even before the real implementation lands."""

    def __init__(self):
        super().__init__()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return features
