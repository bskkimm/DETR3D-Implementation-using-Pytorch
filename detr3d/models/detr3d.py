"""Top-level DETR3D composition module."""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class Detr3DOutputs:
    cls_scores: torch.Tensor
    bbox_preds: torch.Tensor
    init_reference: torch.Tensor
    inter_references: torch.Tensor


class Detr3DModel(nn.Module):
    """Wires backbone, neck, transformer, and head into one module."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, transformer: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.transformer = transformer
        self.head = head

    def forward(self, images: torch.Tensor, img_metas: List[Dict] | None = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        pyramid = self.neck(features)
        hs, init_reference, inter_references = self.transformer(pyramid, img_metas=img_metas)
        cls_scores, bbox_preds = self.head(hs)
        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "init_reference": init_reference,
            "inter_references": inter_references,
        }
