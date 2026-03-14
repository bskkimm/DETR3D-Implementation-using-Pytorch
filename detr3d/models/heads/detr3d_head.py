"""DETR3D detection head."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Detr3DHead(nn.Module):
    def __init__(self, embed_dims: int = 256, num_classes: int = 10, box_dim: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.cls_branch = MLP(embed_dims, embed_dims, num_classes + 1, num_layers=3)
        self.reg_branch = MLP(embed_dims, embed_dims, box_dim, num_layers=3)

    def forward(self, hs: torch.Tensor):
        cls_scores = []
        bbox_preds = []
        for layer_hs in hs:
            cls_scores.append(self.cls_branch(layer_hs))
            bbox_preds.append(self.reg_branch(layer_hs))
        return torch.stack(cls_scores), torch.stack(bbox_preds)
