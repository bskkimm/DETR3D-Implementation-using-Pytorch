"""DETR3D cross-attention."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .feature_sampling import feature_sampling
from .reference_points import inverse_sigmoid


class Detr3DCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        num_levels: int = 3,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.attention_weights = nn.Linear(embed_dims, num_cams * num_levels)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        mlvl_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        img_metas: List[Dict],
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, num_queries, _ = query.shape
        attn_input = query if query_pos is None else query + query_pos
        weights = self.attention_weights(attn_input).view(batch, num_queries, self.num_cams, self.num_levels)
        weights = weights.sigmoid().unsqueeze(1).unsqueeze(4)

        ref_points_3d, sampled_feats, mask = feature_sampling(
            mlvl_feats=mlvl_feats,
            reference_points=reference_points,
            pc_range=self.pc_range,
            img_metas=img_metas,
        )

        fused = (sampled_feats * weights * mask).sum(dim=-1).sum(dim=-1).sum(dim=-1)
        fused = fused.permute(0, 2, 1)
        fused = self.output_proj(fused)

        pos_feat = self.position_encoder(inverse_sigmoid(reference_points))
        return self.dropout(fused + pos_feat)
