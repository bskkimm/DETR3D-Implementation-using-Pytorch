"""DETR3D cross-attention."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .feature_sampling import feature_sampling
from .reference_points import inverse_sigmoid


class Detr3DCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        num_levels: int = 4,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = nn.Linear(embed_dims, num_cams * num_levels)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query: torch.Tensor,
        mlvl_feats: List[torch.Tensor],
        reference_points: torch.Tensor,
        img_metas: List[Dict],
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # `sampled_feats`: [B, C, Q, N_cam, 1, N_level]
        # `mask`:         [B, 1, Q, N_cam, 1, N_level]
        # The mask is 1 only for projections that are in front of the camera and inside
        # the image plane. Invalid projections contribute zero feature mass.
        _, sampled_feats, mask = feature_sampling(
            mlvl_feats=mlvl_feats,
            reference_points=reference_points,
            pc_range=self.pc_range,
            img_metas=img_metas,
        )

        weight_query = query if query_pos is None else (query + query_pos)
        attention_weights = self.attention_weights(weight_query)
        attention_weights = attention_weights.view(
            query.shape[0],
            query.shape[1],
            self.num_cams,
            1,
            self.num_levels,
        )
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4).unsqueeze(4)
        attention_weights = attention_weights.sigmoid().to(sampled_feats.dtype) * mask

        fused = (sampled_feats * attention_weights).sum(dim=-1).sum(dim=-1).sum(dim=-1)

        # Convert from [B, C, Q] back to the decoder layout [B, Q, C].
        fused = fused.permute(0, 2, 1)
        fused = self.output_proj(fused)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points))
        return self.dropout(fused) + pos_feat
