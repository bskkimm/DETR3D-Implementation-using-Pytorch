"""Decoder layer for DETR3D."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .cross_attention import Detr3DCrossAttention


class Detr3DDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_cams: int = 6,
        num_levels: int = 3,
        dropout: float = 0.1,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = Detr3DCrossAttention(
            embed_dims=embed_dims,
            num_cams=num_cams,
            num_levels=num_levels,
            pc_range=pc_range,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        mlvl_feats: List[torch.Tensor],
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        img_metas: List[Dict],
    ) -> torch.Tensor:
        q = query + query_pos
        self_attended, _ = self.self_attn(q, q, query)
        query = self.norm1(query + self.dropout(self_attended))

        cross = self.cross_attn(
            query=query,
            mlvl_feats=mlvl_feats,
            reference_points=reference_points,
            img_metas=img_metas,
            query_pos=query_pos,
        )
        query = self.norm2(query + self.dropout(cross))

        ff = self.ffn(query)
        return self.norm3(query + self.dropout(ff))
