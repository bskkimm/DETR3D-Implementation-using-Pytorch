"""Transformer decoder for DETR3D."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .decoder_layer import Detr3DDecoderLayer


class Detr3DTransformer(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_queries: int = 900,
        num_layers: int = 6,
        num_heads: int = 8,
        num_cams: int = 6,
        num_levels: int = 3,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dims)
        self.query_pos = nn.Embedding(num_queries, embed_dims)
        self.reference_points = nn.Linear(embed_dims, 3)
        self.layers = nn.ModuleList(
            [
                Detr3DDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_cams=num_cams,
                    num_levels=num_levels,
                    pc_range=pc_range,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        pyramid: Dict[str, torch.Tensor],
        img_metas: List[Dict] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if img_metas is None:
            raise ValueError("img_metas is required for DETR3DTransformer.")

        mlvl_feats = list(pyramid.values())
        batch = mlvl_feats[0].shape[0]
        query = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference = reference_points

        intermediate_states = []
        intermediate_refs = []
        hidden = query
        for layer in self.layers:
            hidden = layer(
                query=hidden,
                mlvl_feats=mlvl_feats,
                query_pos=query_pos,
                reference_points=reference_points,
                img_metas=img_metas,
            )
            intermediate_states.append(hidden)
            intermediate_refs.append(reference_points)

        return torch.stack(intermediate_states), init_reference, torch.stack(intermediate_refs)
