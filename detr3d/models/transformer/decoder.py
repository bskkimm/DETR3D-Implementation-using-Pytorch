"""Transformer decoder for DETR3D."""

from typing import Callable, Dict, List, Optional, Tuple

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
        num_levels: int = 4,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dims)
        self.query_pos = nn.Embedding(num_queries, embed_dims)
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

    def init_decoder_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return query, query_pos

    def forward(
        self,
        pyramid: Dict[str, torch.Tensor],
        img_metas: Optional[List[Dict]] = None,
        reference_point_predictor: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if img_metas is None:
            raise ValueError("img_metas is required for DETR3DTransformer.")
        if reference_point_predictor is None:
            raise ValueError(
                "reference_point_predictor is required for paper-aligned DETR3DTransformer.forward(). "
                "Pass a callable such as head.predict_reference_points, or run decoding through Detr3DModel."
            )

        mlvl_feats = list(pyramid.values())
        batch = mlvl_feats[0].shape[0]
        query, query_pos = self.init_decoder_state(batch, mlvl_feats[0].device)
        init_reference = None

        intermediate_states = []
        intermediate_refs = []
        hidden = query
        for layer_idx, layer in enumerate(self.layers):
            # The standalone transformer path follows the same per-layer center
            # prediction contract as the full model: fresh normalized xyz
            # references are predicted from the current query state and then used
            # for projection/sampling in that decoder layer.
            reference_points = reference_point_predictor(layer_idx, hidden)
            if init_reference is None:
                init_reference = reference_points
            intermediate_refs.append(reference_points)
            hidden = layer(
                query=hidden,
                mlvl_feats=mlvl_feats,
                query_pos=query_pos,
                reference_points=reference_points,
                img_metas=img_metas,
            )
            intermediate_states.append(hidden)

        return torch.stack(intermediate_states), init_reference, torch.stack(intermediate_refs)
