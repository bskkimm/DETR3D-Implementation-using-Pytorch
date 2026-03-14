"""Transformer decoder wrapper."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class Detr3DTransformer(nn.Module):
    """Thin adapter around the current scratch transformer implementation."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(self, pyramid: Dict[str, torch.Tensor], img_metas: List[Dict] | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlvl_feats = list(pyramid.values())
        query_embed = getattr(self.transformer, "query_embed", None)
        if query_embed is None:
            raise AttributeError("The wrapped transformer must expose or build its query embedding explicitly.")
        return self.transformer(mlvl_feats, query_embed, img_metas=img_metas)
