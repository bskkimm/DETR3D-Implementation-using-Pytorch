import torch
import torch.nn as nn

from detr3d.models.transformer.decoder_layer import Detr3DDecoderLayer


class RecordingSelfAttention(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, query, key, value):
        self.events.append("self")
        return torch.zeros_like(value), None


class RecordingCrossAttention(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, *, query, **kwargs):
        self.events.append("cross")
        return torch.zeros_like(query)


class RecordingFFN(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, query):
        self.events.append("ffn")
        return torch.zeros_like(query)


def test_decoder_uses_official_operation_order():
    events = []
    layer = Detr3DDecoderLayer(
        embed_dims=4,
        num_heads=1,
        num_cams=1,
        num_levels=1,
        dropout=0.0,
    )
    layer.self_attn = RecordingSelfAttention(events)
    layer.cross_attn = RecordingCrossAttention(events)
    layer.ffn = RecordingFFN(events)
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.norm3 = nn.Identity()
    layer.dropout = nn.Identity()

    layer(
        query=torch.zeros(1, 2, 4),
        query_pos=torch.zeros(1, 2, 4),
        mlvl_feats=[],
        reference_points=torch.full((1, 2, 3), 0.5),
        img_metas=[],
    )

    assert events == ["self", "cross", "ffn"]
