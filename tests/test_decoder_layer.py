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


class FullResidualCrossAttention(nn.Module):
    def forward(self, *, query, **kwargs):
        return torch.full_like(query, 7.0)


class ZeroFFN(nn.Module):
    def forward(self, query):
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


def test_decoder_does_not_add_second_cross_attention_residual():
    layer = Detr3DDecoderLayer(
        embed_dims=4,
        num_heads=1,
        num_cams=1,
        num_levels=1,
        dropout=0.0,
    )
    layer.self_attn = RecordingSelfAttention([])
    layer.cross_attn = FullResidualCrossAttention()
    layer.ffn = ZeroFFN()
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.norm3 = nn.Identity()
    layer.dropout = nn.Identity()

    output = layer(
        query=torch.ones(1, 1, 4),
        query_pos=torch.zeros(1, 1, 4),
        mlvl_feats=[],
        reference_points=torch.full((1, 1, 3), 0.5),
        img_metas=[],
    )

    torch.testing.assert_close(output, torch.full((1, 1, 4), 7.0))


def test_decoder_ffn_has_official_width_512():
    layer = Detr3DDecoderLayer()
    linears = [module for module in layer.ffn if isinstance(module, nn.Linear)]

    assert len(linears) == 2
    assert linears[0].in_features == 256
    assert linears[0].out_features == 512
    assert linears[1].in_features == 512
    assert linears[1].out_features == 256
