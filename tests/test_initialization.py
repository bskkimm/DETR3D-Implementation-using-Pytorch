import math

import torch

from detr3d.models.heads import Detr3DHead
from detr3d.models.transformer import Detr3DTransformer

SENTINEL = 0.123


def sentinel_xavier(tensor, gain=1.0):
    with torch.no_grad():
        return tensor.fill_(SENTINEL)


def test_transformer_xavier_and_cross_attention_zero_initialization(monkeypatch):
    monkeypatch.setattr(torch.nn.init, "xavier_uniform_", sentinel_xavier)

    transformer = Detr3DTransformer(
        embed_dims=8,
        num_queries=3,
        num_layers=1,
        num_heads=1,
        num_cams=1,
        num_levels=1,
    )

    for name, parameter in transformer.named_parameters():
        if parameter.ndim <= 1:
            continue
        if name.endswith("cross_attn.attention_weights.weight"):
            assert torch.count_nonzero(parameter) == 0
        else:
            assert torch.all(parameter == SENTINEL), name

    cross = transformer.layers[0].cross_attn
    assert torch.count_nonzero(cross.attention_weights.weight) == 0
    assert torch.count_nonzero(cross.attention_weights.bias) == 0
    assert torch.count_nonzero(cross.output_proj.bias) == 0


def test_reference_projection_uses_xavier_with_zero_bias(monkeypatch):
    monkeypatch.setattr(torch.nn.init, "xavier_uniform_", sentinel_xavier)

    head = Detr3DHead(
        embed_dims=8,
        num_classes=2,
        num_decoder_layers=1,
    )

    assert torch.all(head.reference_points.weight == SENTINEL)
    assert torch.count_nonzero(head.reference_points.bias) == 0
    expected_cls_bias = -math.log(99.0)
    final_cls = head.cls_branches[0].net[-1]
    torch.testing.assert_close(
        final_cls.bias,
        torch.full_like(final_cls.bias, expected_cls_bias),
    )
