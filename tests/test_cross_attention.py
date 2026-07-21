import torch
import torch.nn as nn

import detr3d.models.transformer.cross_attention as cross_attention_module
from detr3d.models.transformer.cross_attention import Detr3DCrossAttention


class ScaleAndCount(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return 2.0 * value


class FixedPosition(nn.Module):
    def forward(self, reference_logits):
        return reference_logits.new_tensor([5.0, 6.0]).expand(
            *reference_logits.shape[:-1], 2
        )


def test_cross_attention_has_one_residual_and_visual_only_dropout(monkeypatch):
    def fake_feature_sampling(**kwargs):
        reference_points = kwargs["reference_points"]
        sampled = reference_points.new_tensor([2.0, 4.0]).view(
            1, 2, 1, 1, 1, 1
        )
        mask = reference_points.new_ones(1, 1, 1, 1, 1, 1)
        return reference_points, sampled, mask

    monkeypatch.setattr(
        cross_attention_module,
        "feature_sampling",
        fake_feature_sampling,
    )
    attention = Detr3DCrossAttention(
        embed_dims=2,
        num_cams=1,
        num_levels=1,
        dropout=0.0,
    )
    nn.init.zeros_(attention.attention_weights.weight)
    nn.init.zeros_(attention.attention_weights.bias)
    nn.init.eye_(attention.output_proj.weight)
    nn.init.zeros_(attention.output_proj.bias)
    attention.position_encoder = FixedPosition()
    dropout = ScaleAndCount()
    attention.dropout = dropout

    output = attention(
        query=torch.tensor([[[10.0, 20.0]]]),
        mlvl_feats=[],
        reference_points=torch.full((1, 1, 3), 0.5),
        img_metas=[],
    )

    torch.testing.assert_close(output, torch.tensor([[[17.0, 30.0]]]))
    assert dropout.calls == 1
