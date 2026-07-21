import torch
import torch.nn as nn
from torchvision import models

from detr3d.models.backbone.image_backbone import (
    ModulatedDeformConv2dPack,
    MultiViewImageBackbone,
)


def test_modulated_deform_conv_predicts_offsets_and_masks():
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
    packed = ModulatedDeformConv2dPack(conv)

    assert packed.conv_offset.out_channels == 27
    assert torch.count_nonzero(packed.conv_offset.weight) == 0
    assert packed(torch.randn(2, 4, 8, 8)).shape == (2, 8, 8, 8)


def test_official_backbone_uses_caffe_stride_and_modulated_dcn(monkeypatch):
    original_resnet101 = models.resnet101
    monkeypatch.setattr(
        "detr3d.models.backbone.image_backbone.models.resnet101",
        lambda *, weights: original_resnet101(weights=None),
    )
    backbone = MultiViewImageBackbone(pretrained=False, official_style=True)

    assert backbone.stage3[0].conv1.stride == (2, 2)
    assert backbone.stage3[0].conv2.stride == (1, 1)
    assert isinstance(backbone.stage4[0].conv2, ModulatedDeformConv2dPack)
