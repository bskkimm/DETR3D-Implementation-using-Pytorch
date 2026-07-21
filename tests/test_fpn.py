import torch
import torch.nn as nn

from detr3d.models.neck.fpn import ImageFPN


def test_official_extra_fpn_level_uses_relu_input():
    fpn = ImageFPN(in_channels=(1,), out_channels=1, out_names=("p5", "p6"), relu_before_extra_convs=True)
    fpn.lateral_convs[0] = nn.Identity()
    fpn.output_convs[0] = nn.Identity()
    nn.init.ones_(fpn.extra_conv.weight)
    nn.init.zeros_(fpn.extra_conv.bias)

    outputs = fpn({"stage5": -torch.ones(1, 1, 1, 4, 4)})

    assert torch.count_nonzero(outputs["p6"]) == 0
