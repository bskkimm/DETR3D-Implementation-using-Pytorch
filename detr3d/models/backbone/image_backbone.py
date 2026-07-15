"""Multi-view image backbone modules."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision.ops import DeformConv2d


class DeformConv2dPack(nn.Module):
    """Drop-in replacement for a ResNet 3x3 conv using learned offsets."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        kernel_h, kernel_w = conv.kernel_size
        # A deformable kxk kernel predicts 2 offsets per sampling location, so the
        # offset branch needs `2 * k * k` output channels.
        offset_channels = 2 * kernel_h * kernel_w
        self.offset_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=offset_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_h,
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            # Start from the pretrained dense conv weights and zero offsets. This makes
            # the deformable conv initially behave like the original ResNet conv.
            self.deform_conv.weight.copy_(conv.weight)
            if conv.bias is not None and self.deform_conv.bias is not None:
                self.deform_conv.bias.copy_(conv.bias)
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `offset`: [B, 2 * k * k, H_out, W_out] encodes xy displacements for each kernel
        # sampling position at every output spatial location.
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class ModulatedDeformConv2dPack(nn.Module):
    """Torchvision implementation of the modulated DCNv2 pack used upstream."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        kernel_h, kernel_w = conv.kernel_size
        self.offset_channels = 2 * kernel_h * kernel_w
        self.mask_channels = kernel_h * kernel_w
        self.conv_offset = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=self.offset_channels + self.mask_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=True,
        )
        self.deform_conv = DeformConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            self.deform_conv.weight.copy_(conv.weight)
            if conv.bias is not None and self.deform_conv.bias is not None:
                self.deform_conv.bias.copy_(conv.bias)
            nn.init.zeros_(self.conv_offset.weight)
            nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.conv_offset(x)
        offset = offset_mask[:, : self.offset_channels]
        mask = offset_mask[:, self.offset_channels :].sigmoid()
        return self.deform_conv(x, offset, mask)


class MultiViewImageBackbone(nn.Module):
    """Apply a torchvision ResNet backbone to all camera views.

    Input:
        images: [B, N, 3, H, W]
    Output:
        dict[str, Tensor]: [B, N, C, H_l, W_l] per stage
    """

    def __init__(
        self,
        variant: str = "resnet101",
        pretrained: bool = True,
        frozen_stages: int = 1,
        norm_eval: bool = True,
        official_style: bool = False,
    ):
        super().__init__()
        if variant != "resnet101":
            raise ValueError(f"Unsupported backbone variant: {variant}")

        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet101(weights=weights)
        if official_style:
            self._use_caffe_stride(backbone)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stage2 = backbone.layer1
        self.stage3 = backbone.layer2
        self.stage4 = backbone.layer3
        self.stage5 = backbone.layer4
        self._convert_stage_to_deformable(self.stage4, modulated=official_style)
        self._convert_stage_to_deformable(self.stage5, modulated=official_style)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    @staticmethod
    def _use_caffe_stride(backbone: nn.Module) -> None:
        for stage in (backbone.layer2, backbone.layer3, backbone.layer4):
            block = stage[0]
            block.conv1.stride = (2, 2)
            block.conv2.stride = (1, 1)

    @staticmethod
    def _convert_stage_to_deformable(stage: nn.Sequential, *, modulated: bool) -> None:
        pack_type = ModulatedDeformConv2dPack if modulated else DeformConv2dPack
        for block in stage:
            if hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d):
                block.conv2 = pack_type(block.conv2)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        stages = [self.stage2, self.stage3, self.stage4, self.stage5]
        for idx, stage in enumerate(stages, start=1):
            if self.frozen_stages >= idx:
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_cams, channels, height, width = images.shape
        # Flatten cameras into the batch axis so a standard 2D ResNet can process all
        # views in one pass: [B, N_cam, 3, H, W] -> [B * N_cam, 3, H, W].
        flat = images.reshape(batch * num_cams, channels, height, width)

        x = self.stem(flat)
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)

        # Keep C3/C4/C5-style features. They are reshaped back so downstream DETR3D code
        # can still index features by batch and camera separately.
        features = {
            "stage3": stage3,
            "stage4": stage4,
            "stage5": stage5,
        }
        multi_view = {}
        for name, feat in features.items():
            _, out_channels, out_height, out_width = feat.shape
            multi_view[name] = feat.reshape(batch, num_cams, out_channels, out_height, out_width)
        return multi_view
