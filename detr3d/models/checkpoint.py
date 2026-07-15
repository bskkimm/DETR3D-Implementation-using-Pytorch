"""Checkpoint conversion utilities for official detector initialization."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def translate_fcos3d_key(key: str, target_keys: set[str]) -> str | None:
    if key.startswith("module."):
        key = key[len("module.") :]

    backbone_prefixes = {
        "img_backbone.conv1.": "backbone.stem.0.",
        "img_backbone.bn1.": "backbone.stem.1.",
        "img_backbone.layer1.": "backbone.stage2.",
        "img_backbone.layer2.": "backbone.stage3.",
        "img_backbone.layer3.": "backbone.stage4.",
        "img_backbone.layer4.": "backbone.stage5.",
    }
    translated = None
    for source_prefix, target_prefix in backbone_prefixes.items():
        if key.startswith(source_prefix):
            translated = target_prefix + key[len(source_prefix) :]
            break

    if translated is not None:
        if translated in target_keys:
            return translated
        conv2_weight = ".conv2.weight"
        if translated.endswith(conv2_weight):
            wrapped = translated[: -len(conv2_weight)] + ".conv2.deform_conv.weight"
            if wrapped in target_keys:
                return wrapped
        return None

    for index in range(3):
        source_prefix = f"img_neck.lateral_convs.{index}.conv."
        if key.startswith(source_prefix):
            return f"neck.lateral_convs.{index}." + key[len(source_prefix) :]
        source_prefix = f"img_neck.fpn_convs.{index}.conv."
        if key.startswith(source_prefix):
            return f"neck.output_convs.{index}." + key[len(source_prefix) :]

    extra_prefix = "img_neck.fpn_convs.3.conv."
    if key.startswith(extra_prefix):
        return "neck.extra_conv." + key[len(extra_prefix) :]
    return None


def load_fcos3d_initialization(model: nn.Module, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    target_keys = set(target_state)
    converted = {}
    source_keys = []

    for source_key, value in source_state.items():
        target_key = translate_fcos3d_key(source_key, target_keys)
        if target_key is None:
            continue
        if target_key not in target_state:
            raise KeyError(f"FCOS3D key {source_key!r} mapped to unknown key {target_key!r}")
        if target_state[target_key].shape != value.shape:
            raise ValueError(
                f"FCOS3D shape mismatch for {source_key!r} -> {target_key!r}: "
                f"checkpoint={tuple(value.shape)} model={tuple(target_state[target_key].shape)}"
            )
        converted[target_key] = value
        source_keys.append(source_key)

    expected_keys = {
        key for key in target_state if key.startswith("backbone.") or key.startswith("neck.")
    }
    missing_keys = sorted(expected_keys - set(converted))
    if missing_keys:
        preview = ", ".join(missing_keys[:10])
        raise RuntimeError(
            f"FCOS3D initialization did not cover {len(missing_keys)} backbone/FPN tensors: {preview}"
        )

    model.load_state_dict(converted, strict=False)
    loaded_numel = sum(target_state[key].numel() for key in converted)
    expected_numel = sum(target_state[key].numel() for key in expected_keys)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded_tensors": len(converted),
        "loaded_numel": loaded_numel,
        "expected_tensors": len(expected_keys),
        "expected_numel": expected_numel,
        "coverage": loaded_numel / max(expected_numel, 1),
        "source_tensors_used": len(source_keys),
    }
