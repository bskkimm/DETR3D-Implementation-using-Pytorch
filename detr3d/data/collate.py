"""Batch collation for DETR3D samples."""

from __future__ import annotations

from typing import List

import torch


def detr3d_collate(batch: List[dict]) -> dict:
    return {
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "img_metas": [item["img_metas"] for item in batch],
        "gt_boxes_ego": [item["gt_boxes_ego"] for item in batch],
        "gt_boxes_lidar": [item.get("gt_boxes_lidar", item["gt_boxes_ego"]) for item in batch],
        "gt_labels": [item["gt_labels"] for item in batch],
    }


__all__ = ["detr3d_collate"]
