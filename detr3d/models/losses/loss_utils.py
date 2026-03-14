"""Loss helpers and box parameterization utilities."""

from __future__ import annotations

import torch


def box3d_to_10d(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, z, w, l, h, yaw] to [cx, cy, l, w, cz, h, sin, cos, vx, vy]."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 10))
    out = boxes.new_zeros((boxes.shape[0], 10))
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 4]
    out[:, 3] = boxes[:, 3]
    out[:, 4] = boxes[:, 2]
    out[:, 5] = boxes[:, 5]
    out[:, 6] = torch.sin(boxes[:, 6])
    out[:, 7] = torch.cos(boxes[:, 6])
    return out


def expand_bbox_preds(box_preds: torch.Tensor) -> torch.Tensor:
    if box_preds.shape[-1] == 10:
        return box_preds
    if box_preds.shape[-1] != 7:
        raise ValueError(f"Expected bbox dim 7 or 10, got {box_preds.shape[-1]}")
    shape = box_preds.shape[:-1] + (10,)
    out = box_preds.new_zeros(shape)
    out[..., 0] = box_preds[..., 0]
    out[..., 1] = box_preds[..., 1]
    out[..., 2] = box_preds[..., 4]
    out[..., 3] = box_preds[..., 3]
    out[..., 4] = box_preds[..., 2]
    out[..., 5] = box_preds[..., 5]
    out[..., 6] = torch.sin(box_preds[..., 6])
    out[..., 7] = torch.cos(box_preds[..., 6])
    return out


def normalize_bbox(boxes: torch.Tensor, pc_range) -> torch.Tensor:
    pc_range = torch.as_tensor(pc_range, dtype=boxes.dtype, device=boxes.device)
    xyz_min = pc_range[:3]
    xyz_max = pc_range[3:]
    xyz_span = (xyz_max - xyz_min).clamp(min=1e-5)

    out = boxes.clone()
    out[..., 0] = (boxes[..., 0] - xyz_min[0]) / xyz_span[0]
    out[..., 1] = (boxes[..., 1] - xyz_min[1]) / xyz_span[1]
    out[..., 4] = (boxes[..., 4] - xyz_min[2]) / xyz_span[2]
    out[..., 2] = boxes[..., 2] / xyz_span[0]
    out[..., 3] = boxes[..., 3] / xyz_span[1]
    out[..., 5] = boxes[..., 5] / xyz_span[2]
    out[..., 8] = boxes[..., 8] / xyz_span[0]
    out[..., 9] = boxes[..., 9] / xyz_span[1]
    return out
