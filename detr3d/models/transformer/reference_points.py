"""Reference point helpers."""

from __future__ import annotations

import torch


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def denormalize_reference_points(reference_points: torch.Tensor, pc_range) -> torch.Tensor:
    pc_range = torch.as_tensor(pc_range, dtype=reference_points.dtype, device=reference_points.device)
    xyz_min = pc_range[:3]
    xyz_max = pc_range[3:]
    return reference_points * (xyz_max - xyz_min) + xyz_min
