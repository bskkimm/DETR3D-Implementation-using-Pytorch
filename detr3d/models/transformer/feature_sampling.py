"""Project 3D reference points into camera views and sample image features."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .reference_points import denormalize_reference_points


def _stack_projection_matrices(img_metas: List[Dict], key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrices = []
    for meta in img_metas:
        if key not in meta:
            raise KeyError(f"Expected '{key}' in img_metas for DETR3D feature sampling.")
        matrices.append(torch.as_tensor(meta[key], dtype=dtype, device=device))
    return torch.stack(matrices, dim=0)


def _stack_image_shapes(img_metas: List[Dict], num_cams: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    shapes = []
    for meta in img_metas:
        image_shape = meta.get("image_shape")
        if image_shape is None:
            image_shape = meta.get("img_shape")
        if image_shape is None:
            raise KeyError("Expected 'image_shape' or 'img_shape' in img_metas.")
        shape_tensor = torch.as_tensor(image_shape, dtype=dtype, device=device)
        if shape_tensor.ndim == 1:
            shape_tensor = shape_tensor.unsqueeze(0).repeat(num_cams, 1)
        shapes.append(shape_tensor[:, :2])
    return torch.stack(shapes, dim=0)


def feature_sampling(
    mlvl_feats: List[torch.Tensor],
    reference_points: torch.Tensor,
    pc_range,
    img_metas: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample multi-scale image features for 3D reference points.

    Args:
        mlvl_feats: list of [B, N, C, H, W]
        reference_points: [B, Q, 3] in normalized xyz
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        img_metas: list[dict] containing `lidar2img` or `ego2img` and image shapes

    Returns:
        reference_points_3d: [B, Q, 3] in metric xyz
        sampled_feats: [B, C, Q, N, 1, L]
        mask: [B, 1, Q, N, 1, L]
    """

    if not img_metas:
        raise ValueError("img_metas is required for DETR3D feature sampling.")

    batch, num_queries, _ = reference_points.shape
    _, num_cams, channels, _, _ = mlvl_feats[0].shape
    num_levels = len(mlvl_feats)
    device = reference_points.device
    dtype = reference_points.dtype

    ref_points_3d = denormalize_reference_points(reference_points, pc_range)
    ref_points_homo = torch.cat([ref_points_3d, torch.ones_like(ref_points_3d[..., :1])], dim=-1)

    matrix_key = "lidar2img" if "lidar2img" in img_metas[0] else "ego2img"
    proj = _stack_projection_matrices(img_metas, matrix_key, device, dtype)
    if proj.ndim == 3:
        proj = proj.unsqueeze(1).repeat(1, num_cams, 1, 1)
    image_shapes = _stack_image_shapes(img_metas, num_cams, device, dtype)

    points = ref_points_homo[:, None, :, :, None]
    proj_points = torch.matmul(proj[:, :, None], points).squeeze(-1)

    depth = proj_points[..., 2:3]
    xy = proj_points[..., :2] / depth.clamp(min=1e-5)
    width = image_shapes[..., 1:2]
    height = image_shapes[..., 0:1]
    norm_x = xy[..., 0] / width.clamp(min=1.0)
    norm_y = xy[..., 1] / height.clamp(min=1.0)
    grid = torch.stack([norm_x * 2 - 1, norm_y * 2 - 1], dim=-1)

    valid = (depth[..., 0] > 1e-5)
    valid = valid & (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)

    sampled_levels = []
    masks = []
    for level_feat in mlvl_feats:
        feat = level_feat.reshape(batch * num_cams, channels, level_feat.shape[-2], level_feat.shape[-1])
        level_grid = grid.reshape(batch * num_cams, num_queries, 1, 2)
        sampled = F.grid_sample(feat, level_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(batch, num_cams, channels, num_queries, 1).permute(0, 2, 3, 1, 4)
        sampled_levels.append(sampled)
        masks.append(valid.permute(0, 2, 1)[:, None, :, :, None].to(dtype=dtype))

    sampled_feats = torch.stack(sampled_levels, dim=-1)
    mask = torch.stack(masks, dim=-1)
    return ref_points_3d, sampled_feats, mask
