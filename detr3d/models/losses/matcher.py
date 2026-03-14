"""Matching utilities for DETR-style training."""

from __future__ import annotations

from typing import List, Tuple

import torch

from .loss_utils import box3d_to_10d, expand_bbox_preds, normalize_bbox


def _linear_sum_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        device = cost.device
        return (
            torch.as_tensor(row_ind, dtype=torch.long, device=device),
            torch.as_tensor(col_ind, dtype=torch.long, device=device),
        )
    except Exception:
        num_rows, num_cols = cost.shape
        taken_rows = []
        taken_cols = []
        for _ in range(min(num_rows, num_cols)):
            flat_idx = cost.argmin()
            row = flat_idx // num_cols
            col = flat_idx % num_cols
            if row in taken_rows or col in taken_cols:
                cost[row, col] = float("inf")
                continue
            taken_rows.append(int(row))
            taken_cols.append(int(col))
            cost[row] = float("inf")
            cost[:, col] = float("inf")
        device = cost.device
        return (
            torch.as_tensor(taken_rows, dtype=torch.long, device=device),
            torch.as_tensor(taken_cols, dtype=torch.long, device=device),
        )


class HungarianMatcher3D:
    def __init__(self, num_classes: int, pc_range, cls_weight: float = 1.0, bbox_weight: float = 1.0):
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight

    @torch.no_grad()
    def __call__(
        self,
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ):
        box_preds = expand_bbox_preds(box_preds)
        assignments = []
        for batch_idx in range(cls_logits.shape[0]):
            if gt_boxes[batch_idx].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=cls_logits.device)
                assignments.append((empty, empty))
                continue
            gt_boxes_10d = box3d_to_10d(gt_boxes[batch_idx])
            probs = cls_logits[batch_idx].softmax(-1)[:, : self.num_classes]
            cls_cost = -probs[:, gt_labels[batch_idx]]
            pred_norm = normalize_bbox(box_preds[batch_idx], self.pc_range)
            gt_norm = normalize_bbox(gt_boxes_10d, self.pc_range)
            bbox_cost = torch.cdist(pred_norm[:, :8], gt_norm[:, :8], p=1)
            total_cost = self.cls_weight * cls_cost + self.bbox_weight * bbox_cost
            assignments.append(_linear_sum_assignment(total_cost.clone()))
        return assignments
