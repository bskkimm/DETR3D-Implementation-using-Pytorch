"""Matching utilities for DETR-style training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .loss_utils import encode_bbox_targets


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


def _box_l1_cost_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    pred = pred_boxes[:, None, :].expand(-1, gt_boxes.shape[0], -1)
    gt = gt_boxes[None, :, :].expand(pred_boxes.shape[0], -1, -1)
    return (pred - gt).abs().sum(dim=-1)


def _focal_class_cost(
    cls_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    probs = cls_logits.sigmoid().clamp(min=1e-8, max=1 - 1e-8)
    neg_cost = -(1 - probs).log() * (1 - alpha) * probs.pow(gamma)
    pos_cost = -(probs).log() * alpha * (1 - probs).pow(gamma)
    return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]


def _cost_stats(cost: torch.Tensor) -> Dict[str, float]:
    if cost.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(cost.mean().item()),
        "min": float(cost.min().item()),
        "max": float(cost.max().item()),
    }


class HungarianMatcher3D:
    def __init__(
        self,
        num_classes: int,
        pc_range=None,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        alpha: float = 0.25,
        gamma: float = 2.0,
        debug: bool = False,
    ):
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.alpha = alpha
        self.gamma = gamma
        self.debug = debug
        self.last_debug: List[Dict[str, float]] = []

    @torch.no_grad()
    def __call__(
        self,
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ):
        cls_logits = cls_logits.float()
        box_preds = box_preds.float()
        assignments = []
        debug_rows: List[Dict[str, float]] = []
        for batch_idx in range(cls_logits.shape[0]):
            if gt_boxes[batch_idx].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=cls_logits.device)
                assignments.append((empty, empty))
                if self.debug:
                    debug_rows.append(
                        {
                            "matched_queries": 0.0,
                            "num_gt": 0.0,
                            "num_queries": float(cls_logits.shape[1]),
                            "cls_cost_mean": 0.0,
                            "cls_cost_min": 0.0,
                            "cls_cost_max": 0.0,
                            "bbox_cost_mean": 0.0,
                            "bbox_cost_min": 0.0,
                            "bbox_cost_max": 0.0,
                            "total_cost_mean": 0.0,
                            "total_cost_min": 0.0,
                            "total_cost_max": 0.0,
                            "matched_cls_cost_mean": 0.0,
                            "matched_bbox_cost_mean": 0.0,
                            "matched_total_cost_mean": 0.0,
                        }
                    )
                continue

            encoded_gt = encode_bbox_targets(gt_boxes[batch_idx].float(), self.pc_range).to(box_preds.dtype)
            cls_cost = _focal_class_cost(
                cls_logits[batch_idx],
                gt_labels[batch_idx],
                alpha=self.alpha,
                gamma=self.gamma,
            )
            bbox_cost = _box_l1_cost_matrix(box_preds[batch_idx], encoded_gt)
            total_cost = self.cls_weight * cls_cost + self.bbox_weight * bbox_cost
            pred_ids, gt_ids = _linear_sum_assignment(total_cost.clone())
            assignments.append((pred_ids, gt_ids))
            if self.debug:
                cls_stats = _cost_stats(cls_cost)
                bbox_stats = _cost_stats(bbox_cost)
                total_stats = _cost_stats(total_cost)
                matched_cls = cls_cost[pred_ids, gt_ids] if pred_ids.numel() > 0 else cls_cost.new_zeros((0,))
                matched_bbox = bbox_cost[pred_ids, gt_ids] if pred_ids.numel() > 0 else bbox_cost.new_zeros((0,))
                matched_total = total_cost[pred_ids, gt_ids] if pred_ids.numel() > 0 else total_cost.new_zeros((0,))
                debug_rows.append(
                    {
                        "matched_queries": float(pred_ids.numel()),
                        "num_gt": float(gt_boxes[batch_idx].shape[0]),
                        "num_queries": float(cls_logits.shape[1]),
                        "cls_cost_mean": cls_stats["mean"],
                        "cls_cost_min": cls_stats["min"],
                        "cls_cost_max": cls_stats["max"],
                        "bbox_cost_mean": bbox_stats["mean"],
                        "bbox_cost_min": bbox_stats["min"],
                        "bbox_cost_max": bbox_stats["max"],
                        "total_cost_mean": total_stats["mean"],
                        "total_cost_min": total_stats["min"],
                        "total_cost_max": total_stats["max"],
                        "matched_cls_cost_mean": float(matched_cls.mean().item()) if matched_cls.numel() > 0 else 0.0,
                        "matched_bbox_cost_mean": float(matched_bbox.mean().item()) if matched_bbox.numel() > 0 else 0.0,
                        "matched_total_cost_mean": float(matched_total.mean().item()) if matched_total.numel() > 0 else 0.0,
                    }
                )
        self.last_debug = debug_rows
        return assignments
