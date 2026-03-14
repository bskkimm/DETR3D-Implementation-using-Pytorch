"""DETR3D loss implementation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .loss_utils import box3d_to_10d, expand_bbox_preds, normalize_bbox
from .matcher import HungarianMatcher3D


class Detr3DLoss:
    def __init__(
        self,
        num_classes: int,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        code_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        bg_cls_weight: float = 0.1,
    ):
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.bg_cls_weight = bg_cls_weight
        self.code_weights = torch.tensor(code_weights, dtype=torch.float32)
        self.matcher = HungarianMatcher3D(num_classes=num_classes, pc_range=pc_range)

    def _loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_queries, _ = cls_scores.shape
        bbox_preds_10d = expand_bbox_preds(bbox_preds)
        assignments = self.matcher(cls_scores, bbox_preds_10d, gt_boxes, gt_labels)

        labels = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.long,
            device=cls_scores.device,
        )
        bbox_targets = torch.zeros_like(bbox_preds_10d)
        bbox_weights = torch.zeros_like(bbox_preds_10d)

        num_pos = 0
        num_neg = batch_size * num_queries
        for batch_idx, (pred_ids, gt_ids) in enumerate(assignments):
            if pred_ids.numel() == 0:
                continue
            packed_gt = box3d_to_10d(gt_boxes[batch_idx])[gt_ids]
            labels[batch_idx, pred_ids] = gt_labels[batch_idx][gt_ids]
            bbox_targets[batch_idx, pred_ids] = packed_gt
            bbox_weights[batch_idx, pred_ids] = 1.0
            num_pos += pred_ids.numel()
            num_neg -= pred_ids.numel()

        cls_avg_factor = max(float(num_pos) + self.bg_cls_weight * float(num_neg), 1.0)
        loss_cls = F.cross_entropy(cls_scores.reshape(-1, cls_scores.shape[-1]), labels.reshape(-1), reduction="sum")
        loss_cls = loss_cls / cls_avg_factor

        bbox_preds_norm = normalize_bbox(bbox_preds_10d, self.pc_range)
        bbox_targets_norm = normalize_bbox(bbox_targets, self.pc_range)
        code_weights = self.code_weights.to(bbox_preds.device).view(1, 1, -1)
        l1 = F.l1_loss(bbox_preds_norm, bbox_targets_norm, reduction="none")
        loss_bbox = (l1 * bbox_weights * code_weights).sum() / max(float(num_pos), 1.0)
        return {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

    def loss_by_feat(
        self,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        batch_gt_boxes7: List[torch.Tensor],
        batch_gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses_cls = []
        losses_bbox = []
        for layer_idx in range(all_cls_scores.shape[0]):
            loss = self._loss_single(
                cls_scores=all_cls_scores[layer_idx],
                bbox_preds=all_bbox_preds[layer_idx],
                gt_boxes=batch_gt_boxes7,
                gt_labels=batch_gt_labels,
            )
            losses_cls.append(loss["loss_cls"])
            losses_bbox.append(loss["loss_bbox"])

        output = {
            "loss_cls": losses_cls[-1],
            "loss_bbox": losses_bbox[-1],
        }
        for layer_idx in range(len(losses_cls) - 1):
            output[f"d{layer_idx}.loss_cls"] = losses_cls[layer_idx]
            output[f"d{layer_idx}.loss_bbox"] = losses_bbox[layer_idx]
        return output
