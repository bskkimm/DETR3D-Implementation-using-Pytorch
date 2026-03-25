"""DETR3D loss implementation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .loss_utils import decode_bbox_predictions, encode_bbox_targets, wrapped_yaw_difference
from .matcher import HungarianMatcher3D


class Detr3DLoss:
    def __init__(
        self,
        num_classes: int,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        code_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
        matcher_cls_weight: float = 2.0,
        matcher_bbox_weight: float = 0.25,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 0.25,
        use_auxiliary_losses: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        bg_cls_weight: float = 0.1,
        box_group_weights = (1.0, 1.0, 1.0, 1.0),
        debug: bool = False,
    ):
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.use_auxiliary_losses = use_auxiliary_losses
        self.code_weights = torch.tensor(code_weights, dtype=torch.float32)
        self.box_group_weights = torch.tensor(box_group_weights, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.bg_cls_weight = bg_cls_weight
        self.debug = debug
        self.matcher = HungarianMatcher3D(
            num_classes=num_classes,
            pc_range=pc_range,
            cls_weight=matcher_cls_weight,
            bbox_weight=matcher_bbox_weight,
            alpha=alpha,
            gamma=gamma,
            debug=debug,
        )

    def _loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cls_scores = cls_scores.float()
        bbox_preds = bbox_preds.float()
        batch_size, num_queries, _ = cls_scores.shape
        assignments = self.matcher(cls_scores, bbox_preds, gt_boxes, gt_labels)

        label_indices = torch.full(
            (batch_size, num_queries),
            fill_value=self.num_classes,
            dtype=torch.long,
            device=cls_scores.device,
        )
        bbox_targets = torch.zeros_like(bbox_preds)
        bbox_weights = torch.zeros_like(bbox_preds)

        num_pos = 0
        for batch_idx, (pred_ids, gt_ids) in enumerate(assignments):
            if pred_ids.numel() == 0:
                continue
            label_indices[batch_idx, pred_ids] = gt_labels[batch_idx][gt_ids]
            encoded_gt = encode_bbox_targets(gt_boxes[batch_idx].float()[gt_ids], self.pc_range).to(bbox_targets.dtype)
            bbox_targets[batch_idx, pred_ids] = encoded_gt
            bbox_weights[batch_idx, pred_ids] = 1.0
            num_pos += pred_ids.numel()

        num_total = batch_size * num_queries
        num_neg = num_total - num_pos
        normalizer = max(float(num_pos), 1.0)
        cls_avg_factor = max(float(num_pos) + self.bg_cls_weight * float(num_neg), 1.0)
        labels = torch.zeros((batch_size, num_queries, self.num_classes), dtype=cls_scores.dtype, device=cls_scores.device)
        pos_mask = label_indices != self.num_classes
        if pos_mask.any():
            batch_ids, query_ids = pos_mask.nonzero(as_tuple=True)
            labels[batch_ids, query_ids, label_indices[batch_ids, query_ids]] = 1.0

        pred_sigmoid = cls_scores.sigmoid()
        pt = pred_sigmoid * labels + (1 - pred_sigmoid) * (1 - labels)
        focal_weight = (self.alpha * labels + (1 - self.alpha) * (1 - labels)) * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(cls_scores, labels, reduction="none")
        loss_cls_raw = (bce * focal_weight).sum() / cls_avg_factor
        loss_cls = loss_cls_raw * self.loss_cls_weight

        code_weights = self.code_weights.to(bbox_preds.device).view(1, 1, -1)
        abs_diff = (bbox_preds - bbox_targets).abs() * bbox_weights * code_weights
        loss_bbox_raw = abs_diff.sum() / normalizer
        loss_bbox = loss_bbox_raw * self.loss_bbox_weight

        output = {"loss_cls": loss_cls, "loss_bbox": loss_bbox}
        if self.debug:
            semantic_preds = decode_bbox_predictions(bbox_preds, self.pc_range)
            semantic_targets = decode_bbox_predictions(bbox_targets, self.pc_range)
            semantic_weights = bbox_weights[..., :9]
            group_weights = self.box_group_weights.to(bbox_preds.device)
            semantic_diff = (semantic_preds - semantic_targets).abs() * semantic_weights
            semantic_diff[..., 6] = wrapped_yaw_difference(semantic_preds[..., 6], semantic_targets[..., 6]).abs() * semantic_weights[..., 6]

            center_loss = semantic_diff[..., 0:3].sum() / normalizer
            size_loss = semantic_diff[..., 3:6].sum() / normalizer
            yaw_loss = semantic_diff[..., 6].sum() / normalizer
            velocity_loss = semantic_diff[..., 7:9].sum() / normalizer

            output.update(
                {
                    "debug_num_pos": loss_cls.new_tensor(float(num_pos)),
                    "debug_num_neg": loss_cls.new_tensor(float(num_neg)),
                    "debug_cls_avg_factor": loss_cls.new_tensor(float(cls_avg_factor)),
                    "debug_bbox_center": center_loss * group_weights[0] * self.loss_bbox_weight,
                    "debug_bbox_size": size_loss * group_weights[1] * self.loss_bbox_weight,
                    "debug_bbox_yaw": yaw_loss * group_weights[2] * self.loss_bbox_weight,
                    "debug_bbox_velocity": velocity_loss * group_weights[3] * self.loss_bbox_weight,
                }
            )
            matcher_debug = getattr(self.matcher, "last_debug", [])
            if matcher_debug:
                matcher_mean = {
                    key: sum(row[key] for row in matcher_debug) / max(len(matcher_debug), 1)
                    for key in matcher_debug[0]
                }
                for key, value in matcher_mean.items():
                    output[f"debug_matcher_{key}"] = loss_cls.new_tensor(value)
        return output

    def loss_by_feat(
        self,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        batch_gt_boxes7: List[torch.Tensor],
        batch_gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.use_auxiliary_losses:
            return self._loss_single(
                cls_scores=all_cls_scores[-1],
                bbox_preds=all_bbox_preds[-1],
                gt_boxes=batch_gt_boxes7,
                gt_labels=batch_gt_labels,
            )

        losses = []
        for layer_idx in range(all_cls_scores.shape[0]):
            losses.append(
                self._loss_single(
                    cls_scores=all_cls_scores[layer_idx],
                    bbox_preds=all_bbox_preds[layer_idx],
                    gt_boxes=batch_gt_boxes7,
                    gt_labels=batch_gt_labels,
                )
            )

        output = {
            "loss_cls": losses[-1]["loss_cls"],
            "loss_bbox": losses[-1]["loss_bbox"],
        }
        for layer_idx in range(len(losses) - 1):
            output[f"d{layer_idx}.loss_cls"] = losses[layer_idx]["loss_cls"]
            output[f"d{layer_idx}.loss_bbox"] = losses[layer_idx]["loss_bbox"]
        for key, value in losses[-1].items():
            if key not in output and not key.startswith("loss_"):
                output[key] = value
        return output
