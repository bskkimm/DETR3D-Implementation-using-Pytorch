"""Shared evaluation and visualization helpers for DETR3D experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch

from detr3d.data.nuscenes_dataset import CAMERA_NAMES, NUSCENES_CLASSES
from detr3d.models.losses.loss_utils import decode_bbox_predictions


PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def decode_box_predictions(box_preds: torch.Tensor) -> torch.Tensor:
    return decode_bbox_predictions(box_preds, PC_RANGE)


def geometry_boxes(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 7))
    if boxes.shape[-1] < 7:
        raise ValueError(f"Expected boxes with at least 7 dims, got {boxes.shape[-1]}")
    return boxes[..., :7]


def decode_predictions(
    model,
    sample: dict,
    device: torch.device,
    score_threshold: float,
    max_boxes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        images = sample["images"].unsqueeze(0).to(device)
        img_metas = [
            {
                "lidar2img": sample["img_metas"]["lidar2img"].to(device),
                "image_shape": sample["img_metas"]["image_shape"].to(device),
                "sample_token": sample["img_metas"]["sample_token"],
            }
        ]
        outputs = model(images, img_metas)

    cls_scores = outputs["cls_scores"][-1, 0]
    bbox_preds = outputs["bbox_preds"][-1, 0]
    probs = cls_scores.sigmoid()
    scores, labels = probs.max(dim=-1)

    keep = scores >= score_threshold
    if keep.sum() == 0:
        keep_indices = scores.topk(min(max_boxes, scores.numel())).indices
    else:
        keep_indices = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        keep_scores = scores[keep_indices]
        keep_indices = keep_indices[keep_scores.argsort(descending=True)[:max_boxes]]

    pred_boxes = decode_box_predictions(bbox_preds[keep_indices])
    return pred_boxes.cpu(), scores[keep_indices].cpu(), labels[keep_indices].cpu()


def box7_to_bev_corners(boxes: torch.Tensor) -> torch.Tensor:
    boxes7 = geometry_boxes(boxes)
    if boxes7.numel() == 0:
        return boxes7.new_zeros((0, 4, 2))
    x, y, _, w, l, _, yaw = boxes7.unbind(dim=-1)
    template = boxes7.new_tensor([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])
    corners = template.unsqueeze(0).repeat(boxes7.shape[0], 1, 1)
    corners[..., 0] *= l[:, None]
    corners[..., 1] *= w[:, None]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=-1),
            torch.stack([sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )
    corners = torch.matmul(corners, rotation.transpose(-1, -2))
    return corners + torch.stack([x, y], dim=-1)[:, None, :]


def box7_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    boxes7 = geometry_boxes(boxes)
    if boxes7.numel() == 0:
        return boxes7.new_zeros((0, 8, 3))
    x, y, z, w, l, h, yaw = boxes7.unbind(dim=-1)
    template = boxes7.new_tensor(
        [
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )
    corners = template.unsqueeze(0).repeat(boxes7.shape[0], 1, 1)
    corners[..., 0] *= l[:, None]
    corners[..., 1] *= w[:, None]
    corners[..., 2] *= h[:, None]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw, torch.zeros_like(yaw)], dim=-1),
            torch.stack([sin_yaw, cos_yaw, torch.zeros_like(yaw)], dim=-1),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1),
        ],
        dim=-2,
    )
    corners = torch.matmul(corners, rotation.transpose(-1, -2))
    return corners + torch.stack([x, y, z], dim=-1)[:, None, :]


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
    image = (image_tensor.cpu() * std + mean).clamp(0.0, 1.0)
    return image.permute(1, 2, 0)


def get_original_camera_shapes(dataset, sample_index: int, sample_token: str) -> list[tuple[int, int] | None]:
    tables = getattr(dataset, "tables", None)
    samples = getattr(dataset, "samples", None)
    if tables is None or samples is None:
        return [None for _ in CAMERA_NAMES]

    sample_record = None
    if 0 <= sample_index < len(samples):
        candidate = samples[sample_index]
        if candidate.get("token") == sample_token:
            sample_record = candidate
    if sample_record is None:
        for candidate in samples:
            if candidate.get("token") == sample_token:
                sample_record = candidate
                break
    if sample_record is None:
        return [None for _ in CAMERA_NAMES]

    camera_records = tables.camera_data_by_sample_token.get(sample_record["token"], {})
    original_shapes = []
    for camera_name in CAMERA_NAMES:
        record = camera_records.get(camera_name)
        if record is None:
            original_shapes.append(None)
        else:
            original_shapes.append((int(record["height"]), int(record["width"])))
    return original_shapes


def project_corners_to_image(
    corners_ego: torch.Tensor,
    lidar2img: torch.Tensor,
    image_shape: torch.Tensor,
    original_image_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if corners_ego.numel() == 0:
        empty_xy = corners_ego.new_zeros((0, 8, 2))
        empty_mask = torch.zeros((0, 8), dtype=torch.bool)
        return empty_xy, empty_mask, empty_mask
    corners_h = torch.cat([corners_ego, torch.ones_like(corners_ego[..., :1])], dim=-1)
    proj = torch.matmul(corners_h, lidar2img.T.cpu())
    depth = proj[..., 2]
    xy = proj[..., :2] / depth[..., None].clamp(min=1e-5)
    h, w = image_shape.tolist()
    if original_image_shape is not None:
        orig_h, orig_w = original_image_shape
        xy = xy.clone()
        xy[..., 0] *= float(w) / float(orig_w)
        xy[..., 1] *= float(h) / float(orig_h)
    in_front = depth > 1e-5
    in_frame = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    return xy, in_front, in_frame


def _draw_projected_boxes(
    ax,
    corners_2d: torch.Tensor,
    in_front_mask: torch.Tensor,
    in_frame_mask: torch.Tensor,
    color: str,
    label: str,
) -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    first = True
    x_min, x_max = ax.get_xlim()
    y_max, y_min = ax.get_ylim()
    for box_idx in range(corners_2d.shape[0]):
        if int(in_front_mask[box_idx].sum().item()) < 2:
            continue
        if int(in_frame_mask[box_idx].sum().item()) == 0:
            continue
        for start, end in edges:
            if not bool(in_front_mask[box_idx, start] and in_front_mask[box_idx, end]):
                continue
            xs = [float(corners_2d[box_idx, start, 0]), float(corners_2d[box_idx, end, 0])]
            ys = [float(corners_2d[box_idx, start, 1]), float(corners_2d[box_idx, end, 1])]
            xs = [min(max(v, x_min), x_max) for v in xs]
            ys = [min(max(v, y_min), y_max) for v in ys]
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label if first else None)
            first = False


def save_overlay_figure(
    sample: dict,
    pred_boxes: torch.Tensor,
    output_path: Path,
    *,
    original_image_shapes: list[tuple[int, int] | None] | None = None,
) -> None:
    gt_boxes = sample.get("gt_boxes_lidar", sample["gt_boxes_ego"]).cpu()
    pred_corners = box7_to_corners(pred_boxes)
    gt_corners = box7_to_corners(gt_boxes)
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5), constrained_layout=False)
    axes = axes.reshape(-1)
    for cam_idx, ax in enumerate(axes):
        image = denormalize_image(sample["images"][cam_idx])
        lidar2img = sample["img_metas"]["lidar2img"][cam_idx]
        image_shape = sample["img_metas"]["image_shape"][cam_idx]
        original_image_shape = None if original_image_shapes is None else original_image_shapes[cam_idx]
        ax.imshow(image.tolist())
        gt_xy, gt_front, gt_frame = project_corners_to_image(gt_corners, lidar2img, image_shape, original_image_shape)
        pred_xy, pred_front, pred_frame = project_corners_to_image(pred_corners, lidar2img, image_shape, original_image_shape)
        _draw_projected_boxes(ax, gt_xy, gt_front, gt_frame, color="lime", label="GT")
        _draw_projected_boxes(ax, pred_xy, pred_front, pred_frame, color="red", label="Pred")
        ax.set_title(CAMERA_NAMES[cam_idx])
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right")
    fig.suptitle(f"sample token={sample['img_metas']['sample_token']}", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.90, wspace=0.02, hspace=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_bev_figure(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    draw_bev_boxes(ax, gt_boxes, color="lime", label="GT")
    draw_bev_boxes(ax, pred_boxes, color="red", label="Pred", scores=pred_scores)
    ax.set_xlabel("x in lidar frame (m)")
    ax.set_ylabel("y in lidar frame (m)")
    ax.set_title("BEV GT vs predicted boxes")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def draw_bev_boxes(ax, boxes: torch.Tensor, color: str, label: str, scores: torch.Tensor | None = None) -> None:
    boxes7 = geometry_boxes(boxes)
    if boxes7.numel() == 0:
        return
    corners = box7_to_bev_corners(boxes7)
    for idx in range(corners.shape[0]):
        pts = corners[idx]
        closed = torch.cat([pts, pts[:1]], dim=0)
        ax.plot(closed[:, 0].tolist(), closed[:, 1].tolist(), color=color, linewidth=1.5, label=label if idx == 0 else None)
        ax.scatter([float(boxes7[idx, 0])], [float(boxes7[idx, 1])], c=color, s=20)
        if scores is not None:
            ax.text(float(boxes7[idx, 0]), float(boxes7[idx, 1]), f"{float(scores[idx]):.2f}", color=color, fontsize=8)


def summarize_sample(sample: dict, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor, *, verbose: bool = True) -> dict:
    gt_boxes = sample.get("gt_boxes_lidar", sample["gt_boxes_ego"]).cpu()
    gt_labels = sample["gt_labels"].cpu()

    summary: dict[str, object] = {
        "sample_token": sample["img_metas"]["sample_token"],
        "num_gt": int(gt_boxes.shape[0]),
        "num_pred": int(pred_boxes.shape[0]),
        "gt_classes": [NUSCENES_CLASSES[int(label)] for label in gt_labels.tolist()],
        "top_predictions": [(NUSCENES_CLASSES[int(pred_labels[i])], round(float(pred_scores[i]), 4)) for i in range(len(pred_labels))],
    }

    if verbose:
        print(f"sample_token: {summary['sample_token']}")
        print(f"num_gt={summary['num_gt']}, num_pred={summary['num_pred']}")
        print("\nGT classes:")
        print(summary["gt_classes"])
        print("\nTop predicted classes:")
        print(summary["top_predictions"])

    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        if verbose:
            print("\nCannot compute matching summary because GT or predictions are empty.")
        summary.update(
            {
                "mean_center_distance": None,
                "median_center_distance": None,
                "class_matches": 0,
                "assignments": [],
                "query_usage": [],
            }
        )
        return summary

    center_dist = torch.cdist(gt_boxes[:, :3], pred_boxes[:, :3], p=2)
    best_dist, best_pred_idx = center_dist.min(dim=1)

    assignments = []
    matched_class_count = 0
    if verbose:
        print("\nNearest prediction for each GT:")
    for gt_idx in range(gt_boxes.shape[0]):
        pred_idx = int(best_pred_idx[gt_idx])
        gt_name = NUSCENES_CLASSES[int(gt_labels[gt_idx])]
        pred_name = NUSCENES_CLASSES[int(pred_labels[pred_idx])]
        class_match = int(gt_labels[gt_idx]) == int(pred_labels[pred_idx])
        matched_class_count += int(class_match)
        assignment = {
            "gt_index": gt_idx,
            "gt_class": gt_name,
            "pred_index": pred_idx,
            "pred_class": pred_name,
            "score": float(pred_scores[pred_idx]),
            "center_distance": float(best_dist[gt_idx]),
            "class_match": bool(class_match),
            "gt_wlh": [float(gt_boxes[gt_idx, 3]), float(gt_boxes[gt_idx, 4]), float(gt_boxes[gt_idx, 5])],
            "pred_wlh": [float(pred_boxes[pred_idx, 3]), float(pred_boxes[pred_idx, 4]), float(pred_boxes[pred_idx, 5])],
            "gt_velocity": [float(gt_boxes[gt_idx, 7]), float(gt_boxes[gt_idx, 8])],
            "pred_velocity": [float(pred_boxes[pred_idx, 7]), float(pred_boxes[pred_idx, 8])],
        }
        assignments.append(assignment)
        if verbose:
            print(
                f"GT {gt_idx:02d}: class={gt_name:<22} | pred={pred_idx:02d} ({pred_name:<22}) | "
                f"score={assignment['score']:.4f} | center_dist={assignment['center_distance']:.3f} m | "
                f"class_match={assignment['class_match']}"
            )
            print(
                f"         gt_wlh=({assignment['gt_wlh'][0]:.2f}, {assignment['gt_wlh'][1]:.2f}, {assignment['gt_wlh'][2]:.2f}) | "
                f"pred_wlh=({assignment['pred_wlh'][0]:.2f}, {assignment['pred_wlh'][1]:.2f}, {assignment['pred_wlh'][2]:.2f})"
            )

    counts: dict[int, int] = {}
    for pred_idx in best_pred_idx.tolist():
        counts[pred_idx] = counts.get(pred_idx, 0) + 1

    query_usage = []
    if verbose:
        print("\nGT-to-query assignment counts:")
    for pred_idx, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        pred_name = NUSCENES_CLASSES[int(pred_labels[pred_idx])]
        pred_box = pred_boxes[pred_idx]
        entry = {
            "pred_index": pred_idx,
            "pred_class": pred_name,
            "score": float(pred_scores[pred_idx]),
            "used_by_gt": count,
            "center": [float(pred_box[0]), float(pred_box[1]), float(pred_box[2])],
            "size": [float(pred_box[3]), float(pred_box[4]), float(pred_box[5])],
            "yaw": float(pred_box[6]),
            "velocity": [float(pred_box[7]), float(pred_box[8])],
        }
        query_usage.append(entry)
        if verbose:
            print(
                f"pred={pred_idx:02d} class={pred_name:<18} score={entry['score']:.4f} used_by_gt={count} "
                f"center=({entry['center'][0]:.2f}, {entry['center'][1]:.2f}, {entry['center'][2]:.2f}) "
                f"size=({entry['size'][0]:.2f}, {entry['size'][1]:.2f}, {entry['size'][2]:.2f}) yaw={entry['yaw']:.2f} "
                f"vel=({entry['velocity'][0]:.2f}, {entry['velocity'][1]:.2f})"
            )

    mean_dist = float(best_dist.mean())
    median_dist = float(best_dist.median())
    summary.update(
        {
            "mean_center_distance": mean_dist,
            "median_center_distance": median_dist,
            "class_matches": matched_class_count,
            "assignments": assignments,
            "query_usage": query_usage,
        }
    )
    if verbose:
        print("\nAggregate summary:")
        print(f"mean nearest-center distance: {mean_dist:.3f} m")
        print(f"median nearest-center distance: {median_dist:.3f} m")
        print(f"class matches among nearest preds: {matched_class_count}/{gt_boxes.shape[0]}")

    return summary


def evaluate_samples(
    model,
    dataset,
    sample_indices: Iterable[int],
    device: torch.device,
    score_threshold: float,
    max_boxes: int,
    *,
    overlay_dir: Path | None = None,
    bev_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    sample_summaries = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        pred_boxes, pred_scores, pred_labels = decode_predictions(
            model=model,
            sample=sample,
            device=device,
            score_threshold=score_threshold,
            max_boxes=max_boxes,
        )
        if verbose:
            print(f"\n=== sample_index={sample_index} ===")
        summary = summarize_sample(sample, pred_boxes, pred_scores, pred_labels, verbose=verbose)
        summary["sample_index"] = sample_index
        sample_summaries.append(summary)

        token = sample["img_metas"]["sample_token"]
        original_image_shapes = get_original_camera_shapes(dataset, sample_index, token)
        if overlay_dir is not None:
            save_overlay_figure(
                sample,
                pred_boxes,
                overlay_dir / f"{sample_index:04d}_{token}_overlay.png",
                original_image_shapes=original_image_shapes,
            )
        if bev_dir is not None:
            save_bev_figure(sample["gt_boxes_ego"].cpu(), pred_boxes, pred_scores, bev_dir / f"{sample_index:04d}_{token}_bev.png")

    valid_dists = [row["mean_center_distance"] for row in sample_summaries if row["mean_center_distance"] is not None]
    valid_medians = [row["median_center_distance"] for row in sample_summaries if row["median_center_distance"] is not None]
    aggregate = {
        "num_samples": len(sample_summaries),
        "sample_summaries": sample_summaries,
        "mean_center_distance": float(sum(valid_dists) / len(valid_dists)) if valid_dists else None,
        "mean_median_center_distance": float(sum(valid_medians) / len(valid_medians)) if valid_medians else None,
        "total_class_matches": int(sum(row["class_matches"] for row in sample_summaries)),
        "total_gt": int(sum(row["num_gt"] for row in sample_summaries)),
    }
    return aggregate


def parse_sample_indices(text: str | None, dataset_length: int, limit: int | None = None) -> list[int]:
    if text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    upper = dataset_length if limit is None else min(dataset_length, limit)
    return list(range(upper))


def write_summary_json(output_path: Path, summary: dict) -> None:
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
