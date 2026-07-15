"""Official nuScenes result export and evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from pyquaternion import Quaternion

from detr3d.data.nuscenes_dataset import (
    NUSCENES_CLASSES,
    NuScenesTables,
    pose_to_matrix,
)
from detr3d.models.losses.loss_utils import decode_bbox_predictions

OFFICIAL_MAX_NUM = 300
OFFICIAL_POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
DEFAULT_ATTRIBUTES = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}


def decode_nuscenes_predictions(
    cls_logits: torch.Tensor,
    bbox_preds: torch.Tensor,
    *,
    max_num: int = OFFICIAL_MAX_NUM,
    post_center_range: Sequence[float] = OFFICIAL_POST_CENTER_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the final decoder layer with official NMS-free top-k behavior."""
    if cls_logits.ndim != 2 or bbox_preds.ndim != 2:
        raise ValueError("Expected unbatched [query, class/box] predictions")
    if len(post_center_range) != 6:
        raise ValueError("post_center_range must contain six values")

    probabilities = cls_logits.float().sigmoid()
    flat_scores = probabilities.reshape(-1)
    count = min(max_num, flat_scores.numel())
    scores, flat_indices = flat_scores.topk(count)
    labels = flat_indices % cls_logits.shape[-1]
    query_indices = torch.div(flat_indices, cls_logits.shape[-1], rounding_mode="floor")
    boxes = decode_bbox_predictions(bbox_preds[query_indices].float())

    lower = boxes.new_tensor(post_center_range[:3])
    upper = boxes.new_tensor(post_center_range[3:])
    keep = ((boxes[:, :3] >= lower) & (boxes[:, :3] <= upper)).all(dim=-1)
    return boxes[keep], scores[keep], labels[keep]


def prediction_attribute(detection_name: str, velocity_xy: Sequence[float]) -> str:
    speed = math.hypot(float(velocity_xy[0]), float(velocity_xy[1]))
    if speed > 0.2:
        if detection_name in {"car", "construction_vehicle", "bus", "truck", "trailer"}:
            return "vehicle.moving"
        if detection_name in {"bicycle", "motorcycle"}:
            return "cycle.with_rider"
    else:
        if detection_name == "pedestrian":
            return "pedestrian.standing"
        if detection_name == "bus":
            return "vehicle.stopped"
    return DEFAULT_ATTRIBUTES[detection_name]


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def lidar_predictions_to_nuscenes(
    *,
    sample_token: str,
    boxes_lidar: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    tables: NuScenesTables,
    class_range: Mapping[str, float],
    class_names: Sequence[str] = NUSCENES_CLASSES,
) -> list[dict]:
    """Convert semantic LiDAR-frame boxes to nuScenes global result records."""
    lidar_record = tables.lidar_data_by_sample_token[sample_token]
    calibrated = tables.calibrated_sensor_by_token[
        lidar_record["calibrated_sensor_token"]
    ]
    ego_pose = tables.ego_pose_by_token[lidar_record["ego_pose_token"]]
    lidar_to_ego = pose_to_matrix(
        calibrated["rotation"], calibrated["translation"]
    ).astype(np.float64)
    ego_to_global = pose_to_matrix(
        ego_pose["rotation"], ego_pose["translation"]
    ).astype(np.float64)
    lidar_to_global = ego_to_global @ lidar_to_ego

    records = []
    for box_tensor, score_tensor, label_tensor in zip(boxes_lidar, scores, labels):
        box = box_tensor.detach().cpu().double().numpy()
        score = float(score_tensor)
        label = int(label_tensor)
        if label < 0 or label >= len(class_names):
            raise ValueError(f"Invalid detection label {label}")
        detection_name = class_names[label]

        center_lidar = np.asarray([box[0], box[1], box[2], 1.0], dtype=np.float64)
        center_ego = lidar_to_ego @ center_lidar
        if np.linalg.norm(center_ego[:2]) > class_range[detection_name]:
            continue

        center_global = lidar_to_global @ center_lidar
        rotation_global = lidar_to_global[:3, :3] @ _yaw_rotation(float(box[6]))
        quaternion = Quaternion(matrix=rotation_global).elements
        velocity_global = lidar_to_global[:3, :3] @ np.asarray(
            [box[7], box[8], 0.0], dtype=np.float64
        )
        size = box[3:6]

        numeric_values = np.concatenate(
            [
                center_global[:3],
                size,
                quaternion,
                velocity_global[:2],
                np.asarray([score]),
            ]
        )
        if not np.isfinite(numeric_values).all() or np.any(size <= 0):
            raise ValueError(
                f"Non-finite or invalid prediction for sample {sample_token}"
            )

        records.append(
            {
                "sample_token": sample_token,
                "translation": center_global[:3].tolist(),
                "size": size.tolist(),
                "rotation": quaternion.tolist(),
                "velocity": velocity_global[:2].tolist(),
                "detection_name": detection_name,
                "detection_score": score,
                "attribute_name": prediction_attribute(
                    detection_name, velocity_global[:2]
                ),
            }
        )
    return records


def build_nuscenes_submission(results: Mapping[str, list[dict]]) -> dict:
    return {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": dict(results),
    }


def export_nuscenes_results(
    *,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    output_path: str | Path,
    max_num: int = OFFICIAL_MAX_NUM,
    post_center_range: Sequence[float] = OFFICIAL_POST_CENTER_RANGE,
    eval_config_name: str = "detection_cvpr_2019",
    verbose: bool = True,
) -> Path:
    from nuscenes.eval.detection.config import config_factory

    config = config_factory(eval_config_name)
    results = {sample["token"]: [] for sample in dataset.samples}
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            images = sample["images"].unsqueeze(0).to(device)
            img_metas = [
                {
                    "lidar2img": sample["img_metas"]["lidar2img"].to(device),
                    "image_shape": sample["img_metas"]["image_shape"].to(device),
                    "sample_token": sample["img_metas"]["sample_token"],
                }
            ]
            outputs = model(images, img_metas)
            boxes, scores, labels = decode_nuscenes_predictions(
                outputs["cls_scores"][-1, 0],
                outputs["bbox_preds"][-1, 0],
                max_num=max_num,
                post_center_range=post_center_range,
            )
            token = sample["img_metas"]["sample_token"]
            results[token] = lidar_predictions_to_nuscenes(
                sample_token=token,
                boxes_lidar=boxes,
                scores=scores,
                labels=labels,
                tables=dataset.tables,
                class_range=config.class_range,
            )
            if verbose and (index + 1) % 100 == 0:
                print(f"exported {index + 1}/{len(dataset)} samples")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_nuscenes_submission(results), handle, indent=2)
    return path


def run_nuscenes_evaluation(
    *,
    dataroot: str | Path,
    version: str,
    result_path: str | Path,
    eval_set: str,
    output_dir: str | Path,
    eval_config_name: str = "detection_cvpr_2019",
    verbose: bool = True,
) -> dict:
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    evaluator = NuScenesEval(
        nusc=NuScenes(version=version, dataroot=str(dataroot), verbose=verbose),
        config=config_factory(eval_config_name),
        result_path=str(result_path),
        eval_set=eval_set,
        output_dir=str(output_dir),
        verbose=verbose,
    )
    return evaluator.main(plot_examples=0, render_curves=False)
