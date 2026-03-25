"""nuScenes dataset implementation aligned with the notebook batch contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUSCENES_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(NUSCENES_CLASSES)}


def load_table(meta_root: Path, name: str) -> List[dict]:
    with (meta_root / f"{name}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_index(rows: List[dict], key: str = "token") -> Dict[str, dict]:
    return {row[key]: row for row in rows}


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_to_matrix(rotation, translation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_to_rotation_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def yaw_from_rotation_matrix(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def resize_and_normalize_image(image: Image.Image, image_size=(900, 1600)) -> torch.Tensor:
    if image_size is not None and image.size != (image_size[1], image_size[0]):
        image = image.resize((image_size[1], image_size[0]))
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.tensor(array, dtype=torch.float32).permute(2, 0, 1)


def category_to_detection_class(category_name: str) -> str | None:
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    return None


@dataclass
class NuScenesTables:
    samples: List[dict]
    sample_by_token: Dict[str, dict]
    sample_data_by_token: Dict[str, dict]
    sensor_by_token: Dict[str, dict]
    camera_data_by_sample_token: Dict[str, Dict[str, dict]]
    annotation_by_token: Dict[str, dict]
    annotations_by_sample_token: Dict[str, List[dict]]
    instance_by_token: Dict[str, dict]
    calibrated_sensor_by_token: Dict[str, dict]
    ego_pose_by_token: Dict[str, dict]
    category_by_token: Dict[str, dict]

    @classmethod
    def from_dataroot(cls, dataroot: Path, version: str) -> "NuScenesTables":
        meta_root = dataroot / version
        sample_table = load_table(meta_root, "sample")
        sample_data_table = load_table(meta_root, "sample_data")
        annotation_table = load_table(meta_root, "sample_annotation")
        instance_table = load_table(meta_root, "instance")
        calibrated_sensor_table = load_table(meta_root, "calibrated_sensor")
        sensor_table = load_table(meta_root, "sensor")

        calibrated_sensor_by_token = build_index(calibrated_sensor_table)
        sensor_by_token = build_index(sensor_table)
        camera_data_by_sample_token: Dict[str, Dict[str, dict]] = {}
        for row in sample_data_table:
            if row["fileformat"] != "jpg" or not row["is_key_frame"]:
                continue
            calibrated = calibrated_sensor_by_token[row["calibrated_sensor_token"]]
            sensor = sensor_by_token[calibrated["sensor_token"]]
            channel = sensor["channel"]
            if channel not in CAMERA_NAMES:
                continue
            camera_data_by_sample_token.setdefault(row["sample_token"], {})[channel] = row

        annotations_by_sample_token: Dict[str, List[dict]] = {}
        for row in annotation_table:
            annotations_by_sample_token.setdefault(row["sample_token"], []).append(row)

        return cls(
            samples=sample_table,
            sample_by_token=build_index(sample_table),
            sample_data_by_token=build_index(sample_data_table),
            sensor_by_token=sensor_by_token,
            camera_data_by_sample_token=camera_data_by_sample_token,
            annotation_by_token=build_index(annotation_table),
            annotations_by_sample_token=annotations_by_sample_token,
            instance_by_token=build_index(instance_table),
            calibrated_sensor_by_token=calibrated_sensor_by_token,
            ego_pose_by_token=build_index(load_table(meta_root, "ego_pose")),
            category_by_token=build_index(load_table(meta_root, "category")),
        )


class NuScenesDetr3DDataset(Dataset):
    """Builds DETR3D-ready samples directly from extracted nuScenes files."""

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-trainval",
        image_size: tuple[int, int] = (900, 1600),
        max_samples: int | None = None,
    ):
        self.dataroot = Path(dataroot)
        self.version = version
        self.image_size = image_size
        self.tables = NuScenesTables.from_dataroot(self.dataroot, version)
        self.samples = self.tables.samples[:max_samples] if max_samples is not None else self.tables.samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_camera_records(self, sample_record: dict) -> Dict[str, dict]:
        camera_records = self.tables.camera_data_by_sample_token.get(sample_record["token"], {})
        missing = [camera_name for camera_name in CAMERA_NAMES if camera_name not in camera_records]
        if missing:
            raise KeyError(f"Missing camera keyframes for sample {sample_record['token']}: {missing}")
        return camera_records

    def _get_lidar_record(self, sample_record: dict) -> dict:
        sample_token = sample_record["token"]
        for row in self.tables.sample_data_by_token.values():
            if row.get("sample_token") != sample_token:
                continue
            calibrated = self.tables.calibrated_sensor_by_token[row["calibrated_sensor_token"]]
            sensor = self.tables.sensor_by_token[calibrated["sensor_token"]]
            if sensor.get("channel") == "LIDAR_TOP":
                return row
        raise KeyError(f"Missing LIDAR_TOP for sample {sample_token}")

    def _sensor_to_global(self, sample_data_record: dict) -> np.ndarray:
        calibrated_sensor = self.tables.calibrated_sensor_by_token[sample_data_record["calibrated_sensor_token"]]
        ego_pose = self.tables.ego_pose_by_token[sample_data_record["ego_pose_token"]]
        sensor_to_ego = pose_to_matrix(calibrated_sensor["rotation"], calibrated_sensor["translation"])
        ego_to_global = pose_to_matrix(ego_pose["rotation"], ego_pose["translation"])
        return ego_to_global @ sensor_to_ego

    def _build_lidar2img(self, lidar_record: dict, camera_record: dict) -> np.ndarray:
        lidar_to_global = self._sensor_to_global(lidar_record)
        cam_to_global = self._sensor_to_global(camera_record)
        global_to_cam = invert_se3(cam_to_global)
        lidar_to_cam = global_to_cam @ lidar_to_global

        calibrated_sensor = self.tables.calibrated_sensor_by_token[camera_record["calibrated_sensor_token"]]
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
        return intrinsic @ lidar_to_cam

    def _annotation_timestamp_seconds(self, ann: dict) -> float:
        sample_record = self.tables.sample_by_token[ann["sample_token"]]
        return float(sample_record["timestamp"]) * 1e-6

    def _compute_ann_velocity_global(self, ann: dict) -> np.ndarray:
        current_translation = np.asarray(ann["translation"], dtype=np.float32)
        current_time = self._annotation_timestamp_seconds(ann)
        prev_token = ann.get("prev")
        next_token = ann.get("next")

        prev_ann = self.tables.annotation_by_token.get(prev_token) if prev_token else None
        next_ann = self.tables.annotation_by_token.get(next_token) if next_token else None

        if prev_ann is not None and next_ann is not None:
            prev_translation = np.asarray(prev_ann["translation"], dtype=np.float32)
            next_translation = np.asarray(next_ann["translation"], dtype=np.float32)
            prev_time = self._annotation_timestamp_seconds(prev_ann)
            next_time = self._annotation_timestamp_seconds(next_ann)
            dt = max(next_time - prev_time, 1e-6)
            return (next_translation - prev_translation) / dt
        if prev_ann is not None:
            prev_translation = np.asarray(prev_ann["translation"], dtype=np.float32)
            prev_time = self._annotation_timestamp_seconds(prev_ann)
            dt = max(current_time - prev_time, 1e-6)
            return (current_translation - prev_translation) / dt
        if next_ann is not None:
            next_translation = np.asarray(next_ann["translation"], dtype=np.float32)
            next_time = self._annotation_timestamp_seconds(next_ann)
            dt = max(next_time - current_time, 1e-6)
            return (next_translation - current_translation) / dt
        return np.zeros(3, dtype=np.float32)

    def _transform_global_ann_to_lidar(self, ann: dict, lidar_record: dict) -> tuple[np.ndarray, float, np.ndarray]:
        lidar_to_global = self._sensor_to_global(lidar_record)
        global_to_lidar = invert_se3(lidar_to_global)

        center_global = np.ones(4, dtype=np.float32)
        center_global[:3] = np.asarray(ann["translation"], dtype=np.float32)
        center_lidar = global_to_lidar @ center_global

        ann_rotation_global = quaternion_to_rotation_matrix(ann["rotation"])
        ann_rotation_lidar = global_to_lidar[:3, :3] @ ann_rotation_global
        yaw_lidar = yaw_from_rotation_matrix(ann_rotation_lidar)

        velocity_global = self._compute_ann_velocity_global(ann)
        velocity_lidar = global_to_lidar[:3, :3] @ velocity_global
        return center_lidar[:3], yaw_lidar, velocity_lidar[:2]

    def _build_gt_targets(self, sample_record: dict, lidar_record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        labels = []
        for ann in self.tables.annotations_by_sample_token.get(sample_record["token"], []):
            instance = self.tables.instance_by_token[ann["instance_token"]]
            category = self.tables.category_by_token[instance["category_token"]]["name"]
            det_class = category_to_detection_class(category)
            if det_class is None:
                continue
            center_lidar, yaw, velocity_lidar = self._transform_global_ann_to_lidar(ann, lidar_record)
            x, y, z = center_lidar.tolist()
            w, l, h = ann["size"]
            vx, vy = velocity_lidar.tolist()
            boxes.append([x, y, z, w, l, h, yaw, vx, vy])
            labels.append(CLASS_TO_ID[det_class])

        if not boxes:
            return torch.zeros((0, 9), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        sample_record = self.samples[index]
        camera_records = self._get_camera_records(sample_record)
        lidar_record = self._get_lidar_record(sample_record)

        images = []
        lidar2img = []
        image_shape = []
        for camera_name in CAMERA_NAMES:
            record = camera_records[camera_name]
            image_path = self.dataroot / record["filename"]
            image = Image.open(image_path).convert("RGB")
            images.append(resize_and_normalize_image(image, image_size=self.image_size))
            lidar2img.append(torch.tensor(self._build_lidar2img(lidar_record, record), dtype=torch.float32))
            image_shape.append(torch.tensor(self.image_size, dtype=torch.float32))

        gt_boxes_lidar, gt_labels = self._build_gt_targets(sample_record, lidar_record)
        return {
            "images": torch.stack(images, dim=0),
            "img_metas": {
                "lidar2img": torch.stack(lidar2img, dim=0),
                "image_shape": torch.stack(image_shape, dim=0),
                "sample_token": sample_record["token"],
            },
            "gt_boxes_ego": gt_boxes_lidar,
            "gt_boxes_lidar": gt_boxes_lidar,
            "gt_labels": gt_labels,
        }


__all__ = ["NuScenesDetr3DDataset", "NuScenesTables", "CAMERA_NAMES", "NUSCENES_CLASSES"]
