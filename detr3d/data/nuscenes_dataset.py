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


def yaw_from_quaternion(q) -> float:
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_from_rotation_matrix(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def resize_and_normalize_image(image: Image.Image, image_size=(256, 448)) -> torch.Tensor:
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
        image_size: tuple[int, int] = (256, 448),
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

    def _build_lidar2img(self, camera_record: dict) -> np.ndarray:
        calibrated_sensor = self.tables.calibrated_sensor_by_token[camera_record["calibrated_sensor_token"]]

        cam_to_ego = pose_to_matrix(calibrated_sensor["rotation"], calibrated_sensor["translation"])
        ego_to_cam = invert_se3(cam_to_ego)

        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
        return intrinsic @ ego_to_cam

    def _transform_global_ann_to_ego(self, ann: dict, ego_pose: dict) -> tuple[np.ndarray, float]:
        ego_to_global = pose_to_matrix(ego_pose["rotation"], ego_pose["translation"])
        global_to_ego = invert_se3(ego_to_global)

        center_global = np.ones(4, dtype=np.float32)
        center_global[:3] = np.asarray(ann["translation"], dtype=np.float32)
        center_ego = global_to_ego @ center_global

        ann_rotation_global = quaternion_to_rotation_matrix(ann["rotation"])
        ann_rotation_ego = global_to_ego[:3, :3] @ ann_rotation_global
        yaw_ego = yaw_from_rotation_matrix(ann_rotation_ego)
        return center_ego[:3], yaw_ego

    def _build_gt_targets(self, sample_record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        labels = []
        ego_pose = self.tables.ego_pose_by_token[self._get_camera_records(sample_record)["CAM_FRONT"]["ego_pose_token"]]
        for ann in self.tables.annotations_by_sample_token.get(sample_record["token"], []):
            instance = self.tables.instance_by_token[ann["instance_token"]]
            category = self.tables.category_by_token[instance["category_token"]]["name"]
            det_class = category_to_detection_class(category)
            if det_class is None:
                continue
            center_ego, yaw = self._transform_global_ann_to_ego(ann, ego_pose)
            x, y, z = center_ego.tolist()
            w, l, h = ann["size"]
            boxes.append([x, y, z, w, l, h, yaw])
            labels.append(CLASS_TO_ID[det_class])

        if not boxes:
            return torch.zeros((0, 7), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        sample_record = self.samples[index]
        camera_records = self._get_camera_records(sample_record)

        images = []
        lidar2img = []
        image_shape = []
        for camera_name in CAMERA_NAMES:
            record = camera_records[camera_name]
            image_path = self.dataroot / record["filename"]
            image = Image.open(image_path).convert("RGB")
            images.append(resize_and_normalize_image(image, image_size=self.image_size))
            lidar2img.append(torch.tensor(self._build_lidar2img(record), dtype=torch.float32))
            image_shape.append(torch.tensor(self.image_size, dtype=torch.float32))

        gt_boxes_ego, gt_labels = self._build_gt_targets(sample_record)
        return {
            "images": torch.stack(images, dim=0),
            "img_metas": {
                "lidar2img": torch.stack(lidar2img, dim=0),
                "image_shape": torch.stack(image_shape, dim=0),
                "sample_token": sample_record["token"],
            },
            "gt_boxes_ego": gt_boxes_ego,
            "gt_labels": gt_labels,
        }


__all__ = ["NuScenesDetr3DDataset", "NuScenesTables", "CAMERA_NAMES", "NUSCENES_CLASSES"]
