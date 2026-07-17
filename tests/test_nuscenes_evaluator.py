import math

import numpy as np
import pytest
import torch

from detr3d.data.nuscenes_dataset import NuScenesTables
from detr3d.engine.evaluator import (
    build_nuscenes_submission,
    decode_nuscenes_predictions,
    lidar_predictions_to_nuscenes,
    prediction_attribute,
)


def _quaternion(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _tables(
    lidar_yaw=0.0,
    lidar_translation=(0.0, 0.0, 0.0),
    ego_yaw=0.0,
    ego_translation=(0.0, 0.0, 0.0),
):
    lidar_record = {
        "sample_token": "sample",
        "calibrated_sensor_token": "calibrated",
        "ego_pose_token": "pose",
    }
    return NuScenesTables(
        samples=[],
        sample_by_token={},
        scene_by_token={},
        sample_data_by_token={},
        sensor_by_token={},
        camera_data_by_sample_token={},
        lidar_data_by_sample_token={"sample": lidar_record},
        annotation_by_token={},
        annotations_by_sample_token={},
        instance_by_token={},
        calibrated_sensor_by_token={
            "calibrated": {
                "rotation": _quaternion(lidar_yaw),
                "translation": lidar_translation,
            }
        },
        ego_pose_by_token={
            "pose": {"rotation": _quaternion(ego_yaw), "translation": ego_translation}
        },
        category_by_token={},
    )


def test_decode_uses_flattened_query_class_topk():
    logits = torch.tensor([[5.0, 4.0, -5.0], [3.0, -5.0, -5.0]])
    boxes = torch.zeros(2, 10)
    boxes[:, 2:4] = 0.0
    boxes[:, 5] = 0.0
    boxes[:, 7] = 1.0

    decoded, scores, labels = decode_nuscenes_predictions(logits, boxes, max_num=2)

    assert decoded.shape[0] == 2
    assert labels.tolist() == [0, 1]
    torch.testing.assert_close(decoded[0], decoded[1])
    assert scores[0] > scores[1]


def test_decode_filters_after_topk_without_refill():
    logits = torch.tensor([[5.0], [4.0], [3.0]])
    boxes = torch.zeros(3, 10)
    boxes[:, 7] = 1.0
    boxes[0, 0] = 100.0
    boxes[1, 0] = 61.2
    boxes[2, 0] = 0.0

    decoded, _, _ = decode_nuscenes_predictions(logits, boxes, max_num=2)

    assert decoded.shape[0] == 1
    assert decoded[0, 0] == pytest.approx(61.2)


@pytest.mark.parametrize(
    ("name", "velocity", "expected"),
    [
        ("car", (0.21, 0.0), "vehicle.moving"),
        ("car", (0.2, 0.0), "vehicle.parked"),
        ("bus", (0.0, 0.0), "vehicle.stopped"),
        ("pedestrian", (0.0, 0.0), "pedestrian.standing"),
        ("pedestrian", (1.0, 0.0), "pedestrian.moving"),
        ("motorcycle", (1.0, 0.0), "cycle.with_rider"),
        ("traffic_cone", (0.0, 0.0), ""),
    ],
)
def test_prediction_attributes(name, velocity, expected):
    assert prediction_attribute(name, velocity) == expected


def test_lidar_predictions_transform_to_global():
    tables = _tables(
        lidar_yaw=math.pi / 2,
        lidar_translation=(1.0, 0.0, 0.0),
        ego_translation=(10.0, 0.0, 0.0),
    )
    boxes = torch.tensor([[1.0, 0.0, 2.0, 2.0, 4.0, 1.5, math.pi / 3, 1.0, 0.0]])
    records = lidar_predictions_to_nuscenes(
        sample_token="sample",
        boxes_lidar=boxes,
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        tables=tables,
        class_range={
            name: 100.0
            for name in [
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
        },
    )

    record = records[0]
    np.testing.assert_allclose(record["translation"], [11.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(record["velocity"], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(record["size"], [2.0, 4.0, 1.5], atol=1e-6)
    from pyquaternion import Quaternion

    rotation = Quaternion(record["rotation"]).rotation_matrix
    expected_yaw = math.pi / 2 + math.pi / 3
    np.testing.assert_allclose(
        rotation[:2, :2],
        [
            [math.cos(expected_yaw), -math.sin(expected_yaw)],
            [math.sin(expected_yaw), math.cos(expected_yaw)],
        ],
        atol=1e-6,
    )


def test_class_range_uses_ego_center():
    boxes = torch.tensor([[35.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    ranges = {"car": 50.0, "traffic_cone": 30.0}
    car = lidar_predictions_to_nuscenes(
        sample_token="sample",
        boxes_lidar=boxes,
        scores=torch.tensor([0.5]),
        labels=torch.tensor([0]),
        tables=_tables(),
        class_range=ranges,
    )
    cone = lidar_predictions_to_nuscenes(
        sample_token="sample",
        boxes_lidar=boxes,
        scores=torch.tensor([0.5]),
        labels=torch.tensor([8]),
        tables=_tables(),
        class_range=ranges,
    )
    assert len(car) == 1
    assert cone == []


def test_lidar_conversion_uses_supplied_checkpoint_class_order():
    class_names = ["traffic_cone", "car"]
    records = lidar_predictions_to_nuscenes(
        sample_token="sample",
        boxes_lidar=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
        ),
        scores=torch.tensor([0.5]),
        labels=torch.tensor([0]),
        tables=_tables(),
        class_range={"traffic_cone": 30.0},
        class_names=class_names,
    )

    assert records[0]["detection_name"] == "traffic_cone"


def test_submission_schema_loads_with_devkit(tmp_path):
    import json

    from nuscenes.eval.common.loaders import load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox

    record = {
        "sample_token": "sample",
        "translation": [0.0, 0.0, 0.0],
        "size": [1.0, 2.0, 1.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0],
        "detection_name": "car",
        "detection_score": 0.5,
        "attribute_name": "vehicle.parked",
    }
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(build_nuscenes_submission({"sample": [record]})), encoding="utf-8"
    )

    boxes, meta = load_prediction(str(path), 500, DetectionBox, verbose=False)

    assert len(boxes["sample"]) == 1
    assert meta["use_camera"] is True
