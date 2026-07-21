from types import SimpleNamespace

import numpy as np

from detr3d.data.nuscenes_dataset import (
    CLASS_TO_ID,
    NuScenesDetr3DDataset,
    category_to_detection_class,
)


def make_dataset(annotations, categories, centers, official=True):
    dataset = NuScenesDetr3DDataset.__new__(NuScenesDetr3DDataset)
    dataset.samples = [{"token": "sample"}]
    dataset.tables = SimpleNamespace(
        lidar_data_by_sample_token={"sample": {"token": "lidar"}},
        annotations_by_sample_token={"sample": annotations},
        instance_by_token={
            ann["instance_token"]: {"category_token": ann["instance_token"]}
            for ann in annotations
        },
        category_by_token={key: {"name": value} for key, value in categories.items()},
    )
    dataset.point_cloud_range = (-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)
    dataset.official_gt_semantics = official
    dataset.filter_gt_by_range = False
    dataset.filter_zero_point_gt = False
    dataset._transform_global_ann_to_lidar = lambda ann, _: (
        np.asarray(centers[ann["instance_token"]], dtype=np.float32),
        0.0,
        np.zeros(2, dtype=np.float32),
    )
    return dataset


def annotation(name, points=1):
    return {
        "instance_token": name,
        "num_lidar_pts": points,
        "num_radar_pts": 0,
        "size": [1.0, 1.0, 1.0],
    }


def test_official_mapping_uses_only_exact_pedestrian_categories():
    accepted = (
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    )
    for category in accepted:
        assert category_to_detection_class(category, official=True) == "pedestrian"

    assert (
        category_to_detection_class("human.pedestrian.stroller", official=True)
        is None
    )
    assert (
        category_to_detection_class("human.pedestrian.wheelchair", official=True)
        is None
    )
    assert category_to_detection_class("human.pedestrian.stroller") == "pedestrian"


def test_official_filter_is_strict_xy_only_and_requires_points():
    annotations = [
        annotation("inside"),
        annotation("x_min"),
        annotation("x_max"),
        annotation("y_min"),
        annotation("y_max"),
        annotation("z_outside"),
        annotation("empty", points=0),
    ]
    categories = {
        ann["instance_token"]: "vehicle.car" for ann in annotations
    }
    centers = {
        "inside": (0.0, 0.0, 0.0),
        "x_min": (-1.0, 0.0, 0.0),
        "x_max": (1.0, 0.0, 0.0),
        "y_min": (0.0, -2.0, 0.0),
        "y_max": (0.0, 2.0, 0.0),
        "z_outside": (0.0, 0.0, 100.0),
        "empty": (0.0, 0.0, 0.0),
    }
    dataset = make_dataset(annotations, categories, centers)

    assert dataset.get_cat_ids(0) == [CLASS_TO_ID["car"]]
    boxes, labels = dataset._build_gt_targets(dataset.samples[0], {"token": "lidar"})
    assert len(labels) == 2
    assert sorted(boxes[:, 2].tolist()) == [0.0, 100.0]


def test_legacy_filter_keeps_inclusive_xyz_behavior():
    annotations = [annotation("boundary"), annotation("high_z")]
    categories = {ann["instance_token"]: "vehicle.car" for ann in annotations}
    centers = {
        "boundary": (-1.0, 2.0, 3.0),
        "high_z": (0.0, 0.0, 3.1),
    }
    dataset = make_dataset(annotations, categories, centers, official=False)
    dataset.filter_gt_by_range = True

    boxes, _ = dataset._build_gt_targets(dataset.samples[0], {"token": "lidar"})

    assert boxes.shape == (1, 9)
    assert boxes[0, :3].tolist() == [-1.0, 2.0, 3.0]


def test_metadata_cat_ids_match_targets_and_are_unique():
    annotations = [annotation("car_a"), annotation("car_b"), annotation("stroller")]
    categories = {
        "car_a": "vehicle.car",
        "car_b": "vehicle.car",
        "stroller": "human.pedestrian.stroller",
    }
    centers = {name: (0.0, 0.0, 0.0) for name in categories}
    dataset = make_dataset(annotations, categories, centers)

    cat_ids = dataset.get_cat_ids(0)
    _, labels = dataset._build_gt_targets(dataset.samples[0], {"token": "lidar"})

    assert cat_ids == [CLASS_TO_ID["car"]]
    assert set(labels.tolist()) == set(cat_ids)
    assert dataset.has_nonempty_gt(0)
