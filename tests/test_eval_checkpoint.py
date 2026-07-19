import argparse

import pytest

import eval as eval_module
from detr3d.data.nuscenes_dataset import NUSCENES_CLASSES
from eval import resolve_checkpoint_class_names, resolve_checkpoint_config


def _cli_args(**overrides):
    values = {
        "image_height": 900,
        "image_width": 1600,
        "backbone": "cli-backbone",
        "num_queries": 900,
        "official_image_backbone": False,
        "official_image_preprocessing": False,
        "grid_mask": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_checkpoint_config_prefers_saved_c6_construction_args():
    checkpoint = {
        "args": {
            "image_height": 450,
            "image_width": 800,
            "backbone": "resnet101",
            "num_queries": 600,
            "official_image_backbone": True,
            "official_image_preprocessing": True,
            "grid_mask": True,
        }
    }

    config = resolve_checkpoint_config(_cli_args(), checkpoint)

    assert config == {
        "image_height": 450,
        "image_width": 800,
        "backbone": "resnet101",
        "disable_pretrained_backbone": False,
        "num_queries": 600,
        "official_image_backbone": True,
        "official_image_preprocessing": True,
        "official_gt_semantics": False,
        "grid_mask": True,
    }


def test_checkpoint_config_uses_cli_fallback_for_missing_saved_args():
    args = _cli_args(
        image_height=225,
        image_width=400,
        backbone="resnet101",
        num_queries=300,
        official_image_backbone=True,
        official_image_preprocessing=True,
        grid_mask=True,
    )

    assert resolve_checkpoint_config(args, {"model_state_dict": {}}) == {
        "image_height": 225,
        "image_width": 400,
        "backbone": "resnet101",
        "disable_pretrained_backbone": False,
        "num_queries": 300,
        "official_image_backbone": True,
        "official_image_preprocessing": True,
        "official_gt_semantics": False,
        "grid_mask": True,
    }


def test_existing_checkpoint_uses_repository_class_order():
    assert resolve_checkpoint_class_names({}) == tuple(NUSCENES_CLASSES)


def test_future_checkpoint_accepts_validated_class_permutation():
    names = list(reversed(NUSCENES_CLASSES))

    assert resolve_checkpoint_class_names({"class_names": names}) == tuple(names)


@pytest.mark.parametrize(
    "class_names",
    [NUSCENES_CLASSES[:-1], NUSCENES_CLASSES[:-1] + [NUSCENES_CLASSES[0]]],
)
def test_checkpoint_rejects_incomplete_or_duplicate_class_names(class_names):
    with pytest.raises(ValueError, match="every repository nuScenes class"):
        resolve_checkpoint_class_names({"class_names": class_names})


def test_main_loads_checkpoint_before_c6_construction_and_loads_strictly(
    monkeypatch, tmp_path
):
    events = []
    class_names = list(reversed(NUSCENES_CLASSES))
    checkpoint = {
        "args": {
            "image_height": 450,
            "image_width": 800,
            "backbone": "resnet101",
            "num_queries": 600,
            "official_image_backbone": True,
            "official_image_preprocessing": True,
            "grid_mask": True,
        },
        "class_names": class_names,
        "model_state_dict": {"weight": object()},
    }
    args = argparse.Namespace(
        checkpoint=str(tmp_path / "checkpoint.pt"),
        dataroot="unused",
        version="v1.0-mini",
        max_samples=None,
        dataset_split="mini_val",
        filter_gt_by_range=False,
        filter_zero_point_gt=False,
        disable_pretrained_backbone=True,
        device="cpu",
        nuscenes_results_out=str(tmp_path / "results.json"),
        run_nuscenes_eval=False,
        official_max_boxes=300,
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        nuscenes_eval_config="detection_cvpr_2019",
        quiet=True,
        image_height=900,
        image_width=1600,
        backbone="fallback",
        num_queries=900,
        official_image_backbone=False,
        official_image_preprocessing=False,
        grid_mask=False,
    )

    def fake_load(path, *, map_location, weights_only):
        events.append(("load", path, map_location, weights_only))
        return checkpoint

    class FakeDataset:
        def __init__(self, **kwargs):
            events.append(("dataset", kwargs))

    class FakeModel:
        def load_state_dict(self, state_dict, *, strict):
            events.append(("state", state_dict, strict))

        def to(self, device):
            events.append(("to", str(device)))
            return self

        def eval(self):
            events.append(("eval",))
            return self

    def fake_build_model(**kwargs):
        events.append(("model", kwargs))
        return FakeModel()

    def fake_export(**kwargs):
        events.append(("export", kwargs))
        return tmp_path / "results.json"

    monkeypatch.setattr(eval_module, "parse_args", lambda: args)
    monkeypatch.setattr(eval_module.torch, "load", fake_load)
    monkeypatch.setattr(eval_module, "NuScenesDetr3DDataset", FakeDataset)
    monkeypatch.setattr(eval_module, "build_model", fake_build_model)
    monkeypatch.setattr(eval_module, "export_nuscenes_results", fake_export)

    eval_module.main()

    assert [event[0] for event in events] == [
        "load",
        "dataset",
        "model",
        "state",
        "to",
        "eval",
        "export",
    ]
    assert events[0][2:] == ("cpu", False)
    assert events[1][1]["image_size"] == (450, 800)
    assert events[1][1]["official_image_preprocessing"] is True
    assert events[2][1]["official_image_backbone"] is True
    assert events[2][1]["use_grid_mask"] is True
    assert events[3][1:] == (checkpoint["model_state_dict"], True)
    assert events[-1][1]["class_names"] == tuple(class_names)
