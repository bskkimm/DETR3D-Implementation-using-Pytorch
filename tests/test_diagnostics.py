import torch
import torch.nn as nn

import detr3d.engine.diagnostics as diagnostics
from detr3d.engine.diagnostics import decode_predictions, denormalize_image


class StubModel(nn.Module):
    def forward(self, images, img_metas):
        return {
            "cls_scores": torch.full((1, 1, 3, 2), -10.0),
            "bbox_preds": torch.zeros((1, 1, 3, 10)),
        }


def test_decode_predictions_returns_empty_when_none_pass_threshold():
    sample = {
        "images": torch.zeros(1, 3, 2, 2),
        "img_metas": {
            "lidar2img": torch.eye(4).unsqueeze(0),
            "image_shape": torch.tensor([[2, 2]]),
            "sample_token": "test",
        },
    }

    boxes, scores, labels = decode_predictions(
        model=StubModel(),
        sample=sample,
        device=torch.device("cpu"),
        score_threshold=0.9,
        max_boxes=2,
    )

    assert boxes.shape == (0, 9)
    assert scores.shape == (0,)
    assert labels.shape == (0,)
    assert labels.dtype == torch.long


def test_denormalize_image_restores_official_bgr_pixels_as_rgb():
    bgr_pixel = torch.tensor([30.0, 20.0, 10.0]).view(3, 1, 1)
    official_mean = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)

    image = denormalize_image(
        bgr_pixel - official_mean,
        official_preprocessing=True,
    )

    torch.testing.assert_close(
        image[0, 0],
        torch.tensor([10.0, 20.0, 30.0]) / 255.0,
    )


def test_evaluate_samples_limits_artifacts_without_limiting_metrics(
    monkeypatch, tmp_path
):
    dataset = [
        {
            "images": torch.zeros(1, 3, 2, 2),
            "img_metas": {"sample_token": f"sample-{index}"},
            "gt_boxes_ego": torch.zeros((0, 9)),
        }
        for index in range(5)
    ]
    saved_overlays = []
    saved_bevs = []
    pred_boxes = torch.zeros((2, 9))
    pred_scores = torch.tensor([0.2, 0.4])
    pred_labels = torch.zeros((2,), dtype=torch.long)

    monkeypatch.setattr(
        diagnostics,
        "decode_predictions",
        lambda **kwargs: (pred_boxes, pred_scores, pred_labels),
    )
    monkeypatch.setattr(
        diagnostics,
        "summarize_sample",
        lambda *args, **kwargs: {
            "num_gt": 0,
            "mean_center_distance": None,
            "median_center_distance": None,
            "class_matches": 0,
        },
    )
    monkeypatch.setattr(diagnostics, "get_original_camera_shapes", lambda *args: [])
    monkeypatch.setattr(
        diagnostics,
        "save_overlay_figure",
        lambda *args, **kwargs: saved_overlays.append((args[4], args[2].clone())),
    )
    monkeypatch.setattr(
        diagnostics,
        "save_bev_figure",
        lambda *args, **kwargs: saved_bevs.append((args[3], args[2].clone())),
    )

    result = diagnostics.evaluate_samples(
        model=StubModel(),
        dataset=dataset,
        sample_indices=range(5),
        device=torch.device("cpu"),
        score_threshold=0.005,
        max_boxes=100,
        overlay_dir=tmp_path / "overlays",
        bev_dir=tmp_path / "bev",
        artifact_sample_indices=[0, 3],
        artifact_score_threshold=0.3,
        verbose=False,
    )

    assert result["num_samples"] == 5
    assert len(saved_overlays) == 2
    assert len(saved_bevs) == 2
    for _, scores in saved_overlays + saved_bevs:
        torch.testing.assert_close(scores, torch.tensor([0.4]))
