import torch
import torch.nn as nn

from detr3d.engine.diagnostics import decode_predictions


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
