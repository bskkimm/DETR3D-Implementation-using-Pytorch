import math
import sys

import torch

from detr3d.models.losses import Detr3DLoss


def test_official_classification_defaults():
    criterion = Detr3DLoss(num_classes=2)

    assert criterion.loss_cls_weight == 2.0
    assert criterion.alpha == 0.25
    assert criterion.gamma == 2.0
    assert criterion.bg_cls_weight == 0.0
    assert criterion.matcher.cls_weight == 2.0
    assert criterion.matcher.alpha == 0.25
    assert criterion.matcher.gamma == 2.0


def test_classification_is_normalized_by_positive_count_only():
    criterion = Detr3DLoss(
        num_classes=2,
        use_auxiliary_losses=False,
        debug=True,
    )
    cls_scores = torch.zeros(1, 2, 2)
    bbox_preds = torch.zeros(1, 2, 10)
    gt_boxes = [
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    ]
    gt_labels = [torch.tensor([0])]

    losses = criterion._loss_single(cls_scores, bbox_preds, gt_boxes, gt_labels)

    assert losses["debug_num_pos"].item() == 1.0
    assert losses["debug_num_neg"].item() == 1.0
    assert losses["debug_cls_avg_factor"].item() == 1.0
    expected = torch.tensor(1.25 * math.log(2.0))
    torch.testing.assert_close(losses["loss_cls"], expected)


def test_training_cli_uses_official_classification_defaults(monkeypatch):
    from train import parse_args

    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()

    assert args.loss_cls_weight == 2.0
    assert args.focal_alpha == 0.25
    assert args.focal_gamma == 2.0
    assert args.bg_cls_weight == 0.0
