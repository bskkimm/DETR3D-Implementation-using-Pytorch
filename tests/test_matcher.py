import torch

from detr3d.models.losses.loss_utils import encode_bbox_targets
from detr3d.models.losses.matcher import HungarianMatcher3D


def test_matcher_excludes_encoded_velocity_dimensions():
    gt = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 100.0, 100.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -100.0, -100.0],
        ]
    )
    predictions = encode_bbox_targets(gt).clone()
    predictions[:, 8:10] = predictions.flip(0)[:, 8:10]

    matcher = HungarianMatcher3D(
        num_classes=1,
        cls_weight=0.0,
        bbox_weight=1.0,
    )
    assignments = matcher(
        cls_logits=torch.zeros(1, 2, 1),
        box_preds=predictions.unsqueeze(0),
        gt_boxes=[gt],
        gt_labels=[torch.zeros(2, dtype=torch.long)],
    )

    pred_ids, gt_ids = assignments[0]
    mapping = dict(zip(pred_ids.tolist(), gt_ids.tolist()))
    assert mapping == {0: 0, 1: 1}
