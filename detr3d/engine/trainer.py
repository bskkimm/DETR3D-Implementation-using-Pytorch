"""Trainer skeleton."""

from typing import Dict

def train_one_epoch(model, criterion, dataloader, optimizer, device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch["images"].to(device)
        img_metas = [
            {
                "lidar2img": meta["lidar2img"].to(device),
                "image_shape": meta["image_shape"].to(device),
                "sample_token": meta["sample_token"],
            }
            for meta in batch["img_metas"]
        ]
        gt_boxes = [boxes.to(device) for boxes in batch["gt_boxes_ego"]]
        gt_labels = [labels.to(device) for labels in batch["gt_labels"]]
        outputs = model(images, img_metas=img_metas)
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"],
            outputs["bbox_preds"],
            gt_boxes,
            gt_labels,
        )
        loss = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu())
    return {"loss": running_loss / max(len(dataloader), 1)}
