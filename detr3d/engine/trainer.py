"""Trainer skeleton."""

from typing import Dict

import torch


def train_one_epoch(model, criterion, dataloader, optimizer, device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch["images"].to(device)
        outputs = model(images, img_metas=batch.get("img_metas"))
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"],
            outputs["bbox_preds"],
            batch["gt_boxes_ego"],
            batch["gt_labels"],
        )
        loss = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu())
    return {"loss": running_loss / max(len(dataloader), 1)}
