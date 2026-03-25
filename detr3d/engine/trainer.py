"""Training utilities for the DETR3D scaffold."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List

import torch

from detr3d.models.losses.loss_utils import decode_bbox_predictions

_DEBUG_PARAM_NAMES = {
    "query_embed": "transformer.query_embed.weight",
    "query_pos": "transformer.query_pos.weight",
    "ref_branch_last": "head.reference_points.weight",
    "cls_branch_last": "head.cls_branches.5.net.4.weight",
    "reg_branch_last": "head.reg_branches.5.net.4.weight",
}

_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "img_metas": [
            {
                "lidar2img": meta["lidar2img"].to(device, non_blocking=True),
                "image_shape": meta["image_shape"].to(device, non_blocking=True),
                "sample_token": meta["sample_token"],
            }
            for meta in batch["img_metas"]
        ],
        "gt_boxes_ego": [boxes.to(device, non_blocking=True) for boxes in batch["gt_boxes_ego"]],
        "gt_labels": [labels.to(device, non_blocking=True) for labels in batch["gt_labels"]],
    }


def _collect_named_params(model) -> Dict[str, torch.nn.Parameter]:
    params = dict(model.named_parameters())
    return {alias: params[name] for alias, name in _DEBUG_PARAM_NAMES.items() if name in params}


def _grad_norm(param: torch.nn.Parameter) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def _param_delta_norm(param: torch.nn.Parameter, before: torch.Tensor) -> float:
    return float((param.detach() - before).norm().item())


def _prediction_stats(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    cls_scores = outputs["cls_scores"][-1].detach().float()
    bbox_preds = outputs["bbox_preds"][-1].detach().float()
    decoded = decode_bbox_predictions(bbox_preds, _PC_RANGE)
    probs = cls_scores.sigmoid()
    return {
        "debug_pred_score_mean": float(probs.mean().item()),
        "debug_pred_score_max": float(probs.max().item()),
        "debug_pred_center_mean_abs": float(decoded[..., :3].abs().mean().item()),
        "debug_pred_size_mean": float(decoded[..., 3:6].mean().item()),
        "debug_pred_yaw_abs_mean": float(decoded[..., 6].abs().mean().item()),
        "debug_pred_velocity_abs_mean": float(decoded[..., 7:9].abs().mean().item()),
    }


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    debug: bool = False,
) -> Dict[str, float]:
    model.train()
    running = defaultdict(float)
    amp_enabled = use_amp and device.type == "cuda"

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        debug_params = _collect_named_params(model) if debug else {}
        param_before = {name: param.detach().clone() for name, param in debug_params.items()}
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["images"], img_metas=batch["img_metas"])
        loss_dict = criterion.loss_by_feat(
            outputs["cls_scores"],
            outputs["bbox_preds"],
            batch["gt_boxes_ego"],
            batch["gt_labels"],
        )
        loss_terms = [value for name, value in loss_dict.items() if "loss" in name]
        loss = sum(loss_terms)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss during training.")

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if debug or grad_clip_norm is not None:
                scaler.unscale_(optimizer)
            if debug:
                for name, param in debug_params.items():
                    running[f"debug_grad_{name}"] += _grad_norm(param)
            if grad_clip_norm is not None:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                if debug:
                    running["debug_grad_total_norm"] += float(total_grad_norm.item())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if debug:
                for name, param in debug_params.items():
                    running[f"debug_grad_{name}"] += _grad_norm(param)
            if grad_clip_norm is not None:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                if debug:
                    running["debug_grad_total_norm"] += float(total_grad_norm.item())
            optimizer.step()

        if debug:
            for name, param in debug_params.items():
                running[f"debug_delta_{name}"] += _param_delta_norm(param, param_before[name])
            for name, value in _prediction_stats(outputs).items():
                running[name] += value

        running["loss"] += float(loss.detach().cpu())
        for name, value in loss_dict.items():
            if torch.is_tensor(value):
                running[name] += float(value.detach().cpu())
            else:
                running[name] += float(value)

    num_batches = max(len(dataloader), 1)
    return {name: total / num_batches for name, total in running.items()}


def fit(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    epochs: int,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    log_every_epoch: bool = True,
    start_epoch: int = 0,
    epoch_end_callback: Callable[[int, Dict[str, float]], None] | None = None,
    debug: bool = False,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for epoch in range(start_epoch, start_epoch + epochs):
        metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
            debug=debug,
        )
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)

        if log_every_epoch:
            summary = ", ".join(
                f"{name}={value:.4f}" for name, value in metrics.items() if name != "epoch"
            )
            print(f"epoch={epoch + 1}/{start_epoch + epochs} {summary}")

        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, metrics)

    return history
