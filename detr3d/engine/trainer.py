"""Training utilities for the DETR3D scaffold."""

from __future__ import annotations

import time
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
        "gt_boxes_ego": [
            boxes.to(device, non_blocking=True) for boxes in batch["gt_boxes_ego"]
        ],
        "gt_labels": [
            labels.to(device, non_blocking=True) for labels in batch["gt_labels"]
        ],
    }


def _collect_named_params(model) -> Dict[str, torch.nn.Parameter]:
    params = dict(model.named_parameters())
    return {
        alias: params[name]
        for alias, name in _DEBUG_PARAM_NAMES.items()
        if name in params
    }


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
    safety_check: Callable[[], None] | None = None,
    step_scheduler=None,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")

    model.train()
    running = defaultdict(float)
    amp_enabled = use_amp and device.type == "cuda"
    micro_batches = len(dataloader)
    optimizer_steps = 0
    update_boundaries = 0
    epoch_start = time.monotonic()
    previous_step_end = epoch_start
    thermal_pause_start = getattr(safety_check, "total_pause_sec", 0.0)
    thermal_count_start = getattr(safety_check, "pause_count", 0)

    optimizer.zero_grad(set_to_none=True)
    debug_params = _collect_named_params(model) if debug else {}
    param_before: Dict[str, torch.Tensor] = {}

    for batch_index, batch in enumerate(dataloader):
        batch_ready = time.monotonic()
        running["time_data_sec"] += batch_ready - previous_step_end
        if safety_check is not None:
            safety_check()
        step_start = time.monotonic()
        batch = move_batch_to_device(batch, device)
        if debug and batch_index % accumulation_steps == 0:
            param_before = {
                name: param.detach().clone() for name, param in debug_params.items()
            }
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

        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, micro_batches - window_start)
        backward_loss = loss / window_size
        update_boundary = (
            batch_index + 1
        ) % accumulation_steps == 0 or batch_index + 1 == micro_batches

        if amp_enabled and scaler is not None:
            scaler.scale(backward_loss).backward()
            if update_boundary:
                update_boundaries += 1
                if debug or grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                if debug:
                    for name, param in debug_params.items():
                        running[f"debug_grad_{name}"] += _grad_norm(param)
                if grad_clip_norm is not None:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                    if debug:
                        running["debug_grad_total_norm"] += float(
                            total_grad_norm.item()
                        )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_succeeded = scaler.get_scale() >= scale_before
                optimizer_steps += int(step_succeeded)
                if step_scheduler is not None and step_succeeded:
                    step_scheduler.step()
        else:
            backward_loss.backward()
            if update_boundary:
                update_boundaries += 1
                if debug:
                    for name, param in debug_params.items():
                        running[f"debug_grad_{name}"] += _grad_norm(param)
                if grad_clip_norm is not None:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                    if debug:
                        running["debug_grad_total_norm"] += float(
                            total_grad_norm.item()
                        )
                optimizer.step()
                optimizer_steps += 1
                if step_scheduler is not None:
                    step_scheduler.step()

        if debug:
            if update_boundary:
                for name, param in debug_params.items():
                    running[f"debug_delta_{name}"] += _param_delta_norm(
                        param, param_before[name]
                    )
            for name, value in _prediction_stats(outputs).items():
                running[name] += value

        if update_boundary:
            optimizer.zero_grad(set_to_none=True)

        running["loss"] += float(loss.detach().cpu())
        for name, value in loss_dict.items():
            if torch.is_tensor(value):
                running[name] += float(value.detach().cpu())
            else:
                running[name] += float(value)
        running["time_train_sec"] += time.monotonic() - step_start
        previous_step_end = time.monotonic()

    num_batches = max(micro_batches, 1)
    metrics = {
        name: (
            total
            if name.startswith("time_")
            else (
                total / max(update_boundaries, 1)
                if name.startswith(("debug_grad_", "debug_delta_"))
                else total / num_batches
            )
        )
        for name, total in running.items()
    }
    metrics["time_epoch_wall_sec"] = time.monotonic() - epoch_start
    metrics["time_thermal_pause_sec"] = (
        getattr(safety_check, "total_pause_sec", 0.0) - thermal_pause_start
    )
    metrics["thermal_pause_count"] = float(
        getattr(safety_check, "pause_count", 0) - thermal_count_start
    )
    metrics["max_gpu_temp"] = float(getattr(safety_check, "max_observed_gpu_temp", 0.0))
    metrics["max_cpu_temp"] = float(getattr(safety_check, "max_observed_cpu_temp", 0.0))
    metrics["max_gpu_power_watts"] = float(
        getattr(safety_check, "max_observed_power_watts", 0.0)
    )
    metrics["micro_batches"] = float(micro_batches)
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["accumulation_steps"] = float(accumulation_steps)
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    if len(optimizer.param_groups) > 1:
        metrics["backbone_lr"] = float(optimizer.param_groups[1]["lr"])
    return metrics


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
    safety_check: Callable[[], None] | None = None,
    step_scheduler=None,
    accumulation_steps: int = 1,
) -> List[Dict[str, float]]:
    if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
        raise ValueError("accumulation_steps must be a positive integer")

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
            safety_check=safety_check,
            step_scheduler=step_scheduler,
            accumulation_steps=accumulation_steps,
        )
        metrics["epoch"] = float(epoch + 1)
        history.append(metrics)

        if log_every_epoch:
            summary = ", ".join(
                f"{name}={value:.4f}"
                for name, value in metrics.items()
                if name != "epoch"
            )
            print(f"epoch={epoch + 1}/{start_epoch + epochs} {summary}")

        if epoch_end_callback is not None:
            epoch_end_callback(epoch + 1, metrics)

    return history
