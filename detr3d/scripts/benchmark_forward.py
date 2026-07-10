"""Benchmark DETR3D dataloader and GPU training-step throughput."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import torch
from torch.utils.data import DataLoader

from detr3d.data import NuScenesDetr3DDataset, detr3d_collate
from detr3d.engine.trainer import move_batch_to_device
from detr3d.models import Detr3DModel
from detr3d.models.backbone import MultiViewImageBackbone
from detr3d.models.heads import Detr3DHead
from detr3d.models.losses import Detr3DLoss
from detr3d.models.neck import ImageFPN
from detr3d.models.transformer import Detr3DTransformer


def build_benchmark_model(num_queries: int, backbone_name: str, pretrained_backbone: bool) -> Detr3DModel:
    return Detr3DModel(
        backbone=MultiViewImageBackbone(variant=backbone_name, pretrained=pretrained_backbone),
        neck=ImageFPN(),
        transformer=Detr3DTransformer(num_queries=num_queries, num_levels=4),
        head=Detr3DHead(num_decoder_layers=6),
    )


def build_benchmark_optimizer(
    model: Detr3DModel,
    lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DETR3D input pipeline and GPU step time.")
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--image-height", type=int, default=832)
    parser.add_argument("--image-width", type=int, default=1472)
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss-cls-weight", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--grad-clip-norm", type=float, default=35.0)
    parser.add_argument("--disable-auxiliary-losses", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * q)), len(ordered) - 1)
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(args.image_height, args.image_width),
        max_samples=args.max_samples,
    )
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": detr3d_collate,
        "pin_memory": args.pin_memory or device.type == "cuda",
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
    dataloader = DataLoader(**dataloader_kwargs)

    model = None
    criterion = None
    optimizer = None
    scaler = None
    amp_enabled = args.use_amp and device.type == "cuda"
    if not args.data_only:
        model = build_benchmark_model(
            num_queries=args.num_queries,
            backbone_name=args.backbone,
            pretrained_backbone=not args.disable_pretrained_backbone,
        ).to(device)
        criterion = Detr3DLoss(
            num_classes=10,
            loss_cls_weight=args.loss_cls_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            use_auxiliary_losses=not args.disable_auxiliary_losses,
        )
        optimizer = build_benchmark_optimizer(
            model=model,
            lr=args.lr,
            backbone_lr_mult=args.backbone_lr_mult,
            weight_decay=args.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        model.train()

    iterator = iter(dataloader)
    data_times: list[float] = []
    step_times: list[float] = []
    losses: list[float] = []
    status = "ok"
    error = None
    total_steps = args.warmup_steps + args.steps

    try:
        for step_idx in range(total_steps):
            data_start = time.perf_counter()
            batch, iterator = next_batch(iterator, dataloader)
            data_time = time.perf_counter() - data_start

            sync_cuda(device)
            step_start = time.perf_counter()
            if not args.data_only:
                assert model is not None and criterion is not None and optimizer is not None and scaler is not None
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(batch["images"], img_metas=batch["img_metas"])
                loss_dict = criterion.loss_by_feat(
                    outputs["cls_scores"],
                    outputs["bbox_preds"],
                    batch["gt_boxes_ego"],
                    batch["gt_labels"],
                )
                loss = sum(value for name, value in loss_dict.items() if "loss" in name)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
            sync_cuda(device)
            step_time = time.perf_counter() - step_start

            if step_idx >= args.warmup_steps:
                data_times.append(data_time)
                step_times.append(step_time)
                print(
                    f"step={step_idx - args.warmup_steps + 1}/{args.steps} "
                    f"data_time={data_time:.4f}s step_time={step_time:.4f}s"
                )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status = "oom"
            error = str(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            raise

    peak_allocated_gb = None
    peak_reserved_gb = None
    if device.type == "cuda":
        peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    result = {
        "status": status,
        "error": error,
        "config": vars(args),
        "host_cpu_count": os.cpu_count(),
        "num_dataset_samples": len(dataset),
        "mean_data_time_sec": mean(data_times),
        "p90_data_time_sec": percentile(data_times, 0.9),
        "mean_step_time_sec": mean(step_times),
        "p90_step_time_sec": percentile(step_times, 0.9),
        "samples_per_sec": float(args.batch_size / mean(step_times)) if step_times else 0.0,
        "mean_loss": mean(losses[-args.steps:]) if losses and not args.data_only else None,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }

    print("\nBenchmark summary:")
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {output_path}")

    if status == "oom":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
