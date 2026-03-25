"""Training entry point aligned with a paper-oriented DETR3D setup."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detr3d.data import NuScenesDetr3DDataset, detr3d_collate
from detr3d.engine.diagnostics import evaluate_samples, parse_sample_indices, write_summary_json
from detr3d.engine.trainer import fit
from detr3d.models import Detr3DModel
from detr3d.models.backbone import MultiViewImageBackbone
from detr3d.models.heads import Detr3DHead
from detr3d.models.losses import Detr3DLoss
from detr3d.models.neck import ImageFPN
from detr3d.models.transformer import Detr3DTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pure PyTorch DETR3D baseline.")
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--num-queries", type=int, default=900)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-sample-indices", type=str, default=None)
    parser.add_argument("--num-eval-samples", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-score-threshold", type=float, default=0.005)
    parser.add_argument("--eval-max-boxes", type=int, default=50)
    parser.add_argument("--disable-auxiliary-losses", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(num_queries: int, backbone_name: str, pretrained_backbone: bool) -> Detr3DModel:
    return Detr3DModel(
        backbone=MultiViewImageBackbone(variant=backbone_name, pretrained=pretrained_backbone),
        neck=ImageFPN(),
        transformer=Detr3DTransformer(num_queries=num_queries, num_levels=4),
        head=Detr3DHead(num_decoder_layers=6),
    )


def build_optimizer(model: Detr3DModel, lr: float, backbone_lr_mult: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)
    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = build_model(
        num_queries=args.num_queries,
        backbone_name=args.backbone,
        pretrained_backbone=not args.disable_pretrained_backbone,
    ).to(device)
    criterion = Detr3DLoss(num_classes=10, use_auxiliary_losses=not args.disable_auxiliary_losses)
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
    )
    eval_sample_indices = parse_sample_indices(
        args.eval_sample_indices,
        len(dataset),
        limit=args.num_eval_samples if args.num_eval_samples > 0 else None,
    ) if (args.eval_sample_indices or args.num_eval_samples > 0) else []

    prior_history: list[dict] = []
    current_history: list[dict] = []
    start_epoch = 0
    best_eval_metric = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prior_history = checkpoint.get("history", [])
        start_epoch = int(prior_history[-1]["epoch"]) if prior_history else 0
        best_eval_metric = float(checkpoint.get("best_eval_metric", float("inf")))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 11],
        gamma=0.1,
    )

    def save_checkpoint(name: str, history_payload: list[dict]) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history_payload,
            "args": vars(args),
            "best_eval_metric": best_eval_metric,
        }
        if history_payload:
            checkpoint["best_epoch"] = min(history_payload, key=lambda row: row["loss"])
        torch.save(checkpoint, output_dir / name)

    def on_epoch_end(epoch_idx: int, metrics: dict) -> None:
        nonlocal best_eval_metric
        combined_history = prior_history + current_history + [metrics]
        if args.save_every > 0 and epoch_idx % args.save_every == 0:
            save_checkpoint(f"checkpoint_epoch_{epoch_idx:04d}.pt", combined_history)
        should_eval = eval_sample_indices and args.eval_every > 0 and epoch_idx % args.eval_every == 0
        if should_eval:
            eval_dir = output_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            overlays_dir = output_dir / "eval_artifacts" / f"epoch_{epoch_idx:04d}" / "overlays"
            bev_dir = output_dir / "eval_artifacts" / f"epoch_{epoch_idx:04d}" / "bev"
            summary = evaluate_samples(
                model=model,
                dataset=dataset,
                sample_indices=eval_sample_indices,
                device=device,
                score_threshold=args.eval_score_threshold,
                max_boxes=args.eval_max_boxes,
                overlay_dir=overlays_dir,
                bev_dir=bev_dir,
                verbose=False,
            )
            write_summary_json(eval_dir / f"epoch_{epoch_idx:04d}.json", summary)
            metric = summary.get("mean_center_distance")
            if metric is not None:
                print(f"eval epoch={epoch_idx} mean_center_distance={metric:.4f}")
                if metric < best_eval_metric:
                    best_eval_metric = float(metric)
                    save_checkpoint("best_eval_checkpoint.pt", combined_history)
        current_history.append(metrics)
        scheduler.step()

    in_run_history = fit(
        model=model,
        criterion=criterion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.use_amp,
        log_every_epoch=True,
        start_epoch=start_epoch,
        epoch_end_callback=on_epoch_end,
    )
    history = prior_history + in_run_history

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if eval_sample_indices:
        final_overlays_dir = output_dir / "eval_artifacts" / "final" / "overlays"
        final_bev_dir = output_dir / "eval_artifacts" / "final" / "bev"
        final_summary = evaluate_samples(
            model=model,
            dataset=dataset,
            sample_indices=eval_sample_indices,
            device=device,
            score_threshold=args.eval_score_threshold,
            max_boxes=args.eval_max_boxes,
            overlay_dir=final_overlays_dir,
            bev_dir=final_bev_dir,
            verbose=False,
        )
        write_summary_json(output_dir / "final_eval.json", final_summary)
        metric = final_summary.get("mean_center_distance")
        if metric is not None and metric < best_eval_metric:
            best_eval_metric = float(metric)
    save_checkpoint("last_checkpoint.pt", history)
    save_checkpoint("final_checkpoint.pt", history)
    if eval_sample_indices and (output_dir / "best_eval_checkpoint.pt").exists():
        print(f"best_eval_mean_center_distance={best_eval_metric:.4f}")


if __name__ == "__main__":
    main()
