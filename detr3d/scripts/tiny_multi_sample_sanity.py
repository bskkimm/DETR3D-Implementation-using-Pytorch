"""Run a tiny reproducible multi-sample DETR3D sanity experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from detr3d.data import NuScenesDetr3DDataset, detr3d_collate
from detr3d.engine.diagnostics import evaluate_samples, parse_sample_indices
from detr3d.engine.trainer import fit
from detr3d.models.losses import Detr3DLoss
from train import build_model, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny DETR3D multi-sample sanity experiment.")
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loss-cls-weight", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--scheduler", type=str, default="none", choices=["multistep", "none"])
    parser.add_argument("--scheduler-milestones", type=int, nargs="*", default=[8, 11])
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=35.0)
    parser.add_argument("--score-threshold", type=float, default=0.005)
    parser.add_argument("--max-boxes", type=int, default=50)
    parser.add_argument("--disable-auxiliary-losses", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except RuntimeError:
            pass


def build_subset_dataloader(
    dataset: NuScenesDetr3DDataset,
    sample_indices: list[int],
) -> DataLoader:
    invalid = [index for index in sample_indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise IndexError(f"Sample indices out of range for dataset of size {len(dataset)}: {invalid}")
    subset = Subset(dataset, sample_indices)
    return DataLoader(subset, batch_size=1, shuffle=False, collate_fn=detr3d_collate)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(args.image_height, args.image_width),
    )
    sample_indices = parse_sample_indices(
        args.sample_indices,
        len(dataset),
        limit=args.num_samples,
    )
    if not sample_indices:
        raise ValueError("No sample indices selected for sanity run.")

    dataloader = build_subset_dataloader(dataset=dataset, sample_indices=sample_indices)
    model = build_model(
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
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.scheduler_milestones,
            gamma=args.scheduler_gamma,
        )

    def on_epoch_end(epoch_idx: int, metrics: dict) -> None:
        if scheduler is not None:
            scheduler.step()

    history = fit(
        model=model,
        criterion=criterion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=args.use_amp,
        epoch_end_callback=on_epoch_end,
    )

    overlay_dir = None
    bev_dir = None
    if args.artifacts_dir is not None:
        artifacts_dir = Path(args.artifacts_dir)
        overlay_dir = artifacts_dir / "overlays"
        bev_dir = artifacts_dir / "bev"

    eval_summary = evaluate_samples(
        model=model,
        dataset=dataset,
        sample_indices=sample_indices,
        device=device,
        score_threshold=args.score_threshold,
        max_boxes=args.max_boxes,
        overlay_dir=overlay_dir,
        bev_dir=bev_dir,
        verbose=False,
    )

    result = {
        "config": {
            **vars(args),
            "sample_indices": sample_indices,
            "num_samples": len(sample_indices),
        },
        "final_epoch": history[-1] if history else {},
        "eval_summary": eval_summary,
    }

    print("\nFinal training metrics:")
    if history:
        for key in ["loss", "loss_cls", "loss_bbox"]:
            if key in history[-1]:
                print(f"{key}: {history[-1][key]:.4f}")

    print("\nMulti-sample sanity summary:")
    print(f"num_samples: {eval_summary['num_samples']}")
    print(f"mean_center_distance: {eval_summary['mean_center_distance']}")
    print(f"mean_median_center_distance: {eval_summary['mean_median_center_distance']}")
    print(f"total_class_matches: {eval_summary['total_class_matches']}/{eval_summary['total_gt']}")
    for row in eval_summary["sample_summaries"]:
        print(
            f"sample={row['sample_token']} | "
            f"class_matches={row['class_matches']}/{row['num_gt']} | "
            f"mean_center_distance={row['mean_center_distance']}"
        )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
