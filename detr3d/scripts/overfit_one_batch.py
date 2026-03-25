"""Run a one-sample DETR3D overfit experiment without the full notebook."""

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
from detr3d.data.nuscenes_dataset import NUSCENES_CLASSES
from detr3d.engine.diagnostics import decode_predictions
from detr3d.engine.trainer import fit
from detr3d.models.losses import Detr3DLoss
from train import build_model, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit DETR3D on a single nuScenes sample.")
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=832)
    parser.add_argument("--image-width", type=int, default=1472)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
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
    parser.add_argument("--max-boxes", type=int, default=100)
    parser.add_argument("--disable-auxiliary-losses", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
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


def build_single_sample_dataloader(
    dataroot: str,
    version: str,
    sample_index: int,
    image_size: tuple[int, int],
) -> tuple[NuScenesDetr3DDataset, DataLoader]:
    dataset = NuScenesDetr3DDataset(
        dataroot=dataroot,
        version=version,
        image_size=image_size,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} is out of range for dataset of size {len(dataset)}")
    single = Subset(dataset, [sample_index])
    dataloader = DataLoader(single, batch_size=1, shuffle=False, collate_fn=detr3d_collate)
    return dataset, dataloader


def summarize_sample_fit(
    model,
    dataset: NuScenesDetr3DDataset,
    sample_index: int,
    device: torch.device,
    score_threshold: float,
    max_boxes: int,
) -> dict:
    sample = dataset[sample_index]
    pred_boxes, pred_scores, pred_labels = decode_predictions(
        model=model,
        sample=sample,
        device=device,
        score_threshold=score_threshold,
        max_boxes=max_boxes,
    )
    gt_boxes = sample.get("gt_boxes_lidar", sample["gt_boxes_ego"]).cpu()
    gt_labels = sample["gt_labels"].cpu()

    summary = {
        "sample_token": sample["img_metas"]["sample_token"],
        "num_gt": int(gt_boxes.shape[0]),
        "num_pred": int(pred_boxes.shape[0]),
        "mean_center_distance": None,
        "median_center_distance": None,
        "class_matches": 0,
        "gt_classes": [NUSCENES_CLASSES[int(label)] for label in gt_labels.tolist()],
        "top_predictions": [
            {
                "label": NUSCENES_CLASSES[int(label)],
                "score": float(score),
            }
            for score, label in zip(pred_scores.tolist(), pred_labels.tolist())
        ],
        "nearest_matches": [],
    }
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return summary

    center_dist = torch.cdist(gt_boxes[:, :3], pred_boxes[:, :3], p=2)
    best_dist, best_pred_idx = center_dist.min(dim=1)
    class_matches = 0
    for gt_idx in range(gt_boxes.shape[0]):
        pred_idx = int(best_pred_idx[gt_idx])
        class_match = int(int(gt_labels[gt_idx]) == int(pred_labels[pred_idx]))
        class_matches += class_match
        same_class_mask = pred_labels == gt_labels[gt_idx]
        nearest_same_class = None
        if bool(same_class_mask.any()):
            same_class_indices = torch.nonzero(same_class_mask, as_tuple=False).squeeze(-1)
            same_class_dists = center_dist[gt_idx, same_class_indices]
            same_class_best = int(same_class_dists.argmin())
            same_class_pred_idx = int(same_class_indices[same_class_best])
            nearest_same_class = {
                "pred_index": same_class_pred_idx,
                "pred_label": NUSCENES_CLASSES[int(pred_labels[same_class_pred_idx])],
                "pred_score": float(pred_scores[same_class_pred_idx]),
                "center_distance": float(same_class_dists[same_class_best]),
            }
        summary["nearest_matches"].append(
            {
                "gt_index": int(gt_idx),
                "gt_label": NUSCENES_CLASSES[int(gt_labels[gt_idx])],
                "pred_index": pred_idx,
                "pred_label": NUSCENES_CLASSES[int(pred_labels[pred_idx])],
                "pred_score": float(pred_scores[pred_idx]),
                "center_distance": float(best_dist[gt_idx]),
                "class_match": bool(class_match),
                "nearest_same_class": nearest_same_class,
            }
        )

    summary["mean_center_distance"] = float(best_dist.mean())
    summary["median_center_distance"] = float(best_dist.median())
    summary["class_matches"] = int(class_matches)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)

    dataset, dataloader = build_single_sample_dataloader(
        dataroot=args.dataroot,
        version=args.version,
        sample_index=args.sample_index,
        image_size=(args.image_height, args.image_width),
    )
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
        debug=True,
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
        debug=True,
    )
    summary = summarize_sample_fit(
        model=model,
        dataset=dataset,
        sample_index=args.sample_index,
        device=device,
        score_threshold=args.score_threshold,
        max_boxes=args.max_boxes,
    )

    result = {
        "config": vars(args),
        "final_epoch": history[-1] if history else {},
        "summary": summary,
    }

    print("\nFinal training metrics:")
    if history:
        for key in [
            "loss",
            "loss_cls",
            "loss_bbox",
            "debug_bbox_center",
            "debug_bbox_size",
            "debug_bbox_yaw",
            "debug_bbox_velocity",
            "debug_matcher_matched_bbox_cost_mean",
        ]:
            if key in history[-1]:
                print(f"{key}: {history[-1][key]:.4f}")

    print("\nOne-sample fit summary:")
    print(f"sample_token: {summary['sample_token']}")
    print(f"num_gt: {summary['num_gt']}")
    print(f"num_pred: {summary['num_pred']}")
    print(f"mean_center_distance: {summary['mean_center_distance']}")
    print(f"median_center_distance: {summary['median_center_distance']}")
    print(f"class_matches: {summary['class_matches']}/{summary['num_gt']}")
    print("\nGT classes:")
    print(summary["gt_classes"])
    print("\nTop predicted classes:")
    print([(row["label"], round(row["score"], 4)) for row in summary["top_predictions"]])
    if summary["nearest_matches"]:
        print("\nNearest prediction for each GT:")
        for row in summary["nearest_matches"]:
            print(
                f"GT {row['gt_index']:02d}: class={row['gt_label']:<22} | "
                f"pred={row['pred_index']:02d} ({row['pred_label']:<22}) | "
                f"score={row['pred_score']:.4f} | "
                f"center_dist={row['center_distance']:.3f} m | "
                f"class_match={row['class_match']}"
            )
            if row["nearest_same_class"] is not None and not row["class_match"]:
                same = row["nearest_same_class"]
                print(
                    f"         nearest same-class: pred={same['pred_index']:02d} "
                    f"({same['pred_label']:<22}) | score={same['pred_score']:.4f} | "
                    f"center_dist={same['center_distance']:.3f} m"
                )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
