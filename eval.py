"""Evaluation and diagnostics entry point for the pure PyTorch DETR3D baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from detr3d.data import NuScenesDetr3DDataset
from detr3d.engine.diagnostics import evaluate_samples, parse_sample_indices, write_summary_json
from detr3d.models import Detr3DModel
from detr3d.models.backbone import MultiViewImageBackbone
from detr3d.models.heads import Detr3DHead
from detr3d.models.neck import ImageFPN
from detr3d.models.transformer import Detr3DTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pure PyTorch DETR3D checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--num-eval-samples", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--num-queries", type=int, default=900)
    parser.add_argument("--score-threshold", type=float, default=0.005)
    parser.add_argument("--max-boxes", type=int, default=50)
    parser.add_argument("--plot-bev", action="store_true")
    parser.add_argument("--save-overlay-dir", type=str, default=None)
    parser.add_argument("--save-bev-dir", type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(num_queries: int, backbone_name: str, pretrained_backbone: bool) -> Detr3DModel:
    return Detr3DModel(
        backbone=MultiViewImageBackbone(variant=backbone_name, pretrained=pretrained_backbone),
        neck=ImageFPN(),
        transformer=Detr3DTransformer(num_queries=num_queries, num_levels=4),
        head=Detr3DHead(num_decoder_layers=6),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(args.image_height, args.image_width),
        max_samples=args.max_samples,
    )
    model = build_model(
        num_queries=args.num_queries,
        backbone_name=args.backbone,
        pretrained_backbone=not args.disable_pretrained_backbone,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    if args.sample_index is not None:
        sample_indices = [args.sample_index]
    else:
        sample_indices = parse_sample_indices(args.sample_indices, len(dataset), limit=args.num_eval_samples)

    summary = evaluate_samples(
        model=model,
        dataset=dataset,
        sample_indices=sample_indices,
        device=device,
        score_threshold=args.score_threshold,
        max_boxes=args.max_boxes,
        overlay_dir=Path(args.save_overlay_dir) if args.save_overlay_dir else None,
        bev_dir=Path(args.save_bev_dir) if args.save_bev_dir else None,
        verbose=not args.quiet,
    )

    if args.plot_bev and args.save_bev_dir is None:
        print("`--plot-bev` is informational only here. Use `--save-bev-dir` to persist BEV figures.")

    summary_out = Path(args.summary_out) if args.summary_out else Path(args.checkpoint).with_suffix(".eval.json")
    write_summary_json(summary_out, summary)
    print(f"\nSaved summary to {summary_out}")


if __name__ == "__main__":
    main()
