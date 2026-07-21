"""Evaluation and diagnostics entry point for the pure PyTorch DETR3D baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch

from detr3d.data import NuScenesDetr3DDataset
from detr3d.data.nuscenes_dataset import NUSCENES_CLASSES
from detr3d.engine.diagnostics import (
    evaluate_samples,
    parse_sample_indices,
    write_summary_json,
)
from detr3d.engine.evaluator import (
    OFFICIAL_MAX_NUM,
    OFFICIAL_POST_CENTER_RANGE,
    export_nuscenes_results,
    run_nuscenes_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained pure PyTorch DETR3D checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataroot", type=str, default="/home/user/datasets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--num-eval-samples", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        choices=["train", "val", "mini_train", "mini_val"],
    )
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--filter-gt-by-range", action="store_true")
    parser.add_argument("--filter-zero-point-gt", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--official-image-backbone", action="store_true")
    parser.add_argument("--official-image-preprocessing", action="store_true")
    parser.add_argument("--official-gt-semantics", action="store_true")
    parser.add_argument("--grid-mask", action="store_true")
    parser.add_argument("--num-queries", type=int, default=900)
    parser.add_argument("--score-threshold", type=float, default=0.005)
    parser.add_argument("--max-boxes", type=int, default=50)
    parser.add_argument("--plot-bev", action="store_true")
    parser.add_argument("--save-overlay-dir", type=str, default=None)
    parser.add_argument("--save-bev-dir", type=str, default=None)
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--nuscenes-results-out", type=str, default=None)
    parser.add_argument("--run-nuscenes-eval", action="store_true")
    parser.add_argument("--nuscenes-eval-output-dir", type=str, default=None)
    parser.add_argument(
        "--nuscenes-eval-set",
        type=str,
        default=None,
        choices=["train", "val", "mini_train", "mini_val", "test"],
    )
    parser.add_argument(
        "--nuscenes-eval-config", type=str, default="detection_cvpr_2019"
    )
    parser.add_argument("--official-max-boxes", type=int, default=OFFICIAL_MAX_NUM)
    parser.add_argument(
        "--post-center-range", type=float, nargs=6, default=OFFICIAL_POST_CENTER_RANGE
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_model(
    num_queries: int,
    backbone_name: str,
    pretrained_backbone: bool,
    official_image_backbone: bool = False,
    use_grid_mask: bool = False,
) -> torch.nn.Module:
    from detr3d.models import Detr3DModel
    from detr3d.models.backbone import MultiViewImageBackbone
    from detr3d.models.grid_mask import GridMask
    from detr3d.models.heads import Detr3DHead
    from detr3d.models.neck import ImageFPN
    from detr3d.models.transformer import Detr3DTransformer

    return Detr3DModel(
        backbone=MultiViewImageBackbone(
            variant=backbone_name,
            pretrained=pretrained_backbone,
            official_style=official_image_backbone,
        ),
        neck=ImageFPN(relu_before_extra_convs=official_image_backbone),
        transformer=Detr3DTransformer(num_queries=num_queries, num_levels=4),
        head=Detr3DHead(num_decoder_layers=6),
        image_augmentation=GridMask() if use_grid_mask else None,
    )


def resolve_checkpoint_config(
    args: argparse.Namespace, checkpoint: Mapping[str, object]
) -> dict[str, object]:
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, Mapping):
        raise ValueError("checkpoint 'args' must be a mapping")

    def saved_or_cli(name: str):
        value = saved_args.get(name)
        return getattr(args, name, False) if value is None else value

    config = {
        "image_height": int(saved_or_cli("image_height")),
        "image_width": int(saved_or_cli("image_width")),
        "backbone": str(saved_or_cli("backbone")),
        "disable_pretrained_backbone": bool(saved_or_cli("disable_pretrained_backbone")),
        "num_queries": int(saved_or_cli("num_queries")),
        "official_image_backbone": bool(saved_or_cli("official_image_backbone")),
        "official_image_preprocessing": bool(
            saved_or_cli("official_image_preprocessing")
        ),
        "official_gt_semantics": bool(saved_or_cli("official_gt_semantics")),
        "grid_mask": bool(saved_or_cli("grid_mask")),
    }
    if config["image_height"] <= 0 or config["image_width"] <= 0:
        raise ValueError("checkpoint image dimensions must be positive")
    if config["num_queries"] <= 0:
        raise ValueError("checkpoint num_queries must be positive")
    if not config["backbone"]:
        raise ValueError("checkpoint backbone must be non-empty")
    return config


def resolve_checkpoint_class_names(
    checkpoint: Mapping[str, object],
) -> tuple[str, ...]:
    class_names = checkpoint.get("class_names")
    if class_names is None:
        return tuple(NUSCENES_CLASSES)
    if isinstance(class_names, (str, bytes)) or not isinstance(
        class_names, (list, tuple)
    ):
        raise ValueError("checkpoint class_names must be a sequence of strings")
    names = tuple(class_names)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("checkpoint class_names must be a sequence of strings")
    if len(names) != len(NUSCENES_CLASSES) or set(names) != set(NUSCENES_CLASSES):
        raise ValueError(
            "checkpoint class_names must contain every repository nuScenes class exactly once"
        )
    return names


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    config = resolve_checkpoint_config(args, checkpoint)
    class_names = resolve_checkpoint_class_names(checkpoint)
    device = torch.device(args.device)

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(config["image_height"], config["image_width"]),
        max_samples=args.max_samples,
        split=args.dataset_split,
        filter_gt_by_range=args.filter_gt_by_range,
        filter_zero_point_gt=args.filter_zero_point_gt,
        official_image_preprocessing=config["official_image_preprocessing"],
        official_gt_semantics=config["official_gt_semantics"],
    )
    model = build_model(
        num_queries=config["num_queries"],
        backbone_name=config["backbone"],
        pretrained_backbone=not config["disable_pretrained_backbone"],
        official_image_backbone=config["official_image_backbone"],
        use_grid_mask=config["grid_mask"],
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    if args.nuscenes_results_out is not None:
        if args.dataset_split is None:
            raise ValueError("--dataset-split is required for nuScenes result export")
        if args.run_nuscenes_eval:
            if args.nuscenes_eval_output_dir is None or args.nuscenes_eval_set is None:
                raise ValueError(
                    "nuScenes evaluation requires --nuscenes-eval-output-dir "
                    "and --nuscenes-eval-set"
                )
            if args.dataset_split != args.nuscenes_eval_set:
                raise ValueError("--dataset-split must match --nuscenes-eval-set")
            if (
                args.max_samples is not None
                or args.sample_index is not None
                or args.sample_indices is not None
            ):
                raise ValueError(
                    "Official nuScenes evaluation requires the complete split"
                )
        result_path = export_nuscenes_results(
            model=model,
            dataset=dataset,
            device=device,
            output_path=args.nuscenes_results_out,
            max_num=args.official_max_boxes,
            post_center_range=args.post_center_range,
            eval_config_name=args.nuscenes_eval_config,
            class_names=class_names,
            verbose=not args.quiet,
        )
        print(f"Saved nuScenes results to {result_path}")
        if args.run_nuscenes_eval:
            run_nuscenes_evaluation(
                dataroot=args.dataroot,
                version=args.version,
                result_path=result_path,
                eval_set=args.nuscenes_eval_set,
                output_dir=args.nuscenes_eval_output_dir,
                eval_config_name=args.nuscenes_eval_config,
                verbose=not args.quiet,
            )
        return
    if args.run_nuscenes_eval:
        raise ValueError("--run-nuscenes-eval requires --nuscenes-results-out")

    if args.sample_index is not None:
        sample_indices = [args.sample_index]
    else:
        sample_indices = parse_sample_indices(
            args.sample_indices, len(dataset), limit=args.num_eval_samples
        )

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
        print(
            "`--plot-bev` is informational only here. Use `--save-bev-dir` "
            "to persist BEV figures."
        )

    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else Path(args.checkpoint).with_suffix(".eval.json")
    )
    write_summary_json(summary_out, summary)
    print(f"\nSaved summary to {summary_out}")


if __name__ == "__main__":
    main()
