"""Training entry point aligned with a paper-oriented DETR3D setup."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detr3d.data import CBGSDataset, NuScenesDetr3DDataset, detr3d_collate
from detr3d.data.nuscenes_dataset import NUSCENES_CLASSES
from detr3d.engine.diagnostics import (
    evaluate_samples,
    parse_sample_indices,
    write_summary_json,
)
from detr3d.engine.trainer import fit
from detr3d.models import Detr3DModel
from detr3d.models.backbone import MultiViewImageBackbone
from detr3d.models.checkpoint import load_fcos3d_initialization
from detr3d.models.grid_mask import GridMask
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
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-cls-weight", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--bg-cls-weight", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine", "none"])
    parser.add_argument("--scheduler-milestones", type=int, nargs="*", default=[8, 11])
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--warmup-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    parser.add_argument("--scheduler-total-epochs", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--filter-gt-by-range", action="store_true")
    parser.add_argument("--filter-zero-point-gt", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet101")
    parser.add_argument("--disable-pretrained-backbone", action="store_true")
    parser.add_argument("--official-image-backbone", action="store_true")
    parser.add_argument("--official-image-preprocessing", action="store_true")
    parser.add_argument("--init-fcos3d-checkpoint", type=str, default=None)
    parser.add_argument("--grid-mask", action="store_true")
    parser.add_argument("--photometric-distortion", action="store_true")
    parser.add_argument("--cbgs", action="store_true")
    parser.add_argument("--official-gt-semantics", action="store_true")
    parser.add_argument("--num-queries", type=int, default=900)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-split", type=str, default=None, choices=["train", "val", "mini_train", "mini_val"])
    parser.add_argument("--val-split", type=str, default=None, choices=["val", "mini_val"])
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-sample-indices", type=str, default=None)
    parser.add_argument("--num-eval-samples", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-score-threshold", type=float, default=0.005)
    parser.add_argument("--eval-max-boxes", type=int, default=50)
    parser.add_argument("--disable-eval-artifacts", action="store_true")
    parser.add_argument("--num-eval-artifact-samples", type=int, default=None)
    parser.add_argument("--disable-auxiliary-losses", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", type=str, default="detr3d-small-training")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-log-checkpoints", action="store_true")
    parser.add_argument("--thermal-check-interval", type=float, default=60.0)
    parser.add_argument("--max-gpu-temp", type=float, default=85.0)
    parser.add_argument("--max-cpu-temp", type=float, default=90.0)
    parser.add_argument("--thermal-action", choices=["stop", "pause"], default="stop")
    parser.add_argument("--resume-gpu-temp", type=float, default=75.0)
    parser.add_argument("--resume-cpu-temp", type=float, default=80.0)
    parser.add_argument("--thermal-poll-interval", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(dataloader_generator: torch.Generator) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataloader_generator": dataloader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, dataloader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    dataloader_generator.set_state(state["dataloader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _start_mlflow_run(args: argparse.Namespace, dataset_size: int, output_dir: Path):
    if not args.mlflow:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but mlflow is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or output_dir.name
    mlflow.start_run(run_name=run_name)
    params = {key: str(value) for key, value in vars(args).items()}
    params["dataset_size"] = str(dataset_size)
    params["output_dir"] = str(output_dir)
    mlflow.log_params(params)
    tags = {
        "git_branch": _git_output("branch", "--show-current") or "unknown",
        "git_commit": _git_output("rev-parse", "HEAD") or "unknown",
    }
    mlflow.set_tags(tags)
    return mlflow


def _log_mlflow_metrics(mlflow_module, metrics: dict, *, step: int, prefix: str = "") -> None:
    if mlflow_module is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            mlflow_module.log_metric(f"{prefix}{name}", float(value), step=step)


def _log_mlflow_artifact(mlflow_module, path: Path) -> None:
    if mlflow_module is not None and path.exists():
        mlflow_module.log_artifact(str(path))


def _log_mlflow_artifacts(mlflow_module, path: Path, artifact_path: str) -> None:
    if mlflow_module is not None and path.exists():
        mlflow_module.log_artifacts(str(path), artifact_path=artifact_path)


class ThermalSafetyMonitor:
    def __init__(
        self,
        interval: float,
        max_gpu_temp: float,
        max_cpu_temp: float,
        action: str,
        resume_gpu_temp: float,
        resume_cpu_temp: float,
        poll_interval: float,
    ):
        self.interval = interval
        self.max_gpu_temp = max_gpu_temp
        self.max_cpu_temp = max_cpu_temp
        self.action = action
        self.resume_gpu_temp = resume_gpu_temp
        self.resume_cpu_temp = resume_cpu_temp
        self.poll_interval = poll_interval
        self.last_check = 0.0
        self.total_pause_sec = 0.0
        self.pause_count = 0
        self.max_observed_gpu_temp = 0.0
        self.max_observed_cpu_temp = 0.0
        self.max_observed_power_watts = 0.0

    @staticmethod
    def _read_system() -> tuple[float, float, float]:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_rows = [line.split(",") for line in gpu_result.stdout.splitlines() if line.strip()]
        gpu_temp = max(float(row[0].strip()) for row in gpu_rows)
        gpu_power = max(float(row[1].strip()) for row in gpu_rows)
        sensors_result = subprocess.run(
            ["sensors"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        cpu_temps = []
        for line in sensors_result.stdout.splitlines():
            if line.startswith("Package id") and "+" in line:
                cpu_temps.append(float(line.split("+")[1].split("°C")[0]))
        cpu_temp = max(cpu_temps) if cpu_temps else 0.0
        return gpu_temp, cpu_temp, gpu_power

    def __call__(self) -> None:
        now = time.monotonic()
        if now - self.last_check < self.interval:
            return
        self.last_check = now
        gpu_temp, cpu_temp, gpu_power = self._read_system()
        self.max_observed_gpu_temp = max(self.max_observed_gpu_temp, gpu_temp)
        self.max_observed_cpu_temp = max(self.max_observed_cpu_temp, cpu_temp)
        self.max_observed_power_watts = max(self.max_observed_power_watts, gpu_power)
        print(f"thermal gpu={gpu_temp:.0f}C cpu={cpu_temp:.0f}C power={gpu_power:.0f}W")
        if gpu_temp < self.max_gpu_temp and cpu_temp < self.max_cpu_temp:
            return
        if self.action == "stop":
            raise RuntimeError(
                f"Thermal safety stop: GPU {gpu_temp:.0f}C/{self.max_gpu_temp:.0f}C, "
                f"CPU {cpu_temp:.0f}C/{self.max_cpu_temp:.0f}C"
            )
        print(
            f"thermal pause: waiting for GPU <= {self.resume_gpu_temp:.0f}C "
            f"and CPU <= {self.resume_cpu_temp:.0f}C"
        )
        pause_start = time.monotonic()
        self.pause_count += 1
        while gpu_temp > self.resume_gpu_temp or cpu_temp > self.resume_cpu_temp:
            time.sleep(self.poll_interval)
            gpu_temp, cpu_temp, gpu_power = self._read_system()
            self.max_observed_gpu_temp = max(self.max_observed_gpu_temp, gpu_temp)
            self.max_observed_cpu_temp = max(self.max_observed_cpu_temp, cpu_temp)
            self.max_observed_power_watts = max(self.max_observed_power_watts, gpu_power)
            print(f"thermal cooldown gpu={gpu_temp:.0f}C cpu={cpu_temp:.0f}C power={gpu_power:.0f}W")
        self.total_pause_sec += time.monotonic() - pause_start
        self.last_check = time.monotonic()
        print("thermal resume")


def build_model(
    num_queries: int,
    backbone_name: str,
    pretrained_backbone: bool,
    official_image_backbone: bool = False,
    use_grid_mask: bool = False,
) -> Detr3DModel:
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
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be a positive integer")
    if args.cbgs and not args.official_gt_semantics:
        raise ValueError("--cbgs requires --official-gt-semantics")
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and args.init_fcos3d_checkpoint is not None:
        raise ValueError("--resume and --init-fcos3d-checkpoint cannot be used together")
    if args.init_fcos3d_checkpoint is not None and not args.official_image_backbone:
        raise ValueError("--init-fcos3d-checkpoint requires --official-image-backbone")

    dataset = NuScenesDetr3DDataset(
        dataroot=args.dataroot,
        version=args.version,
        image_size=(args.image_height, args.image_width),
        max_samples=args.max_samples,
        split=args.dataset_split,
        filter_gt_by_range=args.filter_gt_by_range,
        filter_zero_point_gt=args.filter_zero_point_gt,
        official_image_preprocessing=args.official_image_preprocessing,
        photometric_distortion=args.photometric_distortion,
        official_gt_semantics=args.official_gt_semantics,
    )
    eval_dataset = dataset
    if args.val_split is not None:
        eval_dataset = NuScenesDetr3DDataset(
            dataroot=args.dataroot,
            version=args.version,
            image_size=(args.image_height, args.image_width),
            max_samples=args.max_val_samples,
            split=args.val_split,
            filter_gt_by_range=args.filter_gt_by_range,
            filter_zero_point_gt=args.filter_zero_point_gt,
            tables=dataset.tables,
            official_image_preprocessing=args.official_image_preprocessing,
            official_gt_semantics=args.official_gt_semantics,
        )
    elif args.photometric_distortion and (args.eval_sample_indices or args.num_eval_samples > 0):
        eval_dataset = NuScenesDetr3DDataset(
            dataroot=args.dataroot,
            version=args.version,
            image_size=(args.image_height, args.image_width),
            max_samples=args.max_samples,
            split=args.dataset_split,
            filter_gt_by_range=args.filter_gt_by_range,
            filter_zero_point_gt=args.filter_zero_point_gt,
            tables=dataset.tables,
            official_image_preprocessing=args.official_image_preprocessing,
            photometric_distortion=False,
            official_gt_semantics=args.official_gt_semantics,
        )
    base_train_size = len(dataset)
    if args.cbgs:
        dataset = CBGSDataset(dataset, seed=args.seed)
        cbgs_report_path = output_dir / "cbgs_report.json"
        cbgs_report_path.write_text(
            json.dumps(
                {
                    "fingerprint": dataset.fingerprint,
                    "stats": dataset.stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    mlflow_run = _start_mlflow_run(args, len(dataset), output_dir)
    if mlflow_run is not None:
        mlflow_run.log_param("base_train_dataset_size", base_train_size)
        mlflow_run.log_param("eval_dataset_size", len(eval_dataset))
        if args.cbgs:
            _log_mlflow_artifact(mlflow_run, cbgs_report_path)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": detr3d_collate,
        "pin_memory": args.pin_memory or device.type == "cuda",
        "generator": dataloader_generator,
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader_kwargs["persistent_workers"] = args.persistent_workers
    dataloader = DataLoader(**dataloader_kwargs)

    model = build_model(
        num_queries=args.num_queries,
        backbone_name=args.backbone,
        pretrained_backbone=not args.disable_pretrained_backbone,
        official_image_backbone=args.official_image_backbone,
        use_grid_mask=args.grid_mask,
    ).to(device)
    if args.init_fcos3d_checkpoint is not None:
        initialization_report = load_fcos3d_initialization(model, args.init_fcos3d_checkpoint)
        report_path = output_dir / "initialization_report.json"
        report_path.write_text(json.dumps(initialization_report, indent=2), encoding="utf-8")
        print(
            f"FCOS3D initialization loaded {initialization_report['loaded_tensors']} tensors "
            f"with {initialization_report['coverage']:.2%} backbone/FPN coverage"
        )
    criterion = Detr3DLoss(
        num_classes=10,
        loss_cls_weight=args.loss_cls_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        bg_cls_weight=args.bg_cls_weight,
        use_auxiliary_losses=not args.disable_auxiliary_losses,
    )
    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
    )
    eval_sample_indices = parse_sample_indices(
        args.eval_sample_indices,
        len(eval_dataset),
        limit=args.num_eval_samples if args.num_eval_samples > 0 else None,
    ) if (args.eval_sample_indices or args.num_eval_samples > 0) else []
    if args.num_eval_artifact_samples is not None and args.num_eval_artifact_samples < 0:
        raise ValueError("--num-eval-artifact-samples must be non-negative")
    eval_artifact_sample_indices = (
        eval_sample_indices
        if args.num_eval_artifact_samples is None
        else eval_sample_indices[: args.num_eval_artifact_samples]
    )
    thermal_monitor = ThermalSafetyMonitor(
        interval=args.thermal_check_interval,
        max_gpu_temp=args.max_gpu_temp,
        max_cpu_temp=args.max_cpu_temp,
        action=args.thermal_action,
        resume_gpu_temp=args.resume_gpu_temp,
        resume_cpu_temp=args.resume_cpu_temp,
        poll_interval=args.thermal_poll_interval,
    )

    prior_history: list[dict] = []
    current_history: list[dict] = []
    start_epoch = 0
    best_eval_metric = float("inf")
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        saved_args = resume_checkpoint.get("args", {})
        for name in ("accumulation_steps", "batch_size", "cbgs"):
            saved_value = saved_args.get(name, 1 if name == "accumulation_steps" else False)
            if saved_value != getattr(args, name):
                raise ValueError(
                    f"Cannot resume with changed {name}: checkpoint={saved_value}, "
                    f"requested={getattr(args, name)}"
                )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        prior_history = resume_checkpoint.get("history", [])
        start_epoch = int(prior_history[-1]["epoch"]) if prior_history else 0
        best_eval_metric = float(resume_checkpoint.get("best_eval_metric", float("inf")))
        if "rng_state" in resume_checkpoint:
            restore_rng_state(resume_checkpoint["rng_state"], dataloader_generator)

    scheduler = None
    step_scheduler = None
    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.scheduler_milestones,
            gamma=args.scheduler_gamma,
        )
    elif args.scheduler == "cosine":
        schedule_epochs = args.scheduler_total_epochs or (start_epoch + args.epochs)
        optimizer_steps_per_epoch = math.ceil(len(dataloader) / args.accumulation_steps)
        total_steps = schedule_epochs * optimizer_steps_per_epoch

        def lr_factor(step: int) -> float:
            if args.warmup_steps > 0 and step < args.warmup_steps:
                progress = step / max(args.warmup_steps, 1)
                return args.warmup_ratio + progress * (1.0 - args.warmup_ratio)
            cosine_steps = max(total_steps - args.warmup_steps, 1)
            progress = min(max(step - args.warmup_steps, 0) / cosine_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

        step_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    active_scheduler = step_scheduler or scheduler
    if resume_checkpoint is not None and active_scheduler is not None and "scheduler_state_dict" in resume_checkpoint:
        active_scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        for group, lr in zip(optimizer.param_groups, active_scheduler.get_last_lr()):
            group["lr"] = lr

    def save_checkpoint(name: str, history_payload: list[dict]) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history_payload,
            "args": vars(args),
            "class_names": list(NUSCENES_CLASSES),
            "training_state": {
                "completed_epoch": int(history_payload[-1]["epoch"]) if history_payload else 0,
                "accumulation_steps": args.accumulation_steps,
                "resume_safe": True,
            },
            "best_eval_metric": best_eval_metric,
            "rng_state": capture_rng_state(dataloader_generator),
        }
        if active_scheduler is not None:
            checkpoint["scheduler_state_dict"] = active_scheduler.state_dict()
        if history_payload:
            checkpoint["best_epoch"] = min(history_payload, key=lambda row: row["loss"])
        checkpoint_path = output_dir / name
        torch.save(checkpoint, checkpoint_path)
        if args.mlflow_log_checkpoints:
            _log_mlflow_artifact(mlflow_run, checkpoint_path)

    def on_epoch_end(epoch_idx: int, metrics: dict) -> None:
        nonlocal best_eval_metric
        combined_history = prior_history + current_history + [metrics]
        if args.save_every > 0 and epoch_idx % args.save_every == 0:
            save_checkpoint(f"checkpoint_epoch_{epoch_idx:04d}.pt", combined_history)
        should_eval = eval_sample_indices and args.eval_every > 0 and epoch_idx % args.eval_every == 0
        if should_eval:
            eval_dir = output_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            overlays_dir = None if args.disable_eval_artifacts else output_dir / "eval_artifacts" / f"epoch_{epoch_idx:04d}" / "overlays"
            bev_dir = None if args.disable_eval_artifacts else output_dir / "eval_artifacts" / f"epoch_{epoch_idx:04d}" / "bev"
            summary = evaluate_samples(
                model=model,
                dataset=eval_dataset,
                sample_indices=eval_sample_indices,
                device=device,
                score_threshold=args.eval_score_threshold,
                max_boxes=args.eval_max_boxes,
                overlay_dir=overlays_dir,
                bev_dir=bev_dir,
                artifact_sample_indices=eval_artifact_sample_indices,
                verbose=False,
            )
            eval_summary_path = eval_dir / f"epoch_{epoch_idx:04d}.json"
            write_summary_json(eval_summary_path, summary)
            _log_mlflow_artifact(mlflow_run, eval_summary_path)
            if not args.disable_eval_artifacts:
                artifact_root = output_dir / "eval_artifacts" / f"epoch_{epoch_idx:04d}"
                _log_mlflow_artifacts(mlflow_run, artifact_root, f"eval_artifacts/epoch_{epoch_idx:04d}")
            _log_mlflow_metrics(mlflow_run, summary, step=epoch_idx, prefix="eval_")
            metric = summary.get("mean_center_distance")
            if metric is not None:
                print(f"eval epoch={epoch_idx} mean_center_distance={metric:.4f}")
                if metric < best_eval_metric:
                    best_eval_metric = float(metric)
                    save_checkpoint("best_eval_checkpoint.pt", combined_history)
        current_history.append(metrics)
        _log_mlflow_metrics(mlflow_run, metrics, step=epoch_idx, prefix="train_")
        if scheduler is not None:
            scheduler.step()

    try:
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
            safety_check=thermal_monitor,
            step_scheduler=step_scheduler,
            accumulation_steps=args.accumulation_steps,
        )
    except Exception:
        save_checkpoint("emergency_checkpoint.pt", prior_history + current_history)
        if mlflow_run is not None:
            mlflow_run.end_run(status="FAILED")
        raise
    history = prior_history + in_run_history

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _log_mlflow_artifact(mlflow_run, history_path)
    if eval_sample_indices:
        final_overlays_dir = None if args.disable_eval_artifacts else output_dir / "eval_artifacts" / "final" / "overlays"
        final_bev_dir = None if args.disable_eval_artifacts else output_dir / "eval_artifacts" / "final" / "bev"
        final_summary = evaluate_samples(
            model=model,
            dataset=eval_dataset,
            sample_indices=eval_sample_indices,
            device=device,
            score_threshold=args.eval_score_threshold,
            max_boxes=args.eval_max_boxes,
            overlay_dir=final_overlays_dir,
            bev_dir=final_bev_dir,
            artifact_sample_indices=eval_artifact_sample_indices,
            verbose=False,
        )
        final_eval_path = output_dir / "final_eval.json"
        write_summary_json(final_eval_path, final_summary)
        _log_mlflow_artifact(mlflow_run, final_eval_path)
        if not args.disable_eval_artifacts:
            _log_mlflow_artifacts(mlflow_run, output_dir / "eval_artifacts" / "final", "eval_artifacts/final")
        _log_mlflow_metrics(mlflow_run, final_summary, step=int(history[-1]["epoch"]) if history else 0, prefix="final_eval_")
        metric = final_summary.get("mean_center_distance")
        if metric is not None and metric < best_eval_metric:
            best_eval_metric = float(metric)
    save_checkpoint("last_checkpoint.pt", history)
    save_checkpoint("final_checkpoint.pt", history)
    if eval_sample_indices and (output_dir / "best_eval_checkpoint.pt").exists():
        print(f"best_eval_mean_center_distance={best_eval_metric:.4f}")
    if mlflow_run is not None:
        mlflow_run.end_run()


if __name__ == "__main__":
    main()
