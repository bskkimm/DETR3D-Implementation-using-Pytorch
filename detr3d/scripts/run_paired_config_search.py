"""Run and aggregate the commit-pinned B0-C6 configuration search."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "detr3d/experiments/paired_config_search_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--phase", choices=["one-sample", "small", "confirmation", "aggregate"], required=True
    )
    parser.add_argument("--variant", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--worktree-root", type=Path, default=Path("/tmp/opencode/detr3d_matrix_worktrees"))
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/paired_config_search_v1")
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def ensure_worktree(variant: str, commit: str, root: Path) -> Path:
    path = root / variant
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), commit],
            cwd=REPO_ROOT,
            check=True,
        )
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    expected = subprocess.run(
        ["git", "rev-parse", commit], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{variant} worktree is {actual}, expected {expected}")
    return path


def base_command(manifest: dict, output_dir: Path, seed: int, phase: str) -> list[str]:
    dataset = manifest["dataset"]
    protocol = manifest["protocol"]
    is_quality_phase = phase in {"small", "confirmation"}
    if phase == "one-sample":
        train_samples = 1
        epochs = 60
        eval_every = 60
        num_queries = 100
    else:
        train_samples = dataset["train_samples"]
        epochs = protocol["epochs"]
        eval_every = protocol["eval_epochs"][0]
        num_queries = protocol["num_queries"]

    command = [
        str(Path.home() / "miniconda3/envs/torch_env/bin/python"),
        "train.py",
        "--dataroot",
        dataset["dataroot"],
        "--version",
        dataset["version"],
        "--dataset-split",
        "train",
        "--max-samples",
        str(train_samples),
        "--batch-size",
        str(protocol["batch_size"] if is_quality_phase else 1),
        "--epochs",
        str(epochs),
        "--num-workers",
        str(protocol["num_workers"]),
        "--prefetch-factor",
        "2",
        "--pin-memory",
        "--persistent-workers",
        "--image-height",
        str(protocol["image_height"]),
        "--image-width",
        str(protocol["image_width"]),
        "--filter-gt-by-range",
        "--filter-zero-point-gt",
        "--num-queries",
        str(num_queries),
        "--lr",
        str(protocol["lr"]),
        "--backbone-lr-mult",
        str(protocol["backbone_lr_mult"]),
        "--weight-decay",
        str(protocol["weight_decay"]),
        "--scheduler",
        protocol["scheduler"],
        "--scheduler-total-epochs",
        str(epochs),
        "--warmup-steps",
        str(protocol["warmup_steps"] if is_quality_phase else 0),
        "--min-lr-ratio",
        str(protocol["min_lr_ratio"]),
        "--grad-clip-norm",
        str(protocol["grad_clip_norm"]),
        "--output-dir",
        str(output_dir),
        "--num-eval-samples",
        str(dataset["val_samples"] if is_quality_phase else 1),
        "--eval-every",
        str(eval_every),
        "--eval-score-threshold",
        "0.005",
        "--eval-max-boxes",
        "100",
        "--disable-eval-artifacts",
        "--seed",
        str(seed),
        "--deterministic",
        "--thermal-action",
        "pause",
        "--max-gpu-temp",
        "90",
        "--max-cpu-temp",
        "90",
        "--resume-gpu-temp",
        "80",
        "--resume-cpu-temp",
        "80",
    ]
    if is_quality_phase:
        command.extend(
            [
                "--val-split",
                "val",
                "--max-val-samples",
                str(dataset["val_samples"]),
            ]
        )
    return command


def selected_values(requested: str, available: list) -> list:
    if requested == "all":
        return available
    value = type(available[0])(requested)
    if value not in available:
        raise ValueError(f"Unknown selection {requested!r}; expected one of {available}")
    return [value]


def run_matrix(args: argparse.Namespace, manifest: dict) -> None:
    variants = selected_values(args.variant, list(manifest["variants"]))
    seeds = selected_values(args.seed, manifest["seeds"])
    for variant in variants:
        definition = manifest["variants"][variant]
        worktree = ensure_worktree(variant, definition["commit"], args.worktree_root)
        for seed in seeds:
            run_dir = args.output_root / args.phase / variant / f"seed_{seed}"
            status_path = run_dir / "status.json"
            if status_path.exists() and json.loads(status_path.read_text())["status"] == "completed":
                print(f"skip completed {args.phase} {variant} seed={seed}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            command = base_command(manifest, run_dir, seed, args.phase)
            command.extend(definition.get("extra_args", []))
            status = {
                "matrix_id": manifest["matrix_id"],
                "phase": args.phase,
                "variant": variant,
                "commit": definition["commit"],
                "seed": seed,
                "command": command,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            try:
                run_command(command, cwd=worktree, log_path=run_dir / "run.log")
            except Exception:
                status["status"] = "failed"
                status["finished_at"] = datetime.now(timezone.utc).isoformat()
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                raise
            status["status"] = "completed"
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            if args.phase in {"small", "confirmation"}:
                removed = []
                for checkpoint_path in run_dir.glob("*.pt"):
                    removed.append(checkpoint_path.name)
                    checkpoint_path.unlink()
                status["removed_checkpoints"] = sorted(removed)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def aggregate(args: argparse.Namespace, manifest: dict) -> None:
    rows = []
    phase = manifest.get("aggregate_phase", "small")
    for variant, definition in manifest["variants"].items():
        for seed in manifest["seeds"]:
            run_dir = args.output_root / phase / variant / f"seed_{seed}"
            for epoch in manifest["protocol"]["eval_epochs"]:
                result_path = run_dir / "eval" / f"epoch_{epoch:04d}.json"
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                rows.append(
                    {
                        "variant": variant,
                        "commit": definition["commit"],
                        "seed": seed,
                        "epoch": epoch,
                        "mean_center_distance": result["mean_center_distance"],
                        "mean_median_center_distance": result["mean_median_center_distance"],
                        "class_matches": result["total_class_matches"],
                        "total_gt": result["total_gt"],
                        "class_match_fraction": result["total_class_matches"] / result["total_gt"],
                    }
                )
    aggregate_path = args.output_root / "aggregate.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.phase == "aggregate":
        aggregate(args, manifest)
    else:
        run_matrix(args, manifest)


if __name__ == "__main__":
    main()
