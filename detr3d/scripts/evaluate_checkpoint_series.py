"""Run official nuScenes evaluation for a sequence of training checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--epochs", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument(
        "--existing-summary",
        action="append",
        default=[],
        metavar="EPOCH=PATH",
    )
    parser.add_argument("--keep-results", action="store_true")
    return parser.parse_args()


def parse_existing_summaries(values: list[str]) -> dict[int, Path]:
    summaries = {}
    for value in values:
        epoch_text, separator, path_text = value.partition("=")
        if not separator or not epoch_text or not path_text:
            raise ValueError(f"Expected EPOCH=PATH, got {value!r}")
        epoch = int(epoch_text)
        if epoch in summaries:
            raise ValueError(f"Duplicate existing summary for epoch {epoch}")
        summaries[epoch] = Path(path_text)
    return summaries


def metric_record(epoch: int, summary_path: Path) -> dict[str, float | int | str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = summary["tp_errors"]
    return {
        "epoch": epoch,
        "mAP": float(summary["mean_ap"]),
        "NDS": float(summary["nd_score"]),
        "mATE": float(errors["trans_err"]),
        "mASE": float(errors["scale_err"]),
        "mAOE": float(errors["orient_err"]),
        "mAVE": float(errors["vel_err"]),
        "mAAE": float(errors["attr_err"]),
        "summary_path": str(summary_path),
    }


def write_aggregate(records: list[dict], output_dir: Path) -> None:
    records = sorted(records, key=lambda row: row["epoch"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "epoch_metrics.json"
    json_temp = json_path.with_suffix(".json.tmp")
    json_temp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    json_temp.replace(json_path)

    csv_path = output_dir / "epoch_metrics.csv"
    csv_temp = csv_path.with_suffix(".csv.tmp")
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    csv_temp.replace(csv_path)


def main() -> None:
    args = parse_args()
    existing = parse_existing_summaries(args.existing_summary)
    repo_root = Path(__file__).resolve().parents[2]
    records = []
    mlflow_client = None
    if args.mlflow_run_id is not None:
        from mlflow.tracking import MlflowClient

        mlflow_client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)

    for epoch in sorted(set(args.epochs)):
        epoch_dir = args.output_dir / f"epoch_{epoch:04d}"
        summary_path = epoch_dir / "metrics" / "metrics_summary.json"
        if epoch in existing:
            summary_path = existing[epoch]
        elif not summary_path.exists():
            checkpoint = args.checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            epoch_dir.mkdir(parents=True, exist_ok=True)
            results_path = epoch_dir / "results_nusc.json"
            log_path = epoch_dir / "evaluation.log"
            command = [
                sys.executable,
                str(repo_root / "eval.py"),
                "--checkpoint",
                str(checkpoint),
                "--dataroot",
                args.dataroot,
                "--version",
                args.version,
                "--dataset-split",
                args.dataset_split,
                "--nuscenes-results-out",
                str(results_path),
                "--run-nuscenes-eval",
                "--nuscenes-eval-set",
                args.dataset_split,
                "--nuscenes-eval-output-dir",
                str(epoch_dir / "metrics"),
                "--device",
                args.device,
            ]
            print(f"Evaluating epoch {epoch}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_handle.write(line)
                    log_handle.flush()
                return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            if not args.keep_results:
                results_path.unlink(missing_ok=True)

        records = [row for row in records if row["epoch"] != epoch]
        records.append(metric_record(epoch, summary_path))
        write_aggregate(records, args.output_dir)
        current = records[-1]
        if mlflow_client is not None:
            for key in ("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"):
                mlflow_client.log_metric(
                    args.mlflow_run_id,
                    f"official_epoch_{key.lower()}",
                    current[key],
                    step=epoch,
                )
        print(
            f"epoch={epoch} mAP={current['mAP']:.4f} NDS={current['NDS']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
