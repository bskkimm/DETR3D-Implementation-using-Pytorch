"""Log training progress and estimated finish time to an MLflow run."""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from mlflow.tracking import MlflowClient

EVAL_PATTERN = re.compile(r"epoch_(\d{4})\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--timezone", default="Asia/Tokyo")
    parser.add_argument("--training-pid", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def completed_eval_times(output_dir: Path) -> list[tuple[int, float]]:
    completed = []
    for path in (output_dir / "eval").glob("epoch_*.json"):
        match = EVAL_PATTERN.match(path.name)
        if match:
            completed.append((int(match.group(1)), path.stat().st_mtime))
    return sorted(completed)


def estimate_progress(
    *,
    start_time: float,
    completion_times: list[tuple[int, float]],
    total_epochs: int,
    now: float,
) -> dict[str, float]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    completed_epochs = completion_times[-1][0] if completion_times else 0
    if completed_epochs > total_epochs:
        raise ValueError("completed epochs exceed total epochs")

    result = {
        "completed_epochs": float(completed_epochs),
        "progress_percent": 100.0 * completed_epochs / total_epochs,
    }
    if completed_epochs == 0:
        return result

    last_completion = completion_times[-1][1]
    mean_epoch_seconds = (last_completion - start_time) / completed_epochs
    expected_finish = last_completion + mean_epoch_seconds * (
        total_epochs - completed_epochs
    )
    result.update(
        {
            "mean_epoch_hours": mean_epoch_seconds / 3600.0,
            "eta_hours": max(expected_finish - now, 0.0) / 3600.0,
            "expected_finish_unix_seconds": expected_finish,
        }
    )
    return result


def process_is_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def log_progress(
    *,
    client: MlflowClient,
    run_id: str,
    output_dir: Path,
    total_epochs: int,
    timezone_name: str,
) -> int:
    run = client.get_run(run_id)
    start_time = run.info.start_time / 1000.0
    now = time.time()
    completion_times = completed_eval_times(output_dir)
    estimate = estimate_progress(
        start_time=start_time,
        completion_times=completion_times,
        total_epochs=total_epochs,
        now=now,
    )
    completed_epochs = int(estimate["completed_epochs"])
    timestamp_ms = int(now * 1000)
    metric_names = {
        "completed_epochs": "training_completed_epochs",
        "progress_percent": "training_progress_percent",
        "mean_epoch_hours": "training_mean_epoch_hours",
        "eta_hours": "training_eta_hours",
        "expected_finish_unix_seconds": "training_expected_finish_unix_seconds",
    }
    for source_name, metric_name in metric_names.items():
        if source_name in estimate:
            client.log_metric(
                run_id,
                metric_name,
                estimate[source_name],
                timestamp=timestamp_ms,
                step=completed_epochs,
            )

    local_zone = ZoneInfo(timezone_name)
    updated_at = datetime.fromtimestamp(now, timezone.utc).astimezone(local_zone)
    client.set_tag(run_id, "eta_monitor_updated_at", updated_at.isoformat())
    client.set_tag(run_id, "eta_monitor_timezone", timezone_name)
    if "expected_finish_unix_seconds" in estimate:
        expected_finish = datetime.fromtimestamp(
            estimate["expected_finish_unix_seconds"], timezone.utc
        ).astimezone(local_zone)
        client.set_tag(
            run_id,
            "training_expected_finish_local",
            expected_finish.strftime("%Y-%m-%d %H:%M:%S %Z"),
        )
    return completed_epochs


def main() -> None:
    args = parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    while True:
        completed_epochs = log_progress(
            client=client,
            run_id=args.run_id,
            output_dir=args.output_dir,
            total_epochs=args.total_epochs,
            timezone_name=args.timezone,
        )
        if completed_epochs >= args.total_epochs or args.once:
            return
        if not process_is_alive(args.training_pid):
            client.set_tag(args.run_id, "eta_monitor_status", "training_process_stopped")
            return
        client.set_tag(args.run_id, "eta_monitor_status", "running")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
