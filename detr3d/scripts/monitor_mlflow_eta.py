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
    parser.add_argument("--overview-interval", type=float, default=300.0)
    parser.add_argument("--timezone", default="Asia/Tokyo")
    parser.add_argument("--training-pid", type=int, default=None)
    parser.add_argument("--current-epoch-start-time", type=float, default=None)
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
    current_epoch_start_time: float | None = None,
) -> dict[str, float]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    completed_epochs = completion_times[-1][0] if completion_times else 0
    if completed_epochs > total_epochs:
        raise ValueError("completed epochs exceed total epochs")

    result = {
        "completed_epochs": float(completed_epochs),
        "progress_percent": 100.0 * completed_epochs / total_epochs,
        "elapsed_hours": max(now - start_time, 0.0) / 3600.0,
    }
    if completed_epochs == 0:
        return result

    last_completion = completion_times[-1][1]
    mean_epoch_seconds = (last_completion - start_time) / completed_epochs
    current_epoch_start = max(
        last_completion,
        current_epoch_start_time or last_completion,
    )
    expected_finish = current_epoch_start + mean_epoch_seconds * (
        total_epochs - completed_epochs
    )
    result.update(
        {
            "mean_epoch_hours": mean_epoch_seconds / 3600.0,
            "estimated_total_hours": (expected_finish - start_time) / 3600.0,
            "eta_hours": max(expected_finish - now, 0.0) / 3600.0,
            "expected_finish_unix_seconds": expected_finish,
        }
    )
    if completed_epochs >= total_epochs:
        result["current_epoch"] = float(total_epochs)
        result["current_epoch_progress_percent"] = 100.0
    elif mean_epoch_seconds > 0:
        result["current_epoch"] = float(completed_epochs + 1)
        result["current_epoch_progress_percent"] = min(
            100.0 * max(now - current_epoch_start, 0.0) / mean_epoch_seconds,
            99.9,
        )
    return result


def format_duration(hours: float) -> str:
    total_minutes = max(int(round(hours * 60)), 0)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    duration_hours, minutes = divmod(remaining_minutes, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if duration_hours or days:
        parts.append(f"{duration_hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def overview_note(
    *,
    estimate: dict[str, float],
    total_epochs: int,
    expected_finish_local: str | None,
    updated_at_local: str,
) -> str:
    completed_epochs = int(estimate["completed_epochs"])
    lines = [
        "## Training Progress",
        "",
        f"- **Elapsed:** {format_duration(estimate['elapsed_hours'])}",
    ]
    if "estimated_total_hours" in estimate:
        lines.extend(
            [
                f"- **Average epoch:** {format_duration(estimate['mean_epoch_hours'])}",
                f"- **Estimated total:** {format_duration(estimate['estimated_total_hours'])}",
                f"- **Remaining:** {format_duration(estimate['eta_hours'])}",
            ]
        )
    lines.append(
        f"- **Progress:** {completed_epochs} / {total_epochs} epochs "
        f"({estimate['progress_percent']:.1f}%)"
    )
    if "current_epoch_progress_percent" in estimate:
        lines.append(
            f"- **Current epoch {int(estimate['current_epoch'])}:** "
            f"{estimate['current_epoch_progress_percent']:.1f}% estimated"
        )
    if expected_finish_local is not None:
        lines.append(f"- **Expected finish:** {expected_finish_local}")
    lines.extend(["", f"Last updated: {updated_at_local}"])
    return "\n".join(lines)


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
    current_epoch_start_time: float | None = None,
    log_epoch_metrics: bool = True,
    update_overview: bool = True,
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
        current_epoch_start_time=current_epoch_start_time,
    )
    completed_epochs = int(estimate["completed_epochs"])
    timestamp_ms = int(now * 1000)
    metric_names = {
        "completed_epochs": "training_completed_epochs",
        "progress_percent": "training_progress_percent",
        "elapsed_hours": "training_elapsed_hours",
        "mean_epoch_hours": "training_mean_epoch_hours",
        "estimated_total_hours": "training_estimated_total_hours",
        "eta_hours": "training_eta_hours",
        "expected_finish_unix_seconds": "training_expected_finish_unix_seconds",
    }
    if log_epoch_metrics:
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
    expected_finish_local = None
    if "expected_finish_unix_seconds" in estimate:
        expected_finish = datetime.fromtimestamp(
            estimate["expected_finish_unix_seconds"], timezone.utc
        ).astimezone(local_zone)
        expected_finish_local = expected_finish.strftime("%Y-%m-%d %H:%M:%S %Z")
    if update_overview:
        updated_at_local = updated_at.strftime("%Y-%m-%d %H:%M:%S %Z")
        client.set_tag(run_id, "eta_monitor_updated_at", updated_at.isoformat())
        client.set_tag(run_id, "eta_monitor_timezone", timezone_name)
        if expected_finish_local is not None:
            client.set_tag(
                run_id, "training_expected_finish_local", expected_finish_local
            )
        client.set_tag(
            run_id,
            "mlflow.note.content",
            overview_note(
                estimate=estimate,
                total_epochs=total_epochs,
                expected_finish_local=expected_finish_local,
                updated_at_local=updated_at_local,
            ),
        )
    return completed_epochs


def main() -> None:
    args = parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)
    last_completed_epochs = -1
    last_overview_update = 0.0
    while True:
        now = time.monotonic()
        completion_times = completed_eval_times(args.output_dir)
        observed_completed_epochs = completion_times[-1][0] if completion_times else 0
        epoch_changed = observed_completed_epochs != last_completed_epochs
        overview_due = now - last_overview_update >= args.overview_interval
        completed_epochs = log_progress(
            client=client,
            run_id=args.run_id,
            output_dir=args.output_dir,
            total_epochs=args.total_epochs,
            timezone_name=args.timezone,
            current_epoch_start_time=args.current_epoch_start_time,
            log_epoch_metrics=epoch_changed,
            update_overview=epoch_changed or overview_due,
        )
        if epoch_changed:
            last_completed_epochs = completed_epochs
        if epoch_changed or overview_due:
            last_overview_update = now
        if completed_epochs >= args.total_epochs or args.once:
            return
        if not process_is_alive(args.training_pid):
            client.set_tag(args.run_id, "eta_monitor_status", "training_process_stopped")
            return
        if epoch_changed or overview_due:
            client.set_tag(args.run_id, "eta_monitor_status", "running")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
