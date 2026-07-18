from datetime import datetime, timezone

import pytest

from detr3d.scripts.monitor_mlflow_eta import (
    completed_eval_times,
    estimate_progress,
)


def test_estimate_progress_uses_completed_epoch_cycle_time():
    start = datetime(2026, 7, 17, tzinfo=timezone.utc).timestamp()
    completion_times = [
        (1, start + 10_000),
        (2, start + 20_000),
        (3, start + 30_000),
    ]

    result = estimate_progress(
        start_time=start,
        completion_times=completion_times,
        total_epochs=6,
        now=start + 35_000,
    )

    assert result["completed_epochs"] == 3
    assert result["progress_percent"] == 50
    assert result["mean_epoch_hours"] == pytest.approx(10_000 / 3600)
    assert result["eta_hours"] == pytest.approx(25_000 / 3600)
    assert result["expected_finish_unix_seconds"] == start + 60_000


def test_completed_eval_times_ignores_other_files(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    first = eval_dir / "epoch_0002.json"
    second = eval_dir / "epoch_0001.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    (eval_dir / "summary.json").write_text("{}", encoding="utf-8")

    result = completed_eval_times(tmp_path)

    assert [epoch for epoch, _ in result] == [1, 2]


def test_estimate_progress_rejects_invalid_total_epochs():
    with pytest.raises(ValueError, match="positive"):
        estimate_progress(
            start_time=0,
            completion_times=[],
            total_epochs=0,
            now=0,
        )
