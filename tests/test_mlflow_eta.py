from datetime import datetime, timezone

import pytest

from detr3d.scripts.monitor_mlflow_eta import (
    completed_eval_times,
    estimate_progress,
    format_duration,
    overview_note,
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
    assert result["elapsed_hours"] == pytest.approx(35_000 / 3600)
    assert result["mean_epoch_hours"] == pytest.approx(10_000 / 3600)
    assert result["current_epoch"] == 4
    assert result["current_epoch_progress_percent"] == pytest.approx(50)
    assert result["estimated_total_hours"] == pytest.approx(60_000 / 3600)
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


def test_estimate_progress_marks_final_epoch_complete():
    result = estimate_progress(
        start_time=0,
        completion_times=[(1, 100), (2, 200)],
        total_epochs=2,
        now=210,
    )

    assert result["current_epoch"] == 2
    assert result["current_epoch_progress_percent"] == 100


def test_estimate_progress_anchors_incomplete_epoch_at_resume_time():
    result = estimate_progress(
        start_time=0,
        completion_times=[(1, 100), (2, 200)],
        total_epochs=4,
        now=1010,
        current_epoch_start_time=1000,
    )

    assert result["current_epoch"] == 3
    assert result["current_epoch_progress_percent"] == pytest.approx(10)
    assert result["eta_hours"] == pytest.approx(190 / 3600)
    assert result["expected_finish_unix_seconds"] == 1200


def test_overview_note_formats_progress_for_mlflow_description():
    note = overview_note(
        estimate={
            "completed_epochs": 4.0,
            "progress_percent": 100 / 6,
            "elapsed_hours": 13.5,
            "mean_epoch_hours": 2.75,
            "estimated_total_hours": 66.6,
            "eta_hours": 53.1,
            "current_epoch": 5.0,
            "current_epoch_progress_percent": 42.25,
        },
        total_epochs=24,
        expected_finish_local="2026-07-20 18:02:25 JST",
        updated_at_local="2026-07-18 13:10:00 JST",
    )

    assert "**Elapsed:** 13h 30m" in note
    assert "**Average epoch:** 2h 45m" in note
    assert "**Estimated total:** 2d 18h 36m" in note
    assert "**Remaining:** 2d 5h 6m" in note
    assert "4 / 24 epochs (16.7%)" in note
    assert "**Current epoch 5:** 42.2% estimated" in note
    assert "2026-07-20 18:02:25 JST" in note


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(0.0, "0m"), (1.5, "1h 30m"), (25.25, "1d 1h 15m")],
)
def test_format_duration(hours, expected):
    assert format_duration(hours) == expected
