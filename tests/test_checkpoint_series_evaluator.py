import json

import pytest

from detr3d.scripts.evaluate_checkpoint_series import (
    metric_record,
    parse_existing_summaries,
    write_aggregate,
)


def test_parse_existing_summaries():
    result = parse_existing_summaries(["18=/tmp/epoch18.json", "24=/tmp/final.json"])

    assert str(result[18]) == "/tmp/epoch18.json"
    assert str(result[24]) == "/tmp/final.json"


def test_parse_existing_summaries_rejects_invalid_value():
    with pytest.raises(ValueError, match="EPOCH=PATH"):
        parse_existing_summaries(["18"])


def test_metric_record_and_aggregate_outputs(tmp_path):
    summary_path = tmp_path / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mean_ap": 0.34,
                "nd_score": 0.42,
                "tp_errors": {
                    "trans_err": 0.7,
                    "scale_err": 0.2,
                    "orient_err": 0.3,
                    "vel_err": 0.8,
                    "attr_err": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    record = metric_record(3, summary_path)
    write_aggregate([record], tmp_path / "aggregate")

    assert record["epoch"] == 3
    assert record["mAP"] == 0.34
    assert json.loads(
        (tmp_path / "aggregate" / "epoch_metrics.json").read_text(encoding="utf-8")
    ) == [record]
    assert "epoch,mAP,NDS" in (
        tmp_path / "aggregate" / "epoch_metrics.csv"
    ).read_text(encoding="utf-8")
