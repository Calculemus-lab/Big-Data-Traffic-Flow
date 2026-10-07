"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trafficbench import table_types
from trafficbench.contracts import (
    LWR_MAINLINE_TOPOLOGY_FILE_STEM,
    QUEUE_FORECAST_STEPS,
    JsonObject,
    NetworkTables,
    Panel,
    ReleasePackageSlice,
    Split,
    validate_predictions,
)
from trafficbench.data import (
    Release,
    load_release_slice,
)
from trafficbench.fixture import make_fixture
from trafficbench.metrics import aggregate, queue_metrics, state_metrics


def test_queue_scores_each_complete_six_step_window_before_averaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench import inference

    panel: Panel = "D12_I5_N"
    release_slice = load_release_slice(
        Release(make_fixture(tmp_path / "release")),
        date(2030, 11, 3),
        date(2030, 11, 20),
        date(2030, 11, 21),
        date(2030, 11, 22),
        [panel],
    )
    forecast_start_timestamps = (
        datetime(2030, 11, 21, 0, 5, tzinfo=UTC),
        datetime(2030, 11, 21, 6, 5, tzinfo=UTC),
    )
    target_tables = [
        pl.DataFrame(
            {
                "window_id": [window_id] * QUEUE_FORECAST_STEPS,
                "timestamp": pl.datetime_range(
                    forecast_start,
                    forecast_start + timedelta(minutes=5 * (QUEUE_FORECAST_STEPS - 1)),
                    interval="5m",
                    eager=True,
                ),
                "link_id": ["L01"] * QUEUE_FORECAST_STEPS,
            }
        )
        for window_id, forecast_start in zip(
            ("window_a", "window_b"), forecast_start_timestamps
        )
    ]
    all_target_keys = pl.concat(target_tables)
    target_mapping: dict[Panel, dict[Split, pl.DataFrame]] = {
        panel: {"train": all_target_keys}
    }
    answer_mapping: dict[Panel, dict[Split, pl.DataFrame]] = {
        panel: {"train": all_target_keys.with_columns(pl.lit(1).alias("queue_pred"))}
    }

    def predict_two_windows(
        _release_slice: ReleasePackageSlice,
        requested_targets: dict[Panel, dict[Split, table_types.QueueFrame]],
        _solution_parameters: JsonObject,
    ) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
        """Predict both test windows while intentionally missing one window."""
        panel_targets = requested_targets[panel]["train"]
        predictions = panel_targets.with_columns(
            (pl.col("window_id") == "window_a").cast(pl.Int64).alias("queue_pred")
        )
        return {panel: {"train": predictions}}

    monkeypatch.setattr(
        inference, "_load_predictor", lambda _name, _task: predict_two_windows
    )
    result = inference.run_solution(
        "per_window_test",
        "queue",
        release_slice,
        target_mapping,
        answer_mapping,
    )

    assert len(result.metrics) == 2
    assert result.metrics.sort("window_id").select(
        "window_id", "queue_proxy_iou"
    ).to_dicts() == [
        {"window_id": "window_a", "queue_proxy_iou": 1.0},
        {"window_id": "window_b", "queue_proxy_iou": 0.0},
    ]


def test_state_formula_per_lane_and_regime() -> None:
    targets = pl.DataFrame(
        {
            "panel": ["A"] * 3,
            "timestamp": pl.datetime_range(
                datetime(2030, 1, 1, tzinfo=UTC),
                datetime(2030, 1, 3, tzinfo=UTC),
                interval="1d",
                eager=True,
            ),
            "station_id": ["S"] * 3,
            "link_id": ["L"] * 3,
            "mask_regime": ["R1", "R2", "R3"],
        }
    )
    truth = targets.with_columns(
        pl.lit(100.0).alias("speed_kmh"), pl.lit(2000.0).alias("flow_vph")
    )
    pred = targets.with_columns(
        pl.lit(90.0).alias("speed_kmh"), pl.lit(2400.0).alias("flow_vph")
    )
    empty_network_table = pl.DataFrame().lazy()
    network = NetworkTables.from_mapping(
        {
            "links": pl.DataFrame(schema={"link_id": pl.String}).lazy(),
            "fd_parameters": pl.DataFrame({"link_id": ["L"], "lanes": [4.0]}).lazy(),
            "path_set": pl.DataFrame(schema={"path_id": pl.String}).lazy(),
            "path_link_incidence": pl.DataFrame(
                schema={"link_id": pl.String, "path_id": pl.String}
            ).lazy(),
            LWR_MAINLINE_TOPOLOGY_FILE_STEM: empty_network_table,
            "ramp_attachment_map": empty_network_table,
            "synthetic_ramp_attachment_map": empty_network_table,
        }
    )
    rows = state_metrics(pred, truth, network)
    assert all(r["flow_per_lane_rmse"] == 100 for r in rows)
    assert all(
        r["state_score"] == pytest.approx(0.54 * 0.6 + 0.46 * (1 - 100 / 600))
        for r in rows
    )


def test_equal_family_and_condition_weighting() -> None:
    rows = [
        {
            "case_id": "a",
            "split": "train",
            "panel": "A_N",
            "family": "A",
            "condition": "onset",
            "queue_proxy_iou": 1.0,
        }
    ] * 20
    rows += [
        {
            "case_id": "a",
            "split": "train",
            "panel": "A_N",
            "family": "A",
            "condition": "ongoing",
            "queue_proxy_iou": 0.0,
        }
    ]
    rows += [
        {
            "case_id": "a",
            "split": "train",
            "panel": "B_N",
            "family": "B",
            "condition": "onset",
            "queue_proxy_iou": 0.0,
        }
    ]
    assert aggregate(pl.DataFrame(rows))["queue_proxy_iou"] == 0.25


def test_queue_ignores_unknown_truth_and_scores_iou() -> None:
    keys = pl.DataFrame(
        {
            "window_id": ["w"] * 3,
            "timestamp": pl.datetime_range(
                datetime(2030, 1, 1, tzinfo=UTC),
                datetime(2030, 1, 3, tzinfo=UTC),
                interval="1d",
                eager=True,
            ),
            "link_id": ["L"] * 3,
        }
    )
    pred = keys.with_columns(pl.Series("queue_pred", [1, 1, 1]))
    truth = keys.with_columns(
        pl.Series("queue_true", [1, 0, 0]),
        pl.Series("eligible", [True, True, False]),
    )
    assert queue_metrics(pred, truth)["queue_proxy_iou"] == 0.5


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "nan", "binary"])
def test_reject_invalid_predictions(bad: str) -> None:
    targets = pl.DataFrame(
        {
            "window_id": ["w", "w"],
            "timestamp": [
                datetime(2030, 1, 1, tzinfo=UTC),
                datetime(2030, 1, 2, tzinfo=UTC),
            ],
            "link_id": ["L", "L"],
        }
    )
    predictions = targets.with_columns(pl.Series("queue_pred", [0.0, 1.0]))
    match bad:
        case "missing":
            predictions = predictions.head(1)
        case "duplicate":
            predictions = pl.concat([predictions, predictions.tail(1)])
        case "unknown":
            predictions = (
                predictions.with_row_index("row_index")
                .with_columns(
                    pl.when(pl.col("row_index") == 1)
                    .then(pl.lit("bad"))
                    .otherwise(pl.col("link_id"))
                    .alias("link_id")
                )
                .drop("row_index")
            )
        case "nan":
            predictions = (
                predictions.with_row_index("row_index")
                .with_columns(
                    pl.when(pl.col("row_index") == 1)
                    .then(pl.lit(float("nan")))
                    .otherwise(pl.col("queue_pred"))
                    .alias("queue_pred")
                )
                .drop("row_index")
            )
        case "binary":
            predictions = (
                predictions.with_row_index("row_index")
                .with_columns(
                    pl.when(pl.col("row_index") == 1)
                    .then(pl.lit(0.5))
                    .otherwise(pl.col("queue_pred"))
                    .alias("queue_pred")
                )
                .drop("row_index")
            )
    with pytest.raises(ValueError):
        validate_predictions("queue", targets.lazy(), predictions.lazy())
