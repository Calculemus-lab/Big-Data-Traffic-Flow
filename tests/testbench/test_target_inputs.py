"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    MASKED_STATE_COLUMNS,
    QUEUE_FORECAST_STEPS,
    VALUES,
    JsonObject,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.data import (
    Release,
    load_release_slice,
    load_target_templates,
)
from trafficbench.fixture import make_fixture
from trafficbench.runner import (
    _partition_answer_key,
)


def test_queue_target_ranges_keep_only_complete_windows(tmp_path: Path) -> None:
    release_directory = make_fixture(tmp_path / "release")
    panel: Panel = "D12_I5_N"
    template_path = (
        release_directory / "task2" / panel / "train" / "sample_submission_queue.csv"
    )
    template_path.parent.mkdir(parents=True)
    complete_window_start = datetime(2030, 11, 21, 10, 5, tzinfo=UTC)
    complete_window = pl.DataFrame(
        {
            "window_id": ["complete"] * QUEUE_FORECAST_STEPS,
            "timestamp": pl.datetime_range(
                complete_window_start,
                complete_window_start
                + timedelta(minutes=5 * (QUEUE_FORECAST_STEPS - 1)),
                interval="5m",
                eager=True,
            ),
            "link_id": ["L01"] * QUEUE_FORECAST_STEPS,
            "queue_pred": [0] * QUEUE_FORECAST_STEPS,
        }
    )
    end_crossing_window_start = datetime(2030, 11, 21, 23, 40, tzinfo=UTC)
    end_crossing_window = pl.DataFrame(
        {
            "window_id": ["crosses_end"] * QUEUE_FORECAST_STEPS,
            "timestamp": pl.datetime_range(
                end_crossing_window_start,
                end_crossing_window_start
                + timedelta(minutes=5 * (QUEUE_FORECAST_STEPS - 1)),
                interval="5m",
                eager=True,
            ),
            "link_id": ["L01"] * QUEUE_FORECAST_STEPS,
            "queue_pred": [0] * QUEUE_FORECAST_STEPS,
        }
    )
    pl.concat([complete_window, end_crossing_window]).write_csv(template_path)

    selected_targets = load_target_templates(
        Release(release_directory),
        "queue",
        date(2030, 11, 21),
        date(2030, 11, 22),
        [panel],
    )

    queue_targets = selected_targets[panel]["train"]
    assert queue_targets["window_id"].unique().to_list() == ["complete"]
    assert len(queue_targets) == QUEUE_FORECAST_STEPS
    incomplete_answer_key = queue_targets.head(1).with_columns(
        pl.lit(0).alias("queue_pred")
    )
    with pytest.raises(ValueError, match="every target row"):
        _partition_answer_key("queue", incomplete_answer_key, selected_targets)


def test_answer_keys_select_a_subset_without_entering_target_tables(
    tmp_path: Path,
) -> None:
    release_directory = make_fixture(tmp_path / "release")
    panel: Panel = "D12_I5_N"
    release = Release(release_directory)
    target_templates = load_target_templates(
        release,
        "state",
        date(2030, 11, 20),
        date(2030, 11, 22),
        [panel],
    )
    answer_table = (
        target_templates[panel]["train"]
        .head(1)
        .with_columns(
            pl.lit(40.0).alias("speed_kmh"),
            pl.lit(1200.0).alias("flow_vph"),
            pl.lit("train").alias("split"),
        )
    )

    answers_by_panel = _partition_answer_key("state", answer_table, target_templates)
    requested_keys = answers_by_panel[panel]["train"][list(KEYS["state"])]
    assert len(requested_keys) == 1
    assert set(requested_keys.columns) == set(KEYS["state"])
    assert answers_by_panel[panel]["train"]["speed_kmh"].item() == 40.0


def test_answer_values_are_scored_after_zero_templates_reach_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench import inference

    panel: Panel = "D12_I5_N"
    release_directory = make_fixture(tmp_path / "release")
    release_slice = load_release_slice(
        Release(release_directory),
        date(2030, 11, 3),
        date(2030, 11, 20),
        date(2030, 11, 21),
        date(2030, 11, 22),
        [panel],
    )
    forecast_timestamp = datetime(2030, 11, 21, 0, 5, tzinfo=UTC)
    target_keys = pl.DataFrame(
        {
            "window_id": ["local_window"],
            "timestamp": [forecast_timestamp],
            "link_id": ["L01"],
        }
    )
    # A later released window can include this forecast timestamp in its
    # permitted history, so redact that overlapping route as well.
    panel_data = release_slice.panel_slices_by_panel[panel]
    queue_history = pl.DataFrame(
        {
            "window_id": ["later_window"],
            "timestamp": [forecast_timestamp],
            "link_id": ["L01"],
            "speed_kmh": [40.0],
            "flow_vph": [100.0],
            "occupancy": [0.2],
            "pct_observed": [100],
            "is_score_eligible": [1],
        }
    ).lazy()
    release_slice = replace(
        release_slice,
        panel_slices_by_panel={
            panel: replace(panel_data, queue_history={"train": queue_history})
        },
    )
    targets: dict[Panel, dict[Split, pl.DataFrame]] = {panel: {"train": target_keys}}
    answers: dict[Panel, dict[Split, pl.DataFrame]] = {
        panel: {"train": target_keys.with_columns(pl.lit(1).alias("queue_pred"))}
    }
    solution_received_zero_template = False

    def predict_queue(
        visible_slice: ReleasePackageSlice,
        requested_targets: dict[Panel, dict[Split, table_types.QueueFrame]],
        _solution_parameters: JsonObject,
    ) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
        """Inspect what the solution sees before returning queue predictions."""
        nonlocal solution_received_zero_template
        requested_template = requested_targets[panel]["train"]
        requested_template_table = requested_template.collect()
        solution_received_zero_template = (
            set(requested_template_table.columns)
            == {
                *KEYS["queue"],
                "queue_pred",
            }
            and requested_template_table.get_column("queue_pred").eq(0).all()
        )
        panel_data = visible_slice.panel_slices_by_panel[panel]
        assert set(VALUES["state"]).issubset(
            panel_data.historical_task_labels.task1_state_answers.collect_schema().names()
        )
        assert (
            "queue_pred"
            in panel_data.historical_task_labels.task2_queue_proxy_labels.collect_schema().names()
        )
        assert (
            "queue_true"
            not in panel_data.queue_history["train"].collect_schema().names()
        )
        queue_history_table = panel_data.queue_history["train"].collect()
        assert queue_history_table.select(
            pl.col(
                [
                    "speed_kmh",
                    "flow_vph",
                    "occupancy",
                    "pct_observed",
                    "is_score_eligible",
                ]
            )
            .is_null()
            .all()
        ).row(0) == (True, True, True, True, True)
        forecast_timestamps = requested_template_table.get_column("timestamp").to_list()
        state_table = panel_data.masked_mainline_states["train"].collect()
        hidden_state_rows = state_table.filter(
            pl.col("timestamp").is_in(forecast_timestamps)
        )
        queue_label_columns = [
            *MASKED_STATE_COLUMNS,
            "pct_observed",
            "is_observed",
            "is_imputed",
            "is_score_eligible",
            "is_missing",
        ]
        visible_mainline_columns = [
            column for column in queue_label_columns if column in state_table.columns
        ]
        assert hidden_state_rows.select(
            pl.col(visible_mainline_columns).is_null().all()
        ).row(0) == (True,) * len(visible_mainline_columns)
        ramp_table = panel_data.ramp_states["train"].collect()
        hidden_ramp_rows = ramp_table.filter(
            pl.col("timestamp").is_in(forecast_timestamps)
        )
        assert hidden_ramp_rows.height > 0
        ramp_measurement_columns = [
            column for column in queue_label_columns if column in ramp_table.columns
        ]
        assert hidden_ramp_rows.select(
            pl.col(ramp_measurement_columns).is_null().all()
        ).row(0) == (True,) * len(ramp_measurement_columns)
        return {
            selected_panel: {
                split: panel_targets.with_columns(pl.lit(0).alias("queue_pred")).lazy()
                for split, panel_targets in panel_targets_by_split.items()
            }
            for selected_panel, panel_targets_by_split in requested_targets.items()
        }

    monkeypatch.setattr(
        inference, "_load_predictor", lambda _name, _task: predict_queue
    )
    result = inference.run_solution(
        "blank_template_test",
        "queue",
        release_slice,
        targets,
        answers,
    )
    assert solution_received_zero_template
    assert result.metrics["queue_proxy_iou"].item() == 0.0
