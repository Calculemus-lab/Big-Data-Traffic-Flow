"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    QUEUE_HISTORY_MINUTES,
    QUEUE_INTERVAL_MINUTES,
    VALUES,
    HistoricalTaskLabels,
    Panel,
    ReleasePackageSlice,
    Split,
    Task,
)
from trafficbench.data import (
    Release,
    load_release_slice,
)
from trafficbench.fixture import make_fixture
from trafficbench.runner import (
    compare,
    run_experiment,
    run_predictions,
    solution_hash,
)


def saved_run_record(
    benchmark_id: str, panels: list[str], evaluator_hash: str
) -> dict[str, object]:
    """Return a complete saved-run record with selectable comparison fields.

    Args:
        benchmark_id: Benchmark identity written into the record.
        panels: Panels recorded as included in the run.
        evaluator_hash: Scoring-code identity used by comparison checks.

    Returns:
        JSON-compatible run fields representing a completed baseline.
    """
    return {
        "run_id": "saved-run",
        "solution": "baseline",
        "task": "state",
        "params": {},
        "status": "completed",
        "timestamp": "2031-03-01T00:00:00+00:00",
        "benchmark_id": benchmark_id,
        "profile": "quick",
        "seed": 1,
        "panels": panels,
        "commit": None,
        "branch": None,
        "dirty": False,
        "code_hash": "code-hash",
        "evaluator_hash": evaluator_hash,
        "baseline_hash": "baseline-hash",
        "python": "3.12",
        "platform": "test",
        "dependencies": {},
        "warnings": [],
        "metrics": {"state_score": 0.5},
        "seconds": 1.0,
        "peak_memory_mb": 1.0,
    }


@pytest.mark.parametrize("task", ["state", "queue", "odme"])
def test_baselines_end_to_end(
    prepared_benchmark_directory: Path, tmp_path: Path, task: Task
) -> None:
    path = run_experiment(
        "baseline",
        task,
        prepared_benchmark_directory,
        runs_directory=tmp_path,
        compare_baseline=False,
    )
    metadata = json.loads((path / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["metrics"]
    assert list((path / "predictions").rglob("*.parquet"))
    assert (path / "source/solutions/baseline/model.py").exists()
    assert (path / "source/solutions/baseline/__init__.py").exists()


def test_direct_run_scores_only_when_an_answer_key_is_supplied(
    tmp_path: Path,
) -> None:
    release = make_fixture(tmp_path / "release")
    panel: Panel = "D12_I5_N"
    state_answer_key = (
        pl.read_csv(
            release / f"task1/{panel}/train/sample_submission_state.csv",
            try_parse_dates=True,
        )
        .with_columns(pl.lit(40.0).alias("speed_kmh"), pl.lit(100.0).alias("flow_vph"))
        .filter(
            pl.col("timestamp").ge(datetime(2030, 12, 1, tzinfo=UTC))
            & pl.col("timestamp").lt(datetime(2030, 12, 2, tzinfo=UTC))
        )
    )
    answer_key_file = tmp_path / "state_answer_key.csv"
    state_answer_key.write_csv(answer_key_file)
    state_run = run_predictions(
        "baseline",
        "state",
        release_directory=release,
        panels=[panel],
        history_start_date=date(2030, 11, 3),
        history_end_date=date(2030, 12, 1),
        prediction_start_date=date(2030, 12, 1),
        prediction_end_date=date(2030, 12, 2),
        answer_key_file=answer_key_file,
        runs_directory=tmp_path / "runs",
    )
    state_record = json.loads((state_run / "run.json").read_text())
    assert state_record["scored"] is True
    assert state_record["final_submission_eligible"] is False
    assert "state_score" in pl.read_csv(state_run / "metrics.csv").columns

    odme_run = run_predictions(
        "baseline",
        "odme",
        release_directory=release,
        panels=[panel],
        history_start_date=date(2030, 11, 3),
        history_end_date=date(2030, 12, 1),
        prediction_start_date=date(2030, 12, 1),
        prediction_end_date=date(2030, 12, 2),
        runs_directory=tmp_path / "runs",
    )
    odme_record = json.loads((odme_run / "run.json").read_text())
    predictions = pl.read_csv(odme_run / "predictions.csv")
    assert odme_record["scored"] is False
    assert set(predictions["path_flow"]) != {0}


def test_solution_call_batches_selected_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench import inference

    release_path = make_fixture(tmp_path / "release")
    release_slice = load_release_slice(
        Release(release_path),
        date(2030, 11, 3),
        date(2030, 12, 1),
        date(2030, 12, 1),
        date(2030, 12, 2),
        ["D12_I5_N"],
    )
    first_panel: Panel = "D12_I5_N"
    second_panel: Panel = "D7_I10_E"
    first_panel_data = release_slice.panel_slices_by_panel[first_panel]
    release_slice = replace(
        release_slice,
        panel_slices_by_panel={
            first_panel: first_panel_data,
            second_panel: replace(
                first_panel_data,
                family_id="D7_I10",
                historical_task_labels=HistoricalTaskLabels(
                    task1_state_answers=pl.DataFrame(
                        schema={
                            **{column: pl.String for column in KEYS["state"]},
                            **{column: pl.Float64 for column in VALUES["state"]},
                        }
                    ).lazy(),
                    task2_queue_proxy_labels=pl.DataFrame(
                        schema={
                            **{column: pl.String for column in KEYS["queue"]},
                            "timestamp": pl.Datetime("us", "UTC"),
                            "queue_pred": pl.Int64,
                        }
                    ).lazy(),
                ),
            ),
        },
    )
    targets: dict[Panel, dict[Split, pl.DataFrame]] = {
        panel: {
            "train": pl.DataFrame(
                {
                    "window_id": [f"{panel}_window"],
                    "timestamp": [datetime(2030, 12, 1, 0, 5, tzinfo=UTC)],
                    "link_id": ["L01"],
                }
            )
        }
        for panel in (first_panel, second_panel)
    }
    solution_calls = 0

    def predict_all_panels(
        visible_slice: ReleasePackageSlice,
        requested_targets: dict[Panel, dict[Split, table_types.QueueFrame]],
    ) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
        """Return zero-valued predictions for every requested panel and split."""
        nonlocal solution_calls
        solution_calls += 1
        assert set(visible_slice.panel_slices_by_panel) == set(requested_targets)
        return {
            panel: {
                split: panel_targets.with_columns(pl.lit(0).alias("queue_pred"))
                for split, panel_targets in panel_targets_by_split.items()
            }
            for panel, panel_targets_by_split in requested_targets.items()
        }

    monkeypatch.setattr(
        inference, "_load_predictor", lambda _name, _task: predict_all_panels
    )
    run_result = inference.run_solution(
        "batch_test",
        "queue",
        release_slice,
        targets,
    )
    assert solution_calls == 1
    assert set(run_result.predictions) == {first_panel, second_panel}


def test_queue_solution_receives_all_windows_in_one_panel_call(
    prepared_benchmark_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solutions import baseline

    original_queue = baseline.queue
    calls: list[tuple[Panel, str]] = []

    def inspect_queue(
        release_slice: ReleasePackageSlice,
        targets: dict[Panel, dict[Split, table_types.QueueFrame]],
    ) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
        """Check window histories for each panel before delegating."""
        assert len(release_slice.panel_slices_by_panel) == len(targets) == 1
        for panel, panel_targets_by_split in targets.items():
            panel_targets = panel_targets_by_split["train"].collect()
            calls.append((panel, release_slice.prediction_start_date.isoformat()))
            panel_data = release_slice.panel_slices_by_panel[panel]
            queue_history = panel_data.queue_history["train"].collect()
            target_window_ids = panel_targets["window_id"].unique().to_list()
            assert set(panel_targets["window_id"]).issubset(
                set(queue_history["window_id"])
            )
            for window_id in target_window_ids:
                history = queue_history.filter(pl.col("window_id") == window_id)
                window_targets = panel_targets.filter(pl.col("window_id") == window_id)
                forecast_origin = window_targets.select(
                    (
                        pl.col("timestamp").min()
                        - timedelta(minutes=QUEUE_INTERVAL_MINUTES)
                    ).alias("forecast_origin")
                )
                history_bounds = history.select(
                    pl.col("timestamp").min().alias("history_start"),
                    pl.col("timestamp").max().alias("history_end"),
                )
                assert history_bounds.join(forecast_origin, how="cross").select(
                    pl.col("history_start").ge(
                        pl.col("forecast_origin")
                        - timedelta(minutes=QUEUE_HISTORY_MINUTES)
                    ),
                    pl.col("history_end").lt(pl.col("forecast_origin")),
                ).row(0) == (True, True)
            masked_states = panel_data.masked_mainline_states["train"].collect()
            requested_times = panel_targets.select("timestamp").unique()
            masked_targets = masked_states.join(
                requested_times, on="timestamp", how="semi"
            )
            assert masked_targets.select(
                pl.col("speed_kmh").is_null().all(),
                pl.col("flow_vph").is_null().all(),
            ).row(0) == (True, True)
            assert panel_data.link_counts["train"].select(pl.len()).collect().item() > 0
            assert panel_data.weak_prior["train"].select(pl.len()).collect().item() > 0
        return original_queue(release_slice, targets)

    monkeypatch.setattr(baseline, "queue", inspect_queue)
    run_experiment(
        "baseline",
        "queue",
        prepared_benchmark_directory,
        runs_directory=tmp_path,
        compare_baseline=False,
    )
    assert len(calls) == 1


def test_new_creates_all_task_predictors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "solutions").mkdir()
    monkeypatch.setattr(sys, "argv", ["bench", "new", "new_solution"])
    with pytest.raises(SystemExit) as completed:
        main()
    assert completed.value.code == 0
    path = tmp_path / "solutions/new_solution"
    assert (
        (path / "__init__.py")
        .read_text()
        .endswith('__all__ = ["odme", "queue", "state"]\n')
    )
    generated_model = (path / "model.py").read_text()
    assert "release_slice: ReleasePackageSlice" in generated_model
    assert "historical_task_labels: HistoricalTaskLabels" not in generated_model
    assert "StateFrame" in generated_model
    assert "QueueFrame" in generated_model
    assert "OdmeFrame" in generated_model
    assert "return baseline.state(" in generated_model
    assert "return baseline.queue(" in generated_model
    assert "return baseline.odme(" in generated_model
    with pytest.raises(SystemExit):
        main()


def test_solution_hash_includes_helper_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "solutions/baseline"
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("from .model import state\n")
    (path / "model.py").write_text("from .helper import value\n")
    helper = path / "helper.py"
    helper.write_text("value = 1\n")
    before = solution_hash("baseline")
    helper.write_text("value = 2\n")
    assert solution_hash("baseline") != before


def test_failure_is_logged(prepared_benchmark_directory: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must define"):
        run_experiment(
            "example", "queue", prepared_benchmark_directory, runs_directory=tmp_path
        )
    metadata = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert metadata["status"] == "failed"
    assert "must define" in metadata["error"]


def test_compare_rejects_different_benchmarks(tmp_path: Path) -> None:
    paths: list[Path] = []
    for ident in ["quick", "full"]:
        p = tmp_path / ident
        p.mkdir()
        (p / "run.json").write_text(
            json.dumps(saved_run_record(ident, ["D12_I5_N"], "same"))
        )
        paths.append(p)
    with pytest.raises(ValueError, match="same task and benchmark"):
        compare(paths)


def test_comparison_rejects_partial_panel_run(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i, panels in enumerate([["D12_I5_N"], ["D12_I5_N", "D12_I5_S"]]):
        path = tmp_path / str(i)
        path.mkdir()
        (path / "run.json").write_text(
            json.dumps(saved_run_record("same", panels, "same"))
        )
        paths.append(path)
    with pytest.raises(ValueError, match="same panels"):
        compare(paths)
