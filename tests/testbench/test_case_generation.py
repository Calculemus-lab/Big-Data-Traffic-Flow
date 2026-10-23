"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    QueueConfig,
)
from trafficbench.data import Release
from trafficbench.fixture import make_fixture
from trafficbench.prepare import build_queue_proxy_windows, prepare
from trafficbench.runner import (
    verify_benchmark,
)


def test_prepared_inputs_and_truth_are_separate(
    prepared_benchmark_directory: Path,
) -> None:
    manifest = verify_benchmark(prepared_benchmark_directory)
    for benchmark_case in manifest.cases:
        case_directory = prepared_benchmark_directory / benchmark_case.case_directory
        targets = pl.read_parquet(case_directory / "targets.parquet")
        assert set(targets.columns) == set(KEYS[benchmark_case.task])
        panel_case_directory = (
            prepared_benchmark_directory / benchmark_case.panel / benchmark_case.case_id
        )
        masked_states = pl.read_parquet(
            panel_case_directory / "masked_mainline_states.parquet"
        )
        if benchmark_case.task == "state" or benchmark_case.task == "queue":
            historical_label_table = pl.read_parquet(
                prepared_benchmark_directory / benchmark_case.historical_solutions_path
            )
            if not historical_label_table.is_empty():
                assert historical_label_table.select(
                    pl.col("timestamp").min()
                    >= datetime.combine(
                        benchmark_case.history_start_date, datetime.min.time(), UTC
                    )
                ).item()
                assert historical_label_table.select(
                    pl.col("timestamp").max()
                    < datetime.combine(
                        benchmark_case.history_end_date, datetime.min.time(), UTC
                    )
                ).item()
            assert masked_states.select(
                pl.col("timestamp").min()
                >= datetime.combine(
                    benchmark_case.history_start_date, datetime.min.time(), UTC
                )
            ).item()
        if benchmark_case.task == "state":
            truth = pl.read_parquet(case_directory / "truth.parquet")
            assert truth.height == targets.height
            assert truth.select(
                pl.col("timestamp")
                .ge(
                    datetime.combine(
                        benchmark_case.history_end_date, datetime.min.time(), UTC
                    )
                )
                .all()
            ).item()
            requested_state_rows = targets.join(
                masked_states, on=["timestamp", "station_id", "link_id"]
            )
            assert requested_state_rows.select(
                pl.col("speed_kmh").is_null().all(),
                pl.col("flow_vph").is_null().all(),
            ).row(0) == (True, True)
        if benchmark_case.task == "queue":
            history = pl.read_parquet(case_directory / "observations.parquet")
            assert set(history.columns) == set(
                table_types.QueueHistoryColumns.__annotations__
            )
            for window in benchmark_case.windows:
                forecast_origin_utc = window.forecast_origin_utc
                window_history = history.filter(pl.col("window_id") == window.window_id)
                window_targets = targets.filter(pl.col("window_id") == window.window_id)
                assert window_history.select(
                    pl.col("timestamp").max().lt(forecast_origin_utc)
                ).item()
                assert window_history.select(
                    pl.col("timestamp")
                    .min()
                    .ge(forecast_origin_utc - timedelta(minutes=60))
                ).item()
                assert window_targets.select(
                    pl.col("timestamp")
                    .min()
                    .eq(forecast_origin_utc + timedelta(minutes=5))
                ).item()
                assert window_targets.select(
                    pl.col("timestamp")
                    .max()
                    .eq(forecast_origin_utc + timedelta(minutes=30))
                ).item()
                assert window_targets["timestamp"].n_unique() == 6
                assert window_targets.select(
                    pl.col("timestamp")
                    .max()
                    .lt(
                        datetime.combine(
                            benchmark_case.prediction_end_date,
                            datetime.min.time(),
                            UTC,
                        )
                    )
                ).item()


def test_task4_synthetic_cases_are_repeatable(tmp_path: Path) -> None:
    release_directory = make_fixture(tmp_path / "release")
    preparation_dates = {
        "history_start_date": date(2030, 11, 3),
        "history_end_date": date(2030, 11, 20),
        "prediction_end_date": date(2030, 11, 28),
        "panels": ["D12_I5_N"],
    }
    first_benchmark = prepare(
        release_directory, tmp_path / "first", **preparation_dates
    )
    second_benchmark = prepare(
        release_directory, tmp_path / "second", **preparation_dates
    )
    first_manifest = verify_benchmark(first_benchmark)
    second_manifest = verify_benchmark(second_benchmark)
    first_cases = {
        case.scenario_name: case
        for case in first_manifest.cases
        if case.task == "odme" and case.scenario_name.startswith("synthetic_")
    }
    second_cases = {
        case.scenario_name: case
        for case in second_manifest.cases
        if case.task == "odme" and case.scenario_name.startswith("synthetic_")
    }
    assert first_cases.keys() == second_cases.keys()
    for scenario_name, first_case in first_cases.items():
        first_directory = first_benchmark / first_case.case_directory
        second_directory = second_benchmark / second_cases[scenario_name].case_directory
        assert pl.read_parquet(first_directory / "counts.parquet").equals(
            pl.read_parquet(second_directory / "counts.parquet")
        )
        assert pl.read_parquet(first_directory / "truth.parquet").equals(
            pl.read_parquet(second_directory / "truth.parquet")
        )


def test_selector_matched_onsets_exclude_earlier_future_queues(
    prepared_benchmark_directory: Path,
) -> None:
    """Keep the earliest onset horizon without changing ongoing eligibility."""
    network = Release(
        prepared_benchmark_directory.parent / "release"
    ).scan_network_tables("D12_I5_N")
    start = datetime(2030, 11, 20, tzinfo=UTC)
    timestamps = [start + timedelta(minutes=5 * step) for step in range(60)]
    links = network.links.select("link_id", "free_speed_kmh").collect()
    # One sustained queue gives six possible forecast onsets but only one
    # origin whose first queued target is exactly thirty minutes ahead.
    states = (
        pl.DataFrame({"timestamp": timestamps})
        .join(links, how="cross")
        .with_columns(
            pl.when(
                pl.col("timestamp").is_between(timestamps[30], timestamps[35])
                & pl.col("link_id").eq(links["link_id"][0])
            )
            .then(pl.col("free_speed_kmh") * 0.4)
            .otherwise(pl.col("free_speed_kmh"))
            .alias("speed_kmh"),
            pl.lit(1000.0).alias("flow_vph"),
            pl.lit(0.1).alias("occupancy"),
            pl.lit(100).alias("pct_observed"),
            pl.lit(1).alias("is_score_eligible"),
        )
    )
    settings = QueueConfig(
        windows_per_condition=100,
        minimum_spacing_minutes=0,
        minimum_history_coverage=1.0,
        minimum_future_coverage=1.0,
        excluded_panels=[],
    )
    arbitrary = build_queue_proxy_windows(
        states,
        "D12_I5_N",
        "I5",
        "test",
        network,
        settings,
        31,
        start.date(),
        start.date(),
        date(2030, 11, 21),
    )
    matched = build_queue_proxy_windows(
        states,
        "D12_I5_N",
        "I5",
        "test",
        network,
        settings.model_copy(update={"onset_at_final_horizon": True}),
        31,
        start.date(),
        start.date(),
        date(2030, 11, 21),
    )
    onsets = [window for window in matched if window.info.condition == "queue_onset"]
    assert len(onsets) == 1
    assert onsets[0].info.forecast_origin_utc == timestamps[24]
    assert any(
        window.info.condition == "queue_onset"
        and not window.answer.filter(
            pl.col("queue_true").eq(1)
            & pl.col("timestamp").lt(
                window.info.forecast_origin_utc + timedelta(minutes=30)
            )
        ).is_empty()
        for window in arbitrary
    )
    assert {
        window.info.forecast_origin_utc
        for window in matched
        if window.info.condition == "queue_ongoing"
    } == {
        window.info.forecast_origin_utc
        for window in arbitrary
        if window.info.condition == "queue_ongoing"
    }
