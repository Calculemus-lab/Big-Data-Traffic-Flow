"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import pytest

from trafficbench.contracts import (
    KEYS,
    Panel,
    Split,
)
from trafficbench.data import (
    Release,
    load_historical_task_labels,
    load_release_slice,
    load_target_templates,
    select_queue_solutions,
)
from trafficbench.fixture import make_fixture


def add_split_inputs(
    release_directory: Path,
    panel: Panel,
    split: Split,
    target_dates: list[date],
) -> None:
    """Copy fixture input rows onto selected calendar dates in another split."""
    panel_directory = release_directory / "corridors" / panel
    train_directory = panel_directory / "train"
    for table_directory_name in ("mainline_states_masked", "ramp_states"):
        train_table_directory = train_directory / table_directory_name
        source_table_path = next(train_table_directory.rglob("*.parquet"))
        source_table = pl.read_parquet(source_table_path)
        source_file_date = "_".join(source_table_path.stem.rsplit("_", 3)[-3:])
        source_date = date.fromisoformat(source_file_date.replace("_", "-"))
        for target_date in target_dates:
            day_offset = target_date - source_date
            shifted_table = source_table.with_columns(
                (pl.col("timestamp") + pl.duration(days=day_offset.days)).alias(
                    "timestamp"
                ),
                pl.lit(target_date.isoformat()).alias("date"),
            )
            source_relative_path = source_table_path.relative_to(train_table_directory)
            destination_relative_parts = [
                (
                    f"year_month={target_date:%Y-%m}"
                    if path_part.startswith("year_month=")
                    else path_part
                )
                for path_part in source_relative_path.parts[:-1]
            ]
            target_file_name = re.sub(
                r"20\d{2}_\d{2}_\d{2}(?=\.parquet$)",
                target_date.strftime("%Y_%m_%d"),
                source_table_path.name,
            )
            destination_relative_path = Path(
                *destination_relative_parts, target_file_name
            )
            destination_file_path = (
                panel_directory
                / split
                / table_directory_name
                / destination_relative_path
            )
            destination_file_path.parent.mkdir(parents=True, exist_ok=True)
            shifted_table.write_parquet(destination_file_path)

    state_template = pl.read_csv(
        release_directory / f"task1/{panel}/train/sample_submission_state.csv"
    ).head(1)
    for target_date in target_dates:
        shifted_targets = state_template.with_columns(
            pl.lit(datetime.combine(target_date, datetime.min.time(), UTC)).alias(
                "timestamp"
            )
        )
        target_file_path = (
            release_directory / "task1" / panel / split / "sample_submission_state.csv"
        )
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        include_header = not target_file_path.exists()
        with target_file_path.open("ab" if not include_header else "wb") as output_file:
            shifted_targets.write_csv(output_file, include_header=include_header)


def test_cutoff_slice_and_cross_split_end_exclusive_selection(
    tmp_path: Path,
) -> None:
    release_directory = make_fixture(tmp_path / "release")
    panel: Panel = "D12_I5_N"
    validation_date = date(2031, 3, 31)
    private_dates = [date(2031, 4, 1), date(2031, 4, 2)]
    add_split_inputs(release_directory, panel, "validation", [validation_date])
    add_split_inputs(release_directory, panel, "private", private_dates)
    release = Release(release_directory)

    # The train/validation and validation/private boundaries come from the release calendar.
    release_slice = load_release_slice(
        release,
        date(2030, 11, 3),
        date(2030, 11, 20),
        validation_date,
        date(2031, 4, 2),
        [panel],
    )
    panel_slice = release_slice.panel_slices_by_panel[panel]
    assert not hasattr(panel_slice, "mainline_history")
    assert set(panel_slice.masked_mainline_states) == {
        "train",
        "validation",
        "private",
    }
    train_states = panel_slice.masked_mainline_states["train"].collect()
    private_states = panel_slice.masked_mainline_states["private"].collect()
    assert train_states.select(
        pl.col("timestamp").min()
        >= datetime.combine(release_slice.history_start_date, datetime.min.time(), UTC)
    ).item()
    assert private_states.select(
        pl.col("timestamp").max() < datetime(2031, 4, 2, tzinfo=UTC)
    ).item()
    assert isinstance(panel_slice.masked_mainline_states["train"], pl.LazyFrame)

    # The target range crosses the split boundary and excludes its end date.
    targets_by_split = load_target_templates(
        release,
        "state",
        validation_date,
        date(2031, 4, 2),
        [panel],
    )
    assert set(targets_by_split[panel]) == {"validation", "private"}
    assert targets_by_split[panel]["validation"]["timestamp"].dt.date().to_list() == [
        validation_date
    ]
    assert targets_by_split[panel]["private"]["timestamp"].dt.date().to_list() == [
        date(2031, 4, 1)
    ]


def test_release_reader_requires_extracted_directory(tmp_path: Path) -> None:
    release_directory = make_fixture(tmp_path / "release")
    archive_path = tmp_path / "release.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as release_archive:
        for file_path in release_directory.rglob("*"):
            if file_path.is_file():
                release_archive.write(
                    file_path, file_path.relative_to(release_directory)
                )

    with pytest.raises(ValueError, match="extracted release directory"):
        Release(archive_path)


def test_historical_task_labels_cover_the_history_before_cutoff(
    tmp_path: Path,
) -> None:
    release = Release(make_fixture(tmp_path / "release"))
    panel: Panel = "D12_I5_N"
    history_start_date = date(2030, 11, 3)
    history_end_date = date(2030, 11, 20)

    historical_task_labels = load_historical_task_labels(
        release, history_start_date, history_end_date, [panel]
    )
    state_solutions = historical_task_labels[panel].task1_state_answers.collect()
    queue_solutions = historical_task_labels[panel].task2_queue_proxy_labels.collect()
    assert set(state_solutions.columns) == {*KEYS["state"], "speed_kmh", "flow_vph"}
    assert state_solutions.select(
        pl.col("timestamp").min()
        >= datetime.combine(history_start_date, datetime.min.time(), UTC)
    ).item()
    assert state_solutions.select(
        pl.col("timestamp").max()
        < datetime.combine(history_end_date, datetime.min.time(), UTC)
    ).item()
    assert state_solutions.select(
        pl.col("speed_kmh").is_not_null().all() & pl.col("flow_vph").is_not_null().all()
    ).item()

    assert set(queue_solutions.columns) == {*KEYS["queue"], "queue_pred"}
    assert queue_solutions.is_empty()


def test_historical_queue_proxy_labels_require_eligible_measurements(
    tmp_path: Path,
) -> None:
    release = Release(make_fixture(tmp_path / "release"))
    panel: Panel = "D12_I5_N"
    timestamp = datetime(2030, 11, 3, 8, tzinfo=UTC)
    unmasked_states = release.read_state_rows(
        panel,
        "train",
        "mainline_states",
        timestamp.date(),
        date(2030, 11, 4),
    )
    unmasked_states = unmasked_states.with_columns(
        pl.when(pl.col("link_id") == "L02")
        .then(pl.lit(50))
        .otherwise(pl.col("pct_observed"))
        .alias("pct_observed")
    )
    queue_template = pl.DataFrame(
        {
            "window_id": ["local_proxy", "local_proxy"],
            "timestamp": [timestamp, timestamp],
            "link_id": ["L01", "L02"],
            "queue_pred": [0, 0],
        }
    )

    queue_solutions = select_queue_solutions(
        queue_template.lazy(),
        unmasked_states.lazy(),
        release.scan_network_tables(panel),
    ).collect()

    assert queue_solutions.select(KEYS["queue"]).to_dicts() == [
        {
            "window_id": "local_proxy",
            "timestamp": timestamp,
            "link_id": "L01",
        }
    ]
    assert queue_solutions["queue_pred"].to_list() == [1]
