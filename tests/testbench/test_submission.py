"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trafficbench.contracts import (
    KEYS,
    Task,
)
from trafficbench.submission import COLUMNS, assemble, validate_submission


def submission_files(tmp_path: Path) -> list[Path]:
    """Write one small prediction file per task for assembly tests.

    Args:
        tmp_path: Temporary directory receiving the three task CSVs.

    Returns:
        Paths to the state, queue, and ODME prediction files, in task order.
    """
    state = pl.DataFrame(
        {
            "panel": ["A"],
            "timestamp": ["2031-03-01T00:00:00Z"],
            "station_id": ["001"],
            "link_id": ["L"],
            "mask_regime": ["R1"],
            "speed_kmh": [80.0],
            "flow_vph": [2000.0],
        }
    )
    queue = pl.DataFrame(
        {
            "window_id": ["w"],
            "timestamp": ["2031-03-01T00:05:00Z"],
            "link_id": ["L"],
            "queue_pred": [1],
        }
    )
    odme = pl.DataFrame(
        {
            "panel": ["A"],
            "departure_time": ["PM"],
            "path_id": ["P"],
            "origin_zone": ["O"],
            "destination_zone": ["D"],
            "path_flow": [25.0],
        }
    )
    for task, frame in [("state", state), ("queue", queue), ("odme", odme)]:
        frame.write_csv(tmp_path / f"{task}.csv")
    task_tables: list[tuple[Task, pl.DataFrame, str]] = [
        ("state", state, "1"),
        ("queue", queue, "2"),
        ("odme", odme, "3"),
    ]
    key_tables = [
        table.select(KEYS[task]).with_columns(
            pl.lit(task).alias("task"),
            pl.lit(submission_id).alias("submission_id"),
        )
        for task, table, submission_id in task_tables
    ]
    key = pl.concat(key_tables, how="diagonal_relaxed").fill_null("")
    key.write_csv(tmp_path / "key.csv")
    template = key.select("submission_id", "task").with_columns(
        pl.lit(0).alias(column) for column in COLUMNS[2:]
    )
    template.select(COLUMNS).write_csv(tmp_path / "template.csv")
    return [
        tmp_path / f"{name}.csv"
        for name in ["state", "queue", "odme", "key", "template", "output"]
    ]


def replace_test_row_value(
    table: pl.DataFrame, row_index: int, column_name: str, value: str | float
) -> pl.DataFrame:
    """Return a test table with one value changed at the selected row."""
    return (
        table.with_row_index("_test_row")
        .with_columns(
            pl.when(pl.col("_test_row") == row_index)
            .then(pl.lit(value))
            .otherwise(pl.col(column_name))
            .alias(column_name)
        )
        .drop("_test_row")
    )


def test_submission_assembly(tmp_path: Path) -> None:
    args = submission_files(tmp_path)
    assemble(*args)
    assert validate_submission(args[-1], args[-2]) == 3
    frame = pl.read_csv(args[-1])
    assert frame["flow_vph"].to_list() == [2000, 0, 0]
    assert frame["queue_pred"].to_list() == [0, 1, 0]
    assert frame["path_flow"].to_list() == [0, 0, 25]


def test_submission_streams_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench import submission

    monkeypatch.setattr(submission, "CHUNK", 1)
    args = submission_files(tmp_path)
    assemble(*args)
    assert validate_submission(args[-1], args[-2]) == 3


def test_duplicate_predictions_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trafficbench import submission

    monkeypatch.setattr(submission, "CHUNK", 1)
    args = submission_files(tmp_path)
    frame = pl.read_csv(args[2])
    pl.concat([frame, frame]).write_csv(args[2])
    with pytest.raises(ValueError, match="Duplicate odme"):
        assemble(*args)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "zones"])
def test_assembly_rejects_incomplete_or_wrong_keys(tmp_path: Path, damage: str) -> None:
    args = submission_files(tmp_path)
    table = pl.read_csv(args[2])
    match damage:
        case "missing":
            table = table.head(0)
        case "extra":
            table = pl.concat(
                [table, table.with_columns(pl.lit("extra").alias("path_id"))]
            )
        case "duplicate":
            table = pl.concat([table, table])
        case "zones":
            table = table.with_columns(pl.lit("wrong").alias("destination_zone"))
    table.write_csv(args[2])
    with pytest.raises(ValueError):
        assemble(*args)
    assert not Path(args[-1]).exists()


@pytest.mark.parametrize(
    "damage",
    ["id", "nonfinite", "unused", "binary", "negative", "missing", "duplicate"],
)
def test_final_csv_rejects_invalid_values(tmp_path: Path, damage: str) -> None:
    args = submission_files(tmp_path)
    assemble(*args)
    frame = pl.read_csv(args[-1])
    match damage:
        case "id":
            frame = replace_test_row_value(frame, 0, "submission_id", 99)
        case "nonfinite":
            frame = replace_test_row_value(frame, 0, "speed_kmh", np.inf)
        case "unused":
            frame = replace_test_row_value(frame, 0, "path_flow", 1)
        case "binary":
            frame = replace_test_row_value(frame, 1, "queue_pred", 0.5)
        case "negative":
            frame = replace_test_row_value(frame, 2, "path_flow", -1)
        case "missing":
            frame = frame.head(2)
        case "duplicate":
            frame = replace_test_row_value(frame, 1, "submission_id", 1)
    frame.write_csv(args[-1])
    with pytest.raises(ValueError):
        validate_submission(args[-1], args[-2])
