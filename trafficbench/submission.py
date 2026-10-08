"""Assemble and validate the single CSV uploaded for all prediction tasks.

Each task prediction file uses its task-specific identifier columns. The
official submission key maps those identifiers to submission IDs and fixes row
order. The upload template defines the final columns. SQLite keeps joins and
uniqueness checks on disk when files are too large to load together.
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl
from pydantic import FilePath

from .contracts import KEYS, VALUES, Task, normalize_lazy, validate_predictions

COLUMNS = ["submission_id", "task", *VALUES["state"], *VALUES["queue"], *VALUES["odme"]]
CHUNK = 100_000
logger = logging.getLogger(__name__)


def table_file_chunks(table_file_path: FilePath) -> Iterator[pl.DataFrame]:
    """Yield rows from a CSV or Parquet file in bounded DataFrame batches.

    Args:
        table_file_path: CSV or Parquet table to read.

    Yields:
        Consecutive batches of rows, each limited to the configured chunk size.
    """
    table_scan = (
        pl.scan_parquet(table_file_path)
        if table_file_path.suffix == ".parquet"
        else pl.scan_csv(table_file_path, infer_schema=False, low_memory=True)
    )
    yield from table_scan.collect_batches(chunk_size=CHUNK)


def canonical_keys(
    input_table: pl.DataFrame, key_columns: Sequence[str]
) -> pl.DataFrame:
    """Return selected task identifier values as text for exact matching.

    Parse timestamps as UTC and format them identically for CSV and Parquet
    inputs. Other identifiers retain their spelling, including leading zeros.

    Args:
        input_table: Table containing the task identifier fields.
        key_columns: Identifier columns to return in their existing order.

    Returns:
        A copy of the selected identifier columns with normalized text values.
    """
    normalized_keys = normalize_lazy(input_table.select(key_columns).lazy()).collect()
    return normalized_keys.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if column == "timestamp"
        else pl.col(column).cast(pl.String)
        for column in key_columns
    )


def validate_submission(
    submission_file: FilePath, sample_submission_file: FilePath
) -> int:
    """Check a final upload CSV against its official row template.

    Both files must have the same columns, row count, submission IDs, task
    labels, and order. Each ID must be unique globally.
    Every value must be finite. Queue values are binary, path flows are
    nonnegative, and columns unused by a row's task contain zero. Read both
    files in bounded chunks and return the accepted row count.

    Args:
        submission_file: Candidate combined upload CSV.
        sample_submission_file: Official template defining exact rows and order.

    Returns:
        Number of validated submission rows.

    Raises:
        ValueError: If rows, columns, identifiers, or prediction values differ
            from the official template contract.
    """
    row_count = 0
    # SQLite keeps the global uniqueness check bounded across CSV chunks.
    with tempfile.TemporaryDirectory(prefix="trafficbench-ids-") as temporary_directory:
        database = sqlite3.connect(Path(temporary_directory) / "ids.sqlite")
        try:
            database.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            for submission_chunk, template_chunk in itertools.zip_longest(
                table_file_chunks(submission_file),
                table_file_chunks(sample_submission_file),
            ):
                # A CSV with the right IDs in a different order is still an
                # invalid upload, because the template fixes row identity.
                if (
                    submission_chunk is None
                    or template_chunk is None
                    or submission_chunk.height != template_chunk.height
                ):
                    raise ValueError("Submission row count differs from template")
                if (
                    list(submission_chunk.columns) != COLUMNS
                    or list(template_chunk.columns) != COLUMNS
                ):
                    raise ValueError(
                        f"Submission must have exactly these columns, in order: {COLUMNS}"
                    )
                if not submission_chunk.select("submission_id", "task").equals(
                    template_chunk.select("submission_id", "task")
                ):
                    raise ValueError(
                        "Submission IDs, order or task assignments differ from template"
                    )
                if submission_chunk["submission_id"].eq("").any() or not (
                    submission_chunk["task"].is_in(list(VALUES)).all()
                ):
                    raise ValueError("Empty ID or unknown task")
                try:
                    database.executemany(
                        "INSERT INTO ids VALUES (?)",
                        [
                            (str(submission_id),)
                            for submission_id in submission_chunk["submission_id"]
                        ],
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("Duplicate submission IDs") from error
                # Parse every value as a finite number. Unused task columns
                # must be zero rather than blank because every cell is required.
                numeric_columns = submission_chunk.select(
                    pl.col(COLUMNS[2:]).cast(pl.Float64, strict=True)
                )
                if not numeric_columns.select(
                    pl.all_horizontal(
                        pl.col(COLUMNS[2:]).is_finite().fill_null(False)
                    ).all()
                ).item():
                    raise ValueError("Submission contains empty or nonfinite values")
                queue_values = numeric_columns.filter(
                    submission_chunk["task"].eq("queue")
                )["queue_pred"]
                if not queue_values.is_in([0, 1]).all():
                    raise ValueError("queue_pred must be binary")
                if (
                    numeric_columns.filter(submission_chunk["task"].eq("odme"))[
                        "path_flow"
                    ]
                    < 0
                ).any():
                    raise ValueError("path_flow must be nonnegative")
                for task, values in VALUES.items():
                    unused_columns = [
                        column for column in COLUMNS[2:] if column not in values
                    ]
                    task_values = numeric_columns.filter(
                        submission_chunk["task"].eq(task)
                    ).select(unused_columns)
                    if task_values.select(
                        pl.any_horizontal(pl.col(unused_columns).ne(0)).any()
                    ).item():
                        raise ValueError(f"{task}: unused output columns must be zero")
                row_count += submission_chunk.height
        finally:
            database.close()
    if not row_count:
        raise ValueError("Empty submission")
    logger.info("Valid submission: %s rows", f"{row_count:,}")
    return row_count


def assemble(
    state_predictions_file: FilePath,
    queue_predictions_file: FilePath,
    odme_predictions_file: FilePath,
    submission_key_file: FilePath,
    sample_submission_template_file: FilePath,
    submission_output_file: Path,
) -> None:
    """Join complete task predictions into one validated upload CSV.

    Each task prediction file contains its task's identifier columns and
    predicted values. ``submission_key_file`` maps those identifiers to
    submission IDs and defines row order. ``sample_submission_template_file``
    defines the final ID, task, and value columns. Reject missing, duplicate,
    or unused task rows. Write to a temporary file and publish the result only
    after it passes the final template check.

    Args:
        state_predictions_file: Task 1 rows with ``panel``, ``timestamp``,
            ``station_id``, ``link_id``, ``mask_regime``, ``speed_kmh``, and
            ``flow_vph``.
        queue_predictions_file: Task 2 rows with ``window_id``, ``timestamp``,
            ``link_id``, and ``queue_pred``.
        odme_predictions_file: Task 4 rows with ``panel``, ``departure_time``,
            ``path_id``, ``origin_zone``, ``destination_zone``, and ``path_flow``.
        submission_key_file: Official task identifiers mapped to submission IDs.
        sample_submission_template_file: Official output columns and row order.
        submission_output_file: New destination for the assembled upload CSV.

    Raises:
        ValueError: If any task predictions are incomplete, duplicated, invalid,
            or absent from the official submission key.
    """
    if submission_output_file.exists():
        raise ValueError(
            f"{submission_output_file} already exists. Choose a new output filename"
        )
    submission_output_file.parent.mkdir(parents=True, exist_ok=True)
    # Join by task row identifiers on disk so the complete submission need not fit
    # in memory and a failed validation leaves no published output.
    with tempfile.TemporaryDirectory(
        prefix="trafficbench-assemble-", dir=submission_output_file.parent
    ) as temporary_directory:
        database = sqlite3.connect(Path(temporary_directory) / "predictions.sqlite")
        staged_submission = Path(temporary_directory) / "submission.csv"
        try:
            task_prediction_paths: dict[Task, FilePath] = {
                "state": state_predictions_file,
                "queue": queue_predictions_file,
                "odme": odme_predictions_file,
            }
            for task, prediction_path in task_prediction_paths.items():
                # Table names and columns come from the fixed task contract,
                # never from prediction file contents. The composite primary
                # key catches duplicates across separate input chunks.
                keys, values = KEYS[task], VALUES[task]
                definitions = [
                    f'"{key_column}" TEXT NOT NULL' for key_column in keys
                ] + [f'"{value_column}" REAL NOT NULL' for value_column in values]
                columns = ", ".join(definitions)
                primary_key = ", ".join(keys)
                database.execute(
                    f"CREATE TABLE {task} ({columns}, used INTEGER DEFAULT 0, "
                    f"PRIMARY KEY ({primary_key}))"
                )
                for prediction_chunk in table_file_chunks(prediction_path):
                    # Using the batch as its own target checks task columns,
                    # duplicate keys within the batch, and value constraints.
                    validated_predictions = validate_predictions(
                        task,
                        prediction_chunk.lazy(),
                        prediction_chunk.lazy(),
                    )
                    canonical_predictions = canonical_keys(
                        validated_predictions, keys
                    ).hstack(validated_predictions.select(values))
                    column_names = ", ".join(keys + values)
                    placeholders = ", ".join("?" for _ in keys + values)
                    try:
                        database.executemany(
                            f"INSERT INTO {task} ({column_names}) VALUES ({placeholders})",
                            canonical_predictions.iter_rows(),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError(
                            f"Duplicate {task} prediction keys across chunks"
                        ) from error
            # Traverse the master key in its original order and fill only the
            # value columns belonging to each task.
            first_chunk = True
            for key_chunk in table_file_chunks(submission_key_file):
                if not key_chunk["task"].is_in(list(VALUES)).all():
                    raise ValueError("Unknown task in submission key")
                submission_columns: dict[str, list[str | int | float]] = {
                    "submission_id": key_chunk["submission_id"].to_list(),
                    "task": key_chunk["task"].to_list(),
                    **{column: [0.0] * key_chunk.height for column in COLUMNS[2:]},
                }
                task_values_by_row = key_chunk["task"].to_list()
                for task, keys in KEYS.items():
                    task_row_positions = [
                        row_position
                        for row_position, row_task in enumerate(task_values_by_row)
                        if row_task == task
                    ]
                    if not task_row_positions:
                        continue
                    # Store requested identifiers with their original positions.
                    # SQL join order does not guarantee the original CSV order.
                    requested_keys = canonical_keys(
                        key_chunk.filter(pl.col("task") == task), keys
                    )
                    database.execute("DROP TABLE IF EXISTS wanted")
                    wanted_columns = ", ".join(f'"{key}" TEXT NOT NULL' for key in keys)
                    database.execute(
                        f"CREATE TABLE wanted ({wanted_columns}, position INTEGER NOT NULL)"
                    )
                    columns_with_position = (*keys, "position")
                    insert_columns = ", ".join(columns_with_position)
                    placeholders = ", ".join("?" for _ in columns_with_position)
                    wanted_rows = (
                        (*row, row_position)
                        for row_position, row in zip(
                            task_row_positions, requested_keys.iter_rows(), strict=True
                        )
                    )
                    database.executemany(
                        f"INSERT INTO wanted ({insert_columns}) VALUES ({placeholders})",
                        wanted_rows,
                    )
                    join_condition = " AND ".join(
                        f'prediction."{key_column}" = requested."{key_column}"'
                        for key_column in keys
                    )
                    value_columns = ", ".join(
                        f'prediction."{value}"' for value in VALUES[task]
                    )
                    matched_rows = database.execute(
                        f"SELECT prediction.rowid, {value_columns} "
                        f"FROM wanted AS requested LEFT JOIN {task} AS prediction "
                        f"ON {join_condition} ORDER BY requested.position"
                    ).fetchall()
                    if any(any(value is None for value in row) for row in matched_rows):
                        raise ValueError(
                            f"Missing {task} predictions, or incorrect row identifiers/zones"
                        )
                    # Mark matched predictions so the final pass can reject
                    # rows supplied by a task file but absent from the key.
                    for row_position, matched_row in zip(
                        task_row_positions, matched_rows, strict=True
                    ):
                        for value_column, value in zip(
                            VALUES[task], matched_row[1:], strict=True
                        ):
                            submission_columns[value_column][row_position] = value
                    database.executemany(
                        f"UPDATE {task} SET used=1 WHERE rowid=?",
                        [(row[0],) for row in matched_rows],
                    )
                with staged_submission.open(
                    "w" if first_chunk else "a", newline="", encoding="utf-8"
                ) as submission_stream:
                    pl.DataFrame(submission_columns).select(COLUMNS).write_csv(
                        submission_stream, include_header=first_chunk
                    )
                first_chunk = False
            # Reject predictions omitted by the master key, then atomically
            # publish only a CSV that passes the exact template checks.
            for task in VALUES:
                if database.execute(
                    f"SELECT COUNT(*) FROM {task} WHERE used=0"
                ).fetchone()[0]:
                    raise ValueError(
                        f"Extra {task} predictions not present in submission key"
                    )
            validate_submission(staged_submission, sample_submission_template_file)
            staged_submission.replace(submission_output_file)
        finally:
            database.close()
    logger.info("Wrote submission: %s", submission_output_file)
