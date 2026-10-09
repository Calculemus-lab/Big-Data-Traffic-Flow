"""Run task solutions with selected release data and zero-filled targets."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import NamedTuple, cast

import polars as pl

from .contracts import (
    KEYS,
    MASKED_STATE_COLUMNS,
    VALUES,
    JsonObject,
    Panel,
    PanelReleaseSlice,
    PanelSplitTableMap,
    ReleasePackageSlice,
    Split,
    Task,
    validate_predictions,
)
from .metrics import odme_metrics, physics_diagnostics, queue_metrics, state_metrics

QUEUE_LABEL_SOURCE_COLUMNS = (
    *MASKED_STATE_COLUMNS,
    "pct_observed",
    "is_observed",
    "is_imputed",
    "is_score_eligible",
    "is_missing",
)


class SliceRunResult(NamedTuple):
    """Validated outputs by panel and split, with any available metrics.

    Attributes:
        predictions: Task tables indexed by panel and then release split.
        metrics: Score rows and local diagnostics produced for this run.
    """

    predictions: PanelSplitTableMap[pl.DataFrame]
    metrics: pl.DataFrame


def _redact_target_period(
    release_slice: ReleasePackageSlice,
    task: Task,
    requested_target_keys_by_panel_and_split: Mapping[
        Panel, Mapping[Split, pl.DataFrame]
    ],
) -> ReleasePackageSlice:
    """Hide measurements that directly reveal requested local target labels.

    State targets are masked at their station, link, and timestamp. Queue
    labels use future speed and coverage measurements, so those fields at
    forecast timestamps are cleared from every released mainline, ramp, and
    queue-history table before the solution is called.

    Args:
        release_slice: Public and historical data prepared for this run.
        task: Target task whose source measurements must be hidden.
        requested_target_keys_by_panel_and_split: Exact target rows to predict.

    Returns:
        A copied slice with direct target measurements removed.
    """
    # Join the requested keys to lazy scans and clear only measurements that
    # reveal the labels. The source files remain untouched and uncollected.
    panel_updates: dict[Panel, PanelReleaseSlice] = {}
    for panel, panel_data in release_slice.panel_slices_by_panel.items():
        masked_states_by_split = dict(panel_data.masked_mainline_states)
        ramps_by_split = dict(panel_data.ramp_states)
        queue_histories_by_split = dict(panel_data.queue_history)

        for split, target_keys in requested_target_keys_by_panel_and_split.get(
            panel, {}
        ).items():
            match task:
                case "state":
                    # State answers are detector-specific, so remove only the
                    # values at the requested timestamp, station, and link.
                    target_join_columns = ["timestamp", "station_id", "link_id"]
                    target_keys_lazy = (
                        target_keys.select(target_join_columns)
                        .lazy()
                        .unique(target_join_columns)
                        .with_columns(pl.lit(True).alias("_hide_target"))
                    )
                    masked_states_by_split[split] = _hide_matching_rows(
                        masked_states_by_split[split],
                        target_keys_lazy,
                        target_join_columns,
                        MASKED_STATE_COLUMNS,
                    )
                case "queue":
                    # Queue labels depend on future speed and coverage. Hide
                    # those fields in every table that can expose the same time.
                    forecast_timestamps = (
                        target_keys.select("timestamp")
                        .unique()
                        .lazy()
                        .with_columns(pl.lit(True).alias("_hide_target"))
                    )
                    for selected_split, state_frame in masked_states_by_split.items():
                        masked_states_by_split[selected_split] = _hide_matching_rows(
                            state_frame,
                            forecast_timestamps,
                            ["timestamp"],
                            QUEUE_LABEL_SOURCE_COLUMNS,
                        )
                    for selected_split, ramp_frame in ramps_by_split.items():
                        ramps_by_split[selected_split] = _hide_matching_rows(
                            ramp_frame,
                            forecast_timestamps,
                            ["timestamp"],
                            QUEUE_LABEL_SOURCE_COLUMNS,
                        )
                    for selected_split, queue_frame in panel_data.queue_history.items():
                        # A later window can expose a target timestamp as its
                        # past history, so redact that second route to the data.
                        queue_histories_by_split[selected_split] = _hide_matching_rows(
                            queue_frame,
                            forecast_timestamps,
                            ["timestamp"],
                            QUEUE_LABEL_SOURCE_COLUMNS,
                        )
                case "odme":
                    pass

        panel_updates[panel] = replace(
            panel_data,
            masked_mainline_states=masked_states_by_split,
            ramp_states=ramps_by_split,
            queue_history=queue_histories_by_split,
        )
    return replace(release_slice, panel_slices_by_panel=panel_updates)


def _hide_matching_rows(
    table: pl.LazyFrame,
    target_keys: pl.LazyFrame,
    join_columns: list[str],
    hidden_columns: tuple[str, ...],
) -> pl.LazyFrame:
    """Replace selected values with null on rows matching target keys.

    Args:
        table: Lazy table that may contain measurements used to form answers.
        target_keys: Rows whose measurements must not be visible to a solution.
        join_columns: Columns that identify matching rows in both tables.
        hidden_columns: Measurement columns to clear when a row matches.

    Returns:
        Lazy table with matching measurements hidden. If none of the requested
        columns exist, return the original lazy table unchanged.
    """
    available_columns = set(table.collect_schema().names())
    columns_to_clear = available_columns.intersection(hidden_columns)
    if not columns_to_clear:
        return table
    return (
        table.join(target_keys, on=join_columns, how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("_hide_target").fill_null(False))
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
            for column in columns_to_clear
        )
        .drop("_hide_target")
    )


def _empty_target_templates(
    task: Task,
    requested_target_keys_by_panel_and_split: Mapping[
        Panel, Mapping[Split, pl.DataFrame]
    ],
) -> PanelSplitTableMap[pl.LazyFrame]:
    """Create official-shaped solution inputs from requested target rows.

    Args:
        task: Task whose prediction columns are added.
        requested_target_keys_by_panel_and_split: Target row identifiers grouped
            by panel and release split.

    Returns:
        The requested rows with the task's prediction columns initialized to zero.
    """
    # Build Polars templates from identifiers only, without runner-held answers.
    return {
        panel: {
            split: target_table_by_panel.select(KEYS[task])
            .with_columns(pl.lit(0).alias(column) for column in VALUES[task])
            .lazy()
            for split, target_table_by_panel in split_targets.items()
        }
        for panel, split_targets in requested_target_keys_by_panel_and_split.items()
    }


def _load_predictor(solution_name: str, task: Task) -> object:
    """Import the task function exported by a solution package.

    Args:
        solution_name: Python package name below ``solutions``.
        task: Exported task function to load.

    Returns:
        The callable solution function.

    Raises:
        TypeError: If the package does not export the requested callable.
    """
    solution_module = importlib.import_module(f"solutions.{solution_name}")
    predictor = getattr(solution_module, task, None)
    if not callable(predictor):
        raise TypeError(
            f"solutions/{solution_name} must define "
            f"{task}(release_slice, target_templates, solution_parameters)"
        )
    return predictor


def _invoke_solution(
    solution_name: str,
    task: Task,
    release_slice: ReleasePackageSlice,
    requested_target_keys_by_panel_and_split: Mapping[
        Panel, Mapping[Split, pl.DataFrame]
    ],
    solution_parameters: JsonObject,
    random_seed: int,
) -> PanelSplitTableMap[pl.DataFrame]:
    """Call one solution once with all requested panels and release splits.

    Args:
        solution_name: Python package name below ``solutions``.
        task: Task function to invoke.
        release_slice: Inputs made visible to the solution after redaction.
        requested_target_keys_by_panel_and_split: Exact target rows to predict.
        solution_parameters: JSON settings supplied to the solution.
        random_seed: Seed supplied to the solution for repeatable behavior.

    Returns:
        Validated predictions grouped by panel and split.

    Raises:
        TypeError: If the solution returns an unsupported mapping or table type.
        ValueError: If returned panels, splits, or target row identifiers differ.
    """
    # Hide observations that would reveal requested answers before loading the
    # solution, then build zero-filled target rows from their identifiers.
    solution_visible_slice = _redact_target_period(
        replace(release_slice, seed=random_seed),
        task,
        requested_target_keys_by_panel_and_split,
    )
    target_templates = _empty_target_templates(
        task, requested_target_keys_by_panel_and_split
    )
    # Load the requested task entry point dynamically, then validate the shape
    # of its result immediately because static types cannot constrain imports.
    predictor = cast(
        Callable[
            [
                ReleasePackageSlice,
                PanelSplitTableMap[pl.LazyFrame],
                JsonObject,
            ],
            object,
        ],
        _load_predictor(solution_name, task),
    )
    raw_predictions = predictor(
        solution_visible_slice,
        target_templates,
        solution_parameters,
    )
    if not isinstance(raw_predictions, Mapping):
        raise TypeError(f"{task} must return a mapping from panel to split tables")
    raw_predictions_by_panel = cast(Mapping[Panel, object], raw_predictions)
    if set(raw_predictions_by_panel) != set(target_templates):
        raise ValueError(f"{task} predictions must cover exactly the requested panels")

    validated_predictions: PanelSplitTableMap[pl.DataFrame] = {}
    for panel, panel_targets in target_templates.items():
        raw_split_predictions_value = raw_predictions_by_panel[panel]
        if not isinstance(raw_split_predictions_value, Mapping):
            raise TypeError(f"{task}/{panel} must return a split-to-table mapping")
        raw_split_predictions = cast(
            Mapping[Split, object], raw_split_predictions_value
        )
        if set(raw_split_predictions) != set(panel_targets):
            raise ValueError(
                f"{task}/{panel} predictions must cover exactly the requested splits"
            )
        validated_predictions[panel] = {
            split: validate_predictions(
                task, target_table, raw_split_predictions[split]
            )
            for split, target_table in panel_targets.items()
        }
    return validated_predictions


def _score_answer_tables(
    task: Task,
    release_slice: ReleasePackageSlice,
    answer_tables_by_panel_and_split: Mapping[Panel, Mapping[Split, pl.DataFrame]],
    prediction_tables_by_panel_and_split: Mapping[Panel, Mapping[Split, pl.DataFrame]],
) -> list[dict[str, str | int | float]]:
    """Score runner-held answers against validated predictions.

    Args:
        task: Scoring task that selects the matching metric calculation.
        release_slice: Network and metadata used by the task scorer.
        answer_tables_by_panel_and_split: Answer values retained by the runner.
        prediction_tables_by_panel_and_split: Validated solution output tables.

    Returns:
        Metric rows with one separate score row for every Task 2 window.
    """
    metric_rows: list[dict[str, str | int | float]] = []
    for panel, answers_by_split in answer_tables_by_panel_and_split.items():
        for split, answer_table in answers_by_split.items():
            prediction_table = prediction_tables_by_panel_and_split[panel][split]
            panel_data = release_slice.panel_slices_by_panel[panel]
            identity = {
                "task": task,
                "split": split,
                "panel": panel,
                "family": panel_data.family_id,
            }
            match task:
                case "state":
                    metric_rows.extend(
                        {**identity, **row}
                        for row in state_metrics(
                            prediction_table,
                            answer_table,
                            panel_data.network,
                        )
                    )
                case "queue":
                    queue_answers = answer_table.rename({"queue_pred": "queue_true"})
                    if "eligible" not in queue_answers.columns:
                        queue_answers = queue_answers.with_columns(
                            pl.lit(True).alias("eligible")
                        )
                    for window_id in queue_answers["window_id"].unique().to_list():
                        window_answers = queue_answers.filter(
                            pl.col("window_id") == window_id
                        )
                        window_predictions = prediction_table.filter(
                            pl.col("window_id") == window_id
                        )
                        window_index = panel_data.queue_window_index.get(split)
                        condition = "generated"
                        if window_index is not None:
                            matching_window = (
                                window_index.filter(pl.col("window_id") == window_id)
                                .select("condition")
                                .collect()
                            )
                            if matching_window.height:
                                condition = matching_window["condition"][0]
                        metric_rows.append(
                            {
                                **identity,
                                "condition": condition,
                                "window_id": str(window_id),
                                **queue_metrics(window_predictions, window_answers),
                            }
                        )
                case "odme":
                    selected_counts = panel_data.link_counts[split].collect()
                    selected_prior = panel_data.weak_prior[split].collect()
                    metric_rows.append(
                        {
                            **identity,
                            **odme_metrics(
                                prediction_table,
                                selected_counts,
                                selected_prior,
                                panel_data.network,
                                answer_table,
                            ),
                        }
                    )
    return metric_rows


def run_solution(
    solution_name: str,
    task: Task,
    release_slice: ReleasePackageSlice,
    requested_target_keys_by_panel_and_split: Mapping[
        Panel, Mapping[Split, pl.DataFrame]
    ],
    answer_tables_by_panel_and_split: Mapping[Panel, Mapping[Split, pl.DataFrame]]
    | None = None,
    solution_parameters: JsonObject | None = None,
    random_seed: int = 0,
) -> SliceRunResult:
    """Predict all supplied target rows and score them when answers are available.

    The runner turns requested target rows into official-shaped templates with
    zero-valued prediction placeholders. The release slice contains historical
    labels, while optional answer maps carry future values and never reach the
    imported solution. Each Task 2 window receives its own IoU calculation.

    Args:
        solution_name: Python package name below ``solutions``.
        task: Task function to invoke and, when possible, score.
        release_slice: Public data, historical labels, and panel networks.
        requested_target_keys_by_panel_and_split: Exact target rows to predict.
        answer_tables_by_panel_and_split: Optional future answer values kept by
            the runner for scoring after predictions return.
        solution_parameters: JSON settings passed to the solution.
        random_seed: Seed passed to the solution for reproducibility.

    Returns:
        Validated predictions and available score or diagnostic rows.

    Raises:
        ValueError: If requested targets refer to panels or splits absent from
            the release slice or answer tables.
    """
    if not requested_target_keys_by_panel_and_split:
        raise ValueError("At least one target panel is required")
    if not set(requested_target_keys_by_panel_and_split).issubset(
        release_slice.panel_slices_by_panel
    ):
        raise ValueError("Targets contain a panel absent from the release slice")
    if answer_tables_by_panel_and_split is not None and any(
        panel not in requested_target_keys_by_panel_and_split
        or not set(answers_by_split).issubset(
            requested_target_keys_by_panel_and_split[panel]
        )
        for panel, answers_by_split in answer_tables_by_panel_and_split.items()
    ):
        raise ValueError(
            "Answer rows refer to a panel or split without requested targets"
        )

    predictions_by_panel_and_split = _invoke_solution(
        solution_name,
        task,
        release_slice,
        requested_target_keys_by_panel_and_split,
        solution_parameters or {},
        random_seed,
    )
    metric_rows = (
        _score_answer_tables(
            task,
            release_slice,
            answer_tables_by_panel_and_split,
            predictions_by_panel_and_split,
        )
        if answer_tables_by_panel_and_split is not None
        else []
    )
    for panel, predictions_by_split in predictions_by_panel_and_split.items():
        for split, prediction_table in predictions_by_split.items():
            panel_data = release_slice.panel_slices_by_panel[panel]
            identity = {
                "task": task,
                "split": split,
                "panel": panel,
                "family": panel_data.family_id,
            }
            match task:
                case "state":
                    metric_rows.append(
                        {
                            **identity,
                            **physics_diagnostics(prediction_table, panel_data.network),
                        }
                    )
                case "odme" if answer_tables_by_panel_and_split is None:
                    metric_rows.append(
                        {
                            **identity,
                            **odme_metrics(
                                prediction_table,
                                panel_data.link_counts[split].collect(),
                                panel_data.weak_prior[split].collect(),
                                panel_data.network,
                            ),
                        }
                    )
                case _:
                    pass
    return SliceRunResult(predictions_by_panel_and_split, pl.DataFrame(metric_rows))
