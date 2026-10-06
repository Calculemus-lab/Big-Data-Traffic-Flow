"""Prepare repeatable local cases from the public competition release.

Each case stores the data a solution may see, the rows it must predict, and
separate answer values for local scoring. The manifest records case locations
and file hashes so repeated runs can use the same prepared data.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray
from pydantic import FilePath

from . import VERSION, table_types
from .contracts import (
    KEYS,
    MIN_OBSERVED_PERCENT,
    QUEUE_FORECAST_STEPS,
    QUEUE_HISTORY_MINUTES,
    QUEUE_INTERVAL_MINUTES,
    QUEUE_SPEED_FRACTION,
    BenchmarkConfig,
    BenchmarkManifest,
    NetworkTables,
    OdmeCase,
    Panel,
    Profile,
    QueueCase,
    QueueCondition,
    QueueConfig,
    QueueWindowInfo,
    ReleaseCorridors,
    StateCase,
)
from .data import (
    Release,
    empty_typed_table,
    file_hash,
    metadata_hash,
    release_calendar,
    save_table,
    select_queue_solutions,
    select_state_solutions,
    validate_experiment_dates,
)
from .metrics import link_path_incidence_matrix


class QueueWindow(NamedTuple):
    """Data and labels for one locally generated queue forecast window.

    Attributes:
        info: Window ID, UTC forecast origin, and onset or ongoing condition.
        history: Visible speed and coverage measurements for the prior 60 minutes.
        targets: Rows to predict for every link at all six future times.
        answer: Estimated queue states and an eligibility flag for each target row.
        index_rows: Forecast times and measured history and forecast coverage.
    """

    info: QueueWindowInfo
    history: pl.DataFrame
    targets: pl.DataFrame
    answer: pl.DataFrame
    index_rows: pl.DataFrame


def load_config(cv_scheme_file: FilePath) -> BenchmarkConfig:
    """Read the YAML benchmark scheme and validate its named settings.

    Args:
        cv_scheme_file: YAML file containing panel, queue, and ODME settings.

    Returns:
        Validated benchmark configuration.
    """
    return BenchmarkConfig.model_validate(yaml.safe_load(cv_scheme_file.read_text()))


def build_queue_proxy_windows(
    state_answer_table: pl.DataFrame,
    panel: Panel,
    family_id: str,
    case_id: str,
    network: NetworkTables,
    benchmark_config: QueueConfig,
    seed: int,
    history_end_date: date,
    prediction_start_date: date,
    prediction_end_date: date,
) -> list[QueueWindow]:
    """Build local queue examples from measured traffic states.

    For each forecast origin, the visible history covers the preceding 60
    minutes, including the timestamp 60 minutes before the origin and excluding
    the origin itself. Target rows cover the next six five-minute timestamps,
    from 5 through 30 minutes after the origin. A link is queued when its
    measured speed is at most 60% of its free-flow speed. Classify a window as
    ongoing when any link is queued at two or more history timestamps. Classify
    the other eligible windows as onset. Keep windows meeting the configured
    measurement coverage and spacing requirements.

    Args:
        state_answer_table: Unmasked train measurements covering history and targets.
        panel: Corridor used to name the generated windows.
        family_id: Freeway-family label used when aggregating queue scores.
        case_id: Date-interval identifier included in generated window IDs.
        network: Link free-flow speeds used to classify queue states.
        benchmark_config: Window count, spacing, and coverage requirements.
        seed: Seed used to select repeatable candidate windows.
        history_end_date: Exclusive end of the earlier train history interval.
        prediction_start_date: First date included in forecast targets.
        prediction_end_date: First date excluded from forecast targets.

    Returns:
        A list of windows, each containing its visible history, target rows,
        queue labels, and forecast-window metadata.

    """
    # Build a complete time-by-link grid. Missing release rows then count as
    # uncovered measurements instead of shortening a candidate window.
    link_table = (
        network.links.select("link_id", "free_speed_kmh").collect().sort("link_id")
    )
    link_ids = link_table["link_id"].to_list()
    free_flow_speeds = link_table["free_speed_kmh"].to_numpy()
    if np.isnan(free_flow_speeds).any():
        raise ValueError("Missing queue threshold")

    grouped_states = state_answer_table.group_by("timestamp", "link_id").agg(
        pl.col("speed_kmh").mean(),
        (
            pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
            & pl.col("speed_kmh").is_not_null()
        )
        .all()
        .alias("eligible"),
    )
    # Polars Series.to_list() does not preserve the column's datetime type in
    # its annotations, so assert the timestamp schema established by the loader.
    state_timestamps = cast(list[datetime], state_answer_table["timestamp"].to_list())
    first_timestamp = min(state_timestamps)
    last_timestamp = max(state_timestamps)
    interval = timedelta(minutes=QUEUE_INTERVAL_MINUTES)
    timestamps = [
        first_timestamp + offset * interval
        for offset in range(int((last_timestamp - first_timestamp) / interval) + 1)
    ]
    time_grid = pl.DataFrame({"timestamp": timestamps})
    link_grid = time_grid.join(link_table.select("link_id"), how="cross")
    state_grid = link_grid.join(
        grouped_states, on=["timestamp", "link_id"], how="left"
    ).sort("timestamp", "link_id")
    speed_by_time_and_link: NDArray[np.float64] = (
        state_grid["speed_kmh"]
        .fill_null(float("nan"))
        .to_numpy()
        .reshape(len(timestamps), len(link_ids))
    )
    eligible_cells: NDArray[np.bool_] = (
        state_grid["eligible"]
        .fill_null(False)
        .to_numpy()
        .reshape(len(timestamps), len(link_ids))
    )
    queued: NDArray[np.bool_] = (
        speed_by_time_and_link <= QUEUE_SPEED_FRACTION * free_flow_speeds[np.newaxis, :]
    ) & eligible_cells
    # Randomize candidate origins with a fixed seed, then select enough
    # spaced examples of each condition for a reproducible local comparison.
    random_generator = np.random.default_rng(seed)
    history_steps = QUEUE_HISTORY_MINUTES // QUEUE_INTERVAL_MINUTES
    possible_origins = np.arange(history_steps, len(timestamps) - QUEUE_FORECAST_STEPS)
    prediction_start_timestamp = datetime.combine(
        prediction_start_date, datetime.min.time(), UTC
    )
    prediction_end_timestamp = datetime.combine(
        prediction_end_date, datetime.min.time(), UTC
    )
    history_end_timestamp = datetime.combine(history_end_date, datetime.min.time(), UTC)
    possible_origins = np.asarray(
        [
            origin_index
            for origin_index in possible_origins
            if timestamps[origin_index] + interval >= prediction_start_timestamp
            and timestamps[origin_index] + interval >= history_end_timestamp
            and timestamps[origin_index] + QUEUE_FORECAST_STEPS * interval
            < prediction_end_timestamp
        ],
        dtype=np.int64,
    )
    candidate_origins: NDArray[np.int64] = random_generator.permutation(
        possible_origins
    )
    selected_origins: dict[QueueCondition, list[int]] = {
        "queue_onset": [],
        "queue_ongoing": [],
    }
    windows: list[QueueWindow] = []
    # Sample without replacement so a fixed seed produces repeatable cases.
    for candidate in candidate_origins:
        origin_index = int(candidate)
        history_queues = queued[origin_index - history_steps : origin_index]
        future_queues = queued[
            origin_index + 1 : origin_index + 1 + QUEUE_FORECAST_STEPS
        ]
        if (
            eligible_cells[origin_index - history_steps : origin_index].mean()
            < benchmark_config.minimum_coverage
            or eligible_cells[
                origin_index + 1 : origin_index + 1 + QUEUE_FORECAST_STEPS
            ].mean()
            < benchmark_config.minimum_coverage
            or not future_queues.any()
        ):
            continue
        # Match the downloaded contract's established-queue definition.
        condition: QueueCondition = (
            "queue_ongoing"
            if (history_queues.sum(axis=0) >= 2).any()
            else "queue_onset"
        )
        previous_origins = selected_origins[condition]
        if len(previous_origins) >= benchmark_config.windows_per_condition or any(
            abs(origin_index - previous_index) * QUEUE_INTERVAL_MINUTES
            < benchmark_config.minimum_spacing_minutes
            for previous_index in previous_origins
        ):
            continue
        # Build a complete target grid so every link is predicted at each
        # forecast timestamp. The ``eligible`` column later excludes poor labels.
        previous_origins.append(origin_index)
        forecast_origin = timestamps[origin_index]
        window_id = f"proxy_{panel}_{case_id}_{condition}_{len(previous_origins):02d}"
        history_columns = list(table_types.QueueHistoryColumns.__annotations__)
        history = (
            state_answer_table.filter(
                pl.col("timestamp").is_between(
                    forecast_origin - timedelta(minutes=QUEUE_HISTORY_MINUTES),
                    forecast_origin,
                    closed="left",
                )
            )
            .select(column for column in history_columns if column != "window_id")
            .with_columns(pl.lit(window_id).alias("window_id"))
            .select(history_columns)
        )
        forecast_timestamps = timestamps[
            origin_index + 1 : origin_index + 1 + QUEUE_FORECAST_STEPS
        ]
        target = pl.DataFrame(
            {
                "window_id": [window_id] * len(forecast_timestamps) * len(link_ids),
                "timestamp": [
                    forecast_timestamp
                    for forecast_timestamp in forecast_timestamps
                    for _ in link_ids
                ],
                "link_id": link_ids * len(forecast_timestamps),
            }
        )
        answer = target.with_columns(
            pl.Series("queue_true", future_queues.ravel().astype(int)),
            pl.Series(
                "eligible",
                eligible_cells[
                    origin_index + 1 : origin_index + 1 + QUEUE_FORECAST_STEPS
                ].ravel(),
            ),
        )
        history_start = forecast_origin - timedelta(minutes=QUEUE_HISTORY_MINUTES)
        forecast_end = forecast_origin + QUEUE_FORECAST_STEPS * interval
        index = pl.DataFrame(
            {
                "window_id": [window_id],
                "panel": [panel],
                "family_id": [family_id],
                "split": ["train"],
                "date": [forecast_origin.date().isoformat()],
                "forecast_origin": [forecast_origin],
                "history_start": [history_start],
                "history_end": [forecast_origin],
                "forecast_start": [forecast_origin + interval],
                "forecast_end": [forecast_end],
                "condition": [condition],
                "history_coverage": [
                    float(
                        eligible_cells[
                            origin_index - history_steps : origin_index
                        ].mean()
                    )
                ],
                "future_coverage": [
                    float(
                        eligible_cells[
                            origin_index + 1 : origin_index + 1 + QUEUE_FORECAST_STEPS
                        ].mean()
                    )
                ],
            }
        )
        windows.append(
            QueueWindow(
                QueueWindowInfo(
                    window_id=window_id,
                    forecast_origin_utc=forecast_origin,
                    condition=condition,
                ),
                history,
                target,
                answer,
                index,
            )
        )
        if all(
            len(indices) == benchmark_config.windows_per_condition
            for indices in selected_origins.values()
        ):
            break
    # Never silently compare experiments on different condition coverage.
    if any(not indices for indices in selected_origins.values()):
        raise ValueError(
            f"{panel}/{case_id}: no windows for both queue conditions. Enlarge prediction dates"
        )
    return sorted(windows, key=lambda window: window.info.window_id)


def prepare(
    release_directory: Path,
    benchmark_directory: Path,
    history_start_date: date,
    history_end_date: date,
    prediction_end_date: date,
    prediction_start_date: date | None = None,
    profile: Profile = "quick",
    cv_scheme_file: FilePath = Path("config/cv_scheme.yaml"),
    panels: Sequence[Panel] | None = None,
) -> Path:
    """Prepare one train holdout interval for local solution experiments.

    The solution receives published masked inputs for the full visible train
    interval, filled historical target rows through ``history_end_date``, and
    target templates with zero-valued prediction placeholders for the
    prediction interval. State and queue answer values are generated from
    later unmasked measurements and remain in the runner for scoring.
    Historical Task 2 labels are local proxies because the
    official queue labels are withheld. Task 4 cases use each selected panel's
    train OD data and include one released-count diagnostic plus deterministic
    synthetic cases with separate path-flow answers.

    All date intervals are half-open. Queue windows must have their complete
    60-minute history inside the selected history period and all six forecast
    timestamps inside the selected prediction interval.

    Args:
        release_directory: Extracted release root containing ``config/``.
        benchmark_directory: Empty or new destination for prepared case tables.
        history_start_date: Inclusive first date of visible and labeled history.
        history_end_date: Exclusive end of the earlier train-label interval.
        prediction_end_date: Exclusive end of the generated target interval.
        prediction_start_date: First target date. Defaults to ``history_end_date``.
        profile: Named configuration selecting default panels and case settings.
        cv_scheme_file: YAML file defining local case-generation settings.
        panels: Optional panel override. Defaults to the selected profile.

    Returns:
        Path to the prepared benchmark directory containing its manifest.

    Raises:
        ValueError: If dates are invalid, the target period leaves train, the
            output directory is non-empty, or no panels are selected.
    """
    if benchmark_directory.exists() and any(benchmark_directory.iterdir()):
        raise ValueError(
            f"{benchmark_directory} is not empty. Choose a new benchmark directory."
        )
    config = load_config(cv_scheme_file)
    if profile not in config.profiles:
        raise ValueError(f"Unknown profile: {profile}")

    release = Release(release_directory)
    calendar = release_calendar(release)
    prediction_start_date = prediction_start_date or history_end_date
    validate_experiment_dates(
        calendar,
        history_start_date,
        history_end_date,
        prediction_start_date,
        prediction_end_date,
    )
    if prediction_end_date > calendar["train"][1]:
        raise ValueError("Generated cases must stay within the train split")
    profile_panel_selection = config.profiles[profile].panels
    selected_panels: list[Panel]
    if panels is None:
        if profile_panel_selection == "all":
            selected_panels = sorted(
                corridor.corridor_id
                for corridor in ReleaseCorridors.model_validate_json(
                    release.read_bytes("config/corridors.json")
                ).panels
            )
        else:
            selected_panels = sorted(profile_panel_selection)
    else:
        selected_panels = list(dict.fromkeys(panels))
    if not selected_panels:
        raise ValueError("Select at least one panel")

    family_ids = {
        corridor.corridor_id: corridor.family_id
        for corridor in ReleaseCorridors.model_validate_json(
            release.read_bytes("config/corridors.json")
        ).panels
    }
    case_id = f"{prediction_start_date.isoformat()}_{prediction_end_date.isoformat()}"
    manifest = BenchmarkManifest(
        version=VERSION,
        profile=profile,
        scheme=config,
        data_fingerprint=release.fingerprint,
        cases=[],
        warnings=[],
        files={},
        benchmark_id="",
    )
    benchmark_directory.mkdir(parents=True, exist_ok=True)

    for panel in selected_panels:
        print(f"Preparing {panel} for {case_id}", flush=True)
        network = release.scan_network_tables(panel)
        for network_table_name, network_table in network.items():
            save_table(
                benchmark_directory
                / "network"
                / panel
                / f"{network_table_name}.parquet",
                network_table,
            )

        panel_directory = Path(panel) / case_id
        unmasked_history_and_forecasts = release.read_state_rows(
            panel,
            "train",
            "mainline_states",
            history_start_date,
            prediction_end_date,
            allow_empty=True,
        )
        history_end_timestamp = datetime.combine(
            history_end_date, datetime.min.time(), UTC
        )
        prediction_start_timestamp = datetime.combine(
            prediction_start_date, datetime.min.time(), UTC
        )
        prediction_end_timestamp = datetime.combine(
            prediction_end_date, datetime.min.time(), UTC
        )
        unmasked_train_history = unmasked_history_and_forecasts.filter(
            pl.col("timestamp") < history_end_timestamp
        )
        unmasked_case_traffic = unmasked_history_and_forecasts.filter(
            pl.col("timestamp") >= history_end_timestamp
        )
        masked_case_traffic = release.read_state_rows(
            panel,
            "train",
            "mainline_states_masked",
            history_start_date,
            prediction_end_date,
            allow_empty=True,
        )
        ramp_case_traffic = release.read_state_rows(
            panel,
            "train",
            "ramp_states",
            history_start_date,
            prediction_end_date,
            allow_empty=True,
        )
        save_table(
            benchmark_directory / panel_directory / "masked_mainline_states.parquet",
            masked_case_traffic,
        )
        save_table(
            benchmark_directory / panel_directory / "ramp_states.parquet",
            ramp_case_traffic,
        )

        # Fill historical Task 1 template rows, keeping those labels outside
        # the feature tables given to the solution.
        historical_state_template = release.read_csv_rows(
            f"task1/{panel}/train/sample_submission_state.csv",
            history_start_date,
            history_end_date,
        )
        state_solutions_path = (
            panel_directory / "state" / "historical_solutions.parquet"
        )
        historical_state_solutions = select_state_solutions(
            historical_state_template.lazy(),
            unmasked_train_history.lazy(),
        )
        save_table(
            benchmark_directory / state_solutions_path,
            historical_state_solutions,
        )

        released_queue_index_path = f"task2/{panel}/train/window_index.csv"
        if released_queue_index_path in release.file_names:
            all_released_queue_index = release.read_csv(released_queue_index_path)
            # Past queue targets are local proxy labels, while published
            # window histories remain ordinary release features.
            historical_queue_index = all_released_queue_index.filter(
                pl.col("history_start").ge(
                    datetime.combine(history_start_date, datetime.min.time(), UTC)
                )
                & pl.col("forecast_end").lt(history_end_timestamp)
            )
            historical_queue_template = release.read_csv(
                f"task2/{panel}/train/sample_submission_queue.csv"
            ).join(
                historical_queue_index.select("window_id"),
                on="window_id",
                how="semi",
            )
            historical_queue_solutions = select_queue_solutions(
                historical_queue_template.lazy(),
                unmasked_train_history.lazy(),
                network,
            )
            queue_solutions_path = (
                panel_directory / "queue" / "historical_solutions.parquet"
            )
            save_table(
                benchmark_directory / queue_solutions_path,
                historical_queue_solutions,
            )

            # Keep all official window-history features visible within the
            # same release interval as the other dated input tables.
            released_queue_index = all_released_queue_index.filter(
                pl.col("forecast_start").ge(
                    datetime.combine(history_start_date, datetime.min.time(), UTC)
                )
                & pl.col("forecast_start").lt(prediction_end_timestamp)
            )
            released_queue_history_path = f"task2/{panel}/train/window_history.parquet"
            all_released_queue_history = release.scan_parquet(
                released_queue_history_path
            )
            released_queue_history = all_released_queue_history.join(
                released_queue_index.select("window_id").lazy(),
                on="window_id",
                how="semi",
            )
        else:
            historical_queue_solutions = empty_typed_table(
                table_types.QueueColumns
            ).collect()
            queue_solutions_path = (
                panel_directory / "queue" / "historical_solutions.parquet"
            )
            save_table(
                benchmark_directory / queue_solutions_path,
                historical_queue_solutions,
            )
            released_queue_index = empty_typed_table(
                table_types.QueueWindowIndexColumns
            ).collect()
            released_queue_history = empty_typed_table(
                table_types.QueueHistoryColumns
            ).collect()
        save_table(
            benchmark_directory
            / panel_directory
            / "released_queue_window_index.parquet",
            released_queue_index,
        )
        save_table(
            benchmark_directory / panel_directory / "released_queue_history.parquet",
            released_queue_history,
        )

        # Official Task 1 target rows identify answers in later unmasked train data.
        state_template = release.read_csv(
            f"task1/{panel}/train/sample_submission_state.csv"
        )
        state_targets = state_template.filter(
            pl.col("timestamp").ge(prediction_start_timestamp)
            & pl.col("timestamp").lt(prediction_end_timestamp)
        ).select(KEYS["state"])
        state_answer_source = unmasked_case_traffic.filter(
            pl.col("timestamp").ge(prediction_start_timestamp)
            & pl.col("timestamp").lt(prediction_end_timestamp)
        )
        state_case_directory = panel_directory / "state"
        state_answer_table = select_state_solutions(
            state_targets.lazy(),
            state_answer_source.lazy(),
        )
        save_table(
            benchmark_directory / state_case_directory / "targets.parquet",
            state_targets,
        )
        save_table(
            benchmark_directory / state_case_directory / "truth.parquet",
            state_answer_table,
        )
        manifest.cases.append(
            StateCase(
                panel=panel,
                family_id=family_ids[panel],
                case_id=case_id,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                prediction_start_date=prediction_start_date,
                prediction_end_date=prediction_end_date,
                case_directory=state_case_directory,
                historical_solutions_path=state_solutions_path,
            )
        )

        # Each queue forecast needs a complete permitted history and six future labels.
        if panel not in config.queue.excluded_panels:
            queue_windows = build_queue_proxy_windows(
                unmasked_history_and_forecasts,
                panel,
                family_ids[panel],
                case_id,
                network,
                config.queue,
                config.seed,
                history_end_date,
                prediction_start_date,
                prediction_end_date,
            )
            condition_counts = {
                condition: sum(
                    window.info.condition == condition for window in queue_windows
                )
                for condition in ("queue_onset", "queue_ongoing")
            }
            if min(condition_counts.values()) < config.queue.windows_per_condition:
                manifest.warnings.append(
                    f"{panel}/{case_id}: only {condition_counts}. Queue conditions "
                    "remain equally weighted"
                )
            queue_case_directory = panel_directory / "queue"
            save_table(
                benchmark_directory / queue_case_directory / "observations.parquet",
                pl.concat([window.history for window in queue_windows]),
            )
            save_table(
                benchmark_directory / queue_case_directory / "targets.parquet",
                pl.concat([window.targets for window in queue_windows]),
            )
            save_table(
                benchmark_directory / queue_case_directory / "truth.parquet",
                pl.concat([window.answer for window in queue_windows]),
            )
            save_table(
                benchmark_directory / queue_case_directory / "window_index.parquet",
                pl.concat([window.index_rows for window in queue_windows]),
            )
            manifest.cases.append(
                QueueCase(
                    panel=panel,
                    family_id=family_ids[panel],
                    case_id=case_id,
                    history_start_date=history_start_date,
                    history_end_date=history_end_date,
                    prediction_start_date=prediction_start_date,
                    prediction_end_date=prediction_end_date,
                    case_directory=queue_case_directory,
                    historical_solutions_path=queue_solutions_path,
                    windows=[window.info for window in queue_windows],
                )
            )

        # Task 4 path-flow cases share the same train scenario inputs regardless
        # of the selected traffic dates.
        odme_case_directory = panel_directory / "odme"
        prior = release.read_csv(f"task4/{panel}/train/synthetic_weak_prior.csv")
        counts = release.read_csv(f"task4/{panel}/train/synthetic_link_counts.csv")
        save_table(
            benchmark_directory / panel_directory / "link_counts.parquet", counts
        )
        save_table(benchmark_directory / panel_directory / "weak_prior.parquet", prior)
        odme_targets = release.read_csv(
            f"task4/{panel}/train/sample_submission_path_flow.csv"
        ).select(KEYS["odme"])
        incidence_matrix = link_path_incidence_matrix(network, odme_targets, counts)
        target_path_order = odme_targets.select("path_id").with_row_index("_order")
        prior_path_flows: NDArray[np.float64] = (
            target_path_order.join(prior.select("path_id", "path_flow"), on="path_id")
            .sort("_order")["path_flow"]
            .to_numpy()
        )
        panel_seed = int(metadata_hash([config.seed, panel])[:8], 16)
        random_generator = np.random.default_rng(panel_seed)

        for scenario_index in range(config.odme.synthetic_cases + 1):
            scenario_name = (
                "released_counts"
                if scenario_index == 0
                else f"synthetic_{scenario_index}"
            )
            scenario_directory = odme_case_directory / scenario_name
            scenario_counts = counts
            scenario_truth: pl.DataFrame | None = None
            if scenario_index:
                synthetic_path_flows: NDArray[np.float64] = (
                    prior_path_flows
                    * random_generator.lognormal(
                        0,
                        config.odme.demand_log_sigma,
                        len(prior_path_flows),
                    )
                )
                count_noise = random_generator.normal(
                    0, config.odme.count_noise_fraction, len(counts)
                )
                scenario_counts = counts.with_columns(
                    pl.Series(
                        "count",
                        np.maximum(
                            0,
                            incidence_matrix @ synthetic_path_flows * (1 + count_noise),
                        ),
                    )
                )
                scenario_truth = odme_targets.with_columns(
                    pl.Series("path_flow", synthetic_path_flows)
                )
            save_table(
                benchmark_directory / scenario_directory / "counts.parquet",
                scenario_counts,
            )
            save_table(
                benchmark_directory / scenario_directory / "prior.parquet", prior
            )
            save_table(
                benchmark_directory / scenario_directory / "targets.parquet",
                odme_targets,
            )
            if scenario_truth is not None:
                save_table(
                    benchmark_directory / scenario_directory / "truth.parquet",
                    scenario_truth,
                )
            manifest.cases.append(
                OdmeCase(
                    panel=panel,
                    family_id=family_ids[panel],
                    case_id=case_id,
                    history_start_date=history_start_date,
                    history_end_date=history_end_date,
                    prediction_start_date=prediction_start_date,
                    prediction_end_date=prediction_end_date,
                    case_directory=scenario_directory,
                    scenario_name=scenario_name,
                )
            )

    # Hash every stored table so later experiments use the exact prepared data.
    manifest.files = {
        str(parquet_path.relative_to(benchmark_directory)): file_hash(parquet_path)
        for parquet_path in sorted(benchmark_directory.rglob("*.parquet"))
    }
    manifest.benchmark_id = metadata_hash(manifest.hash_content())
    (benchmark_directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2, exclude_none=True)
    )
    print(
        f"Ready: {benchmark_directory} ({len(manifest.cases)} cases, "
        f"{manifest.benchmark_id[:12]})"
    )
    return benchmark_directory
