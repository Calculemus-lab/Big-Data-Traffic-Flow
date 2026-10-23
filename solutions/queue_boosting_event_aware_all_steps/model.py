"""Forecast Task 2 queues with LightGBM and condition-aware decoding.

Each panel has one general LightGBM model per forecast horizon and an onset
classifier trained on selector-matched events. Onset windows predict only at
T+30, where the released selector places the first queue. Models learn only from
the historical interval, restoring released Task 1 answers before deriving
speed-threshold labels. A final historical interval selects tree counts and
condition-specific thresholds by window IoU. Requested labels and measurements
after a forecast origin never become input features.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from solutions import baseline
from trafficbench import table_types
from trafficbench.contracts import (
    QUEUE_FORECAST_STEPS,
    QUEUE_INTERVAL_MINUTES,
    QUEUE_SPEED_FRACTION,
    JsonObject,
    Panel,
    PanelReleaseSlice,
    QueueCondition,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels

from .features import (
    CATEGORICAL_FEATURES,
    HISTORY_STEPS,
    MEASUREMENT_COLUMNS,
    FeatureMatrix,
    FloatArray,
    IndexArray,
    QueueNetwork,
    TrafficGrid,
    history_features,
    queue_network,
    traffic_grid,
)

logger = logging.getLogger(__name__)
type BoolArray = NDArray[np.bool_]
FALLBACK_QUEUE_THRESHOLD = 0.35
QUEUE_WINDOW_MINIMUM_COVERAGE = 0.7
ONSET_HORIZON_INDEX = QUEUE_FORECAST_STEPS - 1


class QueueParameters(BaseModel):
    """Training limits and optional overrides for calibrated queue thresholds.

    ``training_stride`` is measured in five-minute origins. Training rows are
    uniformly sampled per horizon up to ``max_training_rows``; forecast targets
    are never sampled. ``calibration_days`` is reserved within historical data
    before the final refit. Explicit thresholds override historical calibration.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    iterations: int = Field(default=250, ge=1)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    num_leaves: int = Field(default=31, ge=2)
    min_data_in_leaf: int = Field(default=100, ge=1)
    positive_weight: float = Field(default=2.0, gt=0)
    training_stride: int = Field(default=3, ge=1)
    max_training_rows: int = Field(default=250_000, ge=1)
    max_training_origins: int = Field(default=3000, ge=1)
    calibration_days: int = Field(default=7, ge=1)
    early_stopping_rounds: int = Field(default=30, ge=1)
    threads: int = Field(default=4, ge=1)
    threshold: float | None = Field(default=None, gt=0, lt=1)
    onset_threshold: float | None = Field(default=None, gt=0, lt=1)
    ongoing_threshold: float | None = Field(default=None, gt=0, lt=1)


class QueueThresholds(NamedTuple):
    """Probability cutoffs for the two published queue-window conditions."""

    queue_onset: float
    queue_ongoing: float


class QueueModels(NamedTuple):
    """Horizon models, optional onset head, and historical IoU thresholds.

    Attributes:
        horizons: General classifier for each of the six forecast steps.
        onset: Classifier fitted on selector-matched onset events, if available.
        thresholds: Calibrated probability cutoffs by published condition.
    """

    horizons: tuple[lgb.Booster, ...]
    onset: lgb.Booster | None
    thresholds: QueueThresholds


def historical_grid(
    release_slice: ReleasePackageSlice,
    panel_data: PanelReleaseSlice,
    network: QueueNetwork,
) -> TrafficGrid:
    """Restore historical state answers and exclude all post-cutoff measurements.

    Args:
        release_slice: Inclusive start and exclusive end of permitted train labels.
        panel_data: Released masked train states, historical answers, and ramps.
        network: Forecast links, static properties, and ramp attachments.

    Returns:
        A regular five-minute traffic grid strictly before ``history_end_date``.
    """
    start = datetime.combine(release_slice.history_start_date, datetime.min.time(), UTC)
    end = datetime.combine(release_slice.history_end_date, datetime.min.time(), UTC)
    period = pl.col("timestamp").is_between(start, end, closed="left")
    state_keys = ("timestamp", "station_id", "link_id")
    # Historical target answers restore speed/flow only at matching detector rows.
    answers = (
        panel_data.historical_task_labels.task1_state_answers.filter(period)
        .group_by(state_keys)
        .agg(pl.col("speed_kmh", "flow_vph").mean())
    )
    states = (
        panel_data.masked_mainline_states["train"]
        .filter(period)
        .select(*state_keys, *MEASUREMENT_COLUMNS, "pct_observed")
        .join(answers, on=state_keys, how="left", suffix="_answer")
        .with_columns(
            pl.coalesce(column, f"{column}_answer").alias(column)
            for column in ("speed_kmh", "flow_vph")
        )
        .collect()
    )
    ramps = panel_data.ramp_states["train"].filter(period).collect()
    timestamps = pl.datetime_range(
        start, end, interval=f"{QUEUE_INTERVAL_MINUTES}m", closed="left", eager=True
    ).to_list()
    return traffic_grid(states, ramps, timestamps, network)


def decode_queue_window(
    probabilities: FloatArray,
    threshold: float,
    condition: QueueCondition,
    eligible: BoolArray,
) -> BoolArray:
    """Decode one forecast while respecting event timing and positive-window rules.

    Args:
        probabilities: Queue probabilities shaped ``(links, horizons)``.
        threshold: Calibrated probability cutoff for the window condition.
        condition: Published onset or ongoing category for this forecast.
        eligible: Cells whose labels are available for historical calibration.

    Returns:
        Binary predictions with at least one eligible queue cell when possible.
        All six forecast steps remain available for onset and ongoing windows.
    """
    available = eligible
    predictions = (probabilities >= threshold) & available
    if not predictions.any() and available.any():
        # A valid competition window always contains a future queue.
        scores = np.where(available, probabilities, -np.inf)
        predictions.flat[np.argmax(scores)] = True
    return predictions


def calibration_thresholds(
    probabilities: FloatArray,
    labels: BoolArray,
    eligible: BoolArray,
    ongoing: BoolArray,
    usable_windows: BoolArray,
) -> QueueThresholds:
    """Maximize mean eligible-cell window IoU separately for onset and ongoing.

    Args:
        probabilities: Queue probabilities shaped ``(origins, links, horizons)``.
        labels: Historical speed-threshold labels with the same shape.
        eligible: Cells with reliable target measurements.
        ongoing: Whether each history contains an established queue.
        usable_windows: Complete windows meeting coverage and positive-queue rules.

    Returns:
        Two thresholds selected using historical windows only. A condition with
        fewer than two usable windows receives the fallback threshold ``0.35``.
    """
    threshold_candidates = np.linspace(0.1, 0.6, 21)
    thresholds: list[float] = []
    for condition in (False, True):
        selected = usable_windows & (ongoing == condition)
        if selected.sum() < 2:
            thresholds.append(FALLBACK_QUEUE_THRESHOLD)
            continue
        known = eligible[selected]
        true = labels[selected] & known
        scores = []
        # Every window receives equal weight even if its queue is much larger.
        condition_name: QueueCondition = "queue_ongoing" if condition else "queue_onset"
        for threshold in threshold_candidates:
            predicted = np.stack(
                [
                    decode_queue_window(
                        window_probabilities,
                        float(threshold),
                        condition_name,
                        window_eligible,
                    )
                    for window_probabilities, window_eligible in zip(
                        probabilities[selected], known, strict=True
                    )
                ]
            )
            intersection = (predicted & true).sum(axis=(1, 2))
            union = (predicted | true).sum(axis=(1, 2))
            scores.append(
                np.divide(
                    intersection, union, out=np.ones(len(union)), where=union > 0
                ).mean()
            )
        thresholds.append(float(threshold_candidates[np.argmax(scores)]))
    return QueueThresholds(*thresholds)


def _fit_horizon(
    features: FeatureMatrix,
    labels: BoolArray,
    eligible: BoolArray,
    training_rows: BoolArray,
    validation_rows: BoolArray,
    parameters: QueueParameters,
    seed: int,
    iterations: int,
) -> lgb.Booster:
    """Fit one horizon, uniformly limiting rows and optionally stopping early.

    Args:
        features: Shared causal link-history feature rows.
        labels: Flattened queue outcomes for this horizon.
        eligible: Flattened validity mask for this horizon's labels.
        training_rows: Row membership of the historical fit interval.
        validation_rows: Row membership of the later historical calibration interval.
        parameters: Tree settings and resource limits.
        seed: Deterministic random seed for row selection and LightGBM.
        iterations: Maximum number of boosting rounds for this fit.

    Returns:
        A fitted binary classifier; validation never includes requested targets.
    """
    rows = np.flatnonzero(eligible & training_rows)
    random_generator = np.random.default_rng(seed)
    if len(rows) > parameters.max_training_rows:
        rows = np.sort(
            random_generator.choice(rows, parameters.max_training_rows, replace=False)
        )
    training = lgb.Dataset(
        features.values[rows],
        label=labels[rows],
        feature_name=list(features.names),
        categorical_feature=list(CATEGORICAL_FEATURES),
    )
    validation_indices = np.flatnonzero(eligible & validation_rows)
    validations: list[lgb.Dataset] = []
    callbacks = []
    if len(validation_indices):
        validations.append(
            lgb.Dataset(
                features.values[validation_indices],
                label=labels[validation_indices],
                reference=training,
            )
        )
        callbacks.append(
            lgb.early_stopping(parameters.early_stopping_rounds, verbose=False)
        )
    return lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": parameters.learning_rate,
            "num_leaves": parameters.num_leaves,
            "min_data_in_leaf": parameters.min_data_in_leaf,
            "scale_pos_weight": parameters.positive_weight,
            "feature_fraction": 0.9,
            "lambda_l2": 1.0,
            "num_threads": parameters.threads,
            "seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        training,
        num_boost_round=iterations,
        valid_sets=validations,
        callbacks=callbacks,
    )


def _onset_candidate_origins(grid: TrafficGrid) -> IndexArray:
    """Find history and forecast origins that match the onset event selector.

    Args:
        grid: Historical five-minute traffic measurements for one panel.

    Returns:
        Origins with an established-queue-free history, no queued cell at T+5
        through T+25, and at least one queued cell at T+30. Both history and
        forecast coverage must meet the local competition contract.
    """
    speed = grid.speed_ratio
    known = np.isfinite(speed)
    queued = known & (speed <= QUEUE_SPEED_FRACTION)
    origins = np.arange(
        HISTORY_STEPS,
        len(grid.timestamps) - QUEUE_FORECAST_STEPS,
        dtype=np.int64,
    )
    if not len(origins):
        return origins
    # Sliding windows align each history queue count with the origin immediately after it.
    history_queues = np.lib.stride_tricks.sliding_window_view(
        queued, HISTORY_STEPS, axis=0
    ).sum(axis=-1, dtype=np.uint8)
    established = (history_queues[origins - HISTORY_STEPS] >= 2).any(axis=1)

    # Prefix sums test event timing and coverage without materializing every future window.
    queue_by_time = queued.any(axis=1)
    queued_prefix = np.concatenate(
        [np.zeros(1, dtype=np.uint32), np.cumsum(queue_by_time, dtype=np.uint32)]
    )
    known_by_time = known.sum(axis=1, dtype=np.uint32)
    known_prefix = np.concatenate(
        [np.zeros(1, dtype=np.uint32), np.cumsum(known_by_time, dtype=np.uint32)]
    )
    earlier_future_queues = (
        queued_prefix[origins + QUEUE_FORECAST_STEPS] - queued_prefix[origins + 1]
    )
    future_queue = queued[origins + QUEUE_FORECAST_STEPS].any(axis=1)
    history_known = known_prefix[origins] - known_prefix[origins - HISTORY_STEPS]
    future_known = (
        known_prefix[origins + QUEUE_FORECAST_STEPS + 1] - known_prefix[origins + 1]
    )
    history_coverage = history_known / (HISTORY_STEPS * speed.shape[1])
    future_coverage = future_known / (QUEUE_FORECAST_STEPS * speed.shape[1])
    enough_coverage = (history_coverage >= QUEUE_WINDOW_MINIMUM_COVERAGE) & (
        future_coverage >= QUEUE_WINDOW_MINIMUM_COVERAGE
    )
    candidates = (
        enough_coverage & ~established & (earlier_future_queues == 0) & future_queue
    )
    return origins[candidates]


def select_training_origins(
    grid_length: int,
    parameters: QueueParameters,
    seed: int,
    required_origins: IndexArray | None = None,
) -> tuple[IndexArray, int]:
    """Bound historical feature memory while retaining the calibration interval.

    Args:
        grid_length: Number of regularly spaced historical timestamps.
        parameters: Origin stride, fit-origin cap, and calibration-day count.
        seed: Reproducible seed for uniformly sampling earlier fit origins.
        required_origins: Selector-matched event origins retained before regular
            origins use the history cap.

    Returns:
        Sorted selected origins and the calibration-start index. Recent calibration
        origins remain intact. Selector-matched onset origins take the earlier-fit
        cap before regular origins; a short history retains all available origins.
    """
    origins = np.arange(
        HISTORY_STEPS,
        grid_length - QUEUE_FORECAST_STEPS,
        parameters.training_stride,
        dtype=np.int64,
    )
    if required_origins is not None:
        origins = np.union1d(origins, required_origins)
    steps_per_day = 24 * 60 // QUEUE_INTERVAL_MINUTES
    cutoff = grid_length - parameters.calibration_days * steps_per_day
    earlier = origins + QUEUE_FORECAST_STEPS < cutoff
    if earlier.sum() > parameters.max_training_origins:
        random_generator = np.random.default_rng(seed)
        required = (
            np.intersect1d(origins[earlier], required_origins)
            if required_origins is not None
            else np.asarray([], dtype=np.int64)
        )
        if len(required) > parameters.max_training_origins:
            required = np.sort(
                random_generator.choice(
                    required, parameters.max_training_origins, replace=False
                )
            )
        remaining = origins[earlier & ~np.isin(origins, required)]
        sample_count = parameters.max_training_origins - len(required)
        selected = random_generator.choice(remaining, sample_count, replace=False)
        origins = np.sort(np.concatenate([selected, origins[~earlier]]))
        if len(required):
            origins = np.sort(np.concatenate([origins, required]))
    return origins, cutoff


def _fit_models(
    grid: TrafficGrid,
    network: QueueNetwork,
    parameters: QueueParameters,
    seed: int,
) -> QueueModels | None:
    """Calibrate on later historical origins, then refit on all allowed history.

    Args:
        grid: Historical traffic observations with no requested target values.
        network: Ordered mainline links and topology.
        parameters: Fit limits, historical calibration period, and threshold overrides.
        seed: Reproducible experiment seed.

    Returns:
        Six general horizon classifiers, an optional onset classifier, and two
        thresholds, or ``None`` without usable train labels.
    """
    origins, cutoff = select_training_origins(
        len(grid.timestamps),
        parameters,
        seed,
        required_origins=_onset_candidate_origins(grid),
    )
    if not len(origins):
        return None
    features = history_features(
        grid, origins, [grid.timestamps[index] for index in origins], network
    )
    future_indices = origins[:, None] + np.arange(1, QUEUE_FORECAST_STEPS + 1)
    future_speed = grid.speed_ratio[future_indices].transpose(0, 2, 1)
    eligible = np.isfinite(future_speed)
    labels = eligible & (future_speed <= QUEUE_SPEED_FRACTION)
    if not eligible.any():
        return None
    history_indices = origins[:, None] + np.arange(-HISTORY_STEPS, 0)
    history_speed = grid.speed_ratio[history_indices]
    history_known = np.isfinite(history_speed)
    history_queues = history_known & (history_speed <= QUEUE_SPEED_FRACTION)
    ongoing = (history_queues.sum(axis=1) >= 2).any(axis=1)
    window_coverage = (
        history_known.mean(axis=(1, 2)) >= QUEUE_WINDOW_MINIMUM_COVERAGE
    ) & (eligible.mean(axis=(1, 2)) >= QUEUE_WINDOW_MINIMUM_COVERAGE)
    # Match onset events selected from clear history with their first forecast
    # queue at the final horizon step.
    onset_windows = (
        window_coverage
        & ~ongoing
        & ~labels[:, :, :ONSET_HORIZON_INDEX].any(axis=(1, 2))
        & labels[:, :, ONSET_HORIZON_INDEX].any(axis=1)
    )
    ongoing_windows = window_coverage & ongoing & labels.any(axis=(1, 2))
    calibration_windows = onset_windows | ongoing_windows
    # Purge the six-step target horizon before the internal chronological split.
    training_origins = origins + QUEUE_FORECAST_STEPS < cutoff
    validation_origins = origins >= cutoff
    if not training_origins.any():
        training_origins = np.ones(len(origins), dtype=bool)
        validation_origins = np.zeros(len(origins), dtype=bool)
    link_count = len(network.link_ids)
    training_rows = np.repeat(training_origins, link_count)
    validation_rows = np.repeat(validation_origins, link_count)
    models: list[lgb.Booster] = []
    for horizon in range(QUEUE_FORECAST_STEPS):
        model = _fit_horizon(
            features,
            labels[:, :, horizon].ravel(),
            eligible[:, :, horizon].ravel(),
            training_rows,
            validation_rows,
            parameters,
            seed + horizon,
            parameters.iterations,
        )
        models.append(model)
    training_onset = training_origins & onset_windows
    validation_onset = validation_origins & onset_windows
    onset_model = None
    if training_onset.any():
        onset_model = _fit_horizon(
            features,
            labels[:, :, ONSET_HORIZON_INDEX].ravel(),
            eligible[:, :, ONSET_HORIZON_INDEX].ravel(),
            np.repeat(training_onset, link_count),
            np.repeat(validation_onset, link_count),
            parameters,
            seed + QUEUE_FORECAST_STEPS,
            parameters.iterations,
        )
    thresholds = QueueThresholds(FALLBACK_QUEUE_THRESHOLD, FALLBACK_QUEUE_THRESHOLD)
    if validation_origins.any() and calibration_windows[validation_origins].any():
        probabilities = np.stack(
            [
                np.asarray(
                    model.predict(features.values[validation_rows]), dtype=np.float32
                ).reshape(-1, link_count)
                for model in models
            ],
            axis=2,
        )
        if onset_model is not None:
            validation_onset_probabilities = np.asarray(
                onset_model.predict(features.values[validation_rows]),
                dtype=np.float32,
            ).reshape(-1, link_count)
            probabilities[onset_windows[validation_origins], :, ONSET_HORIZON_INDEX] = (
                validation_onset_probabilities[onset_windows[validation_origins]]
            )
        thresholds = calibration_thresholds(
            probabilities,
            labels[validation_origins],
            eligible[validation_origins],
            ongoing[validation_origins],
            calibration_windows[validation_origins],
        )
        # Final models use the full historical interval and the selected tree counts.
        all_rows = np.ones(len(features.values), dtype=bool)
        no_validation = np.zeros(len(features.values), dtype=bool)
        models = [
            _fit_horizon(
                features,
                labels[:, :, horizon].ravel(),
                eligible[:, :, horizon].ravel(),
                all_rows,
                no_validation,
                parameters,
                seed + horizon,
                model.best_iteration or parameters.iterations,
            )
            for horizon, model in enumerate(models)
        ]
        if onset_model is not None:
            onset_model = _fit_horizon(
                features,
                labels[:, :, ONSET_HORIZON_INDEX].ravel(),
                eligible[:, :, ONSET_HORIZON_INDEX].ravel(),
                np.repeat(onset_windows, link_count),
                no_validation,
                parameters,
                seed + QUEUE_FORECAST_STEPS,
                onset_model.best_iteration or parameters.iterations,
            )
    onset = parameters.onset_threshold or parameters.threshold or thresholds.queue_onset
    ongoing_threshold = (
        parameters.ongoing_threshold or parameters.threshold or thresholds.queue_ongoing
    )
    logger.info(
        "Queue boosting: %d origins, %d links, %d onset and %d ongoing event candidates; onset threshold %.3f, ongoing threshold %.3f; tree counts %s",
        len(origins),
        link_count,
        int(onset_windows.sum()),
        int(ongoing_windows.sum()),
        onset,
        ongoing_threshold,
        [model.current_iteration() for model in models],
    )
    return QueueModels(
        tuple(models), onset_model, QueueThresholds(onset, ongoing_threshold)
    )


def _predict_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.QueueFrame],
    solution_parameters: JsonObject,
) -> dict[Split, table_types.QueueFrame]:
    """Fit one panel and forecast only the requested windows from their histories.

    Args:
        release_slice: Published tables, historical date limits, and seed.
        panel: Mainline panel whose model and topology are used.
        target_templates_by_split: Requested complete queue-window target grids.
        solution_parameters: JSON settings validated as ``QueueParameters``.

    Returns:
        One complete binary queue-prediction table per requested split.
    """
    parameters = QueueParameters.model_validate(solution_parameters)
    panel_data = release_slice.panel_slices_by_panel[panel]
    network = queue_network(panel_data.network)
    fitted = _fit_models(
        historical_grid(release_slice, panel_data, network),
        network,
        parameters,
        release_slice.seed,
    )
    if fitted is None:
        logger.warning(
            "No usable queue training labels for %s; using persistence", panel
        )
        return baseline.queue(
            release_slice, {panel: dict(target_templates_by_split)}, {}
        )[panel]
    predictions: dict[Split, table_types.QueueFrame] = {}
    link_positions = {link_id: index for index, link_id in enumerate(network.link_ids)}
    threshold_by_condition: dict[QueueCondition, float] = {
        "queue_onset": fitted.thresholds.queue_onset,
        "queue_ongoing": fitted.thresholds.queue_ongoing,
    }
    interval = timedelta(minutes=QUEUE_INTERVAL_MINUTES)
    for split, targets in target_templates_by_split.items():
        target_frame = targets.collect()
        if target_frame.is_empty():
            predictions[split] = target_frame.lazy()
            continue
        window_ids = target_frame.select("window_id").unique()
        index = (
            panel_data.queue_window_index[split]
            .join(window_ids.lazy(), on="window_id", how="semi")
            .collect()
        )
        histories = (
            panel_data.queue_history[split]
            .join(window_ids.lazy(), on="window_id", how="semi")
            .collect()
        )
        ramps = panel_data.ramp_states[split].collect()
        windows: list[pl.DataFrame] = []
        # Each window uses its own history; other windows cannot provide later evidence.
        for window_id, origin, condition in index.select(
            "window_id", "forecast_origin", "condition"
        ).iter_rows():
            timestamps = [
                origin - interval * step for step in range(HISTORY_STEPS, 0, -1)
            ]
            history = histories.filter(pl.col("window_id").eq(window_id))
            history_ramps = ramps.filter(
                pl.col("timestamp").is_between(timestamps[0], origin, closed="left")
            )
            grid = traffic_grid(history, history_ramps, timestamps, network)
            features = history_features(
                grid, np.asarray([HISTORY_STEPS], dtype=np.int64), [origin], network
            )
            probability = np.stack(
                [
                    np.asarray(model.predict(features.values), dtype=np.float32)
                    for model in fitted.horizons
                ],
                axis=1,
            )
            if condition == "queue_onset" and fitted.onset is not None:
                probability[:, ONSET_HORIZON_INDEX] = np.asarray(
                    fitted.onset.predict(features.values), dtype=np.float32
                )
            threshold = threshold_by_condition[condition]
            prediction = decode_queue_window(
                probability,
                threshold,
                condition,
                np.ones(probability.shape, dtype=bool),
            )
            # Keep the caller's row order while indexing each link and horizon forecast.
            window_targets = target_frame.filter(pl.col("window_id").eq(window_id))
            link_indices = np.asarray(
                [link_positions[link_id] for link_id in window_targets["link_id"]],
                dtype=np.int64,
            )
            horizon_indices = np.asarray(
                [
                    (timestamp - origin) // interval - 1
                    for timestamp in window_targets["timestamp"]
                ],
                dtype=np.int64,
            )
            windows.append(
                window_targets.with_columns(
                    pl.Series(
                        "queue_pred",
                        prediction[link_indices, horizon_indices].astype(np.int64),
                    )
                )
            )
        predictions[split] = (
            pl.concat(windows)
            .select("window_id", "timestamp", "link_id", "queue_pred")
            .lazy()
        )
    return predictions


def queue(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.QueueFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
    """Train and forecast all requested Task 2 panels through the runner contract.

    Args:
        release_slice: Selected released inputs and earlier historical labels.
        target_templates_by_panel_and_split: Zero-filled queue rows grouped by panel
            and split. Every supplied row is predicted, without evaluation sampling.
        solution_parameters: JSON settings validated as ``QueueParameters``.

    Returns:
        Complete official-shaped binary queue predictions by panel and split.
    """
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        _predict_panel,
    )
