"""Check causal inputs, coverage, calibration, and complete queue predictions."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import sparse

from solutions.queue_boosting_event_aware.features import (
    HISTORY_STEPS,
    FeatureMatrix,
    QueueNetwork,
    TrafficGrid,
    history_features,
    queue_network,
    traffic_grid,
)
from solutions.queue_boosting_event_aware.model import (
    QueueParameters,
    _add_horizon_context,
    _fit_onset,
    _onset_candidate_origins,
    calibration_thresholds,
    decode_queue_window,
    dropout_history,
    historical_grid,
    select_training_origins,
)
from trafficbench.contracts import JsonObject
from trafficbench.data import Release, load_release_slice
from trafficbench.runner import run_experiment


def test_shared_horizon_context_repeats_condition_per_link() -> None:
    """Shared forecast models receive horizon and condition for every link row."""
    features = FeatureMatrix(
        np.asarray([[0.2, 0.4], [0.8, 1.0], [0.3, 0.5], [0.7, 0.9]], dtype=np.float32),
        ("speed", "occupancy"),
    )

    contextual = _add_horizon_context(
        features, 30, np.asarray([False, True], dtype=np.bool_)
    )

    assert contextual.names == (
        "speed",
        "occupancy",
        "forecast_minutes",
        "ongoing_condition",
    )
    np.testing.assert_array_equal(contextual.values[:, 2], [30, 30, 30, 30])
    np.testing.assert_array_equal(contextual.values[:, 3], [0, 0, 1, 1])


@pytest.fixture
def two_link_network() -> QueueNetwork:
    """Create two connected links with one on-ramp attached upstream."""
    downstream = sparse.csr_matrix(np.asarray([[0, 1], [0, 0]], dtype=np.float32))
    return QueueNetwork(
        ("upstream", "downstream"),
        np.asarray([100, 100], dtype=np.float32),
        np.asarray([2000, 2000], dtype=np.float32),
        {},
        downstream.T.tocsr(),
        downstream,
        pl.DataFrame(
            {
                "ramp_link_id": ["ramp"],
                "ramp_type": ["OR"],
                "nearest_mainline_link_id": ["upstream"],
            }
        ),
    )


def test_features_exclude_origin_and_future_and_preserve_direction(
    two_link_network: QueueNetwork,
) -> None:
    start = datetime(2030, 11, 3, tzinfo=UTC)
    times = tuple(start + timedelta(minutes=5 * step) for step in range(14))
    speeds = np.tile(np.asarray([0.4, 1.0], dtype=np.float32), (len(times), 1))
    zeros = np.zeros_like(speeds)
    grid = TrafficGrid(times, speeds, zeros, zeros, zeros, zeros)
    origins = np.asarray([HISTORY_STEPS], dtype=np.int64)
    before = history_features(grid, origins, [times[HISTORY_STEPS]], two_link_network)
    # Changed values at T and later must never influence this forecast's features.
    speeds[HISTORY_STEPS:] = 50
    after = history_features(grid, origins, [times[HISTORY_STEPS]], two_link_network)
    np.testing.assert_allclose(before.values, after.values, equal_nan=True)
    assert before.values[0, before.names.index("speed_ratio_lag_5")] == pytest.approx(
        0.4
    )
    assert before.values[0, before.names.index("downstream_speed_lag_5")] == 1
    assert before.values[
        1, before.names.index("upstream_speed_lag_5")
    ] == pytest.approx(0.4)


def test_grid_retains_measured_zero_and_distinguishes_missing_ramps(
    two_link_network: QueueNetwork,
) -> None:
    start = datetime(2030, 11, 3, tzinfo=UTC)
    times = (start, start + timedelta(minutes=5))
    states = pl.DataFrame(
        {
            "timestamp": [start, start],
            "link_id": ["upstream", "downstream"],
            "speed_kmh": [100.0, 20.0],
            "flow_vph": [0.0, 1000.0],
            "occupancy": [0.0, 0.2],
            "pct_observed": [90, 50],
        }
    )
    ramps = pl.DataFrame(
        {
            "timestamp": [start],
            "ramp_link_id": ["ramp"],
            "ramp_type": ["OR"],
            "flow_vph": [0.0],
            "pct_observed": [90],
        }
    )
    grid = traffic_grid(states, ramps, times, two_link_network)
    assert grid.flow_ratio[0, 0] == 0
    assert np.isnan(grid.speed_ratio[0, 1])
    assert grid.on_ramp_ratio[0, 0] == 0
    assert np.isnan(grid.on_ramp_ratio[1, 0])
    assert grid.on_ramp_ratio[1, 1] == 0


def test_training_grid_ignores_post_cutoff_values(
    prepared_benchmark_directory: Path,
) -> None:
    release_slice = load_release_slice(
        Release(prepared_benchmark_directory.parent / "release"),
        date(2030, 11, 3),
        date(2030, 11, 20),
        date(2030, 11, 20),
        date(2030, 11, 28),
        panels=("D12_I5_N",),
    )
    panel_data = release_slice.panel_slices_by_panel["D12_I5_N"]
    network = queue_network(panel_data.network)
    original = historical_grid(release_slice, panel_data, network)
    cutoff = datetime(2030, 11, 20, tzinfo=UTC)
    # Replace all later observations with extreme values and check the train boundary.
    changed_states = {
        split: frame.with_columns(
            pl.when(pl.col("timestamp").ge(cutoff))
            .then(1.0)
            .otherwise(pl.col("speed_kmh"))
            .alias("speed_kmh")
        )
        for split, frame in panel_data.masked_mainline_states.items()
    }
    changed_panel = replace(panel_data, masked_mainline_states=changed_states)
    changed = historical_grid(release_slice, changed_panel, network)
    assert max(original.timestamps) < cutoff
    np.testing.assert_allclose(
        original.speed_ratio, changed.speed_ratio, equal_nan=True
    )


def test_calibration_uses_condition_and_eligible_cells() -> None:
    # The third cell has a deliberately misleading probability but is unscored.
    probabilities = np.asarray(
        [
            [[0.2], [0.1], [0.9]],
            [[0.2], [0.1], [0.9]],
            [[0.55], [0.35], [0.9]],
            [[0.55], [0.35], [0.9]],
        ],
        dtype=np.float32,
    )
    labels = np.tile(np.asarray([[True], [False], [False]], dtype=bool), (4, 1, 1))
    eligible = np.tile(np.asarray([[True], [True], [False]], dtype=bool), (4, 1, 1))
    thresholds = calibration_thresholds(
        probabilities,
        labels,
        eligible,
        np.asarray([False, False, True, True], dtype=bool),
        np.ones(4, dtype=bool),
    )
    assert 0.1 <= thresholds.queue_onset <= 0.2
    assert 0.35 < thresholds.queue_ongoing <= 0.55


def test_onset_decoder_targets_final_step_and_guarantees_one_eligible_cell() -> None:
    probabilities = np.asarray(
        [
            [0.99, 0.99, 0.99, 0.99, 0.99, 0.1],
            [0, 0, 0, 0, 0, 0.8],
            [0, 0, 0, 0, 0, 0.7],
        ],
        dtype=np.float32,
    )
    eligible = np.ones(probabilities.shape, dtype=bool)
    eligible[1, -1] = False
    eligible[2, -1] = True

    predictions = decode_queue_window(probabilities, 0.95, "queue_onset", eligible)

    assert predictions.sum() == 1
    assert predictions[2, -1]
    assert not predictions[:, :-1].any()


def test_ongoing_decoder_keeps_thresholded_cells_and_falls_back_when_empty() -> None:
    probabilities = np.asarray([[0.8, 0.1], [0.2, 0.9], [0.2, 0.2]], dtype=np.float32)
    eligible = np.ones(probabilities.shape, dtype=bool)
    eligible[1, 1] = False

    predictions = decode_queue_window(probabilities, 0.75, "queue_ongoing", eligible)

    np.testing.assert_array_equal(
        predictions, np.asarray([[True, False], [False, False], [False, False]])
    )
    fallback = decode_queue_window(probabilities, 0.95, "queue_ongoing", eligible)
    assert fallback.sum() == 1
    assert fallback[0, 0]


def test_onset_candidates_match_first_queue_at_final_horizon_step(
    two_link_network: QueueNetwork,
) -> None:
    start = datetime(2030, 11, 3, tzinfo=UTC)
    times = tuple(start + timedelta(minutes=5 * step) for step in range(40))
    speed = np.ones((len(times), 2), dtype=np.float32)
    speed[26:, 0] = 0.4
    zeros = np.zeros_like(speed)
    grid = TrafficGrid(times, speed, zeros, zeros, zeros, zeros)

    candidates = _onset_candidate_origins(grid)

    np.testing.assert_array_equal(candidates, np.asarray([20], dtype=np.int64))


def test_long_history_origin_sampling_preserves_calibration_and_short_histories() -> (
    None
):
    parameters = QueueParameters(max_training_origins=20, calibration_days=2)
    grid_length = 10 * 288
    all_origins = np.arange(HISTORY_STEPS, grid_length - 6, parameters.training_stride)
    sampled, cutoff = select_training_origins(grid_length, parameters, seed=7)
    assert (sampled + 6 < cutoff).sum() == parameters.max_training_origins
    np.testing.assert_array_equal(
        sampled[sampled + 6 >= cutoff], all_origins[all_origins + 6 >= cutoff]
    )
    repeated, _ = select_training_origins(grid_length, parameters, seed=7)
    np.testing.assert_array_equal(sampled, repeated)
    uncapped, _ = select_training_origins(grid_length, QueueParameters(), seed=7)
    np.testing.assert_array_equal(uncapped, all_origins)


def test_origin_sampling_retains_required_onset_events_within_the_cap() -> None:
    parameters = QueueParameters(max_training_origins=20, calibration_days=2)
    grid_length = 10 * 288
    required = np.asarray([HISTORY_STEPS + 1, HISTORY_STEPS + 2], dtype=np.int64)

    sampled, cutoff = select_training_origins(
        grid_length, parameters, seed=7, required_origins=required
    )

    assert set(required).issubset(sampled)
    assert (sampled + 6 < cutoff).sum() == parameters.max_training_origins


def test_queue_boosting_predicts_all_targets_reproducibly(
    prepared_benchmark_directory: Path,
    tmp_path: Path,
) -> None:
    parameters: JsonObject = {
        "iterations": 8,
        "min_data_in_leaf": 5,
        "max_training_rows": 2000,
        "calibration_days": 3,
        "threads": 1,
    }
    first = run_experiment(
        "queue_boosting_event_aware",
        "queue",
        prepared_benchmark_directory,
        solution_parameters=parameters,
        runs_directory=tmp_path,
        compare_baseline=False,
    )
    second = run_experiment(
        "queue_boosting_event_aware",
        "queue",
        prepared_benchmark_directory,
        solution_parameters=parameters,
        runs_directory=tmp_path,
        compare_baseline=False,
    )
    assert json.loads((first / "run.json").read_text())["status"] == "completed"
    for file in (first / "predictions").rglob("*.parquet"):
        prediction = pl.read_parquet(file).sort(["window_id", "timestamp", "link_id"])
        repeated = pl.read_parquet(second / file.relative_to(first)).sort(
            ["window_id", "timestamp", "link_id"]
        )
        assert prediction.equals(repeated)
        assert set(prediction["queue_pred"]) <= {0, 1}


def test_history_dropout_preserves_original_inputs_and_shared_missingness() -> None:
    """Remove all channels at the same cells without modifying the source grid."""
    values = np.ones((40, 3), dtype=np.float32)
    times = tuple(
        datetime(2030, 11, 3, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(40)
    )
    grid = TrafficGrid(times, values, values, values, values, values)
    augmented = dropout_history(grid, 0.5, 31)
    assert np.isnan(augmented.speed_ratio).any()
    assert np.isfinite(augmented.speed_ratio).any()
    for original, changed in zip(grid[1:], augmented[1:], strict=True):
        np.testing.assert_array_equal(original, values)
        np.testing.assert_array_equal(
            np.isnan(changed), np.isnan(augmented.speed_ratio)
        )
    np.testing.assert_array_equal(
        augmented.speed_ratio, dropout_history(grid, 0.5, 31).speed_ratio
    )


def test_augmented_onset_blend_excludes_nontraining_labels() -> None:
    """Copies of excluded origins must remain excluded from both classifiers."""
    random = np.random.default_rng(31)
    values = np.column_stack(
        [random.random(80), np.tile([0, 1], 40), np.zeros(80)]
    ).astype(np.float32)
    features = FeatureMatrix(values, ("speed", "link_index", "day_of_week"))
    labels = values[:, 0] < 0.5
    training = np.arange(80) < 40
    parameters = QueueParameters(
        iterations=5,
        min_data_in_leaf=2,
        threads=1,
        onset_catboost_weight=0.5,
        onset_catboost_depth=3,
    )
    eligible = np.ones(80, dtype=bool)
    no_validation = np.zeros(80, dtype=bool)
    first = _fit_onset(
        features,
        features,
        labels,
        eligible,
        training,
        no_validation,
        parameters,
        31,
        5,
        5,
    )
    changed = labels.copy()
    changed[~training] = ~changed[~training]
    repeated = _fit_onset(
        features,
        features,
        changed,
        eligible,
        training,
        no_validation,
        parameters,
        31,
        5,
        5,
    )
    assert first.catboost is not None
    probability = first.predict(values)
    assert probability.shape == (80,)
    assert np.all((probability >= 0) & (probability <= 1))
    np.testing.assert_allclose(probability, repeated.predict(values))
