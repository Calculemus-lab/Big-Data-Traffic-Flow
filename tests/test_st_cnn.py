"""Behavior tests for the spatiotemporal CNN state solution."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from solutions.st_cnn import model
from solutions.st_cnn.grid import (
    STEPS_PER_DAY,
    Corridor,
    Normalization,
    Profile,
    build_corridor,
    epoch_step,
    features,
    locate,
    step_range,
    to_grid,
    weekday_and_slot,
)
from solutions.st_cnn.net import Imputer, receptive_steps
from trafficbench.contracts import JsonObject
from trafficbench.fixture import make_fixture
from trafficbench.prepare import prepare
from trafficbench.runner import run_experiment

FAST: JsonObject = {
    "steps": 3,
    "batch": 2,
    "window": 48,
    "width": 8,
    "device": "cpu",
    "log_every": 0,
}
# 2030-01-07 is a Monday.
MONDAY_STEP = model.first_step_of(date(2030, 1, 7))


@pytest.fixture(scope="module")
def benchmark_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare the synthetic fixture with the same dates as the CI smoke run."""
    root = tmp_path_factory.mktemp("st_cnn")
    make_fixture(root / "release")
    prepare(
        root / "release",
        root / "benchmark",
        history_start_date=date(2030, 11, 3),
        history_end_date=date(2030, 11, 20),
        prediction_end_date=date(2030, 11, 28),
        panels=["D12_I5_N"],
    )
    return root / "benchmark"


def run_metrics(
    benchmark_directory: Path, runs_directory: Path, parameters: JsonObject
) -> dict[str, float]:
    """Run st_cnn through the harness and return its aggregated metrics."""
    path = run_experiment(
        "st_cnn",
        "state",
        benchmark_directory,
        solution_parameters=parameters,
        runs_directory=runs_directory,
        compare_baseline=False,
    )
    metadata = json.loads((path / "run.json").read_text())
    assert metadata["status"] == "completed"
    return metadata["metrics"]


def zero_network(*_: object) -> Imputer:
    """Return a network whose output is zero, so predictions equal the profile."""
    net = Imputer(7, 4)
    for parameter in net.out.parameters():
        torch.nn.init.zeros_(parameter)
    return net.eval()


def mainline(rows: list[tuple[str, str, str, float, float, int]]) -> pl.DataFrame:
    """Build a small mainline table from (time, station, link, speed, flow, pct) rows."""
    columns = [
        "timestamp",
        "station_id",
        "link_id",
        "speed_kmh",
        "flow_vph",
        "pct_observed",
    ]
    return pl.DataFrame(rows, schema=columns, orient="row").with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC")
    )


def single_link_corridor() -> Corridor:
    series = pl.DataFrame({"station_id": ["S_A"], "link_id": ["A"]})
    return Corridor(series.with_row_index("column"), np.array([1.0]))


def test_runs_end_to_end_through_the_harness(
    benchmark_directory: Path, tmp_path: Path
) -> None:
    metrics = run_metrics(benchmark_directory, tmp_path, FAST)
    assert 0 < metrics["state_score"] <= 1
    assert metrics["negative_state_fraction_diagnostic"] == 0


def test_profile_path_reproduces_a_repeating_fixture(
    benchmark_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a zero network the output is the profile; the fixture repeats daily.

    Any misalignment of time, columns, lanes or targets breaks the exact score.
    """
    monkeypatch.setattr(model, "fit", zero_network)
    metrics = run_metrics(benchmark_directory, tmp_path, {**FAST, "smooth": 0})
    assert metrics["speed_rmse"] == pytest.approx(0, abs=1e-6)
    assert metrics["flow_per_lane_rmse"] == pytest.approx(0, abs=1e-6)
    assert metrics["state_score"] == pytest.approx(1)


def test_same_seed_repeats_and_seed_offset_changes_the_fit(
    benchmark_directory: Path, tmp_path: Path
) -> None:
    first = run_metrics(benchmark_directory, tmp_path / "a", FAST)
    again = run_metrics(benchmark_directory, tmp_path / "b", FAST)
    other = run_metrics(benchmark_directory, tmp_path / "c", {**FAST, "seed": 1})
    assert first["speed_rmse"] == again["speed_rmse"]
    assert first["speed_rmse"] != other["speed_rmse"]


def test_evaluation_pool_never_contains_target_cells(
    benchmark_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pools_seen: list[list[model.Pool]] = []
    corridors_seen: list[Corridor] = []
    original_build_corridor = model.build_corridor

    def record_corridor(
        topology: pl.LazyFrame, fd_parameters: pl.LazyFrame, frames: list[pl.DataFrame]
    ) -> Corridor:
        corridor = original_build_corridor(topology, fd_parameters, frames)
        corridors_seen.append(corridor)
        return corridor

    def record_pools(pools: list[model.Pool], *_: object) -> Imputer:
        pools_seen.append(pools)
        return zero_network()

    monkeypatch.setattr(model, "build_corridor", record_corridor)
    monkeypatch.setattr(model, "fit", record_pools)
    run_metrics(benchmark_directory, tmp_path / "eval", FAST)
    assert len(pools_seen[-1]) == 2
    evaluation = pools_seen[-1][1]
    targets = pl.read_parquet(next(benchmark_directory.rglob("state/targets.parquet")))
    time_index, column = locate(
        targets, evaluation.first_step, evaluation.values.shape[1], corridors_seen[-1]
    )
    assert (time_index >= 0).all()
    assert np.isnan(evaluation.values[:, time_index, column]).all()

    run_metrics(benchmark_directory, tmp_path / "history", {**FAST, "use_eval": False})
    assert len(pools_seen[-1]) == 1


@pytest.mark.parametrize("bad", [{"steps": 0}, {"unknown": 1}, {"device": "tpu"}])
def test_rejects_invalid_parameters(
    benchmark_directory: Path, tmp_path: Path, bad: JsonObject
) -> None:
    with pytest.raises(ValueError):
        run_metrics(benchmark_directory, tmp_path, {**FAST, **bad})


def test_grid_round_trip_in_corridor_order() -> None:
    data = mainline(
        [
            ("2030-01-01T00:00:00", "S2", "B", 50.0, 3000.0, 100),
            ("2030-01-01T00:05:00", "S1", "A", 90.0, 1000.0, 100),
            ("2030-01-01T00:10:00", "S1", "A", 80.0, 1200.0, 60),
        ]
    )
    corridor = build_corridor(
        pl.LazyFrame({"link_id": ["A", "B"], "order_index": [1, 0]}),
        pl.LazyFrame({"link_id": ["A", "B"], "lanes": [2.0, 3.0]}),
        [data],
    )
    assert corridor.series["link_id"].to_list() == ["B", "A"]
    assert corridor.lanes.tolist() == [3.0, 2.0]
    first, count = step_range([data])
    assert first % STEPS_PER_DAY == 0 and count == 3
    grid = to_grid(data, first, count, corridor)
    assert grid[0, 0, 0] == 50 and grid[1, 0, 0] == 1000  # per-lane flow
    assert grid[0, 1, 1] == 90 and grid[1, 1, 1] == 500
    assert np.isnan(grid[:, 2, 1]).all()  # pct_observed < 75 is not evidence
    time_index, column = locate(data, first, count, corridor)
    assert time_index.tolist() == [0, 1, 2] and column.tolist() == [0, 1, 1]


def test_locate_marks_unknown_and_out_of_range_rows() -> None:
    data = mainline(
        [
            ("2030-01-01T00:00:00", "S9", "Z", 0.0, 0.0, 100),
            ("2030-01-02T00:00:00", "S_A", "A", 0.0, 0.0, 100),
        ]
    )
    first = int(data.select(epoch_step())["step"][0])
    time_index, column = locate(data, first, 1, single_link_corridor())
    assert time_index.tolist() == [-1, -1] and column.tolist() == [-1, -1]


def test_corridor_rejects_links_without_lanes() -> None:
    rows = mainline([("2030-01-01T00:00:00", "S", "B", 1.0, 1.0, 100)])
    with pytest.raises(ValueError, match="lane"):
        build_corridor(
            pl.LazyFrame({"link_id": ["A"], "order_index": [0]}),
            pl.LazyFrame({"link_id": ["A"], "lanes": [2.0]}),
            [rows],
        )


def test_weekday_and_slot_follow_utc_calendar() -> None:
    steps = np.array([MONDAY_STEP, MONDAY_STEP + 287, MONDAY_STEP + 288])
    weekday, slot = weekday_and_slot(steps)
    assert weekday.tolist() == [0, 0, 1] and slot.tolist() == [0, 287, 0]


def test_profile_averages_by_weekday_and_falls_back() -> None:
    count = 14 * STEPS_PER_DAY
    weekday, slot = weekday_and_slot(MONDAY_STEP + np.arange(count))
    monday = weekday == 0
    grid = np.full((2, count, 3), np.nan)
    grid[:, :, 0] = np.where(monday, 10.0, 20.0)
    grid[:, monday, 1] = slot[monday]  # Mondays only
    values = Profile(grid, MONDAY_STEP, half_width=0)(MONDAY_STEP, count)
    assert values.shape == grid.shape
    assert np.allclose(values[:, monday, 0], 10)
    assert np.allclose(values[:, ~monday, 0], 20)
    assert np.allclose(values[:, :, 1], slot)  # time-of-day fallback
    assert np.allclose(values[:, :, 2], np.nanmean(grid, axis=(1, 2))[:, None])


def test_profile_smoothing_continues_across_midnight() -> None:
    grid = np.zeros((2, 7 * STEPS_PER_DAY, 1))
    grid[:, 287, 0] = 30.0  # Monday 23:55
    profile = Profile(grid, MONDAY_STEP, half_width=1)
    values = profile(MONDAY_STEP, 7 * STEPS_PER_DAY)
    assert np.allclose(values[:, 286:289, 0], 10)  # Monday 23:50 to Tuesday 00:00
    assert (values[:, :286, 0] == 0).all() and (values[:, 289:, 0] == 0).all()


def test_profile_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="No usable"):
        Profile(np.full((2, STEPS_PER_DAY, 1), np.nan), MONDAY_STEP)


def test_hidden_cells_follow_regime_rates() -> None:
    rng = np.random.default_rng(0)
    visible = rng.random((200, 200)) < 0.7
    rates: set[float] = set()
    for _ in range(60):
        hidden = model.hide(visible, rng)
        assert not (hidden & ~visible).any()
        rate = float(hidden.sum() / visible.sum())
        assert min(abs(rate - regime_rate) for regime_rate in model.RATES) < 0.02
        rates.add(round(rate, 1))
    assert rates == set(model.RATES)


def test_training_batches_only_show_unhidden_cells() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(100, 10, (2, 96, 6))
    values[:, rng.random((96, 6)) < 0.3] = np.nan
    profile = np.full_like(values, 90.0)
    norm = Normalization(np.array([5.0, 50.0]), np.zeros(2), np.ones(2))
    parameters = model.StCnnParameters(batch=3, window=96)
    pools = [model.Pool(values, profile, MONDAY_STEP)]
    inputs, targets, masks = next(model.batches(pools, parameters, norm, rng))
    visible = ~np.isnan(values[0])
    for x, y, hidden in zip(
        inputs.numpy(), targets.numpy(), masks.numpy().astype(bool), strict=True
    ):
        shown = visible & ~hidden
        assert hidden.any() and not (hidden & ~visible).any()
        np.testing.assert_array_equal(x[2].astype(bool), shown)
        assert (x[:2][:, ~shown] == 0).all()
        np.testing.assert_allclose(
            x[0][shown], ((values[0] - 90) / 5)[shown], rtol=1e-5
        )
        np.testing.assert_allclose(
            x[1][shown], ((values[1] - 90) / 50)[shown], rtol=1e-5
        )
        expected = (values - profile)[:, visible]
        np.testing.assert_allclose(y[:, visible], expected, rtol=1e-5)


def test_features_zero_hidden_residuals() -> None:
    values = np.full((2, 4, 3), 7.0)
    visible = np.zeros((4, 3), bool)
    visible[1, 2] = True
    norm = Normalization(np.array([2.0, 4.0]), np.zeros(2), np.ones(2))
    x = features(values, visible, np.ones((2, 4, 3)), MONDAY_STEP, norm)
    assert x.shape == (7, 4, 3) and x.dtype == np.float32
    assert x[0, 1, 2] == 3 and x[1, 1, 2] == 1.5
    assert (x[:2][:, ~visible] == 0).all() and x[2].sum() == 1


def test_score_loss_matches_competition_formula() -> None:
    target = torch.zeros(1, 2, 2, 2)
    prediction = target.clone()
    prediction[:, 0] = 5.0  # speed error 5 km/h
    prediction[:, 1] = 60.0  # flow error 60 veh/h/lane
    mask = torch.ones(1, 2, 2)
    mask[0, 0, 0] = 0
    prediction[0, :, 0, 0] = 1e4  # unmasked cells must not count
    loss = model.score_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(0.54 * 5 / 25 + 0.46 * 60 / 600)


def test_chunked_prediction_matches_a_direct_pass() -> None:
    torch.manual_seed(0)
    net = Imputer(7, 4).eval()
    rng = np.random.default_rng(0)
    hidden = rng.random((300, 5)) < 0.5
    values = np.where(hidden, np.nan, rng.normal(0, 10, (2, 300, 5)))
    profile = rng.normal(0, 1, (2, 300, 5))
    norm = Normalization(np.array([3.0, 70.0]), np.zeros(2), np.ones(2))
    with torch.no_grad():
        x = torch.from_numpy(features(values, ~hidden, profile, MONDAY_STEP, norm))
        direct = net(x[None])[0].numpy() * norm.residual[:, None, None] + profile
    pool = model.Pool(values, profile, MONDAY_STEP)
    chunked = model.predict(net, pool, norm, chunk=70)
    assert receptive_steps() < 70 and np.isfinite(chunked).all()
    np.testing.assert_allclose(chunked, direct, atol=1e-4)


def test_all_cycle_two_settings_run_together(
    benchmark_directory: Path, tmp_path: Path
) -> None:
    settings: JsonObject = {
        **FAST,
        "blackout_rate": 0.5,
        "blackout_check": 2,
        "ensemble": 2,
        "low_quality_inputs": True,
        "leave_day_out": True,
    }
    metrics = run_metrics(benchmark_directory, tmp_path, settings)
    assert 0 < metrics["state_score"] <= 1


def test_ensemble_changes_and_repeats_the_prediction(
    benchmark_directory: Path, tmp_path: Path
) -> None:
    single = run_metrics(benchmark_directory, tmp_path / "a", FAST)
    pair = run_metrics(benchmark_directory, tmp_path / "b", {**FAST, "ensemble": 2})
    again = run_metrics(benchmark_directory, tmp_path / "c", {**FAST, "ensemble": 2})
    assert pair["speed_rmse"] != single["speed_rmse"]
    assert pair["speed_rmse"] == again["speed_rmse"]


def test_blackout_augmentation_hides_one_full_block() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(100, 10, (2, 96, 6))
    profile = np.full_like(values, 90.0)
    norm = Normalization(np.ones(2), np.zeros(2), np.ones(2))
    parameters = model.StCnnParameters(batch=4, window=96, blackout_rate=1.0)
    pools = [model.Pool(values, profile, MONDAY_STEP)]
    _, _, masks = next(model.batches(pools, parameters, norm, rng))
    for hidden in masks.numpy().astype(bool):
        full_rows = hidden.all(axis=1)
        runs = np.convolve(full_rows, np.ones(model.BLACKOUT_STEPS), mode="valid")
        assert runs.max() == model.BLACKOUT_STEPS


def test_blackout_starts_are_reproducible_daytime_target_days() -> None:
    target_steps = MONDAY_STEP + np.arange(3 * STEPS_PER_DAY)
    starts = model.blackout_starts("D12_I5_N", target_steps, 5)
    assert np.array_equal(starts, model.blackout_starts("D12_I5_N", target_steps, 5))
    assert not np.array_equal(
        starts, model.blackout_starts("D7_I10_E", target_steps, 5)
    )
    _, slot = weekday_and_slot(starts)
    assert ((slot >= 72) & (slot < 240)).all()
    assert set(starts // STEPS_PER_DAY) <= set(target_steps // STEPS_PER_DAY)


def test_blackout_check_blanks_evaluation_inputs(
    benchmark_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pools_seen: list[list[model.Pool]] = []

    def record_pools(pools: list[model.Pool], *_: object) -> Imputer:
        pools_seen.append(pools)
        return zero_network()

    monkeypatch.setattr(model, "fit", record_pools)
    run_metrics(benchmark_directory, tmp_path, {**FAST, "blackout_check": 3})
    evaluation = pools_seen[-1][1]
    targets = pl.read_parquet(next(benchmark_directory.rglob("state/targets.parquet")))
    target_steps = targets.select(epoch_step())["step"].to_numpy()
    for start in model.blackout_starts("D12_I5_N", target_steps, 3):
        rows = slice(start - evaluation.first_step, start - evaluation.first_step + 18)
        assert np.isnan(evaluation.values[:, rows]).all()


def test_low_quality_grid_holds_only_partial_coverage_cells() -> None:
    data = mainline(
        [
            ("2030-01-01T00:00:00", "S_A", "A", 50.0, 1000.0, 50),
            ("2030-01-01T00:05:00", "S_A", "A", 60.0, 1000.0, 100),
            ("2030-01-01T00:10:00", "S_A", "A", 70.0, 1000.0, 0),
        ]
    )
    corridor = single_link_corridor()
    first, count = step_range([data])
    weak = to_grid(data, first, count, corridor, low_quality=True)
    eligible = to_grid(data, first, count, corridor)
    assert weak[0, :, 0].tolist()[0] == 50 and np.isnan(weak[0, 1:, 0]).all()
    assert np.isnan(eligible[0, 0, 0]) and eligible[0, 1, 0] == 60


def test_features_add_low_quality_channels() -> None:
    values = np.full((2, 4, 3), 7.0)
    visible = np.ones((4, 3), bool)
    weak = np.full((2, 4, 3), np.nan)
    weak[:, 2, 1] = 9.0
    norm = Normalization(np.array([2.0, 4.0]), np.zeros(2), np.ones(2))
    x = features(values, visible, np.ones((2, 4, 3)), MONDAY_STEP, norm, weak)
    assert x.shape == (10, 4, 3)
    assert x[7, 2, 1] == 4 and x[8, 2, 1] == 2 and x[9].sum() == 1


def test_leave_day_out_matches_refitting_without_that_day() -> None:
    rng = np.random.default_rng(1)
    count = 21 * STEPS_PER_DAY
    grid = rng.normal(100, 10, (2, count, 2))
    grid[rng.random(grid.shape) < 0.2] = np.nan
    left_out = Profile(grid, MONDAY_STEP, half_width=2).without_own_day(
        grid, MONDAY_STEP
    )
    for day in [0, 9, 20]:
        rows = slice(day * STEPS_PER_DAY, (day + 1) * STEPS_PER_DAY)
        without = grid.copy()
        without[:, rows] = np.nan
        refit = Profile(without, MONDAY_STEP, half_width=2)(MONDAY_STEP, count)
        np.testing.assert_allclose(left_out[:, rows], refit[:, rows], rtol=1e-9)
