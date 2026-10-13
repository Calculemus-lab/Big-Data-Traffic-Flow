"""Spatiotemporal masked CNN for Task 1 state reconstruction.

Self-supervised: visible cells are hidden at the competition rates and the
network learns to fill them in from nearby times and links. It predicts
residuals over a day-of-week x time-of-day profile and is trained on the
competition score itself.

Local experiment: ``bench experiment st_cnn --task state --benchmark DIR``.
Competition rows: ``bench run st_cnn --task state --target-range validation``.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from functools import partial
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field

from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    VALUES,
    JsonObject,
    Panel,
    PanelReleaseSlice,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels

from .grid import (
    SERIES,
    STEP_SECONDS,
    STEPS_PER_DAY,
    BoolArray,
    Corridor,
    FloatArray,
    IntArray,
    Normalization,
    Profile,
    build_corridor,
    epoch_step,
    features,
    locate,
    step_range,
    to_grid,
)
from .net import Imputer, receptive_steps

RATES = (0.2, 0.3, 0.5)
# Validation and private blank the 30-minute Task 2 horizon plus the following hour.
BLACKOUT_STEPS = 18
OBSERVATION_COLUMNS = ["timestamp", *SERIES, "speed_kmh", "flow_vph", "pct_observed"]


class StCnnParameters(BaseModel):
    """Validated settings for training and inference.

    Attributes:
        steps: Optimizer steps per panel.
        batch: Crops per optimizer step.
        window: Crop length in five-minute steps.
        width: Convolution channels.
        lr: Peak learning rate of the one-cycle schedule.
        smooth: Profile smoothing half-width in slots.
        use_eval: Also train on visible cells after the history period.
        device: Torch device; ``auto`` picks MPS when available.
        log_every: Print the running loss every this many steps; 0 is silent.
        seed: Offset on the benchmark seed, for seed-spread runs.
        blackout_rate: Chance that a training crop also loses every visible cell
            in one random 90-minute block, as around Task 2 windows.
        blackout_check: Number of simulated 90-minute blackouts blanked in the
            evaluation inputs of each panel, to measure blackout robustness.
        ensemble: Number of models trained with consecutive seeds and averaged.
        low_quality_inputs: Feed cells with partial coverage (below the
            eligibility threshold) as flagged inputs. They are never targets.
        leave_day_out: Build training residuals against a profile that leaves
            out each cell's own day, as at prediction time.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    steps: int = Field(default=1000, gt=0)
    batch: int = Field(default=8, gt=0)
    window: int = Field(default=192, gt=0)
    width: int = Field(default=32, gt=0)
    lr: float = Field(default=2e-3, gt=0)
    smooth: int = Field(default=2, ge=0)
    use_eval: bool = True
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    log_every: int = Field(default=250, ge=0)
    seed: int = 0
    blackout_rate: float = Field(default=0.0, ge=0, le=1)
    blackout_check: int = Field(default=0, ge=0)
    ensemble: int = Field(default=1, gt=0)
    low_quality_inputs: bool = False
    leave_day_out: bool = False


class Pool(NamedTuple):
    """One period of grid values, its profile, first step and optional weak inputs.

    ``low_quality`` holds partial-coverage cells when ``low_quality_inputs`` is on.
    """

    values: FloatArray
    profile: FloatArray
    first_step: int
    low_quality: FloatArray | None = None


def pick_device(name: str) -> str:
    """Resolve ``auto`` to MPS when available, else CPU."""
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def hide(visible: BoolArray, rng: np.random.Generator) -> BoolArray:
    """Hide visible cells independently at one of the regime rates."""
    return visible & (rng.random(visible.shape) < rng.choice(RATES))


def score_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return 0.54 RMSE_speed/25 + 0.46 RMSE_flow_per_lane/600 over masked cells."""
    count = mask.sum().clamp(min=1)
    mse = (((prediction - target) ** 2) * mask[:, None]).sum((0, 2, 3)) / count
    return (
        0.54 * mse[0].clamp(min=1e-8).sqrt() / 25
        + 0.46 * mse[1].clamp(min=1e-8).sqrt() / 600
    )


def batches(
    pools: list[Pool],
    parameters: StCnnParameters,
    norm: Normalization,
    rng: np.random.Generator,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (inputs, residual targets, hidden mask) for random crops, forever."""
    weights = np.array([pool.values.shape[1] for pool in pools], dtype=float)
    while True:
        inputs, targets, masks = [], [], []
        for _ in range(parameters.batch):
            pool = pools[rng.choice(len(pools), p=weights / weights.sum())]
            length = min(parameters.window, pool.values.shape[1])
            start = int(rng.integers(0, pool.values.shape[1] - length + 1))
            crop = slice(start, start + length)
            values, profile = pool.values[:, crop], pool.profile[:, crop]
            visible = ~np.isnan(values[0])
            hidden = hide(visible, rng)
            if parameters.blackout_rate and rng.random() < parameters.blackout_rate:
                blackout = int(rng.integers(0, max(length - BLACKOUT_STEPS, 0) + 1))
                rows = slice(blackout, blackout + BLACKOUT_STEPS)
                hidden[rows] = visible[rows]
            weak = None if pool.low_quality is None else pool.low_quality[:, crop]
            inputs.append(
                features(
                    values,
                    visible & ~hidden,
                    profile,
                    pool.first_step + start,
                    norm,
                    weak,
                )
            )
            targets.append(np.nan_to_num(values - profile).astype(np.float32))
            masks.append(hidden.astype(np.float32))
        yield (
            torch.from_numpy(np.stack(inputs)),
            torch.from_numpy(np.stack(targets)),
            torch.from_numpy(np.stack(masks)),
        )


def fit(
    pools: list[Pool], parameters: StCnnParameters, norm: Normalization, seed: int
) -> Imputer:
    """Train an imputer on random crops of the pools and return it in eval mode."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(parameters.device)
    net = Imputer(10 if parameters.low_quality_inputs else 7, parameters.width)
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=parameters.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=parameters.lr, total_steps=max(parameters.steps, 2)
    )
    scale = torch.tensor(norm.residual, dtype=torch.float32, device=device)
    scale = scale[None, :, None, None]
    running: float | None = None
    stream = batches(pools, parameters, norm, rng)
    for step in range(parameters.steps):
        inputs, targets, masks = (tensor.to(device) for tensor in next(stream))
        loss = score_loss(net(inputs) * scale, targets, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        running = (
            loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
        )
        if parameters.log_every and (step + 1) % parameters.log_every == 0:
            print(
                f"  step {step + 1}: loss {running:.4f} (~1 - state score)", flush=True
            )
    return net.eval()


@torch.no_grad()
def predict(
    net: Imputer, pool: Pool, norm: Normalization, chunk: int = 576
) -> FloatArray:
    """Predict the full (2, T, S) grid in overlapping chunks."""
    device = next(net.parameters()).device
    pad, total = 2 * receptive_steps(), pool.values.shape[1]
    out = np.full_like(pool.values, np.nan)
    visible = ~np.isnan(pool.values[0])
    for start in range(0, total, chunk):
        low, high, end = (
            max(0, start - pad),
            min(total, start + chunk + pad),
            min(total, start + chunk),
        )
        weak = None if pool.low_quality is None else pool.low_quality[:, low:high]
        inputs = features(
            pool.values[:, low:high],
            visible[low:high],
            pool.profile[:, low:high],
            pool.first_step + low,
            norm,
            weak,
        )
        output = net(torch.from_numpy(inputs)[None].to(device))[0].cpu().numpy()
        out[:, start:end] = (
            output[:, start - low : end - low] * norm.residual[:, None, None]
        )
    return pool.profile + out


def blackout_starts(panel: str, target_steps: IntArray, count: int) -> IntArray:
    """Return reproducible first blanked steps of simulated Task 2 blackouts.

    Each blackout starts on a target day between 06:00 and 20:00 UTC and lasts
    ``BLACKOUT_STEPS``. The choice depends only on the panel name and the
    target days, so an analysis can recompute the same rows.

    Args:
        panel: Panel name, used to seed the choice.
        target_steps: Epoch steps of the requested target rows.
        count: Number of blackouts.

    Returns:
        Sorted epoch steps of each blackout's first blanked row.
    """
    rng = np.random.default_rng(zlib.crc32(panel.encode()))
    days = np.unique(target_steps // STEPS_PER_DAY)
    chosen_days = rng.choice(days, size=count, replace=count > len(days))
    slots = rng.integers(6 * 12, 20 * 12, size=count)
    return np.sort(chosen_days * STEPS_PER_DAY + slots)


def first_step_of(day: date) -> int:
    """Return the epoch step at midnight UTC of ``day``."""
    return (
        int(datetime.combine(day, datetime.min.time(), UTC).timestamp()) // STEP_SECONDS
    )


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow with one CNN trained per panel.

    Args:
        release_slice: Network tables, masked observations and historical answers.
        target_templates_by_panel_and_split: Zero-filled Task 1 rows by panel and split.
        solution_parameters: JSON settings validated as ``StCnnParameters``.

    Returns:
        One prediction table for each requested panel and split.
    """
    parameters = StCnnParameters.model_validate(solution_parameters)
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        partial(state_for_panel, parameters=parameters),
    )


def panel_observations(
    panel_data: PanelReleaseSlice, history_end: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collect the history and evaluation rows, reading only the columns used.

    History rows (before ``history_end``) get their known Task 1 answers
    restored, as ``solutions.baseline.state_observations`` does. Later rows are
    the published masked observations only.

    Args:
        panel_data: One panel's lazy release tables and historical labels.
        history_end: First epoch step after the history period.

    Returns:
        History rows and evaluation rows, each with a ``step`` column.
    """
    masked = pl.concat(
        [
            table.select(OBSERVATION_COLUMNS)
            for table in panel_data.masked_mainline_states.values()
        ]
    ).with_columns(epoch_step())
    answers = panel_data.historical_task_labels.task1_state_answers.select(
        "timestamp", *SERIES, *VALUES["state"]
    )
    history = (
        masked.filter(pl.col("step") < history_end)
        .join(answers, on=["timestamp", *SERIES], how="left", suffix="_answer")
        .with_columns(
            pl.coalesce(column, f"{column}_answer").alias(column)
            for column in VALUES["state"]
        )
        .drop([f"{column}_answer" for column in VALUES["state"]])
        .collect()
    )
    evaluation = masked.filter(pl.col("step") >= history_end).collect()
    return history, evaluation


def build_pools(
    panel: Panel,
    panel_data: PanelReleaseSlice,
    history_end: int,
    targets: list[pl.DataFrame],
    parameters: StCnnParameters,
) -> tuple[Corridor, Pool, Pool]:
    """Build the corridor, the history pool and the evaluation pool for one panel.

    The observation tables are only needed to fill the grids, so they are
    released when this function returns.

    Args:
        panel: Panel name, used to place simulated blackouts.
        panel_data: One panel's lazy release tables and historical labels.
        history_end: First epoch step after the history period.
        targets: Collected target rows of every requested split.
        parameters: Validated model settings.

    Returns:
        The corridor, the history pool and an evaluation pool covering every target.

    Raises:
        ValueError: If the history period has no observations.
    """
    history, evaluation = panel_observations(panel_data, history_end)
    if history.is_empty():
        raise ValueError("No history observations to learn from")
    corridor = build_corridor(
        panel_data.network.lwr_mainline_topology,
        panel_data.network.fd_parameters,
        [history, evaluation, *targets],
    )
    history_first, history_steps = step_range([history])
    history_grid = to_grid(history, history_first, history_steps, corridor)
    profile = Profile(history_grid, history_first, parameters.smooth)
    eval_first, eval_steps = step_range([evaluation, *targets])
    eval_grid = to_grid(evaluation, eval_first, eval_steps, corridor)
    if parameters.blackout_check:
        target_steps = pl.concat([target.select(epoch_step()) for target in targets])
        for start in blackout_starts(
            panel, target_steps["step"].to_numpy(), parameters.blackout_check
        ):
            rows = slice(start - eval_first, start - eval_first + BLACKOUT_STEPS)
            eval_grid[:, rows] = np.nan
    weak_grids = (
        (
            to_grid(history, history_first, history_steps, corridor, low_quality=True),
            to_grid(evaluation, eval_first, eval_steps, corridor, low_quality=True),
        )
        if parameters.low_quality_inputs
        else (None, None)
    )
    history_profile = (
        profile.without_own_day(history_grid, history_first)
        if parameters.leave_day_out
        else profile(history_first, history_steps)
    )
    history_pool = Pool(history_grid, history_profile, history_first, weak_grids[0])
    eval_pool = Pool(
        eval_grid, profile(eval_first, eval_steps), eval_first, weak_grids[1]
    )
    return corridor, history_pool, eval_pool


def state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    _solution_parameters: JsonObject,
    parameters: StCnnParameters,
) -> dict[Split, table_types.StateFrame]:
    """Train on one panel and fill its target rows for every requested split.

    The history period (before ``history_end_date``) gives the profile and the
    main training grid, with known Task 1 answers restored. Everything after it
    is one evaluation grid of visible cells covering all requested targets.

    Args:
        release_slice: Selected release tables, dates and historical labels.
        panel: Panel to predict.
        target_templates_by_split: Zero-filled Task 1 rows by split.
        _solution_parameters: Raw settings, already parsed into ``parameters``.
        parameters: Validated model settings.

    Returns:
        Predictions grouped by split.
    """
    targets: dict[Split, pl.DataFrame] = {
        split: frame.collect() for split, frame in target_templates_by_split.items()
    }
    corridor, history_pool, eval_pool = build_pools(
        panel,
        release_slice.panel_slices_by_panel[panel],
        first_step_of(release_slice.history_end_date),
        list(targets.values()),
        parameters,
    )
    norm = Normalization(
        residual=np.maximum(
            np.nanstd(history_pool.values - history_pool.profile, axis=(1, 2)), 1.0
        ),
        mean=np.nanmean(history_pool.values, axis=(1, 2)),
        spread=np.maximum(np.nanstd(history_pool.values, axis=(1, 2)), 1.0),
    )
    pools = [history_pool, eval_pool] if parameters.use_eval else [history_pool]
    # Average independently seeded models; one model reproduces a single fit.
    prediction = np.mean(
        [
            predict(
                fit(
                    pools,
                    parameters,
                    norm,
                    release_slice.seed + parameters.seed + index,
                ),
                eval_pool,
                norm,
            )
            for index in range(parameters.ensemble)
        ],
        axis=0,
    )

    predictions: dict[Split, table_types.StateFrame] = {}
    for split, target in targets.items():
        time_index, column = locate(
            target, eval_pool.first_step, eval_pool.values.shape[1], corridor
        )
        if (time_index < 0).any():
            raise ValueError(
                f"{panel}/{split}: target rows outside the evaluation grid"
            )
        speed = prediction[0, time_index, column]
        flow = prediction[1, time_index, column] * corridor.lanes[column]
        predictions[split] = (
            target.select(KEYS["state"])
            .with_columns(
                pl.Series("speed_kmh", np.clip(speed, 0, None)),
                pl.Series("flow_vph", np.clip(flow, 0, None)),
            )
            .lazy()
        )
    return predictions
