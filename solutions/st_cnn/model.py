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
from .net import CONTEXTS, Imputer, receptive_steps

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
        conservation_weight: Weight of the conservation term added to the score
            loss (see ``conservation_loss``); 0 trains on the score alone.
        output: ``flow`` predicts speed and per-lane flow residuals; ``density``
            predicts speed and per-lane density residuals and derives flow as
            density times speed.
        context: Dilation set of the network (``base`` or the longer ``wide``).
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
    conservation_weight: float = Field(default=0.0, ge=0)
    output: Literal["flow", "density"] = "flow"
    context: Literal["base", "wide"] = "base"


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


def physical_residuals(
    raw: torch.Tensor,
    profile: torch.Tensor,
    norm: Normalization,
    output: Literal["flow", "density"],
) -> torch.Tensor:
    """Convert network outputs to speed and per-lane flow residuals in physical units.

    Args:
        raw: Network outputs, (batch, 2, T, S), normalized.
        profile: Profile speed and per-lane flow of the same cells.
        norm: Output scales.
        output: Whether the second channel is a flow or a density residual.

    Returns:
        Residuals over the profile for speed and per-lane flow.
    """
    scale = torch.tensor(norm.residual, dtype=raw.dtype, device=raw.device)
    scale = scale[None, :, None, None]
    if output == "flow":
        return raw * scale
    speed_residual = raw[:, :1] * scale[:, :1]
    speed = profile[:, :1] + speed_residual
    profile_density = profile[:, 1:] / profile[:, :1].clamp(min=1)
    density = profile_density + raw[:, 1:] * norm.density_residual
    flow_residual = density * speed.clamp(min=1) - profile[:, 1:]
    return torch.cat([speed_residual, flow_residual], dim=1)


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


class Batch(NamedTuple):
    """One training batch of crops.

    Attributes:
        inputs: Model inputs, (batch, channels, T, S).
        targets: True residuals over the profile, zero where not visible.
        hidden: Cells hidden from the inputs and scored, (batch, T, S).
        profile: Profile speed and per-lane flow of the crops.
        visible: Cells with usable values before hiding, (batch, T, S).
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    hidden: torch.Tensor
    profile: torch.Tensor
    visible: torch.Tensor


def conservation_loss(
    residual_prediction: torch.Tensor, batch: Batch, vehicles_per_flow: torch.Tensor
) -> torch.Tensor:
    """Return 1 - S_LWR of the hidden cells, as the Task 3 conservation term scores it.

    The vehicle count is ``N = flow_per_lane * lanes * length / max(speed, 1)``. The
    evaluator's boundary flows make the true state conserve vehicles exactly, so the
    residual of a five-minute transition is the error in the change of ``N``. Visible
    cells keep their true values; transitions count when both cells had values and
    at least one is hidden.

    Args:
        residual_prediction: Predicted residuals in physical units, like ``targets``.
        batch: The batch the prediction was made for.
        vehicles_per_flow: ``lanes * length_km`` of each grid column.

    Returns:
        Sum of absolute errors in ``N`` changes over the sum of true ``N`` changes.
    """
    true_speed = batch.profile[:, 0] + batch.targets[:, 0]
    true_flow = batch.profile[:, 1] + batch.targets[:, 1]
    predicted_speed = batch.profile[:, 0] + residual_prediction[:, 0]
    predicted_flow = batch.profile[:, 1] + residual_prediction[:, 1]
    true_vehicles = true_flow / true_speed.clamp(min=1) * vehicles_per_flow
    vehicle_error = (
        predicted_flow / predicted_speed.clamp(min=1) * vehicles_per_flow
        - true_vehicles
    ) * batch.hidden
    transitions = (
        batch.visible[:, 1:]
        * batch.visible[:, :-1]
        * (batch.hidden[:, 1:] + batch.hidden[:, :-1]).clamp(max=1)
    )
    residual = (vehicle_error[:, 1:] - vehicle_error[:, :-1]).abs() * transitions
    signal = (true_vehicles[:, 1:] - true_vehicles[:, :-1]).abs() * transitions
    return residual.sum() / signal.sum().clamp(min=1e-6)


def batches(
    pools: list[Pool],
    parameters: StCnnParameters,
    norm: Normalization,
    rng: np.random.Generator,
) -> Iterator[Batch]:
    """Yield batches of random crops with visible cells hidden, forever."""
    weights = np.array([pool.values.shape[1] for pool in pools], dtype=float)
    while True:
        inputs, targets, masks, profiles, visibles = [], [], [], [], []
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
            profiles.append(profile.astype(np.float32))
            visibles.append(visible.astype(np.float32))
        yield Batch(
            *(
                torch.from_numpy(np.stack(arrays))
                for arrays in (inputs, targets, masks, profiles, visibles)
            )
        )


def fit(
    pools: list[Pool],
    parameters: StCnnParameters,
    norm: Normalization,
    seed: int,
    vehicles_per_flow: FloatArray,
) -> Imputer:
    """Train an imputer on random crops of the pools and return it in eval mode.

    ``vehicles_per_flow`` (``lanes * length_km`` per grid column) is only used by
    the conservation term.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(parameters.device)
    net = Imputer(
        10 if parameters.low_quality_inputs else 7,
        parameters.width,
        CONTEXTS[parameters.context],
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=parameters.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=parameters.lr, total_steps=max(parameters.steps, 2)
    )
    column_factor = torch.tensor(vehicles_per_flow, dtype=torch.float32, device=device)
    running: float | None = None
    stream = batches(pools, parameters, norm, rng)
    for step in range(parameters.steps):
        batch = Batch(*(tensor.to(device) for tensor in next(stream)))
        residual_prediction = physical_residuals(
            net(batch.inputs), batch.profile, norm, parameters.output
        )
        loss = score_loss(residual_prediction, batch.targets, batch.hidden)
        if parameters.conservation_weight:
            loss = loss + parameters.conservation_weight * conservation_loss(
                residual_prediction, batch, column_factor
            )
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
    net: Imputer,
    pool: Pool,
    norm: Normalization,
    chunk: int = 576,
    output: Literal["flow", "density"] = "flow",
) -> FloatArray:
    """Predict the full (2, T, S) grid in overlapping chunks."""
    device = next(net.parameters()).device
    pad, total = 2 * receptive_steps(net.dilations), pool.values.shape[1]
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
        profile = torch.from_numpy(pool.profile[:, low:high].astype(np.float32))
        residuals = (
            physical_residuals(
                net(torch.from_numpy(inputs)[None].to(device)),
                profile[None].to(device),
                norm,
                output,
            )[0]
            .cpu()
            .numpy()
        )
        out[:, start:end] = residuals[:, start - low : end - low]
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
    density, profile_density = (
        grid[1] / np.maximum(grid[0], 1)
        for grid in (history_pool.values, history_pool.profile)
    )
    norm = Normalization(
        residual=np.maximum(
            np.nanstd(history_pool.values - history_pool.profile, axis=(1, 2)), 1.0
        ),
        mean=np.nanmean(history_pool.values, axis=(1, 2)),
        spread=np.maximum(np.nanstd(history_pool.values, axis=(1, 2)), 1.0),
        density_residual=max(float(np.nanstd(density - profile_density)), 1e-3),
    )
    pools = [history_pool, eval_pool] if parameters.use_eval else [history_pool]
    lengths = corridor.series.join(
        release_slice.panel_slices_by_panel[panel]
        .network.fd_parameters.select("link_id", "length_km")
        .collect(),
        on="link_id",
        how="left",
        maintain_order="left",
    )["length_km"].to_numpy()
    vehicles_per_flow = corridor.lanes * lengths
    # Average independently seeded models; one model reproduces a single fit.
    prediction = np.mean(
        [
            predict(
                fit(
                    pools,
                    parameters,
                    norm,
                    release_slice.seed + parameters.seed + index,
                    vehicles_per_flow,
                ),
                eval_pool,
                norm,
                output=parameters.output,
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
