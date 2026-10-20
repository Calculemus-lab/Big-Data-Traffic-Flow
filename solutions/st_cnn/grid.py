"""Dense time x corridor grids, the historical profile and model inputs.

Times are integer five-minute steps since the Unix epoch, so grids, profiles
and features index time without datetime objects.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.ndimage import convolve1d

from trafficbench.contracts import MIN_OBSERVED_PERCENT

STEPS_PER_DAY = 288
STEP_SECONDS = 300
SERIES = ["station_id", "link_id"]
# 1970-01-01 was a Thursday; Monday is weekday 0.
EPOCH_WEEKDAY = 3

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]


def epoch_step() -> pl.Expr:
    """Return the ``timestamp`` column as five-minute steps since the epoch."""
    return (pl.col("timestamp").dt.epoch("s") // STEP_SECONDS).alias("step")


def weekday_and_slot(steps: IntArray) -> tuple[IntArray, IntArray]:
    """Return the UTC weekday (Monday 0) and time-of-day slot of each step."""
    return (steps // STEPS_PER_DAY + EPOCH_WEEKDAY) % 7, steps % STEPS_PER_DAY


class Corridor(NamedTuple):
    """Detector series in mainline order, one grid column per series.

    Attributes:
        series: ``station_id`` and ``link_id`` with their grid ``column``.
        lanes: Lane count of each column's link.
    """

    series: pl.DataFrame
    lanes: FloatArray


def build_corridor(
    topology: pl.LazyFrame, fd_parameters: pl.LazyFrame, frames: list[pl.DataFrame]
) -> Corridor:
    """Order every (station, link) series in the frames along the mainline.

    Neighbouring columns are then neighbouring links, which is what the
    convolution relies on. Links missing from the topology go last.

    Args:
        topology: Mainline topology with each link's ``order_index``.
        fd_parameters: Link table with each link's ``lanes``.
        frames: Tables whose ``station_id`` and ``link_id`` pairs need columns.

    Returns:
        The ordered series and their lane counts.

    Raises:
        ValueError: If a series link has no positive lane count.
    """
    order = topology.select("link_id", "order_index").collect()
    lanes = fd_parameters.select("link_id", "lanes").collect()
    series = (
        pl.concat([frame.select(SERIES) for frame in frames])
        .unique()
        .join(order, on="link_id", how="left")
        .join(lanes, on="link_id", how="left")
        .sort(["order_index", "link_id", "station_id"], nulls_last=True)
    )
    lane_counts = series["lanes"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    if np.isnan(lane_counts).any() or (lane_counts <= 0).any():
        raise ValueError("Missing or invalid lane counts")
    return Corridor(series.select(SERIES).with_row_index("column"), lane_counts)


def locate(
    frame: pl.DataFrame, first_step: int, step_count: int, corridor: Corridor
) -> tuple[IntArray, IntArray]:
    """Return each row's grid (time, column) indices, or -1 outside the grid."""
    located = frame.select(epoch_step(), *SERIES).join(
        corridor.series, on=SERIES, how="left", maintain_order="left"
    )
    time_index = located["step"].to_numpy() - first_step
    column = located["column"].cast(pl.Int64).fill_null(-1).to_numpy()
    inside = (time_index >= 0) & (time_index < step_count) & (column >= 0)
    return np.where(inside, time_index, -1), np.where(inside, column, -1)


def to_grid(
    frame: pl.DataFrame,
    first_step: int,
    step_count: int,
    corridor: Corridor,
    low_quality: bool = False,
) -> FloatArray:
    """Return (2, T, S) speed and per-lane flow, NaN where a cell is not used.

    By default a cell is used when both values are present and at least
    ``MIN_OBSERVED_PERCENT`` of the interval was observed (score-eligible
    evidence). With ``low_quality`` the grid instead holds the cells that have
    both values but partial coverage below that threshold.
    """
    grid = np.full((2, step_count, len(corridor.lanes)), np.nan)
    time_index, column = locate(frame, first_step, step_count, corridor)
    coverage = pl.col("pct_observed").fill_null(0)
    covered = (
        coverage.is_between(1, MIN_OBSERVED_PERCENT, closed="left")
        if low_quality
        else coverage.ge(MIN_OBSERVED_PERCENT)
    )
    usable = frame.select(
        covered
        & pl.col("speed_kmh").is_not_null()
        & pl.col("flow_vph").is_not_null()
        & pl.col("speed_kmh").is_not_nan()
        & pl.col("flow_vph").is_not_nan()
    ).to_series().to_numpy() & (time_index >= 0)
    rows, columns = time_index[usable], column[usable]
    grid[0, rows, columns] = frame["speed_kmh"].to_numpy()[usable]
    grid[1, rows, columns] = (
        frame["flow_vph"].to_numpy()[usable] / corridor.lanes[columns]
    )
    return grid


def step_range(frames: list[pl.DataFrame]) -> tuple[int, int]:
    """Return the first step (floored to midnight) and the step count covering the frames."""
    steps = pl.concat([frame.select(epoch_step()) for frame in frames])[
        "step"
    ].to_numpy()
    first = int(steps.min()) // STEPS_PER_DAY * STEPS_PER_DAY
    return first, int(steps.max()) - first + 1


def smoothed(week: FloatArray, half_width: int) -> FloatArray:
    """Moving sum over +-half_width slots along the cyclic week, so 23:55 borders the next day."""
    return np.sum(
        [np.roll(week, shift, axis=0) for shift in range(-half_width, half_width + 1)],
        axis=0,
    )


class Profile:
    """Day-of-week x time-of-day mean per series, smoothed over neighbouring slots.

    Empty cells fall back to the time-of-day mean, then the series mean, then
    the corridor mean of each channel.
    """

    def __init__(self, grid: FloatArray, first_step: int, half_width: int = 2) -> None:
        values = np.moveaxis(grid, 1, 0)
        observed = ~np.isnan(values)
        weekday, slot = weekday_and_slot(first_step + np.arange(values.shape[0]))
        week_shape = (7 * STEPS_PER_DAY,) + values.shape[1:]
        sums, counts = np.zeros(week_shape), np.zeros(week_shape)
        np.add.at(sums, weekday * STEPS_PER_DAY + slot, np.where(observed, values, 0))
        np.add.at(counts, weekday * STEPS_PER_DAY + slot, observed)
        by_day = (7, STEPS_PER_DAY) + values.shape[1:]
        sums = smoothed(sums, half_width).reshape(by_day)
        counts = smoothed(counts, half_width).reshape(by_day)
        with np.errstate(invalid="ignore", divide="ignore"):
            table = sums / counts
            fallbacks = [
                sums.sum(0) / counts.sum(0),
                sums.sum((0, 1)) / counts.sum((0, 1)),
                (sums.sum((0, 1, 3)) / counts.sum((0, 1, 3)))[:, None],
            ]
        for fallback in fallbacks:
            table = np.where(np.isnan(table), fallback, table)
        if np.isnan(table).any():
            raise ValueError("No usable history observations for the profile")
        self.table = table
        self.sums, self.counts, self.half_width = sums, counts, half_width

    def __call__(self, first_step: int, step_count: int) -> FloatArray:
        """Return the (2, T, S) profile for consecutive steps from ``first_step``."""
        weekday, slot = weekday_and_slot(first_step + np.arange(step_count))
        return np.moveaxis(self.table[weekday, slot], 0, 1)

    def without_own_day(self, grid: FloatArray, first_step: int) -> FloatArray:
        """Return the profile at every cell of the fitting grid, leaving out its own day.

        At prediction time the predicted day never contributes to the profile,
        so training residuals should not either. ``grid`` must be the grid the
        profile was fitted on, starting at midnight. Cells whose weekday slot has
        no other observations keep the ordinary profile value.

        Args:
            grid: The (2, T, S) history grid used to fit this profile.
            first_step: Epoch step of the grid's first row.

        Returns:
            A (2, T, S) profile with each cell's own calendar day removed.
        """
        channels, step_count, series = grid.shape
        day_count = -(-step_count // STEPS_PER_DAY)
        padded = np.full((channels, day_count * STEPS_PER_DAY, series), np.nan)
        padded[:, :step_count] = grid
        days = padded.reshape(channels, day_count, STEPS_PER_DAY, series)
        observed = ~np.isnan(days)
        # A day's own share of the smoothed table: its +-half_width neighbours
        # on the same calendar day (slots past midnight belong to other days).
        window = np.ones(2 * self.half_width + 1)
        own_sums = convolve1d(
            np.where(observed, days, 0), window, axis=2, mode="constant"
        )
        own_counts = convolve1d(observed.astype(float), window, axis=2, mode="constant")
        weekday, slot = weekday_and_slot(
            first_step + np.arange(day_count * STEPS_PER_DAY)
        )
        sums = np.moveaxis(self.sums[weekday, slot], 0, 1).reshape(days.shape)
        counts = np.moveaxis(self.counts[weekday, slot], 0, 1).reshape(days.shape)
        remaining = counts - own_counts
        with np.errstate(invalid="ignore", divide="ignore"):
            left_out = (sums - own_sums) / remaining
        ordinary = self(first_step, day_count * STEPS_PER_DAY).reshape(days.shape)
        profile = np.where(remaining > 0, left_out, ordinary)
        return profile.reshape(channels, -1, series)[:, :step_count]


class Normalization(NamedTuple):
    """Per-channel scales for model inputs and outputs.

    Attributes:
        residual: Spread of observed values around the profile.
        mean: Mean observed level, used to centre the profile input.
        spread: Spread of observed levels, used to scale the profile input.
        density_residual: Spread of per-lane density around the profile density,
            used to scale the density output.
    """

    residual: FloatArray
    mean: FloatArray
    spread: FloatArray
    density_residual: float = 1.0


def features(
    values: FloatArray,
    visible: BoolArray,
    profile: FloatArray,
    first_step: int,
    norm: Normalization,
    low_quality: FloatArray | None = None,
) -> NDArray[np.float32]:
    """Return model inputs for one (T, S) window.

    Seven channels: masked residuals (2), visibility, profile level (2) and
    time of day (2). With ``low_quality`` values, three more channels follow:
    the residuals of partial-coverage cells (2) and their presence flag.
    """
    residual = np.where(visible, values - profile, 0) / norm.residual[:, None, None]
    _, slot = weekday_and_slot(first_step + np.arange(visible.shape[0]))
    angle = 2 * np.pi * slot / STEPS_PER_DAY
    clock = np.broadcast_to(
        np.stack([np.sin(angle), np.cos(angle)])[:, :, None], (2,) + visible.shape
    )
    level = (profile - norm.mean[:, None, None]) / norm.spread[:, None, None]
    channels = [residual, visible[None], level, clock]
    if low_quality is not None:
        present = ~np.isnan(low_quality[0])
        weak = (
            np.where(present, low_quality - profile, 0) / norm.residual[:, None, None]
        )
        channels += [weak, present[None]]
    return np.concatenate(channels).astype(np.float32)
