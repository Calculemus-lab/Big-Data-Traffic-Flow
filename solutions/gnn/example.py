"""Cut model inputs and answers out of the link tables."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .grids import (
    FORECAST_STEPS,
    HISTORY_STEPS,
    MIN_FORECAST_COVERAGE,
    MIN_HISTORY_COVERAGE,
    STEPS_PER_DAY,
    LinkGrids,
    queued_cells,
    usual_queue_rate,
)


class Series(NamedTuple):
    """Everything the examples are cut from. Rows are five-minute steps."""

    speed_ratio: NDArray[np.float64]  # speed / free-flow speed, 0 where unknown
    density_ratio: NDArray[np.float64]
    known: NDArray[np.bool_]
    queued: NDArray[np.bool_]
    usual: NDArray[np.float64]  # usual queue rate, (288, links)
    first_weekday: int  # weekday of row 0, Monday = 0



def density_scale(grids: LinkGrids) -> NDArray[np.float64]:
    """95th percentile of each link's known density in training; 1 where none is known."""
    known_density = np.where(grids.known, grids.density_veh_per_km, np.nan)
    scale = np.nanpercentile(known_density, 95, axis=0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def build_series(
    grids: LinkGrids, free_speed_kmh: NDArray[np.float64], scale: NDArray[np.float64]
) -> Series:
    queued = queued_cells(grids, free_speed_kmh)
    return Series(
        speed_ratio=np.nan_to_num(grids.speed_kmh / free_speed_kmh),
        density_ratio=np.nan_to_num(grids.density_veh_per_km / scale),
        known=grids.known,
        queued=queued,
        usual=usual_queue_rate(queued, grids.known),
        first_weekday=grids.start.weekday(),
    )


def example_inputs(series: Series, origins: NDArray[np.int64]) -> NDArray[np.float32]:
    """Model inputs with shape (examples, links, 45).

    Row ``t`` is the unobserved forecast origin. The history is rows t-12 .. t-1.
    """
    history = origins[:, None] + np.arange(-HISTORY_STEPS, 0)  # (examples, 12)
    recent = np.stack(
        [series.speed_ratio[history], series.density_ratio[history], series.known[history]],
        axis=-1,
    )  # (examples, 12, links, 3)
    link_count = recent.shape[2]
    recent = recent.transpose(0, 2, 1, 3).reshape(len(origins), link_count, -1)

    angle = 2 * np.pi * (origins % STEPS_PER_DAY) / STEPS_PER_DAY
    weekend = (series.first_weekday + origins // STEPS_PER_DAY) % 7 >= 5
    clock = np.stack([np.sin(angle), np.cos(angle), weekend], axis=-1)  # (examples, 3)
    clock = np.repeat(clock[:, None, :], link_count, axis=1)

    future_slots = (origins[:, None] + np.arange(1, FORECAST_STEPS + 1)) % STEPS_PER_DAY
    usual = series.usual[future_slots].transpose(0, 2, 1)  # (examples, links, 6)
    return np.concatenate([recent, clock, usual], axis=-1).astype(np.float32)


def example_answers(
    series: Series, origins: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Queued answers and their known-mask, both (examples, links, 6)."""
    future = origins[:, None] + np.arange(1, FORECAST_STEPS + 1)  # rows t+1 .. t+6
    queued = series.queued[future].transpose(0, 2, 1)
    known = series.known[future].transpose(0, 2, 1)
    return queued.astype(np.float32), known.astype(np.float32)


def selected_origins(series: Series) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    zero_row = np.zeros((1, series.queued.shape[1]), dtype=np.int32)
    running_total = np.vstack(
        [zero_row, np.cumsum(series.queued, axis=0, dtype=np.int32)]
    )
    origins = np.arange(HISTORY_STEPS, series.queued.shape[0] - FORECAST_STEPS)
    history_steps = running_total[origins] - running_total[origins - HISTORY_STEPS]
    forecast_steps = (
        running_total[origins + FORECAST_STEPS + 1] - running_total[origins + 1]
    )
    known_total = np.vstack(
        [zero_row, np.cumsum(series.known, axis=0, dtype=np.int32)]
    )
    link_count = series.known.shape[1]
    history_coverage = (
                               known_total[origins] - known_total[origins - HISTORY_STEPS]
                       ).sum(axis=1) / (HISTORY_STEPS * link_count)
    forecast_coverage = (
                                known_total[origins + FORECAST_STEPS + 1] - known_total[origins + 1]
                        ).sum(axis=1) / (FORECAST_STEPS * link_count)
    qualifies = (
            (forecast_steps > 0).any(axis=1)
            & (history_coverage >= MIN_HISTORY_COVERAGE)
            & (forecast_coverage >= MIN_FORECAST_COVERAGE)
    )
    ongoing = (history_steps >= 2).any(axis=1)
    return origins[qualifies], ongoing[qualifies]
