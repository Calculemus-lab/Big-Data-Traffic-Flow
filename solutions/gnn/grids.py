"""Turn masked link measurements into time-by-link arrays for the GCN."""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

STEP_MINUTES = 5
MIN_COVERAGE_PERCENT = 75
QUEUE_FRACTION = 0.60
STEPS_PER_DAY = 288
HISTORY_STEPS = 12
FORECAST_STEPS = 6


class LinkGrids(NamedTuple):
    """Link measurements on a regular five-minute grid; rows are time steps."""

    start: datetime
    speed_kmh: NDArray[np.float64]  # NaN where unknown
    occupancy: NDArray[np.float64]  # NaN where unknown
    known: NDArray[np.bool_]  # True where a queue label can be derived


def link_grids(masked_states: pl.LazyFrame, link_ids: list[str]) -> LinkGrids:
    """Average station rows to one value per link and timestamp."""
    per_link = (
        masked_states.group_by("timestamp", "link_id")
        .agg(
            pl.col("speed_kmh").mean().alias("speed_kmh"),
            pl.col("occupancy").mean().alias("occupancy"),
            (
                pl.col("speed_kmh").is_not_null()
                & (pl.col("pct_observed") >= MIN_COVERAGE_PERCENT)
            )
            .fill_null(False)
            .all()
            .alias("known"),
        )
        .collect()
    )
    start = per_link["timestamp"].min()
    last = per_link["timestamp"].max()
    step_count = int((last - start).total_seconds() // (60 * STEP_MINUTES)) + 1

    link_index = {link_id: index for index, link_id in enumerate(link_ids)}
    per_link = per_link.filter(pl.col("link_id").is_in(link_ids)).with_columns(
        ((pl.col("timestamp") - start).dt.total_minutes() // STEP_MINUTES).alias("row"),
        pl.col("link_id")
        .replace_strict(link_index, return_dtype=pl.Int64)
        .alias("col"),
    )
    rows, columns = per_link["row"].to_numpy(), per_link["col"].to_numpy()

    speed = np.full((step_count, len(link_ids)), np.nan)
    occupancy = np.full_like(speed, np.nan)
    known = np.zeros(speed.shape, dtype=bool)
    speed[rows, columns] = per_link["speed_kmh"].to_numpy()
    occupancy[rows, columns] = per_link["occupancy"].to_numpy()
    known[rows, columns] = per_link["known"].to_numpy()
    return LinkGrids(start, speed, occupancy, known)


def queued_cells(
    grids: LinkGrids, free_speed_kmh: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """True where the link is queued; unknown cells are never counted as queued."""
    return grids.known & (grids.speed_kmh <= QUEUE_FRACTION * free_speed_kmh)


def usual_queue_rate(
    queued: NDArray[np.bool_], known: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Share of known cells that were queued, per time of day (rows) and link (columns)."""
    slots = np.arange(queued.shape[0]) % STEPS_PER_DAY
    queued_count = np.zeros((STEPS_PER_DAY, queued.shape[1]))
    known_count = np.zeros_like(queued_count)
    np.add.at(queued_count, slots, queued)
    np.add.at(known_count, slots, known)
    return queued_count / np.maximum(known_count, 1)
