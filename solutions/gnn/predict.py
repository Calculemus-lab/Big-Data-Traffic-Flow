"""Predict the official queue windows with a trained panel model."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import torch

from .example import Series, example_inputs
from .fit import PanelModel
from .grids import FORECAST_STEPS, STEPS_PER_DAY, link_grids

STEP = timedelta(minutes=5)


def predict_split(
    fitted: PanelModel,
    history: pl.LazyFrame,
    window_index: pl.LazyFrame,
    target_template: pl.LazyFrame,
    threshold: float,
    onset_zero_steps: int = 0,
    onset_last_step_queue: bool = False,
) -> pl.LazyFrame:
    target = target_template.collect()
    windows = window_index.select("window_id", "forecast_origin", "condition").collect()
    if windows.is_empty() or history.limit(1).collect().is_empty():
        return target.with_columns(pl.lit(0).alias("queue_pred")).lazy()

    # 1. One grid from the history rows of all windows (same code as training).
    grids = link_grids(history, fitted.link_ids)
    start = grids.start.replace(tzinfo=None)

    # 2. The clock features expect row 0 to be midnight, so pad the top of the
    #    grid with empty rows up to the first row's time of day.
    offset = (start.hour * 60 + start.minute) // 5

    def pad(values):
        return np.vstack([np.zeros((offset, values.shape[1]), values.dtype), values])

    known = pad(grids.known)
    series = Series(
        speed_ratio=pad(np.nan_to_num(grids.speed_kmh / fitted.free_speed_kmh)),
        occupancy=pad(np.nan_to_num(grids.occupancy)),
        known=known,
        queued=np.zeros_like(known),  # not needed to build inputs
        usual=fitted.usual,
        first_weekday=start.weekday(),
    )

    # 3. The grid row of each window's forecast origin.
    window_ids = windows["window_id"].to_list()
    origin_rows = np.array(
        [
            (origin.replace(tzinfo=None) - start) // STEP + offset
            for origin in windows["forecast_origin"]
        ],
        dtype=np.int64,
    )

    # 4. Same inputs as in training, then the model.
    inputs = torch.from_numpy(example_inputs(series, origin_rows))
    device = next(fitted.model.parameters()).device
    with torch.no_grad():
        probability = torch.sigmoid(fitted.model(inputs.to(device))).cpu().numpy()
    # probability: (windows, links, 6)

    # 5. Lay the probabilities out as a table and join them onto the target rows.
    window_count, link_count = len(window_ids), len(fitted.link_ids)
    table = pl.DataFrame(
        {
            "window_id": np.repeat(window_ids, link_count * FORECAST_STEPS),
            "link_id": np.tile(
                np.repeat(fitted.link_ids, FORECAST_STEPS), window_count
            ),
            "step": np.tile(
                np.arange(1, FORECAST_STEPS + 1), window_count * link_count
            ),
            "probability": probability.reshape(-1),
        }
    )
    return (
        target.join(windows, on="window_id", how="left")
        .with_columns(
            ((pl.col("timestamp") - pl.col("forecast_origin")).dt.total_minutes() // 5)
            .cast(pl.Int64)
            .alias("step")
        )
        .join(table, on=["window_id", "link_id", "step"], how="left")
        .with_columns(
            pl.when(
                (pl.col("condition") == "queue_onset")
                & (pl.col("step") <= onset_zero_steps)
            )
            .then(0)
            .when(
                (pl.col("condition") == "queue_onset")
                & (pl.col("step") == FORECAST_STEPS)
                & onset_last_step_queue
            )
            .then(1)
            .otherwise(
                (pl.col("probability") >= threshold).fill_null(False).cast(pl.Int64)
            )
            .alias("queue_pred")
        )
        .select("window_id", "timestamp", "link_id", "queue_pred")
        .lazy()
    )
