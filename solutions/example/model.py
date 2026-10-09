"""Example state model using visible measurements near masked targets.

Try: ``bench run example --task state --target-range validation``
Tune: ``bench run example --task state --params '{"blend": 0.75}' ...``
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from solutions import baseline
from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    MIN_OBSERVED_PERCENT,
    VALUES,
    JsonObject,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels


class StateParameters(BaseModel):
    """Validated settings for the example state predictor.

    Attributes:
        blend: Weight assigned to interpolated visible measurements, from zero
            (baseline only) through one (interpolated measurements only).
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    blend: float = Field(default=0.5, ge=0, le=1)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Blend baseline estimates with nearby visible state measurements.

    ``blend`` controls the interpolation weight. Missing and low-coverage
    observations are excluded before interpolation.

    Args:
        release_slice: Network data, masked observations, and historical labels.
        target_templates_by_panel_and_split: Zero-filled Task 1 target rows,
            grouped by panel and release split.
        solution_parameters: JSON settings for this state predictor.

    Returns:
        One official-shaped prediction table for each panel and split.
    """
    blend = StateParameters.model_validate(solution_parameters).blend
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        partial(_blend_panel, blend=blend),
    )


def _blend_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    solution_parameters: JsonObject,
    *,
    blend: float,
) -> dict[Split, table_types.StateFrame]:
    """Blend baseline predictions with interpolated measurements for one panel.

    Args:
        release_slice: Selected release tables and historical labels.
        panel: Panel whose target rows and historical measurements are used.
        target_templates_by_split: Zero-filled Task 1 target rows by split.
        solution_parameters: JSON settings passed to the baseline predictor.
        blend: Weight assigned to the interpolated measurements.

    Returns:
        Blended state predictions grouped by split.
    """
    # The baseline supplies finite predictions for every requested target row. The
    # visible-series interpolation is then blended where matching rows exist.
    predictions_by_split = baseline.state(
        release_slice,
        {panel: dict(target_templates_by_split)},
        solution_parameters,
    )[panel]
    blended_predictions: dict[Split, table_types.StateFrame] = {}
    # Interpolate within each station-link series so measurements from another
    # detector or road link cannot influence a target value.
    visible = baseline.state_observations(release_slice, panel).sort(
        ["station_id", "link_id", "timestamp"]
    )
    state_columns = VALUES["state"]
    visible = visible.with_columns(
        pl.when(pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT))
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in state_columns
    ).with_columns(
        pl.col(column)
        .interpolate()
        .forward_fill()
        .backward_fill()
        .over(["station_id", "link_id"])
        .alias(column)
        for column in state_columns
    )
    # Join by the full detector/link/time identity so each target receives only
    # the measurement series from the same station and road link.
    target_keys = ["timestamp", "station_id", "link_id"]
    for split, target_table in target_templates_by_split.items():
        prediction = predictions_by_split[split].collect()
        target_rows = target_table.select(target_keys).collect()
        nearby_estimates = target_rows.join(
            visible.select([*target_keys, *state_columns]),
            on=target_keys,
            how="left",
        )
        # Keep the baseline value if the selected observations do not cover
        # this target's station, link, and timestamp.
        blended_predictions[split] = (
            prediction.join(
                nearby_estimates, on=target_keys, how="left", suffix="_nearby"
            )
            .with_columns(
                (
                    (1 - blend) * pl.col(column)
                    + blend * pl.col(f"{column}_nearby").fill_null(pl.col(column))
                ).alias(column)
                for column in state_columns
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )
    return blended_predictions
