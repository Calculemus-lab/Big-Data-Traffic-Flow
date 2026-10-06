"""Reference predictors for state reconstruction, queue forecasting, and path-flow estimation."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse
from scipy.optimize import lsq_linear

from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    MIN_OBSERVED_PERCENT,
    QUEUE_INTERVAL_MINUTES,
    QUEUE_SPEED_FRACTION,
    VALUES,
    Panel,
    PanelReleaseSlice,
    ReleasePackageSlice,
    Split,
)
from trafficbench.metrics import link_path_incidence_matrix
from trafficbench.panelwise import map_panels


class OdmeParameters(BaseModel):
    """Validated regularization setting for the path-flow baseline.

    Attributes:
        regularization: Positive penalty weight keeping path flows near the prior.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    regularization: float = Field(default=0.05, gt=0)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow for each requested target row.

    For each target, use its link and weekly time slot when available. Fall
    back to a link-wide mean, then to the mean of all visible measurements.
    Before estimating, restore known Task 1 train answers into the published
    masked mainline tables.

    Args:
        release_slice: Network tables, masked observations, and historical labels.
        target_templates_by_panel_and_split: Zero-filled Task 1 target rows,
            grouped by panel and release split.

    Returns:
        One official-shaped prediction table for each panel and split.
    """
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        _state_for_panel,
    )


def _state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
) -> dict[Split, table_types.StateFrame]:
    """Predict one panel's state rows for each requested split.

    Args:
        release_slice: Selected release tables and settings for this run.
        panel: Panel whose historical measurements and network are used.
        target_templates_by_split: Zero-filled Task 1 target rows by split.

    Returns:
        State predictions grouped by split.
    """
    return {
        split: _state_panel(release_slice, panel, target_template)
        for split, target_template in target_templates_by_split.items()
    }


def state_observations(
    release_slice: ReleasePackageSlice,
    panel: Panel,
) -> pl.DataFrame:
    """Combine masked mainline rows with known Task 1 train answers.

    Historical answers replace masked values only at their matching station,
    link, and timestamp. Other missing measurements stay missing.

    Args:
        release_slice: Selected masked mainline tables and historical labels.
        panel: Panel whose state observations are combined.

    Returns:
        Mainline state rows with known train target values restored.
    """
    # Concatenate only released masked features, then index them by the state
    # identity shared with the historical Task 1 answer rows.
    panel_data = release_slice.panel_slices_by_panel[panel]
    observed_state_keys = ("timestamp", "station_id", "link_id")
    state_observations = pl.concat(
        [table.collect() for table in panel_data.masked_mainline_states.values()]
    )
    state_answers = panel_data.historical_task_labels.task1_state_answers.collect()
    return (
        state_observations.join(
            state_answers.select(observed_state_keys + VALUES["state"]),
            on=observed_state_keys,
            how="left",
            suffix="_answer",
        )
        .with_columns(
            pl.coalesce(column, f"{column}_answer").alias(column)
            for column in VALUES["state"]
        )
        .drop([f"{column}_answer" for column in VALUES["state"]])
    )


def _state_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_template: table_types.StateFrame,
) -> table_types.StateFrame:
    """Predict one panel's state rows from weekly speed and flow averages.

    Use each target's weekday and five-minute time slot when historical
    measurements exist. Fall back to that link's mean, then the panel mean.

    Args:
        release_slice: Selected data with requested target measurements hidden.
        panel: Panel whose network and observations are used.
        target_template: Zero-filled Task 1 target rows to predict.

    Returns:
        Requested rows with weekly-profile, link-mean, or panel-mean values.
    """
    panel_data: PanelReleaseSlice = release_slice.panel_slices_by_panel[panel]
    visible = state_observations(release_slice, panel)
    # Remove incomplete measurements before computing either weekly or global
    # averages, so missing detector periods cannot bias a target estimate.
    visible = visible.filter(
        pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
        & pl.col("speed_kmh").is_not_null()
        & pl.col("flow_vph").is_not_null()
    )
    target_frame = target_template.select(KEYS["state"]).collect()
    if visible.is_empty():
        return (
            target_frame.join(
                panel_data.network.links.select("link_id", "free_speed_kmh").collect(),
                on="link_id",
                how="left",
            )
            .with_columns(
                pl.col("free_speed_kmh").fill_null(0).alias("speed_kmh"),
                pl.lit(0.0).alias("flow_vph"),
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )

    # Index each observation by weekday and five-minute slot. The profile then
    # reuses a target's recurring weekly time pattern when one is available.
    slots_per_hour = 60 // QUEUE_INTERVAL_MINUTES
    slots_per_day = 24 * slots_per_hour
    time_slot = (
        (pl.col("timestamp").dt.weekday() - 1) * slots_per_day
        + pl.col("timestamp").dt.hour() * slots_per_hour
        + pl.col("timestamp").dt.minute() // QUEUE_INTERVAL_MINUTES
    )
    visible = visible.with_columns(time_slot.alias("slot"))
    target_frame = target_frame.with_columns(time_slot.alias("slot"))
    value_columns = VALUES["state"]
    weekly_profile = visible.group_by("link_id", "slot").agg(
        pl.col(value_columns).mean()
    )
    link_means = visible.group_by("link_id").agg(
        pl.col(value_columns).mean().name.suffix("_link")
    )
    panel_means = visible.select(pl.col(value_columns).mean().name.suffix("_panel"))
    # Fill missing profile cells first from that link's average, then from the
    # panel's overall average so every requested row receives finite values.
    predictions = (
        target_frame.join(weekly_profile, on=["link_id", "slot"], how="left")
        .join(link_means, on="link_id", how="left")
        .join(panel_means, how="cross")
        .with_columns(
            pl.coalesce(value_column, f"{value_column}_link", f"{value_column}_panel")
            .clip(lower_bound=0)
            .alias(value_column)
            for value_column in value_columns
        )
        .select(KEYS["state"] + VALUES["state"])
    )
    return predictions.lazy()


def queue(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.QueueFrame]
    ],
) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
    """Predict all requested queue windows from their latest history readings.

    For each link, apply the queue-speed threshold to its latest history speed
    when that measurement meets the required coverage. Repeat this status for
    all six forecast times. A link without a valid latest speed is predicted clear.

    Args:
        release_slice: Selected traffic and network tables for each panel.
        target_templates_by_panel_and_split: Zero-filled queue target rows,
            grouped by panel and release split.

    Returns:
        One prediction table per panel and split, with binary queue labels.
    """
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        _queue_for_panel,
    )


def _queue_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.QueueFrame],
) -> dict[Split, table_types.QueueFrame]:
    """Predict the requested queue windows for one panel.

    Args:
        release_slice: Selected release tables and network for this run.
        panel: Panel whose queue histories and free speeds are used.
        target_templates_by_split: Zero-filled queue target rows by split.

    Returns:
        Queue predictions grouped by split.
    """
    panel_data = release_slice.panel_slices_by_panel[panel]
    predictions: dict[Split, table_types.QueueFrame] = {}
    for split, target_table in target_templates_by_split.items():
        target_frame = target_table.collect()
        history = panel_data.queue_history[split].collect()
        if history.is_empty():
            predictions[split] = target_frame.with_columns(
                pl.lit(0).alias("queue_pred")
            ).lazy()
            continue
        # Evaluate links at the latest history timestamp in each window. A
        # low-coverage or missing speed at that timestamp is treated as clear.
        latest_timestamps = history.group_by("window_id").agg(
            pl.col("timestamp").max().alias("_latest_timestamp")
        )
        latest_link_status = (
            history.join(latest_timestamps, on="window_id", how="inner")
            .filter(pl.col("timestamp") == pl.col("_latest_timestamp"))
            .join(
                panel_data.network.links.select("link_id", "free_speed_kmh").collect(),
                on="link_id",
                how="left",
            )
            .group_by("window_id", "link_id")
            .agg(
                (
                    pl.col("speed_kmh").le(
                        QUEUE_SPEED_FRACTION * pl.col("free_speed_kmh")
                    )
                    & pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
                )
                .any()
                .alias("queued")
            )
        )
        predictions[split] = (
            target_frame.join(
                latest_link_status,
                on=["window_id", "link_id"],
                how="left",
            )
            .with_columns(
                pl.col("queued").fill_null(False).cast(pl.Int64).alias("queue_pred")
            )
            .select(KEYS["queue"] + VALUES["queue"])
            .lazy()
        )
    return predictions


def odme(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.OdmeFrame]
    ],
) -> dict[Panel, dict[Split, table_types.OdmeFrame]]:
    """Estimate flow on every requested path for its panel and departure period.

    Task 4 is an origin-destination matrix estimation (ODME) problem. This
    predictor balances agreement with observed link counts against distance
    from the supplied weak prior. ``release_slice.parameters['regularization']``
    controls the relative prior penalty and defaults to ``0.05``.

    Args:
        release_slice: Candidate paths, their links, observed counts, weak
            priors, and solution settings.
        target_templates_by_panel_and_split: Zero-filled Task 4 target rows,
            grouped by panel and release split.

    Returns:
        One table per panel and split with each requested path flow estimated.
    """
    regularization = OdmeParameters.model_validate(
        {"regularization": release_slice.parameters.get("regularization", 0.05)}
    ).regularization
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        partial(_odme_for_panel, regularization=regularization),
    )


def _odme_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.OdmeFrame],
    *,
    regularization: float,
) -> dict[Split, table_types.OdmeFrame]:
    """Estimate path flows for every requested split of one panel.

    Args:
        release_slice: Shared counts, priors, network tables, and settings.
        panel: Panel whose count and prior tables define the path-flow fit.
        target_templates_by_split: Zero-filled Task 4 target rows by split.
        regularization: Penalty weight for distance from the weak prior.

    Returns:
        Estimated path-flow tables grouped by split.
    """
    panel_data = release_slice.panel_slices_by_panel[panel]
    predictions: dict[Split, table_types.OdmeFrame] = {}
    for split, target_frame in target_templates_by_split.items():
        target_table = target_frame.collect()
        link_counts = panel_data.link_counts[split].collect()
        weak_prior = panel_data.weak_prior[split].collect()
        # Keep matrix rows aligned to counts and columns aligned to target paths.
        # The least-squares terms use those same positions below.
        incidence = link_path_incidence_matrix(
            panel_data.network, target_table, link_counts
        )
        prior_path_flow: NDArray[np.float64] = (
            target_table.select("path_id")
            .join(weak_prior.select("path_id", "path_flow"), on="path_id")["path_flow"]
            .to_numpy()
            .astype(np.float64)
        )
        observed_counts: NDArray[np.float64] = (
            link_counts["count"].to_numpy().astype(np.float64)
        )

        # Stack the count equations with the weak-prior penalty as one sparse
        # least-squares problem. SciPy enforces nonnegative path flows.
        prior_weight = np.sqrt(regularization)
        fit_matrix = sparse.vstack(
            [incidence, prior_weight * sparse.eye(len(prior_path_flow))],
            format="csr",
        )
        fit_values: NDArray[np.float64] = np.concatenate(
            [observed_counts, prior_weight * prior_path_flow]
        )
        optimization_result = lsq_linear(
            fit_matrix,
            fit_values,
            bounds=(0, np.inf),
            method="trf",
            tol=1e-3,
            lsmr_tol=1e-2,
            max_iter=500,
        )
        if not optimization_result.success:
            raise RuntimeError(f"ODME solve failed: {optimization_result.message}")
        # Preserve each requested target row and attach only its fitted value.
        predictions[split] = target_table.with_columns(
            pl.Series("path_flow", optimization_result.x)
        ).lazy()
    return predictions
