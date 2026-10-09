"""Build causal traffic-history and network features for queue forecasting.

Each forecast uses the official 60-minute history and summarizes the three hours
before that history. Missing or low-coverage evidence remains missing, rather
than becoming a clear road or a measured zero. Training and inference share the
same feature builder.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import sparse

from trafficbench.contracts import (
    MIN_OBSERVED_PERCENT,
    QUEUE_HISTORY_MINUTES,
    QUEUE_INTERVAL_MINUTES,
    QUEUE_SPEED_FRACTION,
    NetworkTables,
)

type FloatArray = NDArray[np.float32]
type IndexArray = NDArray[np.int64]

HISTORY_STEPS = QUEUE_HISTORY_MINUTES // QUEUE_INTERVAL_MINUTES
CONTEXT_MINUTES = 180
CONTEXT_STEPS = CONTEXT_MINUTES // QUEUE_INTERVAL_MINUTES
CONTEXT_LOOKBACK_STEPS = HISTORY_STEPS + CONTEXT_STEPS
MEASUREMENT_COLUMNS = ("speed_kmh", "flow_vph", "occupancy")
CATEGORICAL_FEATURES = ("link_index", "day_of_week")


class QueueNetwork(NamedTuple):
    """Ordered link properties, directed adjacency, and ramp attachments."""

    link_ids: tuple[str, ...]
    free_speeds: FloatArray
    capacities: FloatArray
    static_features: dict[str, FloatArray]
    upstream: sparse.csr_matrix
    downstream: sparse.csr_matrix
    ramp_attachments: pl.DataFrame


class TrafficGrid(NamedTuple):
    """Regular time-by-link measurements; unavailable measurements are NaN."""

    timestamps: tuple[datetime, ...]
    speed_ratio: FloatArray
    flow_ratio: FloatArray
    occupancy: FloatArray
    on_ramp_ratio: FloatArray
    off_ramp_ratio: FloatArray


class FeatureMatrix(NamedTuple):
    """Origin-major, then link-major feature rows and their column names."""

    values: FloatArray
    names: tuple[str, ...]


def queue_network(network: NetworkTables) -> QueueNetwork:
    """Order network links and build adjacency restricted to forecast links.

    Args:
        network: Released link properties, mainline topology, and ramp map.

    Returns:
        Static link features and upstream/downstream averaging operators.
    """
    links = network.links.collect().sort("link_id")
    link_ids = tuple(links["link_id"].to_list())
    topology = network.lwr_mainline_topology.collect()
    positions = {link_id: index for index, link_id in enumerate(link_ids)}
    sources: list[int] = []
    destinations: list[int] = []
    # Connector links outside the forecast network do not become model nodes.
    for link_id, outgoing in topology.select(
        "link_id", "outgoing_link_ids"
    ).iter_rows():
        if link_id not in positions or not outgoing:
            continue
        for destination in outgoing.split(";"):
            if destination in positions:
                sources.append(positions[link_id])
                destinations.append(positions[destination])
    downstream = sparse.csr_matrix(
        (np.ones(len(sources), dtype=np.float32), (sources, destinations)),
        shape=(len(link_ids), len(link_ids)),
    )
    static_features = {
        column: links[column].to_numpy().astype(np.float32)
        for column in ("length_km", "lanes", "free_speed_kmh", "capacity_vph")
    }
    ordered_links = links.select("link_id").join(
        topology.select("link_id", "order_index"), on="link_id", how="left"
    )
    static_features["position"] = ordered_links["order_index"].to_numpy().astype(
        np.float32
    ) / np.float32(max(len(link_ids) - 1, 1))
    return QueueNetwork(
        link_ids,
        static_features["free_speed_kmh"],
        static_features["capacity_vph"],
        static_features,
        downstream.T.tocsr(),
        downstream,
        network.ramp_attachment_map.collect(),
    )


def traffic_grid(
    states: pl.DataFrame,
    ramps: pl.DataFrame,
    timestamps: Sequence[datetime],
    network: QueueNetwork,
) -> TrafficGrid:
    """Align station or queue-history measurements to a complete link grid.

    Args:
        states: Rows with timestamp, link ID, speed, flow, occupancy, and coverage.
        ramps: Published ramp rows restricted to the same permitted history.
        timestamps: Complete, increasing five-minute grid for these observations.
        network: Ordered forecast links and their measurement normalizers.

    Returns:
        Coverage-filtered measurements and ramp flows on the supplied grid.
    """
    timestamps = tuple(timestamps)
    timestamp_positions = {
        timestamp: index for index, timestamp in enumerate(timestamps)
    }
    link_positions = {link_id: index for index, link_id in enumerate(network.link_ids)}
    shape = (len(timestamps), len(network.link_ids))
    measurements = {
        column: np.full(shape, np.nan, dtype=np.float32)
        for column in MEASUREMENT_COLUMNS
    }
    # A link label requires complete station speeds and sufficient coverage.
    grouped = (
        states.group_by("timestamp", "link_id")
        .agg(
            pl.col("pct_observed")
            .ge(MIN_OBSERVED_PERCENT)
            .fill_null(False)
            .all()
            .alias("covered"),
            *[
                pl.when(pl.col(column).is_finite().fill_null(False).all())
                .then(pl.col(column).mean())
                .otherwise(None)
                .alias(column)
                for column in MEASUREMENT_COLUMNS
            ],
        )
        .filter(pl.col("covered"))
    )
    # Vectorized indexing avoids millions of Python row operations on full history.
    interval_microseconds = QUEUE_INTERVAL_MINUTES * 60 * 1_000_000
    first_timestamp = pl.Series([timestamps[0]]).dt.epoch("us").item()
    elapsed = pl.col("timestamp").dt.epoch("us") - first_timestamp
    indexed = grouped.with_columns(
        (elapsed // interval_microseconds).alias("_time_index"),
        pl.col("link_id")
        .replace_strict(link_positions, default=None, return_dtype=pl.Int64)
        .alias("_link_index"),
    ).filter(
        pl.col("_time_index").is_between(0, len(timestamps), closed="left")
        & pl.col("_link_index").is_not_null()
        & elapsed.mod(interval_microseconds).eq(0)
    )
    rows = indexed["_time_index"].to_numpy()
    columns = indexed["_link_index"].to_numpy()
    for name in MEASUREMENT_COLUMNS:
        measurements[name][rows, columns] = indexed[name].to_numpy()
    # Normalize the dynamic values using the same link properties as queue labels.
    speed_ratio = measurements["speed_kmh"] / network.free_speeds
    flow_ratio = measurements["flow_vph"] / network.capacities
    on_ramp_ratio, off_ramp_ratio = _ramp_grids(
        ramps, network, timestamp_positions, link_positions
    )
    return TrafficGrid(
        timestamps,
        speed_ratio,
        flow_ratio,
        measurements["occupancy"],
        on_ramp_ratio,
        off_ramp_ratio,
    )


def _ramp_grids(
    ramps: pl.DataFrame,
    network: QueueNetwork,
    timestamp_positions: dict[datetime, int],
    link_positions: dict[str, int],
) -> tuple[FloatArray, FloatArray]:
    """Sum observed attached ramps; distinguish missing ramps from no ramps.

    Args:
        ramps: Released history with ramp IDs, types, flow, and coverage.
        network: Attachments and mainline flow capacities.
        timestamp_positions: Row index for each permitted history timestamp.
        link_positions: Column index for each forecast link.

    Returns:
        On-ramp and off-ramp flow divided by mainline capacity.
    """
    shape = (len(timestamp_positions), len(link_positions))
    grids = {ramp_type: np.zeros(shape, dtype=np.float32) for ramp_type in ("OR", "FR")}
    attachments = network.ramp_attachments
    counts = attachments.group_by("nearest_mainline_link_id", "ramp_type").len(
        name="expected"
    )
    for link_id, ramp_type, _ in counts.iter_rows():
        if link_id in link_positions:
            grids[ramp_type][:, link_positions[link_id]] = np.nan
    if ramps.is_empty() or attachments.is_empty():
        return grids["OR"], grids["FR"]
    # Every attached ramp must be present and observed before its sum is evidence.
    joined = ramps.join(
        attachments.select("ramp_link_id", "nearest_mainline_link_id"),
        on="ramp_link_id",
        how="inner",
    )
    grouped = (
        joined.group_by("timestamp", "nearest_mainline_link_id", "ramp_type")
        .agg(
            pl.col("ramp_link_id").n_unique().alias("present"),
            (
                pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
                & pl.col("flow_vph").is_finite()
            )
            .fill_null(False)
            .all()
            .alias("covered"),
            pl.col("flow_vph").sum().alias("flow_vph"),
        )
        .join(counts, on=["nearest_mainline_link_id", "ramp_type"])
        .filter(pl.col("covered") & pl.col("present").eq(pl.col("expected")))
    )
    for timestamp, link_id, ramp_type, _, _, flow, _ in grouped.iter_rows():
        row, column = timestamp_positions.get(timestamp), link_positions.get(link_id)
        if row is not None and column is not None:
            grids[ramp_type][row, column] = flow / network.capacities[column]
    return grids["OR"], grids["FR"]


def _mean(values: FloatArray, axis: int) -> FloatArray:
    """Average finite measurements, returning NaN without available evidence.

    Args:
        values: Measurement array that may contain unavailable values.
        axis: Dimension reduced by the average.

    Returns:
        Float32 averages with the selected dimension removed.
    """
    present = np.isfinite(values)
    counts = present.sum(axis=axis)
    return np.divide(
        np.where(present, values, 0).sum(axis=axis),
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float32),
        where=counts > 0,
    )


def _neighbor_mean(values: FloatArray, adjacency: sparse.csr_matrix) -> FloatArray:
    """Average available directed neighbours without treating missing links as zero.

    Args:
        values: Time-by-link measurement values.
        adjacency: Link-by-link operator whose columns identify each row's neighbours.

    Returns:
        Time-by-link neighbour means, or NaN where no neighbour is observed.
    """
    present = np.isfinite(values)
    totals = (adjacency @ np.where(present, values, 0).T).T
    counts = (adjacency @ present.astype(np.float32).T).T
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def history_features(
    grid: TrafficGrid,
    origins: IndexArray,
    origin_times: Sequence[datetime],
    network: QueueNetwork,
) -> FeatureMatrix:
    """Build lag, trend, queue, ramp, clock, and directed neighbour features.

    Args:
        grid: Observations indexed on a complete five-minute time grid.
        origins: Index of ``T`` for each example; may equal grid length at inference.
        origin_times: Forecast-origin timestamps used for clock features.
        network: Link properties and directed mainline adjacency.

    Returns:
        Feature rows ordered first by origin and then by network link order.
        All dynamic features exclude the origin and its future.
    """
    indices = origins[:, None] + np.arange(-HISTORY_STEPS, 0)
    shape = (len(origins), len(network.link_ids))
    columns: dict[str, FloatArray] = {}
    histories = {
        "speed_ratio": grid.speed_ratio[indices],
        "flow_ratio": grid.flow_ratio[indices],
        "occupancy": grid.occupancy[indices],
        "on_ramp_ratio": grid.on_ramp_ratio[indices],
        "off_ramp_ratio": grid.off_ramp_ratio[indices],
    }
    # Summarize the three hours before the standard 60-minute history.
    context_indices = origins[:, None] + np.arange(
        -CONTEXT_LOOKBACK_STEPS, -HISTORY_STEPS
    )
    context_sources = {
        "speed_ratio": grid.speed_ratio,
        "flow_ratio": grid.flow_ratio,
        "occupancy": grid.occupancy,
        "on_ramp_ratio": grid.on_ramp_ratio,
        "off_ramp_ratio": grid.off_ramp_ratio,
    }
    for name, source in context_sources.items():
        context = source[context_indices]
        columns[f"{name}_context_mean_180"] = _mean(context, axis=1)
        columns[f"{name}_context_change_180"] = context[:, -1] - context[:, 0]
        columns[f"{name}_context_observed_fraction"] = np.isfinite(context).mean(
            axis=1
        ).astype(np.float32)
        if name == "speed_ratio":
            context_known = np.isfinite(context)
            columns["context_queue_fraction_180"] = (
                context_known & (context <= QUEUE_SPEED_FRACTION)
            ).mean(axis=1).astype(np.float32)
    # Trees receive recent individual readings and changes over longer intervals.
    for name, history in histories.items():
        for lag in (1, 2, 3, 6, HISTORY_STEPS):
            columns[f"{name}_lag_{lag * QUEUE_INTERVAL_MINUTES}"] = history[:, -lag]
        columns[f"{name}_mean_15"] = _mean(history[:, -3:], axis=1)
        columns[f"{name}_mean_60"] = _mean(history, axis=1)
        columns[f"{name}_change_15"] = history[:, -1] - history[:, -4]
        columns[f"{name}_change_30"] = history[:, -1] - history[:, -7]
    speed = histories["speed_ratio"]
    known = np.isfinite(speed)
    queued = known & (speed <= QUEUE_SPEED_FRACTION)
    columns["observed_fraction"] = known.mean(axis=1).astype(np.float32)
    columns["queue_fraction_60"] = queued.mean(axis=1).astype(np.float32)
    columns["queue_fraction_15"] = queued[:, -3:].mean(axis=1).astype(np.float32)
    columns["queue_streak_minutes"] = (
        np.cumprod(queued[:, ::-1], axis=1).sum(axis=1) * QUEUE_INTERVAL_MINUTES
    ).astype(np.float32)
    columns["minimum_speed_ratio"] = np.min(speed, axis=1, where=known, initial=np.inf)
    columns["minimum_speed_ratio"][~known.any(axis=1)] = np.nan
    # Last-known speed remains causal and its age tells the model how stale it is.
    age = np.argmax(known[:, ::-1], axis=1)
    latest_known = np.take_along_axis(
        speed, (HISTORY_STEPS - 1 - age)[:, None], axis=1
    )[:, 0]
    columns["last_known_speed_ratio"] = np.where(
        known.any(axis=1), latest_known, np.nan
    )
    columns["last_speed_age_minutes"] = np.where(
        known.any(axis=1), (age + 1) * QUEUE_INTERVAL_MINUTES, np.nan
    ).astype(np.float32)
    # Directional features expose queue growth and recovery across connected links.
    queue_values = np.where(
        np.isfinite(grid.speed_ratio), grid.speed_ratio <= QUEUE_SPEED_FRACTION, np.nan
    ).astype(np.float32)
    for direction, adjacency in (
        ("upstream", network.upstream),
        ("downstream", network.downstream),
    ):
        neighbour_speed = _neighbor_mean(grid.speed_ratio, adjacency)
        second_neighbor_speed = _neighbor_mean(
            grid.speed_ratio, (adjacency @ adjacency).tocsr()
        )
        neighbour_flow = _neighbor_mean(grid.flow_ratio, adjacency)
        neighbour_queue = _neighbor_mean(queue_values, adjacency)
        for lag in (1, 3, 6):
            columns[f"{direction}_speed_lag_{lag * QUEUE_INTERVAL_MINUTES}"] = (
                neighbour_speed[origins - lag]
            )
        columns[f"{direction}_two_hop_speed"] = second_neighbor_speed[origins - 1]
        columns[f"{direction}_flow_ratio"] = neighbour_flow[origins - 1]
        columns[f"{direction}_queue_fraction"] = neighbour_queue[origins - 1]
        columns[f"{direction}_speed_gradient"] = (
            speed[:, -1] - neighbour_speed[origins - 1]
        )
    for name, values in network.static_features.items():
        columns[name] = np.broadcast_to(values, shape)
    columns["link_index"] = np.broadcast_to(
        np.arange(shape[1], dtype=np.float32), shape
    )
    slots_per_day = 24 * 60 // QUEUE_INTERVAL_MINUTES
    slots = np.asarray(
        [
            time.hour * 60 / QUEUE_INTERVAL_MINUTES
            + time.minute // QUEUE_INTERVAL_MINUTES
            for time in origin_times
        ],
        dtype=np.float32,
    )
    clock_features = {
        "time_sin": np.sin(2 * np.pi * slots / slots_per_day),
        "time_cos": np.cos(2 * np.pi * slots / slots_per_day),
        "day_of_week": np.asarray(
            [time.weekday() for time in origin_times], dtype=np.float32
        ),
    }
    for name, values in clock_features.items():
        columns[name] = np.broadcast_to(values[:, None], shape)
    return FeatureMatrix(
        np.column_stack([values.ravel() for values in columns.values()]).astype(
            np.float32, copy=False
        ),
        tuple(columns),
    )
