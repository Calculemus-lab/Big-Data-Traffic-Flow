"""Detector-by-time density fields for one freeway panel.

Each day is a grid whose rows are detectors ordered by milepost and whose
columns are the 288 five-minute slots of that UTC day. A cell stores density
divided by the link jam density. Task 1 targets stay at detector grain, so
two stations on one link occupy two rows and share that link's diagram.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from trafficbench.contracts import MIN_OBSERVED_PERCENT, QUEUE_INTERVAL_MINUTES

SLOTS_PER_DAY = 24 * (60 // QUEUE_INTERVAL_MINUTES)
OBSERVATION_COLUMNS = (
    "timestamp",
    "station_id",
    "link_id",
    "milepost",
    "speed_kmh",
    "flow_vph",
    "pct_observed",
)
LINK_PARAMETER_COLUMNS = (
    "link_id",
    "free_speed_kmh",
    "capacity_vph",
    "critical_density",
    "k_jam",
)


class DetectorAxis(NamedTuple):
    """One detector row and the fundamental-diagram parameters of its link.

    Attributes:
        station_id: Detector station identifier.
        link_id: Mainline link identifier.
        milepost: Median milepost used to order the row.
        free_speed_kmh: Free-flow speed on the link.
        capacity_vph: Link capacity in vehicles per hour.
        critical_density: Density at capacity, in vehicles per kilometre.
        k_jam: Jam density in vehicles per kilometre. Normalized density is
            measured density divided by this value.
    """

    station_id: str
    link_id: str
    milepost: float
    free_speed_kmh: float
    capacity_vph: float
    critical_density: float
    k_jam: float


class DayField(NamedTuple):
    """Normalized density for one calendar day.

    Attributes:
        day: UTC calendar day of the grid.
        density: Density divided by jam density, shape ``(detectors, slots)``.
            Unobserved cells are zero.
        observed: True where a cell has a usable speed and flow measurement.
    """

    day: date
    density: NDArray[np.float32]
    observed: NDArray[np.bool_]


class PanelFields(NamedTuple):
    """Detector axis and daily density grids for one panel.

    Attributes:
        axis: Detectors ordered by milepost, then station, then link.
        days: One grid per calendar day present in the observations.
    """

    axis: tuple[DetectorAxis, ...]
    days: tuple[DayField, ...]


def time_slot(timestamp: pl.Expr | None = None) -> pl.Expr:
    """Return the five-minute slot of a UTC timestamp, from 0 through 287.

    Args:
        timestamp: Timestamp expression to convert. Defaults to the
            ``timestamp`` column.

    Returns:
        Integer slot expression.
    """
    source = pl.col("timestamp") if timestamp is None else timestamp
    slots_per_hour = 60 // QUEUE_INTERVAL_MINUTES
    return (
        source.dt.hour() * slots_per_hour + source.dt.minute() // QUEUE_INTERVAL_MINUTES
    )


def usable_observation() -> pl.Expr:
    """Return a filter for cells that can enter the density field.

    Returns:
        True where coverage meets the scoring threshold and speed and flow
        are finite.
    """
    return (
        pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
        & pl.col("speed_kmh").is_not_null()
        & pl.col("flow_vph").is_not_null()
        & pl.col("speed_kmh").is_finite()
        & pl.col("flow_vph").is_finite()
    )


def speed_and_flow_from_density(
    normalized_density: NDArray[np.floating], axis: tuple[DetectorAxis, ...]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Invert normalized density through the triangular fundamental diagram.

    Density at or below the critical density is free flow: speed is the link
    free-flow speed and flow is that speed times density. Above critical
    density, flow follows the congested wave and speed is flow divided by
    density. Both results are clipped at zero.

    Args:
        normalized_density: Density divided by jam density, shape
            ``(detectors, slots)``.
        axis: Detector rows aligned with the first axis of ``normalized_density``.

    Returns:
        Speed in km/h and flow in vehicles per hour, both ``float32``.
    """
    free_speed = _parameter_column(axis, "free_speed_kmh")
    capacity = _parameter_column(axis, "capacity_vph")
    critical_density = _parameter_column(axis, "critical_density")
    jam_density = _parameter_column(axis, "k_jam")
    density = np.asarray(normalized_density, dtype=np.float64) * jam_density
    wave_speed = capacity / np.maximum(jam_density - critical_density, 1e-9)
    free_flow = density <= critical_density
    flow = np.where(
        free_flow,
        free_speed * density,
        wave_speed * np.maximum(jam_density - density, 0.0),
    )
    speed = np.where(free_flow, free_speed, flow / np.maximum(density, 1e-9))
    return (
        np.asarray(np.clip(speed, 0.0, None), dtype=np.float32),
        np.asarray(np.clip(flow, 0.0, None), dtype=np.float32),
    )


def panel_fields(
    observations: pl.DataFrame, fd_parameters: pl.DataFrame
) -> PanelFields | None:
    """Build daily detector-time density grids from mainline rows.

    A cell is usable when coverage meets the scoring threshold and both speed
    and flow are finite. Density is ``flow / max(speed, 1)``, then divided by
    the link jam density and clipped to ``[0, 1]``. Detectors whose link has
    no usable diagram are omitted.

    Args:
        observations: Mainline rows with timestamp, station, link, milepost,
            speed, flow, and ``pct_observed``.
        fd_parameters: Fundamental-diagram rows keyed by ``link_id``.

    Returns:
        Panel grids, or ``None`` when no detector has both a milepost and a
        usable diagram.
    """
    if observations.is_empty():
        return None
    prepared = _with_coordinates(observations)
    axis_frame = _detector_axis(prepared, fd_parameters)
    if axis_frame.is_empty():
        return None
    axis = tuple(
        DetectorAxis(
            station_id=row["station_id"],
            link_id=row["link_id"],
            milepost=float(row["milepost"]),
            free_speed_kmh=float(row["free_speed_kmh"]),
            capacity_vph=float(row["capacity_vph"]),
            critical_density=float(row["critical_density"]),
            k_jam=float(row["k_jam"]),
        )
        for row in axis_frame.iter_rows(named=True)
    )
    indexed = prepared.join(
        axis_frame.select("station_id", "link_id", "detector_row", "k_jam"),
        on=["station_id", "link_id"],
        how="inner",
    )
    return PanelFields(axis=axis, days=_day_fields(indexed, len(axis)))


def target_speed_and_flow(
    fields: PanelFields,
    completed_by_day: dict[date, tuple[NDArray[np.float32], NDArray[np.float32]]],
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Read completed speed and flow at requested detector timestamps.

    Args:
        fields: Detector axis used to place each target.
        completed_by_day: Speed and flow grids keyed by UTC calendar day.
        targets: Task 1 rows containing ``timestamp``, ``station_id``, and
            ``link_id``.

    Returns:
        Matching target identities with finite speed and flow. Targets that
        miss the axis or the completed day are omitted.
    """
    identity = ["timestamp", "station_id", "link_id"]
    axis_frame = pl.DataFrame(
        {
            "station_id": [detector.station_id for detector in fields.axis],
            "link_id": [detector.link_id for detector in fields.axis],
            "detector_row": list(range(len(fields.axis))),
        }
    )
    selected = (
        targets.select(identity)
        .unique(identity)
        .join(axis_frame, on=["station_id", "link_id"], how="inner")
        .with_columns(
            pl.col("timestamp").dt.date().alias("day"),
            time_slot().alias("slot"),
        )
        .filter(pl.col("slot").is_between(0, SLOTS_PER_DAY - 1))
        .with_row_index("_order")
    )
    speed_values = np.full(selected.height, np.nan, dtype=np.float64)
    flow_values = np.full(selected.height, np.nan, dtype=np.float64)
    if selected.height:
        for key, group in selected.group_by("day"):
            day = _as_date(key[0])
            completed = completed_by_day.get(day)
            if completed is None:
                continue
            speed, flow = completed
            rows = group["detector_row"].to_numpy()
            slots = group["slot"].to_numpy()
            order = group["_order"].to_numpy()
            speed_values[order] = speed[rows, slots]
            flow_values[order] = flow[rows, slots]
    return (
        selected.select(identity)
        .with_columns(
            pl.Series("speed_kmh", speed_values),
            pl.Series("flow_vph", flow_values),
        )
        .filter(pl.col("speed_kmh").is_finite() & pl.col("flow_vph").is_finite())
    )


def _parameter_column(axis: tuple[DetectorAxis, ...], name: str) -> NDArray[np.float64]:
    """Stack one diagram parameter as a column vector.

    Args:
        axis: Detector rows.
        name: ``DetectorAxis`` field to stack.

    Returns:
        Array shaped ``(detectors, 1)``.
    """
    values = [float(getattr(detector, name)) for detector in axis]
    return np.asarray(values, dtype=np.float64)[:, None]


def _with_coordinates(observations: pl.DataFrame) -> pl.DataFrame:
    """Attach the UTC calendar day and five-minute slot.

    Args:
        observations: Mainline rows whose timestamp may be text or datetime.

    Returns:
        The same rows with ``day`` and ``slot`` columns and a UTC timestamp.
    """
    timestamp_dtype = observations.schema["timestamp"]
    frame = observations
    if timestamp_dtype == pl.String:
        frame = frame.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=True)
        )
    elif isinstance(timestamp_dtype, pl.Datetime):
        if timestamp_dtype.time_zone is None:
            frame = frame.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        else:
            frame = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return frame.with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        time_slot().alias("slot"),
    )


def _detector_axis(
    observations: pl.DataFrame, fd_parameters: pl.DataFrame
) -> pl.DataFrame:
    """Order detectors by milepost and attach a usable fundamental diagram.

    Args:
        observations: Rows that already carry ``milepost``.
        fd_parameters: Diagram table keyed by ``link_id``.

    Returns:
        One row per detector with a ``detector_row`` index. Links whose jam
        density does not exceed critical density are dropped.
    """
    parameters = (
        fd_parameters.select(LINK_PARAMETER_COLUMNS)
        .group_by("link_id")
        .agg(pl.all().first())
        .filter(
            pl.col("k_jam").is_not_null()
            & pl.col("k_jam").gt(0)
            & pl.col("critical_density").is_not_null()
            & pl.col("free_speed_kmh").is_not_null()
            & pl.col("capacity_vph").is_not_null()
            & pl.col("k_jam").gt(pl.col("critical_density"))
        )
    )
    return (
        observations.filter(
            pl.col("milepost").is_not_null() & pl.col("milepost").is_finite()
        )
        .group_by("station_id", "link_id")
        .agg(pl.col("milepost").median())
        .join(parameters, on="link_id", how="inner")
        .sort("milepost", "station_id", "link_id")
        .with_row_index("detector_row")
    )


def _day_fields(indexed: pl.DataFrame, detector_count: int) -> tuple[DayField, ...]:
    """Scatter usable densities onto one grid per calendar day.

    Args:
        indexed: Observations joined to ``detector_row`` and ``k_jam``.
        detector_count: Number of rows in the detector axis.

    Returns:
        Daily grids in calendar order. Unobserved cells stay zero.
    """
    day_values = [_as_date(value) for value in indexed["day"].unique().sort().to_list()]
    usable = indexed.filter(usable_observation())
    summaries = usable.group_by("day", "detector_row", "slot").agg(
        (
            pl.col("flow_vph")
            / pl.max_horizontal(pl.col("speed_kmh"), pl.lit(1.0))
            / pl.col("k_jam")
        )
        .clip(0.0, 1.0)
        .mean()
        .alias("density")
    )
    summaries_by_day: dict[date, pl.DataFrame] = {}
    if summaries.height:
        for key, frame in summaries.partition_by("day", as_dict=True).items():
            summaries_by_day[_as_date(key[0])] = frame
    days: list[DayField] = []
    for day in day_values:
        density = np.zeros((detector_count, SLOTS_PER_DAY), dtype=np.float32)
        observed = np.zeros((detector_count, SLOTS_PER_DAY), dtype=np.bool_)
        summary = summaries_by_day.get(day)
        if summary is not None and summary.height:
            rows = summary["detector_row"].to_numpy()
            slots = summary["slot"].to_numpy()
            values = summary["density"].to_numpy()
            valid = (
                (slots >= 0)
                & (slots < SLOTS_PER_DAY)
                & (rows >= 0)
                & (rows < detector_count)
            )
            density[rows[valid], slots[valid]] = values[valid]
            observed[rows[valid], slots[valid]] = True
        days.append(DayField(day=day, density=density, observed=observed))
    return tuple(days)


def _as_date(value: object) -> date:
    """Normalize a Polars date key to ``datetime.date``.

    Args:
        value: Date or datetime drawn from a Polars column or group key.

    Returns:
        The calendar date.

    Raises:
        TypeError: If ``value`` is not a date or datetime.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Expected a date, got {type(value).__name__}")
