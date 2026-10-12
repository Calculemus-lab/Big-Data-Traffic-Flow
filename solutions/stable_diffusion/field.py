"""Milepost-by-time fields for one freeway panel.

Each day becomes a grid whose rows are detectors ordered by milepost and whose
columns are the 288 five-minute slots of that UTC day. Speed and flow stay in
physical units here; the diffusion model normalizes them separately.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from trafficbench.contracts import MIN_OBSERVED_PERCENT, QUEUE_INTERVAL_MINUTES

SLOTS_PER_DAY = 24 * (60 // QUEUE_INTERVAL_MINUTES)
WEEKDAYS = 7
DEFAULT_SPEED_SCALE = 120.0
DEFAULT_FLOW_SCALE = 2000.0
UNIT_CAP = 1.5


class DetectorAxis(NamedTuple):
    """One detector row in a panel's spatiotemporal field.

    Attributes:
        station_id: Detector station identifier.
        link_id: Mainline link identifier.
        milepost: Median milepost used to order the row.
        speed_scale: Free-flow speed used to normalize this detector's speed.
        flow_scale: Link capacity used to normalize this detector's flow.
    """

    station_id: str
    link_id: str
    milepost: float
    speed_scale: float
    flow_scale: float


class DayField(NamedTuple):
    """Observed speed and flow for one calendar day.

    Attributes:
        day: UTC calendar day of the grid.
        speed: Speed in km/h with shape ``(detectors, slots)``. Missing cells
            are NaN.
        flow: Flow in vehicles per hour, same shape as ``speed``.
        observed: True where a cell has a usable speed and flow measurement.
    """

    day: date
    speed: NDArray[np.float32]
    flow: NDArray[np.float32]
    observed: NDArray[np.bool_]


class PanelFields(NamedTuple):
    """Detector axis, daily grids, and weekday-slot means for one panel.

    Attributes:
        axis: Detectors ordered by milepost, then station, then link.
        days: One grid per calendar day present in the observations.
        prior_speed: Mean observed speed with shape ``(detectors, weekdays, slots)``.
        prior_flow: Mean observed flow with the same shape as ``prior_speed``.
    """

    axis: tuple[DetectorAxis, ...]
    days: tuple[DayField, ...]
    prior_speed: NDArray[np.float32]
    prior_flow: NDArray[np.float32]


class ModelSample(NamedTuple):
    """Normalized tensors for one day, ready for the diffusion network.

    Attributes:
        clean: Filled speed and flow in model units, shape ``(2, detectors, slots)``.
        observed_values: Measured cells in model units and zeros elsewhere.
        mask: Usable-cell mask with shape ``(1, detectors, slots)``.
        prior: Weekday-slot mean in model units, shape ``(2, detectors, slots)``.
    """

    clean: NDArray[np.float32]
    observed_values: NDArray[np.float32]
    mask: NDArray[np.float32]
    prior: NDArray[np.float32]


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


def to_model_units(
    values: NDArray[np.floating], scale: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Map a non-negative physical field into ``[-1, 1]``.

    Args:
        values: Speed or flow values.
        scale: Positive per-detector scale broadcastable to ``values``.

    Returns:
        Clipped model units. Values above ``1.5 * scale`` saturate at 1.
    """
    unit = np.clip(values / scale, 0.0, UNIT_CAP) / UNIT_CAP
    return np.asarray(unit * 2.0 - 1.0, dtype=np.float32)


def from_model_units(
    units: NDArray[np.floating], scale: NDArray[np.floating]
) -> NDArray[np.float32]:
    """Invert :func:`to_model_units` and clip at zero.

    Args:
        units: Model outputs, typically in ``[-1, 1]``.
        scale: Positive per-detector scale broadcastable to ``units``.

    Returns:
        Physical speed or flow.
    """
    raw = (units + 1.0) / 2.0 * UNIT_CAP * scale
    return np.asarray(np.clip(raw, 0.0, None), dtype=np.float32)


def panel_fields(
    observations: pl.DataFrame, link_scales: pl.DataFrame
) -> PanelFields | None:
    """Build daily milepost-time grids and weekday priors from mainline rows.

    A cell is usable when coverage meets the scoring threshold and both speed
    and flow are present. Detectors are ordered by median milepost.

    Args:
        observations: Mainline rows with timestamp, station, link, milepost,
            speed, flow, and ``pct_observed``.
        link_scales: Link free-flow speed and capacity, keyed by ``link_id``.

    Returns:
        Panel grids, or ``None`` when the frame has no detector rows.
    """
    if observations.is_empty():
        return None
    prepared = _with_coordinates(observations)
    axis_frame = _detector_axis(prepared, link_scales)
    if axis_frame.is_empty():
        return None
    axis = tuple(
        DetectorAxis(
            station_id=row["station_id"],
            link_id=row["link_id"],
            milepost=float(row["milepost"]),
            speed_scale=float(row["speed_scale"]),
            flow_scale=float(row["flow_scale"]),
        )
        for row in axis_frame.iter_rows(named=True)
    )
    indexed = prepared.join(
        axis_frame.select("station_id", "link_id", "row"),
        on=["station_id", "link_id"],
        how="inner",
    )
    detector_count = len(axis)
    days = _day_fields(indexed, detector_count)
    prior_speed, prior_flow = _weekday_priors(indexed, axis, detector_count)
    return PanelFields(
        axis=axis, days=days, prior_speed=prior_speed, prior_flow=prior_flow
    )


def prior_for_day(
    fields: PanelFields, day: date
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the weekday-slot mean speed and flow for one calendar day.

    Args:
        fields: Panel grids that store the weekday priors.
        day: Calendar day whose weekday selects the prior.

    Returns:
        Speed and flow arrays shaped ``(detectors, slots)``.
    """
    weekday = day.weekday()
    return fields.prior_speed[:, weekday, :], fields.prior_flow[:, weekday, :]


def model_sample(field: DayField, fields: PanelFields) -> ModelSample:
    """Normalize one day, filling unobserved cells in the clean target with the prior.

    Args:
        field: Daily grid to encode.
        fields: Panel axis and priors that supply scales and the weekday mean.

    Returns:
        Clean target, observed conditioning values, mask, and prior.
    """
    prior_speed, prior_flow = prior_for_day(fields, field.day)
    speed_scale, flow_scale = _scale_columns(fields)
    observed = field.observed
    speed_filled = np.where(observed, field.speed, prior_speed)
    flow_filled = np.where(observed, field.flow, prior_flow)
    clean = np.stack(
        (
            to_model_units(speed_filled, speed_scale),
            to_model_units(flow_filled, flow_scale),
        )
    )
    observed_values = np.stack(
        (
            np.where(observed, to_model_units(field.speed, speed_scale), 0.0),
            np.where(observed, to_model_units(field.flow, flow_scale), 0.0),
        )
    ).astype(np.float32)
    prior = np.stack(
        (
            to_model_units(prior_speed, speed_scale),
            to_model_units(prior_flow, flow_scale),
        )
    )
    mask = observed.astype(np.float32)[None, :, :]
    return ModelSample(
        clean=clean,
        observed_values=observed_values,
        mask=mask,
        prior=prior,
    )


def denormalize(
    units: NDArray[np.floating], fields: PanelFields
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert a two-channel model field back to speed and flow.

    Args:
        units: Array shaped ``(2, detectors, slots)`` in model units.
        fields: Panel axis that supplies per-detector scales.

    Returns:
        Speed in km/h and flow in vehicles per hour.
    """
    speed_scale, flow_scale = _scale_columns(fields)
    speed = from_model_units(units[0], speed_scale)
    flow = from_model_units(units[1], flow_scale)
    return speed, flow


def rows_from_completed_field(
    day: date,
    speed: NDArray[np.floating],
    flow: NDArray[np.floating],
    fields: PanelFields,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Read completed grid cells at the target detector and timestamp locations.

    Args:
        day: Calendar day represented by ``speed`` and ``flow``.
        speed: Completed speed grid shaped ``(detectors, slots)``.
        flow: Completed flow grid with the same shape.
        fields: Detector axis aligned with the grids.
        targets: Target rows for this day. Rows whose detector is absent from
            the axis are omitted.

    Returns:
        Matching target identities with speed and flow. ``mask_regime`` is
        preserved when the target frame includes it.
    """
    axis = pl.DataFrame(
        {
            "station_id": [detector.station_id for detector in fields.axis],
            "link_id": [detector.link_id for detector in fields.axis],
            "row": list(range(len(fields.axis))),
        }
    )
    located = (
        targets.filter(pl.col("timestamp").dt.date() == day)
        .with_columns(time_slot().alias("slot"))
        .join(axis, on=["station_id", "link_id"], how="inner")
        .filter(pl.col("slot").is_between(0, SLOTS_PER_DAY - 1))
    )
    if located.is_empty():
        return located
    rows = located["row"].to_numpy()
    slots = located["slot"].to_numpy()
    return located.with_columns(
        pl.Series("speed_kmh", speed[rows, slots]),
        pl.Series("flow_vph", flow[rows, slots]),
    ).drop("row", "slot")


def _with_coordinates(observations: pl.DataFrame) -> pl.DataFrame:
    """Add day, weekday, slot, and usable-cell columns.

    Args:
        observations: Mainline measurement rows.

    Returns:
        The same rows with coordinate and usability columns.
    """
    frame = observations
    if "milepost" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("milepost"))
    # Release parquet stores timestamps as UTC text such as 2030-11-06T00:00:00Z.
    timestamp_dtype = frame.schema["timestamp"]
    if timestamp_dtype == pl.String:
        frame = frame.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=True)
        )
    elif isinstance(timestamp_dtype, pl.Datetime):
        timestamp = pl.col("timestamp")
        frame = frame.with_columns(
            timestamp.dt.replace_time_zone("UTC")
            if timestamp_dtype.time_zone is None
            else timestamp.dt.convert_time_zone("UTC")
        )
    return frame.with_columns(
        pl.col("timestamp").dt.date().alias("day"),
        (pl.col("timestamp").dt.weekday() - 1).alias("weekday"),
        time_slot().alias("slot"),
        (
            pl.col("pct_observed").fill_null(0).ge(MIN_OBSERVED_PERCENT)
            & pl.col("speed_kmh").is_not_null()
            & pl.col("flow_vph").is_not_null()
        ).alias("usable"),
    )


def _detector_axis(
    observations: pl.DataFrame, link_scales: pl.DataFrame
) -> pl.DataFrame:
    """Order detectors by milepost and attach normalization scales.

    Args:
        observations: Rows that already contain ``milepost``.
        link_scales: Free-flow speed and capacity by link.

    Returns:
        One row per detector with a dense ``row`` index.
    """
    scales = link_scales.select("link_id", "free_speed_kmh", "capacity_vph")
    return (
        observations.group_by("station_id", "link_id")
        .agg(pl.col("milepost").drop_nulls().median().alias("milepost"))
        .join(scales, on="link_id", how="left")
        .with_columns(
            pl.col("milepost").fill_null(0.0),
            pl.col("free_speed_kmh")
            .fill_null(DEFAULT_SPEED_SCALE)
            .alias("speed_scale"),
            pl.col("capacity_vph").fill_null(DEFAULT_FLOW_SCALE).alias("flow_scale"),
        )
        .sort("milepost", "station_id", "link_id")
        .with_row_index("row")
    )


def _day_fields(indexed: pl.DataFrame, detector_count: int) -> tuple[DayField, ...]:
    """Scatter usable measurements into one dense grid per day.

    Args:
        indexed: Observations joined to detector row indexes.
        detector_count: Number of detector rows.

    Returns:
        Daily grids in calendar order.
    """
    aggregated = (
        indexed.filter(pl.col("slot").is_between(0, SLOTS_PER_DAY - 1))
        .group_by("day", "row", "slot")
        .agg(
            pl.col("speed_kmh").filter(pl.col("usable")).mean().alias("speed_kmh"),
            pl.col("flow_vph").filter(pl.col("usable")).mean().alias("flow_vph"),
            pl.col("usable").any().alias("observed"),
        )
        .sort("day")
    )
    days: list[DayField] = []
    day_groups = list(aggregated.group_by("day", maintain_order=True))
    for day_value, day_rows in tqdm(day_groups, desc="build days"):
        day = day_value[0]
        speed = np.full((detector_count, SLOTS_PER_DAY), np.nan, dtype=np.float32)
        flow = np.full((detector_count, SLOTS_PER_DAY), np.nan, dtype=np.float32)
        observed = np.zeros((detector_count, SLOTS_PER_DAY), dtype=np.bool_)
        usable_rows = day_rows.filter(pl.col("observed"))
        if not usable_rows.is_empty():
            row_index = usable_rows["row"].to_numpy()
            slot_index = usable_rows["slot"].to_numpy()
            speed[row_index, slot_index] = usable_rows["speed_kmh"].to_numpy()
            flow[row_index, slot_index] = usable_rows["flow_vph"].to_numpy()
            observed[row_index, slot_index] = True
        if isinstance(day, date):
            days.append(DayField(day=day, speed=speed, flow=flow, observed=observed))
    return tuple(days)


def _weekday_priors(
    indexed: pl.DataFrame,
    axis: tuple[DetectorAxis, ...],
    detector_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Average usable measurements by detector, weekday, and time slot.

    Empty weekday-slot cells fall back to that detector's mean, then to a
    fraction of its free-flow speed or capacity.

    Args:
        indexed: Observations joined to detector row indexes.
        axis: Ordered detectors, used for the fallback scales.
        detector_count: Number of detector rows.

    Returns:
        Prior speed and flow shaped ``(detectors, weekdays, slots)``.
    """
    summary = (
        indexed.filter(
            pl.col("usable") & pl.col("slot").is_between(0, SLOTS_PER_DAY - 1)
        )
        .group_by("row", "weekday", "slot")
        .agg(
            pl.col("speed_kmh").mean().alias("speed_kmh"),
            pl.col("flow_vph").mean().alias("flow_vph"),
        )
    )
    speed_sum = np.zeros((detector_count, WEEKDAYS, SLOTS_PER_DAY), dtype=np.float64)
    flow_sum = np.zeros_like(speed_sum)
    counts = np.zeros_like(speed_sum)
    if not summary.is_empty():
        rows = summary["row"].to_numpy()
        weekdays = summary["weekday"].to_numpy()
        slots = summary["slot"].to_numpy()
        np.add.at(speed_sum, (rows, weekdays, slots), summary["speed_kmh"].to_numpy())
        np.add.at(flow_sum, (rows, weekdays, slots), summary["flow_vph"].to_numpy())
        np.add.at(counts, (rows, weekdays, slots), 1.0)
    speed = np.divide(speed_sum, counts, out=np.zeros_like(speed_sum), where=counts > 0)
    flow = np.divide(flow_sum, counts, out=np.zeros_like(flow_sum), where=counts > 0)
    detector_counts = counts.sum(axis=(1, 2))
    detector_speed = np.divide(
        speed_sum.sum(axis=(1, 2)),
        detector_counts,
        out=np.array(
            [0.75 * detector.speed_scale for detector in axis], dtype=np.float64
        ),
        where=detector_counts > 0,
    )
    detector_flow = np.divide(
        flow_sum.sum(axis=(1, 2)),
        detector_counts,
        out=np.array(
            [0.5 * detector.flow_scale for detector in axis], dtype=np.float64
        ),
        where=detector_counts > 0,
    )
    missing = counts == 0
    speed = np.where(missing, detector_speed[:, None, None], speed)
    flow = np.where(missing, detector_flow[:, None, None], flow)
    return speed.astype(np.float32), flow.astype(np.float32)


def _scale_columns(
    fields: PanelFields,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return per-detector speed and flow scales shaped for grid broadcasting.

    Args:
        fields: Panel whose detector axis stores the scales.

    Returns:
        Speed and flow scales shaped ``(detectors, 1)``.
    """
    speed_scale = np.array(
        [detector.speed_scale for detector in fields.axis], dtype=np.float32
    )[:, None]
    flow_scale = np.array(
        [detector.flow_scale for detector in fields.axis], dtype=np.float32
    )[:, None]
    return speed_scale, flow_scale
