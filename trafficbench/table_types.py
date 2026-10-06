"""Polars lazy table aliases and field descriptions for solution code.

The aliases distinguish table roles in function signatures. Their typed
dictionaries document each table's columns for IDE inspection, but a static
type checker still sees every frame as ``pl.LazyFrame`` and cannot verify
names inside expressions such as ``pl.col("speed_kmh")``. Target templates,
historical answers, and predictions share a task's frame alias because their
columns match. Boolean values are stored as integer ``0`` or ``1``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, TypedDict

import polars as pl

type Panel = Literal[
    "D7_I10_E",
    "D7_I10_W",
    "D7_I210_E",
    "D7_I210_W",
    "D7_I405_N",
    "D7_I405_S",
    "D12_I5_N",
    "D12_I5_S",
    "D12_I405_N",
    "D12_I405_S",
]
type Split = Literal["train", "validation", "private"]
# Local queue cases require a future queue. They are "ongoing" when a link
# was queued at two or more history timestamps, and "onset" otherwise.
type QueueCondition = Literal["queue_onset", "queue_ongoing"]
type MaskRegime = Literal["R1", "R2", "R3"]
type BooleanInt = Literal[0, 1]
type RampType = Literal["OR", "FR"]


class LinksColumns(TypedDict):
    """Column schema for the panel's road links and nominal operating limits.

    Each row describes one directed mainline link and its length, lane count,
    free-flow speed, and total vehicle capacity. The release stores lane counts
    as numeric values, so their declared type is ``float``.
    """

    link_id: str
    length_km: float
    lanes: float
    free_speed_kmh: float
    capacity_vph: float


type LinksFrame = Annotated[pl.LazyFrame, LinksColumns]


class FdParametersColumns(LinksColumns):
    """Column schema for link geometry and fundamental diagram parameters.

    Each row describes one link, extending its nominal limits with critical
    and jam density values used by the traffic flow model.
    """

    critical_density: float
    k_jam: float


type FdParametersFrame = Annotated[pl.LazyFrame, FdParametersColumns]


class PathSetColumns(TypedDict):
    """Column schema for candidate paths through the panel's network.

    Each row identifies a path, its origin and destination zones, and its
    ordered sequence of links.
    """

    path_id: str
    origin_zone: str
    destination_zone: str
    link_seq: str


type PathSetFrame = Annotated[pl.LazyFrame, PathSetColumns]


class PathLinkIncidenceColumns(TypedDict):
    """Column schema for the candidate-path to link relationship.

    Each row says that one candidate path uses one network link.
    """

    link_id: str
    path_id: str


type PathLinkIncidenceFrame = Annotated[pl.LazyFrame, PathLinkIncidenceColumns]


class LwrMainlineTopologyColumns(TypedDict):
    """Columns describing mainline links and their connections.

    Each row gives one mainline link, its incoming and outgoing links, any
    attached ramps, and its order and traffic properties. The connection lists
    and their sensor flags are semicolon-separated text. Node IDs and the
    ``has_sensor`` field are stored as integers in the release CSV.
    """

    mainline_link_id: str
    incoming_link_ids: str
    outgoing_link_ids: str
    on_ramp_link_ids: str
    off_ramp_link_ids: str
    incoming_has_sensor: str
    outgoing_has_sensor: str
    lanes: int
    length_km: float
    capacity_vph: float
    free_speed_kmh: float
    order_index: int
    link_id: str
    from_node: int
    to_node: int
    has_sensor: BooleanInt
    detector_id: str
    n_incoming: int
    n_outgoing: int
    n_incoming_no_sensor: int
    n_outgoing_no_sensor: int
    free_flow_speed_kmh: float


type LwrMainlineTopologyFrame = Annotated[pl.LazyFrame, LwrMainlineTopologyColumns]


class RampAttachmentColumns(TypedDict):
    """Column schema for ramp-to-mainline attachment information.

    Each row identifies a ramp, its type (``OR`` for an on-ramp or ``FR`` for
    an off-ramp), and the mainline link to which it is attached.
    """

    ramp_link_id: str
    ramp_type: RampType
    nearest_mainline_link_id: str


type RampAttachmentFrame = Annotated[pl.LazyFrame, RampAttachmentColumns]


class MainlineStateColumns(TypedDict):
    """Column schema for dated mainline traffic measurements.

    Each row describes one panel, detector, link, and Coordinated Universal
    Time (UTC) timestamp. Measurement values and observation flags can be null
    when the release masks a target row. Flags are integer booleans: ``0`` is
    false and ``1`` is true.
    """

    corridor_id: Panel
    date: str
    timestamp: datetime
    station_id: str
    link_id: str
    milepost: float
    direction: str
    speed_kmh: float | None
    flow_vph: float | None
    occupancy: float | None
    density_occ_linear_vehpkm: float | None
    pct_observed: int | None
    is_observed: BooleanInt | None
    is_imputed: BooleanInt | None
    is_score_eligible: BooleanInt | None
    is_missing: BooleanInt | None


type MainlineStatesFrame = Annotated[pl.LazyFrame, MainlineStateColumns]


class MaskedMainlineStateColumns(MainlineStateColumns):
    """Column schema for mainline measurements used in reconstruction cases.

    Each row has the mainline measurements plus the regime that identifies how
    many target measurements were masked for Task 1 reconstruction.
    """

    mask_regime: MaskRegime


type MaskedMainlineStatesFrame = Annotated[pl.LazyFrame, MaskedMainlineStateColumns]


class QueueHistoryColumns(TypedDict):
    """Measurements for each link and timestamp in a queue forecast's history.

    ``window_id`` identifies the forecast that uses the row. Measurements,
    coverage, and eligibility can be null where target data is hidden.
    ``is_score_eligible`` is an integer boolean: ``0`` means the reading does
    not meet the coverage rule and ``1`` means it does.
    """

    window_id: str
    timestamp: datetime
    link_id: str
    speed_kmh: float | None
    flow_vph: float | None
    occupancy: float | None
    pct_observed: int | None
    is_score_eligible: BooleanInt | None


type QueueHistoryFrame = Annotated[pl.LazyFrame, QueueHistoryColumns]


class QueueWindowIndexColumns(TypedDict):
    """Timing, local condition, and measured coverage for one queue forecast.

    ``forecast_origin`` is the last history timestamp. Local cases call a
    window ``queue_ongoing`` if any link was queued at two or more history
    timestamps. They call it ``queue_onset`` otherwise. Both kinds have at
    least one queued target row in the forecast.
    """

    window_id: str
    panel: Panel
    family_id: str
    split: Split
    date: str
    forecast_origin: datetime
    history_start: datetime
    history_end: datetime
    forecast_start: datetime
    forecast_end: datetime
    condition: QueueCondition
    history_coverage: float
    future_coverage: float


type QueueWindowIndexFrame = Annotated[pl.LazyFrame, QueueWindowIndexColumns]


class StateColumns(TypedDict):
    """Task 1 target rows and their known or predicted speed and flow.

    Each row identifies a panel, UTC timestamp, detector station, mainline
    link, and mask regime. In a target template, speed and flow contain zero
    placeholders for the solution to replace with predictions.
    """

    panel: Panel
    timestamp: datetime
    station_id: str
    link_id: str
    mask_regime: MaskRegime
    speed_kmh: float
    flow_vph: float


type StateFrame = Annotated[pl.LazyFrame, StateColumns]


class QueueColumns(TypedDict):
    """Task 2 target rows and their known or predicted queue state.

    Each row identifies a mainline link and one of the six five-minute target
    times in a forecast window. ``queue_pred`` is an integer boolean: ``0`` for
    no queue and ``1`` for a queue.
    """

    window_id: str
    timestamp: datetime
    link_id: str
    queue_pred: BooleanInt


type QueueFrame = Annotated[pl.LazyFrame, QueueColumns]


class OdmeColumns(TypedDict):
    """Task 4 origin-destination matrix estimation target rows.

    Each row identifies a panel, departure period, candidate path, origin zone,
    and destination zone. ``path_flow`` is the number of vehicles assigned to
    that path during the departure period.
    """

    panel: Panel
    departure_time: str
    path_id: str
    origin_zone: str
    destination_zone: str
    path_flow: float


type OdmeFrame = Annotated[pl.LazyFrame, OdmeColumns]


class RampStateColumns(TypedDict):
    """Column schema for dated ramp traffic measurements.

    Each row describes one panel, ramp detector, ramp link, and UTC timestamp.
    Flow and observation flags can be null when the release masks a target row.
    Flags are integer booleans: ``0`` is false and ``1`` is true.
    """

    corridor_id: Panel
    date: str
    timestamp: datetime
    station_id: str
    ramp_link_id: str
    ramp_type: RampType
    flow_vph: float | None
    pct_observed: int | None
    is_observed: BooleanInt | None
    is_imputed: BooleanInt | None
    is_score_eligible: BooleanInt | None
    is_missing: BooleanInt | None


type RampStatesFrame = Annotated[pl.LazyFrame, RampStateColumns]


class LinkCountsColumns(TypedDict):
    """Column schema for Task 4 traffic counts assigned to panel links.

    Each row gives the observed or generated count for one link in a scenario.
    """

    panel: Panel
    link_id: str
    count: float


type LinkCountsFrame = Annotated[pl.LazyFrame, LinkCountsColumns]


class WeakPriorColumns(TypedDict):
    """Column schema for the starting path-flow estimate in a Task 4 case.

    Each row identifies a panel, departure time, path, and origin-destination
    pair, then gives that path's initial flow estimate.
    """

    panel: Panel
    departure_time: str
    path_id: str
    origin_zone: str
    destination_zone: str
    path_flow: float


type WeakPriorFrame = Annotated[pl.LazyFrame, WeakPriorColumns]
