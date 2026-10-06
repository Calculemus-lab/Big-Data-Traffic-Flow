"""Types shared by release slicing, solution execution, scoring, and submission.

Pydantic models describe records saved as JSON. ``ReleasePackageSlice`` and
``NetworkTables`` describe the typed release data passed to a solution. The
task column maps below define the keys and output values used for predictions.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Final, Literal, NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from . import table_types

type JsonObject = dict[str, JsonValue]
type Split = table_types.Split
ALL_SPLITS: Final[tuple[Split, ...]] = ("train", "validation", "private")
type Task = Literal["state", "queue", "odme"]
type TargetShortcut = Literal["validation", "private", "both"]
type Panel = table_types.Panel
ALL_PANELS: Final[tuple[Panel, ...]] = (
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
)
type Profile = Literal["quick", "full"]
type KeyColumn = Literal[
    "panel",
    "timestamp",
    "station_id",
    "link_id",
    "mask_regime",
    "window_id",
    "departure_time",
    "path_id",
    "origin_zone",
    "destination_zone",
]
type ValueColumn = Literal["speed_kmh", "flow_vph", "queue_pred", "path_flow"]
type QueueCondition = table_types.QueueCondition
type RunStatus = Literal["running", "completed", "failed"]

type MaskRegime = table_types.MaskRegime
# These values are shared by case preparation, scoring, and the reference
# predictors. Changing one task rule requires changing it here once.
MASK_REGIMES: Final[tuple[MaskRegime, ...]] = ("R1", "R2", "R3")
MASKED_STATE_COLUMNS: Final[tuple[str, ...]] = (
    "speed_kmh",
    "flow_vph",
    "occupancy",
    "density_occ_linear_vehpkm",
)
MIN_OBSERVED_PERCENT: Final = 75
QUEUE_SPEED_FRACTION: Final = 0.6
QUEUE_INTERVAL_MINUTES: Final = 5
QUEUE_HISTORY_MINUTES: Final = 60
QUEUE_FORECAST_STEPS: Final = 6


class StrictRecord(BaseModel):
    """Validate serialized benchmark records without coercing field types."""

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class ProfileConfig(StrictRecord):
    """Panels selected by a convenience preparation profile.

    Attributes:
        panels: The fixed ``all`` choice or explicit panel list for this profile.
    """

    panels: Literal["all"] | list[Panel]


class QueueConfig(StrictRecord):
    """Number, spacing, and measurement coverage required for queue cases.

    Attributes:
        windows_per_condition: Number of onset and ongoing windows to select.
        minimum_spacing_minutes: Minimum time between selected forecast origins.
        minimum_coverage: Required fraction of usable history records.
        excluded_panels: Panels omitted from local queue case generation.
    """

    windows_per_condition: int
    minimum_spacing_minutes: int
    minimum_coverage: float
    excluded_panels: list[Panel]


class OdmeConfig(StrictRecord):
    """Settings for generated origin-destination matrix estimation cases.

    Attributes:
        synthetic_cases: Number of seeded path-flow truth scenarios per panel.
        demand_log_sigma: Log-space spread applied to synthetic path demand.
        count_noise_fraction: Relative noise applied to synthetic link counts.
    """

    synthetic_cases: int
    demand_log_sigma: float
    count_noise_fraction: float


class BenchmarkConfig(StrictRecord):
    """Local case-generation settings, separate from release date boundaries.

    Attributes:
        name: Human-readable scheme name stored in the prepared manifest.
        seed: Seed used to choose repeatable queue windows and OD scenarios.
        profiles: Named panel-selection settings.
        queue: Queue-window count, spacing, and coverage rules.
        odme: Synthetic path-flow and count-noise settings.
    """

    name: str
    seed: int
    profiles: dict[Profile, ProfileConfig]
    queue: QueueConfig
    odme: OdmeConfig


class StateCase(StrictRecord):
    """Task 1 holdout case with visible inputs and separately stored answers.

    Attributes:
        task: Fixed discriminator identifying this as a state case.
        panel: Panel whose masked measurements are reconstructed.
        family_id: Freeway family used to aggregate panel metrics.
        case_id: Stable case identifier derived from its prediction interval.
        history_start_date: Inclusive beginning of the visible train interval.
        history_end_date: Exclusive end of known historical labels.
        prediction_start_date: Inclusive first state target date.
        prediction_end_date: Exclusive end of state targets.
        case_directory: Relative directory holding features and target keys.
        historical_solutions_path: Relative path to known earlier Task 1 values.
    """

    task: Literal["state"] = "state"
    panel: Panel
    family_id: str
    case_id: str
    history_start_date: date
    history_end_date: date
    prediction_start_date: date
    prediction_end_date: date
    case_directory: Path
    historical_solutions_path: Path


class QueueWindowInfo(StrictRecord):
    """Forecast origin and local queue category for one forecast window.

    Attributes:
        window_id: Identifier shared by the history, target, and answer rows.
        forecast_origin_utc: Last timestamp in the visible history and start of
            the forecast interval.
        condition: ``queue_ongoing`` when any link was queued at two or more
            history timestamps, otherwise ``queue_onset``. Both categories
            include at least one queued target row.
    """

    window_id: str
    forecast_origin_utc: datetime
    condition: QueueCondition


class QueueCase(StrictRecord):
    """Batch of generated queue windows for one panel and prediction interval.

    Attributes:
        task: Fixed task identifier used to select the case schema.
        panel: Panel whose traffic and network produce the windows.
        family_id: Freeway family used to aggregate this panel's score.
        case_id: Stable identifier for the selected prediction interval.
        history_start_date: Inclusive first date of visible train history.
        history_end_date: Exclusive end of the earlier train-label interval.
        prediction_start_date: Inclusive first date of generated forecast targets.
        prediction_end_date: Exclusive end of the generated target interval.
        case_directory: Relative directory containing the panel's case tables.
        historical_solutions_path: Relative path to historical known labels.
        windows: Origins and conditions for the generated forecast windows.
    """

    task: Literal["queue"] = "queue"
    panel: Panel
    family_id: str
    case_id: str
    history_start_date: date
    history_end_date: date
    prediction_start_date: date
    prediction_end_date: date
    case_directory: Path
    historical_solutions_path: Path
    windows: list[QueueWindowInfo]


class OdmeCase(StrictRecord):
    """One published-count diagnostic or synthetic path-flow case.

    Attributes:
        task: Fixed task identifier used to select the case schema.
        panel: Panel whose network and OD inputs define the scenario.
        family_id: Freeway family used to aggregate this panel's score.
        case_id: Stable identifier for the generated case.
        history_start_date: Inclusive first date of visible train history.
        history_end_date: Exclusive end of the earlier train-label interval.
        prediction_start_date: Inclusive first date of the selected interval.
        prediction_end_date: Exclusive end of the selected interval.
        case_directory: Relative directory containing scenario tables.
        scenario_name: Label identifying the released or synthetic scenario.
    """

    task: Literal["odme"] = "odme"
    panel: Panel
    family_id: str
    case_id: str
    history_start_date: date
    history_end_date: date
    prediction_start_date: date
    prediction_end_date: date
    case_directory: Path
    scenario_name: str


type BenchmarkCase = Annotated[
    StateCase | QueueCase | OdmeCase, Field(discriminator="task")
]


class BenchmarkManifest(StrictRecord):
    """Case locations and checksums for one prepared benchmark directory.

    ``benchmark_id`` covers the other manifest fields. ``files`` holds hashes
    of the prepared Parquet files so a runner can detect later changes.

    Attributes:
        version: Trafficbench format version used to prepare the cases.
        profile: Preparation profile selected for this benchmark.
        scheme: Validated case-generation configuration.
        data_fingerprint: Identity of the source release inventory.
        cases: Case records with their panel, dates, and relative paths.
        warnings: Preparation notes that should appear in run reports.
        files: Relative Parquet paths mapped to content hashes.
        benchmark_id: Hash identifying the remaining manifest content.
    """

    version: str
    profile: Profile
    scheme: BenchmarkConfig
    data_fingerprint: str
    cases: list[BenchmarkCase]
    warnings: list[str]
    files: dict[str, str]
    benchmark_id: str

    def hash_content(self) -> dict[str, object]:
        """Return the manifest fields included in the benchmark identifier.

        Returns:
            JSON-compatible manifest content with ``benchmark_id`` omitted.
        """
        return self.model_dump(mode="json", exclude={"benchmark_id"}, exclude_none=True)

    @property
    def seed(self) -> int:
        """Return the random seed used by local case generation.

        Returns:
            Seed stored in the benchmark's validated configuration.
        """
        return self.scheme.seed


type MetricScalar = str | int | float
type MetricRow = dict[str, MetricScalar]


class RunMetadata(StrictRecord):
    """Saved run identity, environment, status, resource use, and scores.

    A record starts with ``status='running'``. On success it gains metrics;
    on failure it gains an error traceback. Baseline fields are added only
    after a compatible baseline run has been found or computed.

    Attributes:
        run_id: Unique directory and report identifier.
        solution: Importable solution package name.
        task: Prediction task evaluated by the run.
        params: JSON settings passed to the solution.
        status: Current lifecycle state of the run.
        timestamp: UTC time when the run started.
        benchmark_id: Prepared benchmark identity used for this run.
        profile: Benchmark preparation profile recorded in the manifest.
        seed: Random seed used for repeatable local case selection.
        panels: Panels represented by the selected cases.
        commit: Git commit recorded at run start, if available.
        branch: Git branch recorded at run start, if available.
        dirty: Whether working-tree changes were present at run start.
        code_hash: Hash of source, configuration, and package metadata files.
        evaluator_hash: Hash of the scoring and evaluation code.
        baseline_hash: Hash of the baseline solution source.
        python: Python version recorded for reproducibility.
        platform: Operating-system description recorded for reproducibility.
        dependencies: Installed distribution names mapped to versions.
        warnings: Preparation warnings carried into the run report.
        metrics: Aggregated scores, present after successful completion.
        error: Traceback text, present when execution fails.
        seconds: Elapsed wall-clock time, when execution has started.
        peak_memory_mb: Peak process memory in megabytes, when measured.
        baseline_run: Matching baseline run identifier, when available.
        baseline_metrics: Matching baseline scores, when available.
        panel_runs: Component run IDs when hosted panel runs are combined.
    """

    run_id: str
    solution: str
    task: Task
    params: JsonObject
    status: RunStatus
    timestamp: str
    benchmark_id: str
    profile: Profile
    seed: int
    panels: list[Panel]
    commit: str | None
    branch: str | None
    dirty: bool
    code_hash: str
    evaluator_hash: str
    baseline_hash: str
    python: str
    platform: str
    dependencies: dict[str, str]
    warnings: list[str]
    metrics: dict[str, float] | None = None
    error: str | None = None
    seconds: float | None = None
    peak_memory_mb: float | None = None
    baseline_run: str | None = None
    baseline_metrics: dict[str, float] | None = None
    panel_runs: list[str] | None = None


class BundleShard(StrictRecord):
    """Archive name, checksum, panel, and tasks for one hosted job.

    Attributes:
        panel: Panel included in the archive.
        asset: Archive filename uploaded as a CI release asset.
        sha256: Digest verified after the hosted runner downloads the archive.
        tasks: Tasks with prepared cases for this panel.
    """

    panel: Panel
    asset: str
    sha256: str
    tasks: list[Task]


class BundleIndex(StrictRecord):
    """All panel archives packaged from one prepared benchmark ID.

    Attributes:
        profile: Preparation profile shared by every panel archive.
        benchmark_id: Prepared benchmark identity shared by every shard.
        shards: Archive records, one for each panel with prepared cases.
    """

    profile: Profile
    benchmark_id: str
    shards: list[BundleShard]


class MatrixEntry(StrictRecord):
    """Panel archive name and checksum passed to one hosted job.

    Attributes:
        panel: Panel evaluated by the hosted job.
        asset: Archive filename downloaded by that job.
        sha256: Expected archive checksum.
    """

    panel: Panel
    asset: str
    sha256: str


class CiMatrix(StrictRecord):
    """Hosted job matrix serialized for the continuous-integration workflow.

    Attributes:
        include: Panel archive entries selected for the requested task.
    """

    include: list[MatrixEntry]


type PanelSplitTableMap[T] = dict[Panel, dict[Split, T]]

# Match the public CSV filename stem in the Python field and type names.
LWR_MAINLINE_TOPOLOGY_FILE_STEM = "lwr_mainline_topology"


class ReleasePanel(BaseModel):
    """Panel identity and family from the public corridor configuration.

    Attributes:
        corridor_id: Directional panel identifier used in release paths.
        family_id: Freeway family containing this directional panel.
    """

    model_config = ConfigDict(strict=True)
    corridor_id: Panel
    family_id: str


class ReleaseCorridors(BaseModel):
    """Panel records from the public release's corridor configuration.

    Attributes:
        panels: All directional panel and freeway-family records.
    """

    model_config = ConfigDict(strict=True)
    panels: list[ReleasePanel]


@dataclass(frozen=True)
class NetworkTables:
    """Typed network tables for one directional panel.

    All seven named tables are present in the public release for every panel.
    ``additional_network_tables`` keeps any other network CSV available by its
    filename stem, so a solution can use release data that has no named field.

    Attributes:
        links: Mainline link lengths, lane counts, speed limits, and capacities.
        fd_parameters: Link attributes plus critical and jam densities.
        path_set: Candidate paths with their origin and destination zones.
        path_link_incidence: Rows linking candidate paths to mainline links.
        lwr_mainline_topology: Directed mainline links,
            their neighboring links, and boundary connections used by the
            Lighthill–Whitham–Richards (LWR) traffic-flow model.
        ramp_attachment_map: Mainline link for each ramp ID in the ramp
            observations. The official Task 3 scorer reads this table.
        synthetic_ramp_attachment_map: Mainline link for each synthetic ramp
            ID. Its rows match ``ramp_attachment_map`` in the current public
            release.
        additional_network_tables: Other network CSV tables keyed by filename
            stem.
    """

    links: table_types.LinksFrame
    fd_parameters: table_types.FdParametersFrame
    path_set: table_types.PathSetFrame
    path_link_incidence: table_types.PathLinkIncidenceFrame
    lwr_mainline_topology: table_types.LwrMainlineTopologyFrame
    ramp_attachment_map: table_types.RampAttachmentFrame
    synthetic_ramp_attachment_map: table_types.RampAttachmentFrame
    additional_network_tables: Mapping[str, pl.LazyFrame] = field(
        default_factory=dict[str, pl.LazyFrame]
    )

    @classmethod
    def from_mapping(cls, network_tables: Mapping[str, pl.LazyFrame]) -> NetworkTables:
        """Build the network object from tables keyed by file stem.

        Args:
            network_tables: Tables indexed by their release CSV filename stems.

        Returns:
            A network object with all seven named tables. Unrecognized network
            CSV tables remain available through ``additional_network_tables``.

        Raises:
            ValueError: If a named network table from the public release is absent.
        """
        # File discovery produces a mapping keyed by CSV or Parquet stem. Check
        # required names before assigning stable attributes to the tables.
        required_names = {
            "links",
            "fd_parameters",
            "path_set",
            "path_link_incidence",
            LWR_MAINLINE_TOPOLOGY_FILE_STEM,
            "ramp_attachment_map",
            "synthetic_ramp_attachment_map",
        }
        missing_table_names = required_names - network_tables.keys()
        if missing_table_names:
            raise ValueError(
                f"Network is missing required tables: {sorted(missing_table_names)}"
            )
        return cls(
            links=network_tables["links"],
            fd_parameters=network_tables["fd_parameters"],
            path_set=network_tables["path_set"],
            path_link_incidence=network_tables["path_link_incidence"],
            lwr_mainline_topology=network_tables[LWR_MAINLINE_TOPOLOGY_FILE_STEM],
            ramp_attachment_map=network_tables["ramp_attachment_map"],
            synthetic_ramp_attachment_map=network_tables[
                "synthetic_ramp_attachment_map"
            ],
            additional_network_tables={
                table_name: table
                for table_name, table in network_tables.items()
                if table_name not in required_names
            },
        )

    def items(self) -> Iterator[tuple[str, pl.LazyFrame]]:
        """Yield all named and additional network tables by their file stems.

        Yields:
            Each loaded network table paired with its release filename stem.
        """
        yield "links", self.links
        yield "fd_parameters", self.fd_parameters
        yield "path_set", self.path_set
        yield "path_link_incidence", self.path_link_incidence
        yield (
            LWR_MAINLINE_TOPOLOGY_FILE_STEM,
            self.lwr_mainline_topology,
        )
        yield "ramp_attachment_map", self.ramp_attachment_map
        yield "synthetic_ramp_attachment_map", self.synthetic_ramp_attachment_map
        yield from self.additional_network_tables.items()


class HistoricalTaskLabels(NamedTuple):
    """Known Task 1 answers and local Task 2 queue labels for one panel.

    Both tables cover the earlier train interval in the parent
    ``ReleasePackageSlice``. Task 1 rows contain the released speed and flow
    answers for masked target cells. Task 2 rows identify a link and time. Their
    queue labels are calculated from unmasked train speeds because the official
    queue answers are not released. These local labels may differ from the
    organizer's labels.

    Attributes:
        task1_state_answers: Released speed and flow values for historical Task 1 rows.
        task2_queue_proxy_labels: Queue labels calculated from train speeds for
            historical Task 2 link and timestamp rows. They may differ from the
            organizer's queue answers.
    """

    task1_state_answers: table_types.StateFrame
    task2_queue_proxy_labels: table_types.QueueFrame


@dataclass(frozen=True)
class PanelReleaseSlice:
    """Published network and traffic tables plus historical labels for one panel.

    Dated tables contain published observations, including masked train rows.
    ``historical_task_labels`` stores released Task 1 answers and queue labels
    calculated from train speeds. The parent release slice defines their date
    interval. Link counts and weak priors describe complete origin-destination
    scenarios for each split; those tables have no date column. Each table is a
    Polars ``LazyFrame`` and its rows are read when a solution collects it.

    Attributes:
        family_id: Freeway family used to group panel scores.
        network: Static network tables for this panel.
        masked_mainline_states: Published mainline rows, grouped by split.
        ramp_states: Published ramp rows, grouped by split.
        queue_history: Published queue-window histories, grouped by split.
        queue_window_index: Published queue-window metadata, grouped by split.
        link_counts: Mainline counts for each origin-destination scenario,
            grouped by split.
        weak_prior: Starting path-flow estimates for each origin-destination
            scenario, grouped by split.
        historical_task_labels: Released state answers and queue labels
            calculated from train speeds.
    """

    family_id: str
    network: NetworkTables
    masked_mainline_states: Mapping[Split, table_types.MaskedMainlineStatesFrame]
    ramp_states: Mapping[Split, table_types.RampStatesFrame]
    queue_history: Mapping[Split, table_types.QueueHistoryFrame]
    queue_window_index: Mapping[Split, table_types.QueueWindowIndexFrame]
    link_counts: Mapping[Split, table_types.LinkCountsFrame]
    weak_prior: Mapping[Split, table_types.WeakPriorFrame]
    historical_task_labels: HistoricalTaskLabels


@dataclass(frozen=True)
class ReleasePackageSlice:
    """Lazy release scans and known historical labels for one experiment.

    Split-keyed release inputs cover ``[history_start_date,
    prediction_end_date)``. Train rows are the public masked tables, even
    before ``history_end_date``. ``prediction_start_date`` and
    ``prediction_end_date`` select requested targets, while released inputs
    in the gap after the history interval remain visible. Historical Task 1
    answers and Task 2 queue proxy labels are stored in
    ``panel_slices_by_panel``. Future target keys remain a separate argument.
    This object never contains unmasked measurements or future answer values.
    Tables remain Polars ``LazyFrame`` query plans until a solution collects
    them, so unused task tables do not occupy memory.

    Attributes:
        history_start_date: Inclusive first date of visible release inputs.
        history_end_date: Exclusive end of the earlier train-label interval.
        prediction_start_date: Inclusive first date requested as a target.
        prediction_end_date: Exclusive end of visible inputs and target selection.
        panel_slices_by_panel: Selected panel data, indexed by panel ID.
        source_fingerprint: Identity of the extracted release file inventory.
        parameters: JSON-compatible solution settings for this run.
        seed: Random seed supplied to the solution.
    """

    history_start_date: date
    history_end_date: date
    prediction_start_date: date
    prediction_end_date: date
    panel_slices_by_panel: Mapping[Panel, PanelReleaseSlice]
    source_fingerprint: str
    parameters: JsonObject = field(default_factory=dict[str, JsonValue])
    seed: int = 0


KEYS: Final[dict[Task, tuple[KeyColumn, ...]]] = {
    "state": ("panel", "timestamp", "station_id", "link_id", "mask_regime"),
    "queue": ("window_id", "timestamp", "link_id"),
    "odme": ("panel", "departure_time", "path_id", "origin_zone", "destination_zone"),
}
VALUES: Final[dict[Task, tuple[ValueColumn, ...]]] = {
    "state": ("speed_kmh", "flow_vph"),
    "queue": ("queue_pred",),
    "odme": ("path_flow",),
}


def normalize_lazy(input_table: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize identifier and timestamp expressions in a lazy table.

    Preserve identifier spelling, including leading zeros, and convert
    timestamps to UTC. Invalid date text fails when a solution collects the
    scan. Prediction validation rejects null identifiers at the output boundary.

    Args:
        input_table: Lazy table whose task-key columns need consistent types.

    Returns:
        A lazy table with text identifiers and UTC timestamps.

    """
    # Only task key columns are identifiers. Numeric measurements keep their
    # original types for scoring and model calculations.
    input_schema = input_table.collect_schema()
    identifier_columns = {
        column
        for columns in KEYS.values()
        for column in columns
        if column != "timestamp" and column in input_schema
    }
    expressions = [pl.col(column).cast(pl.String) for column in identifier_columns]
    if "timestamp" in input_schema:
        timestamp_dtype = input_schema["timestamp"]
        timestamp = pl.col("timestamp")
        if timestamp_dtype == pl.String:
            timestamp = timestamp.str.to_datetime(time_zone="UTC", strict=True)
        elif isinstance(timestamp_dtype, pl.Datetime):
            timestamp = (
                timestamp.dt.replace_time_zone("UTC")
                if timestamp_dtype.time_zone is None
                else timestamp.dt.convert_time_zone("UTC")
            )
        expressions.append(timestamp.alias("timestamp"))
    return input_table.with_columns(expressions)


def validate_predictions(
    task: Task, requested_target_table: pl.LazyFrame, prediction_table: object
) -> pl.DataFrame:
    """Check a solution's output and align values to the requested targets.

    The task selects the natural key and required value columns. The requested
    target table defines the exact rows and their output order. Because solution
    functions are imported dynamically, validate their returned LazyFrame at
    this boundary before scoring or submission assembly.

    Args:
        task: Task whose key and value columns define the prediction table.
        requested_target_table: Lazy requested rows used for key matching and order.
        prediction_table: Dynamically returned table to validate as a LazyFrame.

    Returns:
        Predictions aligned to target order, with finite numeric values.

    Raises:
        TypeError: If the solution did not return a Polars LazyFrame.
        ValueError: If columns, keys, row counts, or task-specific values are invalid.
    """
    # Solution code is imported dynamically, so its return type must be
    # checked at this boundary even though the entry point is type annotated.
    if not isinstance(prediction_table, pl.LazyFrame):
        raise TypeError(f"{task} must return a Polars LazyFrame")
    key_columns, value_columns = KEYS[task], VALUES[task]
    prediction_table = prediction_table.collect()
    requested_keys = requested_target_table.select(key_columns).collect()
    missing_columns = set(key_columns + value_columns) - set(prediction_table.columns)
    if missing_columns:
        raise ValueError(f"{task}: missing columns {sorted(missing_columns)}")

    # Normalize task keys before checking uniqueness and matching target rows.
    requested_keys = (
        normalize_lazy(requested_keys.lazy()).collect().with_row_index("_order")
    )
    returned_predictions = normalize_lazy(
        prediction_table.select(key_columns + value_columns).lazy()
    ).collect()
    for label, key_table in (
        ("targets", requested_keys),
        ("predictions", returned_predictions),
    ):
        if (
            key_table.select(
                pl.any_horizontal(pl.col(key_columns).is_null()).any()
            ).item()
            or key_table.select(pl.struct(key_columns).is_duplicated().any()).item()
        ):
            raise ValueError(f"{task}: null or duplicate keys in {label}")
    requested_row_count = requested_keys.height
    if returned_predictions.height != requested_row_count:
        raise ValueError(
            f"{task}: expected {requested_row_count} rows, "
            f"got {returned_predictions.height}"
        )

    # Anti-joins detect missing and unknown keys. The final join restores the
    # official target order while keeping prediction values separate by name.
    if (
        requested_keys.select(key_columns)
        .join(returned_predictions.select(key_columns), on=key_columns, how="anti")
        .height
        or returned_predictions.select(key_columns)
        .join(requested_keys.select(key_columns), on=key_columns, how="anti")
        .height
    ):
        raise ValueError(f"{task}: missing or unknown target keys (including OD zones)")
    aligned_predictions = (
        requested_keys.join(
            returned_predictions,
            on=key_columns,
            how="left",
            validate="1:1",
        )
        .sort("_order")
        .drop("_order")
    )

    # Numeric conversion catches text. Finite checks reject null, NaN, infinity.
    for value_column in value_columns:
        aligned_predictions = aligned_predictions.with_columns(
            pl.col(value_column).cast(pl.Float64, strict=True)
        )
        if not aligned_predictions.select(
            pl.col(value_column).is_finite().fill_null(False).all()
        ).item():
            raise ValueError(f"{task}: {value_column} must be finite")
    match task:
        case "queue":
            if not aligned_predictions.select(
                pl.col("queue_pred").is_in([0, 1]).all()
            ).item():
                raise ValueError("queue_pred must be binary, not probabilities")
        case "odme":
            if aligned_predictions.select((pl.col("path_flow") < 0).any()).item():
                raise ValueError("path_flow must be nonnegative")
        case "state":
            pass
    return aligned_predictions
