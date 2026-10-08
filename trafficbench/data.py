"""Read release tables from the extracted competition package.

Preparation asks for release-relative files and date ranges. This module
locates source files, preserves identifier text in CSV scans, and returns Polars
tables without modifying the downloaded release.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import UnionType
from typing import Any, Literal, TypeAliasType, get_args, get_origin, get_type_hints

import polars as pl
from pydantic import BaseModel, ConfigDict, FilePath

from . import table_types
from .contracts import (
    ALL_SPLITS,
    KEYS,
    MIN_OBSERVED_PERCENT,
    QUEUE_SPEED_FRACTION,
    VALUES,
    HistoricalTaskLabels,
    NetworkTables,
    Panel,
    PanelReleaseSlice,
    PanelSplitTableMap,
    ReleaseCorridors,
    ReleasePackageSlice,
    Split,
    Task,
    normalize_lazy,
)

type StateDataKind = Literal["mainline_states", "mainline_states_masked", "ramp_states"]

CSV_IDENTIFIER_COLUMNS = {column for columns in KEYS.values() for column in columns} | {
    "ramp_link_id"
}


def empty_typed_table(columns: type[object]) -> pl.LazyFrame:
    """Create an empty lazy table using the scalar types in a column schema.

    ``Literal`` column annotations describe restricted values such as panel IDs
    and integer booleans. Their underlying scalar type determines the Polars
    column type.

    Args:
        columns: Typed dictionary whose annotations describe the table columns.

    Returns:
        Empty lazy table with the declared string, integer, float, and datetime
        columns.
    """
    type_to_polars_dtype: dict[type[object], Any] = {
        str: pl.String,
        int: pl.Int64,
        float: pl.Float64,
        datetime: pl.Datetime("us", "UTC"),
    }
    # Nullable fields use their non-null scalar type, while literal aliases use
    # the type of their allowed values (for example, ``int`` for ``0`` and ``1``).
    schema: dict[str, Any] = {}
    for column_name, annotation in get_type_hints(columns).items():
        while isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        annotation_origin = get_origin(annotation)
        while annotation_origin is UnionType:
            scalar_type = next(
                field_type
                for field_type in get_args(annotation)
                if field_type is not type(None)
            )
            annotation = scalar_type
            while isinstance(annotation, TypeAliasType):
                annotation = annotation.__value__
            annotation_origin = get_origin(annotation)
        if annotation_origin is Literal:
            literal_value = get_args(annotation)[0]
            if isinstance(literal_value, str):
                scalar_type = str
            elif isinstance(literal_value, int):
                scalar_type = int
            else:
                raise TypeError(f"Unsupported table column literal: {literal_value!r}")
        else:
            scalar_type = annotation
        schema[column_name] = type_to_polars_dtype[scalar_type]
    return pl.DataFrame(schema=schema).lazy()


class SplitCalendarEntry(BaseModel):
    """One release split's Coordinated Universal Time (UTC) date boundaries.

    Attributes:
        start: First date included in the split, at 00:00 UTC.
        end_exclusive: First date after the split, at 00:00 UTC.
    """

    model_config = ConfigDict(strict=True)

    start: datetime
    end_exclusive: datetime


class SyntheticReleaseCalendar(BaseModel):
    """Published date boundaries from ``synthetic_release_v1.json``.

    Attributes:
        synthetic_calendar: Date intervals indexed by split name.
    """

    model_config = ConfigDict(strict=True)

    synthetic_calendar: dict[Split, SplitCalendarEntry]


def release_calendar(release_package: Release) -> dict[Split, tuple[date, date]]:
    """Read half-open split dates from the release's published configuration.

    Args:
        release_package: Open public release containing its calendar JSON.

    Returns:
        Each split's inclusive start and exclusive end dates.
    """
    calendar = SyntheticReleaseCalendar.model_validate_json(
        release_package.read_bytes("config/synthetic_release_v1.json")
    )
    return {
        split: (entry.start.date(), entry.end_exclusive.date())
        for split, entry in calendar.synthetic_calendar.items()
    }


def validate_experiment_dates(
    calendar: dict[Split, tuple[date, date]],
    history_start_date: date,
    history_end_date: date,
    prediction_start_date: date,
    prediction_end_date: date,
) -> None:
    """Require the history and prediction intervals to fit the release calendar.

    Args:
        calendar: Inclusive-start, exclusive-end dates for each release split.
        history_start_date: First train-history date to include.
        history_end_date: First date after the train-history interval.
        prediction_start_date: First target date to include.
        prediction_end_date: First date after the target interval.

    Raises:
        ValueError: If dates are out of order or outside the released calendar.
    """
    train_start_date, train_end_date = calendar["train"]
    private_end_date = calendar["private"][1]
    if not train_start_date <= history_start_date < history_end_date <= train_end_date:
        raise ValueError("History start and end dates must fall within the train split")
    if not history_end_date <= prediction_start_date < prediction_end_date:
        raise ValueError("Prediction dates must start at or after history end")
    if prediction_end_date > private_end_date:
        raise ValueError("Prediction end date exceeds the released calendar")


def _split_interval(
    split: Split,
    calendar: dict[Split, tuple[date, date]],
    start_date: date,
    end_date: date,
) -> tuple[date, date] | None:
    """Return the overlap between one split and a half-open date interval.

    Args:
        split: Release split whose boundaries are being intersected.
        calendar: Inclusive-start and exclusive-end dates for all splits.
        start_date: Inclusive start of the requested interval.
        end_date: Exclusive end of the requested interval.

    Returns:
        The non-empty overlap, or ``None`` when the intervals do not overlap.
    """
    split_start_date, split_end_date = calendar[split]
    intersect_start_date = max(split_start_date, start_date)
    intersect_end_date = min(split_end_date, end_date)
    return (
        (intersect_start_date, intersect_end_date)
        if intersect_start_date < intersect_end_date
        else None
    )


def metadata_hash(metadata_value: object) -> str:
    """Return a stable SHA-256 digest for JSON-serializable metadata.

    Args:
        metadata_value: Value encoded as sorted-key JSON before hashing.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(
        json.dumps(metadata_value, sort_keys=True).encode()
    ).hexdigest()


def file_hash(file_path: FilePath) -> str:
    """Return the SHA-256 digest of a file without loading it into memory.

    Args:
        file_path: Existing file whose bytes are hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    with file_path.open("rb") as file_stream:
        return hashlib.file_digest(file_stream, "sha256").hexdigest()


class Release:
    """Read files from the extracted public competition release.

    ``file_names`` lists paths relative to the release root. ``fingerprint``
    identifies the extracted file inventory without reading every byte of a
    large release. Constructing this object indexes file names and metadata
    but does not load table contents.
    """

    def __init__(self, release_directory: Path) -> None:
        """Validate and fingerprint one extracted release directory.

        The path must point to the directory containing ``config/`` and
        ``corridors/``. Downloaded archives must be extracted first.

        Args:
            release_directory: Extracted release root.

        Raises:
            ValueError: If the directory lacks the release configuration.
        """
        if (
            not release_directory.is_dir()
            or not (release_directory / "config/corridors.json").exists()
        ):
            raise ValueError(
                "Point --data at the extracted release directory containing config/"
            )
        self.release_directory = release_directory
        self.file_names = sorted(
            file_path.relative_to(release_directory).as_posix()
            for file_path in release_directory.rglob("*")
            if file_path.is_file()
        )
        # File metadata fingerprints the large release without reading all bytes.
        self.fingerprint = metadata_hash(
            [
                (
                    file_name,
                    (release_directory / file_name).stat().st_size,
                    (release_directory / file_name).stat().st_mtime_ns,
                )
                for file_name in self.file_names
            ]
        )

    def read_bytes(self, release_relative_file_path: str) -> bytes:
        """Read one release file by its path relative to the release root.

        Args:
            release_relative_file_path: POSIX-style path inside the release.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the path is absent from the release directory.
        """
        return (self.release_directory / release_relative_file_path).read_bytes()

    def read_csv(self, release_relative_file_path: str) -> pl.DataFrame:
        """Read a release CSV with identifiers preserved as text.

        Polars infers measurement types. Empty cells become null values, while
        identifier columns remain text, including strings such as ``NA``.

        Args:
            release_relative_file_path: POSIX-style path to a release CSV.

        Returns:
            A parsed DataFrame with identifier columns preserved as strings.
        """
        return self.scan_csv(release_relative_file_path).collect()

    def scan_csv(self, release_relative_file_path: str) -> pl.LazyFrame:
        """Create a lazy scan for one CSV without changing identifier spelling.

        Args:
            release_relative_file_path: POSIX-style path to a release CSV.

        Returns:
            A lazy table that reads the CSV when collected. Identifier columns
            stay strings, even when their values look numeric.
        """
        return normalize_lazy(
            pl.scan_csv(
                self.release_directory / release_relative_file_path,
                schema_overrides={
                    column: pl.String for column in CSV_IDENTIFIER_COLUMNS
                },
                null_values="",
                try_parse_dates=True,
            )
        )

    def scan_parquet(self, release_relative_file_path: str) -> pl.LazyFrame:
        """Create a lazy scan for one Parquet file in the release.

        Args:
            release_relative_file_path: POSIX-style path to a release Parquet file.

        Returns:
            A lazy table that reads the Parquet file when collected.
        """
        return normalize_lazy(
            pl.scan_parquet(self.release_directory / release_relative_file_path)
        )

    def read_csv_rows(
        self,
        release_relative_file_path: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Read a release CSV, optionally keeping a half-open timestamp range.

        The returned rows keep the file's full columns and are normalized like
        :meth:`read_csv`. ``start_date`` is included and ``end_date`` is excluded.

        Args:
            release_relative_file_path: POSIX-style path to a release CSV.
            start_date: Optional inclusive date filter for its ``timestamp``.
            end_date: Optional exclusive date filter for its ``timestamp``.
        Returns:
            All CSV columns for rows within the optional date interval.
        """
        table = self.scan_csv(release_relative_file_path)
        if start_date is not None:
            table = table.filter(
                pl.col("timestamp")
                >= datetime.combine(start_date, datetime.min.time(), UTC)
            )
        if end_date is not None:
            table = table.filter(
                pl.col("timestamp")
                < datetime.combine(end_date, datetime.min.time(), UTC)
            )
        return table.collect()

    def read_parquet(self, release_relative_file_path: str) -> pl.DataFrame:
        """Read one release Parquet file into an eager Polars table.

        Args:
            release_relative_file_path: POSIX-style path to a release Parquet file.

        Returns:
            The file's rows as a Polars DataFrame.
        """
        return self.scan_parquet(release_relative_file_path).collect()

    def scan_network_tables(self, panel: Panel) -> NetworkTables:
        """Create lazy scans for all network CSV files for one panel.

        Args:
            panel: Directional panel whose network tables are read.

        Returns:
            Named network tables and any additional tables, indexed by their
            release filename stems.

        Raises:
            ValueError: If a required named network CSV is missing.
        """
        prefix = f"corridors/{panel}/network/"
        tables = {
            Path(release_relative_path).stem: self.scan_csv(release_relative_path)
            for release_relative_path in self.file_names
            if release_relative_path.startswith(prefix)
            and release_relative_path.endswith(".csv")
        }
        return NetworkTables.from_mapping(tables)

    def read_state_rows(
        self,
        panel: Panel,
        split: Split,
        state_data_kind: StateDataKind,
        start_date: date,
        end_date: date,
        *,
        allow_empty: bool = False,
    ) -> pl.DataFrame:
        """Return state rows from ``start_date`` through before ``end_date``.

        ``split`` selects the release partition and ``state_data_kind`` selects
        its state directory, such as unmasked mainline or ramp states. File
        dates avoid loading unrelated Parquet files. UTC timestamps enforce
        the exact interval inside each file. Raise ValueError if no rows remain.

        Args:
            panel: Panel containing the requested detector records.
            split: Release interval to read.
            state_data_kind: Unmasked, masked, or ramp-state table family.
            start_date: Inclusive first date to return.
            end_date: Exclusive first date to omit.
            allow_empty: Return an empty schema-correct table when no rows match.

        Returns:
            Concatenated records with timestamps in ``[start_date, end_date)``.

        Raises:
            ValueError: If no rows match and ``allow_empty`` is false.
        """
        prefix = f"corridors/{panel}/{split}/{state_data_kind}/"
        start_timestamp_utc = datetime.combine(start_date, datetime.min.time(), UTC)
        end_timestamp_utc = datetime.combine(end_date, datetime.min.time(), UTC)
        daily_state_tables: list[pl.DataFrame] = []
        for release_relative_file_path in self.file_names:
            if not release_relative_file_path.startswith(
                prefix
            ) or not release_relative_file_path.endswith(".parquet"):
                continue

            # The daily filename is only a coarse filter. The timestamp
            # selection below decides which rows fall inside the interval.
            filename_date = re.search(
                r"(20\d\d)_([01]\d)_([0-3]\d)\.parquet$",
                release_relative_file_path,
            )
            if filename_date:
                file_start_utc = datetime.strptime(
                    "-".join(filename_date.groups()), "%Y-%m-%d"
                ).replace(tzinfo=UTC)
                file_end_utc = file_start_utc + timedelta(days=1)
                if (
                    file_end_utc <= start_timestamp_utc
                    or file_start_utc >= end_timestamp_utc
                ):
                    continue
            daily_state_table = (
                self.scan_parquet(release_relative_file_path)
                .filter(
                    pl.col("timestamp").is_between(
                        start_timestamp_utc, end_timestamp_utc, closed="left"
                    )
                )
                .collect()
            )
            if daily_state_table.height:
                daily_state_tables.append(daily_state_table)
        if not daily_state_tables and allow_empty:
            match state_data_kind:
                case "mainline_states":
                    empty_table_type = table_types.MainlineStateColumns
                case "mainline_states_masked":
                    empty_table_type = table_types.MaskedMainlineStateColumns
                case "ramp_states":
                    empty_table_type = table_types.RampStateColumns
            return empty_typed_table(empty_table_type).collect()
        if not daily_state_tables:
            raise ValueError(
                f"No {state_data_kind} records for {panel}/{split} "
                f"in [{start_date}, {end_date})"
            )
        return pl.concat(daily_state_tables, how="diagonal_relaxed")

    def scan_state_rows(
        self,
        panel: Panel,
        split: Split,
        state_data_kind: StateDataKind,
        start_date: date,
        end_date: date,
    ) -> pl.LazyFrame:
        """Scan daily state files that overlap a half-open date range.

        File names first exclude days outside the request. A timestamp filter
        then enforces the exact inclusive-start, exclusive-end interval.

        Args:
            panel: Panel containing the requested state records.
            split: Release split containing the daily files.
            state_data_kind: Mainline, masked-mainline, or ramp state directory.
            start_date: First included UTC date.
            end_date: First excluded UTC date.

        Returns:
            Lazy scan of matching rows from the selected files.
        """
        prefix = f"corridors/{panel}/{split}/{state_data_kind}/"
        start_timestamp = datetime.combine(start_date, datetime.min.time(), UTC)
        end_timestamp = datetime.combine(end_date, datetime.min.time(), UTC)
        daily_scans: list[pl.LazyFrame] = []
        for release_relative_file_path in self.file_names:
            if not release_relative_file_path.startswith(
                prefix
            ) or not release_relative_file_path.endswith(".parquet"):
                continue
            file_date_match = re.search(
                r"(20\d\d)_([01]\d)_([0-3]\d)\.parquet$",
                release_relative_file_path,
            )
            if file_date_match:
                file_date = date(*map(int, file_date_match.groups()))
                if not start_date <= file_date < end_date:
                    continue
            daily_scans.append(self.scan_parquet(release_relative_file_path))
        if not daily_scans:
            raise ValueError(
                f"No {state_data_kind} files for {panel}/{split} "
                f"in [{start_date}, {end_date})"
            )
        return pl.concat(daily_scans).filter(
            pl.col("timestamp").is_between(
                pl.lit(start_timestamp), pl.lit(end_timestamp), closed="left"
            )
        )


def load_release_slice(
    release_package: Release,
    history_start_date: date,
    history_end_date: date,
    prediction_start_date: date,
    prediction_end_date: date,
    panels: Sequence[Panel],
) -> ReleasePackageSlice:
    """Load selected release tables and earlier train labels for a solution run.

    Mainline and ramp rows cover the visible interval from
    ``history_start_date`` to ``prediction_end_date``. Queue windows are
    selected by forecast start, and each selected window keeps its full
    released history even when that history begins before the visible start.
    Complete OD count and prior tables are included for every intersecting
    split. The raw unmasked mainline table is used to build historical labels,
    but is not stored in the returned slice. Release, network, and historical
    label tables are lazy Polars scans until a solution collects them.

    Args:
        release_package: Open competition release package.
        history_start_date: First date of visible mainline and ramp data.
        history_end_date: Exclusive end of the earlier train history interval.
        prediction_start_date: First target date requested from templates.
        prediction_end_date: Exclusive end of visible and requested dates.
        panels: Directional panels loaded for one solution call.

    Returns:
        A ``ReleasePackageSlice`` with the selected panels, dates, and historical
        labels. Released input tables remain lazy until a solution collects them.
    """
    calendar = release_calendar(release_package)
    validate_experiment_dates(
        calendar,
        history_start_date,
        history_end_date,
        prediction_start_date,
        prediction_end_date,
    )
    selected_panels = tuple(dict.fromkeys(panels))
    if not selected_panels:
        raise ValueError("Select at least one panel")
    historical_labels_by_panel = load_historical_task_labels(
        release_package, history_start_date, history_end_date, selected_panels
    )

    family_ids = {
        panel.corridor_id: panel.family_id
        for panel in ReleaseCorridors.model_validate_json(
            release_package.read_bytes("config/corridors.json")
        ).panels
    }
    selected_split_ranges: dict[Split, tuple[date, date]] = {
        split: interval
        for split in ALL_SPLITS
        if (
            interval := _split_interval(
                split, calendar, history_start_date, prediction_end_date
            )
        )
    }
    panel_slices: dict[Panel, PanelReleaseSlice] = {}
    for panel in selected_panels:
        masked_mainline_states: dict[Split, table_types.MaskedMainlineStatesFrame] = {}
        ramp_states: dict[Split, table_types.RampStatesFrame] = {}
        queue_history: dict[Split, table_types.QueueHistoryFrame] = {}
        queue_window_index: dict[Split, table_types.QueueWindowIndexFrame] = {}
        link_counts: dict[Split, table_types.LinkCountsFrame] = {}
        weak_prior: dict[Split, table_types.WeakPriorFrame] = {}

        # Load only the public masked traffic and ramp files for each visible
        # date interval. Earlier answer values remain in separate label tables.
        for split, (split_start_date, split_end_date) in selected_split_ranges.items():
            masked_mainline_states[split] = release_package.scan_state_rows(
                panel,
                split,
                "mainline_states_masked",
                split_start_date,
                split_end_date,
            )
            ramp_states[split] = release_package.scan_state_rows(
                panel,
                split,
                "ramp_states",
                split_start_date,
                split_end_date,
            )

        # Keep only released queue windows whose forecast begins in this slice,
        # and retain each selected window's complete history table.
        for split, (
            visible_start_date,
            visible_end_date,
        ) in selected_split_ranges.items():
            queue_directory = f"task2/{panel}/{split}/"
            index_path = queue_directory + "window_index.csv"
            if index_path in release_package.file_names:
                visible_start_timestamp = datetime.combine(
                    visible_start_date, datetime.min.time(), UTC
                )
                visible_end_timestamp = datetime.combine(
                    visible_end_date, datetime.min.time(), UTC
                )
                index_table = release_package.scan_csv(index_path).filter(
                    pl.col("forecast_start").is_between(
                        visible_start_timestamp,
                        visible_end_timestamp,
                        closed="left",
                    )
                )
                history_path = queue_directory + "window_history.parquet"
                history_table = release_package.scan_parquet(history_path).join(
                    index_table.select("window_id"), on="window_id", how="semi"
                )
            else:
                index_table = empty_typed_table(table_types.QueueWindowIndexColumns)
                history_table = empty_typed_table(table_types.QueueHistoryColumns)
            queue_window_index[split] = index_table
            queue_history[split] = history_table

            # OD scenarios have no date rows, so include their complete tables
            # whenever their enclosing release split intersects the slice.
            odme_directory = f"task4/{panel}/{split}/"
            counts_path = odme_directory + "synthetic_link_counts.csv"
            prior_path = odme_directory + "synthetic_weak_prior.csv"
            link_counts[split] = (
                release_package.scan_csv(counts_path)
                if counts_path in release_package.file_names
                else empty_typed_table(table_types.LinkCountsColumns)
            )
            weak_prior[split] = (
                release_package.scan_csv(prior_path)
                if prior_path in release_package.file_names
                else empty_typed_table(table_types.WeakPriorColumns)
            )

        # Put every table under its source split and keep panel metadata beside
        # it so solution code follows the same panel-first nesting throughout.
        panel_slices[panel] = PanelReleaseSlice(
            family_id=family_ids[panel],
            network=release_package.scan_network_tables(panel),
            masked_mainline_states=masked_mainline_states,
            ramp_states=ramp_states,
            queue_history=queue_history,
            queue_window_index=queue_window_index,
            link_counts=link_counts,
            weak_prior=weak_prior,
            historical_task_labels=historical_labels_by_panel[panel],
        )

    return ReleasePackageSlice(
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        prediction_start_date=prediction_start_date,
        prediction_end_date=prediction_end_date,
        panel_slices_by_panel=panel_slices,
        source_fingerprint=release_package.fingerprint,
    )


def select_state_solutions(
    state_template: pl.LazyFrame, unmasked_state_table: pl.LazyFrame
) -> pl.LazyFrame:
    """Attach unmasked train answers to historical Task 1 target rows.

    Args:
        state_template: Lazy Task 1 target rows in the history interval.
        unmasked_state_table: Lazy unmasked train measurements for those dates.

    Returns:
        Lazy Task 1 target rows with measured speed and flow values.
    """
    # Keep template keys and attach measured answers only for score-eligible rows.
    return (
        state_template.select(KEYS["state"])
        .join(
            unmasked_state_table.select(
                "timestamp",
                "station_id",
                "link_id",
                "speed_kmh",
                "flow_vph",
                "pct_observed",
            ),
            on=["timestamp", "station_id", "link_id"],
            how="left",
            validate="1:1",
        )
        .filter(
            pl.col("speed_kmh").is_not_null()
            & pl.col("flow_vph").is_not_null()
            & pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
        )
        .select(KEYS["state"] + VALUES["state"])
    )


def select_queue_solutions(
    queue_template: pl.LazyFrame,
    unmasked_state_table: pl.LazyFrame,
    network: NetworkTables,
) -> pl.LazyFrame:
    """Estimate historical Task 2 queue labels from unmasked train speeds.

    Average station speeds for each link and timestamp. Keep a queue label only
    when every station measurement for that link and time is present and has
    at least 75% coverage. Mark the link queued when its mean speed is at most
    60% of its free-flow speed. These local estimates can differ from the
    withheld official queue answers.

    Args:
        queue_template: Lazy queue target rows to label.
        unmasked_state_table: Lazy train speed and detector-coverage measurements.
        network: Panel link table with each link's free-flow speed.

    Returns:
        Lazy eligible queue target rows with locally estimated labels.
    """
    # Collapse multiple detectors on one link using the same mean-speed rule
    # and all-detector coverage rule used by local queue case generation.
    # Average detector speeds on each link and require complete detector coverage.
    link_measurements = (
        unmasked_state_table.with_columns(
            (
                pl.col("pct_observed").ge(MIN_OBSERVED_PERCENT)
                & pl.col("speed_kmh").is_not_null()
            ).alias("_queue_eligible")
        )
        .group_by(["timestamp", "link_id"])
        .agg(
            pl.col("speed_kmh").mean(),
            pl.col("_queue_eligible").fill_null(False).all(),
        )
    )

    # Compare eligible link speeds with the published free-flow speed.
    return (
        queue_template.select(KEYS["queue"])
        .join(
            link_measurements,
            on=["timestamp", "link_id"],
            how="left",
            validate="m:1",
        )
        .join(
            network.links.select("link_id", "free_speed_kmh"),
            on="link_id",
            how="left",
            validate="m:1",
        )
        .filter(pl.col("_queue_eligible").fill_null(False))
        .with_columns(
            (pl.col("speed_kmh") <= QUEUE_SPEED_FRACTION * pl.col("free_speed_kmh"))
            .fill_null(False)
            .cast(pl.Int64)
            .alias("queue_pred")
        )
        .select(KEYS["queue"] + VALUES["queue"])
    )


def load_historical_task_labels(
    release_package: Release,
    history_start_date: date,
    history_end_date: date,
    panels: Sequence[Panel],
) -> dict[Panel, HistoricalTaskLabels]:
    """Load available Task 1 and Task 2 labels for each selected panel.

    The unmasked train measurements are used to build lazy Task 1 and Task 2
    label plans, but the measurements themselves are not returned. Task 1
    answers use official target rows. Task 2 labels are local estimates from
    eligible train speeds because official queue answers are withheld. No
    historical Task 4 path-flow answers are available.

    Args:
        release_package: Open competition release.
        history_start_date: First included date of the train history interval.
        history_end_date: First date after the train history interval.
        panels: Directional panels whose labels are loaded.

    Returns:
        A ``HistoricalTaskLabels`` record for each selected panel.
    """
    history_start_timestamp = datetime.combine(
        history_start_date, datetime.min.time(), UTC
    )
    history_end_timestamp = datetime.combine(history_end_date, datetime.min.time(), UTC)
    historical_labels_by_panel: dict[Panel, HistoricalTaskLabels] = {}

    for panel in panels:
        # Keep answer derivation as query plans. Solutions that use the
        # historical tables collect them, while other tasks need not load them.
        unmasked_train_history = release_package.scan_state_rows(
            panel,
            "train",
            "mainline_states",
            history_start_date,
            history_end_date,
        )
        state_template = release_package.scan_csv(
            f"task1/{panel}/train/sample_submission_state.csv"
        ).filter(
            pl.col("timestamp").is_between(
                history_start_timestamp, history_end_timestamp, closed="left"
            )
        )
        task1_state_answers = select_state_solutions(
            state_template, unmasked_train_history
        )

        # Local Task 2 labels are based on complete forecast windows whose
        # history and forecast end before this historical cutoff.
        window_index_path = f"task2/{panel}/train/window_index.csv"
        target_template_path = f"task2/{panel}/train/sample_submission_queue.csv"
        if (
            window_index_path in release_package.file_names
            and target_template_path in release_package.file_names
        ):
            historical_window_ids = (
                release_package.scan_csv(window_index_path)
                .filter(
                    pl.col("history_start").ge(history_start_timestamp)
                    & pl.col("forecast_end").lt(history_end_timestamp)
                )
                .select("window_id")
            )
            queue_targets = release_package.scan_csv(target_template_path).join(
                historical_window_ids, on="window_id", how="semi"
            )
            task2_queue_proxy_labels = select_queue_solutions(
                queue_targets,
                unmasked_train_history,
                release_package.scan_network_tables(panel),
            )
        else:
            task2_queue_proxy_labels = empty_typed_table(table_types.QueueColumns)
        historical_labels_by_panel[panel] = HistoricalTaskLabels(
            task1_state_answers=task1_state_answers,
            task2_queue_proxy_labels=task2_queue_proxy_labels,
        )
    return historical_labels_by_panel


def load_target_templates(
    release_package: Release,
    task: Task,
    prediction_start_date: date,
    prediction_end_date: date,
    panels: Sequence[Panel],
) -> PanelSplitTableMap[pl.DataFrame]:
    """Read official target templates intersecting one date range.

    State rows are selected by timestamp. Queue rows are selected by complete
    windows whose every forecast timestamp fits the requested interval. OD
    scenarios have no timestamp, so each intersecting split contributes its
    complete target table.

    Args:
        release_package: Open public release package.
        task: Competition task whose sample-submission rows are loaded.
        prediction_start_date: Inclusive beginning of the target interval.
        prediction_end_date: Exclusive end of the target interval.
        panels: Selected directional panels.

    Returns:
        Target template tables grouped by panel and intersecting split.
    """
    calendar = release_calendar(release_package)
    path_by_task: dict[Task, str] = {
        "state": "task1/{panel}/{split}/sample_submission_state.csv",
        "queue": "task2/{panel}/{split}/sample_submission_queue.csv",
        "odme": "task4/{panel}/{split}/sample_submission_path_flow.csv",
    }
    targets_by_panel: PanelSplitTableMap[pl.DataFrame] = {}
    for split in ALL_SPLITS:
        split_interval = _split_interval(
            split, calendar, prediction_start_date, prediction_end_date
        )
        if split_interval is None:
            continue
        split_start_date, split_end_date = split_interval
        for panel in panels:
            template_path = path_by_task[task].format(panel=panel, split=split)
            if template_path not in release_package.file_names:
                continue
            if task == "odme":
                target_table = release_package.read_csv(template_path)
            elif task == "queue":
                target_table = release_package.read_csv(template_path)
                target_start = datetime.combine(
                    prediction_start_date, datetime.min.time(), UTC
                )
                target_end = datetime.combine(
                    prediction_end_date, datetime.min.time(), UTC
                )
                # Keep all rows in a scored window. Row filtering could remove
                # forecast steps at either requested date boundary.
                window_ranges = (
                    target_table.group_by("window_id")
                    .agg(
                        pl.col("timestamp").min().alias("_first_target"),
                        pl.col("timestamp").max().alias("_last_target"),
                    )
                    .filter(
                        pl.col("_first_target") >= target_start,
                        pl.col("_last_target") < target_end,
                    )
                    .select("window_id")
                )
                target_table = target_table.join(
                    window_ranges, on="window_id", how="semi"
                )
            else:
                target_table = release_package.read_csv_rows(
                    template_path, split_start_date, split_end_date
                )
                target_table = target_table.filter(
                    pl.col("timestamp").is_between(
                        datetime.combine(
                            prediction_start_date, datetime.min.time(), UTC
                        ),
                        datetime.combine(prediction_end_date, datetime.min.time(), UTC),
                        closed="left",
                    )
                )
            targets_by_panel.setdefault(panel, {})[split] = target_table
    return targets_by_panel


def save_table(
    parquet_file_path: Path, table_data: pl.DataFrame | pl.LazyFrame
) -> None:
    """Create parent directories and write one table as Parquet.

    Args:
        parquet_file_path: Destination path for the Parquet file.
        table_data: Rows to serialize as a Parquet file.
    """
    parquet_file_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(table_data, pl.LazyFrame):
        table_data.sink_parquet(parquet_file_path)
    else:
        table_data.write_parquet(parquet_file_path)
