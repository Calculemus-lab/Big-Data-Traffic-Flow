"""Small deterministic synthetic release for testing plumbing, not model quality."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import (
    KEYS,
    LWR_MAINLINE_TOPOLOGY_FILE_STEM,
    MASK_REGIMES,
    MASKED_STATE_COLUMNS,
    QUEUE_INTERVAL_MINUTES,
    Panel,
    ReleaseCorridors,
    ReleasePanel,
)
from .data import SplitCalendarEntry, SyntheticReleaseCalendar, save_table


def make_fixture(fixture_release_directory: Path) -> Path:
    """Write a small release that exercises the complete benchmark workflow.

    The fixture contains one panel, a two-link network, dated traffic states,
    masked Task 1 inputs, ramp observations, and Task 4 count inputs. It lets
    integration checks run without downloaded competition data. The destination
    must be empty. Return that directory after writing the release files.

    Args:
        fixture_release_directory: New or empty directory for synthetic files.

    Returns:
        The directory containing the generated release configuration and data.

    Raises:
        ValueError: If the destination exists and contains files.
    """
    if fixture_release_directory.exists() and any(fixture_release_directory.iterdir()):
        raise ValueError("Fixture output must be empty")
    panel: Panel = "D12_I5_N"
    upstream_link, downstream_link = "L01", "L02"
    first_path, second_path = "P01", "P02"

    # The fixture carries its own calendar, so CI and tests do not depend on
    # the ignored Kaggle download being present in the checkout.
    (fixture_release_directory / "config").mkdir(parents=True)
    synthetic_release_calendar = SyntheticReleaseCalendar(
        synthetic_calendar={
            "train": SplitCalendarEntry(
                start=datetime(2030, 6, 1, tzinfo=UTC),
                end_exclusive=datetime(2031, 3, 1, tzinfo=UTC),
            ),
            "validation": SplitCalendarEntry(
                start=datetime(2031, 3, 1, tzinfo=UTC),
                end_exclusive=datetime(2031, 4, 1, tzinfo=UTC),
            ),
            "private": SplitCalendarEntry(
                start=datetime(2031, 4, 1, tzinfo=UTC),
                end_exclusive=datetime(2031, 5, 1, tzinfo=UTC),
            ),
        }
    )
    (fixture_release_directory / "config/synthetic_release_v1.json").write_text(
        synthetic_release_calendar.model_dump_json()
    )
    (fixture_release_directory / "config/corridors.json").write_text(
        ReleaseCorridors(
            panels=[ReleasePanel(corridor_id=panel, family_id="D12_I5")]
        ).model_dump_json()
    )
    corridor = fixture_release_directory / "corridors" / panel
    network = corridor / "network"
    measured_dir = corridor / "train" / "mainline_states"
    masked_dir = corridor / "train" / "mainline_states_masked"
    ramp_dir = corridor / "train" / "ramp_states"
    network.mkdir(parents=True)

    # Candidate paths, path-to-link rows, and link counts share these IDs so
    # Task 4 can compare path flows with measured link totals.
    links = pl.DataFrame(
        {
            "link_id": [upstream_link, downstream_link],
            "length_km": [1.0, 1.0],
            "lanes": [2.0, 4.0],
            "free_speed_kmh": [100.0, 100.0],
            "capacity_vph": [4000.0, 8000.0],
            "critical_density": [40.0, 80.0],
            "k_jam": [200.0, 400.0],
        }
    )
    links.write_csv(network / "links.csv")
    links.write_csv(network / "fd_parameters.csv")
    # Keep every network table required by the public release contract in the
    # fixture so solutions can use the same named fields during local tests.
    topology = pl.DataFrame(
        {
            "mainline_link_id": [upstream_link, downstream_link],
            "incoming_link_ids": ["", upstream_link],
            "outgoing_link_ids": [downstream_link, ""],
            "on_ramp_link_ids": ["R01", ""],
            "off_ramp_link_ids": ["", "R02"],
            "incoming_has_sensor": ["", "1"],
            "outgoing_has_sensor": ["1", ""],
            "lanes": [2, 4],
            "length_km": [1.0, 1.0],
            "capacity_vph": [4000.0, 8000.0],
            "free_speed_kmh": [100.0, 100.0],
            "order_index": [0, 1],
            "link_id": [upstream_link, downstream_link],
            "from_node": [1, 2],
            "to_node": [2, 3],
            "has_sensor": [1, 1],
            "detector_id": ["S01", "S02"],
            "n_incoming": [0, 1],
            "n_outgoing": [1, 0],
            "n_incoming_no_sensor": [0, 0],
            "n_outgoing_no_sensor": [0, 0],
            "free_flow_speed_kmh": [100.0, 100.0],
        }
    )
    topology.write_csv(network / f"{LWR_MAINLINE_TOPOLOGY_FILE_STEM}.csv")
    ramp_attachments = pl.DataFrame(
        {
            "ramp_link_id": ["R01", "R02"],
            "ramp_type": ["OR", "FR"],
            "nearest_mainline_link_id": [upstream_link, downstream_link],
        }
    )
    ramp_attachments.write_csv(network / "ramp_attachment_map.csv")
    ramp_attachments.with_columns(
        pl.col("ramp_link_id").str.replace("R", "SYN_RAMP_")
    ).write_csv(network / "synthetic_ramp_attachment_map.csv")
    paths = pl.DataFrame(
        {
            "path_id": [first_path, second_path],
            "origin_zone": ["Z0", "Z0"],
            "destination_zone": ["Z1", "Z2"],
            "link_seq": [upstream_link, f"{upstream_link};{downstream_link}"],
        }
    )
    paths.write_csv(network / "path_set.csv")
    pl.DataFrame(
        {
            "path_id": [first_path, second_path, second_path],
            "link_id": [upstream_link, upstream_link, downstream_link],
        }
    ).write_csv(network / "path_link_incidence.csv")

    # The same daily rows serve as unmasked training data and as held-out answers
    # for reconstruction and queue proxy cases after the selected history end.
    random_generator = np.random.default_rng(42)
    targets: list[pl.DataFrame] = []
    # Each masking fraction corresponds to the regime at the same position.
    mask_fractions = (0.2, 0.3, 0.5)
    first_day = date(2030, 11, 3)
    for day_offset in range(35):
        day = first_day + timedelta(days=day_offset)
        first_timestamp = datetime.combine(day, datetime.min.time(), UTC)
        timestamps = [
            first_timestamp + index * timedelta(minutes=QUEUE_INTERVAL_MINUTES)
            for index in range(24 * 60 // QUEUE_INTERVAL_MINUTES)
        ]
        frame = pl.DataFrame({"timestamp": timestamps}).join(
            pl.DataFrame({"link_id": [upstream_link, downstream_link]}), how="cross"
        )
        frame = frame.with_columns(
            pl.lit(panel).alias("corridor_id"),
            pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("date"),
            pl.when(pl.col("link_id") == upstream_link)
            .then(pl.lit("S01"))
            .otherwise(pl.lit("S02"))
            .alias("station_id"),
            pl.when(pl.col("link_id") == upstream_link)
            .then(pl.lit(0.5))
            .otherwise(pl.lit(1.5))
            .alias("milepost"),
            pl.lit("N").alias("direction"),
            pl.when(pl.col("timestamp").dt.hour().is_between(8, 9))
            .then(pl.lit(40.0))
            .otherwise(pl.lit(95.0))
            .alias("speed_kmh"),
            pl.when(pl.col("link_id") == upstream_link)
            .then(pl.lit(2000.0))
            .otherwise(pl.lit(4000.0))
            .alias("flow_vph"),
            pl.lit(0.2).alias("occupancy"),
            pl.lit(30.0).alias("density_occ_linear_vehpkm"),
            pl.lit(100).alias("pct_observed"),
            pl.lit(1).alias("is_observed"),
            pl.lit(0).alias("is_imputed"),
            pl.lit(1).alias("is_score_eligible"),
            pl.lit(0).alias("is_missing"),
        )
        date_stamp = day.strftime("%Y_%m_%d")
        save_table(measured_dir / f"synthetic_mainline_{date_stamp}.parquet", frame)
        regime_index = day.day % len(MASK_REGIMES)
        regime = MASK_REGIMES[regime_index]
        observed = frame.with_columns(pl.lit(regime).alias("mask_regime"))
        mask = random_generator.random(len(frame)) < mask_fractions[regime_index]
        target = (
            observed.filter(pl.Series("masked", mask))
            .select("timestamp", "station_id", "link_id", "mask_regime")
            .with_columns(pl.lit(panel).alias("panel"))
        )
        targets.append(target.select(KEYS["state"]))

        # Hide midday measurements to simulate gaps around queue forecast windows.
        hidden_cells = pl.Series("hidden", mask) | frame["timestamp"].dt.hour().eq(12)
        observed = observed.with_columns(
            pl.when(hidden_cells).then(None).otherwise(pl.col(column)).alias(column)
            for column in MASKED_STATE_COLUMNS
        )
        save_table(
            masked_dir
            / f"mask_regime={regime}"
            / f"synthetic_mainline_{date_stamp}.parquet",
            observed,
        )
        ramps = frame.select(
            "corridor_id",
            "date",
            "timestamp",
            "station_id",
            "flow_vph",
            "pct_observed",
            "is_observed",
            "is_imputed",
            "is_score_eligible",
            "is_missing",
        ).with_columns(
            pl.lit("R01").alias("ramp_link_id"), pl.lit("OR").alias("ramp_type")
        )
        save_table(ramp_dir / f"synthetic_ramp_{date_stamp}.parquet", ramps)

    # Task templates identify exactly which masked cells and candidate paths
    # preparation may turn into benchmark targets.
    task1 = fixture_release_directory / "task1" / panel / "train"
    task1.mkdir(parents=True)
    state_template = pl.concat(targets).with_columns(
        pl.lit(float("nan")).alias("speed_kmh"),
        pl.lit(float("nan")).alias("flow_vph"),
    )
    state_template.write_csv(task1 / "sample_submission_state.csv")
    task4 = fixture_release_directory / "task4" / panel / "train"
    task4.mkdir(parents=True)
    prior = paths.with_columns(
        pl.lit(panel).alias("panel"),
        pl.lit("TRAIN_PM").alias("departure_time"),
        pl.Series("path_flow", [900.0, 1200.0]),
    )
    prior.write_csv(task4 / "synthetic_weak_prior.csv")
    prior.write_csv(task4 / "sample_submission_path_flow.csv")
    pl.DataFrame(
        {
            "panel": panel,
            "link_id": [upstream_link, downstream_link],
            "count": [2200.0, 1300.0],
        }
    ).write_csv(task4 / "synthetic_link_counts.csv")
    print(f"Synthetic CI fixture: {fixture_release_directory}")
    return fixture_release_directory
