"""Check that parallel panel calls retain their input context and output order."""

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from trafficbench import table_types
from trafficbench.contracts import JsonObject, Panel, ReleasePackageSlice, Split
from trafficbench.data import Release, load_release_slice
from trafficbench.panelwise import map_panels


def _predict_marker(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    targets: Mapping[Split, table_types.QueueFrame],
    parameters: JsonObject,
) -> dict[Split, table_types.QueueFrame]:
    """Return fixed predictions after checking the worker's seed and panel inputs."""
    assert release_slice.seed == parameters["expected_seed"]
    assert targets["train"].collect()["window_id"].to_list() == [panel]
    return {
        split: frame.with_columns(pl.lit(1).alias("queue_pred"))
        for split, frame in targets.items()
    }


def test_parallel_panels_preserve_seed_targets_and_order(
    prepared_benchmark_directory: Path,
) -> None:
    """A parallel prediction call must match the same sequential panel call."""
    release_slice = load_release_slice(
        Release(prepared_benchmark_directory.parent / "release"),
        date(2030, 11, 3),
        date(2030, 11, 20),
        date(2030, 11, 20),
        date(2030, 11, 28),
        panels=("D12_I5_N",),
    )
    panels: tuple[Panel, ...] = ("D12_I5_N", "D7_I10_E")
    targets: dict[Panel, dict[Split, table_types.QueueFrame]] = {
        panel: {
            "train": pl.DataFrame(
                {
                    "window_id": [panel],
                    "timestamp": [datetime(2030, 11, 21, tzinfo=UTC)],
                    "link_id": ["L01"],
                    "queue_pred": [0],
                }
            ).lazy()
        }
        for panel in panels
    }
    parameters: JsonObject = {"expected_seed": release_slice.seed}
    sequential = map_panels(release_slice, targets, parameters, _predict_marker)
    parallel = map_panels(
        release_slice, targets, parameters, _predict_marker, workers=2
    )
    assert tuple(parallel) == panels
    for panel in panels:
        assert (
            parallel[panel]["train"]
            .collect()
            .equals(sequential[panel]["train"].collect())
        )
