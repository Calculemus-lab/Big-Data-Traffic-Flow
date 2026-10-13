"""Score saved st_cnn predictions separately inside simulated blackouts.

Run after ``bench experiment st_cnn ... --params '{"blackout_check": COUNT}'``::

    uv run --locked python -m solutions.st_cnn.blackout_check RUN_DIR BENCHMARK_DIR COUNT

The blackout rows are recomputed with ``blackout_starts``, so they match the
rows the solution blanked. Truth comes from the prepared benchmark's answer
tables, the same source the runner scores against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from trafficbench.contracts import KEYS

from .grid import epoch_step
from .model import BLACKOUT_STEPS, blackout_starts


def subset_scores(frame: pl.DataFrame) -> dict[str, float]:
    """Return row count, speed and per-lane flow RMSE, and the Task 1 score formula."""
    speed_error = frame["speed_kmh"].to_numpy() - frame["speed_kmh_true"].to_numpy()
    flow_error = (
        frame["flow_vph"].to_numpy() - frame["flow_vph_true"].to_numpy()
    ) / frame["lanes"].to_numpy()
    speed = float(np.sqrt(np.mean(speed_error**2)))
    flow = float(np.sqrt(np.mean(flow_error**2)))
    score = 0.54 * max(0.0, 1 - speed / 25) + 0.46 * max(0.0, 1 - flow / 600)
    return {"n": frame.height, "speed_rmse": speed, "flow_rmse": flow, "score": score}


def blackout_table(
    run_directory: Path, benchmark_directory: Path, count: int
) -> pl.DataFrame:
    """Return per-panel scores for targets inside and outside the simulated blackouts.

    Args:
        run_directory: A completed ``bench experiment`` run of st_cnn.
        benchmark_directory: The prepared benchmark the run used.
        count: The ``blackout_check`` setting of that run.

    Returns:
        One row per panel and subset (``blackout`` or ``other``).
    """
    rows: list[dict[str, object]] = []
    for prediction_path in sorted(run_directory.glob("predictions/*/*/state.parquet")):
        panel, case = prediction_path.parts[-3], prediction_path.parts[-2]
        truth = pl.read_parquet(
            benchmark_directory / panel / case / "state" / "truth.parquet"
        )
        lanes = pl.read_parquet(
            benchmark_directory / "network" / panel / "fd_parameters.parquet"
        ).select("link_id", "lanes")
        joined = (
            pl.read_parquet(prediction_path)
            .join(truth, on=KEYS["state"], suffix="_true")
            .join(lanes, on="link_id")
            .with_columns(epoch_step())
        )
        steps = joined["step"].to_numpy()
        inside = np.zeros(joined.height, dtype=bool)
        for start in blackout_starts(panel, steps, count):
            inside |= (steps >= start) & (steps < start + BLACKOUT_STEPS)
        for subset, mask in [("blackout", inside), ("other", ~inside)]:
            rows.append(
                {"panel": panel, "subset": subset, **subset_scores(joined.filter(mask))}
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Print per-panel and mean scores inside and outside simulated blackouts."""
    parser = argparse.ArgumentParser(
        description="Score st_cnn predictions inside simulated blackouts."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("benchmark_directory", type=Path)
    parser.add_argument("count", type=int)
    arguments = parser.parse_args()
    table = blackout_table(
        arguments.run_directory, arguments.benchmark_directory, arguments.count
    )
    with pl.Config(tbl_rows=-1):
        print(table)
        print(
            table.group_by("subset").agg(
                pl.col("n").sum(), pl.exclude("panel", "n").mean()
            )
        )


if __name__ == "__main__":
    main()
