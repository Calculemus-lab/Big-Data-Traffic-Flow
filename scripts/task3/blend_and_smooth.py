"""Cycle-3 experiment: blend two Task 1 runs and smooth accumulation, scored on Tasks 1 and 3.

Variants built from saved predictions of a prepared train case, with no retraining:

- ``blend_w``: ``(1 - w) * st_cnn + w * spatial_low_rank`` for speed and flow.
- ``smooth_a``: at target cells, move the accumulation ``N = flow / max(speed, 1) * length``
  a fraction ``a`` toward the mean of its previous and next 5-minute values on the same
  link (visible cells keep their published values), then set flow from the new ``N``
  with speed unchanged.

Task 1 uses the competition formula; Task 3 uses ``local_score.py``. Both are
averaged over regimes, then panels within a family, then families.

Usage::

    uv run --locked python scripts/task3/blend_and_smooth.py BENCHMARK ST_CNN_RUN LOW_RANK_RUN \\
        --output results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from local_score import KEYS, panel_scores

BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.5, 1.0)
SMOOTH_STRENGTHS = (0.25, 0.5, 1.0)
VALUES = ["speed_kmh", "flow_vph"]


def task1_scores(
    prediction: pl.DataFrame, truth: pl.DataFrame, lanes: pl.DataFrame
) -> list[dict[str, object]]:
    """Return the Task 1 score per regime, with flow error per lane."""
    joined = prediction.join(truth, on=[*KEYS, "mask_regime"], suffix="_true").join(
        lanes, on="link_id"
    )
    rows: list[dict[str, object]] = []
    for regime in ["R1", "R2", "R3"]:
        part = joined.filter(pl.col("mask_regime") == regime)
        speed = float(
            np.sqrt(
                np.mean(
                    (part["speed_kmh"].to_numpy() - part["speed_kmh_true"].to_numpy())
                    ** 2
                )
            )
        )
        flow_error = (
            part["flow_vph"].to_numpy() - part["flow_vph_true"].to_numpy()
        ) / part["lanes"].to_numpy()
        flow = float(np.sqrt(np.mean(flow_error**2)))
        score = 0.54 * max(0.0, 1 - speed / 25) + 0.46 * max(0.0, 1 - flow / 600)
        rows.append({"regime": regime, "task1": score})
    return rows


def blend(first: pl.DataFrame, second: pl.DataFrame, weight: float) -> pl.DataFrame:
    """Return ``(1 - weight) * first + weight * second`` on the shared target keys."""
    joined = first.join(second.select(*KEYS, *VALUES), on=KEYS, suffix="_other")
    return joined.with_columns(
        ((1 - weight) * pl.col(value) + weight * pl.col(f"{value}_other")).alias(value)
        for value in VALUES
    ).select(first.columns)


def smooth(
    prediction: pl.DataFrame,
    observed: pl.DataFrame,
    lengths: pl.DataFrame,
    strength: float,
) -> pl.DataFrame:
    """Move target-cell accumulation toward its temporal neighbours on the same link."""
    state = (
        observed.join(
            prediction.select(*KEYS, *VALUES), on=KEYS, how="left", suffix="_prediction"
        )
        .with_columns(
            pl.col("speed_kmh_prediction").is_not_null().alias("is_target"),
            pl.coalesce("speed_kmh_prediction", "speed_kmh").alias("speed_kmh"),
            pl.coalesce("flow_vph_prediction", "flow_vph").alias("flow_vph"),
        )
        .join(lengths, on="link_id")
        .sort("link_id", "timestamp")
        .with_columns(
            (
                pl.col("flow_vph")
                / pl.col("speed_kmh").clip(lower_bound=1.0)
                * pl.col("length_km")
            ).alias("N")
        )
    )
    step = pl.duration(minutes=5)
    previous_ok = (
        pl.col("timestamp") - pl.col("timestamp").shift(1).over("link_id")
    ) == step
    next_ok = (
        pl.col("timestamp").shift(-1).over("link_id") - pl.col("timestamp")
    ) == step
    neighbours = pl.mean_horizontal(
        pl.when(previous_ok).then(pl.col("N").shift(1).over("link_id")),
        pl.when(next_ok).then(pl.col("N").shift(-1).over("link_id")),
    )
    smoothed = (
        state.with_columns(neighbours.alias("N_neighbours"))
        .filter(pl.col("is_target"))
        .with_columns(
            pl.when(pl.col("N_neighbours").is_not_null())
            .then((1 - strength) * pl.col("N") + strength * pl.col("N_neighbours"))
            .otherwise(pl.col("N"))
            .alias("N")
        )
        .with_columns(
            (
                pl.col("N")
                / pl.col("length_km")
                * pl.col("speed_kmh").clip(lower_bound=1.0)
            ).alias("flow_vph")
        )
        .select(*KEYS, "flow_vph")
    )
    return (
        prediction.drop("flow_vph")
        .join(smoothed, on=KEYS, how="left")
        .select(prediction.columns)
    )


def panel_variants(
    case: Path, network_dir: Path, st_cnn: pl.DataFrame, low_rank: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Return every variant's predictions for one panel case."""
    truth = pl.read_parquet(case / "state" / "truth.parquet")
    observed = (
        pl.scan_parquet(case / "masked_mainline_states.parquet")
        .filter(
            (pl.col("is_score_eligible") == 1)
            & pl.col("timestamp").is_between(
                truth["timestamp"].min(), truth["timestamp"].max()
            )
        )
        .select(*KEYS, *VALUES)
        .collect()
    )
    lengths = pl.read_parquet(network_dir / "fd_parameters.parquet").select(
        "link_id", "length_km"
    )
    variants = {"st_cnn": st_cnn}
    variants |= {
        f"blend_{weight}": blend(st_cnn, low_rank, weight) for weight in BLEND_WEIGHTS
    }
    variants |= {
        f"smooth_{strength}": smooth(st_cnn, observed, lengths, strength)
        for strength in SMOOTH_STRENGTHS
    }
    return variants


def main() -> None:
    """Score every variant on every panel and print the overall Task 1 and Task 3 scores."""
    parser = argparse.ArgumentParser(
        description="Blend and smoothing variants, scored on Tasks 1 and 3."
    )
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("st_cnn_run", type=Path)
    parser.add_argument("low_rank_run", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rows: list[dict[str, object]] = []
    for truth_path in sorted(arguments.benchmark.glob("*/*/state/truth.parquet")):
        case = truth_path.parent.parent
        panel, network_dir = (
            case.parent.name,
            arguments.benchmark / "network" / case.parent.name,
        )
        truth = pl.read_parquet(truth_path)
        lanes = pl.read_parquet(network_dir / "fd_parameters.parquet").select(
            "link_id", "lanes"
        )
        loaded = [
            pl.read_parquet(run / "predictions" / panel / case.name / "state.parquet")
            for run in (arguments.st_cnn_run, arguments.low_rank_run)
        ]
        for name, prediction in panel_variants(case, network_dir, *loaded).items():
            physics = {
                row["regime"]: row
                for row in panel_scores(case, network_dir, prediction)
            }
            for row in task1_scores(prediction, truth, lanes):
                rows.append(
                    {
                        "variant": name,
                        "panel": panel,
                        "family": panel.rsplit("_", 1)[0],
                        **row,
                        **physics[row["regime"]],
                    }
                )
        print(f"{panel} done", flush=True)
    scores = pl.DataFrame(rows)
    metrics = ["task1", "S_FD", "S_LWR", "S_physics"]
    summary = (
        scores.group_by("variant", "family", "panel")
        .agg(pl.col(metrics).mean())
        .group_by("variant", "family")
        .agg(pl.col(metrics).mean())
        .group_by("variant")
        .agg(pl.col(metrics).mean())
        .with_columns(
            (0.35 * pl.col("task1") + 0.15 * pl.col("S_physics")).alias(
                "task1_3_points"
            )
        )
        .sort("task1_3_points", descending=True)
    )
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(summary)
    if arguments.output:
        summary.write_csv(arguments.output)


if __name__ == "__main__":
    main()
