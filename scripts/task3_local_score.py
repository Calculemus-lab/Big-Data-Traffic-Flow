"""Local Task 3 (physical consistency) score for Task 1 predictions on a train case.

The official evaluator (``official_competition_repo/src/task3/score_task3.py``)
compares each link's accumulation change ``N(t+dt) - N(t)``, with
``N = flow / max(speed, 1) * length``, against organizer-held boundary and ramp
flows. Those flows are projected onto the released state, so the true state
conserves vehicles exactly. On a prepared train case the true state is known,
which makes the conservation term computable here:

    S_LWR = max(0, 1 - sum|dN_pred - dN_true| / sum|dN_true|)

over the official transitions (consecutive eligible steps, observable links,
valid attached ramps). ``S_FD`` follows the official per-lane formula, and
``S_physics = (S_FD + 2 S_LWR) / 3`` per panel and regime, averaged over
regimes, then panels within a family, then families.

Usage (a run directory from ``bench experiment``, or ``truth``)::

    uv run --locked python scripts/task3_local_score.py BENCHMARK_DIR RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ["timestamp", "station_id", "link_id"]
EMPTY_FLOOR_VPH = 50.0
EMPTY_SHARE = 0.20
MAX_STEP_HOURS = 5.1 / 60


def link_state(
    observed: pl.DataFrame, answers: pl.DataFrame, network: pl.DataFrame
) -> pl.DataFrame:
    """Return per link and time the eligible state with target cells replaced.

    Stations on one link are averaged, as the official evaluator does.
    """
    return (
        observed.join(answers, on=KEYS, how="left", suffix="_answer")
        .with_columns(
            pl.coalesce("speed_kmh_answer", "speed_kmh").alias("speed_kmh"),
            pl.coalesce("flow_vph_answer", "flow_vph").alias("flow_vph"),
            pl.col("speed_kmh_answer")
            .is_not_null()
            .cast(pl.Float64)
            .alias("is_target"),
        )
        .group_by("timestamp", "link_id", "mask_regime")
        .agg(pl.col("speed_kmh", "flow_vph", "is_target").mean())
        .join(network, on="link_id", how="inner")
        .drop_nulls(["speed_kmh", "flow_vph"])
        .with_columns(
            (
                pl.col("flow_vph")
                / pl.col("speed_kmh").clip(lower_bound=1.0)
                * pl.col("length_km")
            ).alias("accumulation")
        )
        .sort("link_id", "timestamp")
    )


def fd_score(state: pl.DataFrame) -> float:
    """Return the official per-lane fundamental-diagram score of one panel-regime."""
    lanes = state["lanes"].to_numpy().clip(min=1.0)
    flow = state["flow_vph"].to_numpy() / lanes
    density = (
        state["flow_vph"].to_numpy()
        / state["speed_kmh"].to_numpy().clip(min=1.0)
        / lanes
    )
    critical = state["critical_density"].to_numpy() / lanes
    jam = state["k_jam"].to_numpy() / lanes
    capacity = state["capacity_vph"].to_numpy() / lanes
    wave = capacity / np.maximum(jam - critical, 1e-9)
    expected = np.where(
        density <= critical,
        state["free_speed_kmh"].to_numpy() * density,
        wave * np.maximum(jam - density, 0),
    )
    score = max(0.0, 1.0 - np.abs(flow - expected).sum() / (np.abs(flow).sum() + 1e-9))
    submitted = state["is_target"].to_numpy() > 0.5
    if (
        submitted.any()
        and (state["flow_vph"].to_numpy()[submitted] < EMPTY_FLOOR_VPH).mean()
        > EMPTY_SHARE
    ):
        return 0.0
    return score


def scored_transitions(
    state: pl.DataFrame, topology: pl.DataFrame, ramps: pl.DataFrame
) -> pl.DataFrame:
    """Return forward accumulation changes on the transitions the evaluator scores.

    Links are kept only when every internal neighbour is present, and links with
    an attached ramp only at times when that ramp's released cell is valid.
    """
    present = set(state["link_id"].unique().to_list())
    topology_ids = set(topology["link_id"].to_list())
    observable = [
        row["link_id"]
        for row in topology.iter_rows(named=True)
        if all(
            neighbour in present
            for column in ("incoming_link_ids", "outgoing_link_ids")
            for neighbour in (row[column] or "").split(";")
            if neighbour in topology_ids
        )
    ]
    attached = topology.select(
        "link_id",
        (pl.col("on_ramp_link_ids").fill_null("").str.len_chars() > 0).alias("has_on"),
        (pl.col("off_ramp_link_ids").fill_null("").str.len_chars() > 0).alias(
            "has_off"
        ),
    )
    return (
        state.filter(pl.col("link_id").is_in(observable))
        .join(attached, on="link_id", how="left")
        .join(ramps, on=["timestamp", "link_id"], how="left")
        .with_columns(
            (
                pl.col("accumulation").shift(-1).over("link_id")
                - pl.col("accumulation")
            ).alias("dN"),
            (
                (
                    pl.col("timestamp").shift(-1).over("link_id") - pl.col("timestamp")
                ).dt.total_seconds()
                / 3600
            ).alias("dt"),
        )
        .filter(
            (pl.col("dt") > 0)
            & (pl.col("dt") <= MAX_STEP_HOURS)
            & (~pl.col("has_on") | pl.col("on_valid").fill_null(False))
            & (~pl.col("has_off") | pl.col("off_valid").fill_null(False))
        )
        .select("timestamp", "link_id", "dN")
    )


def panel_scores(
    case: Path, network_dir: Path, answers: pl.DataFrame
) -> list[dict[str, object]]:
    """Score one panel case against its truth, one row per regime."""
    truth = pl.read_parquet(case / "state" / "truth.parquet").select(
        *KEYS, "speed_kmh", "flow_vph"
    )
    first, last = truth["timestamp"].min(), truth["timestamp"].max()
    observed = (
        pl.scan_parquet(case / "masked_mainline_states.parquet")
        .filter(
            (pl.col("is_score_eligible") == 1)
            & pl.col("timestamp").is_between(first, last)
        )
        .select(*KEYS, "speed_kmh", "flow_vph", "mask_regime")
        .collect()
    )
    network = pl.read_parquet(network_dir / "fd_parameters.parquet").select(
        "link_id",
        "length_km",
        "lanes",
        "free_speed_kmh",
        "capacity_vph",
        "critical_density",
        "k_jam",
    )
    topology = pl.read_parquet(network_dir / "lwr_mainline_topology.parquet")
    attachment = pl.read_parquet(network_dir / "ramp_attachment_map.parquet").select(
        "ramp_link_id", pl.col("nearest_mainline_link_id").alias("link_id")
    )
    ramps = (
        pl.read_parquet(case / "ramp_states.parquet")
        .filter(
            (pl.col("is_score_eligible") == 1)
            & (pl.col("pct_observed") >= 75)
            & pl.col("flow_vph").is_not_null()
        )
        .join(attachment, on="ramp_link_id", how="inner")
        .group_by("timestamp", "link_id")
        .agg(
            pl.col("ramp_type").is_in(["OR", "ON"]).any().alias("on_valid"),
            pl.col("ramp_type").is_in(["FR", "OFF"]).any().alias("off_valid"),
        )
    )
    true_state = link_state(observed, truth, network)
    predicted_state = link_state(
        observed, answers.select(*KEYS, "speed_kmh", "flow_vph"), network
    )
    rows: list[dict[str, object]] = []
    for regime in ["R1", "R2", "R3"]:
        predicted = predicted_state.filter(pl.col("mask_regime") == regime)
        if predicted.is_empty():
            continue
        changes = scored_transitions(predicted, topology, ramps).join(
            scored_transitions(
                true_state.filter(pl.col("mask_regime") == regime), topology, ramps
            ),
            on=["timestamp", "link_id"],
            suffix="_true",
        )
        residual = np.abs(
            changes["dN"].to_numpy() - changes["dN_true"].to_numpy()
        ).sum()
        signal = np.abs(changes["dN_true"].to_numpy()).sum()
        lwr = max(0.0, 1.0 - float(residual) / (float(signal) + 1e-9))
        fd = fd_score(predicted)
        rows.append(
            {
                "regime": regime,
                "S_FD": fd,
                "S_LWR": lwr,
                "S_physics": (fd + 2 * lwr) / 3,
            }
        )
    return rows


def score_run(benchmark: Path, run: str) -> pl.DataFrame:
    """Return per-panel-regime Task 3 scores for one run directory, or for ``truth``."""
    rows: list[dict[str, object]] = []
    for truth_path in sorted(benchmark.glob("*/*/state/truth.parquet")):
        case = truth_path.parent.parent
        panel = case.parent.name
        answers = (
            pl.read_parquet(truth_path)
            if run == "truth"
            else pl.read_parquet(
                Path(run) / "predictions" / panel / case.name / "state.parquet"
            )
        )
        family = panel.rsplit("_", 1)[0]
        for row in panel_scores(case, benchmark / "network" / panel, answers):
            rows.append({"panel": panel, "family": family, **row})
    return pl.DataFrame(rows)


def overall(scores: pl.DataFrame) -> dict[str, float]:
    """Average regimes, then panels within a family, then families, as the competition does."""
    metrics = ["S_FD", "S_LWR", "S_physics"]
    return (
        scores.group_by("family", "panel")
        .agg(pl.col(metrics).mean())
        .group_by("family")
        .agg(pl.col(metrics).mean())
        .select(pl.col(metrics).mean())
        .row(0, named=True)
    )


def main() -> None:
    """Print the overall local Task 3 score of each run."""
    parser = argparse.ArgumentParser(
        description="Local Task 3 score on a prepared train case."
    )
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("runs", nargs="+", help="run directories, or 'truth'")
    parser.add_argument(
        "--output", type=Path, help="optional CSV of per-panel-regime scores"
    )
    arguments = parser.parse_args()
    tables = []
    for run in arguments.runs:
        scores = score_run(arguments.benchmark, run)
        tables.append(scores.with_columns(pl.lit(Path(run).name).alias("run")))
        summary = overall(scores)
        print(
            f"{Path(run).name}: "
            + ", ".join(f"{k} {v:.4f}" for k, v in summary.items()),
            flush=True,
        )
    if arguments.output:
        pl.concat(tables).write_csv(arguments.output)


if __name__ == "__main__":
    main()
