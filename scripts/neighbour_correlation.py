"""Quick check: do neighbouring links carry information beyond a link's own time series?

Uses the unmasked train measurements of each corridor. For every link, it builds
the deviation from a weekly profile (residual) and compares two simple linear
predictors of a link's residual at time t:

    time only : own residual at t-g and t+g                       (g = gap in steps)
    time+space: the same, plus upstream and downstream residual at t

The improvement in RMSE (and R^2) from adding the neighbours is the value that a
spatial model can at most add on top of the time-only Kalman smoother, roughly.
It is reported for gaps of 1 and 3 steps (5 and 15 minutes), for all times and
for congested times only (speed below 60% of free-flow speed).

Usage:
    python scripts/neighbour_correlation.py --data ~/Downloads/kaggle_public
    python scripts/neighbour_correlation.py --data ~/Downloads/kaggle_public \
        --panel D7_I405_N --start 2030-10-01 --end 2030-12-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SLOTS_PER_DAY = 288


def load_panel(data: Path, panel: str, start: str, end: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Unmasked link-level speed/flow per 5-minute step, plus the ordered link table."""
    corridor = data / "corridors" / panel
    topology = pl.read_csv(
        corridor / "network" / "lwr_mainline_topology.csv",
        schema_overrides={"link_id": pl.Utf8},
    ).select("link_id", "order_index", "lanes", "free_speed_kmh")

    raw = pl.scan_parquet(str(corridor / "train" / "mainline_states" / "**" / "*.parquet"))
    if raw.collect_schema()["timestamp"] == pl.Utf8:
        raw = raw.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    states = (
        raw.with_columns(pl.col("link_id").cast(pl.Utf8))
        .filter(
            (pl.col("timestamp") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
            & (pl.col("timestamp") < pl.lit(end).str.to_datetime(time_zone="UTC"))
            & (pl.col("pct_observed") >= 50)
            & pl.col("speed_kmh").is_not_null()
            & pl.col("flow_vph").is_not_null()
        )
        # Several stations can sit on one link; neighbours are defined per link.
        .group_by("timestamp", "link_id")
        .agg(pl.col("speed_kmh").mean(), pl.col("flow_vph").mean())
        .collect()
    )
    return states, topology


def residual_matrix(
    states: pl.DataFrame, topology: pl.DataFrame
) -> tuple[dict[str, FloatArray], FloatArray, list[str]]:
    """Time x link matrices of residuals (speed: km/h, flow: log-ratio) and a congestion mask."""
    links = (
        topology.filter(pl.col("link_id").is_in(states["link_id"].unique().implode()))
        .sort("order_index")["link_id"]
        .to_list()
    )
    frame = states.with_columns(
        (pl.col("timestamp").dt.weekday() * SLOTS_PER_DAY
         + pl.col("timestamp").dt.hour().cast(pl.Int32) * 12
         + pl.col("timestamp").dt.minute().cast(pl.Int32) // 5).alias("week_slot")
    )
    profile = frame.group_by("link_id", "week_slot").agg(
        pl.col("speed_kmh").mean().alias("p_speed"), pl.col("flow_vph").mean().alias("p_flow")
    )
    frame = frame.join(profile, on=["link_id", "week_slot"]).join(
        topology.select("link_id", "free_speed_kmh"), on="link_id", how="left"
    ).with_columns(
        (pl.col("speed_kmh") - pl.col("p_speed")).alias("r_speed"),
        ((pl.col("flow_vph") + 100) / (pl.col("p_flow") + 100)).log().alias("r_flow"),
        (pl.col("speed_kmh") < 0.6 * pl.col("free_speed_kmh")).cast(pl.Float64).alias("congested"),
    )

    grid = pl.DataFrame(
        {"timestamp": pl.datetime_range(
            frame["timestamp"].min(), frame["timestamp"].max(), "5m", eager=True, time_zone="UTC"
        )}
    )
    out: dict[str, FloatArray] = {}
    for column in ("r_speed", "r_flow", "congested"):
        wide = frame.pivot(on="link_id", index="timestamp", values=column)
        wide = grid.join(wide, on="timestamp", how="left").select(links)
        out[column] = wide.to_numpy().astype(np.float64)
    congested = np.nan_to_num(out.pop("congested"), nan=0.0)
    return out, congested, links


def compare(r: FloatArray, congested: FloatArray, gap: int, only_congested: bool) -> dict[str, float]:
    """RMSE and R^2 of time-only vs time+space linear predictors, pooled over links."""
    n_steps, n_links = r.shape
    t = np.arange(gap, n_steps - gap)
    rows_time, rows_space, targets = [], [], []
    for j in range(1, n_links - 1):  # interior links have both neighbours
        y = r[t, j]
        before, after = r[t - gap, j], r[t + gap, j]
        up, down = r[t, j - 1], r[t, j + 1]
        ok = np.isfinite(y) & np.isfinite(before) & np.isfinite(after) & np.isfinite(up) & np.isfinite(down)
        if only_congested:
            ok &= congested[t, j] > 0
        rows_time.append(np.column_stack([before[ok], after[ok]]))
        rows_space.append(np.column_stack([before[ok], after[ok], up[ok], down[ok]]))
        targets.append(y[ok])
    if not targets or sum(len(y) for y in targets) < 200:
        return {"n": float(sum(len(y) for y in targets))}
    y = np.concatenate(targets)
    result = {"n": float(len(y))}
    for name, rows in (("time", rows_time), ("space", rows_space)):
        x = np.column_stack([np.ones(len(y)), np.concatenate(rows)])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        result[f"rmse_{name}"] = float(np.sqrt(np.mean(resid**2)))
        result[f"r2_{name}"] = float(1 - resid.var() / y.var())
    result["rmse_gain_pct"] = 100 * (1 - result["rmse_space"] / result["rmse_time"])
    return result


def neighbour_corr(r: FloatArray) -> float:
    """Average same-time correlation between each interior link and its neighbours."""
    values = []
    for j in range(r.shape[1] - 1):
        a, b = r[:, j], r[:, j + 1]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 200:
            values.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.nanmean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Extracted kaggle_public folder")
    parser.add_argument("--panel", action="append", help="Panel(s); default: all")
    parser.add_argument("--start", default="2030-10-01", help="First date (train split)")
    parser.add_argument("--end", default="2030-12-01", help="Exclusive end date")
    args = parser.parse_args()

    data = args.data.expanduser()
    panels = args.panel or sorted(p.name for p in (data / "corridors").iterdir() if p.is_dir())
    rows = []
    for panel in panels:
        states, topology = load_panel(data, panel, args.start, args.end)
        residuals, congested, links = residual_matrix(states, topology)
        for value, r in (("speed", residuals["r_speed"]), ("flow", residuals["r_flow"])):
            row: dict[str, object] = {
                "panel": panel, "value": value, "links": len(links),
                "neighbour_corr": round(neighbour_corr(r), 3),
            }
            for gap in (1, 3):
                for only_congested, label in ((False, "all"), (True, "cong")):
                    res = compare(r, congested, gap, only_congested)
                    if "rmse_time" in res:
                        row[f"gain%_gap{gap}_{label}"] = round(res["rmse_gain_pct"], 1)
            rows.append(row)
        print(f"done {panel}", flush=True)

    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(table)
        print("\nMean over panels:")
        print(table.group_by("value").agg(pl.exclude("panel", "value", "links").mean().round(3)))


if __name__ == "__main__":
    main()