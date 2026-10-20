"""Internal validation on the official train windows.

The official train windows of each panel lie in June 1-7, 2030. The model is
trained on June 8 onward, so it has never seen the windows it is scored on.

Run from the repository root:
    uv run python -m solutions.gnn.holdout --panels D12_I5_N
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from trafficbench.data import Release, load_release_slice

from .example import Series, build_series, example_inputs, selected_origins
from .fit import PanelModel, fit_panel
from .grids import FORECAST_STEPS, link_grids

FIRST_DAY = date(2030, 6, 1)  # official train windows start here
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.1)]
STEP = timedelta(minutes=5)


def window_iou(
    predicted: NDArray[np.bool_], truth: NDArray[np.bool_], eligible: NDArray[np.bool_]
) -> float:
    """IoU over eligible cells; a window with nothing queued in both scores 1."""
    predicted = predicted & eligible
    union = (predicted | truth).sum()
    return 1.0 if union == 0 else float((predicted & truth).sum() / union)


def score_windows(
    fitted: PanelModel,
    series: Series,
    origins: NDArray[np.int64],
    ongoing: NDArray[np.bool_],
    start: datetime,
    panel: str,
    output: Path,
) -> list[dict]:
    """Run the model on the windows, save every cell, return one record per window."""
    inputs = torch.from_numpy(example_inputs(series, origins))
    device = next(fitted.model.parameters()).device
    with torch.no_grad():
        probability = torch.cat(
            [
                torch.sigmoid(fitted.model(chunk.to(device))).cpu()
                for chunk in inputs.split(256)
            ]
        ).numpy()  # (windows, links, 6)

    future = origins[:, None] + np.arange(1, FORECAST_STEPS + 1)
    truth = series.queued[future].transpose(0, 2, 1)
    eligible = series.known[future].transpose(0, 2, 1)
    # Persistence repeats the last observed queue status for all six steps.
    persistence = np.repeat(
        series.queued[origins - 1][:, :, None], FORECAST_STEPS, axis=2
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{panel}_cells.npz",
        origins=origins,
        ongoing=ongoing,
        probability=probability,
        truth=truth,
        eligible=eligible,
        link_ids=np.array(fitted.link_ids),
    )

    records = []
    for i, origin in enumerate(origins):
        record = {
            "panel": panel,
            "origin": start + int(origin) * STEP,
            "condition": "queue_ongoing" if ongoing[i] else "queue_onset",
            "queued_cells": int(truth[i].sum()),
            "iou_persistence": window_iou(persistence[i], truth[i], eligible[i]),
        }
        for threshold in THRESHOLDS:
            record[f"iou_t{threshold:.2f}"] = window_iou(
                probability[i] >= threshold, truth[i], eligible[i]
            )
        records.append(record)
    return records


def run_panel(
    release: Release, panel: str, parameters: dict, output: Path
) -> list[dict]:
    """Train after the panel's last official window, then score those windows."""
    index = release.read_csv(f"task2/{panel}/train/window_index.csv")
    last_day = max(index["forecast_origin"]).date()
    train_start = datetime.combine(
        last_day + timedelta(days=1), datetime.min.time(), UTC
    )
    print(f"{panel}: windows end {last_day}, training from {train_start.date()}")

    release_slice = load_release_slice(
        release,
        date(2030, 6, 1),
        date(2031, 3, 1),
        date(2031, 3, 1),
        date(2031, 3, 2),
        [panel],
    )
    fitted = fit_panel(release_slice, panel, parameters, train_start=train_start)

    # Inputs and labels come from the unmasked table, like the official windows.
    states = release.scan_state_rows(
        panel, "train", "mainline_states", FIRST_DAY, last_day + timedelta(days=2)
    )
    grids = link_grids(states, fitted.link_ids)
    series = build_series(grids, fitted.free_speed_kmh, fitted.density_scale)._replace(usual=fitted.usual)

    start = grids.start.replace(tzinfo=None)
    origins = np.array(
        [
            (origin.replace(tzinfo=None) - start) // STEP
            for origin in index["forecast_origin"]
        ],
        dtype=np.int64,
    )
    ongoing = (index["condition"] == "queue_ongoing").to_numpy()

    # Check that our onset/ongoing rule reproduces the organizers' labels.
    own_origins, own_ongoing = selected_origins(series)
    own = dict(zip(own_origins.tolist(), own_ongoing.tolist()))
    agree = sum(own.get(int(o)) == bool(f) for o, f in zip(origins, ongoing))
    print(
        f"{panel}: {agree}/{len(origins)} official windows match our condition labels"
    )

    return score_windows(fitted, series, origins, ongoing, grids.start, panel, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", default=["D12_I5_N"])
    parser.add_argument("--data", type=Path, default=Path("kaggle_public"))
    parser.add_argument("--params", type=json.loads, default={})
    parser.add_argument("--output", type=Path, default=Path("temp/holdout_official"))
    arguments = parser.parse_args()

    release = Release(arguments.data)
    records: list[dict] = []
    for panel in arguments.panels:
        records += run_panel(release, panel, arguments.params, arguments.output)

    table = pl.DataFrame(records)
    table.write_csv(arguments.output / "windows.csv")

    # Mean IoU within panel and condition, then across panels, per condition.
    iou_columns = [c for c in table.columns if c.startswith("iou_")]
    per_condition = (
        table.group_by("panel", "condition")
        .agg(pl.col(c).mean() for c in iou_columns)
        .group_by("condition")
        .agg(pl.col(c).mean() for c in iou_columns)
    )
    window_counts = table.group_by("condition").len()
    print(window_counts)
    print(f"{'setting':<18}{'onset':>8}{'ongoing':>9}{'score':>8}")
    for column in iou_columns:
        by_condition = dict(zip(per_condition["condition"], per_condition[column]))
        onset, ongoing_iou = by_condition["queue_onset"], by_condition["queue_ongoing"]
        print(
            f"{column:<18}{onset:>8.3f}{ongoing_iou:>9.3f}{(onset + ongoing_iou) / 2:>8.3f}"
        )


if __name__ == "__main__":
    main()
