"""Threshold ablation on the probabilities saved by holdout.py (no retraining).

Each variant is scored out of fold: thresholds are fitted on all days except
one, then used on the held-out day, rotating through the days.

Run from the repository root after holdout.py:
    uv run python -m solutions.gnn.threshold_ablation --input temp/holdout_official
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# 1.01 means "never predict a queue", so zeroing a step is one of the options.
THRESHOLDS = np.append(np.round(np.arange(0.05, 0.951, 0.025), 3), 1.01)
FIXED_THRESHOLD = 0.4
STEPS = 6
STEPS_PER_DAY = 288
ONSET, ONGOING = 0, 1


class Windows(NamedTuple):
    """Per-window counts for every forecast step and threshold."""

    panel_id: NDArray[np.int64]
    condition: NDArray[np.int64]  # 0 = onset, 1 = ongoing
    day: NDArray[np.int64]  # day of the window's forecast origin
    intersection: NDArray[np.int64]  # (windows, steps, thresholds)
    union: NDArray[np.int64]


type Fit = Callable[[Windows, list[str], NDArray[np.bool_]], NDArray[np.int64]]


def load_windows(
    input_directory: Path, onset_last_step_only: bool = False
) -> tuple[Windows, list[str]]:
    panel_names: list[str] = []
    columns: dict[str, list[NDArray]] = {
        key: [] for key in ("panel_id", "condition", "day", "intersection", "union")
    }
    for path in sorted(input_directory.glob("*_cells.npz")):
        data = np.load(path)
        panel_id = len(panel_names)
        panel_names.append(path.name.removesuffix("_cells.npz"))
        probability, truth, eligible = (
            data["probability"],
            data["truth"],
            data["eligible"],
        )
        shape = (len(probability), STEPS, len(THRESHOLDS))
        intersection = np.empty(shape, dtype=np.int64)
        union = np.empty(shape, dtype=np.int64)
        ongoing = data["ongoing"].astype(bool)

        for index, threshold in enumerate(THRESHOLDS):
            predicted = (probability >= threshold) & eligible  # (windows, links, steps)
            if onset_last_step_only:
                predicted[~ongoing, :, : STEPS - 1] = False  # onset: step 6 only
            intersection[..., index] = (predicted & truth).sum(axis=1)
            union[..., index] = (predicted | truth).sum(axis=1)
        columns["panel_id"].append(np.full(len(probability), panel_id))
        columns["condition"].append(data["ongoing"].astype(np.int64))
        columns["day"].append(data["origins"] // STEPS_PER_DAY)
        columns["intersection"].append(intersection)
        columns["union"].append(union)
    windows = Windows(**{key: np.concatenate(value) for key, value in columns.items()})
    return windows, panel_names


def window_iou(windows: Windows, spec: NDArray[np.int64]) -> NDArray[np.float64]:
    """IoU of every window; spec[panel, condition, step] indexes THRESHOLDS."""
    steps = np.arange(STEPS)
    chosen = spec[windows.panel_id[:, None], windows.condition[:, None], steps]
    row = np.arange(len(chosen))[:, None]
    intersection = windows.intersection[row, steps, chosen].sum(axis=1)
    union = windows.union[row, steps, chosen].sum(axis=1)
    return np.where(union == 0, 1.0, intersection / np.maximum(union, 1))


def condition_scores(
    iou: NDArray[np.float64],
    windows: Windows,
    panel_names: list[str],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Mean IoU per condition: windows -> panel -> freeway family, as in the official score."""
    scores = np.full(2, np.nan)
    for condition in (ONSET, ONGOING):
        by_family: dict[str, list[float]] = {}
        for panel_id, name in enumerate(panel_names):
            selected = (
                mask & (windows.panel_id == panel_id) & (windows.condition == condition)
            )
            if selected.any():
                family = name.rsplit("_", 1)[0]
                by_family.setdefault(family, []).append(iou[selected].mean())
        if by_family:
            scores[condition] = np.mean([np.mean(v) for v in by_family.values()])
    return scores


def uniform_spec(panel_count: int, index: int) -> NDArray[np.int64]:
    return np.full((panel_count, 2, STEPS), index, dtype=np.int64)


def fit_fixed(
    windows: Windows, names: list[str], mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    return uniform_spec(
        len(names), int(np.argmin(np.abs(THRESHOLDS - FIXED_THRESHOLD)))
    )


def fit_global(
    windows: Windows, names: list[str], mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    scores = [
        np.nanmean(
            condition_scores(
                window_iou(windows, uniform_spec(len(names), t)), windows, names, mask
            )
        )
        for t in range(len(THRESHOLDS))
    ]
    return uniform_spec(len(names), int(np.argmax(scores)))


def fit_by_condition(
    windows: Windows, names: list[str], mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    # The score is a mean over the two conditions, so each is tuned on its own.
    scores = np.array(
        [
            condition_scores(
                window_iou(windows, uniform_spec(len(names), t)), windows, names, mask
            )
            for t in range(len(THRESHOLDS))
        ]
    )  # (thresholds, 2)
    spec = uniform_spec(len(names), 0)
    for condition in (ONSET, ONGOING):
        spec[:, condition, :] = int(np.nanargmax(scores[:, condition]))
    return spec


def fit_by_condition_step(
    windows: Windows, names: list[str], mask: NDArray[np.bool_], passes: int = 2
) -> NDArray[np.int64]:
    spec = fit_by_condition(windows, names, mask)
    for _ in range(passes):  # coordinate ascent: one (condition, step) at a time
        for condition in (ONSET, ONGOING):
            for step in range(STEPS):
                scores = []
                for t in range(len(THRESHOLDS)):
                    trial = spec.copy()
                    trial[:, condition, step] = t
                    iou = window_iou(windows, trial)
                    scores.append(
                        condition_scores(iou, windows, names, mask)[condition]
                    )
                spec[:, condition, step] = int(np.nanargmax(scores))
    return spec


def fit_by_panel_condition(
    windows: Windows, names: list[str], mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    spec = fit_fixed(windows, names, mask)
    all_iou = np.array(
        [
            window_iou(windows, uniform_spec(len(names), t))
            for t in range(len(THRESHOLDS))
        ]
    )  # (thresholds, windows)
    for panel_id in range(len(names)):
        for condition in (ONSET, ONGOING):
            selected = (
                mask & (windows.panel_id == panel_id) & (windows.condition == condition)
            )
            if selected.any():
                spec[panel_id, condition, :] = int(
                    np.argmax(all_iou[:, selected].mean(axis=1))
                )
    return spec


def out_of_fold_iou(
    windows: Windows, names: list[str], fit: Fit
) -> NDArray[np.float64]:
    """Score each day with thresholds fitted on the other days."""
    iou = np.empty(len(windows.condition))
    for day in np.unique(windows.day):
        test = windows.day == day
        iou[test] = window_iou(windows, fit(windows, names, ~test))[test]
    return iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("temp/holdout_official"))
    parser.add_argument("--onset-last-step-only", action="store_true")
    arguments = parser.parse_args()

    windows, names = load_windows(arguments.input, arguments.onset_last_step_only)
    everything = np.ones(len(windows.condition), dtype=bool)
    print(f"{len(windows.condition)} windows, panels: {', '.join(names)}")

    variants: dict[str, tuple[Fit, int]] = {
        "fixed 0.4": (fit_fixed, 0),
        "global": (fit_global, 1),
        "per condition": (fit_by_condition, 2),
        "per condition x step": (fit_by_condition_step, 2 * STEPS),
        "per panel x condition": (fit_by_panel_condition, 2 * len(names)),
    }
    results: dict[str, NDArray[np.float64]] = {}
    print(
        f"\n{'variant':<24}{'params':>7}{'onset':>8}{'ongoing':>9}"
        f"{'out-of-fold':>13}{'in-sample':>11}{'vs fixed':>16}"
    )
    for name, (fit, parameter_count) in variants.items():
        folded = out_of_fold_iou(windows, names, fit)
        in_sample = window_iou(windows, fit(windows, names, everything))
        results[name] = folded
        onset, ongoing = condition_scores(folded, windows, names, everything)
        difference = folded - results["fixed 0.4"]
        standard_error = difference.std() / np.sqrt(len(difference))
        in_sample_score = np.nanmean(
            condition_scores(in_sample, windows, names, everything)
        )
        print(
            f"{name:<24}{parameter_count:>7}{onset:>8.3f}{ongoing:>9.3f}"
            f"{(onset + ongoing) / 2:>13.3f}{in_sample_score:>11.3f}"
            f"{difference.mean():>+10.3f} ±{standard_error:.3f}"
        )

    print("\nThreshold sweep, all windows (one threshold for both conditions):")
    print(f"{'threshold':>10}{'onset':>8}{'ongoing':>9}")
    for t in range(len(THRESHOLDS)):
        iou = window_iou(windows, uniform_spec(len(names), t))
        onset, ongoing = condition_scores(iou, windows, names, everything)
        print(f"{THRESHOLDS[t]:>10.3f}{onset:>8.3f}{ongoing:>9.3f}")

    spec = fit_by_condition_step(windows, names, everything)
    print(
        "\nPer condition x step thresholds fitted on all windows (1.01 = never predict):"
    )
    for condition, label in ((ONSET, "onset"), (ONGOING, "ongoing")):
        row = "  ".join(
            f"{THRESHOLDS[spec[0, condition, s]]:.3f}" for s in range(STEPS)
        )
        print(f"{label:>8}: {row}")


if __name__ == "__main__":
    main()
