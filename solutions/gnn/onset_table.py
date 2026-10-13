"""Holdout table with a hardcoded onset rule, from the saved probabilities.

    uv run python -m solutions.gnn.onset_table --input temp/holdout_official

--onset-rule all   : onset windows predict nothing at steps 1-5 and every link queued at step 6 (default)
--onset-rule model : onset windows predict nothing at steps 1-5, the model's call at step 6
--onset-rule none  : model everywhere, the same numbers as holdout.py (a sanity check)
Ongoing windows always use the model at the threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

THRESHOLDS = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.1)]


def window_iou(predicted, truth, eligible):
    """Per-window IoU over eligible cells; arrays are (windows, links, steps)."""
    predicted = predicted & eligible
    intersection = (predicted & truth).sum(axis=(1, 2))
    union = (predicted | truth).sum(axis=(1, 2))
    return np.where(union == 0, 1.0, intersection / np.maximum(union, 1))


def onset_prediction(probability, threshold, rule):
    model = probability >= threshold
    if rule == "none":
        return model
    predicted = np.zeros_like(model)
    predicted[..., -1] = True if rule == "all" else model[..., -1]
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("temp/holdout_official"))
    parser.add_argument("--onset-rule", choices=("all", "model", "none"), default="all")
    arguments = parser.parse_args()

    panels = []
    for path in sorted(arguments.input.glob("*_cells.npz")):
        panels.append(np.load(path))
    print(f"onset rule: {arguments.onset_rule}")
    print(f"{'setting':<18}{'onset':>8}{'ongoing':>9}{'score':>8}")
    for threshold in THRESHOLDS:
        onset_scores, ongoing_scores = [], []
        for data in panels:
            ongoing = data["ongoing"]
            probability, truth, eligible = (
                data["probability"],
                data["truth"],
                data["eligible"],
            )
            onset = ~ongoing
            onset_scores.append(
                window_iou(
                    onset_prediction(
                        probability[onset], threshold, arguments.onset_rule
                    ),
                    truth[onset],
                    eligible[onset],
                ).mean()
            )
            ongoing_scores.append(
                window_iou(
                    probability[ongoing] >= threshold, truth[ongoing], eligible[ongoing]
                ).mean()
            )
        onset_mean, ongoing_mean = np.mean(onset_scores), np.mean(ongoing_scores)
        print(
            f"{'iou_t' + format(threshold, '.2f'):<18}{onset_mean:>8.3f}"
            f"{ongoing_mean:>9.3f}{(onset_mean + ongoing_mean) / 2:>8.3f}"
        )


if __name__ == "__main__":
    main()
