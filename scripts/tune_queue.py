"""Search event-aware queue parameters on one month and assess a later holdout.

Run as ``python -m scripts.tune_queue`` from the repository root. Every candidate
uses the same prepared cases and seed. The search includes the current defaults,
selects by the runner's condition/panel/family-balanced proxy IoU, and freezes the
winner before evaluating the holdout. Official validation data is never used.
"""

from __future__ import annotations

import argparse
import logging
import random
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from solutions.queue_boosting_event_aware.model import QueueParameters
from trafficbench.contracts import BenchmarkManifest, Panel, QueueCase, RunMetadata
from trafficbench.runner import run_experiment, verify_benchmark


class TrialResult(BaseModel):
    """Parameters, aggregate proxy IoU, and complete runner evidence for one fit."""

    model_config = ConfigDict(frozen=True)
    parameters: QueueParameters
    score: float
    seconds: float
    run_directory: Path


class SearchResult(BaseModel):
    """Search identity, ordered trials, frozen selection, and later-month results."""

    tuning_benchmark_id: str
    holdout_benchmark_id: str
    search_seed: int
    trials: list[TrialResult]
    selected_parameters: QueueParameters | None = None
    holdout_baseline: TrialResult | None = None
    holdout_candidate: TrialResult | None = None


def candidate_parameters(trials: int, seed: int, threads: int) -> list[QueueParameters]:
    """Return defaults followed by a seeded sample without replacement.

    Args:
        trials: Total fit count, including the unchanged default reference.
        seed: Random seed controlling parameter combinations, not benchmark seeds.
        threads: CPU threads available to each LightGBM fit.

    Returns:
        Validated settings varying tree complexity, class weight, and regularization.
    """
    defaults = QueueParameters(threads=threads)
    # Sample combinations rather than independent draws to avoid duplicate trials.
    combinations: list[tuple[int, int, float, float, float]] = list(
        product(
            (15, 31, 63), (30, 100, 300), (1.0, 2.0, 4.0), (0.75, 0.9), (0.0, 1.0, 5.0)
        )
    )
    combinations.remove(
        (
            defaults.num_leaves,
            defaults.min_data_in_leaf,
            defaults.positive_weight,
            defaults.feature_fraction,
            defaults.lambda_l2,
        )
    )
    if not 1 <= trials <= len(combinations) + 1:
        raise ValueError(f"Use 1 to {len(combinations) + 1} trials")
    sampled = random.Random(seed).sample(combinations, trials - 1)
    return [defaults] + [
        QueueParameters(
            threads=threads,
            num_leaves=leaves,
            min_data_in_leaf=minimum_rows,
            positive_weight=positive_weight,
            feature_fraction=feature_fraction,
            lambda_l2=regularization,
        )
        for leaves, minimum_rows, positive_weight, feature_fraction, regularization in sampled
    ]


def validate_folds(tuning: BenchmarkManifest, holdout: BenchmarkManifest) -> None:
    """Reject nonchronological folds or unmatched queue sampling and panel sets.

    Both folds must contain onset and ongoing cases for every selected panel.
    They must use selector-matched onset sampling. The later fold may use earlier
    measurements as training history, but its targets must follow the tuning month.
    """
    tuning_cases = [case for case in tuning.cases if isinstance(case, QueueCase)]
    holdout_cases = [case for case in holdout.cases if isinstance(case, QueueCase)]
    if not tuning_cases or not holdout_cases:
        raise ValueError("Both folds must contain queue cases")
    if (
        tuning.scheme.queue != holdout.scheme.queue
        or not tuning.scheme.queue.onset_at_final_horizon
    ):
        raise ValueError("Both folds must use the same selector-matched queue settings")
    if {case.panel for case in tuning_cases} != {case.panel for case in holdout_cases}:
        raise ValueError("Tuning and holdout panels must match")
    if max(case.prediction_end_date for case in tuning_cases) > min(
        case.prediction_start_date for case in holdout_cases
    ):
        raise ValueError("Holdout targets must follow all tuning targets")
    for case in [*tuning_cases, *holdout_cases]:
        if {window.condition for window in case.windows} != {
            "queue_onset",
            "queue_ongoing",
        }:
            raise ValueError(f"Both queue conditions are required for {case.panel}")


def evaluate(
    parameters: QueueParameters, benchmark: Path, runs: Path, panel: Panel | None = None
) -> TrialResult:
    """Fit and score through trafficbench, retaining its normal evidence files.

    Args:
        parameters: Validated LightGBM and calibration settings.
        benchmark: Prepared cases whose truth remains inside the runner.
        runs: Directory receiving full source snapshots, predictions, and metrics.
        panel: Optional panel restriction for staged optimization and pruning.

    Returns:
        Balanced queue proxy IoU and elapsed time linked to the completed run.
    """
    run_directory = run_experiment(
        "queue_boosting_event_aware",
        task="queue",
        benchmark_directory=benchmark,
        solution_parameters=parameters.model_dump(mode="json"),
        runs_directory=runs,
        compare_baseline=False,
        panel=panel,
    )
    record = RunMetadata.model_validate_json((run_directory / "run.json").read_text())
    if record.metrics is None or record.seconds is None:
        raise RuntimeError("Completed queue run has no score or elapsed time")
    return TrialResult(
        parameters=parameters,
        score=record.metrics["queue_proxy_iou"],
        seconds=record.seconds,
        run_directory=run_directory,
    )


def main() -> None:
    """Run a bounded search and save its selection before opening holdout results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Use at least one CPU thread")
    try:
        candidates = candidate_parameters(args.trials, args.seed, args.threads)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tuning = verify_benchmark(args.tuning)
    holdout = verify_benchmark(args.holdout)
    validate_folds(tuning, holdout)
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "search.json"
    result = SearchResult(
        tuning_benchmark_id=tuning.benchmark_id,
        holdout_benchmark_id=holdout.benchmark_id,
        search_seed=args.seed,
        trials=[],
    )
    # Persist each completed trial so an interrupted search keeps its evidence.
    for parameters in candidates:
        result.trials.append(evaluate(parameters, args.tuning, args.output / "runs"))
        result_path.write_text(result.model_dump_json(indent=2) + "\n")
    selected = max(result.trials, key=lambda trial: trial.score)
    result.selected_parameters = selected.parameters
    result_path.write_text(result.model_dump_json(indent=2) + "\n")
    (args.output / "selected_params.json").write_text(
        selected.parameters.model_dump_json(indent=2) + "\n"
    )
    # Holdout scores assess the frozen winner; they never select a different trial.
    result.holdout_baseline = evaluate(
        result.trials[0].parameters, args.holdout, args.output / "runs"
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n")
    result.holdout_candidate = (
        result.holdout_baseline
        if selected is result.trials[0]
        else evaluate(selected.parameters, args.holdout, args.output / "runs")
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
