"""Optimize queue specialists with Optuna TPE and panel-level pruning.

Run from the repository root. Study storage retains completed and pruned trials.
The objective is condition/panel/family-balanced window IoU from trafficbench.
Later-month evaluation uses the frozen selected parameter file separately.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import optuna
import polars as pl
from optuna.trial import TrialState

from scripts.tune_queue import evaluate
from solutions.queue_boosting_event_aware.model import QueueParameters
from trafficbench.contracts import Panel
from trafficbench.data import file_hash
from trafficbench.metrics import aggregate
from trafficbench.runner import code_hash, verify_benchmark


@dataclass(frozen=True)
class QueueObjective:
    """Score specialist proposals on identical panels and prepared windows."""

    benchmark: Path
    output: Path
    panels: tuple[Panel, ...]
    threads: int

    def __call__(self, trial: optuna.Trial) -> float:
        """Suggest settings and report cumulative panel IoU for pruning.

        Horizon tree parameters retain their reference values. The search varies
        onset capacity, missing-history augmentation, CatBoost blend weight, and
        restriction of horizon training to ongoing event windows.
        """
        parameters = QueueParameters(
            threads=self.threads,
            onset_num_leaves=trial.suggest_categorical(
                "onset_num_leaves", [7, 15, 31, 63]
            ),
            onset_min_data_in_leaf=trial.suggest_categorical(
                "onset_min_data_in_leaf", [10, 30, 100]
            ),
            onset_positive_weight=trial.suggest_categorical(
                "onset_positive_weight", [1.0, 2.0, 4.0]
            ),
            onset_iterations=trial.suggest_categorical("onset_iterations", [250, 750]),
            onset_history_dropout=trial.suggest_categorical(
                "onset_history_dropout", [0.0, 0.05]
            ),
            onset_catboost_weight=trial.suggest_categorical(
                "onset_catboost_weight", [0.0, 0.5, 1.0]
            ),
            ongoing_event_only=trial.suggest_categorical(
                "ongoing_event_only", [False, True]
            ),
        )
        trial.set_user_attr("solution_parameters", parameters.model_dump(mode="json"))
        metrics: list[pl.DataFrame] = []
        run_paths: list[str] = []
        for step, panel in enumerate(self.panels):
            result = evaluate(parameters, self.benchmark, self.output / "runs", panel)
            run_paths.append(str(result.run_directory))
            trial.set_user_attr("run_paths", run_paths)
            metrics.append(pl.read_csv(result.run_directory / "metrics.csv"))
            score = aggregate(pl.concat(metrics))["queue_proxy_iou"]
            trial.report(score, step)
            # Each pruning decision uses the same panel prefix across trials.
            if step < len(self.panels) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        return aggregate(pl.concat(metrics))["queue_proxy_iou"]


def main() -> None:
    """Create or resume a seeded TPE study and save its frozen best parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if min(args.trials, args.threads, args.jobs) < 1:
        parser.error("Trial, thread, and job counts must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manifest = verify_benchmark(args.benchmark)
    if not manifest.scheme.queue.onset_at_final_horizon:
        parser.error("Use selector-matched onset windows")
    queue_panels: set[Panel] = {
        case.panel for case in manifest.cases if case.task == "queue"
    }
    panels = tuple(sorted(queue_panels))
    if not panels:
        parser.error("The benchmark has no queue cases")
    args.output.mkdir(parents=True, exist_ok=True)
    # Reusing a study after changing its model or data would invalidate ranking.
    identity = {
        "benchmark_id": manifest.benchmark_id,
        "code_hash": code_hash(),
        "optimizer_hash": file_hash(Path(__file__)),
        "seed": args.seed,
        "threads": args.threads,
        "jobs": args.jobs,
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        parser.error("Study inputs changed; choose a new output directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    study = optuna.create_study(
        study_name="cal31-queue-specialists",
        storage=optuna.storages.RDBStorage(
            url=f"sqlite:///{(args.output / 'study.sqlite3').resolve()}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        ),
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3),
    )
    if not study.trials:
        # Include the reference and controlled published-method ablations.
        reference = {
            "onset_num_leaves": 31,
            "onset_min_data_in_leaf": 100,
            "onset_positive_weight": 2.0,
            "onset_iterations": 250,
            "onset_history_dropout": 0.0,
            "onset_catboost_weight": 0.0,
            "ongoing_event_only": False,
        }
        for changes in (
            {},
            {"onset_history_dropout": 0.05},
            {"onset_history_dropout": 0.05, "onset_catboost_weight": 0.5},
            {"ongoing_event_only": True},
        ):
            study.enqueue_trial(reference | changes)
    finished = sum(trial.state.is_finished() for trial in study.trials)
    study.optimize(
        QueueObjective(args.benchmark, args.output, panels, args.threads),
        n_trials=max(0, args.trials - finished),
        n_jobs=args.jobs,
    )
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError("No trial completed all panels")
    (args.output / "selected_params.json").write_text(
        json.dumps(study.best_trial.user_attrs["solution_parameters"], indent=2) + "\n"
    )
    study.trials_dataframe().to_csv(args.output / "trials.csv", index=False)
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "trial": study.best_trial.number,
                "tuning_iou": study.best_value,
                "benchmark_id": manifest.benchmark_id,
                "parameters": study.best_trial.user_attrs["solution_parameters"],
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output / "selection.json")


if __name__ == "__main__":
    main()
