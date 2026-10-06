"""Run prepared experiments or direct predictions from release slices.

Experiment runs verify prepared inputs, keep answers outside the solution
slice, validate returned tables, and record per-case metrics. Direct runs read
the requested release slice and official sample-submission rows.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import polars as pl
from pydantic import DirectoryPath, FilePath

from .contracts import (
    KEYS,
    VALUES,
    BenchmarkCase,
    BenchmarkManifest,
    HistoricalTaskLabels,
    JsonObject,
    MetricRow,
    NetworkTables,
    Panel,
    PanelReleaseSlice,
    PanelSplitTableMap,
    ReleasePackageSlice,
    RunMetadata,
    Split,
    TargetShortcut,
    Task,
    normalize_lazy,
)
from .data import (
    Release,
    file_hash,
    load_release_slice,
    load_target_templates,
    metadata_hash,
    release_calendar,
    save_table,
    validate_experiment_dates,
)
from .inference import run_solution
from .metrics import (
    aggregate,
)

logger = logging.getLogger(__name__)


def snapshot_files() -> list[Path]:
    """List source and configuration files copied into a run snapshot.

    Returns:
        Existing Python, YAML, JSON, and project-environment files that
        identify the code and settings used by a run.
    """
    return sorted(
        path
        for source_directory in ["trafficbench", "solutions", "config"]
        for path in Path(source_directory).rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".json"}
    ) + [
        path
        for path in [Path("pyproject.toml"), Path("uv.lock"), Path(".python-version")]
        if path.exists()
    ]


def code_hash() -> str:
    """Hash the paths and contents of files included in a run snapshot.

    Returns:
        Stable digest used to identify the snapshotted code and settings.
    """
    return metadata_hash({str(path): file_hash(path) for path in snapshot_files()})


def solution_hash(solution_name: str) -> str:
    """Hash a solution's Python files for baseline cache identity.

    Args:
        solution_name: Package or module name under ``solutions``.

    Returns:
        Stable digest of its source-file paths and contents.
    """
    solution_path = Path("solutions") / solution_name
    source_files = (
        sorted(solution_path.rglob("*.py"))
        if solution_path.is_dir()
        else [solution_path.with_suffix(".py")]
    )
    return metadata_hash({str(path): file_hash(path) for path in source_files})


def evaluator_hash() -> str:
    """Hash package modules that can affect run execution or scoring.

    Returns:
        Stable digest used to decide whether two runs used the same evaluator.
    """
    return metadata_hash(
        {
            str(path): file_hash(path)
            for path in sorted(Path("trafficbench").glob("*.py"))
        }
    )


def _git_output(*git_arguments: str) -> str | None:
    """Run one Git command and return its trimmed output when it succeeds.

    Args:
        git_arguments: Arguments following the ``git`` command name.

    Returns:
        Standard output, or ``None`` when Git returns a failure status.
    """
    command_result = subprocess.run(
        ["git", *git_arguments], capture_output=True, text=True, check=False
    )
    return command_result.stdout.strip() if command_result.returncode == 0 else None


def verify_benchmark(
    benchmark_directory: DirectoryPath, panel: Panel | None = None
) -> BenchmarkManifest:
    """Verify a prepared benchmark directory against its manifest.

    Verify every prepared Parquet file unless ``panel`` is supplied, in which
    case verify that panel's case files and network tables. Return the
    validated manifest. Raise ``ValueError`` if the manifest or a file changed.

    Args:
        benchmark_directory: Prepared benchmark directory containing
            ``manifest.json`` and the files listed in it.
        panel: Optional panel whose files should be verified for a hosted job.

    Returns:
        The validated manifest describing the prepared benchmark.

    """
    # The identifier covers the manifest fields. Individual file hashes cover
    # the Parquet bytes that the manifest names.
    manifest_path = benchmark_directory / "manifest.json"
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text())
    benchmark_id = manifest.benchmark_id
    if metadata_hash(manifest.hash_content()) != benchmark_id:
        raise ValueError("Benchmark manifest changed. Prepare a new version")
    for relative_path, expected_hash in manifest.files.items():
        if panel and not relative_path.startswith(
            (panel + "/", "network/" + panel + "/")
        ):
            continue
        if file_hash(benchmark_directory / relative_path) != expected_hash:
            raise ValueError(
                f"Benchmark file changed: {relative_path}. Prepare a new version"
            )
    return manifest


def report(run_metadata: RunMetadata) -> str:
    """Format saved scores, baseline differences, warnings, and failures.

    Return Markdown text. Missing metrics in a failed run leave the score
    table empty while the recorded error remains visible below it.

    Args:
        run_metadata: Saved run identity, status, metrics, and warnings.

    Returns:
        Markdown report for display or writing to ``report.md``.
    """
    report_lines = [
        f"# {run_metadata.solution}: {run_metadata.status}",
        "",
        (
            f"Task: {run_metadata.task} | Benchmark: {run_metadata.benchmark_id[:12]} "
            f"| Profile: {run_metadata.profile}"
        ),
        "",
        (
            "Queue and OD synthetic scores are proxies. Physics values are "
            "diagnostics. No local overall competition score is available."
        ),
        "",
        "| Metric | Result | Baseline | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    scores = run_metadata.metrics or {}
    baseline_metrics = run_metadata.baseline_metrics or {}
    for metric_name, score in scores.items():
        baseline_score = baseline_metrics.get(metric_name)
        if baseline_score is None:
            report_lines.append(f"| {metric_name} | {score:.6f} | n/a | n/a |")
        else:
            report_lines.append(
                f"| {metric_name} | {score:.6f} | {baseline_score:.6f} "
                f"| {score - baseline_score:+.6f} |"
            )
    elapsed_seconds = run_metadata.seconds or 0.0
    peak_memory = (
        f"{run_metadata.peak_memory_mb:.0f} MB"
        if run_metadata.peak_memory_mb is not None
        else "not measured on this platform"
    )
    report_lines.extend(
        [
            "",
            (f"Elapsed: {elapsed_seconds:.1f}s. Peak process memory: {peak_memory}."),
        ]
    )
    if error_traceback := run_metadata.error:
        report_lines.extend(["", "```", error_traceback, "```"])
    report_lines.extend(
        [
            "",
            (
                "Per-panel, regime, condition and window results: metrics.csv. "
                "Predictions: predictions/."
            ),
        ]
    )
    for warning in run_metadata.warnings:
        report_lines.extend(["", f"Note: {warning}"])
    return "\n".join(report_lines) + "\n"


def _execute_cases(
    solution_name: str,
    task: Task,
    benchmark_directory: DirectoryPath,
    manifest: BenchmarkManifest,
    selected_benchmark_cases: list[BenchmarkCase],
    solution_parameters: JsonObject,
    run_directory: Path,
) -> Iterator[list[MetricRow]]:
    """Run one generated interval through the normal solution interface.

    All selected panels share one call for each state, queue, or ODME scenario.
    Task 4 has distinct count inputs per synthetic scenario, so those scenarios
    are separate calls with the same split-and-panel target mapping. Answers
    remain in runner-owned tables and are passed only to the scoring boundary.

    Args:
        solution_name: Importable package name below ``solutions``.
        task: Prediction task selected for the prepared cases.
        benchmark_directory: Directory containing case tables and manifest.
        manifest: Validated inventory and configuration for the cases.
        selected_benchmark_cases: Cases matching the requested task and panel.
        solution_parameters: JSON settings passed to each solution call.
        run_directory: Destination for prediction files.

    Yields:
        Per-case metric rows after each grouped solution call completes.

    Raises:
        ValueError: If a prepared case is incomplete or predictions are invalid.
    """
    cases_by_group: dict[str, list[BenchmarkCase]] = {}
    # State and queue panels with the same interval can be predicted together.
    # Task 4 cases are grouped by scenario because each scenario has different
    # link counts and path-flow priors, even when panel and dates match.
    for benchmark_case in selected_benchmark_cases:
        match benchmark_case.task:
            case "odme":
                group_name = benchmark_case.scenario_name
            case "state" | "queue":
                group_name = benchmark_case.case_id
        cases_by_group.setdefault(group_name, []).append(benchmark_case)

    for group_name, grouped_cases in cases_by_group.items():
        panel_slices: dict[Panel, PanelReleaseSlice] = {}
        target_tables_by_panel: dict[Panel, pl.DataFrame] = {}
        answer_tables_by_panel: dict[Panel, pl.DataFrame] = {}
        has_answers = True

        # Rebuild each panel's solution inputs from the prepared case tables.
        # Answer values remain in separate tables and are never added to the slice.
        for benchmark_case in grouped_cases:
            panel_case_directory = (
                benchmark_directory / benchmark_case.panel / benchmark_case.case_id
            )
            case_directory = benchmark_directory / benchmark_case.case_directory
            case_tables = {
                case_table_path.stem: pl.scan_parquet(case_table_path)
                for case_table_path in case_directory.glob("*.parquet")
            }
            panel_network = NetworkTables.from_mapping(
                {
                    network_file.stem: pl.scan_parquet(network_file)
                    for network_file in (
                        benchmark_directory / "network" / benchmark_case.panel
                    ).glob("*.parquet")
                }
            )
            historical_task_labels = HistoricalTaskLabels(
                task1_state_answers=pl.scan_parquet(
                    panel_case_directory / "state" / "historical_solutions.parquet"
                ),
                task2_queue_proxy_labels=pl.scan_parquet(
                    panel_case_directory / "queue" / "historical_solutions.parquet"
                ),
            )
            match benchmark_case.task:
                case "odme":
                    counts_table = case_tables["counts"]
                    prior_table = case_tables["prior"]
                case "state" | "queue":
                    counts_table = pl.scan_parquet(
                        panel_case_directory / "link_counts.parquet"
                    )
                    prior_table = pl.scan_parquet(
                        panel_case_directory / "weak_prior.parquet"
                    )

            # Queue experiments add their generated windows to released
            # historical windows. Other tasks receive only released windows.
            released_queue_history = pl.scan_parquet(
                panel_case_directory / "released_queue_history.parquet"
            )
            released_queue_window_index = pl.scan_parquet(
                panel_case_directory / "released_queue_window_index.parquet"
            )
            match benchmark_case.task:
                case "queue":
                    queue_history_table = pl.concat(
                        [released_queue_history, case_tables["observations"]],
                        how="diagonal_relaxed",
                    )
                    queue_index_table = pl.concat(
                        [released_queue_window_index, case_tables["window_index"]],
                        how="diagonal_relaxed",
                    )
                case "state" | "odme":
                    queue_history_table = released_queue_history
                    queue_index_table = released_queue_window_index
            # All historical labels and released inputs are attached to the
            # panel, while target rows and answer values remain separate mappings.
            panel_slices[benchmark_case.panel] = PanelReleaseSlice(
                family_id=benchmark_case.family_id,
                network=panel_network,
                masked_mainline_states={
                    "train": pl.scan_parquet(
                        panel_case_directory / "masked_mainline_states.parquet"
                    )
                },
                ramp_states={
                    "train": pl.scan_parquet(
                        panel_case_directory / "ramp_states.parquet"
                    )
                },
                queue_history={"train": queue_history_table},
                queue_window_index={"train": queue_index_table},
                link_counts={"train": counts_table},
                weak_prior={"train": prior_table},
                historical_task_labels=historical_task_labels,
            )
            target_tables_by_panel[benchmark_case.panel] = case_tables[
                "targets"
            ].collect()
            if "truth" not in case_tables:
                has_answers = False
            elif task == "queue":
                answer_tables_by_panel[benchmark_case.panel] = (
                    case_tables["truth"].collect().rename({"queue_true": "queue_pred"})
                )
            else:
                answer_tables_by_panel[benchmark_case.panel] = case_tables[
                    "truth"
                ].collect()

        # The grouped cases share one prediction interval and use one seed. The
        # mappings below adapt their per-panel files to the common runner API.
        first_case = grouped_cases[0]
        experiment_slice = ReleasePackageSlice(
            history_start_date=first_case.history_start_date,
            history_end_date=first_case.history_end_date,
            prediction_start_date=first_case.prediction_start_date,
            prediction_end_date=first_case.prediction_end_date,
            panel_slices_by_panel=panel_slices,
            source_fingerprint=manifest.data_fingerprint,
            parameters=solution_parameters.copy(),
            seed=int(metadata_hash([manifest.seed, task, group_name])[:8], 16),
        )
        train_split: Split = "train"
        targets_by_panel_and_split: PanelSplitTableMap[pl.DataFrame] = {
            panel: {train_split: target_table}
            for panel, target_table in target_tables_by_panel.items()
        }
        answers_by_panel_and_split: PanelSplitTableMap[pl.DataFrame] | None = (
            {
                panel: {train_split: answer_table}
                for panel, answer_table in answer_tables_by_panel.items()
            }
            if has_answers
            else None
        )
        # run_solution performs the same leakage redaction and output checks as
        # direct prediction runs, then scores only when every case has truth.
        solution_result = run_solution(
            solution_name,
            task,
            experiment_slice,
            targets_by_panel_and_split,
            answers_by_panel_and_split,
            solution_parameters,
            experiment_slice.seed,
        )

        # Save each panel prediction under the prepared case's relative path.
        for benchmark_case in grouped_cases:
            panel_prediction_table = solution_result.predictions[benchmark_case.panel][
                "train"
            ]
            save_table(
                run_directory
                / "predictions"
                / f"{benchmark_case.case_directory}.parquet",
                panel_prediction_table,
            )

        score_rows: list[MetricRow] = []
        for metric_row in solution_result.metrics.to_dicts():
            metric_row["case_id"] = first_case.case_id
            if task == "odme":
                metric_row["scenario"] = group_name
            score_rows.append(cast(MetricRow, metric_row))
        yield score_rows


def _start_run(
    solution_name: str,
    task: Task,
    runs_directory: Path,
    manifest: BenchmarkManifest,
    selected_benchmark_cases: list[BenchmarkCase],
    solution_parameters: JsonObject,
) -> tuple[Path, RunMetadata]:
    """Create a run directory and snapshot its inputs and environment.

    Return the directory and initial run metadata. Source files, configuration,
    the prepared manifest, and package versions make the result inspectable.

    Args:
        solution_name: Importable package name below ``solutions``.
        task: Task recorded in the new run.
        runs_directory: Parent directory for the new run folder.
        manifest: Prepared benchmark identity and case list.
        selected_benchmark_cases: Cases included in this run.
        solution_parameters: JSON settings supplied to the solution.

    Returns:
        New run directory and metadata with status ``running``.
    """
    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_id = f"{run_timestamp}-{solution_name}-{task}-{uuid.uuid4().hex[:6]}"
    run_directory = runs_directory / run_id
    run_directory.mkdir(parents=True)

    # Record code, data, and dependencies before importing the solution.
    dependency_versions = {
        distribution.name: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    run_metadata = RunMetadata(
        run_id=run_id,
        solution=solution_name,
        task=task,
        params=solution_parameters,
        status="running",
        timestamp=datetime.now(UTC).isoformat(),
        benchmark_id=manifest.benchmark_id,
        profile=manifest.profile,
        seed=manifest.seed,
        panels=sorted(
            {benchmark_case.panel for benchmark_case in selected_benchmark_cases}
        ),
        commit=_git_output("rev-parse", "HEAD"),
        branch=_git_output("branch", "--show-current"),
        dirty=bool(_git_output("status", "--porcelain")),
        code_hash=code_hash(),
        evaluator_hash=evaluator_hash(),
        baseline_hash=solution_hash("baseline"),
        python=sys.version,
        platform=platform.platform(),
        dependencies=dependency_versions,
        warnings=manifest.warnings,
    )
    # Copy the exact source and manifest used by this run.
    for path in snapshot_files():
        snapshot_path = run_directory / "source" / path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot_path)
    (run_directory / "benchmark.json").write_text(
        manifest.model_dump_json(indent=2, exclude_none=True)
    )
    (run_directory / "run.json").write_text(
        run_metadata.model_dump_json(indent=2, exclude_none=True)
    )
    return run_directory, run_metadata


def _save_result(
    run_directory: Path,
    run_metadata: RunMetadata,
    score_rows: list[MetricRow],
    start_time: float,
) -> None:
    """Save case metrics and the completed or failed run report.

    ``score_rows`` can contain partial results when a later case failed. Record
    elapsed time and peak process memory alongside the status and error.

    Args:
        run_directory: Folder receiving metrics, metadata, and report files.
        run_metadata: Mutable record whose final status and resources are saved.
        score_rows: Completed per-case metrics, including partial results.
        start_time: Monotonic clock value recorded before execution.
    """
    run_metadata.seconds = time.perf_counter() - start_time
    # The resource module is unavailable on Windows. Preserve local runs there
    # and record peak memory on platforms that provide this operating-system API.
    try:
        import resource
    except ImportError:
        run_metadata.peak_memory_mb = None
    else:
        peak_resident_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        run_metadata.peak_memory_mb = peak_resident_memory / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    pl.DataFrame(score_rows).write_csv(run_directory / "metrics.csv")
    (run_directory / "run.json").write_text(
        run_metadata.model_dump_json(indent=2, exclude_none=True)
    )
    (run_directory / "report.md").write_text(report(run_metadata))
    logger.info("Results: %s", run_directory)


def _attach_baseline(
    run_metadata: RunMetadata,
    run_directory: Path,
    task: Task,
    benchmark_directory: DirectoryPath,
    runs_directory: Path,
    panel: Panel | None,
) -> None:
    """Compare a run with a matching saved or newly computed local baseline.

    A saved baseline must use the same benchmark, task, panels, evaluator,
    baseline code, and dependencies. Write its metrics into the run report.

    Args:
        run_metadata: Run that receives the baseline ID and scores.
        run_directory: Directory whose metadata and report are updated.
        task: Task used to find or compute the baseline.
        benchmark_directory: Prepared benchmark used for baseline execution.
        runs_directory: Parent directory searched for saved baseline runs.
        panel: Optional hosted-run panel restriction.
    """
    # Reuse only a completed baseline whose inputs and scorer match this run.
    baseline_run: RunMetadata | None = None
    for run_record_path in sorted(runs_directory.glob("*/run.json"), reverse=True):
        try:
            baseline_candidate = RunMetadata.model_validate_json(
                run_record_path.read_text()
            )
        except ValueError:
            continue
        same_environment = (
            baseline_candidate.task == run_metadata.task
            and baseline_candidate.benchmark_id == run_metadata.benchmark_id
            and baseline_candidate.panels == run_metadata.panels
            and baseline_candidate.evaluator_hash == run_metadata.evaluator_hash
            and baseline_candidate.baseline_hash == run_metadata.baseline_hash
            and baseline_candidate.dependencies == run_metadata.dependencies
        )
        if (
            same_environment
            and baseline_candidate.solution == "baseline"
            and baseline_candidate.status == "completed"
            and not baseline_candidate.params
        ):
            baseline_run = baseline_candidate
            break
    if baseline_run is None:
        baseline_directory = run_experiment(
            "baseline", task, benchmark_directory, {}, runs_directory, False, panel
        )
        baseline_run = RunMetadata.model_validate_json(
            (baseline_directory / "run.json").read_text()
        )
    if baseline_run.metrics is None:
        raise ValueError("Completed baseline run has no metrics")

    # Include reference values in the saved run metadata and report.
    run_metadata.baseline_run = baseline_run.run_id
    run_metadata.baseline_metrics = baseline_run.metrics
    (run_directory / "run.json").write_text(
        run_metadata.model_dump_json(indent=2, exclude_none=True)
    )
    (run_directory / "report.md").write_text(report(run_metadata))


def run_experiment(
    solution_name: str,
    task: Task = "state",
    benchmark_directory: DirectoryPath = Path("data/benchmarks/quick"),
    solution_parameters: JsonObject | None = None,
    runs_directory: Path = Path("runs"),
    compare_baseline: bool = True,
    panel: Panel | None = None,
) -> Path:
    """Execute one solution over generated training cases.

    ``benchmark_directory`` contains generated cases, and
    ``solution_parameters`` are passed to the solution. ``runs_directory``
    receives predictions, metrics, source files, and reports.
    ``panel`` limits a hosted run to one panel. By default, a non-baseline run
    also finds or computes a matching baseline. Return the new directory on
    success. If execution fails, write a failed run record before raising the
    original exception.

    Args:
        solution_name: Importable package under ``solutions``.
        task: Task whose generated cases are run and scored.
        benchmark_directory: Prepared directory created by ``bench prepare``.
        solution_parameters: JSON settings passed to each solution call.
        runs_directory: Parent directory for outputs and run reports.
        compare_baseline: Whether to compare with a compatible local baseline.
        panel: Optional panel restriction for a hosted panel job.

    Returns:
        The completed run directory.

    """
    # Verify the prepared case inputs and select only the requested task and
    # panel before creating a run directory.
    manifest = verify_benchmark(benchmark_directory, panel)
    solution_parameters = solution_parameters or {}
    selected_cases = [
        benchmark_case
        for benchmark_case in manifest.cases
        if benchmark_case.task == task
        and (panel is None or benchmark_case.panel == panel)
    ]
    if not selected_cases:
        raise ValueError(f"No {task} cases in this benchmark")
    start_time = time.perf_counter()
    run_directory, run_metadata = _start_run(
        solution_name,
        task,
        runs_directory,
        manifest,
        selected_cases,
        solution_parameters,
    )
    score_rows: list[MetricRow] = []
    run_error: Exception | None = None

    try:
        # Keep completed case rows available if a later case fails.
        for case_score_rows in _execute_cases(
            solution_name,
            task,
            benchmark_directory,
            manifest,
            selected_cases,
            solution_parameters,
            run_directory,
        ):
            score_rows.extend(case_score_rows)
        run_metadata.metrics = aggregate(pl.DataFrame(score_rows))
        run_metadata.status = "completed"
    except Exception as error:  # noqa: BLE001
        # Keep any solution or runner failure in the saved run record, then re-raise it.
        run_metadata.status = "failed"
        run_metadata.error = traceback.format_exc()
        run_error = error
    finally:
        _save_result(run_directory, run_metadata, score_rows, start_time)
    if run_error:
        raise run_error
    if compare_baseline and solution_name != "baseline":
        _attach_baseline(
            run_metadata,
            run_directory,
            task,
            benchmark_directory,
            runs_directory,
            panel,
        )
    if run_metadata.metrics is None:
        raise RuntimeError("Completed run has no metrics")
    logger.info("Aggregated metrics:\n%s", json.dumps(run_metadata.metrics, indent=2))
    return run_directory


def compare(run_directories: Sequence[DirectoryPath]) -> pl.DataFrame:
    """Display metrics from completed, directly comparable run directories.

    ``run_directories`` names the runs to compare. Return their metrics and
    elapsed times as a DataFrame. Reject failed runs or differences in
    benchmark, task, panel coverage, or scoring code.

    Args:
        run_directories: Completed run folders to compare.

    Returns:
        A table containing each run's identity and metric values.

    Raises:
        ValueError: If any run failed or the saved conditions are incompatible.
    """
    # The saved metadata, rather than directory names, determines whether
    # these runs can be compared under the same scoring conditions.
    run_records = [
        RunMetadata.model_validate_json((directory / "run.json").read_text())
        for directory in run_directories
    ]
    if any(record.status != "completed" for record in run_records):
        raise ValueError("Cannot compare a failed run")
    if len({(record.benchmark_id, record.task) for record in run_records}) != 1:
        raise ValueError(
            "Runs must use the same task and benchmark. Quick and full profiles are not interchangeable"
        )
    if len({record.evaluator_hash for record in run_records}) != 1:
        raise ValueError(
            "Scoring code changed between these runs. Rerun using the same evaluator"
        )
    if len({tuple(record.panels) for record in run_records}) != 1:
        raise ValueError(
            "Runs must cover the same panels. A partial run is not a full benchmark"
        )
    comparison_rows: list[dict[str, str | float]] = []
    for record in run_records:
        metrics = record.metrics
        elapsed_seconds = record.seconds
        if metrics is None or elapsed_seconds is None:
            raise ValueError("Completed run is missing metrics or elapsed time")
        comparison_rows.append(
            {"run": record.run_id, **metrics, "seconds": elapsed_seconds}
        )
    comparison_table = pl.DataFrame(comparison_rows)
    logger.info("Run comparison:\n%s", comparison_table)
    return comparison_table


def run_predictions(
    solution_name: str,
    task: Task,
    release_directory: Path,
    panels: Sequence[Panel],
    history_start_date: date | None = None,
    history_end_date: date | None = None,
    prediction_start_date: date | None = None,
    prediction_end_date: date | None = None,
    target_shortcut: TargetShortcut | None = None,
    solution_parameters: JsonObject | None = None,
    answer_key_file: FilePath | None = None,
    runs_directory: Path = Path("runs"),
) -> Path:
    """Predict a date range using selected train history and released tables.

    With no explicit prediction dates, ``target_shortcut`` selects validation,
    private, or both. The default selects both validation and private. Explicit
    target dates may span split boundaries. When an answer file is supplied,
    its identifier columns select the rows to predict. Its answer values stay
    outside the solution call for scoring.

    Args:
        solution_name: Importable package name below ``solutions``.
        task: Task to predict.
        release_directory: Extracted release root containing ``config/``.
        panels: Panels loaded together for the solution call.
        history_start_date: First train date visible to the solution.
        history_end_date: Exclusive end of train answers available as history.
        prediction_start_date: Inclusive first date requested as a target.
        prediction_end_date: Exclusive end of the requested target interval.
        target_shortcut: Optional complete validation, private, or combined range.
        solution_parameters: JSON settings passed to the solution.
        answer_key_file: Optional file with target identifiers and answer values.
        runs_directory: Parent directory for prediction files and run metadata.

    Returns:
        Path to the directory containing this run's predictions and results.

    Raises:
        ValueError: If target selection is invalid or selects no official rows.
    """
    # Resolve default target shortcuts against the release's published calendar.
    release_package = Release(release_directory)
    calendar = release_calendar(release_package)
    train_start_date, train_end_date = calendar["train"]
    history_start_date = history_start_date or train_start_date
    history_end_date = history_end_date or train_end_date

    explicit_prediction_dates = (
        prediction_start_date is not None or prediction_end_date is not None
    )
    if target_shortcut is not None and explicit_prediction_dates:
        raise ValueError("Choose a target shortcut or explicit prediction dates")
    if target_shortcut is None and not explicit_prediction_dates:
        target_shortcut = "both"
    if target_shortcut is not None:
        target_ranges = {
            "validation": calendar["validation"],
            "private": calendar["private"],
            "both": (calendar["validation"][0], calendar["private"][1]),
        }
        prediction_start_date, prediction_end_date = target_ranges[target_shortcut]
    elif prediction_end_date is None:
        raise ValueError("Supply prediction_end_date for an explicit target range")

    prediction_start_date = prediction_start_date or history_end_date
    validate_experiment_dates(
        calendar,
        history_start_date,
        history_end_date,
        prediction_start_date,
        prediction_end_date,
    )
    # Load visible features and target templates separately. Future answer
    # values remain runner-side and never become part of the solution slice.
    release_slice = load_release_slice(
        release_package,
        history_start_date,
        history_end_date,
        prediction_start_date,
        prediction_end_date,
        panels,
    )
    target_templates = load_target_templates(
        release_package,
        task,
        prediction_start_date,
        prediction_end_date,
        panels,
    )
    # An answer file selects target rows and supplies values for later scoring.
    # Its values are not copied into release_slice or the template mapping.
    answer_tables: PanelSplitTableMap[pl.DataFrame] | None = (
        _partition_answer_key(
            task,
            normalize_lazy(
                pl.scan_csv(
                    answer_key_file,
                    schema_overrides={
                        column: pl.String
                        for column in KEYS[task]
                        if column != "timestamp"
                    },
                    null_values="",
                    try_parse_dates=True,
                )
            ).collect(),
            target_templates,
        )
        if answer_key_file is not None
        else None
    )
    if answer_tables is None:
        targets_by_panel_and_split: PanelSplitTableMap[pl.DataFrame] = {
            panel: {
                split: target_table.select(KEYS[task])
                for split, target_table in target_tables_by_split.items()
                if target_table.height
            }
            for panel, target_tables_by_split in target_templates.items()
        }
    else:
        targets_by_panel_and_split = {
            panel: {
                split: answer_table.select(KEYS[task])
                for split, answer_table in answers_by_split.items()
            }
            for panel, answers_by_split in answer_tables.items()
        }
    targets_by_panel_and_split = {
        panel: target_tables_by_split
        for panel, target_tables_by_split in targets_by_panel_and_split.items()
        if target_tables_by_split
    }
    if not targets_by_panel_and_split:
        raise ValueError("No official or supplied target rows fall in this date range")

    run_seed = int(
        metadata_hash(
            [
                solution_name,
                task,
                history_start_date.isoformat(),
                history_end_date.isoformat(),
                prediction_start_date.isoformat(),
                prediction_end_date.isoformat(),
                list(panels),
            ]
        )[:8],
        16,
    )
    # Use the same solution call and prediction validation as local cases.
    solution_result = run_solution(
        solution_name,
        task,
        release_slice,
        targets_by_panel_and_split,
        answer_tables,
        solution_parameters,
        run_seed,
    )
    # Persist predictions and metadata only after the solution passes validation.
    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_id = f"{run_timestamp}-{solution_name}-{task}-{uuid.uuid4().hex[:6]}"
    run_directory = runs_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    # Concatenate only complete task prediction schemas. Split membership is
    # recoverable from timestamp/window/path keys and is recorded in run.json.
    prediction_tables = [
        panel_predictions
        for predictions_by_split in solution_result.predictions.values()
        for panel_predictions in predictions_by_split.values()
    ]
    prediction_file = run_directory / "predictions.csv"
    pl.concat(prediction_tables).write_csv(prediction_file)
    solution_result.metrics.write_csv(run_directory / "metrics.csv")
    target_splits = list(
        dict.fromkeys(
            split
            for panel_predictions in solution_result.predictions.values()
            for split in panel_predictions
        )
    )
    run_metadata_payload = {
        "run_id": run_id,
        "solution": solution_name,
        "task": task,
        "history_start_date": history_start_date.isoformat(),
        "history_end_date": history_end_date.isoformat(),
        "prediction_start_date": prediction_start_date.isoformat(),
        "prediction_end_date": prediction_end_date.isoformat(),
        "target_splits": target_splits,
        "panels": list(release_slice.panel_slices_by_panel),
        "source_fingerprint": release_slice.source_fingerprint,
        "parameters": solution_parameters or {},
        "scored": answer_tables is not None,
        "final_submission_eligible": "train" not in target_splits,
        "metrics": solution_result.metrics.to_dicts(),
    }
    (run_directory / "run.json").write_text(json.dumps(run_metadata_payload, indent=2))
    logger.info("Predictions: %s", prediction_file)
    if not solution_result.metrics.is_empty():
        logger.info("Scores and diagnostics:\n%s", solution_result.metrics)
    return run_directory


def _partition_answer_key(
    task: Task,
    answer_key_table: pl.DataFrame,
    target_templates: PanelSplitTableMap[pl.DataFrame],
) -> PanelSplitTableMap[pl.DataFrame]:
    """Assign supplied answer rows to exactly one split and panel template.

    ``split`` is an optional column for distinguishing OD scenarios, whose
    path identifiers can repeat across release splits. Each answer row must
    match exactly one selected official target table.

    Args:
        task: Task whose identifier columns are matched to target rows.
        answer_key_table: Supplied target identifiers and answer values.
        target_templates: Official target rows grouped by panel and split.

    Returns:
        Matching answers grouped by their unique panel and split.

    Raises:
        ValueError: If columns are missing, an answer row matches zero or
            multiple targets, or a selected queue window is incomplete.
    """
    required_columns = set(KEYS[task] + VALUES[task])
    missing_columns = required_columns - set(answer_key_table.columns)
    if missing_columns:
        raise ValueError(f"Answer file lacks columns {sorted(missing_columns)}")
    answer_key_table = answer_key_table.with_row_index("_answer_row")
    matched_rows_by_location: dict[tuple[Panel, Split], list[int]] = {}
    match_counts: Counter[int] = Counter()

    for panel, target_tables_by_split in target_templates.items():
        for split, target_table in target_tables_by_split.items():
            answer_candidates = answer_key_table
            if "split" in answer_candidates:
                answer_candidates = answer_candidates.filter(pl.col("split") == split)
            if "panel" in answer_candidates:
                answer_candidates = answer_candidates.filter(pl.col("panel") == panel)
            matched_row_ids = answer_candidates.join(
                target_table.select(KEYS[task]).unique(),
                on=KEYS[task],
                how="inner",
            )["_answer_row"].to_list()
            if matched_row_ids:
                match_counts.update(matched_row_ids)
                matched_rows_by_location[(panel, split)] = matched_row_ids

    if any(
        match_counts[row_index] != 1 for row_index in range(answer_key_table.height)
    ):
        raise ValueError(
            "Each answer key must match exactly one target in the selected date range"
        )

    answer_tables_by_panel_and_split: PanelSplitTableMap[pl.DataFrame] = {
        panel: {
            split: answer_key_table.filter(pl.col("_answer_row").is_in(row_ids)).drop(
                "_answer_row", "split", strict=False
            )
            for (matched_panel, split), row_ids in matched_rows_by_location.items()
            if matched_panel == panel
        }
        for panel in target_templates
        if any(matched_panel == panel for matched_panel, _ in matched_rows_by_location)
    }
    # A partial queue window cannot produce the planned full-window overlap score.
    if task == "queue":
        for panel, answers_by_split in answer_tables_by_panel_and_split.items():
            for split, answer_table in answers_by_split.items():
                selected_windows = answer_table.select("window_id").unique()
                complete_window_keys = (
                    target_templates[panel][split]
                    .join(selected_windows, on="window_id", how="semi")
                    .select(KEYS[task])
                )
                missing_window_keys = complete_window_keys.join(
                    answer_table.select(KEYS[task]),
                    on=KEYS[task],
                    how="anti",
                )
                if (
                    complete_window_keys.height != answer_table.height
                    or missing_window_keys.height
                ):
                    raise ValueError(
                        "Queue answer files must include every target row in each selected window"
                    )
    return answer_tables_by_panel_and_split
