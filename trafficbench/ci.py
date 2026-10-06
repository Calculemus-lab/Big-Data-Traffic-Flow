"""Package prepared panels for hosted jobs and combine their saved results.

Each hosted job receives one panel archive and produces a normal run directory.
The combiner requires one compatible run for every expected panel, then
recalculates task metrics from their per-case CSV files.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Final

import polars as pl
import typer
from pydantic import DirectoryPath, FilePath

from .contracts import (
    BundleIndex,
    BundleShard,
    CiMatrix,
    MatrixEntry,
    Panel,
    RunMetadata,
    Task,
)
from .data import file_hash
from .metrics import aggregate
from .runner import report, verify_benchmark

MAX_RELEASE_ASSET_BYTES: Final = 2 * 1024**3
logger = logging.getLogger(__name__)


def package(benchmark_directory: DirectoryPath, package_directory: Path) -> BundleIndex:
    """Write one archive per panel from a verified benchmark directory.

    ``benchmark_directory`` contains the manifest and Parquet files prepared
    locally. ``package_directory`` must be empty. Return and save an index with
    each archive name, SHA-256 checksum, panel, and available tasks for hosted
    job scheduling.

    Args:
        benchmark_directory: Verified prepared benchmark to archive.
        package_directory: Empty output directory for panel archives and index.

    Returns:
        The saved index describing each panel archive.

    Raises:
        ValueError: If the output directory is non-empty or an archive exceeds
            the configured upload size limit.
    """
    # Hash the full prepared benchmark before creating panel archives, so the
    # published assets all come from one unchanged prepared benchmark.
    manifest = verify_benchmark(benchmark_directory)
    if package_directory.exists() and any(package_directory.iterdir()):
        raise ValueError("Package output must be empty")
    package_directory.mkdir(parents=True, exist_ok=True)
    profile = manifest.profile
    bundle_index = BundleIndex(
        profile=profile, benchmark_id=manifest.benchmark_id, shards=[]
    )
    panels: set[Panel] = {case.panel for case in manifest.cases}
    for panel in sorted(panels):
        # Include the full manifest for identity and only this panel's inputs
        # and network files to keep each hosted archive independent.
        archive_file_name = f"{profile}-{panel}.tar.gz"
        archive_path = package_directory / archive_file_name
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive_file:
            archive_file.add(
                benchmark_directory / "manifest.json", arcname="manifest.json"
            )
            for relative_path in sorted(manifest.files):
                if relative_path.startswith((panel + "/", "network/" + panel + "/")):
                    archive_file.add(
                        benchmark_directory / relative_path,
                        arcname=relative_path,
                        recursive=False,
                    )
        # A hosted release asset must fit the configured upload limit.
        if archive_path.stat().st_size >= MAX_RELEASE_ASSET_BYTES:
            raise ValueError(
                f"{archive_file_name} exceeds the GitHub release asset limit"
            )
        task_set: set[Task] = {
            case.task for case in manifest.cases if case.panel == panel
        }
        tasks: list[Task] = sorted(task_set)
        panel_bundle_shard = BundleShard(
            panel=panel,
            asset=archive_file_name,
            sha256=file_hash(archive_path),
            tasks=tasks,
        )
        bundle_index.shards.append(panel_bundle_shard)
        logger.info(
            "Packed %s: %.1f MiB",
            archive_file_name,
            archive_path.stat().st_size / 1024**2,
        )
    (package_directory / f"{profile}-index.json").write_text(
        bundle_index.model_dump_json(indent=2)
    )
    return bundle_index


def matrix(bundle_index: BundleIndex, task: Task) -> CiMatrix:
    """Return only panel archives containing cases for one task.

    Args:
        bundle_index: Archive inventory created by :func:`package`.
        task: Task whose panel jobs are selected.

    Returns:
        CI matrix entries with each panel archive name and checksum.
    """
    entries: list[MatrixEntry] = [
        MatrixEntry(panel=shard.panel, asset=shard.asset, sha256=shard.sha256)
        for shard in bundle_index.shards
        if task in shard.tasks
    ]
    return CiMatrix(include=entries)


def unpack(
    archive_file: FilePath, extraction_directory: Path, expected_sha256: str
) -> None:
    """Verify and extract one downloaded panel archive.

    Refuse a checksum mismatch or an archive containing anything besides
    regular files. Extract them into ``extraction_directory`` using tarfile's
    data filter.

    Args:
        archive_file: Downloaded archive whose checksum is verified.
        extraction_directory: Destination for the verified archive contents.
        expected_sha256: Checksum recorded in the bundle index.

    Raises:
        ValueError: If the checksum differs or the archive includes non-files.
    """
    if file_hash(archive_file) != expected_sha256:
        raise ValueError("Downloaded benchmark checksum does not match the index")
    extraction_directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_file, "r:gz") as tar_archive:
        if any(not member.isfile() for member in tar_archive.getmembers()):
            raise ValueError("Benchmark archive must contain regular files only")
        tar_archive.extractall(extraction_directory, filter="data")


def _read_metrics(metric_file_paths: Iterable[FilePath]) -> pl.DataFrame:
    """Concatenate panel metrics with identical column names and order.

    Args:
        metric_file_paths: One saved metric CSV for each expected panel.

    Returns:
        All metric rows in file order.

    Raises:
        ValueError: If no metrics are supplied or their schemas differ.
    """
    panel_metrics = [
        (metrics_file_path, pl.read_csv(metrics_file_path))
        for metrics_file_path in metric_file_paths
    ]
    if not panel_metrics:
        raise ValueError("No panel metrics to combine")
    columns = panel_metrics[0][1].columns
    for metrics_file_path, metrics_table in panel_metrics[1:]:
        if metrics_table.columns != columns:
            raise ValueError(f"Metric columns differ in {metrics_file_path}")
    return pl.concat([metrics_table for _, metrics_table in panel_metrics])


def combine(
    runs_directory: DirectoryPath,
    bundle_index: BundleIndex,
    solution_name: str,
    task: Task,
    combined_run_directory: Path,
) -> RunMetadata:
    """Write one summary from complete, compatible panel runs.

    ``bundle_index`` lists panels expected for ``task``. Search
    ``runs_directory`` for one completed ``solution_name`` run per panel.
    Require matching benchmark, code, parameters, dependencies, and profile.
    Recalculate scores from per-case rows, then write combined metrics,
    metadata, and Markdown under ``combined_run_directory``.

    Args:
        runs_directory: Parent directory containing panel run records.
        bundle_index: Expected benchmark ID and panel-task assignments.
        solution_name: Solution whose panel runs are combined.
        task: Task that every selected panel run must evaluate.
        combined_run_directory: Destination for the overall report and metrics.

    Returns:
        The combined run metadata after writing all report files.

    Raises:
        ValueError: If panel results are incomplete, duplicated, failed, or
            produced from inconsistent run settings.
    """
    # Choose exactly one completed run for every panel in the index.
    expected_panels: list[Panel] = sorted(
        shard.panel for shard in bundle_index.shards if task in shard.tasks
    )
    run_records: list[tuple[Path, RunMetadata]] = []
    for run_metadata_file in runs_directory.rglob("run.json"):
        panel_run_metadata = RunMetadata.model_validate_json(
            run_metadata_file.read_text()
        )
        run_records.append((run_metadata_file, panel_run_metadata))
    selected_runs = [
        (run_metadata_file, panel_run_metadata)
        for run_metadata_file, panel_run_metadata in run_records
        if panel_run_metadata.solution == solution_name
        and panel_run_metadata.task == task
    ]
    actual_panels = sorted(
        panel
        for _, panel_run_metadata in selected_runs
        for panel in panel_run_metadata.panels
    )
    if len(selected_runs) != len(expected_panels) or actual_panels != expected_panels:
        raise ValueError(
            "Missing, duplicate or unexpected panel results. Refusing a partial overall score"
        )

    # A successful status alone is insufficient: the expected summary and
    # resource fields must also be present in every panel's saved run record.
    for _, panel_run_metadata in selected_runs:
        if (
            panel_run_metadata.status != "completed"
            or panel_run_metadata.benchmark_id != bundle_index.benchmark_id
            or len(panel_run_metadata.panels) != 1
        ):
            raise ValueError("Failed panel or mismatched benchmark")
        if (
            panel_run_metadata.metrics is None
            or panel_run_metadata.seconds is None
            or panel_run_metadata.peak_memory_mb is None
        ):
            raise ValueError("Completed panel run is missing metrics or resource data")
    if not selected_runs:
        raise ValueError("No panel results")

    # Comparisons are meaningful only when all shards ran the same experiment.
    reference_run = selected_runs[0][1]
    if any(
        panel_run_metadata.params != reference_run.params
        or panel_run_metadata.evaluator_hash != reference_run.evaluator_hash
        or panel_run_metadata.code_hash != reference_run.code_hash
        or panel_run_metadata.dependencies != reference_run.dependencies
        or panel_run_metadata.commit != reference_run.commit
        or panel_run_metadata.profile != reference_run.profile
        for _, panel_run_metadata in selected_runs[1:]
    ):
        raise ValueError(
            "Panel runs disagree on parameters, code, evaluator, environment, commit, or profile"
        )

    # Re-aggregate saved case rows instead of averaging already reduced
    # panel scores, which would distort family and condition weighting.
    panel_metric_file_paths = [
        run_metadata_file.parent / "metrics.csv"
        for run_metadata_file, _ in selected_runs
    ]
    combined_case_metrics_table = _read_metrics(panel_metric_file_paths)
    combined_run_metadata = selected_runs[0][1].model_copy()
    combined_run_metadata.run_id = "combined-" + combined_run_metadata.run_id
    combined_run_metadata.panels = expected_panels
    combined_run_metadata.metrics = aggregate(combined_case_metrics_table)
    combined_run_metadata.seconds = sum(
        panel_run_metadata.seconds or 0.0 for _, panel_run_metadata in selected_runs
    )
    combined_run_metadata.peak_memory_mb = max(
        panel_run_metadata.peak_memory_mb or 0.0
        for _, panel_run_metadata in selected_runs
    )
    combined_run_metadata.panel_runs = [
        panel_run_metadata.run_id for _, panel_run_metadata in selected_runs
    ]
    combined_run_metadata.baseline_run = None

    if solution_name != "baseline":
        # Each non-baseline panel points to its own matching baseline run.
        # Combine those baseline case rows under the same weighting scheme.
        baseline_paths: list[Path] = []
        for _, panel_run_metadata in selected_runs:
            baseline_id = panel_run_metadata.baseline_run
            if baseline_id is None:
                raise ValueError("Panel run is missing its baseline reference")
            matches = [
                (run_metadata_file, baseline_metadata)
                for run_metadata_file, baseline_metadata in run_records
                if baseline_metadata.run_id == baseline_id
                and baseline_metadata.status == "completed"
            ]
            if len(matches) != 1:
                raise ValueError("Missing baseline result")
            baseline_paths.append(matches[0][0].parent / "metrics.csv")
        combined_run_metadata.baseline_metrics = aggregate(
            _read_metrics(baseline_paths)
        )

    # Publish the combined tables only after all panels and baseline links
    # have been checked.
    combined_run_directory.mkdir(parents=True, exist_ok=True)
    combined_case_metrics_table.write_csv(combined_run_directory / "metrics.csv")
    (combined_run_directory / "run.json").write_text(
        combined_run_metadata.model_dump_json(indent=2, exclude_none=True)
    )
    (combined_run_directory / "report.md").write_text(report(combined_run_metadata))
    return combined_run_metadata


# Typer needs pathlib.Path in CLI signatures. The package functions above use
# Pydantic FilePath and DirectoryPath where the path kind is fixed.
ci_app = typer.Typer(help="Package and combine hosted benchmark runs.")


@ci_app.command("matrix")
def matrix_command(
    bundle_index_file: Annotated[
        Path,
        typer.Option("--index", exists=True, file_okay=True, dir_okay=False),
    ],
    task: Annotated[Task, typer.Option("--task")],
) -> None:
    """Print the panel job matrix for a task.

    Args:
        bundle_index_file: JSON index created by ``bench package-ci``.
        task: Task whose panel archives are included in the matrix.
    """
    bundle_index = BundleIndex.model_validate_json(bundle_index_file.read_text())
    print(matrix(bundle_index, task).model_dump_json())


@ci_app.command("unpack")
def unpack_command(
    archive_file: Annotated[
        Path,
        typer.Option("--archive", exists=True, file_okay=True, dir_okay=False),
    ],
    extraction_directory: Annotated[Path, typer.Option("--output")],
    expected_sha256: Annotated[str, typer.Option("--sha256")],
) -> None:
    """Verify and unpack a hosted panel archive.

    Args:
        archive_file: Downloaded panel archive.
        extraction_directory: Destination directory for its verified contents.
        expected_sha256: Expected checksum copied from the panel index.
    """
    unpack(archive_file, extraction_directory, expected_sha256)


@ci_app.command("combine")
def combine_command(
    runs_directory: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, dir_okay=True),
    ],
    bundle_index_file: Annotated[
        Path,
        typer.Option("--index", exists=True, file_okay=True, dir_okay=False),
    ],
    solution_name: Annotated[str, typer.Option("--solution")],
    task: Annotated[Task, typer.Option()],
    combined_run_directory: Annotated[Path, typer.Option("--output")],
) -> None:
    """Combine complete panel runs into one task report.

    Args:
        runs_directory: Parent directory containing hosted run folders.
        bundle_index_file: JSON index listing required panel results.
        solution_name: Solution whose results will be combined.
        task: Task to select and aggregate.
        combined_run_directory: Destination for the combined report.
    """
    bundle_index = BundleIndex.model_validate_json(bundle_index_file.read_text())
    combine(runs_directory, bundle_index, solution_name, task, combined_run_directory)


def main() -> None:
    """Run the hosted benchmark commands."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ci_app()


if __name__ == "__main__":
    main()
