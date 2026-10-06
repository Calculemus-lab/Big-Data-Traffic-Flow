"""Command-line entry points for the local benchmark workflow.

Typer parses commands and file paths. Each command delegates preparation,
execution, packaging, or submission work to the corresponding package module.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import typer
from pydantic import TypeAdapter, ValidationError

from .contracts import (
    ALL_PANELS,
    JsonObject,
    Panel,
    RunMetadata,
)

app = typer.Typer(help="One command to test a traffic-flow idea.")
# Typer needs inline Literal choices because it does not resolve PEP 695 aliases.
# Typer needs pathlib.Path for file options. The corresponding package functions
# use Pydantic FilePath and DirectoryPath where the path kind is fixed.
PARAMETERS_ADAPTER = TypeAdapter[JsonObject](JsonObject)
DATE_ADAPTER = TypeAdapter[date](date)
PANEL_LIST_ADAPTER = TypeAdapter[list[Panel]](list[Panel])


def selected_panels(panel_names: list[str]) -> tuple[Panel, ...]:
    """Validate panel options and preserve their order, or select all panels.

    Args:
        panel_names: Panel names supplied on the command line.

    Returns:
        Unique valid panel names in the given order, or all panels when empty.

    Raises:
        typer.BadParameter: If a supplied name is not a released directional panel.
    """
    if not panel_names:
        return ALL_PANELS
    try:
        validated_panels = PANEL_LIST_ADAPTER.validate_python(panel_names, strict=True)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--panel") from error
    return tuple(dict.fromkeys(validated_panels))


@app.command("prepare")
def prepare_command(
    release_directory: Annotated[
        Path,
        typer.Option(
            "--data", exists=True, file_okay=False, help="Extracted release directory"
        ),
    ],
    history_start_date: Annotated[
        date,
        typer.Option(
            ...,
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="First train date available as features and historical targets",
        ),
    ],
    history_end_date: Annotated[
        date,
        typer.Option(
            ...,
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="Exclusive end date for historical train labels",
        ),
    ],
    prediction_end_date: Annotated[
        date,
        typer.Option(
            ...,
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="Exclusive end date for generated train targets",
        ),
    ],
    prediction_start_date: Annotated[
        date | None,
        typer.Option(
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="First generated target date, defaulting to history end",
        ),
    ] = None,
    profile: Literal["quick", "full"] = "quick",
    benchmark_directory: Annotated[Path | None, typer.Option("--output")] = None,
    cv_scheme_file: Annotated[
        Path, typer.Option("--scheme", exists=True, file_okay=True, dir_okay=False)
    ] = Path("config/cv_scheme.yaml"),
    panels: Annotated[list[str], typer.Option("--panel", "-p")] = [],  # noqa: B006
) -> None:
    """Generate reproducible experiment cases from the public train split.

    Args:
        release_directory: Extracted release root containing ``config/``.
        history_start_date: First date included in visible history.
        history_end_date: Exclusive end of historical labels.
        prediction_end_date: Exclusive end of generated target dates.
        prediction_start_date: First target date, defaulting to history end.
        profile: Named panel-selection and case-generation configuration.
        benchmark_directory: Output directory for prepared cases.
        cv_scheme_file: YAML file defining queue and ODME case settings.
        panels: Optional panel names. An empty list uses the profile selection.
    """
    from .prepare import prepare

    prepare(
        release_directory=release_directory,
        benchmark_directory=benchmark_directory or Path("data/benchmarks") / profile,
        profile=profile,
        cv_scheme_file=cv_scheme_file,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        prediction_start_date=prediction_start_date,
        prediction_end_date=prediction_end_date,
        panels=selected_panels(panels),
    )


@app.command("new")
def new_command(
    solution_name: str, task: Literal["state", "queue", "odme"] = "state"
) -> None:
    """Create a solution package exporting the selected predictor function.

    The solution name must be a safe Python package name because the runner imports it
    as ``solutions.<name>``. The generated predictor starts from the baseline.

    Args:
        solution_name: Lowercase package name to create under ``solutions``.
        task: Task function to export from the generated package.

    Raises:
        typer.BadParameter: If the name is unsafe or the package already exists.
    """
    # Reject names that could refer outside the solutions package or collide
    # with an existing module before creating either generated file.
    if not re.fullmatch(r"[a-z][a-z0-9_]*", solution_name):
        raise typer.BadParameter(
            "Use lowercase letters, digits and underscores. Start with a letter"
        )
    solution_directory = Path("solutions") / solution_name
    if solution_directory.with_suffix(".py").exists():
        raise typer.BadParameter(
            f"{solution_directory.with_suffix('.py')} already exists"
        )
    if solution_directory.exists():
        raise typer.BadParameter(f"{solution_directory} already exists")
    solution_directory.mkdir()
    (solution_directory / "__init__.py").write_text(
        f'"""Expose the {task} predictor to the benchmark runner."""\n\n'
        f"from .model import {task}\n"
    )
    task_frame_type_name = {
        "state": "StateFrame",
        "queue": "QueueFrame",
        "odme": "OdmeFrame",
    }[task]
    (solution_directory / "model.py").write_text(
        f'''"""Implement the {task} approach here. Add helper modules as needed."""

from solutions import baseline
from trafficbench.contracts import Panel, ReleasePackageSlice, Split
from trafficbench.table_types import {task_frame_type_name}


def {task}(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, {task_frame_type_name}]
    ],
) -> dict[Panel, dict[Split, {task_frame_type_name}]]:
    """Return predictions for the supplied {task} target rows.

    Use the lazy tables selected for this run. Collect only the tables and
    columns needed by this approach. Preserve each target row and replace its
    zero-valued prediction column with a value.

    Args:
        release_slice: Lazy network and release tables, historical train labels,
            date boundaries, and solution settings.
        target_templates_by_panel_and_split: Zero-valued Polars LazyFrames,
            grouped by panel and release split.

    Returns:
        Polars LazyFrames containing every requested target row and its prediction,
        grouped by panel and release split.
    """
    # Use the baseline predictor as a complete starting implementation.
    return baseline.{task}(
        release_slice,
        target_templates_by_panel_and_split,
    )
'''
    )
    print(
        f"Created {solution_directory}/. Use bench run {solution_name} --task {task} "
        "with --target-range validation, private, or both, or with prediction dates."
    )


@app.command("run")
def run_command(
    solution_name: Annotated[str, typer.Argument()] = "baseline",
    *,
    history_start_date: Annotated[
        date | None,
        typer.Option(
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="First train date available as features and historical targets",
        ),
    ] = None,
    history_end_date: Annotated[
        date | None,
        typer.Option(
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="Exclusive end date for historical train labels",
        ),
    ] = None,
    prediction_start_date: Annotated[
        date | None,
        typer.Option(
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="First target date, defaulting to history end for an explicit range",
        ),
    ] = None,
    prediction_end_date: Annotated[
        date | None,
        typer.Option(
            parser=DATE_ADAPTER.validate_python,
            metavar="YYYY-MM-DD",
            help="Exclusive target and visible-data end date",
        ),
    ] = None,
    task: Literal["state", "queue", "odme"] = "state",
    release_directory: Annotated[
        Path,
        typer.Option(
            "--data", exists=True, file_okay=False, help="Extracted release directory"
        ),
    ] = Path("kaggle_public"),
    target_shortcut: Annotated[
        Literal["validation", "private", "both"] | None,
        typer.Option(
            "--target-range",
            help="Use the complete validation, private, or validation-through-private range",
        ),
    ] = None,
    panels: Annotated[list[str], typer.Option("--panel", "-p")] = [],  # noqa: B006
    answer_key_file: Annotated[
        Path | None,
        typer.Option(
            "--answers",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="CSV containing target rows and answer values for local scoring",
        ),
    ] = None,
    parameters_json: Annotated[str, typer.Option("--params")] = "{}",
    runs_directory: Annotated[Path, typer.Option("--runs")] = Path("runs"),
) -> None:
    """Predict selected target rows and optionally score supplied answers.

    Args:
        solution_name: Importable package name below ``solutions``.
        history_start_date: First train date visible as features and labels.
        history_end_date: Exclusive end of visible historical labels.
        prediction_start_date: First target date in an explicit target interval.
        prediction_end_date: Exclusive end of an explicit target interval.
        task: Prediction task selected for this run.
        release_directory: Extracted release root containing ``config/``.
        target_shortcut: Complete validation, private, or combined date range.
        panels: Optional panel names. An empty list selects every panel.
        answer_key_file: Optional CSV supplying target rows and score answers.
        parameters_json: JSON object passed to the solution function.
        runs_directory: Parent directory for predictions and run reports.
    """
    from .runner import run_predictions

    if not re.fullmatch(r"[a-z][a-z0-9_]*", solution_name):
        raise typer.BadParameter("Invalid solution name")
    # Parse the JSON parameter object once at the CLI boundary. Individual
    # solutions may validate their own named parameters more narrowly.
    try:
        solution_parameters = PARAMETERS_ADAPTER.validate_json(
            parameters_json, strict=True
        )
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--params") from error
    run_predictions(
        solution_name=solution_name,
        task=task,
        release_directory=release_directory,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        prediction_start_date=prediction_start_date,
        prediction_end_date=prediction_end_date,
        target_shortcut=target_shortcut,
        panels=selected_panels(panels),
        solution_parameters=solution_parameters,
        answer_key_file=answer_key_file,
        runs_directory=runs_directory,
    )


@app.command("experiment")
def experiment_command(
    solution_name: Annotated[str, typer.Argument()] = "baseline",
    task: Literal["state", "queue", "odme"] = "state",
    prepared_benchmark_directory: Annotated[
        Path,
        typer.Option("--benchmark", exists=True, file_okay=False),
    ] = Path("data/benchmarks/quick"),
    parameters_json: Annotated[str, typer.Option("--params")] = "{}",
    runs_directory: Annotated[Path, typer.Option("--runs")] = Path("runs"),
    panel: Literal[
        "D7_I10_E",
        "D7_I10_W",
        "D7_I210_E",
        "D7_I210_W",
        "D7_I405_N",
        "D7_I405_S",
        "D12_I5_N",
        "D12_I5_S",
        "D12_I405_N",
        "D12_I405_S",
    ]
    | None = None,
) -> None:
    """Run generated train cases from a prepared benchmark directory.

    Args:
        solution_name: Importable package name below ``solutions``.
        task: Task whose generated cases are executed.
        prepared_benchmark_directory: Directory created by ``bench prepare``.
        parameters_json: JSON object passed to the solution function.
        runs_directory: Parent directory for predictions and run reports.
        panel: Optional panel restriction for one hosted panel job.
    """
    from .runner import run_experiment

    try:
        solution_parameters = PARAMETERS_ADAPTER.validate_json(
            parameters_json, strict=True
        )
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--params") from error
    run_experiment(
        solution_name=solution_name,
        task=task,
        benchmark_directory=prepared_benchmark_directory,
        solution_parameters=solution_parameters,
        runs_directory=runs_directory,
        panel=panel,
    )


@app.command("package-ci")
def package_ci_command(
    package_directory: Annotated[Path, typer.Option("--output")],
    prepared_benchmark_directory: Annotated[
        Path,
        typer.Option("--benchmark", exists=True, file_okay=False, dir_okay=True),
    ] = Path("data/benchmarks/quick"),
) -> None:
    """Package a prepared benchmark into panel archives for hosted runners.

    Args:
        package_directory: Destination for archives and their index.
        prepared_benchmark_directory: Verified cases to package by panel.
    """
    from .ci import package

    package(prepared_benchmark_directory, package_directory)


@app.command("compare")
def compare_command(
    run_directories: Annotated[
        list[Path], typer.Argument(exists=True, file_okay=False)
    ],
) -> None:
    """Compare saved runs that share their benchmark and scoring conditions.

    Args:
        run_directories: Run folders to compare.
    """
    from .runner import compare

    compare(run_directories)


@app.command("summary")
def summary_command(
    runs_directory: Annotated[Path, typer.Option("--runs")] = Path("runs"),
    summary_csv_file: Annotated[Path, typer.Option("--output")] = Path(
        "runs/summary.csv"
    ),
) -> None:
    """Export selected run metadata and metrics as one CSV row per run.

    Args:
        runs_directory: Parent directory containing saved run records.
        summary_csv_file: Destination CSV path. Parent directories are created.
    """
    # Runs already store validated JSON metadata. One CSV row per run makes
    # statuses and scores easy to filter without reading each report by hand.
    run_summary_rows: list[dict[str, object]] = []
    for run_metadata_file in sorted(runs_directory.glob("*/run.json")):
        run_metadata = RunMetadata.model_validate_json(run_metadata_file.read_text())
        run_summary_rows.append(
            {
                "run_id": run_metadata.run_id,
                "timestamp": run_metadata.timestamp,
                "solution": run_metadata.solution,
                "task": run_metadata.task,
                "status": run_metadata.status,
                "profile": run_metadata.profile,
                "benchmark_id": run_metadata.benchmark_id,
                "commit": run_metadata.commit,
                "seconds": run_metadata.seconds,
                **(run_metadata.metrics or {}),
                "path": str(run_metadata_file.parent),
            }
        )
    summary_csv_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(run_summary_rows).write_csv(summary_csv_file)
    print(summary_csv_file)


@app.command("assemble")
def assemble_command(
    state_predictions_file: Annotated[
        Path,
        typer.Option("--state", exists=True, file_okay=True, dir_okay=False),
    ],
    queue_predictions_file: Annotated[
        Path,
        typer.Option("--queue", exists=True, file_okay=True, dir_okay=False),
    ],
    odme_predictions_file: Annotated[
        Path,
        typer.Option("--odme", exists=True, file_okay=True, dir_okay=False),
    ],
    submission_key_file: Annotated[
        Path,
        typer.Option("--key", exists=True, file_okay=True, dir_okay=False),
    ],
    sample_submission_template_file: Annotated[
        Path,
        typer.Option("--template", exists=True, file_okay=True, dir_okay=False),
    ],
    submission_output_file: Annotated[Path, typer.Option("--output")],
) -> None:
    """Assemble the three task prediction files into one upload CSV.

    Args:
        state_predictions_file: Task 1 rows with identifier columns and predictions.
        queue_predictions_file: Task 2 rows with window, timestamp, link, and prediction.
        odme_predictions_file: Task 4 path-flow rows for each requested path.
        submission_key_file: Official table mapping task identifiers to row IDs.
        sample_submission_template_file: Official combined CSV template.
        submission_output_file: Destination for the validated upload CSV.
    """
    from .submission import assemble

    assemble(
        state_predictions_file,
        queue_predictions_file,
        odme_predictions_file,
        submission_key_file,
        sample_submission_template_file,
        submission_output_file,
    )


@app.command("validate")
def validate_command(
    submission_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)
    ],
    sample_submission_template_file: Annotated[
        Path,
        typer.Option("--template", exists=True, file_okay=True, dir_okay=False),
    ],
) -> None:
    """Validate a final CSV against the official upload template.

    Args:
        submission_file: Candidate upload CSV to check.
        sample_submission_template_file: Official row order and value columns.
    """
    from .submission import validate_submission

    validate_submission(submission_file, sample_submission_template_file)


@app.command("fixture")
def fixture_command(
    fixture_release_directory: Annotated[Path, typer.Option("--output")] = Path(
        "data/fixture"
    ),
) -> None:
    """Create a small synthetic release for continuous-integration checks.

    Args:
        fixture_release_directory: New or empty destination for fixture data.
    """
    from .fixture import make_fixture

    make_fixture(fixture_release_directory)


def main() -> None:
    """Run the benchmark command group."""
    app()


if __name__ == "__main__":
    main()
