# Traffic Flow Bench team

This repository contains team solutions and a local runner for the [2026 IEEE
Big Data Traffic Flow Bench competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview).
The guides below take a new contributor from the competition rules through
local experiments to a validated submission.

## Set up the project

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for your
platform, then run these commands from the repository root. The project uses
Python 3.12 and pins the expected uv version in `pyproject.toml`.

```sh
uv python install 3.12
uv sync --locked
```

`uv sync --locked` creates or updates the project environment from
`uv.lock`, including the development tools used by the repository checks.

The [official competition repository](official_competition_repo/README.md) is
a Git submodule. Initialize it so the organizer's schemas, scoring references,
baselines, and the task illustration used in the guides are available locally:

```sh
git submodule update --init --recursive
```

## Check your changes

Run these commands from the repository root. They match the checks used in
continuous integration:

```sh
uv run --locked pyright
uv run --locked ruff check
uv run --locked ruff format --check
uv run --locked python -m pytest -q
```

Pyright checks Python types, Ruff checks Python style, and Ruff's format check
reports files that need formatting. Pytest runs the project test suite. To
apply Ruff formatting instead of checking it, run:

```sh
uv run --locked ruff format
```

## Read the guides in order

1. [Competition and traffic theory](docs/COMPETITION_AND_THEORY.md) explains
   the network, released data, prediction tasks, and competition scoring.
2. [Download the data](docs/GET_DATA.md) explains competition access,
   authentication, download, and extraction.
3. [Trafficbench](docs/TRAFFICBENCH.md) explains the local solution interface,
   repeatable train cases, local scoring, prediction runs, and submission
   assembly.
4. [Submit predictions](docs/SUBMIT.md) explains how to upload the assembled
   competition file.

The [public release package reference](docs/RELEASE_PACKAGE_REFERENCE.md) is a
lookup for downloaded file paths, table columns, and row identifiers. Use it
when you need a specific field definition; it is not required for the main
reading path.

## Team workflow and reference guides

- [Experiment records](docs/EXPERIMENTS.md) explains local run reports and the
  shared record for competition validation scores.
- [Candidate approaches](docs/APPROACHES.md) lists methods to compare by task.
- [Workflow and branch conventions](docs/WORKFLOW.md) describes team
  experiment practices.
- [GitHub-hosted benchmark experiments](docs/HOSTED_BENCHMARK_RUNS.md)
  explains how to run prepared local cases in Actions.
- [Dependencies](docs/DEPENDENCIES.md) describes dependency changes and the
  shared environment.
- [Saved results](results/README.md) describes which research outputs belong
  in `results/`.
