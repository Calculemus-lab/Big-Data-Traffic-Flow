# Traffic Flow Bench team

This repository contains team solutions and a local runner for the [2026 IEEE
Big Data Traffic Flow Bench competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview).

## Start here

Install Python 3.12 and the locked project environment from the repository
root:

```sh
uv python install 3.12
uv sync --locked
```

The following guides explain the competition, data, and solution workflow. Read them in order:

1. [Competition and traffic theory](docs/COMPETITION_AND_THEORY.md): task
   definitions, solution inputs, scoring, and traffic concepts.
2. [Download the data](docs/GET_DATA.md): Kaggle access and release download steps.
3. [Trafficbench](docs/TRAFFICBENCH.md): solution functions, local experiments,
   prediction runs, and submission assembly.
4. [Submit predictions](docs/SUBMIT.md): upload the assembled competition file.

Use the [public release package reference](docs/RELEASE_PACKAGE_REFERENCE.md)
to look up downloaded file paths, table fields, and the fields that identify
each row. You do not need to read it before writing a solution.

## Team workflow

- [Experiment records](docs/EXPERIMENTS.md)
- [Approach notes](docs/APPROACHES.md)
- [Workflow and branch conventions](docs/WORKFLOW.md)
- [GitHub-hosted benchmark experiments](docs/HOSTED_BENCHMARK_RUNS.md)
- [Dependencies](docs/DEPENDENCIES.md)
- [Saved results](results/README.md)

## Run development checks

Run these commands from the repository root. They match the checks used in CI:

```sh
uv run --locked pyright
uv run --locked ruff check
uv run --locked ruff format --check
uv run --locked python -m pytest -q
```

To apply Ruff formatting instead of checking it, run:

```sh
uv run --locked ruff format
```

The [official competition repository](official_competition_repo/README.md)
contains organizer schemas, scorers, and baselines. It is pinned as a Git
submodule. After cloning this repository, initialize it with:

```sh
git submodule update --init --recursive
```
