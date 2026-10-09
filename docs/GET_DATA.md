# Download the competition data

The competition release is hosted on Kaggle. Join the
[competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
before downloading it. You can download the archive from the
[Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
or use the Kaggle command-line interface (CLI).

## Prepare the project environment

From the repository root, install Python 3.12 and the locked project
environment. The environment includes the Kaggle CLI:

```sh
uv python install 3.12
uv sync --locked
```

## Authenticate with Kaggle

Authenticate your Kaggle account with the command-line interface:

```sh
uv run --locked kaggle auth login
```

## Download and extract the release

Download the competition archive into the repository root, then extract it to
`kaggle_public/`:

```sh
uv run --locked kaggle competitions download \
  2026-ieee-big-data-traffic-flow-bench \
  -p .
unzip 2026-ieee-big-data-traffic-flow-bench.zip -d kaggle_public
```

Pass the extracted directory as the `--data` path when preparing a local
benchmark or running predictions.

The [public release package reference](RELEASE_PACKAGE_REFERENCE.md) describes
the directory layout, file columns, and row identifiers. The organizer's
[release data guide](../official_competition_repo/docs/DATA.md) explains the
published data policy. For the information available to each task, see
[what a solution receives](COMPETITION_AND_THEORY.md#what-a-solution-receives).
The [Trafficbench guide](TRAFFICBENCH.md) explains how to prepare local cases
and run solution functions against the extracted release.
