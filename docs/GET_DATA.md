# Download the data

Join the
[Kaggle competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
before downloading its release. You can download the files from the
[Kaggle Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
or use Kaggle's command-line interface.

Install the repository environment from the project root. It includes the
Kaggle CLI:

```sh
uv python install 3.12
uv sync --locked
```

Sign in to Kaggle from the CLI:

```sh
uv run kaggle auth login
```

Download the archive, then unpack it into `kaggle_public/` at the project root:

```sh
uv run kaggle competitions download \
  -c 2026-ieee-big-data-traffic-flow-bench \
  -p .
unzip 2026-ieee-big-data-traffic-flow-bench.zip -d kaggle_public
```

Use the extracted `kaggle_public/` directory as the `--data` path for Trafficbench.

For the downloaded directory structure, table columns, and row identifiers, see the
[public release package reference](RELEASE_PACKAGE_REFERENCE.md). The organizers'
[data guide](../official_competition_repo/docs/DATA.md) describes the release.
For the tables passed to solution functions, see
[what a solution receives](COMPETITION_AND_THEORY.md#what-a-solution-receives).
The [Trafficbench guide](TRAFFICBENCH.md) explains how the runner loads them.
