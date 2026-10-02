# Download the data

Join the [Kaggle competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview) as new team or join an existing one.

Either download the release files from the [Kaggle Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data) or use the Kaggle CLI:

Install the repository's development tools with `uv sync --locked`. This installs the Kaggle CLI. Authenticate in a browser with:

```bash
uv run kaggle auth login
```

Download and unpack the release into a `kaggle_public/` directory:

```bash
uv run kaggle competitions download \
  -c 2026-ieee-big-data-traffic-flow-bench \
  -p .
unzip 2026-ieee-big-data-traffic-flow-bench.zip -d kaggle_public
```

For information about layout and files, see the [repo's own documentation](./competition_and_theory.md) or the [official data guide](../official_competition_repo/docs/DATA.md).
