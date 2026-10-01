# Submit predictions

Make sure you have joined the [Kaggle competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview). Follow [Build the complete submission](competition_and_theory.md#building-the-complete-submission) to create the merged `final_submission.csv` in the required format. 

Either upload the file manually on the [Kaggle submission page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/submit) or use the Kaggle CLI:

Make sure the Kaggle CLI is. It is part of the repository's development tools which can be installed with `uv sync --locked`. Authenticate in a browser with:

```bash
uv run kaggle auth login
```

Upload only the merged CSV from the repository root:

```bash
uv run kaggle competitions submit \
  -c 2026-ieee-big-data-traffic-flow-bench \
  -f final_submission.csv \
  -m "Baseline submission"
```

See the [Kaggle CLI competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) for options such as checking submission status.
