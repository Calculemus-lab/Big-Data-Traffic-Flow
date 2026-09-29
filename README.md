# Big-Data-Traffic-Flow

Shared repository for experiments in the Kaggle **2026 IEEE Big Data Traffic Flow Bench** competition.

Competition: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview

## Goal

Maximize score over the next 3-4 weeks, prioritizing:

1. Local cross-validation (CV) score
2. Public leaderboard score

## Working model

### Python environment

Use `uv sync --locked` after checkout and `uv run` to execute code. See the
brief [dependency guide](docs/dependencies.md) before changing packages.

### General coding practices

- **1 branch ↔ 1 experiment**
- Use shared repo conventions in this document for reproducible experiments.
- Every experiment must include enough artifacts/metadata to be rerun and compared.

### Shared CV scheme (mandatory)

All experiments must use the same fold configuration in:

- `/home/runner/work/Big-Data-Traffic-Flow/Big-Data-Traffic-Flow/config/cv_scheme.yaml`

### Experiment logging (mandatory)

Log **every** experiment, including failures, in:

- `/home/runner/work/Big-Data-Traffic-Flow/Big-Data-Traffic-Flow/experiments/experiment_log.csv`

Use the template columns exactly to keep the log sortable and comparable.

### Required experiment outputs

Each experiment branch should save:

- out-of-fold predictions (`oof_path`)
- test predictions (`test_path`)
- configuration/notes needed to reproduce

## Team guardrails

- Everyone uses the same folds.
- Every experiment gets logged (including failures).
- Keep changes branch-scoped to one experiment.
