# Big-Data-Traffic-Flow

Shared repository for experiments in the Kaggle **2026 IEEE Big Data Traffic Flow Bench** competition.

Competition: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview

## Goal

Maximize score over the next 3-4 weeks, prioritizing:

1. Local cross-validation (CV) score
2. Public leaderboard score

## Working model

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

## Phases

### Phase 1: Setup & Baseline

Main goals:

- Review forum threads and top public notebooks to understand strong approaches.
- Submit a simple baseline to validate end-to-end pipeline.
- Build common CV folds + shared harness expectations.
- Rank 6-10 candidate themes by expected gain, cost, risk, diversity.
- Assign 2-3 themes to the team.

Deliverables:

- baseline score
- agreed CV scheme
- chosen themes
- working shared repo

### Phase 2: Experimentation (bulk of time)

Goals:

- Theme owners run short, frequent experiments and log all results.
- 15-20 minute sync every 3-4 days to review log and share findings.
- At checkpoints classify each theme as:
  - `kill` (no signal)
  - `continue` (CV improving)
  - `blend_candidate` (decent and diverse)
- Timebox rabbit holes and drop low-signal ideas quickly.
- Operate in ~4 day cycles.

### Phase 3: Convergence

- Stop new exploratory directions.
- Take best 2-4 diverse models.
- Lightly tune and build blend/stack from saved OOF predictions.

### Phase 4: Freeze & Submit

- No new experiments.
- Reproduce best results from pipeline.
- Choose final submissions deliberately.
- Keep ~2 days of deadline buffer.

## Team guardrails

- Everyone uses the same folds.
- Every experiment gets logged (including failures).
- Keep changes branch-scoped to one experiment.
