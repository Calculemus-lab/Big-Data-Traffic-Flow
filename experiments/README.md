# Experiments

This folder holds the shared experiment log and experiment-level artifacts references.

## Logging rules

- Add one row per experiment run to `experiment_log.csv`.
- Log all runs, including failed experiments and dead ends.
- Keep branch naming aligned with the `1 branch ↔ 1 experiment` rule.

## Required fields

- `timestamp`: ISO datetime of the run
- `idea`: short experiment idea
- `theme`: assigned theme bucket
- `branch`: experiment branch name
- `cv_scheme`: CV config version (for example `shared_cv_v1`)
- `cv_score`: local CV metric
- `leaderboard_score`: Kaggle public LB score (when submitted)
- `status`: `kill`, `continue`, `blend_candidate`, or `baseline`
- `oof_path`: path/location of saved OOF predictions
- `test_path`: path/location of saved test predictions
- `notes`: short reproducibility notes
