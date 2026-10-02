# Experiment records

Runs are recorded automatically under `runs/`, including failures. Each run has
parameters, source/environment metadata, predictions, metrics and a readable report.
Use `bench summary` to generate `runs/summary.csv`; use `bench compare` for runs on
the same benchmark and task. Save public leaderboard feedback in experiment notes.

The old `experiment_log.csv` is retained as historical data. New experiments do
not need to edit it. Share run directories through Actions artifacts or team storage
and preserve selected final results before artifact retention expires.
