# Experiment records

Trafficbench records every run under `runs/`, including failed runs. A run
records its parameters, source files, installed package versions, predictions,
metrics, and a readable report. Use `bench summary` to generate
`runs/summary.csv`. Use `bench compare` to compare runs on the same benchmark
and task. Save public leaderboard feedback in experiment notes.

The old [`experiment_log.csv`](../experiments/experiment_log.csv) is retained as
historical data. New experiments do not need to edit it. Share run directories
through Actions artifacts or team storage and preserve selected final results
before artifact retention expires.
