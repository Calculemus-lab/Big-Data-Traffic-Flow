# Experiment records

Trafficbench records every run under `runs/`, including failed runs. A run
records its parameters, source files, installed package versions, predictions,
metrics, and a readable report. Use `bench summary` to generate
`runs/summary.csv`. Use `bench compare` to compare runs on the same benchmark
and task.

## Competition validation scores

Record competition validation scores in the shared
[validation scores spreadsheet](https://docs.google.com/spreadsheets/d/1SLmxHqxA-ChrNl3DpUFe4O4uv6fOZSZ6ZFqMfxYNe_8/edit?gid=0#gid=0).
Use the spreadsheet's existing columns and conventions when adding a result.
These scores are evaluator results and should be kept distinct from local
Trafficbench scores, which measure performance on prepared local cases.

Keep the experiment's run directory or source revision associated with its
spreadsheet entry so the result can be traced to the solution that produced it.

The old [`experiment_log.csv`](../experiments/experiment_log.csv) is retained as
historical data. New experiments do not need to edit it. Share run directories
through Actions artifacts or team storage and preserve selected final results
before artifact retention expires.
