# Experiment records

Trafficbench saves every local run under `runs/`, including failed runs. A run
record includes its settings, source files, installed package versions,
predictions, available metrics, and a readable report. A failed run may have
partial metrics from cases that completed before the failure.

Use `bench summary` to write `runs/summary.csv`. Use `bench compare` to compare
runs with the same task and benchmark conditions. The
[Trafficbench guide](TRAFFICBENCH.md#read-the-local-metrics) explains the
metrics and which ones are local diagnostics rather than answer-based scores.

## Competition validation scores

Record official competition validation scores in the shared
[validation scores spreadsheet](https://docs.google.com/spreadsheets/d/1SLmxHqxA-ChrNl3DpUFe4O4uv6fOZSZ6ZFqMfxYNe_8/edit?gid=0#gid=0).
Follow the spreadsheet's existing columns and conventions. Keep each entry
associated with the experiment or source revision that produced it when the
sheet's fields allow.

These evaluator scores use the competition's hidden validation answers.
Trafficbench metrics instead come from local runs, prepared train cases, or
released auxiliary data. Their names and values are not interchangeable with
the competition leaderboard score. The
[competition guide](COMPETITION_AND_THEORY.md#competition-score-and-leaderboards)
describes the official score, and the
[Trafficbench guide](TRAFFICBENCH.md#prepare-and-score-local-train-cases)
describes local score sources and limits.

Use Actions artifacts for reports, metrics, and source snapshots from hosted
runs. Store complete local run directories, which include predictions, in
team storage. Preserve selected results before artifact retention expires.
See the [hosted benchmark guide](HOSTED_BENCHMARK_RUNS.md) for artifact
contents and retention, and the [results guide](../results/README.md) for
outputs that should stay with an experiment branch.
