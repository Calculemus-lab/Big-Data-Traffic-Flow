# Experiment workflow

Use one branch for each experiment. Name the branch
`cycle-<number>/CAL-<number>/<experiment-name>`, for example
`cycle-1/CAL-2/lightgbm-baseline`. The CAL number identifies the assigned work
item in the team's
[Linear project](https://linear.app/calculemus/project/big-data-traffic-flow-competition-21181673811a/issues).
Use lowercase letters, numbers, and hyphens in the experiment name.

Experiment branches keep solution code and results tied to the work item. The
team generally keeps experiment branches separate instead of merging each
one into `main`.

Keep executable code in `.py` files. Do not commit Jupyter notebooks. Use a
`temp/` directory at any depth for disposable files; CI ignores those
directories and checks that notebooks and temporary files are not committed.

## Keep benchmark comparisons reproducible

A **Prepared Benchmark** is a repeatable local case with one history period,
one prediction period, and a selected set of panels. Record its date bounds and
panels when comparing solutions. The history start and prediction start are
included; the history end and prediction end are excluded. Keep these values
the same when comparing experiments. Official train, validation, and private
period boundaries are listed in the
[release calendar](RELEASE_PACKAGE_REFERENCE.md#panels-and-date-splits).

Trafficbench saves a record for every local run, including failed runs, under
`runs/`. The [experiment records guide](EXPERIMENTS.md) explains how to review
and compare these records. Save research outputs that should remain with the
experiment branch under `results/`; see the [results guide](../results/README.md).
