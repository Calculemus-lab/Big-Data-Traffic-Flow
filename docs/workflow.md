# Experiment workflow

Use one branch per experiment. Name branches
`cycle-<number>/CAL-<number>/<experiment-name>`, for example
`cycle-1/CAL-2/lightgbm-baseline`. Use lowercase letters, numbers, and hyphens
in the experiment name.

Each experiment lives on its own branch, so experiment work is generally not
merged into `main`.

Keep executable code in `.py` files; do not commit Jupyter notebooks. Put
disposable files in a `temp/` directory at any depth. These directories and
notebooks are ignored by Git. CI checks branch names and committed file
locations.

All experiments must use the folds in [`config/cv_scheme.yaml`](../config/cv_scheme.yaml).
Log every run, including failures, in
[`experiments/experiment_log.csv`](../experiments/experiment_log.csv). See the
[experiment logging guide](../experiments/README.md) for its fields. Save
persistent outputs under `results/` on the experiment branch; see the
[results guide](../results/README.md).
