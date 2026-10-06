# Experiment workflow

Use one branch per experiment. Name branches
`cycle-<number>/CAL-<number>/<experiment-name>`, for example
`cycle-1/CAL-2/lightgbm-baseline`. Use lowercase letters, numbers, and hyphens
in the experiment name.

Each experiment lives on its own branch, so experiment work is generally not
merged into `main`.

Keep executable code in `.py` files. Do not commit Jupyter notebooks. Put
disposable files in a `temp/` directory at any depth. These directories and
notebooks are ignored by Git. Continuous integration checks branch names and
committed file locations.

For each prepared benchmark, record the inclusive `history_start_date`,
exclusive `history_end_date`, inclusive `prediction_start_date`, and exclusive
`prediction_end_date`, along with the selected panels. Reuse those values when
comparing experiments so each solution receives the same inputs. Official
train, validation, and private boundaries come from the
[release calendar](RELEASE_PACKAGE_REFERENCE.md#panels-and-date-splits).

Trafficbench records every run, including failures, under `runs/`. See
[experiment records](EXPERIMENTS.md) for the saved fields and comparison
commands. Save persistent research outputs under `results/` on the experiment
branch. See the [results guide](../results/README.md).
