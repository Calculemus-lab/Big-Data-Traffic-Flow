# Traffic Flow Bench team experiments

Write an idea, run it, compare the result. The runner handles fixed folds,
scoring, prediction checks, baseline comparisons and saving results.

## First time

Use Python 3.12, then run these commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Get the competition download from Kaggle. You can use the zip directly:

```sh
bench prepare --data 2026-ieee-big-data-traffic-flow-bench.zip
```

This prepares the shared quick benchmark once. Reuse it for every experiment.
You can also point `--data` at an extracted `kaggle_public` directory. Your teammate
can give you an already prepared benchmark directory instead; place it at
`data/benchmarks/quick` without changing its contents.

## Test an idea

```sh
bench new my_idea
bench run my_idea
```

Edit `solutions/my_idea/model.py`. `bench new` creates a package with
`__init__.py` exporting `state(ctx)` from `model.py`. The function receives the
data and returns predictions. Add as many helper modules or subpackages as your
approach needs; import them from `model.py`. The example blends a historical
profile with visible neighbours. Nothing needs registering.

For the other tasks:

```sh
bench new queue_idea --task queue
bench run queue_idea --task queue
bench new demand_idea --task odme
bench run demand_idea --task odme
```

The same package can implement several tasks: define `state(ctx)`, `queue(ctx)`
or `odme(ctx)` in its modules and export each function from `__init__.py`.
Existing single-file `solutions/name.py` modules also remain supported.

Change parameters without editing the code:

```sh
bench run my_idea --params '{"blend": 0.75}'
```

The terminal prints your scores. Each run saves a readable report, detailed
metrics, predictions, parameters, environment information and a source snapshot
under `runs/`. Failed runs are saved too. A successful idea is automatically
compared with the baseline on the same data. Lower error is better; higher score
or IoU is better. Prior movement has no universally better direction.

## What your function gets

- `ctx.train`: observations before the evaluation period; use these to fit.
- `ctx.observations`: visible masked evaluation data for state, or just one
  permitted history window for queue.
- `ctx.targets`: the exact rows to predict, with no answers. Copy this table and
  add `speed_kmh` and `flow_vph`, `queue_pred`, or `path_flow`.
- `ctx.network`: network tables, for example `ctx.network["links"]`.
- `ctx.ramps`: evaluation ramp observations for state. Missing evidence is not zero.
- `ctx.counts` and `ctx.prior`: the current OD problem's inputs.
- `ctx.params`: your command-line parameter dictionary.
- `ctx.cache`: reuse a fitted model across queue windows of the same panel/fold.
- `ctx.seed`: deterministic seed; pass it to your model library too.

Times are UTC and identifiers are strings. Return the target keys unchanged.
Queue outputs must be binary; path flows must be finite and nonnegative.
Do not read benchmark truth files or raw future data from your solution. The
interface keeps answers separate to prevent mistakes; it is not a security sandbox.
For queue training, construct examples wholly inside `ctx.train`, and fit once
using `ctx.cache`. Do not carry fitted state between folds through module globals.

## Larger checks

```sh
bench prepare --data 2026-ieee-big-data-traffic-flow-bench.zip --profile full
bench run my_idea --profile full
```

Quick uses one panel, 28 training days and the first seven days of December.
Full uses every panel, June–November to predict December, then June–December to
predict January. February is reserved for occasional final candidate checks:
use `--profile holdout` for both preparation and running. Preparation needs enough
memory for a panel's training table; the full bundle also needs several GB of disk.

Only compare runs on the same benchmark and task:

```sh
bench compare runs/FIRST_RUN runs/SECOND_RUN
bench summary
```

The summary CSV is generated automatically. No shared spreadsheet needs manual
editing. Keep small source changes on your experiment branch; share result bundles
through Actions artifacts or team storage. Keep selected final results permanently;
Actions artifacts expire. Add public leaderboard results to your experiment notes.

## What the scores mean

State reconstruction is scored against held-out training answers, using the
competition formula, exact template targets, per-lane flow errors, and equal
regime/direction/family weighting. Extra blanks around queue horizons are not targets.

Queue IoU is a proxy: labels come from eligible released speeds, not the hidden
underlying state. The fixed sample balances onset and ongoing conditions and
excludes the two I405 panels that have no competition queue task. Unknown labels
are excluded, never treated as an empty road. The downloaded release uses history
`[T-60, T)` and six future steps from `T+5` to `T+30`; the proxy follows that layout.
Onset/ongoing follows the downloaded contract's established-queue rule: at least
two queued observations on one link in history means ongoing. Sampling guarantees
an observed queue in the horizon and six-hour spacing within each condition.

Physics diagnostics accompany state runs. They check fundamental-diagram error,
low flows and negative values. They are not the official physics score; reliable
conservation evaluation needs withheld boundary flows.

OD runs report released training count fit and prior movement, plus recovery on
three fixed synthetic demand problems using the released network. Synthetic scores
are proxies. These OD cases are independent of the chronological folds. There is
no trustworthy local overall competition score. Use leaderboard feedback to check
whether improvements transfer.

## Build a submission

Generate predictions for both competition splits with your selected models. Save
one CSV or Parquet per task using the same natural keys as `ctx.targets`, including
OD origin/destination zones. Then assemble and validate them:

```sh
bench assemble --state state.csv --queue queue.csv --odme odme.csv \
  --key kaggle_public/submission_key.csv \
  --template kaggle_public/sample_submission.csv --output submission.csv
bench validate submission.csv --template kaggle_public/sample_submission.csv
```

Assembly rejects missing or extra predictions, duplicate keys, invalid values and
incorrect OD zones. It preserves template order and zeroes unused output fields.
It uses disk-backed joins so the final key need not fit in memory. It never uploads
to Kaggle. Task 3 uses the state predictions and has no separate file.

## GitHub

Every PR runs correctness tests and all three baselines on a tiny synthetic release.
This checks whether the plumbing works; these smoke scores say nothing about an
idea's competition quality. Reports are available in the job summary and artifacts.
A lower model score does not fail a PR.

For real experiments, use **Actions → Experiment benchmark → Run workflow**.
Choose the branch, solution name, task, profile and optional JSON parameters.
Both workflows use standard GitHub-hosted Ubuntu machines on GitHub Free. Nobody
needs to keep a laptop or server running. The owner uploads prepared benchmark
packages once to a private data repository and configures read access; see
[CI setup](CI_SETUP.md). CI downloads one panel per job and combines the results
into one report.

CI keeps compact reports, metrics and source snapshots. It does not publish real
predictions or competition data in public Actions artifacts. Run locally to retain
predictions for blending. Intermediate artifacts last one day; combined reports
last seven days. There is no automatic upload to Kaggle.
