# Queue forecasting with event-aware gradient boosting

`queue_boosting_event_aware` predicts Task 2 queues with six general LightGBM horizon
classifiers and a separate LightGBM onset classifier. Each directional panel is
fitted separately. `queue_true` predictions for onset windows are restricted to T+30; ongoing windows
use the six horizon forecasts.

The features describe the preceding 60 minutes: speed relative to free-flow
speed, flow relative to capacity, occupancy, speed changes, queue duration,
measurement availability, attached ramp flows, and upstream/downstream traffic.
Link properties and time of day provide location and calendar context. Missing
or low-coverage measurements remain missing; observed zero flows remain zero.

Training uses published masked train observations with known historical Task 1
speed and flow answers restored at their matching detector rows. Queue training
labels apply the competition's speed threshold to these historical measurements.
They are proxies for the withheld official queue labels.

The final seven days select tree counts using binary log loss and separate
onset/ongoing thresholds using mean window IoU. Calibration requires at least
70% history and target coverage and a positive future queue. The onset head trains
only on historical origins with no established queue, no queue in the first five
forecast steps, and a queue at T+30. This matches the public competition
selector's onset timing. These rare origins are included before regular origins
are sampled. The ongoing condition follows the released contract: at least one
link has queued measurements at two or more history timestamps. Models refit on
the full permitted history. Prediction uses each window's published condition
and its own `[T-60 minutes, T)` observations.

Every official window has at least one queued target cell. The decoder therefore
falls back to the highest eligible queue probability when thresholding would
leave a window empty. For onset windows it considers only the T+30 step. This
uses a published property of the scored windows, not future input data.

## Run an experiment

Prepare the December 2030 holdout on D12_I5_N and run the forecaster:

```sh
uv sync --locked
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 --history-end-date 2030-12-01 \
  --prediction-start-date 2030-12-01 --prediction-end-date 2031-01-01 \
  --profile quick --panel D12_I5_N --output data/benchmarks/queue-december
uv run --locked bench experiment queue_boosting_event_aware --task queue \
  --benchmark data/benchmarks/queue-december
```

The solution predicts every supplied target row. Evaluation window selection
is controlled by `bench prepare`, not by this solution. The default
`config/benchmark_scheme.yaml` selects five onset and five ongoing windows per panel;
`full` changes the panel selection, not this window count. Each prepared
benchmark records its actual windows and warnings in `manifest.json`.

For the January fold, use history dates `2030-12-04` through `2031-01-01` and
prediction dates `2031-01-01` through `2031-02-01`. Date intervals are half-open.
Use a fresh benchmark directory for each preparation.

## Parameters

Pass parameters as JSON through `bench experiment --params`. Unknown parameters
and invalid ranges are rejected.

| Parameter | Default | Behavior |
| --- | ---: | --- |
| `iterations` | 250 | Maximum tree count before historical early stopping. |
| `learning_rate` | 0.05 | Contribution of each boosting round. |
| `num_leaves` | 31 | Maximum leaves per tree. |
| `min_data_in_leaf` | 100 | Minimum training rows per leaf. |
| `positive_weight` | 2.0 | Queue-class weight in the training loss. |
| `feature_fraction` | 0.9 | Fraction of features sampled for each tree. |
| `lambda_l2` | 1.0 | L2 regularization on leaf outputs. |
| `onset_num_leaves` | `null` | Separate onset leaf count; inherits `num_leaves` when omitted. |
| `onset_min_data_in_leaf` | `null` | Separate onset leaf size; inherits the general setting when omitted. |
| `onset_positive_weight` | `null` | Separate onset positive-class weight. |
| `onset_iterations` | `null` | Separate onset maximum tree budget. |
| `onset_history_dropout` | 0.0 | Missing-cell fraction for an additional onset training copy; validation rows are excluded from augmentation training. |
| `onset_catboost_weight` | 0.0 | CatBoost weight in the onset blend; zero uses LightGBM alone. |
| `onset_catboost_depth` | 6 | Depth of CatBoost onset trees. |
| `ongoing_event_only` | `false` | Restrict general horizon fits to eligible ongoing event windows. |
| `training_stride` | 3 | Spacing of training origins in five-minute steps. |
| `max_training_rows` | 250000 | Uniform sample cap per horizon, including the final refit. |
| `max_training_origins` | 3000 | Earlier fit-origin cap before feature construction; selector-matched onset events are reserved before regular origins are sampled. |
| `calibration_days` | 7 | Later historical days reserved before the final refit. |
| `early_stopping_rounds` | 30 | Historical validation patience. |
| `threads` | 4 | LightGBM CPU threads per panel. |
| `threshold` | `null` | Optional common threshold overriding calibration. |
| `onset_threshold` | `null` | Optional onset override, taking priority over `threshold`. |
| `ongoing_threshold` | `null` | Optional ongoing override, taking priority over `threshold`. |

The runner seed controls training-row sampling and LightGBM randomness. A
condition with fewer than two usable historical calibration windows falls back
to threshold `0.35`. Without usable historical labels, the solution uses the
reference persistence predictor. Training never uses the requested prediction
period for calibration, fitting, or feature generation.

Inspect binary predictions, per-window metrics, source snapshots, and run
metadata under the resulting `runs/` directory. Local proxy IoU is not an
official Task 2 or composite competition score.

## Tune parameters on selector-matched windows

The CAL-31 search uses `config/queue_selector_matched.yaml`. Its onset windows
have no queued target at T+5 through T+25 and at least one at T+30. The default
benchmark scheme retains arbitrary onset timing for comparisons with existing
experiments. Both schemes use speed-derived proxy labels.

Prepare two separate months using the selector-matched scheme:

```sh
# December selects parameters; January provides a later assessment.
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 --history-end-date 2030-12-01 \
  --prediction-end-date 2031-01-01 --scheme config/queue_selector_matched.yaml \
  --panel D12_I5_N --panel D7_I10_E \
  --output data/benchmarks/cal31-december
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-12-04 --history-end-date 2031-01-01 \
  --prediction-end-date 2031-02-01 --scheme config/queue_selector_matched.yaml \
  --panel D12_I5_N --panel D7_I10_E \
  --output data/benchmarks/cal31-january
```

Run the bounded search from the repository root:

```sh
# Include defaults, sample five alternatives, and freeze the winner before January.
uv run --locked python -m scripts.tune_queue \
  --tuning data/benchmarks/cal31-december \
  --holdout data/benchmarks/cal31-january \
  --output runs/cal31-hpo --trials 6 --seed 31 --threads 4
```

The output directory must be new. `search.json` records completed trials,
benchmark identifiers, scores, durations, and run locations. `selected_params.json`
is written before the January runs and can be passed through `bench experiment
--params`. Each run retains the normal trafficbench source, predictions, and
per-window metrics. Trials vary leaf count, minimum rows per leaf, class weight,
feature sampling, and L2 regularization. Historical threshold calibration and
early stopping remain inside each fit; the prediction month never enters fitting.
The search does not change the solution defaults or submit to Kaggle.

The [CAL-31 result](../../results/cal31/README.md) records the six-trial search,
selected parameters, and separate January assessment.

## Optuna specialist experiments

`scripts.optimize_queue` uses Optuna TPE to search the onset specialist settings,
missing-history augmentation, CatBoost blend, and ongoing event sampling. Its
first four proposals compare the reference, 5% history dropout, a dropout-trained
LightGBM/CatBoost blend, and ongoing-only training. Later proposals use TPE.
Median pruning can stop a trial after a common prefix of panels. The objective
is the runner's balanced window IoU. Parallel scheduling is controlled by
`--jobs`; each model uses `--threads` CPU threads, including prediction calls.

Run the study on prepared selector-matched cases:

```sh
# Keep artifacts outside the source checkout on the university server.
uv run --locked python -m scripts.optimize_queue \
  --benchmark /data/s4501020/trafficflow/cal31/benchmarks/december \
  --output /data/s4501020/trafficflow/cal31/optuna \
  --trials 32 --jobs 16 --threads 8
```

SQLite study storage preserves completed and pruned trials. The output records
benchmark and source identities and rejects reuse with changed inputs. Parallel
scheduling can change TPE suggestions even with the same seed. The selected
parameter JSON is frozen before a separate later-month assessment. These
specialist methods remain experimental until that assessment and a competition
submission establish their behavior.
