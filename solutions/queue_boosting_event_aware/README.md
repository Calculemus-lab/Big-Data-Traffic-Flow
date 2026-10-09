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
`config/cv_scheme.yaml` selects five onset and five ongoing windows per panel;
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
