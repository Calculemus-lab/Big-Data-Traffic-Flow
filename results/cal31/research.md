# CAL-31 Task 2 optimization research

Research date: 23 October 2026. The experiment starts from commit `ead8b6b`
on `cycle-2/CAL-13/decision-tree`, with work on
`cycle-6/CAL-31/task2-optimization`.

## Starting evidence

The retained official validation record gives the event-aware model approximately
0.71773 Task 2 IoU, compared with 0.45047 for the earlier general horizon model.
These are approximate isolated queue scores derived from displayed Kaggle
composite scores; see the [validation record](../queue_boosting/validation/README.md).
The event-aware package is the optimization reference.

The previous local sampler admits onset queues at any forecast horizon, while
the event-aware model decodes onset only at T+30. A parameter search on those
windows would reward behavior inconsistent with the intended selector rule.
`config/queue_selector_matched.yaml` therefore requires no queued future cell
at T+5 through T+25 for onset windows. Positive-future, coverage, established-queue,
and spacing requirements also apply. Ongoing window eligibility is unchanged.
This reproduces one selector timing property, not the full hidden-label process.

## Online sources and decisions

The public [Task 2 LightGBM implementation](https://github.com/MAYANKSHARMA01010/2026-ieee-big-data-traffic-flow-bench/blob/main/scripts/build_task2_lightgbm.py)
uses upstream/downstream traffic, speed trends, shockwave features, and
condition-specific IoU thresholds. It selects thresholds using grouped random
folds. The existing solution already includes neighbor traffic, temporal changes,
and condition-specific calibration. CAL-31 retains chronological evaluation;
this source does not establish that its advertised leaderboard gains transfer
to our model or the current release.

[LightGBM's tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)
identifies leaf count and minimum observations per leaf as key complexity controls,
and describes feature subsampling and L2 regularization as overfitting controls.
The search varies those settings and positive-class weight. The current tree
budget, learning rate, training caps, historical early stopping, and threshold
calibration stay fixed so that the comparison remains bounded.

The [previously inspected CNN solution](https://www.kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-85),
[LightGBM/CatBoost solution](https://www.kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517),
and [organizer onset discussion](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350)
could not be reopened through the browsing tool during this pass. Their earlier
findings remain in the [Task 2 research record](../queue_boosting/README.md);
CAL-31 does not claim a fresh verification of their contents or rankings.

## Search protocol

Use the same two panels in each month: D12_I5_N and D7_I10_E. Each panel has
10 onset and 10 ongoing windows, for 40 windows per month. The benchmarks use
seed 69420, at least 70% history and target coverage, and 360-minute spacing
within each condition. Both preparations completed without warnings.

| Role | Training history | Scored targets |
| --- | --- | --- |
| Parameter selection | 2030-11-03 to 2030-12-01 | December 2030 |
| Later assessment | 2030-12-04 to 2031-01-01 | January 2031 |

All date bounds are half-open. Each model reserves its final seven historical
days for early stopping and threshold calibration, then refits on allowed
history. Neither scored month enters its own model fitting or calibration.

The six-trial search includes the unchanged reference and five combinations
sampled without replacement with search seed 31. Each fit uses four CPU threads.
The sampling space is:

| Parameter | Values |
| --- | --- |
| `num_leaves` | 15, 31, 63 |
| `min_data_in_leaf` | 30, 100, 300 |
| `positive_weight` | 1, 2, 4 |
| `feature_fraction` | 0.75, 0.9 |
| `lambda_l2` | 0, 1, 5 |

Rank candidates by the existing runner's mean window IoU, balanced over queue
conditions, panels, and freeway families. Freeze the winner and write its JSON
parameters before evaluating January. Evaluate both that winner and the unchanged
reference on January. Do not use January to select an alternative trial.

This is a small random search on two panels and proxy labels. It does not
establish an official validation improvement or optimal parameters. A positive
later-month result supports a subsequent broader comparison. The solution's
default parameters remain the official-score reference.

The [solution guide](../../solutions/queue_boosting_event_aware/README.md#tune-parameters-on-selector-matched-windows)
contains reproduction commands. Full run evidence stays under `runs/cal31-hpo/`.
