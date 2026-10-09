# Task 2 follow-up experiments — 9 October 2026

The pre-existing workspace changes were committed as `325ae89` before these
experiments. The changes described here are uncommitted, as requested.

## Official validation results

Each upload contains the complete competition template. Task 1 and Task 4 rows
are zero in both uploads, so their public composite-score difference comes only
from Task 2. Dividing by Task 2's `0.30` weight gives an approximate queue IoU;
Kaggle rounds the composite score to five decimal places.

| Candidate | Kaggle ref | Public composite | Approx. Task 2 score |
| --- | ---: | ---: | ---: |
| Prior queue boosting candidate | 56917601 | 0.13514 | 0.45047 |
| Selector-aware, onset at T+30 | 57007109 | 0.21532 | 0.71773 |
| Selector-aware, onset at any horizon | 57007717 | 0.21119 | 0.70397 |

The T+30 candidate is a substantial official improvement over the prior
submission: `+0.08018` composite, or approximately `+0.26726` Task 2 score.
This validates the selector-aware candidate as a whole; it does not by itself
show that the T+30 restriction caused the gain.

The all-horizons upload is the controlled ablation. It uses the same selector-
matched training data, features, models, and other submission rows as the
T+30 candidate. Its onset decoder permits predictions at all six forecast
steps, and onset threshold calibration is repeated for that decoder. Thus its
official score directly tests whether restricting new onsets to T+30 helps.
It scored `0.00413` composite lower, about `0.01377` Task 2 IoU. T+30 is the
better choice on public validation, although the difference is modest.

Both uploads predict all 80 validation and all 80 private windows. Only public
validation scores are visible. The all-horizons run and upload were produced
with:

```sh
uv run --locked bench run queue_boosting_event_aware_all_steps --task queue \
  --data kaggle_public --history-start-date 2030-06-01 \
  --history-end-date 2031-03-01 --target-range both \
  --runs runs/queue_t30_ablation_all_steps
```

Kaggle accepted and graded ref `57007717`; its public score is `0.21119`. The
generated upload passed `bench validate` with all 6,980,503 rows.

## Older causal context

The experimental `queue_boosting_context` variant adds summaries of the three
hours before the normal 60-minute history. At prediction time it filters the
selected split's released mainline and ramp observations to timestamps before
the forecast origin. This is rule-compliant causal context.

On one small D12_I5_N December proxy benchmark (10 windows: five onset and five
ongoing), the three-hour context scored `0.57654` versus `0.58919` for the
same event-aware model without that context. Persistence scored `0.30953`.
This single-panel, small, speed-derived proxy fold is weak evidence, but it
does not support promoting the extra context yet. It should not outweigh the
official validation score or be treated as a reliable leaderboard estimate.

## Assessment

The best-supported direction is the selector-aware event-aware model with
onset restricted to T+30. It scores about `0.71773` Task 2 IoU versus `0.70397`
for the unrestricted onset decoder and `0.45047` for the earlier queue
boosting submission. The additional three-hour context has a small negative
proxy result. Keep T+30 as the current Task 2 candidate; do not add the broad
context features based on this proxy result alone. Treat the T+30 edge over
all-horizons as real but modest, and keep public validation monitoring for
later changes.
