# Official Kaggle validation result

Public scores evaluate March 2031; private April scores remain withheld. Task 2
candidate uploads from 9 October use zero outputs for Tasks 1 and 4, so their
composite scores isolate the Task 2 contribution.

The subsequent [baseline, history comparison, and repair assessment](assessment.md)
records persistence at `0.30170` queue IoU and the corrected 28-day model at
`0.43440`. Sixteen malformed local Task 1 templates have been restored from
official copies with backups. A zero-filled comparison export was discarded.
The [selector-aware candidate](event_aware/README.md) and its all-horizons
ablation both have official public scores.

| Upload | Kaggle reference | Public composite score |
| --- | ---: | ---: |
| Queue boosting, other task outputs zero | 56917601 | 0.13514 |
| Selector-aware, onset at T+30 | 57007109 | 0.21532 |
| CAL-31 shared-horizon tuned probe | 57561677 | 0.21169 |
| Selector-aware, onset at all horizons | 57007717 | 0.21119 |
| All-zero control | 56917634 | 0.00000 |

Because the other task outputs are zero, the candidate scores correspond to
approximately `0.71773` and `0.70397` Task 2 IoU, respectively. The T+30
candidate leads the controlled all-horizons ablation by `0.00413` composite,
about `0.01377` Task 2 IoU. See the [follow-up experiment report](experiments_2026-10-09.md)
for the comparison details.

The CAL-31 shared-horizon probe scored below the T+30 candidate by `0.00363`
composite, or approximately `0.01210` Task 2 IoU. It narrowly exceeds the
all-horizons ablation by `0.00050` composite. Its score and training settings are
recorded in the [CAL-31 probe report](../../cal31/university-followup.md#official-kaggle-probe).

Only the queue predictions differ between the two uploads. Every official queue
window contains a positive future queue cell, so an all-clear queue control has
zero queue IoU. Using the official Task 2 weight:

```text
S_queue = (0.13514 - 0.00000) / 0.30
        ≈ 0.45047
```

This is approximate because Kaggle displays composite scores to five decimal
places. In this experiment the other tasks contribute zero, so a composite score
of `0.30` would correspond to perfect Task 2 predictions. When other tasks have
nonzero scores, the composite score cannot be divided by `0.30` directly.

The official Task 2 score is substantially below the approximately `0.69` local
proxy scores. Those local runs used 28 days of training history and different
prediction months. This upload uses a nine-month historical pool and official
hidden labels, so the score difference does not isolate a single cause.

## Predictions and submission format

The forecaster fits on the permitted historical interval `2030-06-01` through
`2031-03-01`, exclusive of the end date. It predicts all 80 official validation
windows and all 80 private windows: 87,000 queue rows per split, 174,000 total.
No official prediction windows are sampled out.

Earlier fit origins are uniformly capped at 3,000 before constructing features;
the final historical calibration week remains intact. Each horizon retains the
existing 250,000-row training cap. This bounds feature memory for the full
nine-month history. On the previously measured December D12_I5_N benchmark,
the memory and indexing changes reproduce the exact `0.6711529808123412` score.

Reproduce the Task 2 predictions with:

```sh
uv run --locked bench run queue_boosting --task queue --data kaggle_public \
  --history-start-date 2030-06-01 --history-end-date 2031-03-01 \
  --target-range both --runs runs/queue_boosting_validation
```

The upload retains all 6,980,503 IDs from the full combined template. Task 1
speed/flow and Task 4 path-flow outputs are zero. Task 3 is scored from Task 1
and has no separate output rows. `bench assemble` validated row identity,
completeness, numeric values, binary queue answers, and unused output columns
before publishing `temp/queue_validation_submission/task2_only.csv`.

The [raw Kaggle results](kaggle_submissions.json),
[prediction run record](prediction_run.json),
[submission metadata and hashes](submission_metadata.json), and
[queue predictions](queue_predictions.csv) preserve the result. `source/`,
`pyproject.toml`, and `uv.lock` preserve the forecaster and environment. Private
predictions are included for upload completeness, but no private score is known.

## Local Task 1 input issue

Seven panels' local Task 1 templates contain malformed mask codes `R10`, `R20`,
or `R30`, where the master key expects `R1`, `R2`, or `R3`. For example, a source
row ends in `R20,0` under a header requiring separate mask, speed, and flow
fields. This caused 4,802,643 Task 1 natural keys to miss the master key in the
initial assembly. Diagnostic normalization of those codes eliminates every
mismatch.

The Task 2 predictions have zero missing or extra keys. For this Task 2-only
upload, zero-valued state and ODME rows were generated directly from the master
key. The malformed templates have since been restored from official copies,
with zero remaining missing or extra state keys. The earlier diagnostic is in
[input consistency checks](input_consistency.json).

The prior baseline suite passed 51 tests. The later event-aware change passed
its 10 focused tests; Ruff lint and format checks and Pyright pass. Its official
predictions are saved under [the event-aware validation record](event_aware/README.md),
but Kaggle's daily submission limit prevented scoring that candidate.
