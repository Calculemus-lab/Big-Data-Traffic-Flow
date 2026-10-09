# Task 2 queue forecasting results

The previously submitted model scored approximately `0.45047` Task 2 IoU on the
official March validation windows. It remains in `solutions/queue_boosting`;
its score and upload are recorded in the [validation report](validation/README.md).
The selector-aware experiment is isolated in
`solutions/queue_boosting_event_aware`. Its official predictions were prepared,
but today's Kaggle submission limit prevented an official score.

Both models use causal traffic, network, ramp, link, and clock features.
Historical Task 1 answers are restored before deriving training labels;
prediction-month answers are never used for fitting or calibration. The
experimental model adds a selector-matched onset head and predicts those
windows only at T+30, with a positive-window fallback.

## Competition code findings

The highest-scoring public notebook inspected during this pass reports that its
onset queue first appears at T+30 because the selector chooses the earliest
origin whose horizon includes the event. It reports a `+0.027` public score
change from predicting onset at T+30 only. That is an author's overall
leaderboard result, not a separately reported Task 2 score. The same notebook
specializes model families by condition, uses a three-seed CNN for onset, and
reports a two-link fallback if thresholding an onset window predicts nothing.
[Notebook and experiment log](https://www.kaggle.com/code/yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-85).
It links the [organizer ruling on allowed history](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068)
and the [onset timing confirmation](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350).

A second public solution with a reported `0.86517` composite uses a separate
onset LightGBM/CatBoost head, samples the earliest selector-style onset from
each event, predicts only T+30, and falls back to its best link when needed.
Its ongoing LightGBM model is trained separately. The experimental candidate
adapts its selector-matched onset idea with a lower-cost LightGBM head; the
reported composite score is not a Task 2 score.
[Notebook and source](https://www.kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517).

I checked the timing against the 40 onset windows in the released train split:
39 have no visible queued speed at T+5 through T+25 and do have one at T+30.
The remaining `D12_I5_S` training window has no visible queued speed in its
six target times. This checks released speed labels, not the hidden evaluation
truth; the organizer's selector ruling remains the basis for the decoder rule.

The public GitHub solution titled “Rank-1” uses a grouped random fold split for
threshold selection and advertises `0.76–0.85+` for Task 2 without a matching
official Task 2 result table. Its linked notebook predates the 11 September
release rebuild. I treat those numbers as unverified and did not use them to
set expected performance.
[Repository and Task 2 script](https://github.com/MAYANKSHARMA01010/2026-ieee-big-data-traffic-flow-bench).

## Evaluation protocol

All intervals are half-open. Each panel receives 28 days of historical data.

| Fold | History start | History end | Prediction start | Prediction end |
| --- | --- | --- | --- | --- |
| December | 2030-11-03 | 2030-12-01 | 2030-12-01 | 2031-01-01 |
| January | 2030-12-04 | 2031-01-01 | 2031-01-01 | 2031-02-01 |

The prepared cases use `cutoff_holdout_v1`, seed `69420`, minimum coverage `0.7`,
and 360 minutes of spacing within each queue condition. They select five onset
and five ongoing windows per panel. They do not score every possible eligible
forecast origin in the month. The all-panel run covers D7_I10_E/W, D7_I210_E/W,
D7_I405_N/S, and D12_I5_N/S: 80 windows in each month. D12_I405_N/S have no Task 2
cases. The exact origins, input hashes, and date limits are saved in each run's
`benchmark.json`.

The original local model sampled general training origins every 15 minutes.
The experimental candidate also keeps selector-matched onset origins before
filling its earlier-origin cap. Each model uses a deterministic sample capped at
250,000 eligible link rows. All requested evaluation rows are predicted.
LightGBM version is `4.7.0` and the CPU thread count is four.

## Completed comparisons

Scores are local queue proxy IoU. They are not official queue or composite
leaderboard scores.

| Evaluation | Windows | Persistence | Original horizon model | Gain |
| --- | ---: | ---: | ---: | ---: |
| December, all eight eligible panels | 80 | 0.372043 | 0.696429 | +0.324386 |
| January, all eight eligible panels | 80 | 0.348996 | 0.688555 | +0.339559 |
| Mean of the two all-panel runs | 160 | 0.360519 | 0.692492 | +0.331973 |

Every panel improves in both months. Condition-level comparisons are:

| Condition | December persistence | December boosting | January persistence | January boosting |
| --- | ---: | ---: | ---: | ---: |
| Onset | 0.109237 | 0.503909 | 0.128862 | 0.529842 |
| Ongoing | 0.634848 | 0.888949 | 0.569130 | 0.847267 |

| Panel | December persistence | December boosting | January persistence | January boosting |
| --- | ---: | ---: | ---: | ---: |
| D12_I5_N | 0.309529 | 0.671153 | 0.267910 | 0.566682 |
| D12_I5_S | 0.321519 | 0.680625 | 0.467080 | 0.719694 |
| D7_I10_E | 0.561386 | 0.709374 | 0.353761 | 0.608126 |
| D7_I10_W | 0.183221 | 0.580165 | 0.245168 | 0.569174 |
| D7_I210_E | 0.470487 | 0.905725 | 0.526662 | 0.990909 |
| D7_I210_W | 0.353108 | 0.785825 | 0.312440 | 0.735890 |
| D7_I405_N | 0.411539 | 0.677141 | 0.302534 | 0.695761 |
| D7_I405_S | 0.365553 | 0.561424 | 0.316412 | 0.622200 |

The all-panel model runs take approximately 7.1 minutes in December and 7.7
minutes in January with four CPU threads. Peak process memory is approximately
1.8 GB and 1.6 GB, respectively. The
[December report](december_all_panels/report.md) and
[January report](january_all_panels/report.md) record exact resource measurements.

Each retained local result directory contains model metrics, persistence metrics,
a run record, a prepared benchmark manifest, and a readable report. Redundant
per-run snapshots and prediction copies were removed after preserving these
aggregate results.

The [event-aware proxy report](event_aware_proxy/README.md) records a two-fold
negative result on random local onset windows and nearly unchanged ongoing
performance. The official event-aware predictions are preserved, but the
candidate could not be submitted because today's Kaggle submission limit was
already exhausted. Its local proxy scores do not justify replacing the
previously submitted model without an official comparison.

The earlier CAL-16 adapter scored approximately `0.43` on the D12_I5_N cases.
That comparison does not isolate the model architecture: its training code used
masked observations without restoring historical Task 1 answers and ran under
an earlier solution API. The matched persistence comparisons above use the
current evaluator and the same prepared targets.

## Verification

The earlier baseline suite passed 51 tests. The event-aware candidate passed
its 10 focused tests for causal inputs, decoding, selector-event sampling, and
complete reproducible output. Ruff lint and format checks, Pyright, and
repository convention checks pass.

The official queue answers are withheld. Proxy labels and local onset/ongoing
classification can differ from the organizer's underlying states. These results
support further Task 2 development but do not establish an official leaderboard
score or a competition-winning model.
