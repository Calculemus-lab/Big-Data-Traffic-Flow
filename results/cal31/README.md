# CAL-31: Task 2 parameter optimization

The six-trial LightGBM search produced a candidate that improves local proxy IoU
in both the tuning month and the later assessment month. January was evaluated
after the candidate parameters were selected and saved from December results.

| Fold | Reference IoU | Selected candidate IoU | Gain |
| --- | ---: | ---: | ---: |
| December, parameter selection | 0.895866 | 0.901710 | +0.005845 |
| January, later assessment | 0.861770 | 0.870681 | +0.008911 |

Each month contains 40 windows across D12_I5_N and D7_I10_E, with ten onset
and ten ongoing windows per panel. Onsets first queue at T+30. Scores use the
runner's condition/panel/family-balanced mean window IoU. They are local
speed-derived proxy scores, not official Task 2 or composite leaderboard scores.

## Candidate settings

The [selected parameter file](selected_params.json) changes four settings in
`queue_boosting_event_aware`:

| Parameter | Reference | Candidate |
| --- | ---: | ---: |
| `num_leaves` | 31 | 63 |
| `min_data_in_leaf` | 100 | 30 |
| `positive_weight` | 2.0 | 4.0 |
| `feature_fraction` | 0.9 | 0.75 |

Other settings retain their defaults, including 250 maximum rounds, learning
rate 0.05, and L2 regularization 1.0. Thresholds and tree counts are selected
inside each fit using historical data. No prediction-month labels enter fitting.

On January, the candidate improves every panel/condition combination except
D7_I10_E onset, where both configurations score 1.0:

| Panel | Condition | Reference IoU | Candidate IoU |
| --- | --- | ---: | ---: |
| D12_I5_N | Onset | 0.916667 | 0.926667 |
| D12_I5_N | Ongoing | 0.664279 | 0.686054 |
| D7_I10_E | Onset | 1.000000 | 1.000000 |
| D7_I10_E | Ongoing | 0.866134 | 0.870003 |

December's gain comes from D12_I5_N onset; ongoing IoU decreases slightly on
both panels. January fitting and prediction take 157.8 seconds for the candidate
and 134.7 seconds for the reference with four CPU threads. The candidate improves
prediction quality on this assessment and costs about 17% more elapsed time.

The candidate is suitable for a broader comparison across the remaining panels
and the official prediction setup. The package defaults remain the previously
scored reference. This search uses only two panels, two months, and 28-day
training histories; it does not establish how the settings perform with the
nine-month official training pool or hidden queue labels. No Kaggle upload was
made.

## Changes and verification

- Added an optional selector-matched onset rule to benchmark preparation.
  The standard benchmark scheme retains its existing onset sampling.
- Exposed feature sampling and L2 regularization as validated model parameters.
- Added a seeded random search that retains defaults, avoids duplicate trials,
  rejects overlapping target periods, and freezes selection before later scoring.
- Corrected persistence fallback calls in the three queue solution packages to
  supply the required parameter argument.

All 58 tests passed. The final edited sampling/search tests also passed after
type-annotation cleanup. Pyright, Ruff lint and format checks, Git whitespace
checks, and repository convention checks pass. All eight experiment runs share
the same source snapshot and evaluator hashes and use LightGBM 4.7.0.

## Evidence and reproduction

The [research and protocol](research.md) records online sources, fold dates,
sampling rules, and limitations. The
[solution guide](../../solutions/queue_boosting_event_aware/README.md#tune-parameters-on-selector-matched-windows)
provides preparation and search commands.

- [All six trial scores and settings](trials.csv).
- [Search record and frozen selection](search.json).
- [Panel and condition comparisons](comparison.csv).
- [Benchmark timing and count audit](benchmark_audit.json).
- [December benchmark manifest](december/benchmark.json) and
  [January benchmark manifest](january/benchmark.json).
- `trials/trial-0` through `trials/trial-5` retain metrics and run metadata.
  Trial 0 is the reference and trial 2 is the selected candidate.
- [January reference metrics](holdout-reference/metrics.csv) and
  [candidate metrics](holdout-candidate/metrics.csv) retain individual window scores.

The original run paths in `search.json` refer to local `runs/cal31-hpo/runs/`
directories. Those directories contain the complete source snapshots and
predictions. The retained records above preserve the comparison in this branch.
