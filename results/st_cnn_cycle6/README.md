# st_cnn cycle 6: final setup (CAL-12)

Branch `cycle-6/CAL-12/final-st-cnn`. Combines cycle 5's two wins (density output; width 48 +
wide context) on top of the cycle-4 setup. Settings only, no code changes. Points = 0.35 x Task 1
+ 0.15 x Task 3; Task 3 uses `scripts/task3/local_score.py`.

Final setup:

```json
{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1, "conservation_weight": 0.3,
 "width": 48, "context": "wide", "output": "density"}
```

## Screen (D12_I5_N, December)

| Setup | Task 1 | S_LWR | Task 3 | Points |
| --- | ---: | ---: | ---: | ---: |
| cycle-4 setup | 0.9373 | 0.580 | 0.717 | 0.4355 |
| B3: width 48 + wide (cycle 5) | 0.9493 | 0.635 | 0.754 | 0.4453 |
| **B3 + density output** | **0.9497** | **0.657** | **0.768** | **0.4476** |

The two gains mostly add up: density output adds +0.0023 on top of B3, almost all of it Task 3.

## Full benchmark (full_dec, all 10 panels)

| full_dec | Task 1 | Speed RMSE | Flow RMSE | S_LWR | Task 3 | Points | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial_low_rank (team) | 0.9305 | 2.06 | 32.5 | 0.394 | 0.591 | 0.4144 | 14 min |
| st_cnn cycle 2 | 0.9443 | 1.47 | 31.2 | 0.549 | 0.695 | 0.4347 | 57 min |
| st_cnn cycle 4 | 0.9366 | 1.74 | 33.6 | 0.603 | 0.731 | 0.4375 | 53 min |
| **st_cnn cycle 6** | **0.9446** | **1.42** | 32.1 | **0.645** | **0.759** | **0.4445** | 160 min |

| Family | Task 1 cycle 4 | Task 1 cycle 6 | Task 3 cycle 4 | Task 3 cycle 6 | Points gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| D12_I405 | 0.9420 | 0.9505 | 0.730 | 0.760 | +0.0074 |
| D12_I5 | 0.9333 | 0.9473 | 0.717 | 0.770 | +0.0129 |
| D7_I10 | 0.9420 | 0.9497 | 0.734 | 0.755 | +0.0058 |
| D7_I210 | 0.9380 | 0.9416 | 0.738 | 0.752 | +0.0033 |
| D7_I405 | 0.9274 | 0.9340 | 0.736 | 0.758 | +0.0055 |

- +0.0070 points over cycle 4, with gains on every family. It is the first setup that is best on
  both tasks: Task 1 recovers cycle 2's level and Task 3 beats cycle 4's. S_FD is unchanged (0.987).
- The D12_I5_N panel reproduces the screen exactly.
- Blending with spatial_low_rank no longer helps: a 10% blend gives 0.4443, and anything stronger is
  worse. Car-count smoothing also hurts (`full_dec_blend.csv`). Both are dropped.
- Cost: 3x the training time (160 min on full_dec, 2 h 53 min for the submission on an M4 Air).
- Minor: density output sometimes predicts near-zero speed deep inside jams. This happens on 50 of
  about 6M December targets (true mean 8.7 km/h, mean speed error 5.5 km/h) and on about 85 rows in
  March/April. The effect on the score is negligible.

## Submission

`temp/submission/st_cnn_cycle6_task1_only.csv` (local, not in git). It holds Task 1 predictions for
March and April from the final setup, with zero rows for Tasks 2 and 4, and passes `bench validate`
(6,980,503 rows). Cycle 4's file scored 0.42771 on the public leaderboard. If the ~60% of local
gains that carried over from cycle 2 to cycle 4 holds again, expect about 0.432.

Per-panel scores: `full_dec_task1_metrics.csv`, `full_dec_task3_scores.csv` (cycles 4 and 6),
`screen_task3_scores.csv`.

## Reproduce

```sh
P='{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1, "conservation_weight": 0.3, "width": 48, "context": "wide", "output": "density"}'
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/full_dec --params "$P"
POLARS_MAX_THREADS=2 uv run --locked python scripts/task3/local_score.py data/benchmarks/full_dec runs/RUN_ID
uv run --locked bench run st_cnn --task state --target-range both --params "$P"   # March + April
```
