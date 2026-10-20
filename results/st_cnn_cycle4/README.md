# st_cnn cycle 4: conservation-aware training (CAL-12)

Branch `cycle-4/CAL-12/density-loss`, on top of cycle 3's local Task 3 score. New setting
`conservation_weight` adds Task 3's conservation error of the hidden training cells to the loss:
the error in step-to-step changes of each link's vehicle count `N = flow / speed * length`
(the evaluator's boundary flows make the true state conserve exactly, so this is the residual
it scores). Everything else is the cycle-2 setup (2000 steps, low-quality inputs, blackout 0.1).

"Points" = 0.35 x Task 1 + 0.15 x Task 3, the share of the total score the two tasks give.

## Screening (D12_I5_N, December)

| conservation_weight | Task 1 | S_LWR | Task 3 | Points |
| ---: | ---: | ---: | ---: | ---: |
| 0 (cycle 2) | 0.9497 | 0.507 | 0.668 | 0.4326 |
| 0.1 | 0.9406 | 0.550 | 0.697 | 0.4337 |
| **0.3** | 0.9373 | 0.580 | 0.717 | **0.4355** |
| 1.0 | 0.9205 | 0.587 | 0.721 | 0.4303 |

Weight 0 reproduces cycle 2's D12_I5_N result exactly. S_FD stays at 0.988-0.990.

## Full benchmark (full_dec, all 10 panels)

| full_dec | Task 1 | Speed RMSE | Flow RMSE | S_LWR | Task 3 | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial_low_rank (team) | 0.9305 | 2.06 | 32.5 | 0.394 | 0.591 | 0.4144 |
| st_cnn cycle 2 | **0.9443** | 1.47 | 31.2 | 0.549 | 0.695 | 0.4347 |
| **st_cnn cycle 4 (weight 0.3)** | 0.9366 | 1.74 | 33.6 | **0.603** | **0.731** | **0.4375** |

| Family | Task 1 cycle 2 | Task 1 cycle 4 | Task 3 cycle 2 | Task 3 cycle 4 |
| --- | ---: | ---: | ---: | ---: |
| D12_I405 | 0.9506 | 0.9420 | 0.713 | 0.730 |
| D12_I5 | 0.9471 | 0.9333 | 0.647 | 0.717 |
| D7_I10 | 0.9490 | 0.9420 | 0.702 | 0.734 |
| D7_I210 | 0.9420 | 0.9380 | 0.721 | 0.738 |
| D7_I405 | 0.9326 | 0.9274 | 0.690 | 0.736 |

The conservation term trades Task 1 accuracy for conservation: Task 3 improves on every family
(+0.037 overall), Task 1 drops on every family (-0.008), net +0.0028 total score. S_FD is unchanged
(0.987). Training time is the same (53 min). The Task 3 numbers rely on the local scorer's
exact-flux assumption, so check the trade-off on the leaderboard before relying on it.
Per-panel scores: `full_dec_task1_metrics.csv`, `full_dec_task3_scores.csv`.

## Reproduce

```sh
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/full_dec \
  --params '{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1, "conservation_weight": 0.3}'
POLARS_MAX_THREADS=2 uv run --locked python scripts/task3/local_score.py data/benchmarks/full_dec runs/RUN_ID
```
