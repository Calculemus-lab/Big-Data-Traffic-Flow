# st_cnn cycle 2: hardening (CAL-12)

Branch `cycle-2/CAL-12/st-cnn-hardening`, on top of the cycle-1 model (`results/st_cnn/README.md`).
Same network, training method and loss; five changes tested as switchable settings.

**Benchmark: full_dec** (all 10 panels, history 2030-06-01 to 2030-12-01, targets December 2030).
Our copy's id is `876ec34baa40` (the id hashes file timestamps), but it is the team's benchmark:
`spatial_low_rank` reproduces its 0.930499 on it exactly.

## Result

| full_dec, all 10 panels | Score | Speed RMSE | Flow RMSE / lane | FD error | Time (M4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6966 | | | | |
| `spatial_low_rank` (team best, CAL-23) | 0.9305 | 2.06 | 32.5 | 0.017 | 14 min |
| st_cnn, cycle-1 settings | 0.9398 | 1.64 | 32.3 | 0.013 | 30 min |
| **st_cnn, cycle-2 final** | **0.9443** | **1.47** | **31.2** | **0.012** | 57 min |

Cycle-2 final = 2000 steps, low-quality inputs, blackout training at 0.1
(`{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1}`).

| Family | spatial_low_rank | st_cnn cycle-1 | st_cnn cycle-2 |
| --- | ---: | ---: | ---: |
| D12_I405 | 0.9342 | 0.9464 | 0.9506 |
| D12_I5 | 0.9272 | 0.9404 | 0.9471 |
| D7_I10 | 0.9421 | 0.9454 | 0.9490 |
| D7_I210 | 0.9305 | 0.9396 | 0.9420 |
| D7_I405 | 0.9185 | 0.9273 | 0.9326 |

st_cnn beats the team's best on every family; most of the gain is speed (RMSE 2.06 to 1.47 km/h).
Per-panel and per-regime metrics: `full_dec_*_metrics.csv`. Out-of-fold predictions stay in the
local run folders (too large to commit).

## The five changes (screened one at a time on D12_I5_N, December)

| Change | Setting | Score (cycle-1 settings: 0.9440) | Decision |
| --- | --- | ---: | --- |
| Longer training | `steps` | 2000: 0.9505, 4000: 0.9554 | 2000 in the final run (time) |
| Low-quality readings (coverage < 75%) as flagged inputs | `low_quality_inputs` | 0.9454 | keep |
| Blackout-aware training (random 90-min all-link gaps) | `blackout_rate` | see below | keep at 0.1 |
| Seed averaging, 3 models | `ensemble` | 0.9462, 3x the time | drop |
| Leave-day-out training profile | `leave_day_out` | 0.9437 | drop |

Seed noise is about 0.0001 (seed 0: 0.94400, seed 1: 0.94406).

**Blackouts.** Validation and private blank all links for 90 minutes around each Task 2 window;
December has none, so `blackout_check` blanks 10 simulated ones (1.9% of targets) and
`solutions/st_cnn/blackout_check.py` scores those targets separately:

| D12_I5_N, 10 simulated blackouts | Inside | Other | All |
| --- | ---: | ---: | ---: |
| cycle-1 settings | 0.600 | 0.942 | 0.923 |
| 4000 steps + low-quality | 0.840 | 0.956 | 0.952 |
| 4000 steps + low-quality + `blackout_rate` 0.1 | **0.891** | 0.954 | **0.954** |

Blackout training costs about 0.002 on normal cells, so full_dec (no blackouts) slightly
understates it; 8 of 10 panels have blackouts on the leaderboard.

**4000 steps instead of 2000** added +0.005 on the screening panel (0.9505 to 0.9554). Scores were
still rising with each doubling (+0.0065, then +0.0049), so the model is undertrained; expect roughly
+0.004 to +0.005 on full_dec for about twice the training time
(about 2 h instead of 1 h for all panels on an M4). Not run here to save time.

## Reproduce

```sh
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/full_dec \
  --params '{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1}'
```

All runs, including dropped ideas, are in `run_log.csv`.
