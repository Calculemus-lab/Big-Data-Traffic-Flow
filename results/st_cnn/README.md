# st_cnn: spatiotemporal masked CNN (Task 1)

CAL-12, branch `cycle-1/CAL-12/deep-learning-t1-k`. A first deep-learning approach for state
reconstruction, kept small so it trains in a few minutes on a laptop.

## How it works

- Each panel becomes a dense grid of 5-minute steps x mainline links in corridor order
  (`lwr_mainline_topology.order_index`), with speed and per-lane flow channels.
- History = masked rows before `history_end_date` with their known Task 1 answers restored
  (`historical_task_labels.task1_state_answers`). A day-of-week x time-of-day profile is fitted
  on it (smoothed over +-2 slots, cyclic across the week); the network predicts residuals over it.
- Everything from `history_end_date` onward is one evaluation grid of visible cells. It covers
  every requested target, including targets in more than one split. Requested answers are never
  in the release slice.
- Inputs: visible residuals (zero where hidden), a visibility mask, the profile level and
  time of day. 7 channels in total.
- Network: 8 residual blocks of dilated 3x3 convolutions, 32 channels, about +-44 steps
  (3.7 h) and +-17 links of context. Fully convolutional, so it runs on the whole evaluation
  period in overlapping chunks.
- Training is self-supervised: random crops of the history grid (and, by default, of the
  visible cells of the evaluation period) have their visible cells hidden at 20/30/50%. The
  loss is the competition score itself on those cells,
  `0.54 RMSE_speed/25 + 0.46 RMSE_flow_per_lane/600`.
- Only the six columns the model needs are read from the lazy release tables.

```sh
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/BENCHMARK
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/BENCHMARK \
  --params '{"steps": 2000, "seed": 1}'
uv run --locked bench run st_cnn --task state --target-range validation   # all panels
uv run --locked python -m pytest -q tests/test_st_cnn.py
```

Settings (`--params`): steps, batch, window, width, lr, smooth, use_eval, device, log_every,
seed (offset on the benchmark seed). One panel-week takes about 4-6 minutes on an M4 (MPS) at
the default 1000 steps; training time does not grow with the history length.

## Quick CV results (old harness: D12_I5_N, December week, benchmark 497cf28c170d)

| Run | Speed RMSE | Flow RMSE (per lane) | State score |
| --- | ---: | ---: | ---: |
| baseline (dow x slot mean) | 16.20 | 118.8 | 0.5590 |
| residual_interp (CAL-14) | 3.62 | 39.5 | 0.8916 |
| **st_cnn, defaults, seeds 0/1/2** | 1.57 | 31.0 | **0.9424 +- 0.0010** |
| st_cnn, no eval-period training* | 1.61 | 30.9 | 0.9416 |
| st_cnn, 2000 steps* | 1.33 | 28.4 | 0.9495 |

Seeds: 0.9431, 0.9413, 0.9427 (about 4 minutes each on an M4). Per regime (seed 0): R1 0.9476,
R2 0.9426, R3 0.9392; residual_interp is 0.905 / 0.895 / 0.875. FD relative error diagnostic
0.010 (residual_interp 0.024).

\* Single runs on the code before the review fixes (profile smoothing now wraps across days,
which changed seed 0 from 0.9432 to 0.9431).

After the port to the Polars Trafficbench, the same data (history 2030-11-03 to 2030-12-01,
targets 2030-12-01 to 2030-12-08, D12_I5_N) gives 0.9409 (seed 0), within the old seed spread.

Files: `quick_metrics.csv`, `quick_report.md` (seed 0) and `quick_oof_D12_I5_N_december.parquet`
(its out-of-fold predictions, for blending).

## Caveats

- One panel and one week. Seed spread alone is about 0.001, so smaller differences are noise.
- The loss was still falling at 1000 steps; 2000 steps gained +0.006, so it is undertrained by default.
- In validation and private, observations around every scored Task 2 window are removed for
  the 30-minute horizon plus the following hour (`trafficflowbench-public/docs/MASK_SPEC.md`),
  so Task 1 targets inside those 90 minutes have no visible same-time neighbours. Quick CV is
  built from train days and has no such blackouts, and st_cnn never sees one in training.
  Expect a lower leaderboard score on the eight Task 2 panels until blackouts are added to
  the training masks.
- On CPU with two threads (the GitHub benchmark runners) the default 1000 steps take roughly
  25 minutes or more per panel, which does not fit the 30-minute panel job. Run it locally, or
  pass fewer steps in CI.
- torch 2.14.1 has wheels for Apple Silicon macOS, Linux and Windows, but not Intel macOS.

## Deep-learning options for the next cycle

From a literature and public-notebook sweep (2026-10-06), ranked by expected value for the time left:

1. **Harden st_cnn**: average 3-5 seeds, train longer (2000+ steps), time it on CPU, run the full
   profile. Cheapest, most reliable gain.
2. **Blackout-aware training**: add 90-minute all-link gaps to the training masks and build a
   blackout CV from the quick benchmark's queue window origins. Protects the leaderboard score
   on the eight Task 2 panels.
3. **Physics-aware outputs (Task 3)**: build a local conservation proxy on the saved predictions,
   then try a density head or gating flow as `q = v * k` in congested cells. Task 3 is 15% of the
   total, against 35% for Task 1, and there is little Task 1 headroom left at ~0.94.
4. **Cheap input fixes**: the regime label, low-quality (`pct_observed < 75`) readings as flagged
   inputs, static per-link channels (lanes, free speed, capacity), ramp flows, and a profile that
   leaves out the crop's own day during training.
5. **Capacity and sharing**: sweep width, steps and link dilation; one model shared by all panels.
6. **Second ensemble member**: a bidirectional GRU with neighbour mixing along the corridor
   (GRIN-style), or a GBDT residual model, which is what most public Task 1 entries use.

Not worth it this cycle: diffusion imputers (CSDI, PriSTI: slow, tuned for MAE), BRITS, SAITS and
TimesNet (lose to interpolation on point-missing traffic data), and PINNs or hard conservation
layers (conservation cannot be identified from the released observations).
