# Task 3 cycle 3: a local physics score (CAL-12)

Branch `cycle-3/CAL-12/task3-local-score`. A first, cheap Task 3 experiment: score the
physical consistency of Task 1 predictions we already have, with no new training. Follow-up
with blending and smoothing: [`blend_and_smoothing.md`](blend_and_smoothing.md). Code is in
`scripts/task3/`.

**Why it can be scored locally.** The official evaluator checks vehicle conservation against
organizer-held boundary flows that make the *true* state conserve vehicles exactly
(`official_competition_repo/src/task3/score_task3.py`, `load_boundary_flux`). On a train case the
truth is known, so the conservation term is
`S_LWR = 1 - sum|dN_pred - dN_true| / sum|dN_true|`, with `N = flow / max(speed, 1) * length`, over
the official transitions. `S_FD` uses the official formula; `S_physics = (S_FD + 2 S_LWR) / 3`,
averaged over regimes, panels and families. The public script cannot do this (without the
boundary flows it scores even the truth at 0).

## Result (full_dec, December 2030, all 10 panels)

| Task 1 predictions | Task 1 score | S_FD | S_LWR | S_physics |
| --- | ---: | ---: | ---: | ---: |
| Truth (ceiling) | 1 | 0.9863 | 1.0000 | 0.9954 |
| Baseline | 0.6966 | 0.9776 | 0.0122 | 0.3340 |
| `spatial_low_rank` | 0.9305 | 0.9855 | 0.3943 | 0.5914 |
| `st_cnn`, cycle-1 settings | 0.9398 | 0.9867 | 0.5051 | 0.6656 |
| `st_cnn`, cycle-2 settings | 0.9443 | 0.9869 | 0.5485 | 0.6947 |

- `S_FD` is saturated: every model already matches the truth's own 0.986.
- All headroom is in conservation: `S_LWR` 0.55 vs 1.0, worth up to about +0.045 total score.
- Better Task 1 moves Task 3 far more than Task 1 itself: cycle 2's +0.0045 on Task 1 (+0.0016
  total) raised `S_LWR` by +0.043 (+0.0043 total).

**Next step if this direction is kept:** `S_LWR` measures errors in accumulation changes
(density = flow / speed) at target cells, which the Task 1 loss does not target. Adding a density
or accumulation term to st_cnn's loss, scored with this script, is the obvious first experiment.

## Run

About 20 seconds for all panels on two CPU threads:

```sh
POLARS_MAX_THREADS=2 uv run --locked python scripts/task3/local_score.py \
  data/benchmarks/full_dec truth runs/RUN_ID [runs/RUN_ID ...] --output scores.csv
```

`scores.csv` has the per-panel, per-regime scores of the runs above.
