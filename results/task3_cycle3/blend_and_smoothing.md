# Blending and smoothing for Task 3 (cycle 3, CAL-12)

Two no-training variants of the st_cnn cycle-2 December predictions on full_dec (all 10 panels),
scored with the competition Task 1 formula and the local Task 3 score (`local_score.py`).
"Points" = 0.35 x Task 1 + 0.15 x S_physics, the share of the total score these two tasks give.

- **Blend:** `(1 - w) * st_cnn + w * spatial_low_rank` for speed and flow.
- **Smooth:** at target cells, move the car count `N = flow / speed * length` a fraction `a`
  toward the mean of the same link's previous and next 5-minute values; speed is kept.

| Variant | Task 1 | S_LWR | S_physics | Points | vs st_cnn |
| --- | ---: | ---: | ---: | ---: | ---: |
| st_cnn (cycle 2) | 0.9443 | 0.5485 | 0.6947 | 0.4347 | |
| **blend w = 0.2** | **0.9451** | 0.5686 | 0.7081 | **0.4370** | **+0.0023** |
| blend w = 0.3 | 0.9448 | 0.5655 | 0.7060 | 0.4366 | +0.0019 |
| blend w = 0.1 | 0.9450 | 0.5636 | 0.7047 | 0.4364 | +0.0017 |
| blend w = 0.5 | 0.9425 | 0.5414 | 0.6898 | 0.4334 | -0.0013 |
| spatial_low_rank alone | 0.9305 | 0.3943 | 0.5914 | 0.4144 | -0.0203 |
| smooth a = 0.25 | 0.9411 | 0.5714 | 0.7099 | 0.4359 | +0.0012 |
| smooth a = 0.5 | 0.9329 | 0.5714 | 0.7099 | 0.4330 | -0.0017 |
| smooth a = 1.0 | 0.9111 | 0.4987 | 0.6611 | 0.4181 | -0.0166 |

S_FD stays at 0.986-0.987 for every variant.

**Findings**

- A 20% blend with spatial_low_rank improves both tasks at once (Task 1 +0.0008, conservation
  +0.020): the two models' errors partly cancel. Best variant, +0.0023 total score.
- Smoothing raises conservation as much as blending (+0.023 at a = 0.25) but costs Task 1
  accuracy, so the net gain is smaller (+0.0012) and turns negative from a = 0.5.
- Caveat: the blend weight and smoothing strength were picked on the same December data they
  are scored on, so the gains are slightly optimistic. One weight per method limits this.

Reproduce (about 3 minutes on two CPU threads):

```sh
POLARS_MAX_THREADS=2 uv run --locked python scripts/task3/blend_and_smooth.py \
  data/benchmarks/full_dec runs/ST_CNN_RUN runs/SPATIAL_LOW_RANK_RUN --output blend_and_smoothing.csv
```
