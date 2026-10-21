# st_cnn cycle 5: architecture probes (CAL-12)

Branch `cycle-5/CAL-12/architecture-probes`. Two quick probes of whether st_cnn's architecture
still has headroom, each on top of the cycle-4 setup (2000 steps, low-quality inputs, blackout 0.1,
conservation weight 0.3), on D12_I5_N for December 2030 (full_dec dates). Task 3 uses
`scripts/task3/local_score.py`. Points = 0.35 x Task 1 + 0.15 x Task 3.

- **A, `output: "density"`:** the network predicts speed and per-lane density; flow = density x speed.
- **B, capacity:** `width: 48` (was 32) and `context: "wide"` (three more dilated blocks: about
  +-94 steps and +-35 links of context instead of +-44 and +-17).

| Probe | Task 1 | Speed RMSE | Flow RMSE | S_LWR | Task 3 | Points | vs cycle 4 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cycle-4 setup | 0.9373 | 1.75 | 32.4 | 0.580 | 0.717 | 0.4355 | | 9 min |
| A: density output | 0.9407 | 1.59 | 32.7 | 0.618 | 0.742 | 0.4406 | +0.0051 | 9 min |
| B1: width 48 | 0.9438 | 1.54 | 29.9 | 0.603 | 0.732 | 0.4402 | +0.0047 | 20 min |
| B2: wide context | 0.9409 | 1.64 | 30.9 | 0.603 | 0.732 | 0.4391 | +0.0036 | 12 min |
| **B3: width 48 + wide** | **0.9493** | **1.34** | **28.3** | **0.635** | **0.754** | **0.4453** | **+0.0098** | 25 min |

The reference reproduces the cycle-4 screening run exactly. S_FD stays at 0.990.

**Findings**

- The architecture is not at its ceiling: more capacity and longer context together give the
  largest gain (+0.0098), recovering cycle 2's Task 1 (0.9497) while keeping a much better Task 3.
- Predicting density is a free win (+0.0051, same training time): it improves both tasks, mostly
  conservation (S_LWR +0.038).
- Not yet tested: density output combined with B3, and 4000 steps. Caveats: one panel and one
  seed (earlier seed noise was about 0.0001); B3 takes about 2.9x the training time, so all
  panels would take about 4 hours on an M4.

```sh
uv run --locked bench experiment st_cnn --task state --benchmark data/benchmarks/full_dec --panel D12_I5_N \
  --params '{"steps": 2000, "low_quality_inputs": true, "blackout_rate": 0.1, "conservation_weight": 0.3, "width": 48, "context": "wide"}'
```
