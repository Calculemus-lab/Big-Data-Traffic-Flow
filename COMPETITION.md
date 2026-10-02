# 2026 IEEE Big Data Traffic Flow Bench: competition reference

This file is the project’s single reference for the competition description, data contract, scoring, submission, and award process. It consolidates the competition material supplied to this repository on 2026-10-01 and was checked against the local [`trafficflowbench-public/`](trafficflowbench-public/) repository. The repository defines the public evaluators and schemas; the pasted competition page supplies award and deadline details that the repository does not document. For an exact Kaggle cutoff or a later rule change, check the [competition page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview).

## At a glance

| Item | Rule |
|---|---|
| Objective | Reconstruct masked freeway speed and flow, forecast queues, maintain physical consistency, and estimate path/OD demand. |
| Data | Synthetic five-minute records on ten directional freeway panels, grouped into five corridor families. Model parameters were calibrated from real detector data; no real detector record is released. |
| Splits | `train`: nine months with unmasked answers; `validation`: one month for the public leaderboard; `private`: one month for final ranking. |
| Setting | Tasks 1, 3, and 4 are offline over the full evaluation period. Task 2 uses 60 minutes of history to predict the next 30 minutes. |
| Leaderboard | `S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME`. Each component is in `[0, 1]`. |
| Aggregation | Score each directional panel, average its two directions within a family, then average the five families equally. |
| Missing output | A wholly missing task scores zero; the other tasks still score. Missing rows within a task follow task-specific evaluator rules (below). |
| Submission | One CSV covering both `validation` and `private`. Task 3 has no separate submission rows. |

## Tasks and scoring

| Task | Inputs | Submitted values | Score |
|---|---|---|---|
| 1. State reconstruction | Masked link-time observations under `R1`, `R2`, `R3` | `speed_kmh`, `flow_vph` on `state` rows | Speed and per-lane flow error on blanked eligible cells |
| 2. Queue forecasting | 60 minutes through forecast origin `T` | Binary `queue_pred` on `queue` rows | Space-time intersection over union over `T+5` through `T+30` |
| 3. Physics | Task 1 submission, topology, fundamental-diagram parameters, ramp flows | Nothing separate | Fundamental-diagram validity and vehicle conservation |
| 4. OD/path flow | Paths, path-link incidence, link counts, weak prior | Nonnegative `path_flow` on `odme` rows | Reference demand, link counts, prior deviation, destination attraction |

### Eligibility and quality

`is_imputed = pct_observed < 100`. A cell is score-eligible when `pct_observed >= 75` and the values required by the task are present. Roughly 76% of mainline and ramp cells qualify. Ineligible cells remain in the release but are not ground truth. A ramp cell below the threshold is **missing evidence**, not a measured zero.

### Task 1: state reconstruction

Predict only the blanked eligible cells listed in `task1/<PANEL>/<split>/sample_submission_state.csv`. The regimes remove 20%, 30%, and 50% of eligible cells for `R1`, `R2`, and `R3`, respectively. Each calendar day appears in exactly one regime. All measured channels are blanked together: speed, flow, occupancy, and occupancy-derived density. Quality fields remain.

For each regime `r`:

```text
S_speed(r) = max(0, 1 - RMSE_speed(r) / 25)
S_flow(r)  = max(0, 1 - RMSE_flow_per_lane(r) / 600)
S_state(r) = 0.54*S_speed(r) + 0.46*S_flow(r)
S_state    = [S_state(R1) + S_state(R2) + S_state(R3)] / 3
```

Speed RMSE is in km/h; flow RMSE is in veh/h/lane, using lane counts from `fd_parameters.csv`. Density is neither submitted nor scored for Task 1. Task 2 horizons and the 60 minutes after each horizon are also blanked in the state files; those blanks are **not** Task 1 targets. Use the task template rather than deriving targets from the mask hash.

### Task 2: queue forecasting

For each window, predict every link at `T+5`, `T+10`, `T+15`, `T+20`, `T+25`, and `T+30` minutes. The origin `T` is excluded. The label comes from the underlying state: a link is queued when `speed <= 0.60 * free_speed`; noisy released observations do not determine the label.

Each included panel and split has ten windows: five `queue_onset` windows, with no queued link in the history and at least one in the horizon, and five `queue_ongoing` windows, with a queued link at `T`. Every horizon contains a queue, so predicting all zeroes cannot win a window. The horizon and following 60 minutes are hidden to prevent reading or interpolation.

```text
IoU(window) = |predicted queued cells ∩ true queued cells|
              / |predicted queued cells ∪ true queued cells|
S_queue     = mean window IoU, with equal weight for each window and condition
```

If both sets are empty, IoU is 1, though released evaluation windows have a nonempty true set. `D12_I405_N` and `D12_I405_S` have no Task 2 windows; they remain in Tasks 1, 3, and 4.

### Task 3: physical consistency

Task 3 reads the Task 1 speed and flow submission. The evaluator derives density from flow and speed, vehicle accumulation `N = k*L`, organizer-held boundary flows using the released topology, and ramp terms from released observations. Improving Task 1 values is the only way to change this score.

```text
S_physics = (1/3)*S_FD + (2/3)*S_LWR

k = q / max(v, 1)           evaluator safeguard at very low speed
q_FD(k) = v_f*k             if k <= k_crit
        = w*max(k_jam-k, 0) if k > k_crit

N(t+dt) - N(t) = dt*(q_in + q_on_ramp - q_out - q_off_ramp)
```

`S_FD` checks the normalized gap between submitted and fundamental-diagram-consistent flow at the same density, in per-lane terms. `S_LWR` penalizes conservation residuals over five-minute steps and carries two thirds of the physics weight. The [scoring specification](trafficflowbench-public/docs/SCORING_SPEC.md) gives the exact normalization and safeguards.

Ramp coverage fixes which transitions are checked. Mode A uses eligible on/off-ramp observations as anchors; Mode B scores transitions with valid attached-ramp observations and ramp-free mainline transitions, excluding attached-ramp transitions with invalid observations; Mode C checks only ramp-free mainline transitions. Organizers fix and publish each panel’s mode. All ten panels in this release are Mode A, with 0.756 ramp coverage; see [`config/task3_lwr_modes.json`](trafficflowbench-public/config/task3_lwr_modes.json).

**Local-score limitation:** Public `score_task3.py` lacks organizer boundary flows and substitutes an estimate from submitted flows. Its `S_LWR` can floor at zero even for a correct answer, so its printed physics score is not a reliable ranking signal. Train against locally scoreable Task 1 and use leaderboard feedback for Task 3. The competition material gives an illustrative `D12_I5_N` comparison of approximately 0.33 locally versus 0.96 with organizer flows for a perfect answer; exact figures may vary by evaluator version.

### Task 4: OD and path-flow estimation

Let `A` be the released path-link incidence matrix and `f` the submitted path-flow vector. Loaded link flow is `A*f`.

```text
S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr
```

`S_od` compares submitted and withheld reference path flows; `S_link` compares `A*f` with released counts; `S_dev` compares distance from the weak prior with the reference’s distance from that prior; `S_attr` compares destination-attraction distributions. `S_dev` discourages both leaving the prior untouched and fitting counts with implausible demand. The [scoring specification](trafficflowbench-public/docs/SCORING_SPEC.md) contains the exact formulas. Submitted path IDs and origin/destination zones must match the released network files; link references belong to the released incidence asset, not the submission. Path flows must be finite and nonnegative. `departure_time` is a period token rather than a timestamp; the public release defines one period per split.

## Dataset and file layout

The actual data is downloaded from the [Kaggle Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data), not from the code repository. The extracted root is `kaggle_public/`; use that as `--release-root` for scripts.

```text
kaggle_public/
  config/
    synthetic_release_v1.json       calendar and release contract
    corridors.json                  directional panels
  corridors/<PANEL>/
    network/
      links.csv                     lengths, lanes, free speeds, capacities
      fd_parameters.csv             FD parameters and lane counts
      lwr_mainline_topology.csv     mainline connections
      ramp_attachment_map.csv       ramp connections
      path_set.csv                  paths and OD zones
      path_link_incidence.csv       path-to-link membership
    train/
      mainline_states/             unmasked observations
      mainline_states_masked/      Task 1 targets and Task 2 horizon hidden
      ramp_states/                 on/off-ramp counts
    validation/ and private/
      mainline_states_masked/
      ramp_states/
  task1/<PANEL>/<split>/sample_submission_state.csv
  task2/<PANEL>/<split>/window_index.csv
  task2/<PANEL>/<split>/window_history.parquet
  task2/<PANEL>/<split>/sample_submission_queue.csv
  task4/<PANEL>/<split>/synthetic_link_counts.csv
  task4/<PANEL>/<split>/synthetic_weak_prior.csv
  task4/<PANEL>/<split>/sample_submission_path_flow.csv
  sample_submission.csv
  submission_key.csv
```

The ten directional panels are:

| Family | Directions |
|---|---|
| `D7_I10` | `D7_I10_E`, `D7_I10_W` |
| `D7_I210` | `D7_I210_E`, `D7_I210_W` |
| `D7_I405` | `D7_I405_N`, `D7_I405_S` |
| `D12_I5` | `D12_I5_N`, `D12_I5_S` |
| `D12_I405` | `D12_I405_N`, `D12_I405_S` |

| Split | Synthetic dates | Days | Masked inputs | Unmasked observations |
|---|---|---:|:---:|:---:|
| `train` | 2030-06-01 to 2031-02-28 | 273 | Yes | Yes |
| `validation` | 2031-03 | 31 | Yes | No |
| `private` | 2031-04 | 30 | Yes | No |

Join masked and unmasked training mainline records on `(timestamp, station_id, link_id)` to obtain training answers. The pasted Kaggle dataset description lists speed, flow, occupancy, occupancy-derived density, station/link IDs, milepost, and quality fields on mainline records, and crossing counts, IDs, and quality fields but no freeway speed on ramp records. The public code repository does not establish that full released-file schema; inspect the downloaded data before relying on the occupancy-derived density field. Validation and private were generated independently, with separate demand draws and incident schedules (five and seven incidents, respectively); their leaderboard rankings can differ. The calendar corresponds to no real period, and the data cannot be matched to external detector records.

### Local evaluation limits

| Task | Local evaluation | Reason |
|---|---|---|
| 1 | Yes, on `train` | Unmasked observations are released. |
| 2 | No | Evaluation queue labels are withheld on every split. |
| 3 | No reliable score | Organizer boundary flows are withheld; the public fallback is too coarse. |
| 4 | Partial | Local reference construction checks a solver more than leaderboard rank. |

## Submission contract

Fill the Kaggle `sample_submission.csv` template and upload **one CSV**. It covers both evaluation splits and contains one row for every required value. The columns are:

```csv
submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
1,state,0,0,0,0
```

| `task` | Fill | Keep at zero |
|---|---|---|
| `state` | `speed_kmh` (km/h), `flow_vph` (veh/h) | `queue_pred`, `path_flow` |
| `queue` | `queue_pred` (`0` or `1`) | `speed_kmh`, `flow_vph`, `path_flow` |
| `odme` | finite `path_flow >= 0` | `speed_kmh`, `flow_vph`, `queue_pred` |

The pasted Kaggle schema says `submission_key.csv` maps each `submission_id` to `task`, `split`, `panel`, `timestamp`, `station_id`, `link_id`, `mask_regime`, `window_id`, `departure_time`, `path_id`, `origin_zone`, and `destination_zone`. The repository confirms the key’s role and the natural keys used by the merge script, but does not document its complete column list. Join the downloaded key to make predictions, then write values into the upload template; **do not upload the key**. Per-task templates also cover `train` for local development. Task 3 has no rows.

The pasted Kaggle upload instructions require no empty cells, unchanged and unique `submission_id` values, no added rows, complete task row sets, binary `queue_pred`, and finite nonnegative `path_flow`. They state that a missing required row makes the whole task score zero and an unknown ID makes the whole submission score zero. **The public per-task evaluators differ:** a missing Task 1 row is scored as a zero speed/flow prediction, a missing Task 2 row as `queue_pred=0`, and missing, duplicate, or invalid Task 4 paths invalidate the affected panel. To satisfy the stated Kaggle upload rule, retain every template row regardless of the local scorer’s tolerance.

To combine per-task files, from `trafficflowbench-public/` run:

```bash
python src/merge_submissions.py \
  --state state_submission.csv \
  --queue queue_submission.csv \
  --odme path_flow_submission.csv \
  --key submission_key.csv \
  --output submission.csv
```

## Leaderboards and final awards

The public leaderboard uses `validation`; the private leaderboard uses `private` and determines Kaggle rank. Fit methods on `train`, use public results to assess transfer, and expect rankings to shift because validation and private are independently generated months.

Award eligibility requires all of the following: registration by email by **2026-10-25**; final code and PDF report by **2026-11-10**; and compliance with the Competition Rules. The top ten eligible teams on the private leaderboard enter review. Organizers run their code; predictions must reproduce their selected final submission within **1% of its private leaderboard score**. A failed team leaves the pool and the next eligible team enters.

For verified teams in the review pool:

```text
Final Score = 0.85*L + 0.15*R
```

`L` is the private score min-max normalized within the review pool to `[0, 100]` (all teams get 100 if tied). `R` is the committee’s average report score on `[0, 100]`: technical soundness 40 points, traffic-flow analysis and insight 40, clarity and documentation 20. Private leaderboard score breaks final-score ties.

| Award | Amount |
|---|---:|
| Gold | USD 1,500 |
| Silver | USD 1,000 |
| Bronze | USD 500 |
| K–12 Student Award | USD 500 |

The K–12 award goes to the highest-ranked eligible team whose members are all currently enrolled K–12 students, who passes reproducibility verification and submits a report; it need not be in the top ten. A team receives only one award, so the K–12 award passes to the next eligible K–12 team if its first candidate wins Gold, Silver, or Bronze. Prizes remain subject to Competition Rules.

### Deadlines and delivery

| Date in 2026 | Milestone |
|---|---|
| October 25 | Registration email deadline for awards |
| November 6 | Final Kaggle submission deadline; private leaderboard revealed |
| November 10 | Code and PDF report deadline |
| November 11–17 | Reproducibility verification and committee review |
| November 18 | Winners and selected teams announced |
| November 20 | Camera-ready deadline for selected reports |
| December 14–17 | IEEE BigData 2026 presentations and awards in Phoenix, Arizona |

Email deadlines are **23:59 Anywhere on Earth (AoE)**. The Kaggle page is authoritative for the exact submission cutoff, team merger deadline, daily submission limit, and schedule changes.

Register by emailing **trafficflowbench@gmail.com** with subject **“Traffic Flow Bench Registration”**. Include Kaggle team name; each member’s Kaggle username, full name, affiliation, and email; and a primary contact. Registration affects award eligibility, not data access or leaderboard submissions.

Email the final package to the same address with subject **“Traffic Flow Bench Final Package – [Team Name]”**. Include a GitHub link or zip containing code, README, environment details, and one entry script that reproduces the selected final submission, plus a complete PDF report. There is no required report template or page limit. A short recorded presentation link is encouraged but optional.

Winning and selected teams may be invited to present at [IEEE BigData 2026](https://bigdataieee.org/BigData2026/), December 14–17 in Phoenix. Presenting is encouraged but not required for a prize; conference registration is separate. Other possible recognition includes certificates, registration support, and a challenge-report slot.

## Organizers, sponsors, and sources

The challenge is organized by the [IEEE Intelligent Transportation Systems Society Technical Committee on Travel Information and Traffic Management](https://ieee-itss.org/chapters-committees/traffic-travel-management/) with [RERITE, the Reproducible Research in Transportation Engineering Working Group](https://reriteworkinggroup.github.io/RERITE_website/). Both sponsor and co-organize it. Co-chairs are Xuesong (Simon) Zhou and Ruolin Li, with Cathy Wu and Yudai Honma. Organizers and designated judges perform eligibility, reproducibility, and private-score verification.

- [Kaggle competition and rules](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
- [Kaggle Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)
- [Public code, baselines, schemas, and scoring specification](https://github.com/jacky850/trafficflowbench-public), also checked out locally as [`trafficflowbench-public/`](trafficflowbench-public/)
- [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/)
- [OpenStreetMap copyright and license](https://www.openstreetmap.org/copyright)

The citation supplied by the Kaggle competition text is: Jinxi Wu and Xuesong (Simon) Zhou. *2026 IEEE Big Data-Traffic Flow Bench*. Kaggle, 2026. <https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench>. The repository’s separate [`CITATION.cff`](trafficflowbench-public/CITATION.cff) cites Jinxi Wu, *TrafficFlowBench: a four-task freeway traffic benchmark*, version 1.0, 2026-09-03, using the GitHub repository URL.
