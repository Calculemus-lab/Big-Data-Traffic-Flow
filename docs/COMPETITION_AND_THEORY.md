# Competition and traffic theory

TrafficFlowBench evaluates predictions of traffic conditions on synthetic
freeway networks. Participants use released measurements, road-network
attributes, and task templates to make predictions. The competition evaluates
four tasks and accepts one combined comma-separated values (CSV) submission.

This guide starts with the competition format, then defines the road network
and data before explaining the inputs, tasks, and scores. The
[release package reference](RELEASE_PACKAGE_REFERENCE.md) is a lookup for the
downloaded files and their columns. This repository's local solution interface
is described in the [Trafficbench guide](TRAFFICBENCH.md).

## Competition at a glance

The public release contains eleven months of synthetic traffic records at
five-minute intervals. The generator was calibrated using real detector data.
Each task asks for a different output:

| Task | Prediction or evaluation |
| --- | --- |
| 1. Traffic-state reconstruction | Speed and total flow for masked mainline measurements. |
| 2. Queue forecasting | Queued or clear status for each mainline link at six future times in a forecast window. |
| 3. Physical consistency | A score computed from the Task 1 speed and flow predictions. |
| 4. Origin-destination matrix estimation (ODME) | Nonnegative flow for each candidate route in a traffic scenario. |

The submission contains prediction rows for Tasks 1, 2, and 4. Task 3 has no
separate prediction rows: the evaluator calculates its score from the Task 1
values. The public leaderboard evaluates the validation period; the final
ranking uses the private period. The date table below defines those periods.

## Road networks, panels, and traffic measurements

A road network can be represented as a directed graph. An intersection or
network boundary is a **node**. A one-way road segment between two nodes is a
**directed link**. The freeway's **mainline** is its ordered sequence of
freeway links. On-ramps add traffic to the mainline; off-ramps remove traffic.

A **panel** is one directional freeway network. A **freeway family** groups
the two panels for the same freeway in the same district. The family ID gives
the district and freeway; the panel ID adds a direction suffix. The ten panels
in the public release are:

| Freeway family | Freeway | Directional panels |
| --- | --- | --- |
| `D7_I10` | District 7, I-10 | `D7_I10_E` eastbound; `D7_I10_W` westbound |
| `D7_I210` | District 7, I-210 | `D7_I210_E` eastbound; `D7_I210_W` westbound |
| `D7_I405` | District 7, I-405 | `D7_I405_N` northbound; `D7_I405_S` southbound |
| `D12_I5` | District 12, I-5 | `D12_I5_N` northbound; `D12_I5_S` southbound |
| `D12_I405` | District 12, I-405 | `D12_I405_N` northbound; `D12_I405_S` southbound |

The evaluator scores a panel and combines the two directional panel scores
within each freeway family. It then gives the five families equal weight. Task
2 has no windows for `D12_I405_N` or `D12_I405_S`; those panels remain in the
other three tasks. The
[release package reference](RELEASE_PACKAGE_REFERENCE.md#panels-and-date-splits)
lists the same panel identifiers with their release files.

A **detector station** measures traffic at one location on a link. A link may
have several detector stations, and station measurements may contain gaps.
Each record describes one five-minute interval.

For one link, speed $v$ is measured in kilometres per hour, total flow $F$ in
vehicles per hour across all lanes, and lane count $n$ in lanes. Per-lane flow
$q$ and per-lane density $k$ are

$$
q=\frac{F}{n},\qquad
k=\frac{q}{v}=\frac{F}{nv},\quad v>0.
$$

The traffic-flow identity $q=kv$ relates per-lane flow and density. Total
density across all lanes is $K=nk=F/v$. For a link of length $L$ kilometres,
the accumulation $N$ is the number of vehicles on that link:

$$
N=KL=\frac{F}{v}L.
$$

For example, a three-lane link with 3,600 vehicles per hour and a speed of
80 km/h has a per-lane flow of 1,200 vehicles per hour per lane and a
per-lane density of 15 vehicles per kilometre per lane.

## The synthetic release and its date splits

The public archive contains eleven months of synthetic traffic at
five-minute intervals. The archive includes unmasked and masked mainline
measurements for `train`. Validation and private contain masked mainline
measurements; their answer values are withheld from participants. Every date
uses Coordinated Universal Time (UTC). Each date interval includes its start
and excludes its end:

| Split | Start, included | End, excluded | Purpose and published mainline data |
| --- | --- | --- | --- |
| `train` | `2030-06-01` | `2031-03-01` | Model development and local cases. The archive contains masked and unmasked measurements. |
| `validation` | `2031-03-01` | `2031-04-01` | Public leaderboard. The archive contains masked measurements; answers are withheld. |
| `private` | `2031-04-01` | `2031-05-01` | Final evaluation. The archive contains masked measurements; answers are withheld. |

The archive contains complete unmasked mainline measurements for `train`, but
solution functions do not receive that raw table. Validation and private are
separate generated months with different demand draws and incident schedules.
A strong training result can therefore fail to transfer, and a validation
score cannot guarantee a private result.

## What a solution receives

Kaggle publishes input files and target templates. A **target template** lists
the rows a prediction must fill and has placeholders in its prediction
columns. The competition submission is a combined CSV made from the Task 1,
Task 2, and Task 4 prediction rows. It does not call a Python function.

For local development, this repository groups selected published inputs into
a **Release Slice**: static network tables, dated traffic records, and any
historical labels available for the selected panels and dates. Trafficbench
calls one Python function for each task with separate prediction rows (Tasks
1, 2, and 4). Each function receives the Release Slice, the target template
rows it must fill, and solution-specific settings. It returns predictions
for those same rows. Task 3 is calculated from the returned Task 1 values.
The exact function signature and prediction checks are in the
[Trafficbench function contract](TRAFFICBENCH.md#solution-function-contract).

When local answer values exist, Trafficbench keeps them in an **answer
table** outside the function arguments. After the function returns, the runner
compares predictions with the answer table. The function can see target row
identifiers and zero placeholders, but never the requested answer values.
Historical labels are separate from those requested answers.

Each Release Slice has one panel record for every selected directional
network. Static network tables are grouped together; dated tables are grouped
by panel and split. A solution can read the tables needed for its task:

| Release Slice field | Information available to a solution | Published file schema |
| --- | --- | --- |
| `network.links` | Mainline link lengths, lane counts, free-flow speeds, and capacities. | [Mainline link attributes](RELEASE_PACKAGE_REFERENCE.md#mainline-link-attributes) |
| `network.fd_parameters` | Link attributes and critical and jam densities used by the fundamental diagram, which describes feasible flow at each density. | [Fundamental-diagram parameters](RELEASE_PACKAGE_REFERENCE.md#fundamental-diagram-parameters) |
| `network.lwr_mainline_topology` | Mainline link connections and boundary links used for vehicle conservation. | [Mainline connectivity](RELEASE_PACKAGE_REFERENCE.md#mainline-connectivity) |
| `network.ramp_attachment_map`, `network.synthetic_ramp_attachment_map` | The mainline link associated with each ramp identifier. | [Ramp attachment maps](RELEASE_PACKAGE_REFERENCE.md#ramp-attachment-map) and [synthetic ramp attachment map](RELEASE_PACKAGE_REFERENCE.md#synthetic-ramp-attachment-map) |
| `network.path_set`, `network.path_link_incidence` | Candidate routes and the mainline links each route uses. | [Candidate path set](RELEASE_PACKAGE_REFERENCE.md#candidate-path-set) and [path-link incidence](RELEASE_PACKAGE_REFERENCE.md#path-link-incidence) |
| `network.additional_network_tables` | Other static network tables, keyed by their filename without `.csv`. | Listed in [static network tables](RELEASE_PACKAGE_REFERENCE.md#static-network-tables). |
| `masked_mainline_states[split]` | Published mainline measurements, including the masked train view. | [Mainline state records](RELEASE_PACKAGE_REFERENCE.md#mainline-state-records) |
| `ramp_states[split]` | Published ramp measurements for visible dates. | [Ramp state records](RELEASE_PACKAGE_REFERENCE.md#ramp-state-records) |
| `queue_history[split]`, `queue_window_index[split]` | Task 2 history measurements, forecast times, and window conditions. | [Queue-window metadata](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-metadata) and [queue-window history](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-history) |
| `link_counts[split]`, `weak_prior[split]` | Observed mainline counts and starting path-flow estimates for each Task 4 scenario. These scenarios are keyed by split, not by date. | [Link-count scenario](RELEASE_PACKAGE_REFERENCE.md#task-4-link-count-scenario) and [weak-prior scenario](RELEASE_PACKAGE_REFERENCE.md#task-4-weak-prior-scenario) |
| `historical_task_labels.task1_state_answers` | Released Task 1 speed and flow values for eligible historical target rows. | [Task 1 state target template](RELEASE_PACKAGE_REFERENCE.md#task-1-state-target-template) |
| `historical_task_labels.task2_queue_proxy_labels` | Local queue labels calculated from historical train speeds because official Task 2 answers are not released. | Columns match the [queue target template](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-target-template). These are estimates, not official answers. |

The field names above belong to this repository's local Python interface. The
downloaded files use their published column names. Trafficbench converts
timestamp columns to UTC datetimes and groups dated data by panel and split.
The [release package reference](RELEASE_PACKAGE_REFERENCE.md) documents the
on-disk files; [Data passed to the solution](TRAFFICBENCH.md#data-passed-to-the-solution)
documents the in-memory tables.

## Task 1: traffic-state reconstruction

Task 1 predicts speed and total flow for masked mainline station-time records.
Its target template lists the panel, timestamp, station, link, and mask regime
for each row, with speed and flow as the values to predict. The function can
use masked mainline records, network link attributes, and the historical Task
1 labels in the Release Slice. The published target columns are listed in the
[Task 1 template schema](RELEASE_PACKAGE_REFERENCE.md#task-1-state-target-template).
The release applies its masks before participants receive the data. A null
measurement does not by itself mean that the cell is a Task 1 target. Some nulls
are natural detector gaps, and additional measurements are hidden around Task 2
forecast windows. The Task 1 target rows identify which cells are scored. A
cell is eligible for scoring when both required measurements, speed and total
flow, are present and detector coverage is at least 75%.

Each date is assigned one mask regime, which sets the share of eligible speed
and flow cells that are hidden:

| Regime | Eligible cells selected for masking |
| ------ | ----------------------------------: |
| `R1`   |                                 20% |
| `R2`   |                                 30% |
| `R3`   |                                 50% |

![Task 1 masking example for one weekday across three directional panels. The first panel shows natural gaps, while the other panels show the R1, R2, and R3 masks.](../official_competition_repo/figures/task1_what_is_asked.png)

_The official example shows how the three regimes hide increasing shares of eligible measurements._

For each regime $r$, the evaluator computes a speed root mean squared error
(RMSE) and a per-lane flow RMSE:

$$
S_{\text{speed}}(r)=\max\left(0,1-\frac{\operatorname{RMSE}_{\text{speed}}(r)}{25}\right),
\qquad
S_{\text{flow}}(r)=\max\left(0,1-\frac{\operatorname{RMSE}_{\text{flow per lane}}(r)}{600}\right),
$$

$$
S_{\text{state}}(r)=0.54S_{\text{speed}}(r)+0.46S_{\text{flow}}(r),
\qquad
S_{\text{state}}=\frac{1}{3}\sum_{r\in\{R1,R2,R3\}}S_{\text{state}}(r).
$$

The RMSE scales are km/h for speed and vehicles per hour per lane for flow.
Both predicted and true total flow are divided by the link's lane count before
the flow error is calculated. The [mask specification](../official_competition_repo/docs/MASK_SPEC.md)
explains how the target templates identify scored cells, so not every visible
null is a prediction target.

The official [historical-mean baseline](../official_competition_repo/src/task1/baseline_task1_historical_mean.py)
uses a weekday and time-of-day profile for each link. An optional
[structural Kalman baseline](../official_competition_repo/src/task1/baseline_task1_structural_kf.py)
smooths same-day deviations from that profile.

## Task 2: short-term queue forecasting

Each official queue window has a 60-minute history ending at forecast origin
$T$. It asks for queue predictions on every mainline link at six future
five-minute times, from $T+5$ through $T+30$. The target template identifies
each window, link, and forecast timestamp. The function can use the published
queue history and window metadata, masked mainline records, network link
attributes, and historical queue proxy labels. See the
[Task 2 target schema](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-target-template),
[window metadata schema](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-metadata),
and [window history schema](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-history).

A link is queued when its speed is at or below 60% of that link's free-flow
speed:

$$
\operatorname{queue}(e,t)=
\begin{cases}
1,&v(e,t)\leq0.60v_f(e),\\
0,&v(e,t)>0.60v_f(e).
\end{cases}
$$

The official windows have two conditions. A `queue_onset` window has no
established queue in its history and at least one queued link in its forecast.
A `queue_ongoing` window already has an established queue in its history and
also has a queue in its forecast. A queue is established when the same link
is queued at two or more history timestamps. Each included panel and split
has five windows of each condition.
`D12_I405_N` and `D12_I405_S` have no Task 2 windows.

The evaluator calculates space-time intersection over union (IoU) for each
window. IoU is the number of link-time cells predicted and truly queued divided
by the number of cells queued in either the prediction or the answer. It
averages window scores within each condition and gives both conditions equal
weight:

$$
\operatorname{IoU}(w)=
\frac{|Q_{\text{pred}}(w)\cap Q_{\text{true}}(w)|}
{|Q_{\text{pred}}(w)\cup Q_{\text{true}}(w)|}.
$$

Only forecast cells with at least 75% detector coverage and the required
measurements count. Official queue labels use the underlying traffic state to
avoid queue changes caused by detector measurement error. The public release
does not provide those labels, so its published
[Task 2 evaluator](../official_competition_repo/src/task2/score_task2.py)
cannot calculate a label-based score from the public files alone. The official
[persistence baseline](../official_competition_repo/src/task2/build_task2_persistence_submission.py)
repeats the queue status visible at $T$ for all six forecast steps.

## Task 3: physical consistency

Task 3 scores the physical consistency of the Task 1 speed and flow
predictions. It has no separate target table or prediction function. The
evaluator checks a **Fundamental Diagram (FD)** and whether vehicle
conservation follows the Lighthill-Whitham-Richards (LWR) equation.
Trafficbench reports local diagnostics for Task 1 predictions using the
network's fundamental-diagram parameters. These diagnostics measure flow
mismatch, low-flow predictions, and negative values. They do not calculate
vehicle conservation or the full official Task 3 score. The relevant published
values are in the [fundamental-diagram parameter table](RELEASE_PACKAGE_REFERENCE.md#fundamental-diagram-parameters).

### Fundamental diagram

An FD gives the feasible flow at each traffic density. The triangular FD has a
free-flow branch with slope $v_f$ and a congested branch that reaches zero at
the jam density $k_{\text{jam}}$:

$$
q(k)=
\begin{cases}
v_f k,&0\leq k\leq k_{\text{crit}},\\
w(k_{\text{jam}}-k),&k_{\text{crit}}<k\leq k_{\text{jam}}.
\end{cases}
$$

Here $k_{\text{crit}}$ is density at capacity, and $w$ is the magnitude of the
upstream-moving congestion-wave speed. The maximum flow occurs at the critical
density: $C_{\text{lane}}=v_fk_{\text{crit}}$. The public network tables store
some density and capacity values as totals across lanes. Convert them to
per-lane values before using this diagram. Exact field definitions are in the
[release package reference](RELEASE_PACKAGE_REFERENCE.md).

![Triangular fundamental diagram with free-flow and congested branches, capacity flow, critical density, and jam density.](images/triangular_fundamental_diagram.png)

_The diagram labels capacity flow as $q_{\mathrm{cap}}$ and critical density as $k_{\mathrm{cap}}$. These correspond to $C_{\mathrm{lane}}$ and $k_{\mathrm{crit}}$ in the equation above._

For each target, the evaluator derives density from submitted speed and flow,
then compares the submitted per-lane flow with the FD flow at that density.
The FD score is zero if more than 20% of a panel-regime's target cells have
submitted total flow below 50 vehicles per hour. Otherwise, it is one minus
the normalized absolute difference between submitted per-lane flow and FD
flow. For vehicle conservation, net flow $F_{\text{net}}$ is the amount
entering the link from its upstream mainline link and on-ramps, minus the
amount leaving toward its downstream mainline link and off-ramps:

$$
F_{\text{net}}=F_{\text{in}}+F_{\text{on}}-F_{\text{out}}-F_{\text{off}}.
$$

The conservation score is one minus the normalized absolute conservation
residual. Both scores are floored at zero. In the equations, $\varepsilon$ is
a small positive constant that prevents division by zero. The FD sum covers
scored target cells. The LWR sum covers scored transitions between consecutive
timestamps on the same mainline link:

$$
S_{\text{FD}}=\max\left(0,1-\frac{\sum|q_{\text{lane}}-q_{\text{FD}}(k_{\text{lane}})|}{\sum|q_{\text{lane}}|+\varepsilon}\right),
\qquad
S_{\text{LWR}}=\max\left(0,1-\frac{\sum|\Delta N-\Delta tF_{\text{net}}|}{\sum|\Delta tF_{\text{net}}|+\varepsilon}\right).
$$

In the conservation equation, $\Delta t$ is elapsed time in hours. Multiplying
it by flow in vehicles per hour gives the number of vehicles that entered or
left during that interval.

The total physics score weights conservation twice as much as the FD score:

$$
S_{\text{physics}}=\frac{1}{3}S_{\text{FD}}+\frac{2}{3}S_{\text{LWR}}.
$$

### Conservation of vehicles: Lighthill-Whitham-Richards (LWR) equation

During a five-minute interval, the change in the number of vehicles on a link
must equal total incoming flow minus total outgoing flow. Mainline flows and
ramp flows use total vehicles per hour across all lanes:

$$
N_e(t+\Delta t)-N_e(t)=
\Delta t\left(
F_{\text{in}}(t)+F_{\text{on}}(t)-F_{\text{out}}(t)-F_{\text{off}}(t)
\right).
$$

An unattached ramp contributes zero flow. A missing measurement from an
attached ramp is missing evidence, so the affected transition is not scored.
The public release does not include the organizer's boundary flows: traffic
entering or leaving the panel at its external ends. The organizer also
publishes a local scoring script, but it estimates those flows from network
topology and submitted values. Its conservation score can differ substantially
from the leaderboard score and is not a useful leaderboard proxy. The
competition evaluator has the boundary flows. See the official
[scoring specification](../official_competition_repo/docs/SCORING_SPEC.md)
for exact evaluator equations. Task 3 has no separate prediction baseline
because it evaluates whichever Task 1 values are submitted.

## Task 4: origin-destination path-flow estimation

Task 4 estimates how much traffic follows each candidate path from an origin
zone to a destination zone. This is an **origin-destination matrix estimation
(ODME)** problem. Its target rows ask for path flows in an evening-peak
scenario rather than speed, flow, or queue values at five-minute timestamps.
Each panel and split has a separate scenario. Its `departure_time` is a label
such as `SYN_PM_TRAIN`, not a timestamp.

Task 4 receives the same **Release Slice** as the other prediction tasks, so
its solution can also use other published tables when they are useful. Each
scenario provides candidate paths, the mainline links used by each path,
observed link counts $\mathbf c$, and a **Weak Prior** $\mathbf b$, an initial
flow estimate for each candidate path. A path-flow vector
$\hat{\mathbf f}$ has one value for each candidate path. The path-link
incidence matrix $A$ has one row for each mainline link and one column for
each candidate path. Its entry $A_{mp}$ is 1 when path $p$ uses link $m$, and 0
otherwise. The product $A\hat{\mathbf f}$ gives the link counts implied by the
estimated path flows. In the Release Slice, these inputs are available as
`network.path_set`, `network.path_link_incidence`, `link_counts[split]`, and
`weak_prior[split]`. The target template asks for one flow per candidate path.
See the [path-flow target schema](RELEASE_PACKAGE_REFERENCE.md#task-4-path-flow-target-template).

The Weak Prior guides the estimate, but it is not the hidden answer. Because
several paths can share links, different path-flow estimates can produce
similar link counts. The score therefore compares submitted flows with hidden
path flows as well as checking how well the estimates match the released
counts and prior.

The official score combines four parts:

$$
S_{\text{ODME}}=0.45S_{\text{od}}+0.25S_{\text{link}}+
0.15S_{\text{dev}}+0.15S_{\text{attr}}.
$$

**Path-flow accuracy** compares submitted path flows $\hat f_p$ with hidden
reference flows $f_p^*$:

$$
S_{\text{od}}=\max\left(0,
1-\frac{\sum_p|\hat f_p-f_p^*|}{\max(\sum_p f_p^*,\varepsilon)}\right).
$$

**Link-count fit** compares counts implied by the submitted paths with observed
link counts:

$$
S_{\text{link}}=\max\left(0,
1-\frac{\sum_m|(A\hat{\mathbf f})_m-c_m|}
{\max(\sum_m c_m,\varepsilon)}\right).
$$

In these first two formulas, $\varepsilon$ is a small positive constant that
prevents division by zero.

**Prior deviation** compares how far the submission and reference are from the
weak prior. It gives a higher score when the submission's distance from the
prior matches the reference distance:

$$
D_{\text{sub}}=\|\hat{\mathbf f}-\mathbf b\|_1,
\qquad
D_{\text{ref}}=\|\mathbf f^*-\mathbf b\|_1,
\qquad
S_{\text{dev}}=\exp\left(-\left|\frac{D_{\text{sub}}}{D_{\text{ref}}}-1\right|\right).
$$

The destination-attraction component compares how total path flow is
distributed across destination zones. For destination $d$, let $P_d$ be the
paths ending there. The predicted share is

$$
\hat a_d=\frac{\sum_{p\in P_d}\hat f_p}{\sum_p\hat f_p}.
$$

The evaluator calculates the same distribution $a_d^*$ from hidden reference
flows and scores the total absolute difference between the two distributions:

$$
S_{\text{attr}}=\max\left(0,
1-\frac{1}{2}\sum_d|\hat a_d-a_d^*|\right).
$$

The public release has no reference path-flow answers. The published
[Task 4 scorer](../official_competition_repo/src/task4/score_task4.py) can
calculate the link-count component $S_{\text{link}}$, which is one quarter of
the Task 4 score. It cannot score path-flow accuracy, prior deviation, or
destination attraction without the hidden reference flows. The official
[regularised ODME baseline](../official_competition_repo/src/task4/build_task4_odme_artifacts.py)
fits counts while penalising deviation from the weak prior.

## Competition score and leaderboards

The evaluator combines the four task scores using these weights:

$$
S_{\text{total}}=
0.35S_{\text{state}}+
0.30S_{\text{queue}}+
0.15S_{\text{physics}}+
0.20S_{\text{ODME}}.
$$

For each task, the evaluator first combines the included directional panels
within each freeway family, then gives the five families equal weight. Task 2
has targets in four families because both `D12_I405` directions are excluded;
the other tasks cover all five families. If a required task output is missing,
that task contributes zero to the combined score. Task 3 contributes its
physics score from the Task 1 values and requires no separate prediction rows.

The public leaderboard uses validation answers. Private answers are used in
the final evaluation and determine the competition ranking. Validation and
private are different generated months, so a validation result is evidence
about performance on unseen data but does not guarantee a private result. The
organizers' [task connection and suggested order](../official_competition_repo/README.md#how-the-tasks-connect)
explains why they recommend beginning with Task 1.

## Official references

- [Competition rules and leaderboard](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
- [Scoring specification](../official_competition_repo/docs/SCORING_SPEC.md)
- [Mask specification](../official_competition_repo/docs/MASK_SPEC.md)
- [Submission schemas](../official_competition_repo/docs/SUBMISSION_SCHEMAS.md)
- [Release data guide](../official_competition_repo/docs/DATA.md)
