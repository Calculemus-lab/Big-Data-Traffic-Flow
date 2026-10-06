# Competition and traffic theory

TrafficFlowBench is a competition for predicting traffic conditions on
synthetic freeway networks. Its tasks use related traffic data but ask for
different outputs and evaluate different parts of a solution. This guide
explains the tasks, the data available for each one, and the scoring rules.

The [release package reference](RELEASE_PACKAGE_REFERENCE.md) documents the
original downloaded files and their columns. **Trafficbench** is this
repository's local runner for calling solution functions and assembling their
prediction files. The [Trafficbench guide](TRAFFICBENCH.md) explains that
interface and its commands.

## Competition at a glance

The public release contains synthetic records at five-minute intervals for
ten directional freeway panels, grouped into five freeway families. Its
traffic generator was calibrated using real detector data. The release divides
dates into three data splits:

| Split | Dates, start included and end excluded | Purpose | Mainline information available through Trafficbench |
| --- | --- | --- | --- |
| `train` | 2030-06-01 to 2031-03-01 | Model development and local cases | Masked measurements and selected labels from an earlier train interval. |
| `validation` | 2031-03-01 to 2031-04-01 | Public leaderboard | Masked measurements. Answers for these dates are withheld. |
| `private` | 2031-04-01 to 2031-05-01 | Final ranking | Masked measurements. Answers for these dates are withheld. |

The downloaded archive also contains unmasked mainline measurements for
`train`. Trafficbench does not pass that full table to solution functions. It
passes the published masked view as input and provides selected historical
Task 1 answers and locally derived Task 2 labels in
`historical_task_labels`. This makes those labels available without passing
the unmasked measurement table to a solution. The split boundaries come from
the release configuration and use Coordinated Universal Time (UTC).

Validation and private are separate generated months with different demand
draws and incident schedules. A strong train score does not guarantee transfer
to either month, and validation performance does not guarantee a private
ranking. The public leaderboard uses validation, and the private leaderboard
determines the final competition ranking. Exact split dates are listed in the
[release package reference](RELEASE_PACKAGE_REFERENCE.md#panels-and-date-splits).

| Task                           | Prediction or score                                |
| ------------------------------ | -------------------------------------------------- |
| 1. State reconstruction        | Mainline speed and total flow at masked targets    |
| 2. Queue forecasting           | Queued or clear state per link at six future times |
| 3. Physical consistency        | Physics score calculated from Task 1 predictions   |
| 4. Origin-destination path-flow estimation (ODME) | Nonnegative traffic flow on each candidate path |

The total score is

$$
S_{\text{total}}=
0.35S_{\text{state}}+
0.30S_{\text{queue}}+
0.15S_{\text{physics}}+
0.20S_{\text{ODME}}.
$$

Within each freeway family, the evaluator averages scores from the included
directions. It then gives each family equal weight. Task 2 has targets in four
families because both `D12_I405` panels are excluded. The other tasks cover
all five families. If a required task output is missing, that task contributes
zero. Task 3 has no separate prediction rows because it evaluates the Task 1
values. The organizers'
[task connection and suggested order](../official_competition_repo/README.md#how-the-tasks-connect)
explains why they recommend starting with Task 1.

## Traffic fundamentals

### Panels, links, and detectors

A **Panel** is one directional freeway network. Its mainline is a sequence of
directed road links. On-ramps add traffic to the mainline, and off-ramps remove
traffic. A detector station measures one location on a link. Several stations
can measure the same link.

For one link, speed $v$ is measured in kilometres per hour, total flow $F$ in
vehicles per hour across all lanes, and lane count $n$ in lanes. Per-lane flow
$q$ and per-lane density $k$ are

$$
q=\frac{F}{n},\qquad
k=\frac{q}{v}=\frac{F}{nv}.
$$

The identity $q=kv$ relates these quantities. Total density across all lanes
is $K=nk=F/v$. For a link of length $L$ kilometres, its accumulation $N$ is the
number of vehicles on the link:

$$
N=KL=\frac{F}{v}L.
$$

For example, a three-lane link with 3,600 vehicles per hour and a speed of
80 km/h has a per-lane flow of 1,200 vehicles per hour per lane and a per-lane
density of 15 vehicles per kilometre per lane.

## What a solution receives

Every local task function receives the same **Release Slice**
([`ReleasePackageSlice`](../trafficbench/contracts.py#L604)) and target
templates grouped by panel and split. Each function returns predictions in the
same panel-then-split layout.

The release slice records four dates. `history_start_date` begins the visible
inputs and historical labels. `history_end_date` ends the historical labels.
`prediction_start_date` and `prediction_end_date` bound the requested targets.
The history interval must fall inside `train`. Every interval includes its
start and excludes its end. Dated input tables cover
`[history_start_date, prediction_end_date)`, so released observations between
the historical labels and the targets remain available as input data.

Its `panel_slices_by_panel` field contains one
[`PanelReleaseSlice`](../trafficbench/contracts.py#L566) for each selected
panel. Each panel slice also has a `family_id` for score aggregation. It groups
static network tables together and stores dated tables in maps keyed by split.
In code, for example, a solution can read
`release_slice.panel_slices_by_panel[panel].masked_mainline_states[split]`.

| Release Slice field | What the table contains | Published file schema |
| --- | --- | --- |
| `network.links` | Mainline link lengths, lane counts, free-flow speeds, and capacities. | [Mainline link attributes](RELEASE_PACKAGE_REFERENCE.md#mainline-link-attributes) |
| `network.fd_parameters` | Link attributes and critical and jam densities for the fundamental diagram. | [Fundamental-diagram parameters](RELEASE_PACKAGE_REFERENCE.md#fundamental-diagram-parameters) |
| `network.lwr_mainline_topology` | Mainline link connections and boundary links used for vehicle conservation. | [Mainline connectivity](RELEASE_PACKAGE_REFERENCE.md#mainline-connectivity) |
| `network.ramp_attachment_map`, `network.synthetic_ramp_attachment_map` | The mainline link associated with each ramp ID. | [Ramp attachment maps](RELEASE_PACKAGE_REFERENCE.md#ramp-attachment-map) and [synthetic ramp attachment map](RELEASE_PACKAGE_REFERENCE.md#synthetic-ramp-attachment-map) |
| `network.path_set`, `network.path_link_incidence` | Candidate paths and the mainline links each path uses. | [Candidate path set](RELEASE_PACKAGE_REFERENCE.md#candidate-path-set) and [path-link incidence](RELEASE_PACKAGE_REFERENCE.md#path-link-incidence) |
| `network.additional_network_tables` | Any other network CSV tables, keyed by their file name without the `.csv` suffix. | Listed in the [static network tables](RELEASE_PACKAGE_REFERENCE.md#static-network-tables) section. |
| `masked_mainline_states[split]` | Published mainline measurements, including the masked train view. | [Mainline state records](RELEASE_PACKAGE_REFERENCE.md#mainline-state-records) |
| `ramp_states[split]` | Published ramp measurements for the visible dates. | [Ramp state records](RELEASE_PACKAGE_REFERENCE.md#ramp-state-records) |
| `queue_history[split]`, `queue_window_index[split]` | Released Task 2 history measurements and the times and conditions of forecast windows. | [Queue-window metadata](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-metadata) and [queue-window history](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-window-history) |
| `link_counts[split]`, `weak_prior[split]` | Observed mainline counts and starting path-flow estimates for each Task 4 scenario. These scenarios are keyed by split, not by date. | [Link-count scenario](RELEASE_PACKAGE_REFERENCE.md#task-4-link-count-scenario) and [weak-prior scenario](RELEASE_PACKAGE_REFERENCE.md#task-4-weak-prior-scenario) |
| [`historical_task_labels.task1_state_answers`](../trafficbench/contracts.py#L562) | Released Task 1 speed and flow values for eligible historical target rows. | [Task 1 state target template](RELEASE_PACKAGE_REFERENCE.md#task-1-state-target-template) |
| [`historical_task_labels.task2_queue_proxy_labels`](../trafficbench/contracts.py#L563) | Local queue labels calculated from historical train speeds because official Task 2 answers are not released. | Columns match the [queue target template](RELEASE_PACKAGE_REFERENCE.md#task-2-queue-target-template). These labels are local estimates, not official answers. |

The linked release schemas describe columns in files on disk. Trafficbench groups
dated tables by panel and split and reads timestamps as Coordinated Universal
Time (UTC) datetimes. The in-memory [`StateFrame`](../trafficbench/table_types.py#L260),
[`QueueFrame`](../trafficbench/table_types.py#L277), and
[`OdmeFrame`](../trafficbench/table_types.py#L296) types document the columns
available in each task table. Their usage is shown in the
[Trafficbench function contract](TRAFFICBENCH.md#solution-function-contract).

The second argument is named `target_templates_by_panel_and_split`. It contains
the rows to predict, with fields that identify each row and zero placeholders
in the prediction columns. Its shape is `panel -> split -> table`, matching
the prediction mapping returned by the function.

For requested target rows, the solution receives the row identifiers and zero
placeholders, while the runner keeps the true answer values for scoring. The
separate `historical_task_labels` field contains earlier Task 1 answers and
locally calculated Task 2 labels. No unmasked mainline table or historical
Task 4 path-flow answers are passed to a solution.

All three functions receive this same set of panel data. A solution can choose
which tables to read. Tables are Polars `LazyFrame` query plans, so a function
that does not collect Task 4 tables does not read them into memory. The
[Trafficbench guide](TRAFFICBENCH.md) describes the command-line workflow and
how local cases are prepared.

The slice also carries a `source_fingerprint` that identifies the release
files, a `parameters` mapping for solution settings, and a `seed` that makes
randomized choices repeatable.

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
cell is eligible for scoring when all required measurements are present and
detector coverage is at least 75%.

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

The evaluator calculates space-time intersection over union (IoU) separately
for every window. It then averages window scores within each condition and
gives both conditions equal weight:

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
The local diagnostic uses the Task 1 predictions together with network FD
parameters, mainline topology, and ramp observations. The relevant published
tables are the [FD parameters](RELEASE_PACKAGE_REFERENCE.md#fundamental-diagram-parameters),
[mainline connectivity](RELEASE_PACKAGE_REFERENCE.md#mainline-connectivity),
and [ramp state records](RELEASE_PACKAGE_REFERENCE.md#ramp-state-records).

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
flow. Let $F_{\text{net}}=F_{\text{in}}+F_{\text{on}}-F_{\text{out}}-F_{\text{off}}$.
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
The public release does not include the organizer's boundary flows. These are
the traffic flows entering or leaving the panel at its external ends. The
published local scorer estimates them from network topology and submitted
values. Its conservation score can differ substantially from the leaderboard
score and is not a useful leaderboard proxy. The organizer's
evaluator has the boundary flows. See the official
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

The score combines four parts:

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

**Destination Attraction** is the share of total predicted path flow that
ends in each destination zone. For destination $d$, let $P_d$ be the paths
ending there. The predicted share is

$$
\hat a_d=\frac{\sum_{p\in P_d}\hat f_p}{\sum_p\hat f_p}.
$$

The evaluator calculates the same distribution $a_d^*$ from hidden reference
flows and scores the total absolute difference between the two distributions:

$$
S_{\text{attr}}=\max\left(0,
1-\frac{1}{2}\sum_d|\hat a_d-a_d^*|\right).
$$

The public release has no reference path-flow answers. Its published
[Task 4 scorer](../official_competition_repo/src/task4/score_task4.py) can
calculate $S_{\text{link}}$, which is one quarter of the Task 4 score. It
cannot score path-flow accuracy, prior deviation, or destination attraction
without the hidden reference flows. The official
[regularised ODME baseline](../official_competition_repo/src/task4/build_task4_odme_artifacts.py)
fits counts while penalising deviation from the weak prior.

## Official references

- [Competition rules and leaderboard](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
- [Scoring specification](../official_competition_repo/docs/SCORING_SPEC.md)
- [Mask specification](../official_competition_repo/docs/MASK_SPEC.md)
- [Submission schemas](../official_competition_repo/docs/SUBMISSION_SCHEMAS.md)
- [Release data guide](../official_competition_repo/docs/DATA.md)
