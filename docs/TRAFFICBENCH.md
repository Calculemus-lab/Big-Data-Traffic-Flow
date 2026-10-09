# Trafficbench: local cases and final predictions

The competition evaluates one combined CSV submission. This repository's
**Trafficbench** package helps a team develop the Task 1, Task 2, and Task 4
prediction rows, compare solutions on local cases, and assemble the final CSV.
Task 3 has no separate prediction rows; its score is calculated from Task 1.
Task 4 is origin-destination matrix estimation (ODME): it predicts how much
traffic follows each candidate route in a scenario.

There are two prediction workflows. `bench experiment` runs a solution on a
prepared train case whose available answer values let Trafficbench calculate
local metrics. `bench run` fills rows from the competition templates for
validation or private evaluation; the public release does not include those
target answers. The guides introduce these workflows in that order, then show
how to assemble and upload a submission.

For competition task definitions, see
[competition and traffic theory](COMPETITION_AND_THEORY.md). For the downloaded
files and their columns, see the [release package reference](RELEASE_PACKAGE_REFERENCE.md).
Obtain the release as described in [download the data](GET_DATA.md).

## Date splits and available answers

The official release has one training split followed by two evaluation splits.
The intervals use Coordinated Universal Time (UTC), include their start date,
and exclude their end date. The split durations are nine months for train and
one month each for validation and private:

```text
2030-06-01                           2031-03-01               2031-04-01            2031-05-01
    |---------- train (9 months) ----------|- validation (1 month) -|- private (1 month) -|
```

| Split | Start, included | End, excluded | Published mainline data and use |
| --- | --- | --- | --- |
| `train` | `2030-06-01` | `2031-03-01` | Masked and unmasked measurements; used to develop solutions and prepare local cases. |
| `validation` | `2031-03-01` | `2031-04-01` | Masked measurements; answers are withheld and this split determines the public leaderboard. |
| `private` | `2031-04-01` | `2031-05-01` | Masked measurements; answers are withheld and this split determines the final ranking. |

The public archive contains unmasked mainline measurements for `train`. A
prediction function does not receive that raw table. It receives the published
masked view and, for dates selected as history, only the historical labels that
the local interface is allowed to provide. Validation and private are separate
generated months with different demand draws and incident schedules, so a
validation result does not guarantee the private result.

For example, this diagram shows a default `bench run` that requests private
targets. The top line shows the official date splits. The lower lines show
what the local runner makes visible and which rows the function must predict:

```text
official split:       |----------------- train ----------------|- validation -|-------- private --------|

released input:       |-------------------- masked observations and target views ----------------------|
historical labels:    |- Task 1 answers + Task 2 proxy labels -|
requested targets:                                                                  |- private rows -|
```

For this private run, the full train period is the default history. Released
validation observations remain available as inputs before the private targets.
The runner withholds the requested answer values, even when an answer file is
provided for scoring.

The answer source depends on the task. This matters because local metrics do
not all compare predictions with the same kind of answer:

| Task | Answer values in the public release | Reference used by a local score |
| --- | --- | --- |
| 1. Traffic-state reconstruction | Unmasked mainline measurements are released for train; validation and private answers are withheld. | Exact train values can score eligible Task 1 target rows. |
| 2. Queue forecasting | Official queue answers are withheld for every split. | On train, Trafficbench derives proxy labels from measured speeds. They can differ from the official labels. |
| 3. Physical consistency | There is no separate target table. The official evaluator calculates this score from Task 1 predictions and organizer-held boundary flows. | Trafficbench reports local diagnostics but cannot calculate the full official physics score. |
| 4. ODME path-flow estimation | Link counts and an initial path-flow estimate (the **Weak Prior**) are released, but path-flow answers are not released for any split. | A released-count case measures count fit. Separate generated cases have generated path-flow answers for local checks. |

The Task 2 proxy labels are local estimates, not competition answers. Likewise,
a score on generated Task 4 cases measures behavior on those generated cases,
not accuracy against hidden competition path flows.

## Choose history and prediction periods

Trafficbench divides a run's date range into a **history period** and a
**prediction period**. The history period supplies labels that may be used as
features. The prediction period selects the target rows that the solution
function must fill. Four dates define these periods; each start is included and
each end is excluded:

| Date field | Meaning |
| --- | --- |
| `history_start_date` | First date of visible mainline and ramp input data and historical labels. |
| `history_end_date` | First date after the history period. No later date adds historical labels. |
| `prediction_start_date` | First date of the requested target period; it must be on or after `history_end_date`. |
| `prediction_end_date` | First date after the targets and the exclusive end of visible date-based inputs. |

The history period must lie within `train`. For `bench prepare`, the prediction
period must also lie entirely within `train`; that is how Trafficbench has
answer values for local Task 1 and Task 2 scoring. A `bench run` prediction
period can instead extend from a later train date through validation and
private. Released observations in a gap between history and prediction remain
available as inputs, but do not add historical labels.

Read the next diagram from left to right. The first bar is the official train
split. The second bar marks dates selected as history and dates selected as
prediction. The final bar separates the observations the solution may read
from the target rows it must return:

```text
split timeline:    |-------------------- train -----------------------------|------- later train/validation/private ----------|

run dates:    history_start_date                        history_end_date        prediction_start_date          prediction_end_date
                      |----------------------------------------|----- (optional gap) -----|-------------------------------|

given information:    |-------------------------------- released inputs ------------------------------------|
                      |- Task 1 answers + Task 2 proxy labels -|                          |---- requested target rows ----|
```

For Task 4, dates select complete origin-destination scenarios by split. Each
scenario has its own link counts and Weak Prior. A prediction period selects
the full scenario for each split it intersects; it does not trim a scenario to
a subset of dates. Task 4 has no historical path-flow answers.

Use `--target-range validation`, `--target-range private`, or
`--target-range both` to request the published validation period, private
period, or both. Without custom dates, `bench run` uses the full train period
as history and requests both evaluation periods. A private target run includes
released validation observations as inputs. Each selected Task 2 window also
includes its full 60-minute history, even if that history begins before
`history_start_date`.

## Which local scores can a prepared case report?

The answer source limits what each local metric means. Use this table to
read a local result before preparing a case:

| Task | Local answer source | What the resulting metric measures |
| --- | --- | --- |
| 1. State reconstruction | Exact unmasked train measurements for eligible target rows. | Accuracy against the synthetic train values. |
| 2. Queue forecasting | Proxy labels calculated from train station speeds; official queue labels are not released. | Agreement with the local proxy labels, not with the hidden official queue labels. |
| 3. Physical consistency | No separate answer table is used. | Flow-density and value-range diagnostics; no full local version of the official physics score. |
| 4. Path-flow estimation | Released link counts for one case; generated path flows for three synthetic cases. | Count fit on released data and path-flow accuracy on generated examples only. |

## Prepare and score local train cases

`bench prepare` turns an earlier train interval and a later train interval into
a repeatable local case: the solution uses the earlier history, predicts the
later target rows, and Trafficbench scores the predictions afterward. It saves
the selected inputs, target rows, and answer tables separately. One **Prepared
Benchmark** contains one history period, one prediction period, and a selected
set of panels.

The next diagram zooms in on this local case. Both periods lie within official
`train`, so train answers are available to the scorer. Task 1 answers and Task
2 proxy labels stop at `history_end_date`. Masked observations after that date
remain available as inputs, while requested target answers stay with the
runner:

```text
split timeline:   |------------------------------------------------------train-------------------------------------------------|

run dates:    history_start_date                        history_end_date        prediction_start_date          prediction_end_date
                      |----------------------------------------|----- (optional gap) -----|-------------------------------|

given information:    |------------------------------------------------ traffic data -------------------------------------|
                      |- Task 1 answers + Task 2 proxy labels -|                          |---- requested target rows ----|
```

The prediction period is one continuous date range shared by all selected
panels. Prepare another case if you want to compare a different date interval.

Run this command from the repository root. `--data` points to the extracted
release. The four date options define the history and prediction periods shown
above. `--profile quick` selects one panel; `--profile full` selects all ten.
`--scheme config/cv_scheme.yaml` selects the default YAML settings for choosing
queue windows and generating Task 4 cases:

```sh
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-24 \
  --prediction-end-date 2030-12-01 \
  --profile quick --scheme config/cv_scheme.yaml \
  --output data/benchmarks/november
```

The default `quick` profile selects `D12_I5_N`; `full` selects all panels. Use
`--panel` to select panel IDs directly. `--output` chooses where the prepared
case is written; without it, output goes to `data/benchmarks/<profile>`.

`--scheme PATH` selects a YAML file that controls local case generation. If
the option is omitted, Trafficbench reads `config/cv_scheme.yaml` from the
repository root. The file records the random seed, Task 2 window count, spacing
and coverage rules, and Task 4 generated-case count and noise settings. Its
default selects five Task 2 windows for each condition, requires at least 70%
eligible link-time cells in both the 60-minute history and the six-step forecast
when choosing a window, spaces windows in one condition at least 360 minutes
apart, and creates three generated Task 4 cases. This 70% rule selects complete
windows; it does not change the separate 75% detector-coverage rule for an
individual queue label to be eligible for scoring. Keep the same scheme, dates,
and panels when comparing solutions.

Preparation saves selected release inputs, target rows, and answer tables
separately. The [`bench experiment` runner](../trafficbench/runner.py#L614)
uses the same function interface and prediction checks as `bench run`.

### Task 1 local cases

The target rows come from the official train state template. Preparation looks
up their true values in unmasked train measurements. Those later answer values
stay outside the solution call. The Release Slice contains the earlier Task 1
historical answers and published masked measurements through the target
interval. A local Task 1 score therefore measures predictions against the
answer values for those eligible train rows. These are exact answers for the
synthetic benchmark, not measurements of real-world traffic.

### Task 2 local cases

The official queue answers are withheld, so local cases use proxy labels from
unmasked train speed measurements. For each link and timestamp, Trafficbench
averages the station speeds. It keeps a label only when every station has a
speed measurement and at least 75% observation coverage. An eligible link is
queued when its mean speed is at or below 60% of its free-flow speed. This 75%
rule determines which link-time cells can be scored. Separately, the scheme's
70% coverage requirement determines whether a full history and forecast window
can be selected.

Preparation selects windows that meet the configured coverage and spacing
requirements. Each selected window has its full 60-minute history and all six
forecast steps. Trafficbench keeps proxy answers with the runner and hides
their future source measurements before calling the solution. It calculates
intersection over union (IoU) for each complete window, then aggregates the
window scores by condition.

The proxy labels apply the published threshold and coverage rules to measured
speeds. Official labels use the underlying traffic state, so measurements near
the threshold can produce different labels. A local score compares solutions
on the same proxy cases, but it is not an official Task 2 score.

### Task 3 local diagnostics

Task 3 evaluates the physical consistency of Task 1 predictions. Trafficbench
reports a flow-density mismatch, the fraction of predictions below a low-flow
threshold, and the fraction with a negative value. These diagnostics do not
calculate vehicle conservation or the official Task 3 score.

The official conservation calculation needs traffic flows entering and
leaving each panel at its outer boundaries. Those boundary flows are not in
the public release. The organizer's local scoring script estimates them from
the network and submitted predictions; for this release, its estimate gives
the conservation component a score of zero even for exact Task 1 answers.
Because conservation is two-thirds of Task 3's official score, that local
script's combined physics score is not a reliable leaderboard proxy. See the
[official scoring specification](../official_competition_repo/docs/SCORING_SPEC.md)
and [Task 3 scoring notes](../official_competition_repo/README.md#task-3-is-your-reconstruction-physical)
for the organizer's scoring details.

### Task 4 local cases

Task 4 has no released path-flow answers. For each selected panel, preparation
creates one case from released link counts and three reproducible synthetic
cases with known path flows and matching noisy link counts. The released-count
case has no known path-flow answer, so it can check only how well the predicted
flows reproduce the observed link counts. Link fit is one-quarter of the
official Task 4 score. See the [official Task 4
scorer](../official_competition_repo/src/task4/score_task4.py) for the scope of
this public check.

The three synthetic cases also have known generated path flows. They let
Trafficbench score path-flow and destination-attraction accuracy against those
generated answers, but that does not establish accuracy against the hidden
competition path flows. Each scenario belongs to a panel and split, not to
specific dates. Preparation therefore uses the same released train scenario
regardless of the chosen history and prediction dates. The solution is called
separately for each synthetic scenario because each has its own counts and
prior.

## Solution function contract

From the repository root, create starter functions with this command:

```sh
uv run --locked bench new my_solution
```

It creates the `solutions/my_solution/` package with `state`, `queue`, and
`odme` functions. Each starter delegates to its corresponding baseline;
replace that logic with your method. Each `bench run` or `bench experiment` call selects one function with
`--task` and passes three arguments: a
[`ReleasePackageSlice`](../trafficbench/contracts.py#L604), a mapping of target
templates, and a JSON object containing solution-specific settings. The competition receives the assembled CSV, not these Python functions.
The following example shows the Task 1 function signature:

```python
from trafficbench.contracts import JsonObject, Panel, ReleasePackageSlice, Split
from trafficbench.table_types import StateFrame


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, StateFrame]]:
    ...
```

Use `QueueFrame` or `OdmeFrame` for the other task functions. The target
mapping is nested by panel, then split. A Task 2 prediction table contains all
six future steps for every included queue window.

Every task table is a Polars `LazyFrame`. Build predictions with Polars
expressions and return a lazy frame. Call `.collect()` only when the approach
needs materialized values, such as a NumPy array for an optimizer. A lazy frame
stores a scan plan, not the table's rows. Collecting a plan reads the needed
source columns and rows, and collecting it again runs that scan again.

Return the same panel-then-split mapping with one prediction table for each
requested template. Every requested target row must appear exactly once with
the required prediction columns. Trafficbench checks and aligns results with
[`validate_predictions`](../trafficbench/contracts.py#L690), rejecting missing,
extra, or duplicate target rows and invalid prediction values. The
[`map_panels` helper](../trafficbench/panelwise.py#L15) is optional for a
solution that predicts each panel independently. It applies a panel function
and returns the same panel-then-split mapping.

[`StateFrame`](../trafficbench/table_types.py#L260),
[`QueueFrame`](../trafficbench/table_types.py#L277), and
[`OdmeFrame`](../trafficbench/table_types.py#L296) distinguish task tables in
function annotations. Their companion schemas document the expected columns.
Pyright does not use those schemas to verify Polars column expressions such as
`pl.col("speed_kmh")`, so column names are not statically checked.

## Data passed to the solution

For each call, the runner selects released inputs and official template
rows for the chosen panels and dates. It removes any released values that
would reveal requested Task 1 or Task 2 answers, then gives the solution the
selected inputs and zero-filled target templates. The runner keeps requested
answer values outside both arguments and scores predictions after the function
returns.

A **Release Slice** is the selected published inputs and available historical
labels. A **Target Template** lists the requested output rows and has zero
placeholders in prediction columns. An optional **Answer Table** contains true
values for those rows; the solution never receives those values.

The [competition guide's Release Slice reference](COMPETITION_AND_THEORY.md#what-a-solution-receives)
describes the data available to each task and links to the downloaded file
schemas. This section explains how Trafficbench passes that data to a function.

The `ReleasePackageSlice` has one panel record per selected panel and dated
tables grouped by split. All task tables are available, but Polars reads a
table only when the solution collects its `LazyFrame`. A solution can therefore
choose which tables it needs.

The separate `target_templates_by_panel_and_split` argument contains the
fields that identify each requested row and zero placeholders in the
prediction columns. It uses the same `panel -> split -> table` shape as the
function's return value. The runner keeps true values for those requested rows
outside both arguments. It removes released measurements that would reveal
requested targets and scores after the function returns. The slice may contain
selected historical Task 1 answers and locally calculated Task 2 labels, but
never the unmasked train table or answers for the requested targets.

## What release inputs the solution can see

Mainline and ramp rows begin on `history_start_date` and stop before
`prediction_end_date`. Historical Task 1 answers and Task 2 proxy labels begin
on `history_start_date` and stop before `history_end_date`. Released inputs
after `history_end_date` do not add labels to the history period. The runner
never gives the solution the raw unmasked train table or answer values for
requested target rows.

With `--target-range validation`, the visible date range ends at the validation
boundary, so the slice contains no private observations. With
`--target-range private`, it ends at the private boundary and includes released
validation observations before the private targets. An explicit prediction
period may cross a split boundary. Target tables keep their
original split grouping even when one run requests targets from more than one
split.

For each published Task 2 window whose forecast starts in the visible date
range, Trafficbench includes its complete 60-minute history. That history can
begin before `history_start_date` when the window starts near the beginning of
the range.

Trafficbench creates a lazy scan for each table in the extracted release
directory. The rows are read only when a solution collects that table, so
unused tables do not take up memory. A `LazyFrame` does not cache collected
rows: collecting the same plan again rereads its source. Each collected result
uses memory until the solution releases it.

## Run a prepared case with `bench experiment`

`--benchmark` names the directory created by `bench prepare`. The solution
argument names a package under `solutions/`; `--task` chooses which
function to call.

`bench experiment` calls a solution function on every applicable case in a
Prepared Benchmark, then scores the returned predictions using that case's
answer tables and released inputs. Select the function with `--task` and the
prepared directory with `--benchmark`. Run the command from the repository
root:

```sh
uv run --locked bench experiment baseline --task state \
  --benchmark data/benchmarks/november
```

The `baseline` argument names an importable package under `solutions/`. Use the
package created with `bench new` after implementing a solution. Task choices
are `state` for Task 1, `queue` for Task 2, and `odme` for Task 4. Trafficbench
may call a Task 4 function more than once because each generated scenario has
its own counts and Weak Prior.

A solution may accept its own settings through `--params`. Trafficbench parses
the option as a JSON object and passes it as the function's third argument,
`solution_parameters`. The default is an empty object. For example, this call
passes a `blend` setting to the state function. Run it from the repository
root as well:

```sh
uv run --locked bench experiment my_solution --task state \
  --benchmark data/benchmarks/november \
  --params '{"blend": 0.75}'
```

These are method settings. They do not change the Prepared Benchmark's dates,
panel selection, queue windows, or generated Task 4 answers; use `bench
prepare` settings to control those.

## Read the local metrics

Trafficbench writes metrics only when their required reference values are
available. `state_score` and `queue_proxy_iou` compare predictions with answer
tables. The Task 3 fields are diagnostics calculated from predictions and
network parameters, not the official physics score. Task 4 count fit uses the
released counts; the full local Task 4 score appears only for generated cases
with generated path-flow answers.

The `n` column is a count, not a score. For Task 1 it counts answer rows in a
mask regime. Task 1 scoring uses target rows with both speed and total flow
present and at least 75% detector coverage. For Task 2, `n` counts eligible
link-time cells in a complete forecast window; the local scorer excludes
cells below the 75% detector coverage rule.

| Metric | When it is available and what it compares | Valid range and interpretation |
| --- | --- | --- |
| `state_score` | When a Task 1 answer table is supplied; calculated separately for each represented mask regime from speed and per-lane flow RMSE. | 0 to 1; higher is better. |
| `speed_rmse` | When a Task 1 answer table is supplied; speed error on eligible target rows. | 0 or higher, in km/h; lower is better. |
| `flow_per_lane_rmse` | When a Task 1 answer table is supplied; flow error after dividing by each link's lane count. | 0 or higher, in vehicles per hour per lane; lower is better. |
| `queue_proxy_iou` | When Task 2 answers are supplied; intersection-over-union on eligible cells of one complete forecast window. Prepared train cases use speed-derived proxy labels. | 0 to 1; higher is better. It is not an official Task 2 score when the answer values are proxies. |
| `fd_relative_error_diagnostic` | Calculated for Task 1 predictions using the network's fundamental-diagram parameters. | 0 or higher, with no fixed upper bound; lower means a closer flow-density fit. This is a diagnostic, not the official Task 3 score. |
| `low_flow_fraction_diagnostic` | Fraction of requested Task 1 rows with total predicted flow below 50 vehicles per hour. | 0 to 1; lower means fewer low-flow predictions. It is not the official Task 3 low-flow component. |
| `negative_state_fraction_diagnostic` | Fraction of requested Task 1 rows with negative speed or flow. | 0 to 1; zero means no negative predictions. |
| `odme_link_score_diagnostic` | Fit between predicted path flows and the released link counts for a Task 4 scenario. | 0 to 1; higher is better for count fit, but it does not establish hidden path-flow accuracy. |
| `odme_prior_relative_movement` | Absolute change from the released Weak Prior, divided by its total flow. | 0 or higher, with no fixed upper bound. This describes movement from the prior; neither a higher nor lower value is always better. |
| `odme_synthetic_score` | Calculated only for a generated Task 4 case with known generated path flows. It combines path-flow accuracy (45%), link-count fit (25%), prior deviation (15%), and destination attraction (15%). | 0 to 1; higher is better on that generated case only. It is not a score against hidden competition path flows. |

For Task 1, Trafficbench calculates `state_score` separately for each
represented mask regime, then averages those regime scores. For Task 2,
Trafficbench calculates intersection over union (IoU) using only
eligible link-time cells in a complete forecast window. If both the answer and
prediction mark every eligible cell clear, the window receives an IoU of 1.
Trafficbench averages windows within each condition and gives
`queue_onset` and `queue_ongoing` equal weight. Local aggregates also average
cases, splits, panels, and freeway families according to the hierarchy used by
the scorer. These local metrics help compare methods on the same prepared case;
they do not reproduce the hidden competition leaderboard.

## Make competition predictions with `bench run`

`bench run` applies a solution to rows selected from the competition
templates. It defaults to the `baseline` package and all panels. Set `--task`
to `state`, `queue`, or `odme`; set `--target-range` to choose validation,
private, or both official target periods. The `--data` option defaults to the
extracted release at `kaggle_public/`. Obtain and unpack that release using
[download the data](GET_DATA.md). Run these commands from the repository root:

```sh
uv run --locked bench run baseline --task state --target-range validation
uv run --locked bench run baseline --task queue --target-range both
uv run --locked bench run baseline --task odme --target-range both
```

Each task template defines the row identifiers and output columns. These
commands write predictions for every requested row, but the public release
has no validation or private target answers. The runs therefore report no
answer-based Task 1 or Task 2 score. Task 1 runs still report local physics
diagnostics, and Task 4 runs can report fit to released link counts.

From the repository root, pass date boundaries for custom history and
prediction periods. The prediction start defaults to `history_end_date` when
custom dates are supplied:

```sh
uv run --locked bench run baseline --task state \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-24 \
  --prediction-end-date 2030-12-01 \
  --panel D12_I5_N
```

An explicit prediction period may include train targets. Such predictions are
local outputs and cannot be assembled into the official validation/private
submission.

### Score a run against supplied answers

Use `--answers FILE` when you have answer values for a `bench run`. The CSV
must contain the task target identifiers and the corresponding answer-value
columns. Its rows select which template rows to predict. Trafficbench keeps the
answer values for scoring after the function returns; the function receives
those row identifiers with zero placeholders, not the answers.

Without `--answers`, a run reports no answer-based Task 1 or Task 2 score.
Task 1 still reports local physics diagnostics. Task 4 can report fit to the
released counts, which does not establish accuracy against hidden path flows.

For Task 2, the answer file must include every row in each selected forecast
window. Trafficbench rejects a partial window because queue intersection over
union (IoU) is calculated over the complete window.

Each run writes validated `predictions.csv` rows, `metrics.csv`, `run.json`,
and a human-readable report beneath `runs/`. The metadata records the dates,
panels, target splits, and whether answer values were supplied. The command
logs the prediction path and any available metrics at the `INFO` level. These
messages go to standard error; prediction rows remain in `predictions.csv`.

## Validation leaderboard and final score

Validation is the public evaluation against the competition's hidden answers.
The leaderboard reports one combined score across all four tasks, so it does
not show which task caused a score change. Validation and private are separate
generated months; a validation result is evidence about transfer but cannot
guarantee the private result. Private determines the final ranking. Record
official validation scores in the team's
[validation-score spreadsheet](https://docs.google.com/spreadsheets/d/1SLmxHqxA-ChrNl3DpUFe4O4uv6fOZSZ6ZFqMfxYNe_8/edit?gid=0#gid=0);
keep those evaluator results distinct from local run metrics. See the
[experiment records guide](EXPERIMENTS.md) for the team's recording convention
and the [competition overview](../official_competition_repo/README.md#scoring)
for the combined score.

## Build and upload the competition file

For a complete final submission, run each prediction task across both official
evaluation splits and all panels. Run the commands from the repository root.
Each `bench run` writes a prediction file under `runs/`; replace `STATE_RUN`,
`QUEUE_RUN`, and `ODME_RUN` below with the corresponding run-directory names.
`bench assemble` joins those three files to the official submission key and
combined template, preserving the template's row order. `bench validate` then
checks that the assembled file has the required submission rows and values.
Both commands use this repository's assembler and validator.

```sh
uv run --locked bench run my_idea --task state --target-range both
uv run --locked bench run my_idea --task queue --target-range both
uv run --locked bench run my_idea --task odme --target-range both

uv run --locked bench assemble \
  --state runs/STATE_RUN/predictions.csv \
  --queue runs/QUEUE_RUN/predictions.csv \
  --odme runs/ODME_RUN/predictions.csv \
  --key kaggle_public/submission_key.csv \
  --template kaggle_public/sample_submission.csv \
  --output final_submission.csv
uv run --locked bench validate final_submission.csv \
  --template kaggle_public/sample_submission.csv
```

The assembled CSV follows the official template's row order and includes one
row for every submission ID. The
[`assemble` function](../trafficbench/submission.py#L177) performs the join.
See [submit predictions](SUBMIT.md) for the
manual and Kaggle command-line interface upload steps. See
[experiment records](EXPERIMENTS.md)
for comparing and recording local runs, and
[GitHub-hosted benchmark experiments](HOSTED_BENCHMARK_RUNS.md) for running
prepared experiments on GitHub's hosted machines.

## Command reference

Run commands from the repository root with `uv run --locked bench`. This table
collects the commands introduced in the preceding workflow sections; use
`bench COMMAND --help` for each command's complete options.

| Command | Purpose and guide section |
| --- | --- |
| `bench new SOLUTION` | Create the starter package described in [the function contract](#solution-function-contract). |
| `bench prepare` | Build a repeatable local train case using the dates and scheme explained in [local case preparation](#prepare-and-score-local-train-cases). |
| `bench experiment SOLUTION` | Run a solution and score available local answers as described in [the prepared-case workflow](#run-a-prepared-case-with-bench-experiment). |
| `bench run SOLUTION` | Predict competition-template rows as described in [competition predictions](#make-competition-predictions-with-bench-run). |
| `bench assemble` | Combine the Task 1, Task 2, and Task 4 prediction CSVs using the official key and template. |
| `bench validate SUBMISSION.csv` | Check that a combined submission matches the official template. |
| `bench compare RUN_DIR...` | Compare saved run records; see the [experiment records guide](EXPERIMENTS.md). |
| `bench summary` | Write a CSV summary of saved runs; see the [experiment records guide](EXPERIMENTS.md). |
| `bench package-ci` | Package a prepared case for the [GitHub-hosted benchmark workflow](HOSTED_BENCHMARK_RUNS.md). |
| `bench fixture` | Create synthetic release data for repository checks. |
