# Trafficbench: local cases and final predictions

The competition asks participants to submit one combined comma-separated
values (CSV) file of predictions. **Trafficbench** is this repository's local
tool for producing that file. Developers write one function per prediction
task. Trafficbench gives each function selected competition inputs and target
rows, and the function returns predictions for those rows.

For task definitions and scoring, see
[competition and traffic theory](COMPETITION_AND_THEORY.md). For the files
downloaded from Kaggle, see the [release package reference](RELEASE_PACKAGE_REFERENCE.md).
For installation and download steps, see [download the data](GET_DATA.md).

## The release dates and available labels

The release covers eleven months. Each date split includes its start date and
excludes its end date. The bar shows the order of the splits, not their relative
durations:

```text
2030-06-01                             2031-03-01               2031-04-01            2031-05-01
    |---------- train (9 months) ----------|- validation (1 month) -|- private (1 month) -|
```

Every task function receives the same **Release Slice**: the released tables
selected for the run and any available historical labels. The points below
summarize which answers are available for each task. They do not limit which
tables a solution can read: all tasks can access the shared network and
time-series inputs described later in this guide.

- **Task 1:** Train includes unmasked mainline measurements. Trafficbench uses
  them to obtain true values for eligible Task 1 rows. Validation and private
  include masked measurements, but their target answers are withheld.
- **Task 2:** Official queue answers are not in the public release. For train,
  Trafficbench derives **proxy labels**, local queue estimates calculated from
  measured speeds. Official answers use the underlying traffic state, so the
  estimates may differ from them. Validation and private include masked state
  observations and queue-window information, but their queue answers are
  withheld.
- **Task 3:** Scores the physical consistency of Task 1 predictions and has no
  separate target rows.
- **Task 4:** Each split includes observed link counts and the Weak Prior, a
  starting flow estimate for each candidate path. The public release does not
  include path-flow answers for any split.

For the default private-only prediction, the solution receives the full train
history and predicts private target rows. Released validation data lies between
them and remains available as input. The Task 2 labels shown here are
Trafficbench's locally derived proxies; the competition does not publish queue
answers:

```text
split timeline:       |----------------- train ----------------|- validation -|-------- private --------|

given information:    |----------------------------------- traffic data --------------------------------|
                      |- Task 1 answers + Task 2 proxy labels -|              |- requested target rows -|
```

Trafficbench supports two workflows. `bench run` predicts rows selected from
the published templates for validation, private, or a custom date interval. It
can also score rows when an answer file is supplied. For local comparisons,
`bench prepare` creates a smaller train case with separate target and answer
tables, and `bench experiment` runs a solution against that case using the
same function interface and prediction checks as `bench run`.

## Prediction run: periods and inputs

Each run selects a **history period** and a **prediction period**. The
history period supplies available Task 1 answers and Trafficbench's Task 2
proxy labels. The prediction period selects the rows the solution must fill.
These four dates define the periods. Each includes its start date and excludes
its end date:

| Date field | Meaning |
|---|---|
| `history_start_date` | First date in the history period and the mainline/ramp input range. |
| `history_end_date` | First date after the history period. |
| `prediction_start_date` | First date in the requested prediction period. |
| `prediction_end_date` | First date after the prediction period and the exclusive end of visible inputs. |

The history period must lie within train, where unmasked mainline measurements
are available to obtain Task 1 answers and derive Task 2 proxy labels.
`prediction_start_date` must be on or after `history_end_date`, and
`prediction_end_date` must be no later than the exclusive end of private. The
prediction period can start later in train and continue across validation into
private. A gap between the two periods is allowed: released inputs in the gap
remain visible, but they do not add labels to the history period.

```text
split timeline:    |-------------------- train -----------------------------|------- later train/validation/private ----------|

run dates:    history_start_date                        history_end_date        prediction_start_date          prediction_end_date
                      |----------------------------------------|----- (optional gap) -----|-------------------------------|

given information:    |------------------------------------------------ traffic data -------------------------------------|
                      |- Task 1 answers + Task 2 proxy labels -|                          |---- requested target rows ----|
```

Use `--target-range validation`, `--target-range private`, or
`--target-range both` to select the published validation period, private
period, or both periods together. With no custom dates, `bench run` uses the
full train period for history and requests validation and private targets.
Explicit prediction dates can instead start later in train and continue across
either or both later splits.

Task 4 uses complete origin-destination scenarios rather than timestamped
target rows. Each scenario belongs to a panel and split. The prediction period
selects one complete target scenario for each split it intersects. Changing
dates within a split does not trim or change that scenario, while extending the
period across another split adds that split's scenario. Panel selection
determines which corridors are included. Task 4 has no historical path-flow
answers. Its complete link-count and weak-prior tables are available for each
split in the visible Release Slice; the target mapping contains the requested
path rows for each target split.

Trafficbench keeps the true values for requested target rows away from the
solution function while providing the inputs needed to make predictions:

1. It selects the published input tables for the requested panels and dates.
2. It selects target rows from the official task templates. If an answer file
   is supplied, its rows choose which targets to predict and its values are
   kept by the runner for scoring.
3. It removes released measurements that would reveal the requested Task 1 or
   Task 2 answers.
4. It gives the solution the selected inputs and target rows with zeros in
   the prediction columns.
5. It checks the returned predictions. When answer values are available, it
   scores them after the solution returns.

The **Release Slice** contains published inputs and known labels from the
selected history period in train. A **Target Template** lists the rows the
solution must predict and has zeros in the prediction columns. An optional
**Answer Table** contains the true values for those rows, which the runner uses
to score predictions. The solution receives the target row identifiers, but
not their answer values.

The competition does not receive the solution function. The function is a local
development interface for producing the same kind of prediction rows that are
eventually assembled into the competition's combined CSV.

`bench experiment` may call a Task 4 solution more than once because each
generated origin-destination scenario has its own link counts and weak prior.
It uses the same solution function and prediction checks for every call.

## Solution function contract

Create a solution package with `bench new`, then export a function named for
the task: `state`, `queue`, or `odme`. Each function receives two arguments: a
[`ReleasePackageSlice`](../trafficbench/contracts.py#L604) and a mapping of
target templates:

```python
from trafficbench.contracts import Panel, ReleasePackageSlice, Split
from trafficbench.table_types import StateFrame


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, StateFrame]
    ],
) -> dict[Panel, dict[Split, StateFrame]]:
    ...
```

Use `QueueFrame` or `OdmeFrame` for the other task functions. The target
mapping is nested by panel, then split. Return the same mapping shape with one
prediction table for every requested table. Task 2 tables contain all six
future steps for each included window.

Every task table is a Polars `LazyFrame`. Build predictions with Polars
expressions and return a lazy frame. Call `.collect()` only when the approach
needs materialized values, such as a NumPy array for an optimizer. A lazy frame
stores a scan plan, not the table's rows. Collecting a plan reads the needed
source columns and rows, and collecting it again runs that scan again.

The runner requires every requested target row exactly once and the required
prediction columns. It checks and aligns each result with
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

The [competition guide's Release Slice reference](COMPETITION_AND_THEORY.md#what-a-solution-receives)
defines the fields available to each task, links to the file schemas, and
explains which historical labels are known. This section describes how the
runner passes those values to a function.

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

## Run a solution with `bench run`

Install the development environment and obtain the release as described in
[download the data](GET_DATA.md). The
[`bench run` command](../trafficbench/cli.py#L216) selects all panels by
default. Pass `--panel` to run on selected panels:

```sh
uv run bench run baseline --task state --target-range validation
uv run bench run baseline --task queue --target-range both
uv run bench run baseline --task odme --target-range both
```

The task's official template determines which rows these commands request.
Validation and private answers are withheld, so these runs cannot report
label-based scores for those targets. Task 1 runs still report local physics
diagnostics.

For custom history and prediction periods, pass their date boundaries. The
prediction start defaults to `history_end_date` when custom dates are supplied:

```sh
uv run bench run baseline --task state \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-24 \
  --prediction-end-date 2030-12-01 \
  --panel D12_I5_N
```

An explicit prediction period may include train targets. Such predictions are
local outputs and cannot be assembled into the official validation/private
submission.

### Score against supplied answers

Pass a CSV with the task's target identifier columns and answer-value columns
using `--answers`. Its rows select which official template rows to predict.
The answer values remain with the runner and are used for scoring after the
solution returns. The solution receives those same identifier values in
zero-filled target templates, not the answers. Without `--answers`, the run makes
predictions but reports no label-based target score. Task 1 still reports its
local physics diagnostic. Task 4 can report its fit to released counts, which
does not establish accuracy against the hidden path flows.

For Task 2, an answer file must include every target row in each selected
window. The runner rejects a partial window because intersection over union
(IoU) is scored over the complete window.

Each run writes `predictions.csv`, `metrics.csv`, and `run.json` beneath
`runs/`. The CSV has validated task rows. The metadata records the selected
date intervals, panels, target splits, and whether answer values were supplied.

## Prepare and score local train cases

`bench run` can score predictions when an answer file is supplied. For a
repeatable local benchmark, the
[`bench prepare` command](../trafficbench/cli.py#L56) creates a train case with
an earlier history period and a later prediction period. The history period
supplies Task 1 answers and Task 2 proxy labels. The prediction period supplies
Task 1 and Task 2 target rows and their answers. Both periods must be in train because
Task 1 answers come from unmasked train measurements and Task 2 proxy labels
are derived from train speeds. The runner keeps the target answers away from
the solution and compares them with predictions afterward. One **Prepared
Benchmark** represents one history period and one prediction period. Prepare
another benchmark to compare a different date range.

The diagram below zooms in on train. Both the history and prediction periods
lie within it, so the runner can supply historical labels and keep target
answers for scoring. Released masked inputs between the periods remain
available to the solution.

```text
split timeline:   |------------------------------------------------------train-------------------------------------------------|

run dates:    history_start_date                        history_end_date        prediction_start_date          prediction_end_date
                      |----------------------------------------|----- (optional gap) -----|-------------------------------|

given information:    |------------------------------------------------ traffic data -------------------------------------|
                      |- Task 1 answers + Task 2 proxy labels -|                          |---- requested target rows ----|
```

Masked inputs between the periods remain available, but their hidden values do
not become historical labels. The prediction period is one continuous date
range shared by all selected panels.

For example, use the default `quick` profile to prepare one panel, or use
`full` to prepare all panels:

```sh
uv run bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-24 \
  --prediction-end-date 2030-12-01 \
  --profile quick --output data/benchmarks/november

uv run bench experiment baseline --task state \
  --benchmark data/benchmarks/november
```

Preparation saves selected release inputs, target rows, and answer tables
separately. The [`bench experiment` runner](../trafficbench/runner.py#L614)
rebuilds the same solution input shape used by `bench run` and calls the same
prediction validation and scoring code.

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
speed measurement and each station has at least 75% observation coverage. An
eligible link is queued when its mean speed is at or below 60% of its free-flow
speed.

Preparation selects windows that meet the configured coverage and spacing
requirements. Each selected window has its full 60-minute history and all six
forecast steps. Trafficbench keeps proxy answers with the runner and hides
their future source measurements before calling the solution. It calculates
IoU for each complete window, then aggregates the window scores by condition.

The proxy labels apply the published threshold and coverage rules to measured
speeds. Official labels use the underlying traffic state, so measurements near
the threshold can produce different labels. A local score compares solutions
on the same proxy cases, but it is not an official Task 2 score.

### Task 3 local diagnostics

Task 3 evaluates the physical consistency of Task 1 predictions. Trafficbench
reports local flow-density and value-range diagnostics, not the competition's
Task 3 score. The public release omits organizer boundary flows: traffic
entering or leaving a panel at its outer ends. The official repository's local
conservation scorer estimates those flows from topology and submitted values.

That estimate is too coarse for the conservation calculation, so the scorer
floors `S_LWR` at zero even for the exact Task 1 answers. Because this component
accounts for two-thirds of the official Task 3 score, the public local Task 3
score is a poor leaderboard proxy. Its fundamental-diagram component can still
be calculated.
See the [official scoring specification](../official_competition_repo/docs/SCORING_SPEC.md)
and [Task 3 scoring notes](../official_competition_repo/README.md#task-3-is-your-reconstruction-physical).

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

## Validation leaderboard and final score

Validation is the only participant-visible evaluation against the official
hidden answers before the final evaluation. The leaderboard reports one
combined score across all four tasks, so it does not show which task caused a
score change. Validation and private are separately generated months, so a
validation result is useful evidence about transfer but cannot guarantee the
private result. Private determines the final ranking and is scored at the final
evaluation. The [competition overview](../official_competition_repo/README.md#scoring)
describes the combined score and leaderboard.

## Build and upload the competition file

For a complete final submission, run each prediction task across both official
evaluation splits and all panels. `bench assemble` joins those three output
files to the official key and combined template. It uses this repository's
assembler and validator. It does not call the official repository's merge
helper.

```sh
uv run bench run my_idea --task state --target-range both
uv run bench run my_idea --task queue --target-range both
uv run bench run my_idea --task odme --target-range both

uv run bench assemble \
  --state runs/STATE_RUN/predictions.csv \
  --queue runs/QUEUE_RUN/predictions.csv \
  --odme runs/ODME_RUN/predictions.csv \
  --key kaggle_public/submission_key.csv \
  --template kaggle_public/sample_submission.csv \
  --output final_submission.csv
uv run bench validate final_submission.csv \
  --template kaggle_public/sample_submission.csv
```

The assembled CSV follows the official template's row order and includes one
row for every submission ID. The
[`assemble` function](../trafficbench/submission.py#L177) performs the join.
See [submit predictions](SUBMIT.md) for the
manual and Kaggle command-line interface upload steps. See
[experiment records](EXPERIMENTS.md)
for comparing and recording local runs, and [hosted benchmark setup](CI_SETUP.md)
for experiments on GitHub's hosted machines.
