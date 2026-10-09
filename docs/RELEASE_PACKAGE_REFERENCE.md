# Public release package reference

This reference describes the files downloaded from the competition. A panel
is one directional freeway network; the competition groups two directions
into each freeway family. A split is one date interval, such as `train` or
`validation`. In paths below, `<PANEL>`, `<SPLIT>`, `<REGIME>`, and date
components are placeholders for values in the release. All paths start at the
unpacked release directory.

Solution functions read tables exposed by Trafficbench rather than opening the
downloaded files directly. The
[competition guide](COMPETITION_AND_THEORY.md#what-a-solution-receives)
explains how those tables are grouped and used by each task. The
[Trafficbench guide](TRAFFICBENCH.md) explains how the local interface builds
the in-memory tables and calls solution functions.

The package contains configuration files, network tables for each panel,
time-series traffic records, task target and scenario tables, and the combined
submission key and template.

The files are available from the [Kaggle competition Data page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data).
See [data download instructions](GET_DATA.md) to obtain and unpack it. The official
[data notes](../official_competition_repo/docs/DATA.md),
[mask specification](../official_competition_repo/docs/MASK_SPEC.md), and
[submission schemas](../official_competition_repo/docs/SUBMISSION_SCHEMAS.md)
give further detail on those subjects.

## Package layout

The unpacked release has this directory structure:

```text
<release-root>/
  config/
    competition_contract.json
    corridors.json
    synthetic_release_v1.json
  corridors/<PANEL>/
    network/
      fd_parameters.csv
      links.csv
      lwr_mainline_topology.csv
      path_link_incidence.csv
      path_set.csv
      ramp_attachment_map.csv
      synthetic_network_manifest.json
      synthetic_ramp_attachment_map.csv
    train/
      mainline_states/year_month=<YYYY>-<MM>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet
      mainline_states_masked/mask_regime=<REGIME>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet
      ramp_states/year_month=<YYYY>-<MM>/synthetic_ramp_<YYYY>_<MM>_<DD>.parquet
    validation/
      mainline_states_masked/mask_regime=<REGIME>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet
      ramp_states/year_month=<YYYY>-<MM>/synthetic_ramp_<YYYY>_<MM>_<DD>.parquet
    private/
      mainline_states_masked/mask_regime=<REGIME>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet
      ramp_states/year_month=<YYYY>-<MM>/synthetic_ramp_<YYYY>_<MM>_<DD>.parquet
  task1/<PANEL>/<SPLIT>/sample_submission_state.csv
  task2/<PANEL>/<SPLIT>/
    window_index.csv
    window_history.parquet
    sample_submission_queue.csv
  task4/<PANEL>/<SPLIT>/
    synthetic_link_counts.csv
    synthetic_weak_prior.csv
    sample_submission_path_flow.csv
  sample_submission.csv
  submission_key.csv
```

`train` also contains unmasked mainline records. Validation and private contain
masked mainline records and ramp records, but no unmasked mainline files. Task 2
tables are present for the eight panels named by the release contract. Task 1
and Task 4 tables are present for all ten panels. Each task directory has one
set of files per available panel and split.

## Panels and date splits

`corridors.json` maps each directional panel to its freeway family. The ten
panels are:

| Freeway family | Panel IDs |
| --- | --- |
| `D7_I10` | `D7_I10_E`, `D7_I10_W` |
| `D7_I210` | `D7_I210_E`, `D7_I210_W` |
| `D7_I405` | `D7_I405_N`, `D7_I405_S` |
| `D12_I5` | `D12_I5_N`, `D12_I5_S` |
| `D12_I405` | `D12_I405_N`, `D12_I405_S` |

`synthetic_release_v1.json` defines the Coordinated Universal Time (UTC) split
boundaries. Each interval includes its start and excludes its end.

| Split | Start, inclusive | End, exclusive | Released mainline views |
| --- | --- | --- | --- |
| `train` | `2030-06-01` | `2031-03-01` | Unmasked and masked |
| `validation` | `2031-03-01` | `2031-04-01` | Masked only |
| `private` | `2031-04-01` | `2031-05-01` | Masked only |

The release records are synthetic and use five-minute traffic intervals.
Parquet `date` and `timestamp` columns are stored as strings in ISO date and
UTC timestamp formats. In the schemas below, `float` and `int` describe the
field's data type. Nullable measurement fields are marked explicitly. Flags
shown as booleans are stored as integer `0` or `1` values.

## Configuration files

### `config/competition_contract.json`

This JSON file records the release contract: panels and families, split labels,
data-quality definitions, task-level input and output rules, and score
aggregation weights. Its `data_quality` section defines `pct_observed` and
score eligibility. Its `tasks` section records the task-specific contract.

### `config/corridors.json`

This JSON file lists each family and its directional panels. Each panel record
contains `corridor_id`, `family_id`, `district`, `freeway_number`, and
`direction`.

### `config/synthetic_release_v1.json`

This JSON file contains the synthetic calendar, scenario contract, local-data
use policy, and Task 2 window-selection policy. The policy identifies
`aggregate calibration only` as the allowed local-source use and lists row
replay, timestamp shifting, and copying or sampling historical traffic values
as prohibited. It also records the Task 2 window conditions, counts, spacing,
and coverage settings.

### `corridors/<PANEL>/network/synthetic_network_manifest.json`

Each panel has one network manifest. It identifies the panel and records which
network inputs are structural and which were supplied as synthetic
replacements.

## Static network tables

Network files describe one panel and are shared by its split-specific traffic
records. A table's key is the column or combination of columns that identifies
one row; keys in these files are unique within a panel. Field names such as
`link_id`, `path_id`, and `ramp_link_id` contain identifiers. These values are
strings even when they contain digits.

### Mainline link attributes

**Example file:** [`links.csv`](../kaggle_public/corridors/D7_I10_E/network/links.csv)

**Path pattern:** `corridors/<PANEL>/network/links.csv`  
**Key within a panel:** `link_id`

Each row gives the geometry and nominal operating limits of one directed
mainline link.

| Field | Type | Meaning |
| --- | --- | --- |
| `link_id` | string | Mainline link identifier. |
| `length_km` | float | Link length in kilometres. |
| `lanes` | float | Number of lanes, stored as a numeric value such as `3.0`. |
| `free_speed_kmh` | float | Free-flow speed in kilometres per hour. |
| `capacity_vph` | float | Total link capacity in vehicles per hour. |

### Fundamental-diagram parameters

**Example file:** [`fd_parameters.csv`](../kaggle_public/corridors/D7_I10_E/network/fd_parameters.csv)

**Path pattern:** `corridors/<PANEL>/network/fd_parameters.csv`  
**Key within a panel:** `link_id`

Each row adds per-link density parameters to the mainline link attributes.
`critical_density` and `k_jam` are total densities across the link's lanes.
Divide them by `lanes` for per-lane values.

| Field | Type | Meaning |
| --- | --- | --- |
| `link_id` | string | Mainline link identifier. |
| `length_km` | float | Link length in kilometres. |
| `lanes` | float | Number of lanes, stored as a numeric value such as `3.0`. |
| `free_speed_kmh` | float | Free-flow speed in kilometres per hour. |
| `capacity_vph` | float | Total link capacity in vehicles per hour. |
| `critical_density` | float | Total density at capacity, in vehicles per kilometre. |
| `k_jam` | float | Total density at jam, in vehicles per kilometre. |

### Mainline connectivity

**Example file:** [`lwr_mainline_topology.csv`](../kaggle_public/corridors/D7_I10_E/network/lwr_mainline_topology.csv)

**Path pattern:** `corridors/<PANEL>/network/lwr_mainline_topology.csv`  
**Key within a panel:** `link_id` (equal to `mainline_link_id`)

Each row describes one mainline link, its connected links and ramps, its
position in the mainline sequence, and sensor coverage used to interpret
network boundaries.

| Field | Type | Meaning |
| --- | --- | --- |
| `mainline_link_id` | string | Mainline link identifier. |
| `incoming_link_ids` | string | Semicolon-separated IDs of incoming links. |
| `outgoing_link_ids` | string | Semicolon-separated IDs of outgoing links. |
| `on_ramp_link_ids` | string | Semicolon-separated IDs of on-ramps attached to this link. |
| `off_ramp_link_ids` | string | Semicolon-separated IDs of off-ramps attached to this link. |
| `incoming_has_sensor` | string | Semicolon-separated `0`/`1` sensor flags aligned with `incoming_link_ids`. |
| `outgoing_has_sensor` | string | Semicolon-separated `0`/`1` sensor flags aligned with `outgoing_link_ids`. |
| `lanes` | int | Lane count. |
| `length_km` | float | Link length in kilometres. |
| `capacity_vph` | float | Total capacity in vehicles per hour. |
| `free_speed_kmh` | float | Free-flow speed in kilometres per hour. |
| `order_index` | int | Position in the ordered mainline. |
| `link_id` | string | Mainline link identifier, equal to `mainline_link_id`. |
| `from_node` | int | Numeric ID of the directed link's starting node. |
| `to_node` | int | Numeric ID of the directed link's ending node. |
| `has_sensor` | boolean (`0`/`1`) | Whether the mainline link has a detector. |
| `detector_id` | string | Associated detector identifier, if one is assigned. |
| `n_incoming` | int | Number of incoming links. |
| `n_outgoing` | int | Number of outgoing links. |
| `n_incoming_no_sensor` | int | Number of incoming links without sensors. |
| `n_outgoing_no_sensor` | int | Number of outgoing links without sensors. |
| `free_flow_speed_kmh` | float | Free-flow speed recorded for the topology link. |

### Ramp attachment map

**Example file:** [`ramp_attachment_map.csv`](../kaggle_public/corridors/D7_I10_E/network/ramp_attachment_map.csv)

**Path pattern:** `corridors/<PANEL>/network/ramp_attachment_map.csv`  
**Key within a panel:** `ramp_link_id`

Each row connects a ramp ID to the mainline link where the ramp joins or leaves
the freeway. `ramp_type` distinguishes an on-ramp (`OR`) from an off-ramp
(`FR`).

| Field | Type | Meaning |
| --- | --- | --- |
| `ramp_link_id` | string | Ramp link identifier. |
| `ramp_type` | string enum: `OR`, `FR` | Ramp type. `OR` joins the mainline and `FR` leaves it. |
| `nearest_mainline_link_id` | string | Mainline link to which the ramp is attached. |

### Synthetic ramp attachment map

**Example file:** [`synthetic_ramp_attachment_map.csv`](../kaggle_public/corridors/D7_I10_E/network/synthetic_ramp_attachment_map.csv)

**Path pattern:** `corridors/<PANEL>/network/synthetic_ramp_attachment_map.csv`  
**Key within a panel:** `ramp_link_id`

This table is labeled for the ramp IDs in the synthetic ramp observations. In
the current public release, `synthetic_ramp_attachment_map.csv` is an exact
copy of `ramp_attachment_map.csv` for every panel: both files contain the same
rows and map the same ramp IDs to the same mainline links. The official Task 3
scorer reads `ramp_attachment_map.csv`. The second file does not provide a
different ramp-to-mainline mapping in this release.

| Field | Type | Meaning |
| --- | --- | --- |
| `ramp_link_id` | string | Synthetic ramp link identifier. |
| `ramp_type` | string enum: `OR`, `FR` | Ramp type. `OR` joins the mainline and `FR` leaves it. |
| `nearest_mainline_link_id` | string | Mainline link to which the synthetic ramp is attached. |

### Candidate path set

**Example file:** [`path_set.csv`](../kaggle_public/corridors/D7_I10_E/network/path_set.csv)

**Path pattern:** `corridors/<PANEL>/network/path_set.csv`  
**Key within a panel:** `path_id`

Each row defines one candidate route by its origin, destination, and ordered
mainline links.

| Field | Type | Meaning |
| --- | --- | --- |
| `path_id` | string | Candidate path identifier. |
| `origin_zone` | string | Origin zone for the path. |
| `destination_zone` | string | Destination zone for the path. |
| `link_seq` | string | Ordered, semicolon-separated mainline link IDs on the path. |

### Path-link incidence

**Example file:** [`path_link_incidence.csv`](../kaggle_public/corridors/D7_I10_E/network/path_link_incidence.csv)

**Path pattern:** `corridors/<PANEL>/network/path_link_incidence.csv`  
**Key within a panel:** (`path_id`, `link_id`)

Each row records one candidate path's use of one mainline link.

| Field | Type | Meaning |
| --- | --- | --- |
| `path_id` | string | Candidate path identifier. |
| `link_id` | string | Mainline link used by the path. |

## Time-series traffic tables

Traffic observations have one row per detector or link and five-minute time
interval. In the daily Parquet files, the `date` and `timestamp` values use
UTC. The fields `is_observed`, `is_imputed`, `is_score_eligible`, and
`is_missing` are integer-encoded booleans. `pct_observed` records the percentage
of the interval with detector coverage. `is_imputed` indicates partial
coverage, `is_score_eligible` records whether required measurements were
present with at least 75% coverage, and `is_missing` indicates a naturally
missing measurement. These quality fields describe the underlying record even
when a masked view hides its measurements.

### Mainline state records

The unmasked table is available only in `train`. The masked table is published
for all three splits. Masked files are organized under the regime assigned to
the date. `R1`, `R2`, and `R3` identify the three mask-regime folders. The
unmasked and masked views share the same columns.

**Unmasked path pattern:** `corridors/<PANEL>/train/mainline_states/year_month=<YYYY>-<MM>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet`  
**Masked path pattern:** `corridors/<PANEL>/<SPLIT>/mainline_states_masked/mask_regime=<REGIME>/synthetic_mainline_<YYYY>_<MM>_<DD>.parquet`  
**Example file:** [`synthetic_mainline_2031_04_03.parquet`](../kaggle_public/corridors/D7_I10_E/private/mainline_states_masked/mask_regime=R1/synthetic_mainline_2031_04_03.parquet)  
**Key within a panel and split:** (`timestamp`, `station_id`, `link_id`)

Each row is a mainline detector's record for one link and five-minute interval.

| Field | Type | Meaning |
| --- | --- | --- |
| `corridor_id` | string | Directional panel ID. |
| `date` | string (`YYYY-MM-DD`) | UTC calendar date of the observation. |
| `timestamp` | string (UTC timestamp) | Start of the five-minute interval. |
| `station_id` | string | Detector station identifier. |
| `link_id` | string | Mainline link measured by the station. |
| `milepost` | float | Position along the freeway in miles. |
| `direction` | string enum: `E`, `W`, `N`, `S` | Travel direction. |
| `speed_kmh` | nullable float | Measured speed in kilometres per hour. |
| `flow_vph` | nullable float | Total measured flow across lanes, in vehicles per hour. |
| `occupancy` | nullable float | Measured detector occupancy. |
| `density_occ_linear_vehpkm` | nullable float | Density derived linearly from occupancy. |
| `pct_observed` | int (`0`–`100`) | Percentage of the interval during which the detector reported. |
| `is_observed` | boolean (`0`/`1`) | Whether the detector reported an observation. |
| `is_imputed` | boolean (`0`/`1`) | Whether interval coverage was partial. |
| `is_score_eligible` | boolean (`0`/`1`) | Whether required measurements were present and coverage was at least 75%. |
| `is_missing` | boolean (`0`/`1`) | Whether a required measurement was naturally unavailable. |
| `mask_regime` | string enum: `R1`, `R2`, `R3` | Mask regime assigned to the observation date. |

### Ramp state records

Ramp state files are published for all three splits. They report measured flow
at on-ramps and off-ramps. Ramp IDs join to the panel's attachment tables.

**Path pattern:** `corridors/<PANEL>/<SPLIT>/ramp_states/year_month=<YYYY>-<MM>/synthetic_ramp_<YYYY>_<MM>_<DD>.parquet`  
**Example file:** [`synthetic_ramp_2031_04_03.parquet`](../kaggle_public/corridors/D7_I10_E/private/ramp_states/year_month=2031-04/synthetic_ramp_2031_04_03.parquet)  
**Key within a panel and split:** (`timestamp`, `station_id`, `ramp_link_id`)

Each row is one ramp detector's record for one ramp and five-minute interval.

| Field | Type | Meaning |
| --- | --- | --- |
| `corridor_id` | string | Directional panel ID. |
| `date` | string (`YYYY-MM-DD`) | UTC calendar date of the observation. |
| `timestamp` | string (UTC timestamp) | Start of the five-minute interval. |
| `station_id` | string | Ramp detector station identifier. |
| `ramp_link_id` | string | Ramp link measured by the station. |
| `ramp_type` | string enum: `OR`, `FR` | Ramp type. `OR` joins the mainline and `FR` leaves it. |
| `flow_vph` | nullable float | Measured ramp flow in vehicles per hour. |
| `pct_observed` | int (`0`–`100`) | Percentage of the interval during which the detector reported. |
| `is_observed` | boolean (`0`/`1`) | Whether the detector reported any part of the interval. |
| `is_imputed` | boolean (`0`/`1`) | Whether interval coverage was partial. |
| `is_score_eligible` | boolean (`0`/`1`) | Whether flow was present and coverage was at least 75%. |
| `is_missing` | boolean (`0`/`1`) | Whether the ramp measurement was naturally unavailable. |

## Task target and scenario tables

Task directories contain one table per panel and split. The task names and
directory structure identify the table purpose. The `split` value is not
repeated as a column in every task table.

### Task 1 state target template

This CSV lists the mainline records where participants must predict speed and
flow for one panel and split. Those two prediction fields are blank in the
published template.

**Path pattern:** `task1/<PANEL>/<SPLIT>/sample_submission_state.csv`  
**Example file:** [`sample_submission_state.csv`](../kaggle_public/task1/D7_I10_E/train/sample_submission_state.csv)  
**Key within one file:** (`timestamp`, `station_id`, `link_id`, `mask_regime`)  
**Key across panel files:** (`panel`, `timestamp`, `station_id`, `link_id`, `mask_regime`)

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | string | Directional panel containing the observation. |
| `timestamp` | string (UTC timestamp) | Start of the five-minute observation interval. |
| `station_id` | string | Detector station measuring the link. |
| `link_id` | string | Mainline link measured. |
| `mask_regime` | string enum: `R1`, `R2`, `R3` | Mask regime assigned to the date. |
| `speed_kmh` | nullable float | Target speed in kilometres per hour. Blank in the template. |
| `flow_vph` | nullable float | Target total flow across lanes, in vehicles per hour. Blank in the template. |

### Task 2 queue-window metadata

The window index describes each forecast window and supplies its time bounds,
condition, and coverage measurements. An established queue means the same link
is queued at two or more history timestamps. Both conditions require a queue
in the forecast.

**Path pattern:** `task2/<PANEL>/<SPLIT>/window_index.csv`  
**Example file:** [`window_index.csv`](../kaggle_public/task2/D7_I10_E/train/window_index.csv)  
**Key within one file:** `window_id`

| Field | Type | Meaning |
| --- | --- | --- |
| `window_id` | string | Forecast-window identifier. |
| `panel` | string | Directional panel containing the window. |
| `family_id` | string | Freeway family containing the panel. |
| `split` | string enum: `train`, `validation`, `private` | Split containing the window. |
| `date` | string (`YYYY-MM-DD`) | UTC date of the forecast origin. |
| `forecast_origin` | string (UTC timestamp) | Forecast origin time. |
| `history_start` | string (UTC timestamp) | Start of the history interval. |
| `history_end` | string (UTC timestamp) | End of the history interval. |
| `forecast_start` | string (UTC timestamp) | Start of the forecast interval. |
| `forecast_end` | string (UTC timestamp) | Timestamp of the last requested forecast step, 30 minutes after the origin. |
| `condition` | string enum: `queue_onset`, `queue_ongoing` | `queue_onset`: no established queue in the history. `queue_ongoing`: an established queue is in the history. |
| `history_coverage` | float (`0`–`1`) | Fraction of expected link and timestamp records present in the history. |
| `future_coverage` | float (`0`–`1`) | Fraction of expected link and timestamp records present in the forecast. |

### Task 2 queue-window history

This Parquet table contains the released 60-minute history for each window.

**Path pattern:** `task2/<PANEL>/<SPLIT>/window_history.parquet`  
**Example file:** [`window_history.parquet`](../kaggle_public/task2/D7_I10_E/train/window_history.parquet)  
**Key within a panel and split:** (`window_id`, `timestamp`, `link_id`)

| Field | Type | Meaning |
| --- | --- | --- |
| `window_id` | string | Forecast window containing the observation. |
| `timestamp` | string (UTC timestamp) | Time of the historical observation. |
| `link_id` | string | Mainline link observed. |
| `speed_kmh` | nullable float | Historical speed in kilometres per hour. |
| `flow_vph` | nullable float | Historical total flow across lanes, in vehicles per hour. |
| `occupancy` | nullable float | Historical detector occupancy. |
| `pct_observed` | int (`0`–`100`) | Percentage of the interval during which the detector reported. |
| `is_score_eligible` | boolean (`0`/`1`) | Whether required measurements were present and coverage was at least 75%. |

### Task 2 queue target template

This CSV lists each link and forecast time where participants must predict
whether the link is queued.

**Path pattern:** `task2/<PANEL>/<SPLIT>/sample_submission_queue.csv`  
**Example file:** [`sample_submission_queue.csv`](../kaggle_public/task2/D7_I10_E/train/sample_submission_queue.csv)  
**Key within a panel and split:** (`window_id`, `timestamp`, `link_id`)

| Field | Type | Meaning |
| --- | --- | --- |
| `window_id` | string | Forecast window containing the prediction row. |
| `timestamp` | string (UTC timestamp) | Forecast time for this row. |
| `link_id` | string | Mainline link whose queue state is requested. |
| `queue_pred` | boolean (`0`/`1`) | Queue status field. Template rows contain `0`. |

### Task 4 link-count scenario

This CSV contains the synthetic observed traffic total for each counted
mainline link in one Task 4 panel and split scenario.

**Path pattern:** `task4/<PANEL>/<SPLIT>/synthetic_link_counts.csv`  
**Example file:** [`synthetic_link_counts.csv`](../kaggle_public/task4/D7_I10_E/train/synthetic_link_counts.csv)  
**Key within one file:** `link_id`  
**Key across panel and split files:** (`panel`, `split`, `link_id`)

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | string | Directional panel containing the link. |
| `link_id` | string | Mainline link with a count in this scenario. |
| `count` | float | Synthetic traffic count for the link. |

### Task 4 weak-prior scenario

This CSV contains a starting flow estimate for each candidate path in one
Task 4 scenario. The **Weak Prior** guides the path-flow estimate, but it is
not the hidden answer. `departure_time` is a scenario label such as
`SYN_PM_TRAIN`, not a clock time.

**Path pattern:** `task4/<PANEL>/<SPLIT>/synthetic_weak_prior.csv`  
**Example file:** [`synthetic_weak_prior.csv`](../kaggle_public/task4/D7_I10_E/train/synthetic_weak_prior.csv)  
**Key within one file:** (`departure_time`, `path_id`)  
**Key across panel and split files:** (`panel`, `split`, `departure_time`, `path_id`)

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | string | Directional panel for the scenario. |
| `departure_time` | string | Scenario label, not a timestamp. |
| `path_id` | string | Candidate path identifier. |
| `origin_zone` | string | Origin zone for the path. |
| `destination_zone` | string | Destination zone for the path. |
| `path_flow` | float | Weak-prior estimate of flow on the path. |

### Task 4 path-flow target template

This CSV lists each candidate path that needs a Task 4 prediction. Its path
and zone fields identify the row, and `path_flow` starts at zero in the
published template.

**Path pattern:** `task4/<PANEL>/<SPLIT>/sample_submission_path_flow.csv`  
**Example file:** [`sample_submission_path_flow.csv`](../kaggle_public/task4/D7_I10_E/train/sample_submission_path_flow.csv)  
**Key within one file:** (`departure_time`, `path_id`)  
**Key across panel and split files:** (`panel`, `split`, `departure_time`, `path_id`)

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | string | Directional panel for the scenario. |
| `departure_time` | string | Scenario label, not a timestamp. |
| `path_id` | string | Candidate path identifier. |
| `origin_zone` | string | Origin zone for the path. |
| `destination_zone` | string | Destination zone for the path. |
| `path_flow` | float | Path-flow field. The release template contains `0.0`. |

## Submission key and combined template

### Submission key

**Example file:** [`submission_key.csv`](../kaggle_public/submission_key.csv)

This CSV identifies every scored submission row and gives the task-specific
keys that locate its target. Fields unrelated to a row's task are empty or use
the `-` placeholder.

**Path:** `submission_key.csv`  
**Key:** `submission_id`

| Field | Type | Meaning |
| --- | --- | --- |
| `submission_id` | int | Unique identifier for one combined-submission row. |
| `task` | string enum: `state`, `queue`, `odme` | Task for this row. |
| `split` | string enum: `train`, `validation`, `private` | Data split containing the target. |
| `panel` | string | Directional panel containing the target. |
| `timestamp` | string (UTC timestamp) | State or queue target time. |
| `station_id` | string | State target detector station. |
| `link_id` | string | State or queue target link. |
| `mask_regime` | string enum: `R1`, `R2`, `R3` | State target's mask regime. |
| `window_id` | string | Queue target's forecast window. |
| `departure_time` | string | Task 4 scenario label. |
| `path_id` | string | Task 4 candidate path. |
| `origin_zone` | string | Task 4 path origin zone. |
| `destination_zone` | string | Task 4 path destination zone. |

### Combined submission template

**Example file:** [`sample_submission.csv`](../kaggle_public/sample_submission.csv)

This CSV has one row for each `submission_id` in the key. Only the value field
or fields for that row's `task` are populated. Value fields for the other tasks
are zero placeholders. Task 1 rows have both `speed_kmh` and `flow_vph` values.

**Path:** `sample_submission.csv`  
**Key:** `submission_id`

| Field | Type | Meaning |
| --- | --- | --- |
| `submission_id` | int | Row identifier matching `submission_key.csv`. |
| `task` | string enum: `state`, `queue`, `odme` | Task associated with this row. |
| `speed_kmh` | float | Task 1 speed value. |
| `flow_vph` | float | Task 1 total flow value across lanes. |
| `queue_pred` | boolean (`0`/`1`) | Task 2 queue status. |
| `path_flow` | float | Task 4 flow for the candidate path. |

For the three task-specific table schemas, see the official
[submission schema reference](../official_competition_repo/docs/SUBMISSION_SCHEMAS.md).
