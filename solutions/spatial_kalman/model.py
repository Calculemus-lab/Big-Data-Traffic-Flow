"""Task 1: Kalman smoother in time + spatial correction from neighbouring links.

Builds on ``kalman_smoother`` (same profile, same AR(1) RTS smoother per
station) and adds a second, spatial stage:

1. Time stage: for every station, smooth its deviation from the profile along
   time (exactly as ``kalman_smoother``). This gives a time-only estimate m and
   its uncertainty v at every cell.
2. Spatial stage: for a masked cell on link l, combine its own time estimate
   with what the neighbouring links show (their observed value if visible, else
   their time estimate) using linear weights. The weights are fitted per
   corridor by regression on cells hidden *artificially* (whole links at random
   times, like the real mask), so they are learned in the same situation as the
   real targets. Separate weights are fitted per group of cells (by how
   uncertain the time estimate is, and optionally by time of day).

Optional extensions (all off by default, switch on with --params):
    n_neighbours   1, 2 or 3 links on each side instead of 1
    neighbour_lags also use neighbours at t-1 and t+1 (jams moving between links)
    cross          speed features help predict flow and vice versa
    time_bins      separate weights for four 6-hour blocks of the day

Local experiment:  bench experiment spatial_kalman --task state --benchmark data/benchmarks/quick_v2
Tune:              ... --params '{"neighbour_lags": true}'   or  '{"spatial": false}' (= kalman_smoother)
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field

from solutions import baseline
from solutions.kalman_smoother.model import (
    KalmanParameters,
    _back_transform,  # pyright: ignore[reportPrivateUsage]
    _fit_ar1,  # pyright: ignore[reportPrivateUsage]
    _residuals,  # pyright: ignore[reportPrivateUsage]
)
from solutions.residual_interp.model import (
    CELL,
    SERIES,
    STEP_SECONDS,
    _add_profile,  # pyright: ignore[reportPrivateUsage]
    _fit_profile,  # pyright: ignore[reportPrivateUsage]
)
from trafficbench import table_types
from trafficbench.contracts import KEYS, VALUES, Panel, ReleasePackageSlice, Split
from trafficbench.panelwise import map_panels

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
UNCERTAINTY_EDGES = (0.15, 0.5)  # smoothed variance / stationary variance -> 3 bins
OWN_ESTIMATE_COLUMN = 1  # column of the own time estimate in the design matrix


class SpatialParameters(KalmanParameters):
    """Kalman settings plus the spatial stage.

    Defaults use the best Kalman settings found so far (min_pct 25, no
    coverage-based noise scaling).

    Attributes:
        spatial: False switches the spatial stage off (= kalman_smoother).
        mask_rate: Share of (link, time) cells hidden to train the spatial weights.
        max_train_rows: Cap on training cells per corridor.
        ridge: Ridge penalty for the weight regression (relative to data scale).
        seed: Random seed for the artificial mask.
        n_neighbours: Number of links used on each side (upstream and downstream).
        neighbour_lags: Also use the neighbours at t-1 and t+1.
        cross: Add the other variable's features (speed for flow and vice versa).
        time_bins: Separate weights for four 6-hour blocks of the day (UTC).
    """

    min_pct: float = Field(default=25, ge=0, le=100)
    noise_by_coverage: bool = False
    spatial: bool = True
    mask_rate: float = Field(default=0.35, gt=0, lt=1)
    max_train_rows: int = Field(default=300_000, ge=1_000)
    ridge: float = Field(default=1e-3, ge=0)
    seed: int = 0
    n_neighbours: int = Field(default=2, ge=1, le=3)
    neighbour_lags: bool = True
    cross: bool = False
    time_bins: bool = False
    day_groups: Literal["dow", "pooled"] = "pooled"
    smooth: int = Field(default=2, ge=0, le=12)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow: profile + time smoother + spatial correction."""
    p = SpatialParameters.model_validate(dict(release_slice.parameters))
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        partial(_state_for_panel, p=p),
    )


def _state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    *,
    p: SpatialParameters,
) -> dict[Split, table_types.StateFrame]:
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)
    obs = _add_profile(observations, profile, link_means, panel_means, p)

    # Grid: row = 5-minute step, column = station series. Links ordered along the corridor.
    first = obs.select(pl.col("timestamp").min()).item()
    topology = release_slice.panel_slices_by_panel[
        panel
    ].network.lwr_mainline_topology.collect()
    series = _ordered_series(obs, topology)
    obs = (
        obs.join(series.select([*SERIES, "sid"]), on=SERIES, how="left")
        .with_columns(
            ((pl.col("timestamp") - first).dt.total_seconds() // STEP_SECONDS)
            .cast(pl.Int64)
            .alias("step")
        )
        .with_row_index("row")
    )
    steps = obs["step"].to_numpy().astype(np.int64)
    sids = obs["sid"].to_numpy().astype(np.int64)
    hours = obs["timestamp"].dt.hour().to_numpy().astype(np.int64)
    link_of_series = series["lid"].to_numpy().astype(np.int64)
    n_links = int(link_of_series.max()) + 1
    shape = (int(steps.max()) + 1, len(series))
    pct = obs["pct_observed"].cast(pl.Float64).fill_null(0.0).to_numpy()
    rng = np.random.default_rng(p.seed)

    # Rows we must predict: every cell requested by any split's template.
    target_keys = pl.concat(
        [t.select(CELL).collect() for t in target_templates_by_split.values()]
    ).unique()
    target_table = obs.select([*CELL, "row"]).join(target_keys, on=CELL, how="inner")
    target_rows = target_table["row"].to_numpy().astype(np.int64)

    modes = {"speed_kmh": p.speed_mode, "flow_vph": p.flow_mode}
    residuals = {v: _residuals(obs, v, modes[v], p) for v in VALUES["state"]}
    anchors = {
        v: np.isfinite(residuals[v][0]) & (pct >= p.min_pct) for v in VALUES["state"]
    }

    # Artificial mask for learning the spatial weights: whole links at random times.
    hidden_links = rng.random((shape[0], n_links)) < p.mask_rate
    hidden_row = hidden_links[steps, link_of_series[sids]]
    both = anchors["speed_kmh"] & anchors["flow_vph"]
    train_rows = np.flatnonzero(hidden_row & both)
    if len(train_rows) > p.max_train_rows:
        train_rows = rng.choice(train_rows, p.max_train_rows, replace=False)

    blocks_train: dict[str, FloatArray] = {}
    blocks_target: dict[str, FloatArray] = {}
    uncertainty_train: dict[str, FloatArray] = {}
    uncertainty_target: dict[str, FloatArray] = {}
    for value in VALUES["state"]:
        y_obs = residuals[value][0]
        anchor = anchors[value]
        y = np.full(shape, np.nan, dtype=np.float32)
        y[steps[anchor], sids[anchor]] = y_obs[anchor]
        phi, q, r = _fit_ar1(y.astype(np.float64), p)
        stationary = q / (1 - phi**2)

        if p.spatial:
            y_train = y.copy()
            drop = hidden_row & anchor
            y_train[steps[drop], sids[drop]] = np.nan
            m_train, v_train = _smooth(y_train, phi, q, r)
            blocks_train[value], uncertainty_train[value] = _value_features(
                y_train,
                m_train,
                v_train,
                steps[train_rows],
                sids[train_rows],
                link_of_series,
                n_links,
                stationary,
                p,
            )
            del y_train, m_train, v_train

        m, v = _smooth(y, phi, q, r)
        blocks_target[value], uncertainty_target[value] = _value_features(
            y,
            m,
            v,
            steps[target_rows],
            sids[target_rows],
            link_of_series,
            n_links,
            stationary,
            p,
        )
        del y, m, v

    predictions_at_targets: dict[str, FloatArray] = {}
    for value in VALUES["state"]:
        other = "flow_vph" if value == "speed_kmh" else "speed_kmh"
        y_obs, prof = residuals[value]
        r_hat = blocks_target[value][:, 0].copy()  # time-only estimate
        if p.spatial:
            x_train = _design(blocks_train, value, other, p)
            x_target = _design(blocks_target, value, other, p)
            groups_train = _groups(uncertainty_train[value], hours[train_rows], p)
            groups_target = _groups(uncertainty_target[value], hours[target_rows], p)
            beta = _fit_weights(x_train, y_obs[train_rows], groups_train, p)
            r_hat = np.einsum("ij,ij->i", x_target, beta[groups_target])
        visible = anchors[value][target_rows]
        r_hat[visible] = y_obs[target_rows][visible]
        predictions_at_targets[value] = _back_transform(
            prof[target_rows], r_hat, modes[value], p
        )

    target_table = target_table.with_columns(
        pl.Series(f"pred_{v}", predictions_at_targets[v]) for v in VALUES["state"]
    ).drop("row")

    predictions: dict[Split, table_types.StateFrame] = {}
    for split, template in target_templates_by_split.items():
        targets = _add_profile(
            template.select(KEYS["state"]).collect(),
            profile,
            link_means,
            panel_means,
            p,
        )
        predictions[split] = (
            targets.join(target_table, on=CELL, how="left")
            .with_columns(
                pl.coalesce(f"pred_{v}", f"p_{v}").clip(lower_bound=0).alias(v)
                for v in VALUES["state"]
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )
    return predictions


def _ordered_series(obs: pl.DataFrame, topology: pl.DataFrame) -> pl.DataFrame:
    """One row per station series with sid (grid column) and lid (link position)."""
    series = obs.select(SERIES).unique()
    if "order_index" in topology.columns:
        order = topology.select(
            pl.col("link_id").cast(pl.Utf8), pl.col("order_index").cast(pl.Float64)
        ).unique("link_id")
    else:
        order = pl.DataFrame(
            {"link_id": [], "order_index": []},
            schema={"link_id": pl.Utf8, "order_index": pl.Float64},
        )
    series = series.join(order, on="link_id", how="left")
    # Links without a known position go to the end, in name order.
    links = (
        series.select("link_id", "order_index")
        .unique("link_id")
        .sort([pl.col("order_index").is_null(), "order_index", "link_id"])
        .with_row_index("lid")
        .select("link_id", "lid")
    )
    return (
        series.join(links, on="link_id", how="left")
        .sort(["lid", "station_id"])
        .with_row_index("sid")
        .select([*SERIES, "sid", "lid"])
    )


def _smooth(
    y: Float32Array, phi: FloatArray, q: FloatArray, r: FloatArray
) -> tuple[Float32Array, Float32Array]:
    """AR(1) Kalman filter + RTS smoother per column; returns smoothed mean and variance.

    Memory-lean: stores only filtered mean/variance (float32) and overwrites them
    in place with the smoothed values during the backward pass.
    """
    n_steps, n_series = y.shape
    mean = np.empty((n_steps, n_series), dtype=np.float32)
    var = np.empty((n_steps, n_series), dtype=np.float32)
    m, v = np.zeros(n_series), q / (1 - phi**2)
    for t in range(n_steps):
        if t:
            m, v = phi * m, phi**2 * v + q
        row = y[t].astype(np.float64)
        observed = np.isfinite(row)
        gain = v / (v + r)
        m = np.where(observed, m + gain * (np.nan_to_num(row) - m), m)
        v = np.where(observed, (1 - gain) * v, v)
        mean[t], var[t] = m, v

    next_m, next_v = mean[-1].astype(np.float64), var[-1].astype(np.float64)
    for t in range(n_steps - 2, -1, -1):
        fm, fv = mean[t].astype(np.float64), var[t].astype(np.float64)
        pred_m, pred_v = phi * fm, phi**2 * fv + q
        c = fv * phi / pred_v
        next_m = fm + c * (next_m - pred_m)
        next_v = fv + c**2 * (next_v - pred_v)
        mean[t], var[t] = next_m, next_v
    return mean, var


def _link_summaries(
    y: Float32Array, m: Float32Array, link_of_series: IntArray, n_links: int
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Per (time, link): mean observed residual, mean estimated residual, visible share."""
    membership = np.zeros((len(link_of_series), n_links), dtype=np.float32)
    membership[np.arange(len(link_of_series)), link_of_series] = 1.0
    visible = np.isfinite(y).astype(np.float32)
    n_visible = visible @ membership
    n_hidden = (1.0 - visible) @ membership
    observed_mean = (np.nan_to_num(y) @ membership) / np.maximum(n_visible, 1.0)
    estimated_mean = ((m * (1.0 - visible)) @ membership) / np.maximum(n_hidden, 1.0)
    share = n_visible / np.maximum(n_visible + n_hidden, 1.0)
    return observed_mean, estimated_mean, share


def _value_features(
    y: Float32Array,
    m: Float32Array,
    v: Float32Array,
    cell_steps: IntArray,
    cell_sids: IntArray,
    link_of_series: IntArray,
    n_links: int,
    stationary: FloatArray,
    p: SpatialParameters,
) -> tuple[FloatArray, FloatArray]:
    """Features of one variable at the given cells, plus the relative time uncertainty.

    Column 0 is the cell's own time estimate. Then, for each neighbour link
    (upstream and downstream, up to n_neighbours away) and each time lag, two
    columns: observed mean * visible share and estimated mean * hidden share.
    """
    observed_mean, estimated_mean, share = _link_summaries(
        y, m, link_of_series, n_links
    )
    n_steps = y.shape[0]
    links = link_of_series[cell_sids]
    columns = [m[cell_steps, cell_sids].astype(np.float64)]
    lags = (-1, 0, 1) if p.neighbour_lags else (0,)
    for side in (-1, 1):
        for distance in range(1, p.n_neighbours + 1):
            neighbour = links + side * distance
            has_link = (neighbour >= 0) & (neighbour < n_links)
            nb = np.clip(neighbour, 0, n_links - 1)
            for lag in lags:
                t = cell_steps + lag
                ok = has_link & (t >= 0) & (t < n_steps)
                tc = np.clip(t, 0, n_steps - 1)
                vis = share[tc, nb].astype(np.float64)
                columns.append(np.where(ok, observed_mean[tc, nb] * vis, 0.0))
                columns.append(np.where(ok, estimated_mean[tc, nb] * (1.0 - vis), 0.0))
    relative = v[cell_steps, cell_sids] / np.maximum(stationary[cell_sids], 1e-12)
    return np.column_stack(columns), relative.astype(np.float64)


def _design(
    blocks: Mapping[str, FloatArray], value: str, other: str, p: SpatialParameters
) -> FloatArray:
    """Intercept + this variable's features (+ the other variable's, if cross)."""
    parts = [np.ones((len(blocks[value]), 1)), blocks[value]]
    if p.cross:
        parts.append(blocks[other])
    return np.hstack(parts)


def _groups(uncertainty: FloatArray, hours: IntArray, p: SpatialParameters) -> IntArray:
    """Weight group per cell: uncertainty level (3) x optional 6-hour block (4)."""
    group = np.digitize(uncertainty, UNCERTAINTY_EDGES).astype(np.int64)
    if p.time_bins:
        group = group * 4 + hours // 6
    return group


def _fit_weights(
    x: FloatArray, truth: FloatArray, groups: IntArray, p: SpatialParameters
) -> FloatArray:
    """Ridge regression per group; groups with too few rows keep the time estimate."""
    n_groups = (len(UNCERTAINTY_EDGES) + 1) * (4 if p.time_bins else 1)
    n_features = x.shape[1]
    fallback = np.zeros(n_features)
    fallback[OWN_ESTIMATE_COLUMN] = 1.0
    beta = np.tile(fallback, (n_groups, 1))
    for g in range(n_groups):
        rows = groups == g
        if rows.sum() < max(500, 20 * n_features):
            continue
        xg, yg = x[rows], truth[rows]
        gram = xg.T @ xg
        penalty = p.ridge * np.diag(
            np.diag(gram)
        )  # scale-free: each column vs its own size
        penalty[0, 0] = 0.0
        beta[g] = np.linalg.solve(gram + penalty, xg.T @ yg)
    return beta
