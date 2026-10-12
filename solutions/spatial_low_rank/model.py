"""Task 1: spatial Kalman + low-rank completion, combined by learned weights.

Same pipeline as ``spatial_kalman`` (profile -> AR(1) Kalman smoother per
station -> spatial correction from neighbouring links), with one extra input in
the final regression: the low-rank (SoftImpute) estimate of the cell, computed
on the same time x station grid as the Kalman smoother.

The two parts are complementary: the Kalman/spatial part captures local,
short-lived events (best for speed), the low-rank part captures corridor-wide
recurring patterns (best for flow). The regression learns, per uncertainty
level, how much to trust each, using the same artificially hidden cells as
``spatial_kalman``.

Local experiment:  bench experiment spatial_low_rank --task state --benchmark data/benchmarks/quick_v2
Tune:              ... --params '{"lr_rank": 15}'   or  '{"low_rank": false}' (= spatial_kalman)
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field

from solutions import baseline
from solutions.kalman_smoother.model import (
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
from solutions.spatial_kalman.model import (
    SpatialParameters,
    _fit_weights,  # pyright: ignore[reportPrivateUsage]
    _groups,  # pyright: ignore[reportPrivateUsage]
    _ordered_series,  # pyright: ignore[reportPrivateUsage]
    _smooth,  # pyright: ignore[reportPrivateUsage]
    _value_features,  # pyright: ignore[reportPrivateUsage]
)
from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    VALUES,
    JsonObject,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]


class SpatialLowRankParameters(SpatialParameters):
    """Spatial Kalman settings (incl. its tuned defaults) plus the low-rank part.

    Attributes:
        low_rank: False drops the low-rank input (= spatial_kalman).
        lr_rank: Maximum number of low-rank components.
        lr_shrink: Soft threshold as a fraction of the largest singular value.
        lr_iter: Maximum SoftImpute iterations.
        lr_tol: Relative change at which SoftImpute stops.
    """

    low_rank: bool = True
    lr_rank: int = Field(default=20, ge=1, le=500)
    lr_shrink: float = Field(default=0.02, ge=0, lt=1)
    lr_iter: int = Field(default=50, ge=1, le=500)
    lr_tol: float = Field(default=1e-4, gt=0)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow: profile + Kalman/spatial + low-rank, combined."""
    p = SpatialLowRankParameters.model_validate(solution_parameters)
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        partial(_state_for_panel, p=p),
    )


def _state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    solution_parameters: JsonObject,
    *,
    p: SpatialLowRankParameters,
) -> dict[Split, table_types.StateFrame]:
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)
    obs = _add_profile(observations, profile, link_means, panel_means, p)

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

        y_train = y.copy()
        drop = hidden_row & anchor
        y_train[steps[drop], sids[drop]] = np.nan
        m_train, v_train = _smooth(y_train, phi, q, r)
        block, uncertainty_train[value] = _value_features(
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
        del m_train, v_train
        if p.low_rank:
            lr_train = _soft_impute(y_train, p)
            block = np.column_stack(
                [block, lr_train[steps[train_rows], sids[train_rows]]]
            )
            del lr_train
        blocks_train[value] = block
        del y_train

        m, v = _smooth(y, phi, q, r)
        block, uncertainty_target[value] = _value_features(
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
        del m, v
        if p.low_rank:
            lr = _soft_impute(y, p)
            block = np.column_stack([block, lr[steps[target_rows], sids[target_rows]]])
            del lr
        blocks_target[value] = block
        del y

    predictions_at_targets: dict[str, FloatArray] = {}
    for value in VALUES["state"]:
        y_obs, prof = residuals[value]
        x_train = np.hstack([np.ones((len(train_rows), 1)), blocks_train[value]])
        x_target = np.hstack([np.ones((len(target_rows), 1)), blocks_target[value]])
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


def _soft_impute(matrix: Float32Array, p: SpatialLowRankParameters) -> Float32Array:
    """Low-rank completion (SoftImpute) of a time x station grid with NaN gaps."""
    observed = np.isfinite(matrix)
    values = np.where(observed, matrix, 0.0).astype(np.float32)
    filled = values.copy()
    if not observed.any():
        return filled
    lam: float | None = None
    low_rank = filled
    previous = np.zeros(int((~observed).sum()), dtype=np.float32)
    for _ in range(p.lr_iter):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        if lam is None:
            lam = p.lr_shrink * float(s[0])
        s = np.maximum(s[: p.lr_rank] - lam, 0.0)
        k = int((s > 0).sum())
        if k == 0:
            return np.zeros_like(filled)
        low_rank = (u[:, :k] * s[:k]) @ vt[:k]
        filled = np.where(observed, values, low_rank).astype(np.float32)
        current = low_rank[~observed]
        change = np.linalg.norm(current - previous) / max(
            float(np.linalg.norm(previous)), 1e-9
        )
        previous = current
        if change < p.lr_tol:
            break
    return low_rank.astype(np.float32)
