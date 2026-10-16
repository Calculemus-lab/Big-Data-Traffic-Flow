"""Task 1: profile + low-rank autoregressive tensor completion (LATC).

Same profile as the other solutions. The deviations from the profile are
arranged as a 3-way tensor

    stations x days x 5-min slots of the day

and completed with LATC (Chen, Lei, Saunier & Sun, IEEE T-ITS 2022). LATC
combines two assumptions:

* the tensor is low-rank in every direction (a few typical station patterns,
  day patterns and daily shapes) -- the "time" and "day" layouts of
  ``low_rank`` in one model;
* each station's time series follows an autoregressive (AR) model: the value
  now is a weighted sum of the values ``time_lags`` steps earlier, so a missing
  cell is also pulled towards what its own recent past predicts.

Solved with ADMM, repeating:
  1. shrink the singular values of each unfolding of the tensor
     (truncated nuclear norm: the top ``theta`` share is not shrunk);
  2. refit the AR coefficients of every station;
  3. set the missing cells to a weighted average of the low-rank estimate and
     the AR prediction (weight ``ar_weight``); observed cells stay fixed.

Only the last ``window_days`` days before the first target day plus the target
period are completed; the profile still uses the full history. The residuals
are divided by their standard deviation first, so the same ``rho0`` works for
speed (km/h) and flow (log-ratio).

Local experiment:  bench experiment latc --task state --benchmark data/benchmarks/quick_v2
Tune:              ... --params '{"ar_weight": 5}'  or  '{"time_lags": [1, 2, 3, 288]}'
Ablation:          ... --params '{"ar_weight": 0}'  (= plain low-rank tensor completion)
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from functools import partial
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field, PositiveInt

from solutions import baseline
from solutions.residual_interp.model import (
    CELL,
    SERIES,
    STEP_SECONDS,
    Parameters,
    _add_profile,  # pyright: ignore[reportPrivateUsage]
    _fit_profile,  # pyright: ignore[reportPrivateUsage]
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
IntArray = NDArray[np.int64]
SLOTS_PER_DAY = 24 * 3600 // STEP_SECONDS  # 288
RHO_GROWTH = 1.05
RHO_MAX = 1e5


class LatcParameters(Parameters):
    """Profile settings plus the LATC settings.

    Attributes:
        window_days: Days of history before the first target day used in the
            tensor (the profile always uses the full history).
        time_lags: AR lags in 5-min steps (288 = same time yesterday).
        ar_weight: Weight of the AR prediction vs the low-rank estimate for the
            missing cells; 0 switches the AR part off.
        alpha: Weights of the station, day and slot directions of the tensor
            (normalised to sum 1; 0 drops that direction).
        theta: Share of the largest singular values left unshrunk in each
            direction (0 = ordinary nuclear norm).
        rho0: Starting ADMM step parameter (grows 5% per iteration). Smaller
            means stronger shrinkage early on.
        n_iter: Maximum ADMM iterations.
        tol: Stop when the estimate changes less than this (relative).
    """

    min_pct: float = Field(default=25, ge=0, le=100)
    day_groups: Literal["dow", "pooled"] = "pooled"
    smooth: int = Field(default=2, ge=0, le=12)
    window_days: int = Field(default=56, ge=1, le=400)
    time_lags: list[PositiveInt] = Field(
        default_factory=lambda: [1, 2, 3], min_length=1, max_length=12
    )
    ar_weight: float = Field(default=0.0, ge=0)
    alpha: list[float] = Field(
        default_factory=lambda: [6.0, 1.0, 1.0], min_length=3, max_length=3
    )
    theta: float = Field(default=0.2, ge=0, lt=1)
    rho0: float = Field(default=1e-3, gt=0)
    n_iter: int = Field(default=100, ge=1, le=1000)
    tol: float = Field(default=1e-4, gt=0)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow as profile + LATC-completed residual."""
    p = LatcParameters.model_validate(solution_parameters)
    if not any(a > 0 for a in p.alpha):
        msg = "alpha needs at least one positive weight"
        raise ValueError(msg)
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
    p: LatcParameters,
) -> dict[Split, table_types.StateFrame]:
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)

    target_keys = pl.concat(
        [t.select(CELL).collect() for t in target_templates_by_split.values()]
    ).unique()
    first_target = target_keys.select(pl.col("timestamp").min()).item()
    start = first_target.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=p.window_days)

    obs = _add_profile(observations, profile, link_means, panel_means, p).filter(
        pl.col("timestamp") >= start
    )
    first = obs.select(pl.col("timestamp").min()).item()
    origin = first.replace(hour=0, minute=0, second=0, microsecond=0)

    series = obs.select(SERIES).unique().sort(SERIES).with_row_index("sid")
    obs = (
        obs.join(series, on=SERIES, how="left")
        .with_columns(
            ((pl.col("timestamp") - origin).dt.total_seconds() // STEP_SECONDS)
            .cast(pl.Int64)
            .alias("step")
        )
        .with_row_index("row")
    )
    sids = obs["sid"].to_numpy().astype(np.int64)
    steps = obs["step"].to_numpy().astype(np.int64)
    n_series = len(series)
    n_days = int(steps.max()) // SLOTS_PER_DAY + 1
    shape = (n_series, n_days, SLOTS_PER_DAY)
    pct = obs["pct_observed"].cast(pl.Float64).fill_null(0.0).to_numpy()

    target_table = obs.select([*CELL, "row"]).join(target_keys, on=CELL, how="inner")
    target_rows = target_table["row"].to_numpy().astype(np.int64)

    modes = {"speed_kmh": p.speed_mode, "flow_vph": p.flow_mode}
    predictions_at_targets: dict[str, FloatArray] = {}
    for value in VALUES["state"]:
        y_obs, prof = _residuals(obs, value, modes[value], p)
        anchor = np.isfinite(y_obs) & (pct >= p.min_pct)
        r_hat = np.zeros(len(target_rows), dtype=np.float64)
        if anchor.any():
            scale = float(np.std(y_obs[anchor])) or 1.0
            matrix = np.full(
                (n_series, n_days * SLOTS_PER_DAY), np.nan, dtype=np.float32
            )
            matrix[sids[anchor], steps[anchor]] = y_obs[anchor] / scale
            completed = _latc(matrix, shape, p)
            r_hat = (
                completed[sids[target_rows], steps[target_rows]].astype(np.float64)
                * scale
            )
            del matrix, completed
        visible = anchor[target_rows]
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


def _residuals(
    obs: pl.DataFrame, value: str, mode: str, p: LatcParameters
) -> tuple[FloatArray, FloatArray]:
    """Deviation from the profile (log-ratio in multiplicative mode) and the profile."""
    y = obs[value].cast(pl.Float64).fill_null(np.nan).to_numpy()
    prof = obs[f"p_{value}"].cast(pl.Float64).to_numpy()
    if mode == "mult":
        c = p.offset
        with np.errstate(invalid="ignore"):
            resid = np.log((np.clip(y, 0, None) + c) / (np.clip(prof, 0, None) + c))
    else:
        resid = y - prof
    return resid, prof


def _back_transform(
    prof: FloatArray, r_hat: FloatArray, mode: str, p: LatcParameters
) -> FloatArray:
    """Profile + deviation, back in speed/flow units."""
    if mode == "mult":
        c = p.offset
        pred = (np.clip(prof, 0, None) + c) * np.exp(np.clip(r_hat, -1.6, 1.6)) - c
    else:
        pred = prof + r_hat
    return np.clip(pred, 0, None)


def _latc(
    matrix: Float32Array, shape: tuple[int, int, int], p: LatcParameters
) -> Float32Array:
    """Complete a stations x time matrix (NaN = missing) with LATC via ADMM.

    ``shape`` is (stations, days, slots); the time axis of ``matrix`` is
    day-major, so ``matrix.reshape(shape)`` is the tensor. Returns the completed
    matrix: observed cells unchanged, missing cells filled.
    """
    observed = np.isfinite(matrix)
    missing = ~observed
    z = np.where(observed, matrix, 0.0).astype(np.float32)
    n_series, n_time = z.shape

    weights = np.asarray(p.alpha, dtype=np.float32)
    weights /= weights.sum()
    active = [k for k in range(3) if weights[k] > 0]
    lags = np.asarray(sorted(set(p.time_lags)), dtype=np.int64)
    max_lag = int(lags.max())
    use_ar = p.ar_weight > 0 and max_lag + 10 * len(lags) < n_time

    x = np.zeros((3, *shape), dtype=np.float32)
    dual = np.zeros((3, *shape), dtype=np.float32)
    rho = p.rho0
    previous = z.copy()
    norm = max(float(np.linalg.norm(z)), 1e-9)

    for _ in range(p.n_iter):
        rho = min(rho * RHO_GROWTH, RHO_MAX)
        z_tensor = z.reshape(shape)
        for k in active:
            unfolded = _unfold(z_tensor - dual[k] / rho, k)
            x[k] = _fold(_svt(unfolded, float(weights[k]) / rho, p.theta), k, shape)

        estimate = np.zeros(shape, dtype=np.float32)
        consensus = np.zeros(shape, dtype=np.float32)
        for k in active:
            estimate += weights[k] * x[k]
            consensus += x[k] + dual[k] / rho
        estimate_mat = estimate.reshape(n_series, n_time)
        update = (consensus / len(active)).reshape(n_series, n_time)

        if use_ar:
            ar = _ar_prediction(estimate_mat, z, lags)
            c = p.ar_weight
            update[:, max_lag:] = (update[:, max_lag:] + c * ar) / (1 + c)

        z = np.where(missing, update, z).astype(np.float32)
        z_tensor = z.reshape(shape)
        for k in active:
            dual[k] += rho * (x[k] - z_tensor)

        change = float(np.linalg.norm(z - previous)) / norm
        previous = z.copy()
        if change < p.tol:
            break
    return z


def _unfold(tensor: Float32Array, mode: int) -> Float32Array:
    """Mode-k unfolding: rows = the k-th direction, columns = everything else."""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _fold(
    matrix: Float32Array, mode: int, shape: tuple[int, int, int]
) -> Float32Array:
    """Inverse of ``_unfold``."""
    moved = [shape[mode], *(s for i, s in enumerate(shape) if i != mode)]
    return np.moveaxis(matrix.reshape(moved), 0, mode)


def _svt(matrix: Float32Array, tau: float, theta: float) -> Float32Array:
    """Singular value thresholding with truncation.

    Shrinks every singular value by ``tau`` except the largest ``theta`` share,
    which are kept as they are (truncated nuclear norm). Uses the small Gram
    matrix instead of a full SVD, which is much faster for wide matrices.
    """
    m, n = matrix.shape
    if m > n:
        return _svt(matrix.T, tau, theta).T
    gram = (matrix @ matrix.T).astype(np.float64)
    eigval, vecs = np.linalg.eigh(gram)
    eigval, vecs = eigval[::-1], vecs[:, ::-1]
    s = np.sqrt(np.clip(eigval, 0.0, None))
    shrunk = np.maximum(s - tau, 0.0)
    keep = int(np.ceil(theta * m))
    shrunk[:keep] = s[:keep]
    idx = np.flatnonzero((shrunk > 0) & (s > 0))
    if idx.size == 0:
        return np.zeros_like(matrix)
    u = vecs[:, idx].astype(np.float32)
    factor = (shrunk[idx] / s[idx]).astype(np.float32)
    return (u * factor) @ (u.T @ matrix)


def _ar_prediction(
    estimate: Float32Array, z: Float32Array, lags: IntArray
) -> Float32Array:
    """Per-station AR fit and prediction for steps max_lag..end.

    Regressors are the lagged low-rank estimate, the response is the current
    completed matrix (as in LATC). Returns an array of shape
    (stations, n_time - max_lag).
    """
    n_time = estimate.shape[1]
    max_lag = int(lags.max())
    lagged = [estimate[:, max_lag - int(h) : n_time - int(h)] for h in lags]
    target = z[:, max_lag:]
    d = len(lagged)

    gram = np.empty((estimate.shape[0], d, d), dtype=np.float64)
    rhs = np.empty((estimate.shape[0], d), dtype=np.float64)
    for i in range(d):
        rhs[:, i] = (lagged[i] * target).sum(axis=1, dtype=np.float64)
        for j in range(i, d):
            gram[:, i, j] = (lagged[i] * lagged[j]).sum(axis=1, dtype=np.float64)
            gram[:, j, i] = gram[:, i, j]
    ridge = 1e-6 * np.trace(gram, axis1=1, axis2=2) + 1e-9
    gram[:, np.arange(d), np.arange(d)] += ridge[:, None]
    coef = np.linalg.solve(gram, rhs[..., None])[..., 0].astype(np.float32)

    prediction = np.zeros_like(target)
    for i in range(d):
        prediction += coef[:, i : i + 1] * lagged[i]
    return prediction
