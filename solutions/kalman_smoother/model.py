"""Task 1: profile + Kalman (RTS) smoother on same-day residuals.

Same profile as ``residual_interp``; only the residual step differs. Each
station's deviation from its profile is modelled as an AR(1) process observed
with noise:

    x_t = phi * x_{t-1} + w_t,   w_t ~ N(0, q)      true deviation from normal
    y_t = x_t + v_t,             v_t ~ N(0, r_t)    visible measurement (if any)

A forward Kalman filter plus a backward Rauch-Tung-Striebel pass gives the best
linear estimate of x_t at every masked cell, using data before and after it.
phi, q and r are fitted per station series from the residual autocovariances,
instead of a hand-tuned tau. prediction = profile + smoothed deviation.

Local experiment:  bench experiment kalman_smoother --task state --benchmark data/benchmarks/quick_v2
Tune:              ... --params '{"fit_lag": 3}'  or  '{"phi": 0.95}'
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field

from solutions import baseline
from solutions.residual_interp.model import (
    CELL,
    SERIES,
    STEP_SECONDS,
    Parameters,
    _add_profile,  
    _fit_profile,  
)
from trafficbench import table_types
from trafficbench.contracts import KEYS, VALUES, Panel, ReleasePackageSlice, Split
from trafficbench.panelwise import map_panels

FloatArray = NDArray[np.float64]


class KalmanParameters(Parameters):
    """Profile settings from ``residual_interp`` plus the smoother's settings.

    Attributes:
        fit_lag: Lag k (5-min steps) used to estimate persistence:
            phi = (gamma(2k) / gamma(k)) ** (1/k). A larger k targets the slow
            component of the deviation; faster wiggles count as noise.
        phi: Optional fixed persistence per step, overriding the fit.
        min_pairs: Minimum lag pairs for a per-station fit, else panel-wide fit.
        noise_by_coverage: Scale measurement noise by 100 / pct_observed, so
            partially covered intervals count as weaker evidence.
        tau: Unused here (kept from the shared profile settings).
    """

    fit_lag: int = Field(default=6, ge=1, le=72)
    phi: float | None = Field(default=None, gt=0, lt=1)
    min_pairs: int = Field(default=200, ge=10)
    min_pct: float = Field(default=25, ge=0, le=100)
    noise_by_coverage: bool = False


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow as profile + Kalman-smoothed residual."""
    p = KalmanParameters.model_validate(dict(release_slice.parameters))
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
    p: KalmanParameters,
) -> dict[Split, table_types.StateFrame]:
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)
    obs = _add_profile(observations, profile, link_means, panel_means, p)

    # Place every cell on a regular grid: row = 5-minute step, column = station series.
    first = obs.select(pl.col("timestamp").min()).item()
    series_ids = (
        obs.select(SERIES).unique().sort(SERIES).with_row_index("sid", offset=0)
    )
    obs = obs.join(series_ids, on=SERIES, how="left").with_columns(
        ((pl.col("timestamp") - first).dt.total_seconds() // STEP_SECONDS)
        .cast(pl.Int64)
        .alias("step")
    )
    steps = obs["step"].to_numpy()
    sids = obs["sid"].to_numpy().astype(np.int64)
    shape = (int(steps.max()) + 1, len(series_ids))
    pct = obs["pct_observed"].cast(pl.Float64).fill_null(0.0).to_numpy()

    modes = {"speed_kmh": p.speed_mode, "flow_vph": p.flow_mode}
    for value in VALUES["state"]:
        y_obs, prof = _residuals(obs, value, modes[value], p)
        anchor = np.isfinite(y_obs) & (pct >= p.min_pct)
        y = np.full(shape, np.nan)
        y[steps[anchor], sids[anchor]] = y_obs[anchor]
        noise_scale = np.ones(shape)
        if p.noise_by_coverage:
            noise_scale[steps[anchor], sids[anchor]] = 100.0 / np.clip(
                pct[anchor], 1.0, 100.0
            )
        phi, q, r = _fit_ar1(y, p)
        smoothed = _rts_smooth(y, noise_scale, phi, q, r)
        r_hat = smoothed[steps, sids]
        obs = obs.with_columns(
            pl.Series(f"pred_{value}", _back_transform(prof, r_hat, modes[value], p))
        )

    obs = obs.select([*CELL, *[f"pred_{v}" for v in VALUES["state"]]])
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
            targets.join(obs, on=CELL, how="left")
            .with_columns(
                pl.coalesce(f"pred_{v}", f"p_{v}").clip(lower_bound=0).alias(v)
                for v in VALUES["state"]
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )
    return predictions


def _residuals(
    obs: pl.DataFrame, value: str, mode: str, p: KalmanParameters
) -> tuple[FloatArray, FloatArray]:
    """Deviation from the profile (log-ratio in multiplicative mode)."""
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
    prof: FloatArray, r_hat: FloatArray, mode: str, p: KalmanParameters
) -> FloatArray:
    if mode == "mult":
        c = p.offset
        pred = (np.clip(prof, 0, None) + c) * np.exp(np.clip(r_hat, -1.6, 1.6)) - c
    else:
        pred = prof + r_hat
    return np.clip(pred, 0, None)


def _lag_moment(y: FloatArray, k: int) -> tuple[FloatArray, FloatArray]:
    """Per-column mean of y_t * y_{t-k} over pairs where both are observed."""
    prod = y[k:] * y[:-k] if k > 0 else y * y
    count = np.isfinite(prod).sum(axis=0).astype(float)
    total = np.nansum(prod, axis=0)
    return total, count


def _fit_ar1(y: FloatArray, p: KalmanParameters) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit phi, process noise q and measurement noise r for each column.

    Uses autocovariances gamma(0), gamma(k), gamma(2k) of the observed
    residuals: phi^k = gamma(2k)/gamma(k), signal variance = gamma(k)/phi^k,
    measurement noise = gamma(0) - signal variance. Columns with too few pairs
    use the pooled (panel-wide) estimate.
    """
    k = p.fit_lag
    moments = [_lag_moment(y, lag) for lag in (0, k, 2 * k)]

    def estimate(sums: list[FloatArray], counts: list[FloatArray]) -> tuple[FloatArray, ...]:
        g0, gk, g2k = (s / np.maximum(c, 1.0) for s, c in zip(sums, counts, strict=True))
        g0 = np.maximum(g0, 1e-9)
        if p.phi is not None:
            phi = np.full_like(g0, p.phi)
        else:
            ratio = np.where((gk > 0) & (g2k > 0), g2k / np.where(gk > 0, gk, 1.0), np.nan)
            phi = np.clip(np.power(ratio, 1.0 / k), 0.5, 0.9995)
            phi = np.where(np.isfinite(phi), phi, 0.95)
        signal = np.clip(np.where(gk > 0, gk, 0.5 * g0) / phi**k, 0.05 * g0, 0.99 * g0)
        noise = np.maximum(g0 - signal, 0.01 * g0)
        return phi, signal * (1 - phi**2), noise

    per_column = estimate([m[0] for m in moments], [m[1] for m in moments])
    pooled = estimate(
        [np.atleast_1d(m[0].sum()) for m in moments],
        [np.atleast_1d(m[1].sum()) for m in moments],
    )
    enough = np.minimum(moments[1][1], moments[2][1]) >= p.min_pairs
    phi, q, r = (np.where(enough, col, pool[0]) for col, pool in zip(per_column, pooled, strict=True))
    return phi, q, r


def _rts_smooth(
    y: FloatArray, noise_scale: FloatArray, phi: FloatArray, q: FloatArray, r: FloatArray
) -> FloatArray:
    """Kalman filter + RTS smoother, vectorised across columns (station series).

    Missing values (NaN) are skipped in the update, so the estimate decays
    toward 0 (= the profile) across gaps at rate phi per step.
    """
    n_steps, n_series = y.shape
    pred_mean = np.zeros((n_steps, n_series))
    pred_var = np.zeros((n_steps, n_series))
    filt_mean = np.zeros((n_steps, n_series))
    filt_var = np.zeros((n_steps, n_series))
    stationary_var = q / (1 - phi**2)

    m, v = np.zeros(n_series), stationary_var.copy()
    for t in range(n_steps):
        if t:
            m, v = phi * m, phi**2 * v + q
        pred_mean[t], pred_var[t] = m, v
        observed = np.isfinite(y[t])
        gain = v / (v + r * noise_scale[t])
        m = np.where(observed, m + gain * (np.nan_to_num(y[t]) - m), m)
        v = np.where(observed, (1 - gain) * v, v)
        filt_mean[t], filt_var[t] = m, v

    smoothed = filt_mean.copy()
    for t in range(n_steps - 2, -1, -1):
        c = filt_var[t] * phi / pred_var[t + 1]
        smoothed[t] = filt_mean[t] + c * (smoothed[t + 1] - pred_mean[t + 1])
    return smoothed
