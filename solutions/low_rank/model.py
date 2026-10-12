"""Task 1: profile + low-rank matrix completion (SoftImpute) of the residuals.

Same profile as the other solutions (here with the best profile settings found
so far: pooled weekdays, smooth 2, min_pct 25). The deviations from the profile
are arranged as a matrix with missing entries and completed by assuming the
matrix is approximately low-rank:

    layout "day"  : rows = days,          columns = (station, 5-min slot of day)
                    -> each day is a mix of a few typical daily patterns
    layout "time" : rows = 5-min steps,   columns = stations
                    -> the whole corridor moves in a few typical spatial patterns

SoftImpute: fill the gaps with 0 (= the profile), take an SVD, shrink the
singular values by lambda (and keep at most ``rank`` of them), put the observed
values back, and repeat until the filled-in values stop changing.

Local experiment:  bench experiment low_rank --task state --benchmark data/benchmarks/quick_v2
Tune:              ... --params '{"layout": "time"}'   or  '{"rank": 10, "shrink": 0.05}'
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
SLOTS_PER_DAY = 24 * 3600 // STEP_SECONDS  # 288


class LowRankParameters(Parameters):
    """Profile settings plus the matrix-completion settings.

    Attributes:
        layout: "day" (days x station-slots) or "time" (5-min steps x stations).
        rank: Maximum number of components kept.
        shrink: Soft threshold as a fraction of the largest singular value.
        n_iter: Maximum SoftImpute iterations.
        tol: Stop when the filled-in values change less than this (relative).
    """

    min_pct: float = Field(default=25, ge=0, le=100)
    day_groups: Literal["dow", "pooled"] = "pooled"
    smooth: int = Field(default=2, ge=0, le=12)
    layout: Literal["day", "time"] = "time"
    rank: int = Field(default=20, ge=1, le=500)
    shrink: float = Field(default=0.02, ge=0, lt=1)
    n_iter: int = Field(default=50, ge=1, le=500)
    tol: float = Field(default=1e-4, gt=0)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow as profile + low-rank completed residual."""
    p = LowRankParameters.model_validate(solution_parameters)
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
    p: LowRankParameters,
) -> dict[Split, table_types.StateFrame]:
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)
    obs = _add_profile(observations, profile, link_means, panel_means, p)

    series = obs.select(SERIES).unique().sort(SERIES).with_row_index("sid")
    first = obs.select(pl.col("timestamp").min()).item()
    first_midnight = first.replace(hour=0, minute=0, second=0, microsecond=0)
    obs = (
        obs.join(series, on=SERIES, how="left")
        .with_columns(
            ((pl.col("timestamp") - first).dt.total_seconds() // STEP_SECONDS)
            .cast(pl.Int64)
            .alias("step"),
            ((pl.col("timestamp") - first_midnight).dt.total_days())
            .cast(pl.Int64)
            .alias("day"),
            (
                pl.col("timestamp").dt.hour().cast(pl.Int64) * 12
                + pl.col("timestamp").dt.minute().cast(pl.Int64) // 5
            ).alias("slot"),
        )
        .with_row_index("row")
    )
    sids = obs["sid"].to_numpy().astype(np.int64)
    n_series = len(series)
    if p.layout == "day":
        rows = obs["day"].to_numpy().astype(np.int64)
        cols = sids * SLOTS_PER_DAY + obs["slot"].to_numpy().astype(np.int64)
        shape = (int(rows.max()) + 1, n_series * SLOTS_PER_DAY)
    else:
        rows = obs["step"].to_numpy().astype(np.int64)
        cols = sids
        shape = (int(rows.max()) + 1, n_series)
    pct = obs["pct_observed"].cast(pl.Float64).fill_null(0.0).to_numpy()

    target_keys = pl.concat(
        [t.select(CELL).collect() for t in target_templates_by_split.values()]
    ).unique()
    target_table = obs.select([*CELL, "row"]).join(target_keys, on=CELL, how="inner")
    target_rows = target_table["row"].to_numpy().astype(np.int64)

    modes = {"speed_kmh": p.speed_mode, "flow_vph": p.flow_mode}
    predictions_at_targets: dict[str, FloatArray] = {}
    for value in VALUES["state"]:
        y_obs, prof = _residuals(obs, value, modes[value], p)
        anchor = np.isfinite(y_obs) & (pct >= p.min_pct)
        matrix = np.full(shape, np.nan, dtype=np.float32)
        matrix[rows[anchor], cols[anchor]] = y_obs[anchor]
        completed = _soft_impute(matrix, p)
        r_hat = completed[rows[target_rows], cols[target_rows]].astype(np.float64)
        visible = anchor[target_rows]
        r_hat[visible] = y_obs[target_rows][visible]
        predictions_at_targets[value] = _back_transform(
            prof[target_rows], r_hat, modes[value], p
        )
        del matrix, completed

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
    obs: pl.DataFrame, value: str, mode: str, p: LowRankParameters
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
    prof: FloatArray, r_hat: FloatArray, mode: str, p: LowRankParameters
) -> FloatArray:
    """Profile + deviation, back in speed/flow units."""
    if mode == "mult":
        c = p.offset
        pred = (np.clip(prof, 0, None) + c) * np.exp(np.clip(r_hat, -1.6, 1.6)) - c
    else:
        pred = prof + r_hat
    return np.clip(pred, 0, None)


def _soft_impute(matrix: Float32Array, p: LowRankParameters) -> Float32Array:
    """Low-rank completion of a matrix with NaN gaps (SoftImpute).

    Returns the low-rank estimate for every entry. Missing entries start at 0,
    i.e. at the profile, so rows or columns without any data stay at 0.
    """
    observed = np.isfinite(matrix)
    values = np.where(observed, matrix, 0.0).astype(np.float32)
    filled = values.copy()
    if not observed.any():
        return filled

    lam: float | None = None
    low_rank = filled
    previous = np.zeros(int((~observed).sum()), dtype=np.float32)
    for _ in range(p.n_iter):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        if lam is None:
            lam = p.shrink * float(s[0])
        s = np.maximum(s[: p.rank] - lam, 0.0)
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
        if change < p.tol:
            break
    return low_rank.astype(np.float32)
