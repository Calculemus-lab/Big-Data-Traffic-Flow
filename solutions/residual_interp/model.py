"""Statistical approach that:
1) Improves the provided baseline 
2) On top of that interpolates the same-day residuals

Local experiment:  bench experiment residual_interp --task state --benchmark data/benchmarks/quick
Tune:              ... --params '{"tau": 20, "flow_mode": "mult"}'
Release targets:   bench run residual_interp --task state --target-range validation
"""
 
from __future__ import annotations
 
from collections.abc import Mapping
from functools import partial
from typing import Literal
 
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
 
from solutions import baseline
from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    QUEUE_INTERVAL_MINUTES,
    VALUES,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels
 
SERIES = ["station_id", "link_id"]
CELL = ["timestamp", "station_id", "link_id"]
SLOTS_PER_DAY = 24 * 60 // QUEUE_INTERVAL_MINUTES  # 288 five-minute slots
STEP_SECONDS = QUEUE_INTERVAL_MINUTES * 60
 
 
class Parameters(BaseModel):
    """Validated settings. Unknown keys are rejected to catch typos in --params.
 
    Attributes:
        smooth: Average each profile slot with +-k neighbouring slots.
        half_life_days: Weight of a day halves every this many days (0 = equal).
        day_groups: "dow" = 7 day types; "pooled" = Mon-Fri share one profile.
        tau: Decay of an anchor's weight, in 5-minute steps.
        min_pct: Minimum pct_observed for profile data and anchors.
        speed_mode, flow_mode: Additive or multiplicative (log-ratio) deviation.
        offset: Added before ratios in multiplicative mode so low values stay stable.
    """
 
    model_config = ConfigDict(strict=True, extra="forbid")
    smooth: int = Field(default=1, ge=0, le=12)
    half_life_days: float = Field(default=56, ge=0)
    day_groups: Literal["dow", "pooled"] = "dow"
    tau: float = Field(default=20, gt=0)
    min_pct: float = Field(default=75, ge=0, le=100)
    speed_mode: Literal["add", "mult"] = "add"
    flow_mode: Literal["add", "mult"] = "mult"
    offset: float = Field(default=100.0, gt=0)
 
 
def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Predict Task 1 speed and flow as profile + interpolated residual."""
    p = Parameters.model_validate(dict(release_slice.parameters))
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
    p: Parameters,
) -> dict[Split, table_types.StateFrame]:
    # All published mainline cells for this panel, with known historical
    # Task 1 answers restored. Masked cells have null speed/flow.
    observations = baseline.state_observations(release_slice, panel)
    profile, link_means, panel_means = _fit_profile(observations, p)
 
    # Interpolate on every observed cell once; targets then just look up their row.
    obs = _add_profile(observations, profile, link_means, panel_means, p).sort(
        [*SERIES, "timestamp"]
    )
    first = obs.select(pl.col("timestamp").min()).item()
    obs = obs.with_columns(
        ((pl.col("timestamp") - first).dt.total_seconds() / STEP_SECONDS).alias("step")
    )
    obs = obs.with_columns(
        _interpolate("speed_kmh", p.speed_mode, p).alias("pred_speed_kmh"),
        _interpolate("flow_vph", p.flow_mode, p).alias("pred_flow_vph"),
    ).select([*CELL, "pred_speed_kmh", "pred_flow_vph"])
 
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
                # Fall back to the profile if a target has no matching observed row.
                pl.coalesce(f"pred_{v}", f"p_{v}").clip(lower_bound=0).alias(v)
                for v in VALUES["state"]
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )
    return predictions
 
 
def _day_slot(p: Parameters) -> list[pl.Expr]:
    weekday = pl.col("timestamp").dt.weekday()  # 1 = Monday ... 7 = Sunday
    day = (
        weekday
        if p.day_groups == "dow"
        else pl.when(weekday >= 6).then(weekday).otherwise(1)
    )
    slot = (
        pl.col("timestamp").dt.hour().cast(pl.Int32) * (60 // QUEUE_INTERVAL_MINUTES)
        + pl.col("timestamp").dt.minute().cast(pl.Int32) // QUEUE_INTERVAL_MINUTES
    )
    return [day.cast(pl.Int32).alias("day"), slot.alias("slot")]
 
 
def _fit_profile(
    observations: pl.DataFrame, p: Parameters
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Recency-weighted, slot-smoothed mean per (station, link, day type, slot)."""
    values = list(VALUES["state"])
    tr = observations.filter(
        pl.col("pct_observed").ge(p.min_pct)
        & pl.col("speed_kmh").is_not_null()
        & pl.col("flow_vph").is_not_null()
    ).with_columns(_day_slot(p))
    if p.half_life_days > 0:
        age_days = (
            pl.col("timestamp").max() - pl.col("timestamp")
        ).dt.total_seconds() / 86400
        tr = tr.with_columns((0.5 ** (age_days / p.half_life_days)).alias("w"))
    else:
        tr = tr.with_columns(pl.lit(1.0).alias("w"))
 
    keys = [*SERIES, "day", "slot"]
    sums = tr.group_by(keys).agg(
        pl.col("w").sum().alias("sw"),
        *[(pl.col("w") * pl.col(v)).sum().alias(f"swv_{v}") for v in values],
    )
    # Circular smoothing: each slot also counts its +-k neighbours (23:55 wraps to 00:00).
    shifted = pl.concat(
        sums.with_columns(((pl.col("slot") + s) % SLOTS_PER_DAY).alias("slot"))
        for s in range(-p.smooth, p.smooth + 1)
    )
    profile = (
        shifted.group_by(keys)
        .agg(pl.all().sum())
        .select(
            *keys,
            *[(pl.col(f"swv_{v}") / pl.col("sw")).alias(f"p_{v}") for v in values],
        )
    )
    link_means = tr.group_by("link_id").agg(
        pl.col(v).mean().alias(f"link_{v}") for v in values
    )
    panel_means = tr.select(pl.col(v).mean().alias(f"panel_{v}") for v in values)
    return profile, link_means, panel_means
 
 
def _add_profile(
    frame: pl.DataFrame,
    profile: pl.DataFrame,
    link_means: pl.DataFrame,
    panel_means: pl.DataFrame,
    p: Parameters,
) -> pl.DataFrame:
    """Attach the profile value, falling back to link mean, then panel mean."""
    values = VALUES["state"]
    return (
        frame.with_columns(_day_slot(p))
        .join(profile, on=[*SERIES, "day", "slot"], how="left")
        .join(link_means, on="link_id", how="left")
        .join(panel_means, how="cross")
        .with_columns(
            pl.coalesce(f"p_{v}", f"link_{v}", f"panel_{v}")
            .fill_null(0.0)
            .alias(f"p_{v}")
            for v in values
        )
        .drop(
            "day",
            "slot",
            *[f"link_{v}" for v in values],
            *[f"panel_{v}" for v in values],
        )
    )
 
 
def _interpolate(value: str, mode: str, p: Parameters) -> pl.Expr:
    """Profile + residual from the nearest anchors before/after on the same series."""
    prof, obs_value, c = pl.col(f"p_{value}"), pl.col(value), p.offset
    anchor = pl.col("pct_observed").ge(p.min_pct) & obs_value.is_not_null()
 
    if mode == "mult":
        resid = (
            (obs_value.clip(lower_bound=0) + c) / (prof.clip(lower_bound=0) + c)
        ).log()
    else:
        resid = obs_value - prof
    r = pl.when(anchor).then(resid)
    t = pl.when(anchor).then(pl.col("step"))
 
    prev_r, next_r = r.forward_fill().over(SERIES), r.backward_fill().over(SERIES)
    prev_t, next_t = t.forward_fill().over(SERIES), t.backward_fill().over(SERIES)
    w_prev = (-(pl.col("step") - prev_t) / p.tau).exp().fill_null(0.0)
    w_next = (-(next_t - pl.col("step")) / p.tau).exp().fill_null(0.0)
    total = w_prev + w_next
    blended = (
        w_prev * prev_r.fill_null(0.0) + w_next * next_r.fill_null(0.0)
    ) / pl.when(total > 0).then(total)
    # Shrink toward the profile (residual 0) when the nearest anchor is far away.
    r_hat = (blended * pl.max_horizontal(w_prev, w_next)).fill_null(0.0)
 
    if mode == "mult":
        pred = (prof.clip(lower_bound=0) + c) * r_hat.clip(-1.6, 1.6).exp() - c
    else:
        pred = prof + r_hat
    return pred.clip(lower_bound=0)
