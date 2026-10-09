"""CatBoost residuals on switching-mode mixture Kalman baselines.

GPU SMM: ``uv pip install "cupy-cuda13x[ctk]"`` (CUDA 13) or ``cupy-cuda12x[ctk]`` (CUDA 12).

  uv run bench run catboost_smm --task state --profile quick \\
    --params '{"smm_device": "auto", "catboost_device": "auto"}'

  Default policy: **fast SMM** (small ``M``, coarse strides), **slow CatBoost** (many trees/samples).
  Non-mainline target links use the slot baseline; mainline uses SMM + CatBoost.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from tqdm import tqdm

from solutions.switch_mode_model.model import MixtureKalmanEstimator, _PARAMS as MKF_PARAMS

from .features import (
    FeaturePanel,
    _chain_topology,
    feature_rows,
    matrices,
    sampled_targets,
    simulated_mask,
)

MKF_DEFAULTS = {
    "M": 2,
    "eps": 1e-3,
    "sigma_v": (2.0, 5.0),
    "sigma_w": (5.0, 8.0),
    "trans": 0.95,
    "sigma_walk": 50.0,
}

DEFAULTS = {
    **MKF_DEFAULTS,
    "reps": 2,
    "samples_per_day": 300,
    "iterations_d8": 800,
    "iterations_d10": 450,
    "early_stopping_rounds": 100,
    "learning_rate": 0.04,
    "valid_days": 5,
    "threads": 8,
    "catboost_device": "auto",
    "gpu_device": "0",
    "smm_device": "auto",
    "smm_day_stride": 4,
    "smm_step_stride": 4,
    "smm_infer_step_stride": 6,
}


def _smm_runtime_params(ctx) -> dict:
    p = _params(ctx)
    return {"smm_device": p["smm_device"], "gpu_device": p["gpu_device"]}


def _catboost_task_type(ctx) -> tuple[str, str | None]:
    choice = str(_param(ctx, "catboost_device")).lower()
    if choice == "auto":
        choice = "gpu" if get_gpu_device_count() > 0 else "cpu"
    if choice in ("gpu", "cuda"):
        return "GPU", str(_params(ctx).get("gpu_device", DEFAULTS["gpu_device"]))
    return "CPU", None


def _params(ctx) -> dict:
    merged = dict(DEFAULTS)
    if isinstance(getattr(ctx, "params", None), dict):
        merged.update(ctx.params)
    return merged


def _param(ctx, name):
    return _params(ctx)[name]


def _mkf_params(ctx) -> dict:
    """Mixture-Kalman hyperparameters (MKF_DEFAULTS, overridable via ctx.params)."""
    merged = dict(MKF_DEFAULTS)
    raw = _params(ctx)
    for name in MKF_PARAMS:
        if name in raw:
            merged[name] = raw[name]
    return merged


def _upsample_smm(
    speed: np.ndarray,
    flow: np.ndarray,
    coarse_times,
    fine_times,
    link_ids,
) -> tuple[np.ndarray, np.ndarray]:
    c_idx = _time_index(coarse_times)
    f_idx = _time_index(fine_times)
    if len(c_idx) == len(f_idx):
        return speed, flow
    cols = list(link_ids)
    sp = pd.DataFrame(speed, index=c_idx, columns=cols)
    fp = pd.DataFrame(flow, index=c_idx, columns=cols)
    sp = sp.reindex(f_idx).interpolate(method="time").bfill().ffill()
    fp = fp.reindex(f_idx).interpolate(method="time").bfill().ffill()
    return sp.to_numpy(dtype=np.float32), fp.to_numpy(dtype=np.float32)


def _obs_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["link_id"] = out["link_id"].astype(str)
    for col in ("speed_kmh", "flow_vph", "pct_observed"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    if "pct_observed" not in out.columns:
        out["pct_observed"] = np.where(out["speed_kmh"].notna() & out["flow_vph"].notna(), 100.0, 0.0)
    return out


def _time_index(times) -> pd.DatetimeIndex:
    if isinstance(times, pd.DatetimeIndex):
        return times
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True))


def _network_for_mkf(ctx) -> dict:
    network = {k: v.copy(deep=True) for k, v in ctx.network.items()}
    if "lwr_mainline_topology" in network:
        return network
    fd = network["fd_parameters"].copy()
    fd["link_id"] = fd["link_id"].astype(str)
    links = fd.drop_duplicates("link_id")
    if "links" in network and "milepost" in network["links"].columns:
        order = (
            network["links"]
            .astype({"link_id": str})
            .drop_duplicates("link_id")
            .sort_values("milepost")
            .link_id.tolist()
        )
        links = links.set_index("link_id").reindex(order).reset_index()
    link_ids = links.link_id.tolist()
    network["lwr_mainline_topology"] = _chain_topology(link_ids)
    return network


def _fit_mkf(ctx) -> MixtureKalmanEstimator:
    model = MixtureKalmanEstimator()
    model.set_params(**_mkf_params(ctx))
    mkf_ctx = replace(ctx, network=_network_for_mkf(ctx))
    model.fit(mkf_ctx)
    return model


def _smm_grid(
    est: MixtureKalmanEstimator,
    ctx,
    observations: pd.DataFrame,
    times,
    *,
    time_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fitted filter on one timeline; return speed and flow (n_time, n_link)."""
    obs = _obs_frame(observations)
    index = _time_index(times)
    stride = max(1, int(time_stride))
    if stride > 1 and len(index) > 1:
        coarse_index = index[::stride]
    else:
        coarse_index = index
    if obs.empty or len(index) == 0:
        n = len(est.link_ids_)
        empty = np.full((len(index), n), np.nan, dtype=np.float32)
        return empty, empty
    if stride > 1:
        obs = obs.loc[obs["timestamp"].isin(coarse_index)]
    targets = pd.MultiIndex.from_product([coarse_index, est.link_ids_], names=["timestamp", "link_id"]).to_frame(index=False)
    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True)
    targets["link_id"] = targets["link_id"].astype(str)
    mini = replace(
        ctx,
        observations=obs,
        targets=targets,
        network=_network_for_mkf(ctx),
        params=_smm_runtime_params(ctx),
    )
    pred = est.predict(mini)
    speed = pred.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=coarse_index, columns=est.link_ids_)
    flow = pred.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=coarse_index, columns=est.link_ids_)
    speed_c = speed.to_numpy(dtype=np.float32)
    flow_c = flow.to_numpy(dtype=np.float32)
    if len(coarse_index) != len(index):
        speed_c, flow_c = _upsample_smm(speed_c, flow_c, coarse_index, index, est.link_ids_)
    return speed_c, flow_c


def _panel(ctx, est: MixtureKalmanEstimator) -> FeaturePanel:
    key = "panel_obj"
    if isinstance(ctx.cache, dict) and key in ctx.cache:
        return ctx.cache[key]
    panel = FeaturePanel(ctx.network, list(est.link_ids_), panel_no=0)
    if isinstance(ctx.cache, dict):
        ctx.cache[key] = panel
    return panel


def _train_day_sample(ctx, est, panel, train, day, seed_base, per_day, holdout):
    day_obs = train.loc[train["timestamp"].dt.floor("D").eq(day)]
    times, speed, flow, elig, _ = matrices(day_obs, panel)
    if not elig.any():
        return None
    mask, regime = simulated_mask(ctx.panel, seed_base, times, speed, flow, elig)
    rows, cols = sampled_targets(mask, seed_base, times, per_day)
    if len(rows) == 0:
        return None
    masked_speed = np.where(mask, np.nan, speed)
    masked_flow = np.where(mask, np.nan, flow)
    masked_frame = day_obs.copy()
    if mask.any():
        masked_frame = masked_frame.set_index(["timestamp", "link_id"])
        t_index = _time_index(times)
        hide = pd.MultiIndex.from_arrays(
            [t_index[np.where(mask)[0]], np.asarray(panel.links)[np.where(mask)[1]]],
            names=["timestamp", "link_id"],
        )
        present = hide.intersection(masked_frame.index)
        if len(present):
            masked_frame.loc[present, ["speed_kmh", "flow_vph"]] = np.nan
        masked_frame = masked_frame.reset_index()

    step_stride = max(1, int(_param(ctx, "smm_step_stride")))
    smm_v, smm_q = _smm_grid(est, ctx, masked_frame, times, time_stride=step_stride)
    X, bv, bq = feature_rows(
        panel,
        times,
        masked_speed,
        masked_flow,
        elig,
        smm_v,
        smm_q,
        rows,
        cols,
        regime[rows],
    )
    rv = speed[rows, cols] - bv
    rq = flow[rows, cols] / panel.lanes[cols] - bq
    split = "val" if day in holdout else "train"
    return split, X, rv, rq


def _train_models(ctx, est: MixtureKalmanEstimator, panel: FeaturePanel) -> dict:
    train = _obs_frame(ctx.train)
    if train.empty:
        raise ValueError("Training observations are empty")
    seed = int(ctx.seed)
    reps = int(_param(ctx, "reps"))
    per_day = int(_param(ctx, "samples_per_day"))
    valid_days = int(_param(ctx, "valid_days"))
    threads = int(_param(ctx, "threads"))
    days = sorted(train["timestamp"].dt.floor("D").unique())
    if len(days) <= valid_days:
        valid_days = max(1, len(days) // 5)
    holdout = set(days[-valid_days:])
    day_stride = max(1, int(_param(ctx, "smm_day_stride")))
    smm_days = days[::day_stride] if day_stride > 1 else list(days)
    job_days = sorted(set(smm_days) | holdout)

    x_train, yv_train, yq_train = [], [], []
    x_val, yv_val, yq_val = [], [], []

    jobs = [(rep, day) for rep in range(reps) for day in job_days]
    day_label = f"{ctx.panel} SMM train days"
    results = []
    for rep, day in tqdm(jobs, desc=day_label, unit="day", mininterval=0.5):
        seed_base = seed + rep * 7919
        results.append(_train_day_sample(ctx, est, panel, train, day, seed_base, per_day, holdout))

    for item in results:
        if item is None:
            continue
        split, X, rv, rq = item
        if split == "val":
            x_val.append(X)
            yv_val.append(rv)
            yq_val.append(rq)
        else:
            x_train.append(X)
            yv_train.append(rv)
            yq_train.append(rq)

    if not x_train:
        raise ValueError("No training samples collected for CatBoost")

    X_tr = np.concatenate(x_train)
    yv_tr = np.concatenate(yv_train)
    yq_tr = np.concatenate(yq_train)
    has_val = bool(x_val)
    if has_val:
        X_va = np.concatenate(x_val)
        yv_va = np.concatenate(yv_val)
        yq_va = np.concatenate(yq_val)
        eval_speed = (X_va, yv_va)
        eval_flow = (X_va, yq_va)
    else:
        eval_speed = eval_flow = None

    models = {}
    specs = (
        ("speed", yv_tr, eval_speed, "iterations_d8", 8),
        ("speed", yv_tr, eval_speed, "iterations_d10", 10),
        ("flow", yq_tr, eval_flow, "iterations_d8", 8),
        ("flow", yq_tr, eval_flow, "iterations_d10", 10),
    )
    task_type, devices = _catboost_task_type(ctx)
    cb_label = f"{ctx.panel} CatBoost ({task_type})"
    for channel, y_tr, eval_set, iter_key, depth in tqdm(specs, desc=cb_label, unit="model"):
        tag = f"{channel}_d{depth}"
        kwargs = dict(
            iterations=int(_param(ctx, iter_key)),
            depth=depth,
            learning_rate=float(_param(ctx, "learning_rate")),
            l2_leaf_reg=10,
            loss_function="RMSE",
            task_type=task_type,
            thread_count=threads,
            random_seed=seed + depth,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            verbose=100 if task_type == "GPU" else False,
        )
        if task_type == "GPU":
            kwargs["devices"] = devices
        if has_val:
            kwargs["early_stopping_rounds"] = int(_param(ctx, "early_stopping_rounds"))
        reg = CatBoostRegressor(**kwargs)
        if eval_set is not None:
            reg.fit(X_tr, y_tr, eval_set=eval_set, use_best_model=True)
        else:
            reg.fit(X_tr, y_tr)
        models[tag] = reg
    return models


def _residual(models: dict, X: np.ndarray, channel: str, ctx) -> np.ndarray:
    task_type, _devices = _catboost_task_type(ctx)
    kwargs = {"task_type": task_type} if task_type == "GPU" else {}
    d8 = models[f"{channel}_d8"].predict(X, **kwargs)
    d10 = models[f"{channel}_d10"].predict(X, **kwargs)
    return 0.5 * d8 + 0.5 * d10


def _ensure_trained(ctx) -> tuple[MixtureKalmanEstimator, FeaturePanel, dict]:
    cache = ctx.cache if isinstance(ctx.cache, dict) else {}
    if "catboost_smm" in cache:
        bundle = cache["catboost_smm"]
        return bundle["mkf"], bundle["panel"], bundle["models"]
    mkf = _fit_mkf(ctx)
    panel = _panel(ctx, mkf)
    models = _train_models(ctx, mkf, panel)
    cache["catboost_smm"] = {"mkf": mkf, "panel": panel, "models": models}
    return mkf, panel, models


def state(ctx):
    from solutions import baseline

    mkf, panel, models = _ensure_trained(ctx)
    obs = _obs_frame(ctx.observations)
    if obs.empty:
        raise ValueError("State task requires evaluation observations")
    times, speed, flow, elig, _ = matrices(obs, panel)
    time_index = _time_index(times)

    targets = ctx.targets.copy()
    targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True)
    targets["link_id"] = targets["link_id"].astype(str)
    mainline = set(map(str, mkf.link_ids_))
    out = targets.copy()
    off_mainline = ~targets["link_id"].isin(mainline)
    if off_mainline.any():
        off_pred = baseline.state(replace(ctx, targets=targets.loc[off_mainline]))
        out.loc[off_mainline, ["speed_kmh", "flow_vph"]] = off_pred[["speed_kmh", "flow_vph"]].to_numpy()

    on_mainline = targets["link_id"].isin(mainline)
    if not on_mainline.any():
        return out

    sub = targets.loc[on_mainline]
    infer_stride = max(1, int(_param(ctx, "smm_infer_step_stride")))
    smm_v, smm_q = _smm_grid(mkf, ctx, obs, times, time_stride=infer_stride)

    rows = time_index.get_indexer(sub["timestamp"])
    cols = pd.Index(panel.links).get_indexer(sub["link_id"])
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("Target timestamp/link outside observation timeline")
    regime = sub["mask_regime"].map({"R1": 0, "R2": 1, "R3": 2}).to_numpy(dtype=np.uint8)
    X, bv, bq = feature_rows(
        panel,
        times,
        speed,
        flow,
        elig,
        smm_v,
        smm_q,
        rows,
        cols,
        regime,
    )
    lanes = np.maximum(panel.lanes[cols], 1.0)
    rv = _residual(models, X, "speed", ctx)
    rq = _residual(models, X, "flow", ctx)
    out.loc[on_mainline, "speed_kmh"] = np.clip(bv + rv, 1.0, 180.0)
    out.loc[on_mainline, "flow_vph"] = np.clip((bq + rq) * lanes, 0.0, 150_000.0)
    return out


def queue(ctx):
    from solutions.switch_mode_model.model import queue as smm_queue

    return smm_queue(ctx)


def odme(ctx):
    from solutions.switch_mode_model.model import odme as smm_odme

    return smm_odme(ctx)
