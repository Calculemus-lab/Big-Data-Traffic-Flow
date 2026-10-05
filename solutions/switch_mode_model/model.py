"""
Switching CTM mixture Kalman filter
https://horowitz.me.berkeley.edu/Publications_files/All_papers_numbered/133c_c_Sun_Highway_traffic_state_estimation_CDC_03.pdf
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm import tqdm

_PARAMS = ("M", "eps", "sigma_v", "sigma_w", "trans", "sigma_walk")
_TS = 1.0 / 12.0
_SIGMA_FLOOR = 1.0


def _pair(value, name):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if arr.size != 2 or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError(f"{name} must be two nonnegative standard deviations")
    return arr


def _slots(timestamps):
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    return ts.hour * 12 + ts.minute // 5


def _split_ids(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _log_gauss(innov, S):
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e12
    try:
        quad = float(innov @ np.linalg.solve(S, innov))
    except np.linalg.LinAlgError:
        return -1e12
    if not np.isfinite(quad):
        return -1e12
    return -0.5 * (innov.size * np.log(2.0 * np.pi) + logdet + quad)


def _integrate(A, B, b, n_sub):
    """Compose n_sub CFL-stable steps with a constant input."""
    n = A.shape[0]
    acc = np.zeros((n, n))
    power = np.eye(n)
    for _ in range(n_sub):
        acc += power
        power = A @ power
    return power, acc @ B, acc @ b


def _pivot(frame, times, key_col, keys, value_col):
    out = np.full((len(times), len(keys)), np.nan)
    if frame.empty or value_col not in frame or not keys:
        return out
    grouped = frame.groupby(["timestamp", key_col], sort=False)[value_col].mean()
    index = pd.MultiIndex.from_product([times, keys], names=["timestamp", key_col])
    return grouped.reindex(index).to_numpy(dtype=float).reshape(len(times), len(keys))


class MixtureKalmanEstimator:
    """Mixture Kalman filter on a two-mode cell transmission model."""

    def __init__(self, M=10, eps=1e-3, sigma_v=(2.0, 5.0), sigma_w=(5.0, 8.0), trans=0.95, sigma_walk=50.0):
        self.M = M
        self.eps = eps
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.trans = trans
        self.sigma_walk = sigma_walk

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _PARAMS}

    def set_params(self, **params):
        unknown = set(params) - set(_PARAMS)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, ctx, y=None):
        self._bind(ctx)
        self._build_matrices()
        self._input_stats(ctx)
        self._augment()
        self._initial_state(ctx)
        return self

    def predict(self, ctx):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("MixtureKalmanEstimator is not fitted")
        obs = _frame(getattr(ctx, "observations", None))
        ramps = _frame(getattr(ctx, "ramps", None))
        targets = ctx.targets.copy()
        targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True)
        targets["link_id"] = targets["link_id"].astype(str)
        for col in ("speed_kmh", "flow_vph"):
            if col in targets:
                targets = targets.drop(columns=col)
        pieces = []
        if not obs.empty:
            obs = _mainline(obs)
            ramps = _ramps(ramps)
            rng = np.random.default_rng(int(getattr(ctx, "seed", 0)))
            days = obs["timestamp"].dt.floor("D")
            schedule = []
            for day in sorted(days.unique()):
                day_obs = obs.loc[days.eq(day)]
                times = pd.DatetimeIndex(np.sort(day_obs["timestamp"].unique()))
                day_ramps = ramps.loc[ramps["timestamp"].dt.floor("D").eq(day)] if not ramps.empty else ramps
                schedule.append((day_obs, day_ramps, times))
            label = f"{getattr(ctx, 'panel', '')} mixture KF".strip()
            bar = tqdm(total=sum(len(times) for _, _, times in schedule), desc=label, unit="step", mininterval=0.25)
            try:
                for day_obs, day_ramps, times in schedule:
                    rho = self._filter_day(day_obs, day_ramps, times, rng, bar)
                    speed, flow = self._to_speed_flow(rho)
                    n = self.n_cells_
                    pieces.append(pd.DataFrame({
                        "timestamp": np.repeat(times.to_numpy(), n),
                        "link_id": np.tile(self.link_ids_, len(times)),
                        "speed_kmh": speed.reshape(-1),
                        "flow_vph": flow.reshape(-1),
                    }))
            finally:
                bar.close()
        pred = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
        if not pred.empty:
            pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
            pred["link_id"] = pred["link_id"].astype(str)
            pred = pred.drop_duplicates(["timestamp", "link_id"], keep="last")
        out = targets.merge(pred, on=["timestamp", "link_id"], how="left", validate="many_to_one")
        speed_fb, flow_fb = self._fallbacks
        out["speed_kmh"] = out["speed_kmh"].fillna(out["link_id"].map(speed_fb)).fillna(float(np.mean(self.v_f_)))
        out["flow_vph"] = out["flow_vph"].fillna(out["link_id"].map(flow_fb)).fillna(0.0)
        out["speed_kmh"] = np.clip(pd.to_numeric(out["speed_kmh"], errors="coerce"), 0, None)
        out["flow_vph"] = np.clip(pd.to_numeric(out["flow_vph"], errors="coerce"), 0, None)
        return out

    def _bind(self, ctx):
        topo = ctx.network["lwr_mainline_topology"].copy()
        fd = ctx.network["fd_parameters"].copy()
        link_col = "mainline_link_id" if "mainline_link_id" in topo.columns else "link_id"
        topo[link_col] = topo[link_col].astype(str)
        graph = nx.DiGraph()
        for _, row in topo.iterrows():
            graph.add_edge(row["from_node"], row["to_node"], link_id=str(row[link_col]))
        path = nx.dag_longest_path(graph)
        self.link_ids_ = [str(graph[path[i]][path[i + 1]]["link_id"]) for i in range(len(path) - 1)]
        if not self.link_ids_:
            raise ValueError("Mainline path has no cells")
        fd["link_id"] = fd["link_id"].astype(str)
        fd = fd.drop_duplicates("link_id").set_index("link_id").loc[self.link_ids_]
        self.n_cells_ = len(self.link_ids_)
        self.lanes_ = np.maximum(fd["lanes"].to_numpy(dtype=float), 1.0)
        self.l_ = np.maximum(fd["length_km"].to_numpy(dtype=float), 1e-3)
        self.v_f_ = np.maximum(fd["free_speed_kmh"].to_numpy(dtype=float), 1e-3)
        self.Q_m_ = np.maximum(fd["capacity_vph"].to_numpy(dtype=float) / self.lanes_, 0.0)
        self.k_c_ = np.maximum(fd["critical_density"].to_numpy(dtype=float) / self.lanes_, 0.0)
        self.k_j_ = np.maximum(fd["k_jam"].to_numpy(dtype=float) / self.lanes_, self.k_c_ + 1e-3)
        self.w_c_ = self.Q_m_ / np.maximum(self.k_j_ - self.k_c_, 1e-3)
        topo = topo.drop_duplicates(link_col).set_index(link_col)
        self.specs_ = [{"name": "q_in", "kind": "boundary", "cell": 0}, {"name": "q_out", "kind": "boundary", "cell": self.n_cells_ - 1}]
        seen = set()
        for kind, column in (("on", "on_ramp_link_ids"), ("off", "off_ramp_link_ids")):
            if column not in topo.columns:
                continue
            for cell, link in enumerate(self.link_ids_):
                for ramp_id in _split_ids(topo.at[link, column] if link in topo.index else None):
                    if ramp_id in seen:
                        continue
                    seen.add(ramp_id)
                    self.specs_.append({"name": ramp_id, "kind": kind, "cell": cell})
        self.sigma_v_ = _pair(self.sigma_v, "sigma_v")
        self.sigma_w_ = _pair(self.sigma_w, "sigma_w")
        stay = float(np.clip(self.trans, 1e-6, 1.0 - 1e-6))
        self.Trans_ = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]])
        self.sigma_walk_ = float(self.sigma_walk)
        if self.sigma_walk_ < 0 or not np.isfinite(self.sigma_walk_):
            raise ValueError("sigma_walk must be a nonnegative finite number")
        if int(self.M) < 1:
            raise ValueError("M must be positive")
        if not 0 < float(self.eps) < 1:
            raise ValueError("eps must satisfy 0 < eps < 1")

    def _build_matrices(self):
        ratio = np.maximum(self.v_f_, self.w_c_) * _TS / self.l_
        self.n_sub_ = max(1, int(np.ceil(ratio.max())))
        dt = _TS / self.n_sub_
        n = self.n_cells_
        n_in = len(self.specs_)
        A = np.zeros((2, n, n))
        B = np.zeros((2, n, n_in))
        gain = dt / self.l_
        for i in range(n):
            A[0, i, i] = 1.0 - self.v_f_[i] * gain[i]
            A[1, i, i] = 1.0 - self.w_c_[i] * gain[i]
            if i > 0:
                A[0, i, i - 1] = self.v_f_[i - 1] * gain[i]
            if i + 1 < n:
                A[1, i, i + 1] = self.w_c_[i + 1] * gain[i]
        BJ = np.zeros((n, n))
        for i in range(n):
            BJ[i, i] = self.w_c_[i] * gain[i]
            if i + 1 < n:
                BJ[i, i + 1] = -self.w_c_[i + 1] * gain[i]
        for j, spec in enumerate(self.specs_):
            coef = gain[spec["cell"]]
            if spec["name"] == "q_in":
                B[0, spec["cell"], j] = coef
            elif spec["name"] == "q_out":
                B[1, spec["cell"], j] = -coef
            elif spec["kind"] == "on":
                B[:, spec["cell"], j] = coef
            else:
                B[:, spec["cell"], j] = -coef
        b = np.zeros((2, n))
        b[1] = BJ @ self.k_j_
        self.A_step_ = np.zeros_like(A)
        self.B_step_ = np.zeros_like(B)
        self.b_step_ = np.zeros_like(b)
        for mode in (0, 1):
            self.A_step_[mode], self.B_step_[mode], self.b_step_[mode] = _integrate(A[mode], B[mode], b[mode], self.n_sub_)

    def _input_stats(self, ctx):
        train = _mainline(getattr(ctx, "train", None))
        ramps = _ramps(getattr(ctx, "ramps", None))
        self.stats_ = []
        for spec in self.specs_:
            if spec["kind"] == "boundary":
                link = self.link_ids_[spec["cell"]]
                part = train.loc[train["link_id"].eq(link)] if not train.empty else train
                flow = part["flow_vph"].to_numpy(dtype=float) / self.lanes_[spec["cell"]] if not part.empty else np.array([])
                pct = part["pct_observed"].to_numpy(dtype=float) if not part.empty else np.array([])
                times = part["timestamp"] if not part.empty else []
            else:
                part = ramps.loc[ramps["ramp_link_id"].eq(spec["name"])] if not ramps.empty else ramps
                flow = part["flow_vph"].to_numpy(dtype=float) / self.lanes_[spec["cell"]] if not part.empty else np.array([])
                pct = part["pct_observed"].to_numpy(dtype=float) if not part.empty else np.array([])
                times = part["timestamp"] if not part.empty else []
            self.stats_.append(_flow_stat(times, flow, pct))
        self.walk_ = np.array([stat is None for stat in self.stats_])
        self.kept_ = np.flatnonzero(~self.walk_)
        self.walk_idx_ = np.flatnonzero(self.walk_)
        self.n_walk_ = int(self.walk_.sum())
        self.n_x_ = self.n_cells_ + self.n_walk_

    def _augment(self):
        n = self.n_cells_
        nx = self.n_x_
        n_kept = int(self.kept_.size)
        self.A_ = np.zeros((2, nx, nx))
        self.B_ = np.zeros((2, n, n_kept))
        self.b_ = self.b_step_.copy()
        for mode in (0, 1):
            self.A_[mode, :n, :n] = self.A_step_[mode]
            if self.n_walk_:
                self.A_[mode, n:, n:] = np.eye(self.n_walk_)
                for local, src in enumerate(self.walk_idx_):
                    self.A_[mode, :n, n + local] = self.B_step_[mode, :, src]
            if n_kept:
                self.B_[mode] = self.B_step_[mode][:, self.kept_]

    def _initial_state(self, ctx):
        train = _mainline(getattr(ctx, "train", None))
        density = np.full(self.n_cells_, np.nan)
        if not train.empty:
            ok = train["pct_observed"].ge(75) & train["speed_kmh"].gt(1e-3) & train["flow_vph"].notna()
            part = train.loc[ok, ["link_id", "flow_vph", "speed_kmh"]].copy()
            if not part.empty:
                part["lane"] = part["link_id"].map(dict(zip(self.link_ids_, self.lanes_)))
                part = part.dropna(subset=["lane"])
                part["k"] = part["flow_vph"] / (part["lane"] * part["speed_kmh"])
                density = part.groupby("link_id")["k"].mean().reindex(self.link_ids_).to_numpy(dtype=float)
        self.x0_ = np.where(np.isfinite(density), density, 0.5 * self.k_c_)
        self.x0_ = np.clip(self.x0_, 0.0, self.k_j_)
        scale = np.maximum(self.k_j_, 1.0)
        p0 = np.concatenate([scale ** 2, np.full(self.n_walk_, (self.sigma_walk_ * 10.0) ** 2)])
        self.P0_ = np.diag(p0)
        train_ok = train.loc[train["pct_observed"].ge(75)] if not train.empty else train
        if train_ok.empty:
            speed_fb, flow_fb = {}, {}
        else:
            means = train_ok.groupby("link_id")[["speed_kmh", "flow_vph"]].mean()
            speed_fb = means["speed_kmh"].to_dict()
            flow_fb = means["flow_vph"].to_dict()
        self._fallbacks = (speed_fb, flow_fb)
        self.fitted_ = True

    def _filter_day(self, obs, ramps, times, rng, bar=None):
        rho_obs, mask, values, pct = self._day_inputs(obs, ramps, times)
        u_hat, sigma_u, z_meas, z_var = self._impute(times, values, pct)
        M = int(self.M)
        n = self.n_cells_
        prior = self.Trans_[0]
        modes = rng.choice(2, size=M, p=prior)
        xs = []
        Ps = []
        log_w = np.zeros(M)
        x0 = np.zeros(self.n_x_)
        x0[:n] = self.x0_
        for m in range(M):
            y, C, R = self._observe(0, modes[m], rho_obs, mask, z_meas, z_var)
            x, P, loglik = _update(x0.copy(), self.P0_.copy(), y, C, R)
            xs.append(x)
            Ps.append(P)
            log_w[m] = loglik
        weights = _normalize(log_w, float(self.eps), M)
        log_w = np.log(weights)
        rho = np.zeros((len(times), n))
        rho[0] = sum(weights[m] * xs[m][:n] for m in range(M))
        if bar is not None:
            bar.set_postfix_str(pd.Timestamp(times[0]).strftime("%Y-%m-%d %H:%M"), refresh=False)
            bar.update()
        for t in range(1, len(times)):
            log_zeta = np.zeros(M)
            for m in range(M):
                cands = []
                log_mu = np.zeros(2)
                for mode in (0, 1):
                    xp, Pp = self._predict(xs[m], Ps[m], mode, u_hat[t], sigma_u[t])
                    y, C, R = self._observe(t, mode, rho_obs, mask, z_meas, z_var)
                    xn, Pn, loglik = _update(xp, Pp, y, C, R)
                    cands.append((xn, Pn))
                    log_mu[mode] = np.log(self.Trans_[modes[m], mode]) + loglik
                log_zeta[m] = logsumexp(log_mu)
                pick = int(rng.choice(2, p=_normalize(log_mu, 0.0, 1)))
                modes[m] = pick
                xs[m], Ps[m] = cands[pick]
            log_w = log_w + log_zeta
            weights = _normalize(log_w, float(self.eps), M)
            log_w = np.log(weights)
            rho[t] = sum(weights[m] * xs[m][:n] for m in range(M))
            if bar is not None:
                bar.set_postfix_str(pd.Timestamp(times[t]).strftime("%Y-%m-%d %H:%M"), refresh=False)
                bar.update()
        return rho

    def _day_inputs(self, obs, ramps, times):
        n = self.n_cells_
        lane = dict(zip(self.link_ids_, self.lanes_))
        part = obs.loc[obs["link_id"].isin(self.link_ids_)].copy()
        part["lane"] = part["link_id"].map(lane)
        finite = part["flow_vph"].notna() & part["lane"].notna()
        flow_ok = finite & part["pct_observed"].ge(75)
        density_ok = flow_ok & part["speed_kmh"].gt(1e-3)
        inaccurate = finite & ~flow_ok
        part["k"] = np.where(density_ok, part["flow_vph"] / (part["lane"] * part["speed_kmh"]), np.nan)
        part["q_trusted"] = np.where(flow_ok, part["flow_vph"] / part["lane"], np.nan)
        part["q_inacc"] = np.where(inaccurate, part["flow_vph"] / part["lane"], np.nan)
        part["pct_inacc"] = np.where(inaccurate, part["pct_observed"], np.nan)
        rho = _pivot(part, times, "link_id", self.link_ids_, "k")
        q_trusted = _pivot(part, times, "link_id", self.link_ids_, "q_trusted")
        q_inacc = _pivot(part, times, "link_id", self.link_ids_, "q_inacc")
        pct_inacc = _pivot(part, times, "link_id", self.link_ids_, "pct_inacc")
        rho = np.clip(rho, 0.0, self.k_j_ * 2.0)
        mask = np.isfinite(rho)
        n_in = len(self.specs_)
        values = np.full((len(times), n_in), np.nan)
        pct = np.full((len(times), n_in), np.nan)
        ramp_ids = [spec["name"] for spec in self.specs_ if spec["kind"] != "boundary"]
        ramp_q = None
        ramp_index = {}
        if ramp_ids and not ramps.empty:
            sub = ramps.loc[ramps["ramp_link_id"].isin(ramp_ids)].copy()
            sub["lane"] = sub["ramp_link_id"].map({spec["name"]: self.lanes_[spec["cell"]] for spec in self.specs_ if spec["kind"] != "boundary"})
            sub["q"] = sub["flow_vph"] / sub["lane"]
            trusted_r = sub["q"].notna() & sub["pct_observed"].ge(75)
            sub["q_trusted"] = np.where(trusted_r, sub["q"], np.nan)
            sub["q_inacc"] = np.where(sub["q"].notna() & ~trusted_r, sub["q"], np.nan)
            sub["pct_inacc"] = np.where(sub["q"].notna() & ~trusted_r, sub["pct_observed"], np.nan)
            ramp_q = (
                _pivot(sub, times, "ramp_link_id", ramp_ids, "q_trusted"),
                _pivot(sub, times, "ramp_link_id", ramp_ids, "q_inacc"),
                _pivot(sub, times, "ramp_link_id", ramp_ids, "pct_inacc"),
            )
            ramp_index = {ramp_id: i for i, ramp_id in enumerate(ramp_ids)}
        q_any = np.where(np.isfinite(q_trusted), q_trusted, q_inacc)
        self._q_any = q_any
        for j, spec in enumerate(self.specs_):
            if spec["kind"] == "boundary":
                trusted_col, inacc_col, inacc_pct = q_trusted[:, spec["cell"]], q_inacc[:, spec["cell"]], pct_inacc[:, spec["cell"]]
            elif ramp_q is not None and spec["name"] in ramp_index:
                i = ramp_index[spec["name"]]
                trusted_col, inacc_col, inacc_pct = ramp_q[0][:, i], ramp_q[1][:, i], ramp_q[2][:, i]
            else:
                continue
            use_t = np.isfinite(trusted_col)
            use_i = ~use_t & np.isfinite(inacc_col)
            values[use_t, j] = trusted_col[use_t]
            pct[use_t, j] = 100.0
            values[use_i, j] = inacc_col[use_i]
            pct[use_i, j] = inacc_pct[use_i]
        return rho, mask, values, pct

    def _impute(self, times, values, pct):
        T = len(times)
        n_kept = int(self.kept_.size)
        u_hat = np.zeros((T, n_kept))
        sigma_u = np.zeros((T, n_kept))
        z_meas = np.full((T, self.n_walk_), np.nan)
        z_var = np.full((T, self.n_walk_), np.nan)
        slots = _slots(times)
        last = np.full(n_kept, np.nan)
        neighbors = []
        for spec in self.specs_:
            if spec["kind"] == "boundary":
                neighbors.append([])
            else:
                neighbors.append([k for k, other in enumerate(self.specs_) if other["kind"] == spec["kind"] and other["name"] != spec["name"]])
        for t in range(T):
            for local, j in enumerate(self.kept_):
                stat = self.stats_[int(j)]
                if stat is None:
                    raise RuntimeError("kept input has no imputation mean")
                std = stat["std"]
                raw, raw_pct = values[t, j], pct[t, j]
                if not np.isfinite(raw_pct):
                    raw_pct = 0.0
                if np.isfinite(raw) and raw_pct >= 75:
                    guess, sig = raw, 0.0
                elif np.isfinite(raw):
                    frac = 1.0 - np.clip(raw_pct, 0.0, 100.0) / 100.0
                    guess, sig = raw, max(std * frac, _SIGMA_FLOOR)
                else:
                    neigh = self._neighbor(t, j, neighbors[j], values)
                    if np.isfinite(last[local]):
                        guess = last[local]
                    elif np.isfinite(neigh):
                        guess = neigh
                    else:
                        guess = stat["slot"][slots[t]]
                        if not np.isfinite(guess):
                            guess = stat["mean"]
                    if not np.isfinite(guess):
                        guess = 0.0
                    sig = max(std, _SIGMA_FLOOR)
                u_hat[t, local] = max(guess, 0.0)
                sigma_u[t, local] = sig
                last[local] = u_hat[t, local]
        for local, j in enumerate(self.walk_idx_):
            finite = np.isfinite(values[:, j])
            z_meas[finite, local] = np.maximum(values[finite, j], 0.0)
            for t in np.flatnonzero(finite):
                raw_pct = pct[t, j] if np.isfinite(pct[t, j]) else 0.0
                if raw_pct >= 75:
                    z_var[t, local] = _SIGMA_FLOOR ** 2
                else:
                    frac = 1.0 - np.clip(raw_pct, 0.0, 100.0) / 100.0
                    z_var[t, local] = max(_SIGMA_FLOOR, abs(values[t, j]) * frac) ** 2
        return u_hat, sigma_u, z_meas, z_var

    def _neighbor(self, t, j, others, values):
        spec = self.specs_[j]
        if spec["name"] == "q_in" and self.n_cells_ > 1:
            return self._q_any[t, 1]
        if spec["name"] == "q_out" and self.n_cells_ > 1:
            return self._q_any[t, self.n_cells_ - 2]
        if not others:
            return np.nan
        sample = values[t, others]
        sample = sample[np.isfinite(sample)]
        return float(np.mean(sample)) if sample.size else np.nan

    def _predict(self, x, P, mode, u_hat, sigma_u):
        n = self.n_cells_
        x_pred = self.A_[mode] @ x
        B = self.B_[mode]
        if u_hat.size:
            x_pred[:n] += B @ u_hat
        if mode == 1:
            x_pred[:n] += self.b_[mode]
        Q = np.zeros((self.n_x_, self.n_x_))
        Q[:n, :n] = self.sigma_v_[mode] ** 2 * np.eye(n)
        if u_hat.size:
            variance = np.asarray(sigma_u, dtype=float) ** 2
            Q[:n, :n] += (B * variance) @ B.T
        if self.n_walk_:
            Q[n:, n:] = self.sigma_walk_ ** 2 * np.eye(self.n_walk_)
        P_pred = self.A_[mode] @ P @ self.A_[mode].T + Q
        return x_pred, 0.5 * (P_pred + P_pred.T)

    def _observe(self, t, mode, rho, mask, z_meas, z_var):
        rows, y, var = [], [], []
        for i in np.flatnonzero(mask[t]):
            rows.append(i)
            y.append(rho[t, i])
            var.append(self.sigma_w_[mode] ** 2)
        for k in range(self.n_walk_):
            if np.isfinite(z_meas[t, k]):
                rows.append(self.n_cells_ + k)
                y.append(z_meas[t, k])
                var.append(max(z_var[t, k], 1e-8))
        if not rows:
            return np.zeros(0), np.zeros((0, self.n_x_)), np.zeros((0, 0))
        C = np.zeros((len(rows), self.n_x_))
        C[np.arange(len(rows)), rows] = 1.0
        return np.asarray(y, dtype=float), C, np.diag(var)

    def _to_speed_flow(self, rho):
        rho = np.clip(rho, 0.0, self.k_j_)
        q = np.minimum(self.v_f_ * rho, self.w_c_ * (self.k_j_ - rho))
        q = np.clip(np.minimum(q, self.Q_m_), 0.0, None)
        speed = np.where(rho > 1e-6, q / np.maximum(rho, 1e-6), self.v_f_)
        return np.clip(speed, 0.0, None), q * self.lanes_


def _update(x, P, y, C, R):
    if y.size == 0:
        return x, P, 0.0
    innov = y - C @ x
    S = C @ P @ C.T + R
    S = 0.5 * (S + S.T) + 1e-8 * np.eye(y.size)
    loglik = _log_gauss(innov, S)
    try:
        K = np.linalg.solve(S, C @ P).T
    except np.linalg.LinAlgError:
        return x, P, -1e12
    x_new = x + K @ innov
    eye = np.eye(x.size)
    gain = eye - K @ C
    P_new = gain @ P @ gain.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    if not np.isfinite(x_new).all():
        x_new = np.nan_to_num(x, nan=0.0)
    if not np.isfinite(P_new).all():
        P_new = np.nan_to_num(P, nan=0.0)
        P_new = 0.5 * (P_new + P_new.T) + np.eye(x.size)
    return x_new, P_new, loglik


def _normalize(log_w, eps, M):
    log_w = np.asarray(log_w, dtype=float)
    log_w = np.where(np.isfinite(log_w), log_w, -1e12)
    weights = np.exp(log_w - np.max(log_w))
    total = weights.sum()
    weights = np.full_like(weights, 1.0 / weights.size) if total <= 0 else weights / total
    if eps > 0 and M > 1:
        weights = np.maximum(weights, eps / M)
        weights = weights / weights.sum()
    return weights


def _flow_stat(timestamps, flow, pct):
    flow = np.asarray(flow, dtype=float)
    pct = np.asarray(pct, dtype=float)
    if flow.size == 0:
        return None
    trusted = np.isfinite(flow) & np.isfinite(pct) & (pct >= 75)
    if not np.any(trusted):
        return None
    vals = np.maximum(flow[trusted], 0.0)
    slot_mean = np.full(288, np.nan)
    slots = _slots(timestamps)
    for slot in range(288):
        chosen = trusted & (slots == slot)
        if np.any(chosen):
            slot_mean[slot] = float(np.mean(np.maximum(flow[chosen], 0.0)))
    return {"std": float(np.std(vals)), "mean": float(np.mean(vals)), "slot": slot_mean}


def _frame(frame):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    return frame.copy()


def _mainline(frame):
    frame = _frame(frame)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["link_id"] = frame["link_id"].astype(str)
    for col in ("speed_kmh", "flow_vph", "pct_observed"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame else np.nan
    frame["pct_observed"] = frame["pct_observed"].fillna(0.0)
    return frame


def _ramps(frame):
    frame = _frame(frame)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ramp_link_id"] = frame["ramp_link_id"].astype(str)
    frame["flow_vph"] = pd.to_numeric(frame["flow_vph"], errors="coerce") if "flow_vph" in frame else np.nan
    frame["pct_observed"] = pd.to_numeric(frame["pct_observed"], errors="coerce").fillna(0.0) if "pct_observed" in frame else 0.0
    if "is_missing" in frame:
        missing = pd.to_numeric(frame["is_missing"], errors="coerce").fillna(0).astype(bool)
        frame.loc[missing, "flow_vph"] = np.nan
    return frame


def _estimator(ctx):
    model = ctx.cache.get("mkf") if isinstance(getattr(ctx, "cache", None), dict) else None
    if model is None:
        model = MixtureKalmanEstimator()
        model.set_params(**{name: ctx.params[name] for name in _PARAMS if name in ctx.params})
        model.fit(ctx)
        if isinstance(getattr(ctx, "cache", None), dict):
            ctx.cache["mkf"] = model
    return model


def state(ctx):
    return _estimator(ctx).predict(ctx)


def queue(ctx):
    """Filter the 60-minute history, then roll the mixture through the unobserved horizon."""
    from dataclasses import replace

    model = _estimator(ctx)
    hist = _mainline(ctx.observations)
    start = hist["timestamp"].min()
    end = pd.to_datetime(ctx.targets["timestamp"], utc=True).max()
    missing = pd.date_range(start, end, freq="5min", tz="UTC").difference(pd.DatetimeIndex(hist["timestamp"]))
    extra = pd.DataFrame({"timestamp": missing, "link_id": model.link_ids_[0], "speed_kmh": np.nan, "flow_vph": np.nan, "pct_observed": 0})
    pred = model.predict(replace(ctx, observations=pd.concat([ctx.observations, extra], ignore_index=True)))
    free = ctx.network["links"].set_index("link_id")["free_speed_kmh"]
    ratio = float(ctx.params.get("queue_ratio", 0.6))
    pred["queue_pred"] = pred["speed_kmh"].le(ratio * pred["link_id"].map(free)).fillna(False).astype(int)
    return pred


def _mode_flow(incidence, prior, counts, regularization, start):
    """Nonnegative ridge solve. An iteration-limit point is kept when it is finite."""
    from scipy.optimize import minimize

    def objective(flow):
        residual, movement = incidence @ flow - counts, flow - prior
        loss = 0.5 * (residual @ residual + regularization * (movement @ movement))
        gradient = np.asarray(incidence.T @ residual + regularization * movement).ravel()
        return loss, gradient

    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(start),
                      options={"maxiter": 20000, "ftol": 1e-8, "gtol": 1e-4})
    flow = np.clip(np.asarray(result.x, dtype=float), 0, None)
    if not np.isfinite(flow).all():
        raise RuntimeError(f"ODME solve failed: {result.message}")
    return flow


def odme(ctx):
    """Two mode-conditioned path-flow posteriors, mixed by the count marginal likelihood."""
    from scipy import sparse
    from trafficbench.metrics import operator

    incidence = operator(ctx.network, ctx.targets, ctx.counts)
    counts = ctx.counts["count"].to_numpy(float)
    prior = ctx.prior.set_index("path_id").reindex(ctx.targets.path_id).path_flow.to_numpy(float)
    innov = counts - np.asarray(incidence @ prior).ravel()
    gram = (incidence @ incidence.T).toarray() if sparse.issparse(incidence) else np.asarray(incidence @ incidence.T)
    regs = np.sort(np.asarray(ctx.params.get("odme_regs", (0.2, 0.02)), dtype=float).reshape(-1))[::-1]
    flows, scores = [], []
    eye = np.eye(len(counts))
    start = prior
    for reg in regs:
        if reg <= 0:
            raise ValueError("odme_regs must be positive")
        start = _mode_flow(incidence, prior, counts, float(reg), start)
        flows.append(start)
        cov = gram / float(reg) + eye
        scores.append(_log_gauss(innov, 0.5 * (cov + cov.T) + 1e-8 * eye))
    weights = _normalize(np.asarray(scores, dtype=float), float(ctx.params.get("eps", 1e-3)), len(regs))
    out = ctx.targets.copy()
    out["path_flow"] = sum(weight * flow for weight, flow in zip(weights, flows))
    return out
