"""IMM on an edge-by-edge density matrix.

Each mainline link, ramp, and connector is one edge. The state is the E by E
density matrix, with a full covariance on its vectorization. Free-flow and
congested modes are linear and have no external input.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from tqdm import tqdm

from trafficbench.metrics import operator

from .imm import IMMEstimator

_PARAMS = ("sigma_v", "sigma_w", "trans")
_TS = 1.0 / 12.0
_KINDS = ("mainline", "on", "off", "conn-in", "conn-out", "on-conn", "off-conn")


def _pair(value, name):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if arr.size != 2 or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError(f"{name} must be two nonnegative standard deviations")
    return arr


def _split_ids(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


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
    _numeric(frame, "speed_kmh")
    _numeric(frame, "flow_vph")
    _numeric(frame, "pct_observed")
    frame["pct_observed"] = frame["pct_observed"].fillna(0.0)
    return frame


def _ramps(frame):
    frame = _frame(frame)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ramp_link_id"] = frame["ramp_link_id"].astype(str)
    _numeric(frame, "flow_vph")
    _numeric(frame, "pct_observed")
    frame["pct_observed"] = frame["pct_observed"].fillna(0.0)
    if "is_missing" in frame:
        missing = pd.Series(pd.to_numeric(frame["is_missing"], errors="coerce"), index=frame.index).fillna(0).astype(bool)
        frame.loc[missing, "flow_vph"] = np.nan
    return frame


def _field(row, name, default):
    if not isinstance(row, pd.Series) or name not in row.index:
        return float(default)
    try:
        number = float(cast(Any, row[name]))
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _numeric(frame, col):
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    else:
        frame[col] = pd.Series(np.nan, index=frame.index, dtype=float)


def _pivot(frame, times, key_col, keys, value_col):
    out = np.full((len(times), len(keys)), np.nan)
    if frame.empty or value_col not in frame or not keys:
        return out
    grouped = frame.groupby(["timestamp", key_col], sort=False)[value_col].mean()
    index = pd.MultiIndex.from_product([times, keys], names=["timestamp", key_col])
    return grouped.reindex(index).to_numpy(dtype=float).reshape(len(times), len(keys))


def _link_col(topo):
    return "mainline_link_id" if "mainline_link_id" in topo.columns else "link_id"


def build_edges(topo):
    """Unique edges from one mainline topology table."""
    topo = topo.copy()
    link_col = _link_col(topo)
    topo[link_col] = topo[link_col].astype(str)
    if "order_index" in topo.columns:
        topo["_order"] = pd.Series(pd.to_numeric(topo["order_index"], errors="coerce"), index=topo.index).fillna(0)
    else:
        topo["_order"] = np.arange(len(topo))
    topo = topo.sort_values(["_order", link_col])
    mainline = list(dict.fromkeys(topo[link_col].tolist()))
    mainline_set = set(mainline)
    on_ramps, off_ramps, incoming_ids, outgoing_ids = set(), set(), set(), set()
    on_conn, off_conn, attached = set(), set(), {}
    parsed = []
    for _, row in topo.iterrows():
        host = str(row[link_col])
        incoming = _split_ids(row["incoming_link_ids"] if "incoming_link_ids" in topo.columns else None)
        outgoing = _split_ids(row["outgoing_link_ids"] if "outgoing_link_ids" in topo.columns else None)
        ons = _split_ids(row["on_ramp_link_ids"] if "on_ramp_link_ids" in topo.columns else None)
        offs = _split_ids(row["off_ramp_link_ids"] if "off_ramp_link_ids" in topo.columns else None)
        parsed.append((host, incoming, outgoing, ons, offs))
        on_ramps.update(ons)
        off_ramps.update(offs)
        incoming_ids.update(incoming)
        outgoing_ids.update(outgoing)
        on_conn.update(edge for edge in ons if edge in incoming)
        off_conn.update(edge for edge in offs if edge in outgoing)
        for edge_id in list(ons) + list(offs) + incoming + outgoing:
            if edge_id not in mainline_set:
                attached.setdefault(edge_id, host)

    def kind_of(edge_id):
        if edge_id in on_conn:
            return "on-conn"
        if edge_id in off_conn:
            return "off-conn"
        if edge_id in on_ramps:
            return "on"
        if edge_id in off_ramps:
            return "off"
        if edge_id in incoming_ids:
            return "conn-in"
        return "conn-out"

    grouped = {kind: [] for kind in _KINDS if kind != "mainline"}
    for edge_id in list(on_ramps | off_ramps | incoming_ids | outgoing_ids):
        if edge_id in mainline_set:
            continue
        grouped[kind_of(edge_id)].append(edge_id)
    edges, kinds = [], []
    for edge_id in mainline:
        edges.append(edge_id)
        kinds.append("mainline")
    for kind in ("on", "off", "conn-in", "conn-out", "on-conn", "off-conn"):
        for edge_id in sorted(set(grouped[kind])):
            edges.append(edge_id)
            kinds.append(kind)
    index = {edge_id: i for i, edge_id in enumerate(edges)}
    succ = [[] for _ in edges]
    pairs = set()

    def link(upstream, downstream):
        if upstream not in index or downstream not in index or upstream == downstream:
            return
        pair = (index[upstream], index[downstream])
        if pair in pairs:
            return
        pairs.add(pair)
        succ[pair[0]].append(pair[1])

    for host, incoming, outgoing, ons, offs in parsed:
        for edge_id in incoming:
            if edge_id in mainline_set:
                link(edge_id, host)
        for edge_id in outgoing:
            if edge_id in mainline_set:
                link(host, edge_id)
        for edge_id in list(ons) + [edge for edge in incoming if edge not in mainline_set]:
            link(edge_id, host)
        for edge_id in list(offs) + [edge for edge in outgoing if edge not in mainline_set]:
            link(host, edge_id)
    return edges, kinds, attached, succ


def _lane_shares(indices, lanes):
    if not indices:
        return []
    weight = np.maximum(lanes[list(indices)], 0.0)
    total = float(weight.sum())
    if total <= 0:
        return [1.0 / len(indices)] * len(indices)
    return (weight / total).tolist()


def _layout(succ):
    edges, cols, starts = [], [], []
    for edge, downstream in enumerate(succ):
        starts.append(len(edges))
        seen = set()
        for col in (edge, *downstream):
            if col in seen:
                continue
            seen.add(col)
            edges.append(edge)
            cols.append(col)
    edges = np.asarray(edges, dtype=int)
    cols = np.asarray(cols, dtype=int)
    index = {(int(edge), int(col)): i for i, (edge, col) in enumerate(zip(edges, cols))}
    return edges, cols, np.asarray(starts, dtype=int), index


def _compose_dense(step, bias, n_sub):
    """A^n and (I + A + ... + A^{n-1}) b by squaring. BLAS uses the CPU cores."""
    matrix = np.asarray(step.toarray() if sparse.issparse(step) else step, dtype=float)
    carry = np.asarray(bias, dtype=float).reshape(-1)
    powers, partials = [], []
    span = 1
    while span <= n_sub:
        powers.append(matrix)
        partials.append(carry)
        if span > n_sub // 2:
            break
        carry = carry + matrix @ carry
        matrix = matrix @ matrix
        span *= 2
    total = np.eye(powers[0].shape[0])
    shift = np.zeros(powers[0].shape[0])
    for bit, (block, piece) in enumerate(zip(powers, partials)):
        if n_sub & (1 << bit):
            shift = total @ piece + shift
            total = total @ block
    if not np.isfinite(total).all() or not np.isfinite(shift).all():
        raise FloatingPointError("composed transition is not finite")
    return total, shift


def _mode_step(mode, succ, pred, v, w, length, lanes, k_jam, dt, index):
    count = len(v)
    rows, cols, data = [], [], []

    def add(row, col, value):
        rows.append(row)
        cols.append(col)
        data.append(value)

    bias = np.zeros(len(index))
    real = []
    for edge, downstream in enumerate(succ):
        seen = []
        for col in (edge, *downstream):
            if col not in seen:
                seen.append(col)
        real.append(seen)
    if mode == 0:
        alpha = v * dt / length
        for edge in range(count):
            for col in real[edge]:
                slot = index[(edge, col)]
                add(slot, slot, 1.0 - alpha[edge])
            for downstream, share in zip(succ[edge], _lane_shares(succ[edge], lanes)):
                gain = share * (lanes[edge] / lanes[downstream]) * (length[edge] / length[downstream]) * alpha[edge]
                for col in real[edge]:
                    add(index[(downstream, downstream)], index[(edge, col)], gain)
    else:
        for edge in range(count):
            for col in real[edge]:
                slot = index[(edge, col)]
                add(slot, slot, 1.0)
        beta = w * dt / length
        for edge in range(count):
            upstream = pred[edge]
            if not upstream:
                continue
            for col in real[edge]:
                add(index[(edge, edge)], index[(edge, col)], -beta[edge])
            bias[index[(edge, edge)]] += beta[edge] * k_jam[edge]
            for source, share in zip(upstream, _lane_shares(upstream, lanes)):
                gamma = share * beta[edge] * (lanes[edge] / lanes[source]) * (length[edge] / length[source])
                bias[index[(source, source)]] -= gamma * k_jam[edge]
                for col in real[edge]:
                    add(index[(source, source)], index[(edge, col)], gamma)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(index), len(index)))
    matrix.sum_duplicates()
    return matrix, bias


def _clip_rows(state, k_jam, starts):
    values = np.asarray(state, dtype=float).reshape(-1).copy()
    sums = np.add.reduceat(values, starts)
    scale = np.ones(sums.size)
    scale[sums < 0] = 0.0
    over = sums > k_jam
    scale[over] = k_jam[over] / sums[over]
    counts = np.diff(np.append(starts, values.size))
    values *= np.repeat(scale, counts)
    return values.reshape(-1, 1)


class InteractingMultipleModelEstimator:
    """Density IMM. propagate() does not take a flow or control vector."""

    def __init__(self, sigma_v=(2.0, 5.0), sigma_w=(5.0, 8.0), trans=0.95):
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.trans = trans

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _PARAMS}

    def set_params(self, **params):
        unknown = set(params) - set(_PARAMS)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, ctx):
        self._bind(ctx)
        self._dynamics()
        self._initial_state(ctx)
        self.imm.reset(self.x0_, self.P0_)
        self.fitted_ = True
        return self

    def propagate(self, *, imm=None):
        """One IMM prediction. This step takes no flow and no control input."""
        imm = self.imm if imm is None else imm
        imm.predict()
        self._clip_modes(imm)

    def assimilate(self, density, flow, *, imm=None):
        """Mode-matched update. An empty measurement leaves the state unchanged."""
        imm = self.imm if imm is None else imm
        if not density and not flow:
            imm.update(None)
            return
        packs = [self._pack(mode, density, flow) for mode in (0, 1)]
        imm.update(packs)
        self._clip_modes(imm)

    @property
    def X(self):
        matrix = np.zeros((self.n_edges_, self.n_edges_))
        matrix[self.active_edge_, self.active_col_] = self.imm.x.reshape(-1)
        return matrix

    @property
    def P(self):
        dim = self.n_edges_ ** 2
        position = self.active_edge_ * self.n_edges_ + self.active_col_
        return sparse.csr_matrix((np.asarray(self.imm.p).ravel(), (position, position)), shape=(dim, dim))

    @property
    def mu(self):
        return self.imm.mu

    def predict(self, ctx):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("InteractingMultipleModelEstimator is not fitted")
        obs = _mainline(getattr(ctx, "observations", None))
        ramps = _ramps(getattr(ctx, "ramps", None))
        targets = ctx.targets.copy()
        targets["timestamp"] = pd.to_datetime(targets["timestamp"], utc=True)
        targets["link_id"] = targets["link_id"].astype(str)
        for col in ("speed_kmh", "flow_vph"):
            if col in targets:
                targets = targets.drop(columns=col)
        pieces = []
        if not obs.empty:
            days = obs["timestamp"].dt.floor("D")
            schedule = []
            for day in sorted(days.unique()):
                day_obs = obs.loc[days.eq(day)]
                times = pd.DatetimeIndex(np.sort(day_obs["timestamp"].unique()))
                day_ramps = ramps.loc[ramps["timestamp"].dt.floor("D").eq(day)] if not ramps.empty else ramps
                schedule.append((day_obs, day_ramps, times))
            label = f"{getattr(ctx, 'panel', '')} IMM".strip()
            bar = tqdm(total=sum(len(times) for _, _, times in schedule), desc=label, unit="step", mininterval=0.25)
            try:
                pieces.extend(self._filter_schedule(schedule, bar))
            finally:
                bar.close()
        pred = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
        if not pred.empty:
            pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
            pred["link_id"] = pred["link_id"].astype(str)
            pred = pred.drop_duplicates(["timestamp", "link_id"], keep="last")
        out = targets.merge(pred, on=["timestamp", "link_id"], how="left", validate="many_to_one")
        speed_fb, flow_fb = self._fallbacks
        speed = pd.Series(pd.to_numeric(out["speed_kmh"], errors="coerce"), index=out.index, dtype=float)
        flow = pd.Series(pd.to_numeric(out["flow_vph"], errors="coerce"), index=out.index, dtype=float)
        speed = speed.fillna(out["link_id"].map(speed_fb)).fillna(float(np.mean(self.v_f_))).clip(lower=0)
        flow = flow.fillna(out["link_id"].map(flow_fb)).fillna(0.0).clip(lower=0)
        out["speed_kmh"] = speed
        out["flow_vph"] = flow
        return out

    def filter_horizon(self, ctx):
        """Filter the queue history, then propagate across the unobserved horizon."""
        from dataclasses import replace

        hist = _mainline(ctx.observations)
        start = hist["timestamp"].min()
        end = pd.to_datetime(ctx.targets["timestamp"], utc=True).max()
        missing = pd.date_range(start, end, freq="5min", tz="UTC").difference(pd.DatetimeIndex(hist["timestamp"]))
        extra = pd.DataFrame({
            "timestamp": missing,
            "link_id": self.edge_ids_[0],
            "speed_kmh": np.nan,
            "flow_vph": np.nan,
            "pct_observed": 0,
        })
        observations = pd.concat([ctx.observations, extra], ignore_index=True)
        return self.predict(replace(ctx, observations=observations))

    def estimate_path_flow(self, ctx):
        """One IMM update of the density matrix, then a mode-mixed path-flow solve."""
        incidence = operator(ctx.network, ctx.targets, ctx.counts)
        prior = ctx.prior.set_index("path_id").reindex(ctx.targets.path_id).path_flow.to_numpy(float)
        counts = ctx.counts.copy()
        counts["link_id"] = counts["link_id"].astype(str)
        measured = counts["count"].to_numpy(float)
        self._plant_prior(np.asarray(incidence @ prior).ravel(), counts["link_id"].tolist())
        density, flow = {}, {}
        for row, link_id in enumerate(counts["link_id"]):
            edge = self._index.get(link_id)
            if edge is None or not np.isfinite(measured[row]):
                continue
            flow[edge] = measured[row] / self.lanes_[edge]
        self.assimilate(density, flow)
        reg = float(ctx.params.get("odme_reg", 0.05))
        if reg <= 0:
            raise ValueError("odme_reg must be positive")
        label = f"{getattr(ctx, 'panel', '')} IMM odme".strip()
        flows = []
        for mode in tqdm(range(self.imm.N), desc=label, unit="mode"):
            target = self._mode_link_flow(mode, counts["link_id"].tolist(), np.asarray(incidence @ prior).ravel())
            flows.append(_path_flow(incidence, prior, target, reg))
        out = ctx.targets.copy()
        out["path_flow"] = sum(weight * flow for weight, flow in zip(self.mu, flows))
        out["path_flow"] = np.clip(out["path_flow"], 0, None)
        return out

    def _bind(self, ctx):
        topo = ctx.network["lwr_mainline_topology"]
        self.edge_ids_, self.kinds_, attached, succ = build_edges(topo)
        if not self.edge_ids_:
            raise ValueError("topology has no edges")
        self.n_edges_ = len(self.edge_ids_)
        self._index = {edge_id: i for i, edge_id in enumerate(self.edge_ids_)}
        self.succ_ = succ
        self.pred_ = [[] for _ in range(self.n_edges_)]
        for upstream, downstreams in enumerate(succ):
            for downstream in downstreams:
                self.pred_[downstream].append(upstream)
        fd = self._fundamental(ctx, attached)
        self.lanes_ = fd["lanes"]
        self.l_ = fd["length"]
        self.v_f_ = fd["v"]
        self.w_c_ = fd["w"]
        self.k_c_ = fd["k_c"]
        self.k_j_ = fd["k_j"]
        self.Q_m_ = fd["q_m"]
        self.sigma_v_ = _pair(self.sigma_v, "sigma_v")
        self.sigma_w_ = _pair(self.sigma_w, "sigma_w")
        stay = float(np.clip(self.trans, 1e-6, 1.0 - 1e-6))
        transition = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]])
        self.imm = IMMEstimator([0.5, 0.5], transition)
        self._mainline_edges = [i for i, kind in enumerate(self.kinds_) if kind == "mainline"]
        self._flow_edges = [i for i, kind in enumerate(self.kinds_) if kind in {"on", "off", "on-conn", "off-conn"}]

    def _fundamental(self, ctx, attached):
        fd = ctx.network["fd_parameters"].copy()
        fd["link_id"] = fd["link_id"].astype(str)
        fd = fd.drop_duplicates("link_id").set_index("link_id")
        count = self.n_edges_
        lanes = np.ones(count)
        length = np.ones(count)
        v = np.full(count, 80.0)
        q_m = np.full(count, 1800.0)
        k_c = np.full(count, 20.0)
        k_j = np.full(count, 100.0)
        mainline_index = {edge_id: i for i, edge_id in enumerate(self.edge_ids_) if self.kinds_[i] == "mainline"}

        def read(edge_id):
            if edge_id not in fd.index:
                return None
            row = fd.loc[edge_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            lane = _field(row, "lanes", 1.0)
            if not np.isfinite(lane) or lane < 1 or lane > 16:
                lane = 1.0
            span = _field(row, "length_km", 1.0)
            free = _field(row, "free_speed_kmh", 80.0)
            capacity = _field(row, "capacity_vph", np.nan)
            critical = _field(row, "critical_density", np.nan)
            jam = _field(row, "k_jam", np.nan)
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            if not np.isfinite(free) or free <= 0:
                free = 80.0
            if not np.isfinite(capacity) or capacity <= 0:
                capacity = free * 20.0 * lane
            if not np.isfinite(critical) or critical <= 0:
                critical = capacity / free
            per_critical = critical / lane
            per_jam = jam / lane if np.isfinite(jam) else np.nan
            if not np.isfinite(per_jam) or per_jam <= per_critical:
                per_jam = per_critical + max(per_critical, 1.0)
            return {
                "lanes": lane,
                "length": span,
                "v": free,
                "q_m": capacity / lane,
                "k_c": per_critical,
                "k_j": per_jam,
            }

        for edge, edge_id in enumerate(self.edge_ids_):
            source = edge_id if self.kinds_[edge] == "mainline" else attached.get(edge_id)
            spec = read(source) if source is not None else None
            if spec is None and source in mainline_index:
                spec = read(self.edge_ids_[mainline_index[source]])
            if spec is None:
                continue
            lanes[edge] = spec["lanes"]
            length[edge] = spec["length"]
            v[edge] = spec["v"]
            q_m[edge] = spec["q_m"]
            k_c[edge] = spec["k_c"]
            k_j[edge] = spec["k_j"]
        wave = q_m / np.maximum(k_j - k_c, 1e-3)
        return {"lanes": lanes, "length": length, "v": v, "w": wave, "k_c": k_c, "k_j": k_j, "q_m": q_m}

    def _dynamics(self):
        ratio = np.maximum(self.v_f_, self.w_c_) * _TS / self.l_
        self.n_sub_ = max(1, int(np.ceil(float(ratio.max()))))
        dt = _TS / self.n_sub_
        self.active_edge_, self.active_col_, self.active_starts_, index = _layout(self.succ_)
        self.n_active_ = int(self.active_edge_.size)
        self._self_index = np.flatnonzero(self.active_edge_ == self.active_col_)
        steps, shifts, operators, biases = [], [], [], []
        for mode in (0, 1):
            step, bias = _mode_step(mode, self.succ_, self.pred_, self.v_f_, self.w_c_, self.l_, self.lanes_, self.k_j_, dt, index)
            steps.append(step)
            shifts.append(bias)
            try:
                composed, carry = _compose_dense(step, bias, self.n_sub_)
            except FloatingPointError:
                operators, biases = [], []
                break
            operators.append(composed)
            biases.append(carry)
        repeats = 1
        if len(operators) != 2:
            operators = [np.asarray(step.toarray(), dtype=float) for step in steps]
            biases = shifts
            repeats = self.n_sub_
        process = [np.full(self.n_active_, self.sigma_v_[mode] ** 2) for mode in (0, 1)]
        self._operators = operators
        self._biases = biases
        self.imm.set_models(operators, biases, process, np.zeros(self.n_active_), np.ones(self.n_active_), repeats)

    def _initial_state(self, ctx):
        train = _mainline(getattr(ctx, "train", None))
        density = np.full(self.n_edges_, np.nan)
        if not train.empty:
            ok = train["pct_observed"].ge(75) & train["speed_kmh"].gt(1e-3) & train["flow_vph"].notna()
            part = train.loc[ok, ["link_id", "flow_vph", "speed_kmh"]].copy()
            if not part.empty:
                lane = dict(zip(self.edge_ids_, self.lanes_))
                part["lane"] = part["link_id"].map(lane)
                part = part.dropna(subset=["lane"])
                part["k"] = part["flow_vph"] / (part["lane"] * part["speed_kmh"])
                density = part.groupby("link_id")["k"].mean().reindex(self.edge_ids_).to_numpy(dtype=float)
        diagonal = np.where(np.isfinite(density), density, 0.5 * self.k_c_)
        diagonal = np.clip(diagonal, 0.0, self.k_j_)
        self.x0_ = self._from_diagonal(diagonal)
        scale = float(np.maximum(self.k_j_, 1.0).max() ** 2)
        self.P0_ = np.full(self.n_active_, scale)
        train_ok = train.loc[train["pct_observed"].ge(75)] if not train.empty else train
        if train_ok.empty:
            speed_fb, flow_fb = {}, {}
        else:
            means = pd.DataFrame(train_ok.groupby("link_id")[["speed_kmh", "flow_vph"]].mean())
            speed_fb = pd.Series(means["speed_kmh"]).to_dict()
            flow_fb = pd.Series(means["flow_vph"]).to_dict()
        self._fallbacks = (speed_fb, flow_fb)

    def _clip_modes(self, imm=None):
        imm = self.imm if imm is None else imm
        for index in range(imm.N):
            imm.xs[index] = _clip_rows(imm.xs[index], self.k_j_, self.active_starts_)
        imm._compute_state_estimate()
        imm.x = _clip_rows(imm.x, self.k_j_, self.active_starts_)

    def _from_diagonal(self, diagonal):
        state = np.zeros((self.n_active_, 1))
        state[self._self_index, 0] = np.asarray(diagonal, dtype=float).reshape(-1)
        return state

    def _row_sums(self, state):
        return np.add.reduceat(np.asarray(state, dtype=float).reshape(-1), self.active_starts_)

    def _spawn_filter(self):
        return self.imm.spawn()

    def _filter_schedule(self, schedule, bar):
        def run(item):
            day_obs, day_ramps, times = item
            rho = self._filter_day(day_obs, day_ramps, times, bar, self._spawn_filter())
            speed, flow = self._to_speed_flow(rho)
            return pd.DataFrame({
                "timestamp": np.repeat(times.to_numpy(), self.n_edges_),
                "link_id": np.tile(self.edge_ids_, len(times)),
                "speed_kmh": speed.reshape(-1),
                "flow_vph": flow.reshape(-1),
            })

        workers = min(len(schedule), os.cpu_count() or 1)
        lock = threading.Lock()
        previous = getattr(self, "_bar_lock", None)
        self._bar_lock = lock
        try:
            if workers <= 1:
                return [run(item) for item in schedule]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(run, schedule))
        finally:
            self._bar_lock = previous

    def _pack(self, mode, density, flow):
        entries, columns, values, variance, coeffs = [], [], [], [], []

        def observe(edge, coefficient, measured, noise):
            span = np.flatnonzero(self.active_edge_ == edge)
            entries.append(np.full(span.size, len(values)))
            columns.append(span)
            coeffs.append(np.full(span.size, coefficient))
            values.append(measured)
            variance.append(noise)

        for edge, measured in density.items():
            if np.isfinite(measured):
                observe(edge, 1.0, measured, self.sigma_w_[mode] ** 2)
        for edge, measured in flow.items():
            if not np.isfinite(measured):
                continue
            if mode == 0:
                observe(edge, self.v_f_[edge], measured, (self.sigma_w_[mode] * max(self.v_f_[edge], 1.0)) ** 2)
            else:
                observe(
                    edge,
                    -self.w_c_[edge],
                    measured - self.w_c_[edge] * self.k_j_[edge],
                    (self.sigma_w_[mode] * max(self.w_c_[edge], 1.0)) ** 2,
                )
        if not values:
            return np.zeros(0), sparse.csr_matrix((0, self.n_active_)), np.zeros(0), np.zeros((0, 0))
        observation = sparse.csr_matrix(
            (np.concatenate(coeffs), (np.concatenate(entries), np.concatenate(columns))),
            shape=(len(values), self.n_active_),
        )
        return np.asarray(values, dtype=float), observation, np.zeros(len(values)), np.diag(variance)

    def _filter_day(self, obs, ramps, times, bar, imm=None):
        density, flow = self._day_inputs(obs, ramps, times)
        imm = self.imm if imm is None else imm
        imm.reset(self.x0_, self.P0_)
        rho = np.zeros((len(times), self.n_edges_))
        for t, stamp in enumerate(times):
            if t:
                self.propagate(imm=imm)
            observed_k = {edge: density[t, edge] for edge in range(self.n_edges_) if np.isfinite(density[t, edge])}
            observed_q = {edge: flow[t, edge] for edge in range(self.n_edges_) if np.isfinite(flow[t, edge])}
            self.assimilate(observed_k, observed_q, imm=imm)
            rho[t] = self._row_sums(imm.x)
            if bar is not None:
                lock = getattr(self, "_bar_lock", None)
                if lock is None:
                    bar.set_postfix_str(pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M"), refresh=False)
                    bar.update()
                else:
                    with lock:
                        bar.set_postfix_str(pd.Timestamp(stamp).strftime("%Y-%m-%d %H:%M"), refresh=False)
                        bar.update()
        return rho

    def _day_inputs(self, obs, ramps, times):
        lane = dict(zip(self.edge_ids_, self.lanes_))
        density = np.full((len(times), self.n_edges_), np.nan)
        flow = np.full((len(times), self.n_edges_), np.nan)
        mainline_ids = [self.edge_ids_[edge] for edge in self._mainline_edges]
        if mainline_ids and not obs.empty:
            part = obs.loc[obs["link_id"].isin(mainline_ids)].copy()
            part["lane"] = part["link_id"].map(lane)
            ok = part["flow_vph"].notna() & part["lane"].notna() & part["pct_observed"].ge(75) & part["speed_kmh"].gt(1e-3)
            part["k"] = np.where(ok, part["flow_vph"] / (part["lane"] * part["speed_kmh"]), np.nan)
            pivoted = _pivot(part, times, "link_id", mainline_ids, "k")
            for local, edge in enumerate(self._mainline_edges):
                density[:, edge] = pivoted[:, local]
        ramp_ids = [self.edge_ids_[edge] for edge in self._flow_edges]
        if ramp_ids and not ramps.empty:
            sub = ramps.loc[ramps["ramp_link_id"].isin(ramp_ids)].copy()
            sub["lane"] = sub["ramp_link_id"].map(lane)
            ok = sub["flow_vph"].notna() & sub["lane"].notna() & sub["pct_observed"].ge(75)
            sub["q"] = np.where(ok, sub["flow_vph"] / sub["lane"], np.nan)
            pivoted = _pivot(sub, times, "ramp_link_id", ramp_ids, "q")
            for local, edge in enumerate(self._flow_edges):
                flow[:, edge] = pivoted[:, local]
        density = np.clip(density, 0.0, None)
        flow = np.clip(flow, 0.0, None)
        return density, flow

    def _to_speed_flow(self, rho):
        rho = np.clip(rho, 0.0, self.k_j_)
        q = np.minimum(self.v_f_ * rho, self.w_c_ * (self.k_j_ - rho))
        q = np.clip(np.minimum(q, self.Q_m_), 0.0, None)
        speed = np.where(rho > 1e-6, q / np.maximum(rho, 1e-6), self.v_f_)
        return np.clip(speed, 0.0, None), q * self.lanes_

    def _plant_prior(self, link_flow, link_ids):
        diagonal = 0.5 * self.k_c_.copy()
        for flow, link_id in zip(link_flow, link_ids):
            edge = self._index.get(str(link_id))
            if edge is None or not np.isfinite(flow):
                continue
            per_lane = flow / self.lanes_[edge]
            diagonal[edge] = np.clip(per_lane / self.v_f_[edge], 0.0, self.k_j_[edge])
        self.imm.reset(self._from_diagonal(diagonal), self.P0_)

    def _mode_link_flow(self, mode, link_ids, fallback):
        density = np.clip(self._row_sums(self.imm.xs[mode]), 0.0, self.k_j_)
        if mode == 0:
            per_lane = self.v_f_ * density
        else:
            per_lane = self.w_c_ * np.maximum(self.k_j_ - density, 0.0)
        total = per_lane * self.lanes_
        out = np.asarray(fallback, dtype=float).copy()
        for row, link_id in enumerate(link_ids):
            edge = self._index.get(str(link_id))
            if edge is not None:
                out[row] = total[edge]
        return np.clip(out, 0.0, None)


def _path_flow(incidence, prior, target, regularization):
    def objective(flow):
        residual, movement = incidence @ flow - target, flow - prior
        loss = 0.5 * (residual @ residual + regularization * (movement @ movement))
        gradient = np.asarray(incidence.T @ residual + regularization * movement).ravel()
        return loss, gradient

    result = minimize(
        objective,
        prior,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] * len(prior),
        options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-5},
    )
    flow = np.clip(np.asarray(result.x, dtype=float), 0, None)
    if not np.isfinite(flow).all():
        raise RuntimeError(f"ODME solve failed: {result.message}")
    return flow


def _estimator(ctx):
    model = ctx.cache.get("imm") if isinstance(getattr(ctx, "cache", None), dict) else None
    if model is None:
        model = InteractingMultipleModelEstimator()
        model.set_params(**{name: ctx.params[name] for name in _PARAMS if name in ctx.params})
        model.fit(ctx)
        if isinstance(getattr(ctx, "cache", None), dict):
            ctx.cache["imm"] = model
    return model


def state(ctx):
    return _estimator(ctx).predict(ctx)


def queue(ctx):
    model = _estimator(ctx)
    pred = model.filter_horizon(ctx)
    free = ctx.network["links"].set_index("link_id")["free_speed_kmh"]
    ratio = float(ctx.params.get("queue_ratio", 0.6))
    pred["queue_pred"] = pred["speed_kmh"].le(ratio * pred["link_id"].map(free)).fillna(False).astype(int)
    return pred


def odme(ctx):
    return _estimator(ctx).estimate_path_flow(ctx)
