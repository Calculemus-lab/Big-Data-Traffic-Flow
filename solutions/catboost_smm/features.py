"""Quality-aware state features with SMM baseline slots (94 + 3 SMM extras)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import sparse

RATE = np.array([0.20, 0.30, 0.50], dtype=np.float32)
TIME_OFFSETS = (-12, -6, -3, -2, -1, 1, 2, 3, 6, 12)


def eligible_mask(pct_observed, speed, flow):
    pct = np.asarray(pct_observed, dtype=float)
    return (pct >= 75.0) & np.isfinite(speed) & np.isfinite(flow)


def simulated_mask(panel_name, seed_base, times, speed, flow, eligible):
    dates = pd.Index([text[:10] for text in times])
    result = np.zeros(speed.shape, dtype=bool)
    regimes = np.empty(speed.shape[0], dtype=np.uint8)
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        digest = hashlib.blake2b(f"{seed_base}|{panel_name}|{date}".encode(), digest_size=8).digest()
        day_seed = int.from_bytes(digest, "big")
        regime = day_seed % 3
        rng = np.random.default_rng(day_seed)
        regimes[rows] = regime
        result[rows] = (
            (rng.random((len(rows), speed.shape[1])) < RATE[regime])
            & eligible[rows]
            & np.isfinite(speed[rows])
            & np.isfinite(flow[rows])
        )
    return result, regimes


def sampled_targets(mask, seed_base, times, per_day):
    dates = pd.Index([text[:10] for text in times])
    row_out, col_out = [], []
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        coords = np.argwhere(mask[rows])
        if not len(coords):
            continue
        digest = hashlib.blake2b(f"{seed_base}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        chosen = coords[rng.choice(len(coords), size=min(per_day, len(coords)), replace=False)]
        row_out.append(rows[chosen[:, 0]])
        col_out.append(chosen[:, 1])
    if not row_out:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.concatenate(row_out), np.concatenate(col_out)


def graph_mean(graph, values):
    observed = np.isfinite(values).astype(np.float32)
    count = (graph @ observed.T).T
    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)
    mean = np.where(count > 0, mean, np.nan)
    return mean, count


class FeaturePanel:
    def __init__(self, network, link_ids, panel_no=0):
        self.panel_no = panel_no
        self.links = [str(x) for x in link_ids]
        self.link_index = {key: i for i, key in enumerate(self.links)}
        fd = network["fd_parameters"].copy()
        fd["link_id"] = fd["link_id"].astype(str)
        fd = fd.drop_duplicates("link_id").set_index("link_id").reindex(self.links)
        self.lanes = np.maximum(fd["lanes"].to_numpy(dtype=np.float32), 1.0)
        self.vf = np.maximum(fd["free_speed_kmh"].to_numpy(dtype=np.float32), 1e-3)
        self.cap = np.maximum(fd["capacity_vph"].to_numpy(dtype=np.float32), 0.0)
        self.length = np.maximum(fd["length_km"].to_numpy(dtype=np.float32), 1e-3)
        topo = network.get("lwr_mainline_topology")
        if topo is None or topo.empty:
            topo = _chain_topology(self.links)
        self.incoming = self._adjacency(topo, "incoming_link_ids")
        self.outgoing = self._adjacency(topo, "outgoing_link_ids")
        self.milepost = self._mileposts(network)

    def _adjacency(self, topo, column):
        link_col = "mainline_link_id" if "mainline_link_id" in topo.columns else "link_id"
        topo = topo.copy()
        topo[link_col] = topo[link_col].astype(str)
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(getattr(row, link_col))
            if link not in self.link_index:
                continue
            raw = getattr(row, column, None)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            for other in str(raw).split(";"):
                other = other.strip()
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        mat = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(self.links), len(self.links)),
        )
        mat.data[:] = 1.0
        return mat

    def _mileposts(self, network):
        links = network.get("links")
        if links is not None and "milepost" in links.columns:
            loc = links.copy()
            loc["link_id"] = loc["link_id"].astype(str)
            mp = loc.drop_duplicates("link_id").set_index("link_id").reindex(self.links)["milepost"]
            mp = mp.to_numpy(dtype=np.float32)
            if np.isfinite(mp).all():
                self.sorted_links = np.argsort(mp)
                self.link_rank = np.empty(len(self.links), dtype=np.int32)
                self.link_rank[self.sorted_links] = np.arange(len(self.links))
                return mp
        mp = np.arange(len(self.links), dtype=np.float32)
        self.sorted_links = mp.astype(np.int32)
        self.link_rank = mp.astype(np.int32)
        return mp


def _chain_topology(links):
    nodes = [f"n{i}" for i in range(len(links) + 1)]
    rows = []
    for i, link in enumerate(links):
        incoming = links[i - 1] if i > 0 else ""
        outgoing = links[i + 1] if i + 1 < len(links) else ""
        rows.append(
            {
                "from_node": nodes[i],
                "to_node": nodes[i + 1],
                "link_id": link,
                "incoming_link_ids": incoming,
                "outgoing_link_ids": outgoing,
            }
        )
    return pd.DataFrame(rows)


def matrices(frame, panel):
    frame = frame.copy()
    frame["link_id"] = frame["link_id"].astype(str)
    index = pd.Index(sorted(frame.timestamp.unique()))
    cols = panel.links
    speed = (
        frame.pivot(index="timestamp", columns="link_id", values="speed_kmh")
        .reindex(index=index, columns=cols)
        .to_numpy(dtype=np.float32)
    )
    flow = (
        frame.pivot(index="timestamp", columns="link_id", values="flow_vph")
        .reindex(index=index, columns=cols)
        .to_numpy(dtype=np.float32)
    )
    if "pct_observed" in frame.columns:
        pct = (
            frame.pivot(index="timestamp", columns="link_id", values="pct_observed")
            .reindex(index=index, columns=cols)
            .to_numpy(dtype=np.float32)
        )
    else:
        pct = np.full_like(speed, 100.0)
    eligible = eligible_mask(pct, speed, flow)
    times = index.astype(str).tolist()
    return times, speed, flow, eligible, pct


def compute_features(
    panel,
    times,
    observed_speed,
    observed_flow,
    eligible,
    smm_speed,
    smm_flow,
    rows,
    cols,
    regime,
):
    n_times, n_links = observed_speed.shape
    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])
    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, 0.6 * panel.cap[None, :])

    graph = (panel.incoming + panel.outgoing).tocsr()
    graph.data[:] = 1.0
    spatial_flow, spatial_count = graph_mean(graph, observed_flow)
    flow_base = np.where(spatial_count > 0, 0.5 * temporal_flow + 0.5 * spatial_flow, temporal_flow)
    up_speed, up_count = graph_mean(panel.incoming, observed_speed)
    down_speed, down_count = graph_mean(panel.outgoing, observed_speed)
    up_flow, _ = graph_mean(panel.incoming, observed_flow)
    down_flow, _ = graph_mean(panel.outgoing, observed_flow)

    valid = np.isfinite(observed_speed)
    positions = np.arange(n_times, dtype=np.int32)[:, None]
    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)
    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]
    p = prev_index[rows, cols]
    n = next_index[rows, cols]
    prev_v = observed_speed[np.maximum(p, 0), cols].copy()
    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()
    prev_q = observed_flow[np.maximum(p, 0), cols].copy()
    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()
    prev_v[p < 0] = np.nan
    prev_q[p < 0] = np.nan
    next_v[n >= n_times] = np.nan
    next_q[n >= n_times] = np.nan
    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)
    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)

    stamp = pd.to_datetime(times, utc=True)
    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60
    dow = stamp.dayofweek.to_numpy(dtype=np.float32)
    lanes = np.maximum(panel.lanes[cols], 1.0)
    smm_v = smm_speed[rows, cols]
    smm_q_lane = smm_flow[rows, cols] / lanes
    base_speed = smm_v
    base_flow = smm_q_lane
    temporal_flow_lane = temporal_flow[rows, cols] / lanes
    spatial_flow_lane = spatial_flow[rows, cols] / lanes
    up_q = up_flow[rows, cols] / lanes
    down_q = down_flow[rows, cols] / lanes
    regime_vector = regime if len(regime) == len(rows) else regime[rows]
    features = [
        np.full(len(rows), panel.panel_no),
        cols.astype(np.float32),
        regime_vector.astype(np.float32),
        tod[rows],
        np.sin(2 * np.pi * tod[rows] / 24),
        np.cos(2 * np.pi * tod[rows] / 24),
        dow[rows],
        panel.vf[cols],
        panel.cap[cols] / lanes,
        lanes,
        panel.length[cols],
        base_speed,
        base_flow,
        temporal_flow_lane,
        spatial_flow_lane,
        base_speed / panel.vf[cols],
        base_flow * lanes / panel.cap[cols],
        prev_v,
        next_v,
        prev_q / lanes,
        next_q / lanes,
        prev_gap,
        next_gap,
        next_v - prev_v,
        (next_q - prev_q) / lanes,
        up_speed[rows, cols],
        down_speed[rows, cols],
        up_q,
        down_q,
        up_count[rows, cols],
        down_count[rows, cols],
        spatial_count[rows, cols],
        up_speed[rows, cols] - down_speed[rows, cols],
        up_q - down_q,
    ]
    for offset in (-2, -1, 1, 2):
        ranks = panel.link_rank[cols] + offset
        valid_neighbor = (ranks >= 0) & (ranks < n_links)
        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links - 1)]
        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1.0)
        same_time_v = observed_speed[rows, neighbor]
        same_time_q = observed_flow[rows, neighbor] / neighbor_lanes
        filled_v = temporal_speed[rows, neighbor]
        filled_q = temporal_flow[rows, neighbor] / neighbor_lanes
        features.extend(
            (
                np.where(valid_neighbor, np.abs(panel.milepost[neighbor] - panel.milepost[cols]), np.nan),
                np.where(valid_neighbor, same_time_v, np.nan),
                np.where(valid_neighbor, same_time_q, np.nan),
                np.where(valid_neighbor, filled_v, np.nan),
                np.where(valid_neighbor, filled_q, np.nan),
            )
        )
    extra = []
    for offset in TIME_OFFSETS:
        rr = rows + offset
        valid_t = (rr >= 0) & (rr < n_times)
        rr = np.clip(rr, 0, n_times - 1)
        vv = observed_speed[rr, cols]
        qq = observed_flow[rr, cols] / lanes
        extra.extend(
            [
                np.where(valid_t, vv, np.nan),
                np.where(valid_t, qq, np.nan),
                np.where(valid_t, qq / np.maximum(vv, 1.0), np.nan),
                np.where(valid_t & eligible[rr, cols] & np.isfinite(vv), 1.0, 0.0),
            ]
        )
    smm_density = smm_q_lane / np.maximum(smm_v, 1.0)
    extra.extend([smm_v, smm_q_lane, smm_density])
    x = np.column_stack([*features, *extra]).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x, base_speed, base_flow


def feature_rows(
    panel,
    times,
    speed,
    flow,
    eligible,
    smm_speed,
    smm_flow,
    rows,
    cols,
    regime,
):
    masked_v = np.where(np.isfinite(speed), speed, np.nan)
    masked_q = np.where(np.isfinite(flow), flow, np.nan)
    return compute_features(
        panel,
        times,
        masked_v,
        masked_q,
        eligible,
        smm_speed,
        smm_flow,
        rows,
        cols,
        regime,
    )
