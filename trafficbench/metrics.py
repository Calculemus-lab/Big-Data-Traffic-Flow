import numpy as np
import pandas as pd
from scipy import sparse

from .contracts import KEYS


def operator(network, targets, counts):
    paths = targets.path_id.tolist()
    if len(set(paths)) != len(paths):
        raise ValueError("ODME context must contain one departure period")
    path_index = {p: i for i, p in enumerate(paths)}
    link_index = {link: i for i, link in enumerate(counts.link_id)}
    incidence = network["path_link_incidence"]
    if set(incidence.path_id) - set(paths) or set(counts.link_id) - set(incidence.link_id):
        raise ValueError("Incidence, counts and path IDs do not match")
    edges = incidence[incidence.link_id.isin(link_index)].drop_duplicates(["link_id", "path_id"])
    return sparse.csr_matrix((np.ones(len(edges)), (edges.link_id.map(link_index), edges.path_id.map(path_index))), shape=(len(counts), len(paths)))


def state_metrics(pred, truth, network):
    frame = truth.merge(pred, on=KEYS["state"], suffixes=("_true", "_pred"), validate="one_to_one")
    fd = network["fd_parameters"].set_index("link_id")
    lanes = frame.link_id.map(fd.lanes)
    if lanes.isna().any() or (lanes <= 0).any():
        raise ValueError("Missing or invalid lane counts")
    frame["speed_sq"] = (frame.speed_kmh_pred - frame.speed_kmh_true) ** 2
    frame["flow_sq"] = ((frame.flow_vph_pred - frame.flow_vph_true) / lanes) ** 2
    rows = []
    for regime in ["R1", "R2", "R3"]:
        group = frame[frame.mask_regime.eq(regime)]
        if group.empty:
            raise ValueError(f"No targets for {regime}; benchmark is incomplete")
        speed, flow = np.sqrt(group[["speed_sq", "flow_sq"]].mean())
        rows.append({"regime": regime, "n": len(group), "speed_rmse": speed, "flow_per_lane_rmse": flow,
                     "state_score": .54 * max(0, 1 - speed / 25) + .46 * max(0, 1 - flow / 600)})
    return rows


def physics_diagnostics(pred, network):
    fd = network["fd_parameters"].set_index("link_id").reindex(pred.link_id)
    q, v = pred.flow_vph.to_numpy(), pred.speed_kmh.to_numpy()
    density = q / np.maximum(v, 1)
    critical = fd.critical_density.to_numpy(float)
    jam = fd.k_jam.to_numpy(float)
    wave = fd.capacity_vph.to_numpy(float) / np.maximum(jam - critical, 1e-9)
    expected = np.where(density <= critical, fd.free_speed_kmh.to_numpy(float) * density, wave * np.maximum(jam - density, 0))
    return {"fd_relative_error_diagnostic": float(np.abs(q - expected).sum() / max(np.abs(q).sum(), 1e-9)),
            "low_flow_fraction_diagnostic": float((q < 50).mean()),
            "negative_state_fraction_diagnostic": float(((q < 0) | (v < 0)).mean())}


def queue_metrics(pred, truth):
    frame = truth.merge(pred, on=KEYS["queue"], validate="one_to_one")
    frame = frame[frame.eligible.astype(bool)]
    if frame.empty:
        raise ValueError("Queue window has no eligible labels")
    true, guess = frame.queue_true.astype(bool), frame.queue_pred.astype(bool)
    union = int((true | guess).sum())
    return {"queue_proxy_iou": float((true & guess).sum() / union) if union else 1.0, "n": len(frame)}


def odme_metrics(pred, counts, prior, network, truth=None):
    A = operator(network, pred, counts)
    f = pred.path_flow.to_numpy(float)
    b = prior.set_index("path_id").reindex(pred.path_id).path_flow.to_numpy(float)
    c = counts["count"].to_numpy(float)
    result = {"odme_link_score_diagnostic": max(0., 1 - float(np.abs(A @ f - c).sum() / max(c.sum(), 1e-9))),
              "odme_prior_relative_movement": float(np.abs(f - b).sum() / max(b.sum(), 1e-9))}
    if truth is not None:
        star = truth.set_index("path_id").reindex(pred.path_id).path_flow.to_numpy(float)
        score_od = max(0., 1 - float(np.abs(f - star).sum() / max(star.sum(), 1e-9)))
        dev = float(np.exp(-abs(np.abs(f - b).sum() / max(np.abs(star - b).sum(), 1e-9) - 1)))
        attraction = pd.DataFrame({"zone": pred.destination_zone, "f": f, "star": star}).groupby("zone")[["f", "star"]].sum()
        attr = max(0., 1 - float(np.abs(attraction.f / max(f.sum(), 1e-9) - attraction.star / max(star.sum(), 1e-9)).sum()) / 2)
        result["odme_synthetic_score"] = .45 * score_od + .25 * result["odme_link_score_diagnostic"] + .15 * dev + .15 * attr
    return result


def aggregate(rows):
    """Equal means at each level. No fabricated overall competition score."""
    frame = pd.DataFrame(rows)
    result = {}
    metrics = [c for c in frame if c.endswith(("score", "iou", "diagnostic", "movement", "rmse"))]
    for metric in metrics:
        d = frame.dropna(subset=[metric]).copy()
        if metric == "queue_proxy_iou":
            d = d.groupby(["fold", "panel", "family", "condition"], as_index=False)[metric].mean()
        d = d.groupby(["fold", "panel", "family"], as_index=False)[metric].mean()
        d = d.groupby(["fold", "family"], as_index=False)[metric].mean()
        result[metric] = float(d.groupby("fold")[metric].mean().mean())
    return result
