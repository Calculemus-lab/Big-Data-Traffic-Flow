"""Build a directed LWR graph from benchmark network tables."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

EdgeType = str  # link | on-ramp | off-ramp | conn-in | conn-out


@dataclass
class LwrNetwork:
    """Directed graph of mainline links, connectors, and ramps."""

    graph: nx.DiGraph
    topo: pd.DataFrame
    panel: str
    seed: int


def _split_ids(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _is_mainline(link_id: str, mainline_ids: set[str]) -> bool:
    return link_id in mainline_ids


def _is_conn(link_id: str) -> bool:
    return link_id.startswith("CONN-")


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and (not value.strip() or value.lower() == "nan"):
        return None
    return value


def _scalar_row(row: pd.Series, key: str):
    if key not in row.index:
        return None
    return _scalar(row[key])


def _as_float(value):
    value = _scalar(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    value = _scalar(value)
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _link_edge_attrs(
    link_id: str,
    row: pd.Series,
    links_rec: dict,
    fd_rec: dict,
) -> dict:
    lanes = _as_int(_scalar_row(row, "lanes") or links_rec.get("lanes"))
    length = _as_float(_scalar_row(row, "length_km") or links_rec.get("length_km"))
    v_f = _as_float(
        _scalar_row(row, "free_speed_kmh")
        or _scalar_row(row, "free_flow_speed_kmh")
        or links_rec.get("free_speed_kmh")
    )
    capacity = _as_float(_scalar_row(row, "capacity_vph") or links_rec.get("capacity_vph"))
    k_c = _as_float(fd_rec.get("critical_density"))
    k_j = _as_float(fd_rec.get("k_jam"))
    attrs: dict = {
        "type": "link",
        "id": link_id,
    }
    if lanes is not None:
        attrs["lanes"] = lanes
    if length is not None:
        attrs["length"] = length
    if v_f is not None:
        attrs["v_f"] = v_f
    if capacity is not None:
        attrs["C"] = capacity
    if k_c is not None:
        attrs["k_c"] = k_c
    if k_j is not None:
        attrs["k_j"] = k_j
    return attrs


def _aux_edge_attrs(edge_type: EdgeType, link_id: str) -> dict:
    return {"type": edge_type, "id": link_id}


def _build_graph(topo: pd.DataFrame, network: dict) -> nx.DiGraph:
    mainline_ids = set(topo["link_id"].astype(str))
    ramp_map = network.get("ramp_attachment_map")
    links_table = network.get("links")
    fd_table = network.get("fd_parameters")

    ramp_by_id: dict[str, dict] = {}
    if ramp_map is not None and not ramp_map.empty:
        ramp_by_id = ramp_map.set_index("ramp_link_id").to_dict("index")

    links_by_id: dict[str, dict] = {}
    if links_table is not None and not links_table.empty:
        links_by_id = links_table.set_index("link_id").to_dict("index")

    fd_by_id: dict[str, dict] = {}
    if fd_table is not None and not fd_table.empty:
        fd_by_id = fd_table.set_index("link_id").to_dict("index")

    max_node = int(max(topo["from_node"].max(), topo["to_node"].max()))
    next_external = max_node + 1
    external_nodes: dict[str, int] = {}

    def external_node(link_id: str) -> int:
        nonlocal next_external
        if link_id not in external_nodes:
            external_nodes[link_id] = next_external
            next_external += 1
        return external_nodes[link_id]

    graph = nx.DiGraph()
    added_link_ids: set[str] = set()

    def add_edge(u: int, v: int, link_id: str, attrs: dict) -> None:
        if link_id in added_link_ids:
            return
        added_link_ids.add(link_id)
        graph.add_edge(u, v, **attrs)

    for _, row in topo.iterrows():
        link_id = str(row["link_id"])
        u = int(row["from_node"])
        v = int(row["to_node"])
        links_rec = links_by_id.get(link_id, {})
        fd_rec = fd_by_id.get(link_id, {})
        add_edge(
            u,
            v,
            link_id,
            _link_edge_attrs(link_id, row, links_rec, fd_rec),
        )

        incoming = _split_ids(row["incoming_link_ids"])
        for inc_id in incoming:
            if _is_mainline(inc_id, mainline_ids):
                continue
            if _is_conn(inc_id):
                ext = external_node(inc_id)
                add_edge(
                    ext,
                    u,
                    inc_id,
                    _aux_edge_attrs("conn-in", inc_id),
                )

        outgoing = _split_ids(row["outgoing_link_ids"])
        for out_id in outgoing:
            if _is_mainline(out_id, mainline_ids):
                continue
            if _is_conn(out_id):
                ext = external_node(out_id)
                add_edge(
                    v,
                    ext,
                    out_id,
                    _aux_edge_attrs("conn-out", out_id),
                )

        for ramp_id in _split_ids(row["on_ramp_link_ids"]):
            ext = external_node(ramp_id)
            add_edge(ext, u, ramp_id, _aux_edge_attrs("on-ramp", ramp_id))

        for ramp_id in _split_ids(row["off_ramp_link_ids"]):
            ext = external_node(ramp_id)
            add_edge(v, ext, ramp_id, _aux_edge_attrs("off-ramp", ramp_id))

    return graph


_BUILD_CACHE_KEY = "lwr_network"


def build_network(ctx) -> LwrNetwork | None:
    """Build (or return cached) LWR directed graph for ctx.panel."""
    cached = ctx.cache.get(_BUILD_CACHE_KEY)
    if cached is not None:
        return cached

    topo = ctx.network.get("lwr_mainline_topology")
    if topo is None or topo.empty:
        return None

    graph = _build_graph(topo, ctx.network)
    built = LwrNetwork(
        graph=graph,
        topo=topo,
        panel=ctx.panel,
        seed=ctx.seed,
    )
    ctx.cache[_BUILD_CACHE_KEY] = built
    return built
