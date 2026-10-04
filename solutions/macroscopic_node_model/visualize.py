"""Draw LWR mainline topology as an interactive HTML graph."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from .network import LwrNetwork

_COLOR_MAINLINE = "#1f77b4"
_COLOR_IN = "#2ca02c"
_COLOR_OUT = "#d62728"
_NODE_MAINLINE = "#dbeafe"
_NODE_IN = "#86efac"
_NODE_OUT = "#fca5a5"
_NODE_DEFAULT = "#e5e7eb"
_CACHE_KEY = "topology_interactive_v5"
_HTML_SCALE = 70.0
_MAINLINE_SPACING = 2.4
_EXTERNAL_LAYER = 1.15
_JUNCTION_FAN_X = 0.4
_EXTERNAL_STAGGER = 0.07
_EXTERNAL_MIN_DX = 0.45
_EXTERNAL_X_SLACK = 1.2
_LABEL_VADJUST_STEP = 22
_SPRING_K = 0.62
_SPRING_ITERATIONS = 140


def _mainline_nodes(topo: pd.DataFrame) -> set[int]:
    nodes: set[int] = set()
    for _, row in topo.iterrows():
        nodes.add(int(row["from_node"]))
        nodes.add(int(row["to_node"]))
    return nodes


def _ordered_mainline_nodes(topo: pd.DataFrame) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for _, row in topo.sort_values("order_index").iterrows():
        for node in (int(row["from_node"]), int(row["to_node"])):
            if node not in seen:
                ordered.append(node)
                seen.add(node)
    return ordered


def _external_groups(
    graph: nx.DiGraph, mainline_nodes: set[int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    incoming_at: dict[int, list[int]] = defaultdict(list)
    outgoing_at: dict[int, list[int]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("type", "link")
        if edge_type in ("conn-in", "on-ramp") and u not in mainline_nodes:
            incoming_at[v].append(u)
        if edge_type in ("conn-out", "off-ramp") and v not in mainline_nodes:
            outgoing_at[u].append(v)
    return incoming_at, outgoing_at


def _initial_positions(
    graph: nx.DiGraph, topo: pd.DataFrame, mainline_nodes: set[int]
) -> dict[int, tuple[float, float]]:
    """Seed layout: mainline on a horizontal spine; externals near their junction."""
    positions: dict[int, tuple[float, float]] = {
        node: (index * _MAINLINE_SPACING, 0.0)
        for index, node in enumerate(_ordered_mainline_nodes(topo))
    }
    incoming_at, outgoing_at = _external_groups(graph, mainline_nodes)

    for junction, externals in incoming_at.items():
        anchor_x = positions.get(junction, (0.0, 0.0))[0]
        ordered = sorted(externals)
        count = len(ordered)
        for index, node in enumerate(ordered):
            x_offset = (index - (count - 1) / 2.0) * _JUNCTION_FAN_X
            positions[node] = (anchor_x + x_offset, _EXTERNAL_LAYER + index * _EXTERNAL_STAGGER)

    for junction, externals in outgoing_at.items():
        anchor_x = positions.get(junction, (0.0, 0.0))[0]
        ordered = sorted(externals)
        count = len(ordered)
        for index, node in enumerate(ordered):
            x_offset = (index - (count - 1) / 2.0) * _JUNCTION_FAN_X
            positions[node] = (anchor_x + x_offset, -_EXTERNAL_LAYER - index * _EXTERNAL_STAGGER)

    for node in graph.nodes:
        positions.setdefault(node, (0.0, 0.0))
    return positions


def _smart_layout(
    graph: nx.DiGraph, topo: pd.DataFrame, seed: int
) -> dict[int, tuple[float, float]]:
    """Force-directed polish: fix the mainline spine, relax CONN / ramp endpoints."""
    mainline_nodes = _mainline_nodes(topo)
    initial = _initial_positions(graph, topo, mainline_nodes)
    relaxed = nx.spring_layout(
        graph.to_undirected(),
        pos=initial,
        fixed=list(mainline_nodes),
        seed=seed,
        k=_SPRING_K,
        iterations=_SPRING_ITERATIONS,
        threshold=1e-4,
    )
    for node in mainline_nodes:
        relaxed[node] = initial[node]
    for node in graph.nodes:
        if node in mainline_nodes:
            continue
        init_x, init_y = initial[node]
        spring_x, _ = relaxed[node]
        x_delta = max(-_EXTERNAL_X_SLACK, min(_EXTERNAL_X_SLACK, spring_x - init_x))
        relaxed[node] = (init_x + x_delta, init_y)

    externals = sorted(
        (node for node in graph.nodes if node not in mainline_nodes),
        key=lambda node: (relaxed[node][1], relaxed[node][0], node),
    )
    min_dx = _EXTERNAL_MIN_DX
    for index in range(1, len(externals)):
        left, right = externals[index - 1], externals[index]
        lx, ly = relaxed[left]
        rx, ry = relaxed[right]
        if abs(ly - ry) > 0.05:
            continue
        if rx - lx < min_dx:
            shift = (min_dx - (rx - lx)) / 2.0
            relaxed[left] = (lx - shift, ly)
            relaxed[right] = (rx + shift, ry)
            externals[index - 1] = left
            externals[index] = right
    return relaxed


def _fmt_num(value, digits: int = 0) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    if digits == 0:
        return str(int(round(number)))
    formatted = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _edge_color(data: dict) -> str:
    edge_type = data.get("type", "link")
    if edge_type == "link":
        return _COLOR_MAINLINE
    if edge_type in ("on-ramp", "conn-in"):
        return _COLOR_IN
    if edge_type in ("off-ramp", "conn-out"):
        return _COLOR_OUT
    return "#999999"


def _endpoint_node_sets(graph: nx.DiGraph, mainline_nodes: set[int]) -> tuple[set[int], set[int]]:
    inbound: set[int] = set()
    outbound: set[int] = set()
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("type")
        if edge_type in ("on-ramp", "conn-in") and u not in mainline_nodes:
            inbound.add(u)
        if edge_type in ("off-ramp", "conn-out") and v not in mainline_nodes:
            outbound.add(v)
    return inbound, outbound


def _node_color(node: int, mainline_nodes: set[int], inbound: set[int], outbound: set[int]) -> str:
    if node in mainline_nodes:
        return _NODE_MAINLINE
    if node in inbound:
        return _NODE_IN
    if node in outbound:
        return _NODE_OUT
    return _NODE_DEFAULT


def _mainline_edge_label(data: dict) -> str:
    lines = [str(data.get("id", "")).strip()]
    lanes = _fmt_num(data.get("lanes"))
    length = _fmt_num(data.get("length"), 2)
    v_free = _fmt_num(data.get("v_f"))
    capacity = _fmt_num(data.get("C"))
    k_c = _fmt_num(data.get("k_c"), 1)
    k_jam = _fmt_num(data.get("k_j"), 0)
    if lanes:
        lines.append(f"L{lanes}")
    if length:
        lines.append(f"ℓ{length}")
    if v_free:
        lines.append(f"v_free{v_free}")
    if capacity:
        lines.append(f"C{capacity}")
    if k_c:
        lines.append(f"k_c{k_c}")
    if k_jam:
        lines.append(f"k_jam{k_jam}")
    return "\n".join(lines)


def _spread_junction_edges(keys: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], dict]]:
    """Offset labels and curve parallel edges that meet at the same node."""
    vadjusts: dict[tuple[int, int], int] = {}
    smooths: dict[tuple[int, int], dict] = {}
    if len(keys) <= 1:
        return vadjusts, smooths
    ordered = sorted(keys)
    count = len(ordered)
    for index, key in enumerate(ordered):
        vadjusts[key] = int((index - (count - 1) / 2.0) * _LABEL_VADJUST_STEP)
        curve = "curvedCW" if index % 2 == 0 else "curvedCCW"
        roundness = min(0.08 + (index // 2) * 0.06, 0.32)
        smooths[key] = {"enabled": True, "type": curve, "roundness": roundness}
    return vadjusts, smooths


def _junction_edge_styles(graph: nx.DiGraph) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], dict]]:
    vadjusts: dict[tuple[int, int], int] = {}
    smooths: dict[tuple[int, int], dict] = {}
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v in graph.edges():
        incoming[v].append((u, v))
        outgoing[u].append((u, v))

    for keys in incoming.values():
        adj, sm = _spread_junction_edges(keys)
        vadjusts.update(adj)
        smooths.update(sm)
    for keys in outgoing.values():
        adj, sm = _spread_junction_edges(keys)
        for key, value in adj.items():
            if key not in vadjusts:
                vadjusts[key] = value
        for key, value in sm.items():
            if key not in smooths:
                smooths[key] = value
    return vadjusts, smooths


def _aux_edge_label(link_id: str) -> tuple[str, dict]:
    """Full CONN / ramp id on the edge; wrap long synthetic ramp ids."""
    text = str(link_id).strip()
    if len(text) <= 28:
        return text, {"size": 8, "align": "middle"}
    if "_" in text:
        head, tail = text.rsplit("_", 1)
        return f"{head}\n{tail}", {"size": 7, "align": "middle", "multi": True}
    return text, {"size": 7, "align": "middle"}


def _vis_edge_label(data: dict) -> tuple[str, dict]:
    if data.get("type") == "link":
        return _mainline_edge_label(data), {"size": 7, "align": "middle", "multi": True}
    label, font = _aux_edge_label(str(data.get("id", "")))
    font.setdefault("multi", True)
    return label, font


def _legend_swatch(color: str) -> str:
    return (
        f'<span style="display:inline-block;width:14px;height:14px;margin-right:6px;'
        f'border:1px solid #333;background:{color};"></span>'
    )


def _legend_html() -> str:
    items = (
        f"<div>{_legend_swatch(_COLOR_MAINLINE)}mainline</div>"
        f"<div>{_legend_swatch(_COLOR_IN)}CONN-IN / on-ramp</div>"
        f"<div>{_legend_swatch(_COLOR_OUT)}CONN-OUT / off-ramp</div>"
        f"<div>{_legend_swatch(_NODE_IN)}in / on-ramp node</div>"
        f"<div>{_legend_swatch(_NODE_OUT)}out / off-ramp node</div>"
    )
    return (
        '<div style="position:fixed;top:12px;right:12px;z-index:9999;background:#fff;'
        'padding:10px 12px;border:1px solid #ccc;border-radius:6px;font:13px sans-serif;'
        'line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,.08);">'
        "<strong>Legend</strong>"
        f"{items}"
        "<div style='margin-top:8px;color:#555;font-size:12px;'>"
        "Mainline: link_id + one metric per line; parallel edges fan at junctions"
        "</div>"
        "</div>"
    )


def _draw_html(
    graph: nx.DiGraph, topo: pd.DataFrame, panel: str, output_path: Path, seed: int
) -> Path:
    positions = _smart_layout(graph, topo, seed)
    mainline_nodes = _mainline_nodes(topo)

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        cdn_resources="remote",
        heading=f"LWR mainline topology — {panel}",
        select_menu=False,
        filter_menu=False,
    )
    net.set_options(
        """
    {
      "physics": { "enabled": false },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "navigationButtons": true,
        "keyboard": { "enabled": true }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.65 } },
        "smooth": { "type": "continuous" },
        "font": {
          "size": 8,
          "face": "monospace",
          "align": "middle",
          "multi": true,
          "strokeWidth": 3,
          "strokeColor": "#ffffff",
          "background": "rgba(255,255,255,0.45)"
        },
        "scaling": {
          "label": { "enabled": true, "min": 6, "max": 12 }
        }
      },
      "nodes": {
        "shape": "dot",
        "size": 14,
        "font": { "size": 13, "face": "monospace" },
        "borderWidth": 2
      }
    }
    """
    )

    inbound_nodes, outbound_nodes = _endpoint_node_sets(graph, mainline_nodes)
    for node in graph.nodes:
        x, y = positions[node]
        net.add_node(
            node,
            label=str(node),
            x=float(x * _HTML_SCALE),
            y=float(-y * _HTML_SCALE),
            color=_node_color(node, mainline_nodes, inbound_nodes, outbound_nodes),
            borderWidth=2 if node in mainline_nodes else 1,
            title=f"Node {node}",
        )

    vadjusts, smooths = _junction_edge_styles(graph)
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("type", "link")
        label, font = _vis_edge_label(data)
        key = (u, v)
        if key in vadjusts:
            font = {**font, "vadjust": vadjusts[key]}
        edge_kwargs: dict = {
            "label": label,
            "font": font,
            "title": "",
            "width": 1.5 if edge_type == "link" else 1.0,
        }
        if key in smooths:
            edge_kwargs["smooth"] = smooths[key]
        net.add_edge(
            u,
            v,
            color=_edge_color(data),
            **edge_kwargs,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
    html = output_path.read_text(encoding="utf-8")
    if "<body>" in html:
        html = html.replace("<body>", f"<body>{_legend_html()}", 1)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def draw_topology(network: LwrNetwork, ctx=None) -> Path | None:
    """Save an interactive topology HTML for a built LwrNetwork (optional ctx cache)."""
    if ctx is not None and ctx.cache.get(_CACHE_KEY):
        return ctx.cache.get("topology_path")

    output = Path("results") / "macroscopic_node_model" / f"{network.panel}_topology.html"
    path = _draw_html(
        network.graph,
        network.topo,
        network.panel,
        output,
        seed=network.seed,
    )
    if ctx is not None:
        ctx.cache[_CACHE_KEY] = True
        ctx.cache["topology_path"] = path
    return path
