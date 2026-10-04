"""Draw LWR mainline topology as a directed graph from ctx.network tables."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

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


def _split_ids(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _split_flags(value) -> list[int | None]:
    parts = _split_ids(value)
    out: list[int | None] = []
    for part in parts:
        if part in ("", "nan"):
            out.append(None)
        else:
            out.append(int(float(part)))
    return out


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


def _conn_port(link_id: str) -> str:
    if link_id.endswith("-IN"):
        return "IN"
    if link_id.endswith("-OUT"):
        return "OUT"
    return "unknown"


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

    def add_edge(**kwargs) -> None:
        link_id = str(kwargs["link_id"])
        if link_id in added_link_ids:
            return
        added_link_ids.add(link_id)
        graph.add_edge(kwargs.pop("u"), kwargs.pop("v"), **kwargs)

    for _, row in topo.iterrows():
        link_id = str(row["link_id"])
        u = int(row["from_node"])
        v = int(row["to_node"])
        links_rec = links_by_id.get(link_id, {})
        fd_rec = fd_by_id.get(link_id, {})
        add_edge(
            u=u,
            v=v,
            kind="mainline",
            link_id=link_id,
            mainline_link_id=str(_scalar_row(row, "mainline_link_id") or link_id),
            order_index=_scalar_row(row, "order_index"),
            from_node=u,
            to_node=v,
            lanes=_scalar_row(row, "lanes"),
            length_km=_scalar_row(row, "length_km"),
            capacity_vph=_scalar_row(row, "capacity_vph"),
            free_speed_kmh=_scalar_row(row, "free_speed_kmh"),
            free_flow_speed_kmh=_scalar_row(row, "free_flow_speed_kmh"),
            has_sensor=_scalar_row(row, "has_sensor"),
            detector_id=_scalar_row(row, "detector_id"),
            incoming_link_ids=_scalar_row(row, "incoming_link_ids"),
            outgoing_link_ids=_scalar_row(row, "outgoing_link_ids"),
            on_ramp_link_ids=_scalar_row(row, "on_ramp_link_ids"),
            off_ramp_link_ids=_scalar_row(row, "off_ramp_link_ids"),
            incoming_has_sensor=_scalar_row(row, "incoming_has_sensor"),
            outgoing_has_sensor=_scalar_row(row, "outgoing_has_sensor"),
            n_incoming=_scalar_row(row, "n_incoming"),
            n_outgoing=_scalar_row(row, "n_outgoing"),
            n_incoming_no_sensor=_scalar_row(row, "n_incoming_no_sensor"),
            n_outgoing_no_sensor=_scalar_row(row, "n_outgoing_no_sensor"),
            links_length_km=_scalar(links_rec.get("length_km")),
            links_lanes=_scalar(links_rec.get("lanes")),
            links_free_speed_kmh=_scalar(links_rec.get("free_speed_kmh")),
            links_capacity_vph=_scalar(links_rec.get("capacity_vph")),
            critical_density=_scalar(fd_rec.get("critical_density")),
            k_jam=_scalar(fd_rec.get("k_jam")),
        )

        incoming = _split_ids(row["incoming_link_ids"])
        incoming_sensors = _split_flags(row["incoming_has_sensor"])
        for idx, inc_id in enumerate(incoming):
            if _is_mainline(inc_id, mainline_ids):
                continue
            sensor = incoming_sensors[idx] if idx < len(incoming_sensors) else None
            if _is_conn(inc_id):
                ext = external_node(inc_id)
                add_edge(
                    u=ext,
                    v=u,
                    kind="conn",
                    link_id=inc_id,
                    conn_port=_conn_port(inc_id),
                    has_sensor=sensor,
                    host_mainline_link_id=link_id,
                    host_from_node=u,
                    host_to_node=v,
                    attach_node=u,
                    host_incoming_link_ids=_scalar_row(row, "incoming_link_ids"),
                    host_outgoing_link_ids=_scalar_row(row, "outgoing_link_ids"),
                    host_incoming_has_sensor=_scalar_row(row, "incoming_has_sensor"),
                    host_outgoing_has_sensor=_scalar_row(row, "outgoing_has_sensor"),
                )

        outgoing = _split_ids(row["outgoing_link_ids"])
        outgoing_sensors = _split_flags(row["outgoing_has_sensor"])
        for idx, out_id in enumerate(outgoing):
            if _is_mainline(out_id, mainline_ids):
                continue
            sensor = outgoing_sensors[idx] if idx < len(outgoing_sensors) else None
            if _is_conn(out_id):
                ext = external_node(out_id)
                add_edge(
                    u=v,
                    v=ext,
                    kind="conn",
                    link_id=out_id,
                    conn_port=_conn_port(out_id),
                    has_sensor=sensor,
                    host_mainline_link_id=link_id,
                    host_from_node=u,
                    host_to_node=v,
                    attach_node=v,
                    host_incoming_link_ids=_scalar_row(row, "incoming_link_ids"),
                    host_outgoing_link_ids=_scalar_row(row, "outgoing_link_ids"),
                    host_incoming_has_sensor=_scalar_row(row, "incoming_has_sensor"),
                    host_outgoing_has_sensor=_scalar_row(row, "outgoing_has_sensor"),
                )

        for ramp_id in _split_ids(row["on_ramp_link_ids"]):
            ext = external_node(ramp_id)
            ramp_rec = ramp_by_id.get(ramp_id, {})
            add_edge(
                u=ext,
                v=u,
                kind="on_ramp",
                link_id=ramp_id,
                ramp_type=_scalar(ramp_rec.get("ramp_type")) or "OR",
                ramp_link_id=ramp_id,
                nearest_mainline_link_id=_scalar(ramp_rec.get("nearest_mainline_link_id")),
                host_mainline_link_id=link_id,
                host_from_node=u,
                host_to_node=v,
                attach_node=u,
                has_sensor=None,
            )

        for ramp_id in _split_ids(row["off_ramp_link_ids"]):
            ext = external_node(ramp_id)
            ramp_rec = ramp_by_id.get(ramp_id, {})
            add_edge(
                u=v,
                v=ext,
                kind="off_ramp",
                link_id=ramp_id,
                ramp_type=_scalar(ramp_rec.get("ramp_type")) or "FR",
                ramp_link_id=ramp_id,
                nearest_mainline_link_id=_scalar(ramp_rec.get("nearest_mainline_link_id")),
                host_mainline_link_id=link_id,
                host_from_node=u,
                host_to_node=v,
                attach_node=v,
                has_sensor=None,
            )

    return graph


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
        kind = data.get("kind", "mainline")
        if kind in ("conn", "on_ramp") and u not in mainline_nodes:
            incoming_at[v].append(u)
        if kind in ("conn", "off_ramp") and v not in mainline_nodes:
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


def _sensor_label(has_sensor) -> str:
    if has_sensor is None:
        return "?"
    return "S" if int(has_sensor) else "0"


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
    kind = data.get("kind", "mainline")
    if kind == "mainline":
        return _COLOR_MAINLINE
    if kind == "on_ramp" or (kind == "conn" and data.get("conn_port") == "IN"):
        return _COLOR_IN
    if kind == "off_ramp" or (kind == "conn" and data.get("conn_port") == "OUT"):
        return _COLOR_OUT
    return "#999999"


def _endpoint_node_sets(graph: nx.DiGraph, mainline_nodes: set[int]) -> tuple[set[int], set[int]]:
    inbound: set[int] = set()
    outbound: set[int] = set()
    for u, v, data in graph.edges(data=True):
        kind = data.get("kind")
        if kind == "on_ramp" or (kind == "conn" and data.get("conn_port") == "IN"):
            if u not in mainline_nodes:
                inbound.add(u)
        if kind == "off_ramp" or (kind == "conn" and data.get("conn_port") == "OUT"):
            if v not in mainline_nodes:
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
    lines = [str(data.get("link_id", "")).strip()]
    lanes = _fmt_num(data.get("lanes") or data.get("links_lanes"))
    length = _fmt_num(data.get("length_km") or data.get("links_length_km"), 2)
    v_free = _fmt_num(
        data.get("free_speed_kmh")
        or data.get("links_free_speed_kmh")
        or data.get("free_flow_speed_kmh")
    )
    capacity = _fmt_num(data.get("capacity_vph") or data.get("links_capacity_vph"))
    k_c = _fmt_num(data.get("critical_density"), 1)
    k_jam = _fmt_num(data.get("k_jam"), 0)
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
    kind = data.get("kind", "mainline")
    if kind == "mainline":
        return _mainline_edge_label(data), {"size": 7, "align": "middle", "multi": True}
    label, font = _aux_edge_label(str(data.get("link_id", "")))
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
        kind = data.get("kind", "mainline")
        label, font = _vis_edge_label(data)
        key = (u, v)
        if key in vadjusts:
            font = {**font, "vadjust": vadjusts[key]}
        edge_kwargs: dict = {
            "label": label,
            "font": font,
            "title": "",
            "width": 1.5 if kind == "mainline" else 1.0,
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


def draw_topology(ctx) -> Path | None:
    """Build and save a directed topology figure for ctx.panel (once per cache)."""
    if ctx.cache.get(_CACHE_KEY):
        return ctx.cache.get("topology_path")

    topo = ctx.network.get("lwr_mainline_topology")
    if topo is None or topo.empty:
        return None

    graph = _build_graph(topo, ctx.network)
    output = Path("results") / "macroscopic_node_model" / f"{ctx.panel}_topology.html"
    path = _draw_html(graph, topo, ctx.panel, output, seed=ctx.seed)
    ctx.cache[_CACHE_KEY] = True
    ctx.cache["topology_path"] = path
    return path
