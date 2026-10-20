"""Train the queue GCN for one panel."""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from trafficbench.contracts import JsonObject, Panel, ReleasePackageSlice

from .example import build_series, density_scale
from .graph import neighbour_matrices
from .grids import link_grids, fill_masked_states
from .network import QueueSTGNN
from .train import train_model


class PanelModel(NamedTuple):
    """Everything prediction needs from the training of one panel."""

    model: QueueSTGNN
    link_ids: list[str]
    free_speed_kmh: NDArray[np.float64]
    usual: NDArray[np.float64]  # usual queue rate, (288, links)
    density_scale: NDArray[np.float64]


def fit_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    parameters: JsonObject,
    train_start: datetime | None = None,  # inclusive
    train_end: datetime | None = None,  # exclusive
) -> PanelModel:
    panel_data = release_slice.panel_slices_by_panel[panel]

    links = panel_data.network.links.collect()
    link_ids = sorted(links["link_id"].to_list())
    free_speed_by_link = dict(zip(links["link_id"], links["free_speed_kmh"]))
    free_speed_kmh = np.array([free_speed_by_link[link_id] for link_id in link_ids])

    train_states = fill_masked_states(
        panel_data.masked_mainline_states["train"],
        panel_data.historical_task_labels.task1_state_answers,
    )
    if train_start is not None:
        train_states = train_states.filter(pl.col("timestamp") >= train_start)
    if train_end is not None:
        train_states = train_states.filter(pl.col("timestamp") < train_end)

    grids = link_grids(train_states, link_ids)
    print(f"{panel}: {grids.known.mean():.1%} of link-steps usable", flush=True)
    scale = density_scale(grids)
    series = build_series(grids, free_speed_kmh, scale)
    ahead, behind = neighbour_matrices(
        panel_data.network.lwr_mainline_topology, link_ids
    )

    model = train_model(
        series,
        ahead,
        behind,
        layers=int(parameters.get("layers", 2)),
        epochs=int(parameters.get("epochs", 6)),
        positive_weight=float(parameters.get("positive_weight", 3.0)),
        onset_weight=float(parameters.get("onset_weight", 3.0)),
    )
    return PanelModel(model, link_ids, free_speed_kmh, series.usual, scale)
