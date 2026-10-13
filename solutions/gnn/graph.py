"""Who is next to whom: the road graph used by the GCN."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray


def neighbour_matrices(
    topology: pl.LazyFrame, link_ids: list[str]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Row-normalised "link ahead" and "link behind" matrices, each (links, links)."""
    link_index = {link_id: index for index, link_id in enumerate(link_ids)}
    ahead = np.zeros((len(link_ids), len(link_ids)), dtype=np.float32)
    rows = topology.select("link_id", "outgoing_link_ids").collect().iter_rows()
    for link_id, outgoing in rows:
        if link_id not in link_index or outgoing is None:
            continue
        for next_link in outgoing.split(";"):
            if next_link in link_index:  # skips CONN-... connectors
                ahead[link_index[link_id], link_index[next_link]] = 1.0
    behind = ahead.T.copy()
    return _row_average(ahead), _row_average(behind)


def _row_average(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Divide each row by its number of neighbours; rows without neighbours stay 0."""
    return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
