"""Tests for the local Task 3 scorer in scripts/task3_local_score.py."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

SPEC = importlib.util.spec_from_file_location(
    "task3_local_score", Path(__file__).parents[1] / "scripts" / "task3_local_score.py"
)
assert SPEC and SPEC.loader
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)

START = datetime(2030, 12, 1, tzinfo=UTC)


def link_rows(link: str, steps: list[int], accumulation: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=5 * step) for step in steps],
            "link_id": [link] * len(steps),
            "accumulation": accumulation,
        }
    )


def fd_state(
    flow: list[float], speed: list[float], is_target: list[float]
) -> pl.DataFrame:
    rows = len(flow)
    return pl.DataFrame(
        {
            "flow_vph": flow,
            "speed_kmh": speed,
            "is_target": is_target,
            "lanes": [2.0] * rows,
            "critical_density": [40.0] * rows,
            "k_jam": [200.0] * rows,
            "capacity_vph": [4000.0] * rows,
            "free_speed_kmh": [100.0] * rows,
        }
    )


def test_fd_score_is_one_on_the_free_flow_branch_and_zero_for_empty_roads() -> None:
    # Free-flow points (speed = free speed) lie exactly on the diagram.
    assert score.fd_score(
        fd_state([1000.0, 3000.0], [100.0, 100.0], [1, 1])
    ) == pytest.approx(1)
    assert score.fd_score(fd_state([10.0, 3000.0], [100.0, 100.0], [1, 0])) == 0


def test_transitions_follow_the_official_filters() -> None:
    state = pl.concat(
        [
            link_rows(
                "A", [0, 1, 3], [10.0, 12.0, 15.0]
            ),  # gap of two steps is not a transition
            link_rows("B", [0, 1], [5.0, 4.0]),  # needs neighbour C, which is absent
            link_rows("R", [0, 1], [7.0, 9.0]),  # on-ramp valid only at step 0
        ]
    )
    topology = pl.DataFrame(
        {
            "link_id": ["A", "B", "C", "R"],
            "incoming_link_ids": ["", "C", "", ""],
            "outgoing_link_ids": ["", "", "", ""],
            "on_ramp_link_ids": ["", "", "", "RAMP"],
            "off_ramp_link_ids": ["", "", "", ""],
        }
    )
    ramps = pl.DataFrame(
        {
            "timestamp": [START],
            "link_id": ["R"],
            "on_valid": [True],
            "off_valid": [False],
        }
    )
    changes = score.scored_transitions(state, topology, ramps).sort("link_id")
    assert changes["link_id"].to_list() == ["A", "R"]
    assert changes["dN"].to_list() == [2.0, 2.0]
