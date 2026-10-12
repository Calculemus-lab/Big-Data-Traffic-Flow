"""CPU checks for the milepost-time diffusion inpainter."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from solutions.stable_diffusion.field import panel_fields
from solutions.stable_diffusion.model import state
from solutions.stable_diffusion.train import fields_from_unmasked_release, fit_panels
from trafficbench.contracts import KEYS, VALUES
from trafficbench.data import Release, load_release_slice
from trafficbench.fixture import make_fixture


def test_observed_speed_lands_on_the_milepost_time_grid() -> None:
    """A usable measurement keeps its detector row and five-minute column."""
    timestamp = "2030-11-10T08:05:00Z"
    observations = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "station_id": ["S01", "S02"],
            "link_id": ["L01", "L02"],
            "milepost": [1.5, 0.5],
            "speed_kmh": [40.0, 95.0],
            "flow_vph": [2000.0, 4000.0],
            "pct_observed": [100, 100],
        }
    )
    scales = pl.DataFrame(
        {
            "link_id": ["L01", "L02"],
            "free_speed_kmh": [100.0, 100.0],
            "capacity_vph": [4000.0, 8000.0],
        }
    )
    fields = panel_fields(observations, scales)
    assert fields is not None
    assert [detector.link_id for detector in fields.axis] == ["L02", "L01"]
    day = fields.days[0]
    assert day.day == date(2030, 11, 10)
    assert day.observed[0, 8 * 12 + 1]
    assert day.speed[0, 8 * 12 + 1] == 95.0
    assert not day.observed[0, 0]


def test_state_diffusion_fills_target_keys(tmp_path: Path) -> None:
    """Sampling writes finite speed and flow on exactly the requested rows."""
    release_directory = make_fixture(tmp_path / "release")
    panel_grids = fields_from_unmasked_release(release_directory, "D12_I5_N")
    assert panel_grids is not None
    checkpoint = tmp_path / "checkpoint.pt"
    fit_panels(
        [panel_grids],
        timesteps=4,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        base_channels=4,
        device_name="cpu",
        seed=1,
        output_path=checkpoint,
        max_days=1,
    )
    release_slice = load_release_slice(
        Release(release_directory),
        date(2030, 11, 3),
        date(2030, 11, 10),
        date(2030, 11, 10),
        date(2030, 11, 11),
        ["D12_I5_N"],
    )
    timestamps = [
        datetime(2030, 11, 10, 8, 0, tzinfo=UTC),
        datetime(2030, 11, 10, 8, 5, tzinfo=UTC),
    ]
    targets = pl.DataFrame(
        {
            "panel": ["D12_I5_N", "D12_I5_N"],
            "timestamp": timestamps,
            "station_id": ["S01", "S02"],
            "link_id": ["L01", "L02"],
            "mask_regime": ["R1", "R1"],
        }
    )
    predictions = state(
        release_slice,
        {"D12_I5_N": {"train": targets.lazy()}},
        {
            "checkpoint": str(checkpoint),
            "device": "cpu",
            "sample_steps": 2,
            "timesteps": 4,
            "base_channels": 4,
        },
    )
    predicted = predictions["D12_I5_N"]["train"].collect().sort(list(KEYS["state"]))
    expected = targets.sort(list(KEYS["state"]))
    assert predicted.select(KEYS["state"]).equals(expected)
    assert predicted.select(VALUES["state"]).null_count().sum_horizontal().item() == 0
    assert predicted["speed_kmh"].is_finite().all()
    assert predicted["flow_vph"].is_finite().all()
    assert (predicted["speed_kmh"] >= 0).all()
    assert (predicted["flow_vph"] >= 0).all()
