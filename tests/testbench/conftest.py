"""Shared fixtures for trafficbench tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trafficbench.fixture import make_fixture
from trafficbench.prepare import prepare


@pytest.fixture(scope="session")
def prepared_benchmark_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepare one shared benchmark fixture from synthetic release inputs.

    Args:
        tmp_path_factory: Pytest factory that creates an isolated session directory.

    Returns:
        Directory containing the generated benchmark manifest and tables.
    """
    root = tmp_path_factory.mktemp("benchmark")
    make_fixture(root / "release")
    prepare(
        root / "release",
        root / "prepared_benchmark_directory",
        history_start_date=date(2030, 11, 3),
        history_end_date=date(2030, 11, 20),
        prediction_end_date=date(2030, 11, 28),
    )
    return root / "prepared_benchmark_directory"
