"""Focused behavior tests for the trafficbench package."""

from __future__ import annotations

from pathlib import Path

import pytest

from trafficbench.contracts import (
    BundleShard,
    RunMetadata,
)
from trafficbench.runner import (
    run_experiment,
)


def test_hosted_ci_package_run_and_combine(
    prepared_benchmark_directory: Path, tmp_path: Path
) -> None:
    from trafficbench.ci import combine, package, unpack

    index = package(prepared_benchmark_directory, tmp_path / "packages")
    shard = index.shards[0]
    unpack(tmp_path / "packages" / shard.asset, tmp_path / "shard", shard.sha256)
    path = run_experiment(
        "example",
        "state",
        tmp_path / "shard",
        runs_directory=tmp_path / "runs",
        panel=shard.panel,
    )
    original = RunMetadata.model_validate_json((path / "run.json").read_text())
    merged = combine(
        tmp_path / "runs", index, "example", "state", tmp_path / "combined"
    )
    assert original.metrics is not None
    assert original.baseline_metrics is not None
    assert merged.metrics is not None
    assert merged.baseline_metrics is not None
    assert merged.metrics["state_score"] == pytest.approx(
        original.metrics["state_score"]
    )
    assert merged.baseline_metrics["state_score"] == pytest.approx(
        original.baseline_metrics["state_score"]
    )
    assert merged.panels == ["D12_I5_N"]
    index.shards.append(
        BundleShard(
            panel="D12_I5_S", asset="missing.tar.gz", sha256="missing", tasks=["state"]
        )
    )
    with pytest.raises(ValueError, match="Missing, duplicate"):
        combine(tmp_path / "runs", index, "example", "state", tmp_path / "bad")


def test_hosted_ci_rejects_changed_download(tmp_path: Path) -> None:
    from trafficbench.ci import unpack

    path = tmp_path / "corrupt.tar.gz"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        unpack(path, tmp_path / "out", "wrong")
