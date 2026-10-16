"""Train the density transformer on unmasked train days.

Weights are written under ``results/`` so the checkpoint can stay with this
experiment branch. Run::

    uv run --locked python -m solutions.transformer_model.train \\
        --release-root kaggle_public --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from solutions.transformer_model.field import (
    OBSERVATION_COLUMNS,
    SLOTS_PER_DAY,
    PanelFields,
    panel_fields,
)
from solutions.transformer_model.transformer import (
    DensityTransformer,
    TransformerConfig,
    masked_density_loss,
    resolve_device,
    save_checkpoint,
)
from trafficbench.data import Release

DEFAULT_CHECKPOINT = Path("results/transformer_model/checkpoint.pt")


def fit_panels(
    panels: list[PanelFields],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    max_detectors: int,
    device_name: str,
    seed: int,
    output_path: Path,
    max_days: int | None = None,
) -> Path:
    """Fit one density transformer on daily fields and save its checkpoint.

    Each step samples one panel, then a batch of its days. Training hides a
    random 20, 30, or 50 percent of measured cells.

    Args:
        panels: Grids to learn. Empty panels are skipped.
        epochs: Passes over the available days.
        batch_size: Days per optimizer step.
        learning_rate: AdamW learning rate.
        d_model: Token width.
        n_heads: Attention heads.
        n_layers: Temporal and spatial attention pairs.
        max_detectors: Position table size. Must cover every panel axis.
        device_name: ``cuda`` or ``cpu``. CUDA falls back to CPU when absent.
        seed: Seed for initialization, shuffling, and the training mask.
        output_path: ``.pt`` path for the weights.
        max_days: Optional cap on days used from each panel.

    Returns:
        Path of the written checkpoint.

    Raises:
        ValueError: If no day has a measurement, or a panel has more detectors
            than ``max_detectors``.
    """
    trainable = [
        panel for panel in panels if any(day.observed.any() for day in panel.days)
    ]
    if not trainable:
        raise ValueError("Transformer training needs at least one observed day")
    widest = max(len(panel.axis) for panel in trainable)
    if widest > max_detectors:
        raise ValueError(
            f"A panel has {widest} detectors, above max_detectors={max_detectors}"
        )
    device = resolve_device(device_name)
    torch.manual_seed(seed)
    config = TransformerConfig(d_model, n_heads, n_layers, max_detectors, SLOTS_PER_DAY)
    model = DensityTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    day_indexes = [_limited_days(panel, max_days) for panel in trainable]
    steps = epochs * sum(
        (len(indexes) + batch_size - 1) // batch_size for indexes in day_indexes
    )
    progress = tqdm(total=steps, desc="train density transformer")
    model.train()
    for epoch in range(epochs):
        progress.set_description(f"train epoch {epoch + 1}/{epochs}")
        panel_order = generator.permutation(len(trainable))
        for panel_index in panel_order:
            panel = trainable[int(panel_index)]
            indexes = day_indexes[int(panel_index)].copy()
            generator.shuffle(indexes)
            for start in range(0, len(indexes), batch_size):
                batch = indexes[start : start + batch_size]
                loss = _batch_loss(model, panel, batch, generator, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
                progress.update(1)
    progress.close()
    save_checkpoint(output_path, model)
    return output_path


def fields_from_unmasked_release(release_root: Path, panel: str) -> PanelFields | None:
    """Load one panel's unmasked train days and build density grids.

    Args:
        release_root: Extracted release directory.
        panel: Corridor directory name.

    Returns:
        Density grids, or ``None`` when the panel has no unmasked train files
        or no detector with a usable diagram.
    """
    release = Release(release_root)
    prefix = f"corridors/{panel}/train/mainline_states/"
    files = [
        name
        for name in release.file_names
        if name.startswith(prefix) and name.endswith(".parquet")
    ]
    if not files:
        return None
    observations = pl.concat(
        [
            release.scan_parquet(name).select(OBSERVATION_COLUMNS).collect()
            for name in tqdm(files, desc=f"load {panel}", unit="day", leave=False)
        ],
        how="diagonal_relaxed",
    )
    fd_parameters = release.read_csv(f"corridors/{panel}/network/fd_parameters.csv")
    return panel_fields(observations, fd_parameters)


def main() -> None:
    """Train from unmasked train parquet files and write ``results/`` weights."""
    parser = argparse.ArgumentParser(
        description="Train the Task 1 density transformer."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--panel", action="append", dest="panels")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-detectors", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    panel_names = args.panels or _corridor_names(args.release_root)
    panels: list[PanelFields] = []
    for name in tqdm(panel_names, desc="load train panels"):
        fields = fields_from_unmasked_release(args.release_root, name)
        if fields is not None:
            panels.append(fields)
    fit_panels(
        panels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_detectors=args.max_detectors,
        device_name=args.device,
        seed=args.seed,
        output_path=args.output,
        max_days=args.max_days,
    )


def _corridor_names(release_root: Path) -> list[str]:
    """List corridor directory names under the release.

    Args:
        release_root: Extracted release directory.

    Returns:
        Sorted corridor names that contain a ``network`` directory.
    """
    corridors = release_root / "corridors"
    return sorted(
        path.name
        for path in corridors.iterdir()
        if path.is_dir() and (path / "network").is_dir()
    )


def _limited_days(panel: PanelFields, max_days: int | None) -> list[int]:
    """Choose day indexes that still contain at least one measurement.

    Args:
        panel: Panel whose days are candidates.
        max_days: Optional maximum number of days to keep.

    Returns:
        Indexes into ``panel.days``.
    """
    observed = [
        index for index, day in enumerate(panel.days) if bool(day.observed.any())
    ]
    if max_days is None:
        return observed
    return observed[:max_days]


def _batch_loss(
    model: DensityTransformer,
    panel: PanelFields,
    day_indexes: list[int],
    generator: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Stack one batch of days and return the masked density loss.

    Args:
        model: Transformer being trained.
        panel: Panel that owns the selected days.
        day_indexes: Indexes into ``panel.days``.
        generator: Generator for the training mask.
        device: Device that stores the batch.

    Returns:
        Scalar loss for the batch.
    """
    density = np.stack([panel.days[index].density for index in day_indexes])
    observed = np.stack([panel.days[index].observed for index in day_indexes])
    clean = torch.as_tensor(density, device=device)
    mask = torch.as_tensor(observed, device=device)
    return masked_density_loss(model, clean, mask, generator)


if __name__ == "__main__":
    main()
