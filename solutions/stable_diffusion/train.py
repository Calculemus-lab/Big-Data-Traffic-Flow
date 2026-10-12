"""Train the spatiotemporal diffusion model on unmasked train days.

Weights are written under ``results/`` so the checkpoint can stay with this
experiment branch. The same fitting routine trains from release files or from
fields already built by the Task 1 predictor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from solutions.stable_diffusion.diffusion import (
    ConditionalUNet,
    cosine_alpha_bar,
    diffusion_loss,
    resolve_device,
    save_checkpoint,
)
from solutions.stable_diffusion.field import (
    PanelFields,
    model_sample,
    panel_fields,
)

DEFAULT_CHECKPOINT = Path("results/stable_diffusion/checkpoint.pt")
OBSERVATION_COLUMNS = (
    "timestamp",
    "station_id",
    "link_id",
    "milepost",
    "speed_kmh",
    "flow_vph",
    "pct_observed",
)


def fit_panels(
    panels: list[PanelFields],
    *,
    timesteps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    base_channels: int,
    device_name: str,
    seed: int,
    output_path: Path,
    max_days: int | None = None,
) -> Path:
    """Fit one denoiser on daily fields and save its checkpoint.

    Each step samples one panel, then a batch of its days. Training hides a
    random 20, 30, or 50 percent of measured cells so the model learns the
    Task 1 mask rates.

    Args:
        panels: Grids to learn. Empty panels are skipped.
        timesteps: Length of the cosine noise schedule.
        epochs: Passes over the available days.
        batch_size: Days per optimizer step.
        learning_rate: AdamW learning rate.
        base_channels: U-Net width.
        device_name: ``cuda`` or ``cpu``. CUDA falls back to CPU when absent.
        seed: Seed for initialization, shuffling, and the diffusion noise.
        output_path: ``.pt`` path for the weights.
        max_days: Optional cap on days used from each panel.

    Returns:
        Path of the written checkpoint.
    """
    trainable = [panel for panel in panels if panel.days]
    if not trainable:
        raise ValueError("Diffusion training needs at least one observed day")
    device = resolve_device(device_name)
    torch.manual_seed(seed)
    model = ConditionalUNet(base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    alpha_bar = cosine_alpha_bar(timesteps, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    prepared = [(panel, _limited_days(panel, max_days)) for panel in trainable]
    steps_per_epoch = sum(
        (len(day_indexes) + batch_size - 1) // batch_size
        for _panel, day_indexes in prepared
        if day_indexes
    )
    with tqdm(total=epochs * steps_per_epoch, desc="train") as progress:
        for epoch in range(epochs):
            order = np.random.default_rng(seed + epoch).permutation(len(prepared))
            for panel_index in order:
                panel, day_indexes = prepared[int(panel_index)]
                for start in range(0, len(day_indexes), batch_size):
                    batch_days = day_indexes[start : start + batch_size]
                    loss = _batch_loss(model, panel, batch_days, alpha_bar, generator)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    progress.set_postfix(epoch=epoch + 1, loss=f"{loss.item():.4f}")
                    progress.update(1)
    save_checkpoint(
        output_path,
        model,
        timesteps=timesteps,
        base_channels=base_channels,
    )
    return output_path


def fields_from_unmasked_release(release_root: Path, panel: str) -> PanelFields | None:
    """Read one panel's unmasked train mainline files into daily grids.

    Args:
        release_root: Extracted release directory that contains ``corridors``.
        panel: Corridor directory name.

    Returns:
        Panel grids, or ``None`` when that corridor has no unmasked rows.
    """
    mainline = release_root / "corridors" / panel / "train" / "mainline_states"
    files = sorted(mainline.rglob("*.parquet"))
    if not files:
        return None
    observations = pl.concat(
        [
            pl.read_parquet(path, columns=list(OBSERVATION_COLUMNS))
            for path in tqdm(files, desc=f"load {panel}")
        ],
        how="diagonal",
    )
    links = pl.read_csv(release_root / "corridors" / panel / "network" / "links.csv")
    return panel_fields(observations, links)


def main() -> None:
    """Train from unmasked train parquet files and write ``results/`` weights."""
    parser = argparse.ArgumentParser(
        description="Train the Task 1 field diffusion model."
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--panel", action="append", dest="panels")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    panel_names = args.panels or _corridor_names(args.release_root)
    panels = [
        fields
        for name in panel_names
        if (fields := fields_from_unmasked_release(args.release_root, name)) is not None
    ]
    fit_panels(
        panels,
        timesteps=args.timesteps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
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
    model: ConditionalUNet,
    panel: PanelFields,
    day_indexes: list[int],
    alpha_bar: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Stack one batch of days and return the diffusion loss.

    Args:
        model: Denoiser being trained.
        panel: Panel that owns the selected days.
        day_indexes: Indexes into ``panel.days``.
        alpha_bar: Cosine noise schedule.
        generator: Generator for mask and noise draws.

    Returns:
        Scalar loss for the batch.
    """
    samples = [model_sample(panel.days[index], panel) for index in day_indexes]
    device = alpha_bar.device
    clean = torch.stack([torch.as_tensor(sample.clean) for sample in samples]).to(
        device
    )
    observed = torch.stack(
        [torch.as_tensor(sample.observed_values) for sample in samples]
    ).to(device)
    mask = torch.stack([torch.as_tensor(sample.mask) for sample in samples]).to(device)
    prior = torch.stack([torch.as_tensor(sample.prior) for sample in samples]).to(
        device
    )
    return diffusion_loss(model, clean, observed, mask, prior, alpha_bar, generator)


if __name__ == "__main__":
    main()
