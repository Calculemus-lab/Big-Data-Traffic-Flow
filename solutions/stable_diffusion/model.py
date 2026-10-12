"""Task 1 speed and flow reconstruction by denoising a milepost-time field.

Each corridor day is a grid: detector rows ordered by milepost, and 288
five-minute columns. A conditional diffusion model fills cells that the Task 1
mask removed. Queue and path-flow predictions stay on the baseline.

Try, after training::

    uv run bench run stable_diffusion --task state --target-range validation \\
        --params '{"device": "cuda", "sample_steps": 20}'

Train from the unmasked release with ``python -m solutions.stable_diffusion.train``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from tqdm import tqdm

from solutions import baseline
from solutions.stable_diffusion.diffusion import (
    ConditionalUNet,
    as_batch,
    cosine_alpha_bar,
    load_checkpoint,
    resolve_device,
    sample_field,
)
from solutions.stable_diffusion.field import (
    DayField,
    PanelFields,
    denormalize,
    model_sample,
    panel_fields,
    rows_from_completed_field,
)
from trafficbench import table_types
from trafficbench.contracts import (
    KEYS,
    VALUES,
    JsonObject,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.panelwise import map_panels

CHECKPOINT = "results/stable_diffusion/checkpoint.pt"


class DiffusionParameters(BaseModel):
    """Settings for training and sampling the field diffusion model.

    Attributes:
        checkpoint: Path of the ``.pt`` weights. A sibling ``.json`` stores the
            schedule length and channel width.
        timesteps: Cosine schedule length used when training from scratch.
        sample_steps: DDIM steps used to fill one day.
        epochs: Optimizer passes when ``train_if_missing`` starts training.
        batch_size: Days per training step.
        learning_rate: AdamW step size.
        base_channels: U-Net width used when training. Loading a checkpoint
            uses the width stored beside the weights.
        device: ``cuda`` when a GPU is available, otherwise CPU.
        train_if_missing: Train on the visible history when the checkpoint is
            absent. Otherwise missing weights fall back to the weekly baseline.
        max_days: Optional cap on training days per panel.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    checkpoint: str = CHECKPOINT
    timesteps: int = Field(default=200, ge=2)
    sample_steps: int = Field(default=20, ge=1)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    base_channels: int = Field(default=32, ge=4)
    device: str = "cuda"
    train_if_missing: bool = False
    max_days: int | None = Field(default=None, ge=1)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Fill Task 1 speed and flow by denoising each day's milepost-time field.

    Args:
        release_slice: Network tables, masked observations, and historical labels.
        target_templates_by_panel_and_split: Zero-filled Task 1 target rows,
            grouped by panel and release split.
        solution_parameters: JSON settings validated as :class:`DiffusionParameters`.

    Returns:
        One official-shaped prediction table for each panel and split.
    """
    parameters = DiffusionParameters.model_validate(solution_parameters)
    model_bundle = _load_or_train(release_slice, parameters)

    def predict_panel(
        release: ReleasePackageSlice,
        panel: Panel,
        templates: Mapping[Split, table_types.StateFrame],
        _solution_parameters: JsonObject,
    ) -> dict[Split, table_types.StateFrame]:
        """Predict one panel with the checkpoint loaded for this call.

        Args:
            release: Slice shared by every panel.
            panel: Panel to predict.
            templates: Zero-filled Task 1 rows for that panel.
            _solution_parameters: Settings already validated above.

        Returns:
            Predictions for each requested split.
        """
        return _state_for_panel(release, panel, templates, parameters, model_bundle)

    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        predict_panel,
    )


def queue(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.QueueFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.QueueFrame]]:
    """Return the baseline queue forecast.

    Args:
        release_slice: Released inputs and historical labels.
        target_templates_by_panel_and_split: Zero-filled Task 2 rows.
        solution_parameters: JSON settings forwarded to the baseline.

    Returns:
        Queue predictions grouped by panel and split.
    """
    return baseline.queue(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
    )


def odme(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.OdmeFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.OdmeFrame]]:
    """Return the baseline path-flow estimate.

    Args:
        release_slice: Released inputs and network tables.
        target_templates_by_panel_and_split: Zero-filled Task 4 rows.
        solution_parameters: JSON settings forwarded to the baseline.

    Returns:
        Path-flow predictions grouped by panel and split.
    """
    return baseline.odme(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
    )


def _state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    parameters: DiffusionParameters,
    model_bundle: tuple[ConditionalUNet, int] | None,
) -> dict[Split, table_types.StateFrame]:
    """Predict one panel, using the weekly baseline where diffusion has no row.

    Args:
        release_slice: Selected release tables and historical labels.
        panel: Panel whose measurements and targets are used.
        target_templates_by_split: Zero-filled Task 1 rows by split.
        parameters: Validated diffusion settings.
        model_bundle: Loaded denoiser and schedule length, or ``None`` to keep
            the weekly baseline.

    Returns:
        State predictions grouped by split.
    """
    fallback = baseline.state(
        release_slice,
        {panel: dict(target_templates_by_split)},
        {},
    )[panel]
    if model_bundle is None:
        return fallback
    model, timesteps = model_bundle
    fields = _fields_for_panel(release_slice, panel)
    if fields is None:
        return fallback
    completed_by_day = _complete_target_days(
        fields,
        target_templates_by_split,
        model,
        timesteps,
        parameters,
        release_slice.seed,
    )
    return {
        split: _blend_split(fallback[split], completed_by_day)
        for split, target_template in target_templates_by_split.items()
    }


def _load_or_train(
    release_slice: ReleasePackageSlice,
    parameters: DiffusionParameters,
) -> tuple[ConditionalUNet, int] | None:
    """Load the checkpoint, train it, or report that the baseline should be used.

    Args:
        release_slice: Visible history used when training is requested.
        parameters: Checkpoint path and training switch.

    Returns:
        Denoiser and schedule length, or ``None`` when no weights are available.
    """
    checkpoint = Path(parameters.checkpoint)
    device = resolve_device(parameters.device)
    if checkpoint.is_file():
        return load_checkpoint(checkpoint, device)
    if not parameters.train_if_missing:
        return None
    panels = [
        fields
        for panel in release_slice.panel_slices_by_panel
        if (fields := _fields_for_panel(release_slice, panel)) is not None
    ]
    from solutions.stable_diffusion.train import fit_panels

    fit_panels(
        panels,
        timesteps=parameters.timesteps,
        epochs=parameters.epochs,
        batch_size=parameters.batch_size,
        learning_rate=parameters.learning_rate,
        base_channels=parameters.base_channels,
        device_name=parameters.device,
        seed=release_slice.seed,
        output_path=checkpoint,
        max_days=parameters.max_days,
    )
    return load_checkpoint(checkpoint, device)


def _fields_for_panel(
    release_slice: ReleasePackageSlice, panel: Panel
) -> PanelFields | None:
    """Build daily grids from the measurements visible to the solution.

    Args:
        release_slice: Slice whose historical answers have been restored.
        panel: Panel to encode.

    Returns:
        Panel grids, or ``None`` when the panel has no mainline rows.
    """
    observations = baseline.state_observations(release_slice, panel)
    links = release_slice.panel_slices_by_panel[panel].network.links.collect()
    return panel_fields(observations, links)


def _complete_target_days(
    fields: PanelFields,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    model: ConditionalUNet,
    timesteps: int,
    parameters: DiffusionParameters,
    seed: int,
) -> dict[date, pl.DataFrame]:
    """Sample a completed field for each day that contains target rows.

    Args:
        fields: Detector axis and observed daily grids.
        target_templates_by_split: Target rows whose dates select the days.
        model: Loaded denoiser.
        timesteps: Schedule length stored with the checkpoint.
        parameters: Sampling device and step count.
        seed: Seed for the DDIM noise.

    Returns:
        Completed speed and flow rows keyed by calendar day.
    """
    targets = pl.concat(
        [template.collect() for template in target_templates_by_split.values()]
    ).with_columns(pl.col("timestamp").dt.date().alias("day"))
    days_by_date = {field.day: field for field in fields.days}
    device = resolve_device(parameters.device)
    alpha_bar = cosine_alpha_bar(timesteps, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    completed: dict[date, pl.DataFrame] = {}
    for day_value, day_targets in tqdm(
        list(targets.group_by("day")), desc="sample days"
    ):
        day = day_value[0]
        if not isinstance(day, date):
            continue
        field = days_by_date.get(day)
        if field is None:
            continue
        speed, flow = _sample_day(
            field, fields, model, alpha_bar, parameters, generator
        )
        filled = rows_from_completed_field(day, speed, flow, fields, day_targets)
        if "speed_kmh" in filled.columns and not filled.is_empty():
            completed[day] = filled
    return completed


def _sample_day(
    field: DayField,
    fields: PanelFields,
    model: ConditionalUNet,
    alpha_bar: Tensor,
    parameters: DiffusionParameters,
    generator: torch.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Denoise one daily grid and return physical speed and flow.

    Args:
        field: Observed day to complete.
        fields: Panel scales and priors.
        model: Denoiser.
        alpha_bar: Noise schedule.
        parameters: Step count.
        generator: Noise generator.

    Returns:
        Speed and flow grids in km/h and vehicles per hour.
    """
    sample = model_sample(field, fields)
    device = alpha_bar.device
    with torch.no_grad():
        units = sample_field(
            model,
            as_batch(sample.observed_values, device),
            as_batch(sample.mask, device),
            as_batch(sample.prior, device),
            alpha_bar,
            parameters.sample_steps,
            generator,
        )
    return denormalize(units[0].cpu().numpy(), fields)


def _blend_split(
    fallback: table_types.StateFrame,
    completed_by_day: dict[date, pl.DataFrame],
) -> table_types.StateFrame:
    """Prefer diffusion values and keep the baseline where a cell was not sampled.

    Args:
        fallback: Weekly-profile predictions for this split.
        completed_by_day: Diffusion rows keyed by day.

    Returns:
        Lazy prediction table with the official state columns.
    """
    identity = ["timestamp", "station_id", "link_id"]
    baseline_frame = fallback.collect()
    if not completed_by_day:
        return baseline_frame.lazy()
    completed = (
        pl.concat(list(completed_by_day.values()), how="diagonal")
        .select(identity + list(VALUES["state"]))
        .unique(identity, keep="first")
    )
    return (
        baseline_frame.join(completed, on=identity, how="left", suffix="_diffusion")
        .with_columns(
            pl.coalesce(f"{column}_diffusion", column).alias(column)
            for column in VALUES["state"]
        )
        .select(list(KEYS["state"]) + list(VALUES["state"]))
        .lazy()
    )
