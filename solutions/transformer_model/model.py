"""Task 1 speed and flow reconstruction from a normalized density field.

Each corridor day is a grid: detector rows ordered by milepost, and 288
five-minute columns. Cell values are density divided by jam density. A
transformer fills cells that the Task 1 mask removed, and the triangular
fundamental diagram turns that density back into speed and flow. Queue and
path-flow predictions stay on the baseline.

Train, then run::

    uv run --locked python -m solutions.transformer_model.train \\
        --release-root kaggle_public --device cuda
    uv run --locked bench run transformer_model --task state \\
        --target-range validation --params '{"device": "cuda"}'
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from solutions import baseline
from solutions.transformer_model.field import (
    PanelFields,
    panel_fields,
    speed_and_flow_from_density,
    target_speed_and_flow,
)
from solutions.transformer_model.transformer import (
    DensityTransformer,
    load_checkpoint,
    predict_normalized_density,
    resolve_device,
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

CHECKPOINT = "results/transformer_model/checkpoint.pt"


class TransformerParameters(BaseModel):
    """Settings for loading the density transformer.

    Attributes:
        checkpoint: Path of the ``.pt`` weights. A sibling ``.json`` stores
            the architecture.
        device: ``cuda`` when a GPU is available, otherwise CPU.
        batch_size: Days completed in one forward pass.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    checkpoint: str = CHECKPOINT
    device: str = "cuda"
    batch_size: int = Field(default=4, ge=1)


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.StateFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.StateFrame]]:
    """Fill Task 1 speed and flow by completing each day's density field.

    Args:
        release_slice: Network tables, masked observations, and historical labels.
        target_templates_by_panel_and_split: Zero-filled Task 1 target rows,
            grouped by panel and release split.
        solution_parameters: JSON settings validated as
            :class:`TransformerParameters`.

    Returns:
        One official-shaped prediction table for each panel and split.
    """
    parameters = TransformerParameters.model_validate(solution_parameters)
    model = _load_model(parameters)

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
        return _state_for_panel(release, panel, templates, parameters, model)

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
        solution_parameters: JSON settings accepted by this solution. The
            baseline queue forecast does not read them.

    Returns:
        Queue predictions grouped by panel and split.
    """
    del solution_parameters
    return baseline.queue(release_slice, target_templates_by_panel_and_split, {})


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
        solution_parameters: JSON settings accepted by this solution. The
            baseline path-flow estimate keeps its own default regularization.

    Returns:
        Path-flow predictions grouped by panel and split.
    """
    del solution_parameters
    return baseline.odme(release_slice, target_templates_by_panel_and_split, {})


def _load_model(parameters: TransformerParameters) -> DensityTransformer | None:
    """Load the checkpoint, or return ``None`` when the file is absent.

    Args:
        parameters: Checkpoint path and device.

    Returns:
        The density transformer, or ``None`` so the caller can use the baseline.
    """
    checkpoint = Path(parameters.checkpoint)
    if not checkpoint.is_file():
        return None
    return load_checkpoint(checkpoint, resolve_device(parameters.device))


def _state_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.StateFrame],
    parameters: TransformerParameters,
    model: DensityTransformer | None,
) -> dict[Split, table_types.StateFrame]:
    """Predict one panel, using the weekly baseline where the field has no row.

    Args:
        release_slice: Selected release tables and historical labels.
        panel: Panel whose measurements and targets are used.
        target_templates_by_split: Zero-filled Task 1 rows by split.
        parameters: Device and batch size for the forward pass.
        model: Loaded transformer, or ``None`` when the checkpoint is missing.

    Returns:
        State predictions grouped by split.
    """
    baseline_predictions = baseline.state(
        release_slice,
        {panel: dict(target_templates_by_split)},
        {},
    )[panel]
    if model is None:
        return dict(baseline_predictions)
    observations = baseline.state_observations(release_slice, panel)
    fd_parameters = release_slice.panel_slices_by_panel[
        panel
    ].network.fd_parameters.collect()
    fields = panel_fields(observations, fd_parameters)
    if fields is None:
        return dict(baseline_predictions)
    completed = _completed_speed_and_flow(fields, model, parameters, panel)
    predictions: dict[Split, table_types.StateFrame] = {}
    for split, target_template in target_templates_by_split.items():
        base = baseline_predictions[split].collect()
        model_rows = target_speed_and_flow(fields, completed, target_template.collect())
        predictions[split] = (
            base.join(
                model_rows,
                on=["timestamp", "station_id", "link_id"],
                how="left",
                suffix="_model",
            )
            .with_columns(
                pl.coalesce(pl.col(f"{column}_model"), pl.col(column)).alias(column)
                for column in VALUES["state"]
            )
            .select(KEYS["state"] + VALUES["state"])
            .lazy()
        )
    return predictions


def _completed_speed_and_flow(
    fields: PanelFields,
    model: DensityTransformer,
    parameters: TransformerParameters,
    panel: Panel,
) -> dict[date, tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """Run the transformer on every day and invert density to speed and flow.

    Args:
        fields: Observed density grids for one panel.
        model: Loaded density transformer.
        parameters: Device and batch size.
        panel: Panel name shown on the progress bar.

    Returns:
        Speed and flow arrays keyed by calendar day.
    """
    device = resolve_device(parameters.device)
    model = model.to(device)
    model.eval()
    completed: dict[date, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    days = fields.days
    batch_starts = range(0, len(days), parameters.batch_size)
    for start in tqdm(batch_starts, desc=f"fill {panel}", unit="batch"):
        batch = days[start : start + parameters.batch_size]
        density = np.stack([day.density for day in batch])
        observed = np.stack([day.observed for day in batch])
        filled = predict_normalized_density(model, density, observed, device)
        for index, day in enumerate(batch):
            speed, flow = speed_and_flow_from_density(filled[index], fields.axis)
            completed[day.day] = (speed, flow)
    return completed
