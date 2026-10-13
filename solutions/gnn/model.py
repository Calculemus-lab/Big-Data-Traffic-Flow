"""Starting predictors for Tasks 1, 2, and 4.

Each function starts from its matching baseline and can be developed
independently. Add helper modules as needed.
"""

from __future__ import annotations

from solutions import baseline
from trafficbench.contracts import JsonObject, Panel, ReleasePackageSlice, Split
from trafficbench.table_types import OdmeFrame, QueueFrame, StateFrame

from collections.abc import Mapping

from trafficbench.panelwise import map_panels

from .fit import fit_panel
from .predict import predict_split


def state(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[Panel, dict[Split, StateFrame]],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, StateFrame]]:
    """Return Task 1 speed and flow predictions for the supplied target rows.

    Args:
        release_slice: Released inputs, historical labels, and network data.
        target_templates_by_panel_and_split: Zero-filled Task 1 rows to predict.
        solution_parameters: JSON settings passed to this solution.

    Returns:
        Task 1 predictions grouped by panel and release split.
    """
    return baseline.state(
        release_slice, target_templates_by_panel_and_split, solution_parameters
    )


def queue(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[Panel, dict[Split, QueueFrame]],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, QueueFrame]]:
    """Return Task 2 queue predictions from one trained GCN per panel."""
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        _queue_for_panel,
    )


def _queue_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, QueueFrame],
    solution_parameters: JsonObject,
) -> dict[Split, QueueFrame]:
    fitted = fit_panel(release_slice, panel, solution_parameters)
    threshold = float(solution_parameters.get("threshold", 0.4))
    panel_data = release_slice.panel_slices_by_panel[panel]
    onset_zero_steps = int(solution_parameters.get("onset_zero_steps", 0))
    onset_last_step_queue = bool(
        solution_parameters.get("onset_last_step_queue", False)
    )
    return {
        split: predict_split(
            fitted,
            panel_data.queue_history[split],
            panel_data.queue_window_index[split],
            template,
            threshold,
            onset_zero_steps,
            onset_last_step_queue,
        )
        for split, template in target_templates_by_split.items()
    }


def odme(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[Panel, dict[Split, OdmeFrame]],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, OdmeFrame]]:
    """Return Task 4 path-flow predictions for the supplied target rows.

    Args:
        release_slice: Released inputs, historical labels, and network data.
        target_templates_by_panel_and_split: Zero-filled Task 4 rows to predict.
        solution_parameters: JSON settings passed to this solution.

    Returns:
        Path-flow predictions grouped by panel and release split.
    """
    return baseline.odme(
        release_slice, target_templates_by_panel_and_split, solution_parameters
    )
