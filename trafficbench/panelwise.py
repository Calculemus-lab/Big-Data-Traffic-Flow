"""Helpers for solutions that predict each panel independently."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from . import table_types
from .contracts import JsonObject, Panel, ReleasePackageSlice, Split

type PredictionFrame = (
    table_types.StateFrame | table_types.QueueFrame | table_types.OdmeFrame
)


def map_panels[PredictionFrameT: PredictionFrame](
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: Mapping[
        Panel, Mapping[Split, PredictionFrameT]
    ],
    solution_parameters: JsonObject,
    predict_panel: Callable[
        [ReleasePackageSlice, Panel, Mapping[Split, PredictionFrameT], JsonObject],
        Mapping[Split, PredictionFrameT],
    ],
) -> dict[Panel, dict[Split, PredictionFrameT]]:
    """Apply one panel predictor and collect its split results for every panel.

    Args:
        release_slice: Selected release tables, dates, and historical labels.
        target_templates_by_panel_and_split: Requested target tables grouped by
            panel and then by split.
        solution_parameters: JSON settings supplied to the solution function.
        predict_panel: Function that predicts all requested splits for one
            panel. The function receives the release slice, panel name, that
            panel's target templates by split, and the solution settings.

    Returns:
        Predictions grouped by panel and split, matching the runner's solution
        interface.
    """
    return {
        panel: dict(
            predict_panel(release_slice, panel, split_templates, solution_parameters)
        )
        for panel, split_templates in target_templates_by_panel_and_split.items()
    }
