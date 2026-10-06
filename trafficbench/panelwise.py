"""Helpers for solutions that predict each panel independently."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from . import table_types
from .contracts import Panel, ReleasePackageSlice, Split

type PredictionFrame = (
    table_types.StateFrame | table_types.QueueFrame | table_types.OdmeFrame
)


def map_panels[PredictionFrameT: PredictionFrame](
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: Mapping[
        Panel, Mapping[Split, PredictionFrameT]
    ],
    predict_panel: Callable[
        [ReleasePackageSlice, Panel, Mapping[Split, PredictionFrameT]],
        Mapping[Split, PredictionFrameT],
    ],
) -> dict[Panel, dict[Split, PredictionFrameT]]:
    """Apply one panel predictor and collect its split results for every panel.

    Args:
        release_slice: Selected release tables, dates, and solution settings.
        target_templates_by_panel_and_split: Requested target tables grouped by
            panel and then by split.
        predict_panel: Function that predicts all requested splits for one
            panel. The function receives the shared slice, panel name, and that
            panel's target templates grouped by release split.

    Returns:
        Predictions grouped by panel and split, matching the runner's solution
        interface.
    """
    return {
        panel: dict(predict_panel(release_slice, panel, split_templates))
        for panel, split_templates in target_templates_by_panel_and_split.items()
    }
