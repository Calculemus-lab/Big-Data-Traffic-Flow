"""Prior-aware path-flow estimation for Task 4."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from trafficbench import table_types
from trafficbench.contracts import (
    JsonObject,
    Panel,
    ReleasePackageSlice,
    Split,
)
from trafficbench.metrics import link_path_incidence_matrix
from trafficbench.panelwise import map_panels

MAXIMUM_OPTIMIZER_ITERATIONS = 8_000


class OdmeParameters(BaseModel):
    """Tunable regularization for the constrained ODME estimator.

    Attributes:
        regularization: Weight on squared distance from the released weak prior.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    regularization: float = Field(default=0.05, gt=0)


def odme(
    release_slice: ReleasePackageSlice,
    target_templates_by_panel_and_split: dict[
        Panel, dict[Split, table_types.OdmeFrame]
    ],
    solution_parameters: JsonObject,
) -> dict[Panel, dict[Split, table_types.OdmeFrame]]:
    """Estimate nonnegative path flows for each requested scenario.

    The optimizer minimizes squared differences between loaded link flows and
    released counts, plus a regularized squared distance from the weak prior.
    It enforces nonnegative path flows and solves each split independently.

    Args:
        release_slice: Candidate paths, path-link incidence, observed counts,
            and the split-specific weak priors.
        target_templates_by_panel_and_split: Requested Task 4 path rows,
            grouped by panel and split.
        solution_parameters: Optional estimator settings validated by
            :class:`OdmeParameters`.

    Returns:
        One path-flow prediction table per requested panel and split.
    """
    parameters = OdmeParameters.model_validate(solution_parameters)
    return map_panels(
        release_slice,
        target_templates_by_panel_and_split,
        solution_parameters,
        lambda selected_slice, panel, targets, _parameters: _odme_for_panel(
            selected_slice, panel, targets, parameters
        ),
    )


def _odme_for_panel(
    release_slice: ReleasePackageSlice,
    panel: Panel,
    target_templates_by_split: Mapping[Split, table_types.OdmeFrame],
    parameters: OdmeParameters,
) -> dict[Split, table_types.OdmeFrame]:
    """Estimate flows for each requested split of one corridor panel.

    Args:
        release_slice: Network, counts, and priors for the selected panels.
        panel: Corridor panel being estimated.
        target_templates_by_split: Requested path rows grouped by split.
        parameters: Validated prior-regularization setting.

    Returns:
        Estimated path-flow tables keyed by split.
    """
    panel_data = release_slice.panel_slices_by_panel[panel]
    predictions: dict[Split, table_types.OdmeFrame] = {}
    for split, target_template in target_templates_by_split.items():
        target_table = target_template.collect()
        link_counts = panel_data.link_counts[split].collect()
        weak_prior = panel_data.weak_prior[split].collect()

        # Align the sparse incidence matrix, counts, and prior to the exact
        # ordering of the requested path rows before optimization.
        incidence = link_path_incidence_matrix(
            panel_data.network, target_table, link_counts
        )
        prior_frame = target_table.select("path_id").join(
            weak_prior.select("path_id", "path_flow"),
            on="path_id",
            how="left",
            maintain_order="left",
        )
        if (
            prior_frame.height != target_table.height
            or prior_frame["path_flow"].null_count()
        ):
            raise ValueError(
                f"{panel}/{split} weak prior must have one row for every target path"
            )
        prior = prior_frame["path_flow"].to_numpy()
        counts = link_counts["count"].to_numpy()
        path_flow = _estimate_path_flow(
            incidence,
            counts,
            prior,
            parameters,
        )
        predictions[split] = target_table.with_columns(
            pl.Series("path_flow", path_flow)
        ).lazy()
    return predictions


def _estimate_path_flow(
    incidence: csr_matrix,
    counts: NDArray[np.float64],
    prior: NDArray[np.float64],
    parameters: OdmeParameters,
) -> NDArray[np.float64]:
    """Fit nonnegative path flows to counts with a penalty toward the prior.

    Args:
        incidence: Sparse link-by-path incidence matrix in submission order.
        counts: Observed counts in the same row order as ``incidence``.
        prior: Weak-prior path flows in the same column order as ``incidence``.
        parameters: Validated prior-regularization setting.

    Returns:
        Nonnegative estimated path flows in target-template order.

    Raises:
        ValueError: If counts or prior flows are negative or non-finite.
        RuntimeError: If the constrained optimization fails.
    """
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.isfinite(prior).all()
        or (prior < 0).any()
    ):
        raise ValueError(
            "ODME counts and weak-prior flows must be finite and nonnegative"
        )

    regularization = parameters.regularization

    # A sparse gradient keeps the solve practical when a corridor has thousands
    # of candidate paths but only a few hundred measured links.
    def objective(
        path_flow: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        link_residual = incidence @ path_flow - counts
        prior_change = path_flow - prior
        value = 0.5 * (
            link_residual @ link_residual
            + regularization * (prior_change @ prior_change)
        )
        gradient = (
            np.asarray(incidence.T @ link_residual) + regularization * prior_change
        )
        return float(value), gradient

    result = minimize(
        objective,
        prior,
        jac=True,
        bounds=[(0.0, None)] * prior.size,
        method="L-BFGS-B",
        options={
            "maxiter": MAXIMUM_OPTIMIZER_ITERATIONS,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "maxcor": 30,
        },
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Task 4 ODME optimization failed: {result.message}")
    return np.maximum(result.x, 0.0)
