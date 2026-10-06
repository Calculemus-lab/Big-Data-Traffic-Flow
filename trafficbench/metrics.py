"""Score local cases using answer values withheld from solutions.

State and queue functions compare predictions with held-out values. Task 4
origin-destination matrix estimation (ODME) metrics compare predicted path
flows with link counts and, for synthetic cases, known path flows. The
``aggregate`` function gives panels and freeway families equal weight.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix

from .contracts import KEYS, MASK_REGIMES, VALUES, MetricRow, NetworkTables


def link_path_incidence_matrix(
    network: NetworkTables,
    requested_path_table: pl.DataFrame,
    observed_link_count_table: pl.DataFrame,
) -> csr_matrix:
    """Build the link-count-by-path incidence matrix for one Task 4 case.

    Each row follows the order of ``observed_link_count_table``. Each column
    follows the order of ``requested_path_table``. A value is one when a
    candidate path uses a counted link and zero otherwise. Reject duplicate
    requested path IDs, incidence rows for unrequested paths, and counted links
    absent from the panel's path-to-link table.

    Args:
        network: Candidate paths and the links each path uses.
        requested_path_table: Requested paths in output and matrix-column order.
        observed_link_count_table: Measured links in matrix-row order.

    Returns:
        Sparse binary matrix mapping candidate path flows to link counts.

    Raises:
        ValueError: If requested paths or counted links do not match the
            panel's path-to-link table.
    """
    # The row and column positions must follow the inputs used by the scorer,
    # regardless of how the network incidence CSV happened to be ordered.
    path_ids = requested_path_table["path_id"].to_list()
    if len(set(path_ids)) != len(path_ids):
        raise ValueError("ODME targets contain duplicate path IDs")
    path_positions = {path_id: index for index, path_id in enumerate(path_ids)}
    link_positions = {
        link_id: index
        for index, link_id in enumerate(observed_link_count_table["link_id"])
    }
    incidence = network.path_link_incidence.collect()
    if set(incidence["path_id"]) - set(path_ids) or set(
        observed_link_count_table["link_id"]
    ) - set(incidence["link_id"]):
        raise ValueError("Incidence, counts and path IDs do not match")

    # A path may list the same link more than once in the source. One measured
    # link contributes once to that path's count prediction.
    path_link_pairs = (
        incidence.filter(pl.col("link_id").is_in(list(link_positions)))
        .select("link_id", "path_id")
        .unique()
    )
    incidence_values: NDArray[np.float64] = np.ones(path_link_pairs.height)
    row_indices: NDArray[np.int64] = np.asarray(
        [link_positions[link_id] for link_id in path_link_pairs["link_id"]],
        dtype=np.int64,
    )
    column_indices: NDArray[np.int64] = np.asarray(
        [path_positions[path_id] for path_id in path_link_pairs["path_id"]],
        dtype=np.int64,
    )
    return sparse.csr_matrix(
        (incidence_values, (row_indices, column_indices)),
        shape=(observed_link_count_table.height, len(path_ids)),
    )


def state_metrics(
    state_prediction_table: pl.DataFrame,
    state_answer_table: pl.DataFrame,
    network: NetworkTables,
) -> list[MetricRow]:
    """Calculate speed and per-lane flow errors for each mask regime.

    ``state_prediction_table`` and ``state_answer_table`` contain the same
    Task 1 target rows. Divide each link's flow error by its lane count before
    computing root mean square error (RMSE). Return one row per regime with the local weighted score
    and target count.

    Args:
        state_prediction_table: Predicted speed and flow for Task 1 target rows.
        state_answer_table: Known speed and flow for those same target rows.
        network: Panel link attributes containing lane counts.

    Returns:
        One score and sample count for each represented mask regime.
    """
    # Lane counts come from the same network panel as the state targets.
    joined_state_table = state_answer_table.join(
        state_prediction_table.select(KEYS["state"] + VALUES["state"]),
        on=KEYS["state"],
        how="left",
        suffix="_pred",
        validate="1:1",
    ).join(
        network.fd_parameters.select("link_id", "lanes").collect(),
        on="link_id",
        how="left",
    )
    lane_counts = joined_state_table["lanes"]
    if lane_counts.null_count() or (lane_counts <= 0).any():
        raise ValueError("Missing or invalid lane counts")
    joined_state_table = joined_state_table.with_columns(
        (pl.col("speed_kmh_pred") - pl.col("speed_kmh")).pow(2).alias("speed_sq"),
        ((pl.col("flow_vph_pred") - pl.col("flow_vph")) / pl.col("lanes"))
        .pow(2)
        .alias("flow_sq"),
    )

    # Keep regimes separate so the later aggregation does not give a larger
    # regime more weight merely because it has more masked cells.
    grouped_regimes = joined_state_table.group_by("mask_regime").agg(
        pl.len().alias("n"),
        pl.col("speed_sq").mean().sqrt().alias("speed_rmse"),
        pl.col("flow_sq").mean().sqrt().alias("flow_per_lane_rmse"),
    )
    rows_by_regime = {row["mask_regime"]: row for row in grouped_regimes.to_dicts()}
    regime_scores: list[MetricRow] = [
        {
            "regime": regime,
            "n": rows_by_regime[regime]["n"],
            "speed_rmse": rows_by_regime[regime]["speed_rmse"],
            "flow_per_lane_rmse": rows_by_regime[regime]["flow_per_lane_rmse"],
            "state_score": 0.54 * max(0, 1 - rows_by_regime[regime]["speed_rmse"] / 25)
            + 0.46 * max(0, 1 - rows_by_regime[regime]["flow_per_lane_rmse"] / 600),
        }
        for regime in MASK_REGIMES
        if regime in rows_by_regime
    ]
    if not regime_scores:
        raise ValueError("State template contains no scoreable regimes")
    return regime_scores


def physics_diagnostics(
    state_prediction_table: pl.DataFrame, network: NetworkTables
) -> MetricRow:
    """Calculate local flow-density diagnostics for predicted Task 1 values.

    Convert predicted flow and speed to density, then compare flow with the
    free-flow or congested branch selected by the link's critical density.
    Also report low-flow and negative-value fractions. These are local
    diagnostics, not the competition's physics score.

    Args:
        state_prediction_table: Predicted speed and flow values.
        network: Panel link parameters for the fundamental diagram.

    Returns:
        Physics consistency and value-range diagnostics for the panel.
    """
    # Reindex link parameters to prediction row order before NumPy arithmetic.
    link_parameters = (
        state_prediction_table.select("link_id")
        .with_row_index("_prediction_order")
        .join(network.fd_parameters.collect(), on="link_id", how="left")
        .sort("_prediction_order")
    )
    predicted_flow: NDArray[np.float64] = (
        state_prediction_table["flow_vph"].to_numpy().astype(np.float64)
    )
    predicted_speed: NDArray[np.float64] = (
        state_prediction_table["speed_kmh"].to_numpy().astype(np.float64)
    )
    density: NDArray[np.float64] = predicted_flow / np.maximum(predicted_speed, 1)
    critical_density: NDArray[np.float64] = (
        link_parameters["critical_density"].to_numpy().astype(np.float64)
    )
    jam_density: NDArray[np.float64] = (
        link_parameters["k_jam"].to_numpy().astype(np.float64)
    )
    congestion_wave_speed: NDArray[np.float64] = link_parameters[
        "capacity_vph"
    ].to_numpy().astype(np.float64) / np.maximum(jam_density - critical_density, 1e-9)
    expected_flow: NDArray[np.float64] = np.where(
        density <= critical_density,
        link_parameters["free_speed_kmh"].to_numpy().astype(np.float64) * density,
        congestion_wave_speed * np.maximum(jam_density - density, 0),
    )
    return {
        "fd_relative_error_diagnostic": float(
            np.abs(predicted_flow - expected_flow).sum()
            / max(np.abs(predicted_flow).sum(), 1e-9)
        ),
        "low_flow_fraction_diagnostic": float((predicted_flow < 50).mean()),
        "negative_state_fraction_diagnostic": float(
            ((predicted_flow < 0) | (predicted_speed < 0)).mean()
        ),
    }


def queue_metrics(
    queue_prediction_table: pl.DataFrame, queue_answer_table: pl.DataFrame
) -> MetricRow:
    """Compute intersection over union (IoU) for one queue forecast window.

    Compare only future rows marked eligible for each link and forecast time.
    The score is the number of correctly predicted queued rows divided by the
    number of rows queued in either table. Two all-clear forecasts score one.

    Args:
        queue_prediction_table: Predicted queue state for each target row.
        queue_answer_table: Answer queue state and coverage eligibility per row.

    Returns:
        Intersection over union and the number of eligible forecast rows.

    Raises:
        ValueError: If the window has no eligible cells to score.
    """
    # The proxy builder keeps low-coverage labels for shape consistency but
    # excludes them here so missing measurements cannot decide the score.
    eligible_queue_table = queue_answer_table.join(
        queue_prediction_table, on=KEYS["queue"], validate="1:1"
    ).filter(pl.col("eligible").cast(pl.Boolean))
    if eligible_queue_table.is_empty():
        raise ValueError("Queue window has no eligible labels")
    # Count queued cells in either table for this window, then divide the
    # correctly forecast queued cells by that union.
    actual_queue = eligible_queue_table["queue_true"].to_numpy().astype(bool)
    predicted_queue = eligible_queue_table["queue_pred"].to_numpy().astype(bool)
    union_size = int(np.logical_or(actual_queue, predicted_queue).sum())
    return {
        "queue_proxy_iou": (
            float(np.logical_and(actual_queue, predicted_queue).sum() / union_size)
            if union_size
            else 1.0
        ),
        "n": eligible_queue_table.height,
    }


def odme_metrics(
    path_flow_prediction_table: pl.DataFrame,
    observed_link_count_table: pl.DataFrame,
    weak_prior_table: pl.DataFrame,
    network: NetworkTables,
    path_flow_answer_table: pl.DataFrame | None = None,
) -> MetricRow:
    """Calculate count fit and path-flow accuracy for a Task 4 case.

    Task 4 is origin-destination matrix estimation (ODME). The prediction and
    weak-prior tables give a flow for each requested candidate path, while the
    link-count table gives one observed total per network link. Synthetic cases
    also have known path flows, so they can be scored for path accuracy and for
    how closely predicted vehicle shares match each destination's true share.

    Args:
        path_flow_prediction_table: Predicted flow for each requested candidate path.
        observed_link_count_table: Released link counts for this scenario.
        weak_prior_table: Initial path-flow estimate for each candidate path.
        network: Panel candidate paths and the links used by each path.
        path_flow_answer_table: Known path flows for a generated synthetic case.

    Returns:
        Link-count fit and prior movement metrics. Synthetic cases also include
        path-flow accuracy and destination-attraction metrics.
    """
    # Align paths and measured links before multiplying path flows into link
    # totals. Reindex the prior to the prediction's path order as well.
    link_path_matrix = link_path_incidence_matrix(
        network, path_flow_prediction_table, observed_link_count_table
    )
    predicted_path_flow: NDArray[np.float64] = (
        path_flow_prediction_table["path_flow"].to_numpy().astype(np.float64)
    )
    prior_path_flow: NDArray[np.float64] = (
        path_flow_prediction_table.select("path_id")
        .join(weak_prior_table.select("path_id", "path_flow"), on="path_id")[
            "path_flow"
        ]
        .to_numpy()
        .astype(np.float64)
    )
    observed_link_counts: NDArray[np.float64] = (
        observed_link_count_table["count"].to_numpy().astype(np.float64)
    )
    link_count_score = max(
        0.0,
        1
        - float(
            np.abs(link_path_matrix @ predicted_path_flow - observed_link_counts).sum()
            / max(observed_link_counts.sum(), 1e-9)
        ),
    )
    scores: MetricRow = {
        "odme_link_score_diagnostic": link_count_score,
        "odme_prior_relative_movement": float(
            np.abs(predicted_path_flow - prior_path_flow).sum()
            / max(prior_path_flow.sum(), 1e-9)
        ),
    }
    if path_flow_answer_table is not None:
        # Synthetic answer values make it possible to assess path-level accuracy and
        # how much predicted demand ends at each destination zone.
        true_path_flow: NDArray[np.float64] = (
            path_flow_prediction_table.select("path_id")
            .join(path_flow_answer_table.select("path_id", "path_flow"), on="path_id")[
                "path_flow"
            ]
            .to_numpy()
            .astype(np.float64)
        )
        path_flow_score = max(
            0.0,
            1
            - float(
                np.abs(predicted_path_flow - true_path_flow).sum()
                / max(true_path_flow.sum(), 1e-9)
            ),
        )
        prior_deviation_score = float(
            np.exp(
                -abs(
                    np.abs(predicted_path_flow - prior_path_flow).sum()
                    / max(np.abs(true_path_flow - prior_path_flow).sum(), 1e-9)
                    - 1
                )
            )
        )
        destination_flow = (
            pl.DataFrame(
                {
                    "zone": path_flow_prediction_table["destination_zone"],
                    "predicted": predicted_path_flow,
                    "true": true_path_flow,
                }
            )
            .group_by("zone")
            .agg(pl.col("predicted").sum(), pl.col("true").sum())
        )
        attraction_score = max(
            0.0,
            1
            - float(
                np.abs(
                    destination_flow["predicted"].to_numpy()
                    / max(predicted_path_flow.sum(), 1e-9)
                    - destination_flow["true"].to_numpy()
                    / max(true_path_flow.sum(), 1e-9)
                ).sum()
            )
            / 2,
        )
        scores["odme_synthetic_score"] = (
            0.45 * path_flow_score
            + 0.25 * link_count_score
            + 0.15 * prior_deviation_score
            + 0.15 * attraction_score
        )
    return scores


def aggregate(per_case_metrics_table: pl.DataFrame) -> dict[str, float]:
    """Average local metrics equally across cases, splits, panels, and families.

    Queue overlap is first averaged by condition within each panel and case so
    a condition with more windows cannot receive more weight. Other metrics
    start at the panel level. Non-numeric identity columns are not aggregated.

    Args:
        per_case_metrics_table: Metric rows with case, split, panel, and family IDs.

    Returns:
        Aggregated numeric metrics with equal weight across the documented levels.
    """
    if per_case_metrics_table.is_empty():
        return {}
    # Case identifiers and counts are not score columns. The metric naming
    # suffixes pick only outputs intended for aggregation.
    metric_names = [
        column
        for column in per_case_metrics_table.columns
        if column.endswith(("score", "iou", "diagnostic", "movement", "rmse"))
    ]
    aggregate_scores: dict[str, float] = {}
    for metric_name in metric_names:
        grouped_scores = per_case_metrics_table.with_columns(
            pl.col(metric_name).cast(pl.Float64, strict=False)
        ).filter(pl.col(metric_name).is_finite())
        if grouped_scores.is_empty():
            continue

        # Reduce finer groups first so a panel with more windows does not
        # receive a larger weight than another panel or family.
        grouping_levels: list[tuple[str, ...]] = (
            [("case_id", "split", "panel", "family", "condition")]
            if metric_name == "queue_proxy_iou"
            else []
        )
        grouping_levels.extend(
            [
                ("case_id", "split", "panel", "family"),
                ("case_id", "split", "family"),
                ("case_id", "split"),
            ]
        )
        for group_keys in grouping_levels:
            if grouped_scores.select(
                pl.any_horizontal(pl.col(group_keys).is_null()).any()
            ).item():
                raise ValueError(f"Cannot aggregate {metric_name}: missing group key")
            grouped_scores = grouped_scores.group_by(group_keys).agg(
                pl.col(metric_name).mean()
            )
        aggregate_scores[metric_name] = float(
            grouped_scores[metric_name].to_numpy().mean()
        )
    return aggregate_scores
