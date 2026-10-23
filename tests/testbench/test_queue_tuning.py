"""Check reproducible searches and separation of tuning from later targets."""

from datetime import timedelta
from pathlib import Path

import pytest

from scripts.tune_queue import candidate_parameters, validate_folds
from solutions.queue_boosting_event_aware.model import QueueParameters
from trafficbench.runner import verify_benchmark


def test_parameter_search_includes_reference_without_duplicates() -> None:
    """Keep defaults first and reproduce all sampled combinations from the seed."""
    candidates = candidate_parameters(12, 31, 2)
    assert candidates[0] == QueueParameters(threads=2)
    assert candidates == candidate_parameters(12, 31, 2)
    assert len({parameters.model_dump_json() for parameters in candidates}) == 12
    assert all(parameters.threads == 2 for parameters in candidates)
    with pytest.raises(ValueError, match="trials"):
        candidate_parameters(0, 31, 2)


def test_search_requires_selector_matched_chronological_folds(
    prepared_benchmark_directory: Path,
) -> None:
    """Reject random onset sampling and reuse of tuning targets as holdout."""
    tuning = verify_benchmark(prepared_benchmark_directory)
    with pytest.raises(ValueError, match="selector-matched"):
        validate_folds(tuning, tuning)
    tuning.scheme.queue.onset_at_final_horizon = True
    with pytest.raises(ValueError, match="Holdout targets"):
        validate_folds(tuning, tuning)
    holdout = tuning.model_copy(deep=True)
    for case in holdout.cases:
        case.prediction_start_date += timedelta(days=31)
        case.prediction_end_date += timedelta(days=31)
    validate_folds(tuning, holdout)
