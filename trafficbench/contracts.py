from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id", "origin_zone", "destination_zone"],
}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


@dataclass
class Context:
    """Permitted inputs only. Times are UTC; identifiers are strings.

    train: earlier unmasked observations (empty for ODME).
    observations: masked evaluation month for state; ONE history for queue.
    targets: exact keys to predict, without answers.
    network: named CSV tables, e.g. network['links'].
    cache: reusable model state within this task/panel/fold.
    """

    task: str
    panel: str
    fold: str
    train: pd.DataFrame
    observations: pd.DataFrame
    targets: pd.DataFrame
    network: dict
    params: dict = field(default_factory=dict)
    ramps: pd.DataFrame = field(default_factory=pd.DataFrame)
    counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    prior: pd.DataFrame = field(default_factory=pd.DataFrame)
    cache: dict = field(default_factory=dict)
    seed: int = 69420


def normalize(frame):
    frame = frame.copy()
    for col in set(sum(KEYS.values(), [])) - {"timestamp"}:
        if col in frame:
            if frame[col].isna().any():
                raise ValueError(f"Null identifier: {col}")
            frame[col] = frame[col].astype(str)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    return frame


def validate_predictions(task, targets, predictions):
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError(f"{task} must return a pandas DataFrame")
    keys, values = KEYS[task], VALUES[task]
    missing = set(keys + values) - set(predictions)
    if missing:
        raise ValueError(f"{task}: missing columns {sorted(missing)}")
    wanted, actual = normalize(targets[keys]), normalize(predictions[keys + values])
    for label, table in [("targets", wanted), ("predictions", actual)]:
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError(f"{task}: null or duplicate keys in {label}")
    if len(actual) != len(wanted):
        raise ValueError(f"{task}: expected {len(wanted)} rows, got {len(actual)}")
    aligned = wanted.merge(actual, on=keys, how="left", validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError(f"{task}: missing or unknown target keys (including OD zones)")
    aligned = aligned.drop(columns="_merge")
    for col in values:
        aligned[col] = pd.to_numeric(aligned[col], errors="raise")
        if not np.isfinite(aligned[col]).all():
            raise ValueError(f"{task}: {col} must be finite")
    if task == "queue" and not aligned.queue_pred.isin([0, 1]).all():
        raise ValueError("queue_pred must be binary, not probabilities")
    if task == "odme" and (aligned.path_flow < 0).any():
        raise ValueError("path_flow must be nonnegative")
    return aligned
