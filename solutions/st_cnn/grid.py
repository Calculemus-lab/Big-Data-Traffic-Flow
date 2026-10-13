"""Dense time x corridor grids, the historical profile and model inputs."""
import numpy as np
import pandas as pd

DAY = 288
SERIES = ['station_id', 'link_id']
STEP = pd.Timedelta(minutes=5)


def clean(frame):
    """Plain numpy dtypes (train arrives pyarrow-backed)."""
    out = pd.DataFrame({
        'timestamp': pd.to_datetime(frame['timestamp'], utc=True).astype('datetime64[ns, UTC]'),
        'station_id': frame['station_id'].astype(str),
        'link_id': frame['link_id'].astype(str),
    })
    for col in ['speed_kmh', 'flow_vph', 'pct_observed']:
        if col in frame:
            out[col] = frame[col].to_numpy(dtype=float, na_value=np.nan)
    return out


def series_order(network, frames):
    """(station, link) pairs in mainline order, so neighbouring columns are neighbouring links."""
    pairs = pd.concat([f[SERIES] for f in frames]).drop_duplicates()
    topology = network.get('lwr_mainline_topology')
    rank = topology.set_index('link_id').order_index if topology is not None else pd.Series(dtype=float)
    pairs = pairs.assign(rank=pairs.link_id.map(rank).astype(float).fillna(np.inf))
    pairs = pairs.sort_values(['rank', 'link_id', 'station_id'])
    return pd.MultiIndex.from_arrays([pairs.station_id, pairs.link_id], names=SERIES)


def lanes_for(series, network):
    lanes = series.get_level_values('link_id').map(network['fd_parameters'].set_index('link_id').lanes)
    lanes = np.asarray(lanes, dtype=float)
    if np.isnan(lanes).any() or (lanes <= 0).any():
        raise ValueError('Missing or invalid lane counts')
    return lanes


def period(frame):
    return pd.date_range(frame.timestamp.min().floor('D'), frame.timestamp.max(), freq=STEP)


def locate(frame, times, series):
    """Grid (time, series) indices of each row; -1 when outside the grid."""
    t = ((frame.timestamp - times[0]) // STEP).to_numpy()
    s = series.get_indexer(pd.MultiIndex.from_arrays([frame.station_id, frame.link_id]))
    t = np.where((t >= 0) & (t < len(times)) & (s >= 0), t, -1)
    return t, np.where(t >= 0, s, -1)


def to_grid(frame, times, series, lanes):
    """(2, T, S) speed and per-lane flow; NaN wherever a cell is not usable evidence."""
    grid = np.full((2, len(times), len(series)), np.nan)
    t, s = locate(frame, times, series)
    ok = (t >= 0) & frame.pct_observed.ge(75).to_numpy() & frame.speed_kmh.notna().to_numpy() & frame.flow_vph.notna().to_numpy()
    grid[0, t[ok], s[ok]] = frame.speed_kmh.to_numpy()[ok]
    grid[1, t[ok], s[ok]] = frame.flow_vph.to_numpy()[ok] / lanes[s[ok]]
    return grid


def slot(times):
    return np.asarray(times.hour * 12 + times.minute // 5)


def smoothed(week, k):
    """Moving sum over +-k slots along the cyclic week, so 23:55 borders the next day."""
    return np.sum([np.roll(week, shift, axis=0) for shift in range(-k, k + 1)], axis=0)


class Profile:
    """Day-of-week x time-of-day mean per series, smoothed over +-smooth neighbouring slots."""

    def __init__(self, grid, times, smooth=2):
        values = np.moveaxis(grid, 1, 0)
        ok = ~np.isnan(values)
        key = np.asarray(times.dayofweek) * DAY + slot(times)
        sums, counts = np.zeros((7 * DAY,) + values.shape[1:]), np.zeros((7 * DAY,) + values.shape[1:])
        np.add.at(sums, key, np.where(ok, values, 0))
        np.add.at(counts, key, ok)
        sums, counts = (smoothed(a, smooth).reshape((7, DAY) + values.shape[1:]) for a in (sums, counts))
        with np.errstate(invalid='ignore', divide='ignore'):
            table = sums / counts
            fallbacks = [sums.sum(0) / counts.sum(0), sums.sum((0, 1)) / counts.sum((0, 1)),
                         (sums.sum((0, 1, 3)) / counts.sum((0, 1, 3)))[:, None]]
        for fallback in fallbacks:
            table = np.where(np.isnan(table), fallback, table)
        if np.isnan(table).any():
            raise ValueError('No usable training observations for the profile')
        self.table = table

    def __call__(self, times):
        return np.moveaxis(self.table[np.asarray(times.dayofweek), slot(times)], 0, 1)


def features(values, visible, profile, times, norm):
    """(7, T, S) inputs: masked residuals, visibility, profile and time of day."""
    resid = np.where(visible, values - profile, 0) / norm['resid'][:, None, None]
    angle = 2 * np.pi * slot(times) / DAY
    clock = np.broadcast_to(np.stack([np.sin(angle), np.cos(angle)])[:, :, None], (2,) + visible.shape)
    level = (profile - norm['mean'][:, None, None]) / norm['std'][:, None, None]
    return np.concatenate([resid, visible[None], level, clock]).astype(np.float32)
