"""Statistical approach that:
1) Improves the provided baseline 
2) On top of that interpolates the same-day residuals
"""
import numpy as np
import pandas as pd
 
VALUES = ['speed_kmh', 'flow_vph']
KEYS = ['timestamp', 'station_id', 'link_id']
SERIES = ['station_id', 'link_id']
DAY = 288  # 5-minute slots per day
 
DEFAULTS = {
    # --- baseline profile ---
    'smooth': 1,             # average over +-k neighbouring slots (1 = 15-min window)
    'half_life_days': 56,    # weight recent training days more; 0 = equal weights
    'day_groups': 'dow',     # 'dow' = 7 day types; 'pooled' = Mon-Fri share one profile
    # --- interpolation ---
    'tau': 20,                # decay of anchor weight, in 5-min steps (6 = 30 min)
    'min_pct': 75,           # visible cells with pct_observed >= this are anchors
    'speed_mode': 'add',     # 'add' or 'mult'
    'flow_mode': 'mult',     # 'add' or 'mult'
    'offset': 100.0,         # added before taking ratios, so low values don't explode
}
 
 
def _clean(frame):
    """Plain numpy dtypes and consistent keys (train arrives as pyarrow-backed)."""
    out = pd.DataFrame({
        'timestamp': pd.to_datetime(frame['timestamp'], utc=True).astype('datetime64[ns, UTC]'),
        'station_id': frame['station_id'].astype(str),
        'link_id': frame['link_id'].astype(str),
    })
    for col in VALUES + ['pct_observed']:
        if col in frame:
            out[col] = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    return out
 
 
def _day_type(ts, mode):
    dow = ts.dt.dayofweek
    return dow if mode == 'dow' else pd.Series(np.where(dow >= 5, dow, 0), index=ts.index)
 
 
def _slot(ts):
    return ts.dt.hour * 12 + ts.dt.minute // 5
 
 
def fit_profile(train, p):
    """Weighted, slot-smoothed mean per (station, link, day type, slot)."""
    tr = train[train.pct_observed.ge(p['min_pct'])].dropna(subset=VALUES).copy()
    if p['half_life_days'] > 0:
        age_days = (tr.timestamp.max() - tr.timestamp).dt.total_seconds() / 86400
        tr['w'] = 0.5 ** (age_days / p['half_life_days'])
    else:
        tr['w'] = 1.0
    tr['day'] = _day_type(tr.timestamp, p['day_groups'])
    tr['slot'] = _slot(tr.timestamp)
    for v in VALUES:
        tr['w_' + v] = tr.w * tr[v]
    sums = tr.groupby(SERIES + ['day', 'slot'])[['w'] + ['w_' + v for v in VALUES]].sum()
 
    # Smooth over neighbouring slots: circular moving sum over the 288 slots.
    k = int(p['smooth'])
    smoothed, index = {}, None
    for col in sums.columns:
        wide = sums[col].unstack('slot').reindex(columns=range(DAY)).fillna(0.0)
        index = wide.index
        arr = wide.to_numpy()
        smoothed[col] = sum(np.roll(arr, s, axis=1) for s in range(-k, k + 1))
    profile = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for v in VALUES:
            ratio = np.where(smoothed['w'] > 0, smoothed['w_' + v] / smoothed['w'], np.nan)
            profile['p_' + v] = pd.DataFrame(ratio, index=index, columns=pd.Index(range(DAY), name='slot')).stack()
    profile = pd.DataFrame(profile).reset_index()
 
    fallback = {'link': tr.groupby('link_id')[VALUES].mean(), 'global': tr[VALUES].mean()}
    return profile, fallback
 
 
def add_profile(frame, profile, fallback, p):
    frame = frame.assign(day=_day_type(frame.timestamp, p['day_groups']).to_numpy(),
                         slot=_slot(frame.timestamp).to_numpy())
    out = frame.merge(profile, on=SERIES + ['day', 'slot'], how='left', validate='many_to_one')
    for v in VALUES:
        out['p_' + v] = (out['p_' + v]
                         .fillna(out.link_id.map(fallback['link'][v]))
                         .fillna(fallback['global'][v]))
    return out.drop(columns=['day', 'slot'])
 
 
def interpolate(obs, value, mode, p):
    """obs must be sorted by station, link, timestamp and have a 'step' column."""
    prof, c = obs['p_' + value], float(p['offset'])
    anchor = obs.pct_observed.ge(p['min_pct']) & obs[value].notna()
 
    # Residual: additive difference, or log ratio for multiplicative mode.
    if mode == 'mult':
        resid = np.log((obs[value].clip(lower=0) + c) / (prof.clip(lower=0) + c))
    else:
        resid = obs[value] - prof
    anchors = pd.DataFrame({'r': resid.where(anchor), 't': obs.step.where(anchor)})
 
    groups = anchors.groupby([obs.station_id, obs.link_id], sort=False)
    prev, nxt = groups.ffill(), groups.bfill()   # nearest anchor before / after
 
    tau = float(p['tau'])
    w_prev = np.exp(-(obs.step - prev.t) / tau).fillna(0.0)
    w_next = np.exp(-(nxt.t - obs.step) / tau).fillna(0.0)
    total = w_prev + w_next
    blended = (w_prev * prev.r.fillna(0) + w_next * nxt.r.fillna(0)) / total.where(total > 0)
    # Shrink toward the profile (residual 0) when the nearest anchor is far away.
    r_hat = (blended * np.maximum(w_prev, w_next)).fillna(0.0)
 
    if mode == 'mult':
        pred = (prof.clip(lower=0) + c) * np.exp(r_hat.clip(-1.6, 1.6)) - c
    else:
        pred = prof + r_hat
    return pred.clip(lower=0)
 
 
def state(ctx):
    p = {**DEFAULTS, **ctx.params}
    for key in ['speed_mode', 'flow_mode']:
        if p[key] not in ('add', 'mult'):
            raise ValueError(f'{key} must be "add" or "mult"')
    if p['tau'] <= 0:
        raise ValueError('tau must be positive')
 
    profile, fallback = fit_profile(_clean(ctx.train), p)
 
    # All evaluation-period cells: visible ones are anchors, blanked ones get predictions.
    obs = add_profile(_clean(ctx.observations), profile, fallback, p)
    obs = obs.sort_values(SERIES + ['timestamp'], ignore_index=True)
    obs['step'] = (obs.timestamp - obs.timestamp.min()).dt.total_seconds() // 300
    obs['pred_speed_kmh'] = interpolate(obs, 'speed_kmh', p['speed_mode'], p)
    obs['pred_flow_vph'] = interpolate(obs, 'flow_vph', p['flow_mode'], p)
 
    # Predict exactly the target rows; fall back to the profile if a key is not in obs.
    targets = add_profile(_clean(ctx.targets), profile, fallback, p)
    merged = targets[KEYS + ['p_' + v for v in VALUES]].merge(
        obs[KEYS + ['pred_' + v for v in VALUES]], on=KEYS, how='left', validate='one_to_one')
 
    out = ctx.targets.copy()
    for v in VALUES:
        out[v] = merged['pred_' + v].fillna(merged['p_' + v]).clip(lower=0).to_numpy()
    return out
