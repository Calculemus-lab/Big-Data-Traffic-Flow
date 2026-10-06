"""Spatiotemporal masked CNN for Task 1.

Self-supervised: visible cells are hidden at the competition rates and the network
learns to fill them in from nearby times and links. It predicts residuals over a
day-of-week x time-of-day profile and is trained on the competition loss.
"""
import numpy as np
import pandas as pd
import torch

from .grid import Profile, clean, features, lanes_for, locate, period, series_order, to_grid
from .net import Imputer, receptive_steps

DEFAULTS = {
    'steps': 1000,
    'batch': 8,
    'window': 192,       # training crop length in 5-minute steps
    'width': 32,
    'lr': 2e-3,
    'smooth': 2,         # profile smoothing, +- slots
    'use_eval': True,    # also learn from visible cells of the evaluation period
    'device': 'auto',
    'log_every': 250,
    'seed': 0,           # offset on the harness seed, for seed-spread runs
}
RATES = (.2, .3, .5)


def pick_device(name):
    if name == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    return name


def hide(visible, rng):
    """Hide visible cells independently at one of the regime rates."""
    return visible & (rng.random(visible.shape) < rng.choice(RATES))


def score_loss(pred, target, mask):
    """0.54 RMSE_speed/25 + 0.46 RMSE_flow_per_lane/600 over hidden cells."""
    n = mask.sum().clamp(min=1)
    mse = (((pred - target) ** 2) * mask[:, None]).sum((0, 2, 3)) / n
    return .54 * mse[0].clamp(min=1e-8).sqrt() / 25 + .46 * mse[1].clamp(min=1e-8).sqrt() / 600


def batches(pools, p, rng):
    weights = np.array([g.shape[1] for g, _, _ in pools], dtype=float)
    while True:
        xs, ys, ms = [], [], []
        for _ in range(p['batch']):
            values, profile, times = pools[rng.choice(len(pools), p=weights / weights.sum())]
            length = min(p['window'], values.shape[1])
            start = rng.integers(0, values.shape[1] - length + 1)
            crop = slice(start, start + length)
            v, prof, t = values[:, crop], profile[:, crop], times[crop]
            visible = ~np.isnan(v[0])
            hidden = hide(visible, rng)
            xs.append(features(v, visible & ~hidden, prof, t, p['norm']))
            ys.append(np.nan_to_num(v - prof).astype(np.float32))
            ms.append(hidden.astype(np.float32))
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms))


def fit(pools, p, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(p['device'])
    net = Imputer(7, p['width']).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p['lr'], total_steps=max(p['steps'], 2))
    scale = torch.tensor(p['norm']['resid'], dtype=torch.float32, device=device)[None, :, None, None]
    running, stream = None, batches(pools, p, rng)
    for step in range(p['steps']):
        x, y, m = (tensor.to(device) for tensor in next(stream))
        loss = score_loss(net(x) * scale, y, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
        running = loss.item() if running is None else .95 * running + .05 * loss.item()
        if p['log_every'] and (step + 1) % p['log_every'] == 0:
            print(f'  step {step + 1}: loss {running:.4f} (~1 - state score on hidden train cells)', flush=True)
    return net.eval()


@torch.no_grad()
def predict(net, values, profile, times, p, chunk=576):
    """Full-grid prediction in overlapping chunks; returns (2, T, S)."""
    device = next(net.parameters()).device
    pad, total = 2 * receptive_steps(), values.shape[1]
    out = np.full_like(values, np.nan)
    visible = ~np.isnan(values[0])
    for a in range(0, total, chunk):
        lo, hi, b = max(0, a - pad), min(total, a + chunk + pad), min(total, a + chunk)
        x = features(values[:, lo:hi], visible[lo:hi], profile[:, lo:hi], times[lo:hi], p['norm'])
        y = net(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
        out[:, a:b] = y[:, a - lo:b - lo] * p['norm']['resid'][:, None, None]
    return profile + out


def state(ctx):
    p = {**DEFAULTS, **ctx.params}
    if p['steps'] < 1 or p['window'] < 1:
        raise ValueError('steps and window must be positive')
    train, obs, targets = clean(ctx.train), clean(ctx.observations), clean(ctx.targets)
    series = series_order(ctx.network, [train, obs, targets])
    lanes = lanes_for(series, ctx.network)
    train_times, eval_times = period(train), period(obs)
    train_grid, eval_grid = to_grid(train, train_times, series, lanes), to_grid(obs, eval_times, series, lanes)
    profile = Profile(train_grid, train_times, p['smooth'])
    train_profile, eval_profile = profile(train_times), profile(eval_times)

    resid = train_grid - train_profile
    p['norm'] = {'resid': np.maximum(np.nanstd(resid, axis=(1, 2)), 1.), 'mean': np.nanmean(train_grid, axis=(1, 2)),
                 'std': np.maximum(np.nanstd(train_grid, axis=(1, 2)), 1.)}
    pools = [(train_grid, train_profile, train_times)]
    if p['use_eval']:
        pools.append((eval_grid, eval_profile, eval_times))
    net = fit(pools, p, ctx.seed + int(p['seed']))
    pred = predict(net, eval_grid, eval_profile, eval_times, p)

    t, s = locate(targets, eval_times, series)
    inside = t >= 0
    speed, flow = np.full(len(targets), np.nan), np.full(len(targets), np.nan)
    speed[inside] = pred[0, t[inside], s[inside]]
    flow[inside] = pred[1, t[inside], s[inside]] * lanes[s[inside]]
    # Rare keys outside the grid fall back to the series' profile mean.
    fallback = pd.DataFrame({'speed_kmh': train_profile[0].mean(0), 'flow_vph': train_profile[1].mean(0) * lanes}, index=series)
    keys = pd.MultiIndex.from_arrays([targets.station_id, targets.link_id])
    out = ctx.targets.copy()
    out['speed_kmh'] = np.where(np.isnan(speed), fallback.speed_kmh.reindex(keys).to_numpy(), speed)
    out['flow_vph'] = np.where(np.isnan(flow), fallback.flow_vph.reindex(keys).to_numpy(), flow)
    out[['speed_kmh', 'flow_vph']] = out[['speed_kmh', 'flow_vph']].clip(lower=0)
    return out
