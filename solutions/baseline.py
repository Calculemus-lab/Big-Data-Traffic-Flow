"""Three independent baselines. Copy one function to start an experiment."""
import numpy as np
from scipy.optimize import minimize

from trafficbench.metrics import operator


def state(ctx):
    train = ctx.train[ctx.train.pct_observed.ge(75)].copy()
    train['slot'] = train.timestamp.dt.dayofweek * 288 + train.timestamp.dt.hour * 12 + train.timestamp.dt.minute // 5
    values = ['speed_kmh', 'flow_vph']
    profile = train.groupby(['link_id', 'slot'])[values].mean()
    fallback = train.groupby('link_id')[values].mean()
    out = ctx.targets.copy()
    out['slot'] = out.timestamp.dt.dayofweek * 288 + out.timestamp.dt.hour * 12 + out.timestamp.dt.minute // 5
    out = out.merge(profile, on=['link_id','slot'], how='left', validate='many_to_one')
    for value in values:
        out[value] = out[value].fillna(out.link_id.map(fallback[value])).fillna(train[value].mean()).clip(lower=0)
    return out.drop(columns='slot')


def queue(ctx):
    threshold = .6 * ctx.network['links'].set_index('link_id').free_speed_kmh
    history = ctx.observations
    last = history[history.timestamp.eq(history.timestamp.max())].copy()
    last['queued'] = last.speed_kmh.le(last.link_id.map(threshold)) & last.pct_observed.ge(75)
    now = last.groupby('link_id').queued.any()
    out = ctx.targets.copy()
    out['queue_pred'] = out.link_id.map(now).fillna(False).astype(int)
    return out


def odme(ctx):
    A = operator(ctx.network, ctx.targets, ctx.counts)
    prior = ctx.prior.set_index('path_id').reindex(ctx.targets.path_id).path_flow.to_numpy(float)
    regularization = float(ctx.params.get('regularization', .05))
    if regularization <= 0:
        raise ValueError('regularization must be positive')
    counts = ctx.counts['count'].to_numpy(float)

    def objective(f):
        residual, movement = A @ f - counts, f - prior
        loss = .5 * (residual @ residual + regularization * (movement @ movement))
        gradient = A.T @ residual + regularization * movement
        return loss, gradient

    result = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(prior),
                      options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-5})
    if not result.success:
        raise RuntimeError(f'ODME solve failed: {result.message}')
    out = ctx.targets.copy()
    out['path_flow'] = result.x
    return out
