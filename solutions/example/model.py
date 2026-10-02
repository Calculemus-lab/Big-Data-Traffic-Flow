"""Example idea: use visible same-link neighbours to reconstruct missing cells.

Try: bench run example --task state
Tune: bench run example --task state --params '{"blend": 0.75}'
"""
from solutions import baseline


def state(ctx):
    out = baseline.state(ctx)
    blend = float(ctx.params.get('blend', .5))
    if not 0 <= blend <= 1:
        raise ValueError('blend must be between 0 and 1')
    # Offline Task 1 permits interpolation using the visible evaluation month.
    obs = ctx.observations.sort_values(['station_id', 'link_id', 'timestamp']).copy()
    values = ['speed_kmh', 'flow_vph']
    obs.loc[obs.pct_observed.lt(75), values] = float('nan')
    obs[values] = obs.groupby(['station_id', 'link_id'])[values].transform(lambda s: s.interpolate(limit_direction='both'))
    near = out[['timestamp','station_id','link_id']].merge(obs[['timestamp','station_id','link_id'] + values], how='left', validate='one_to_one')
    for col in values:
        out[col] = (1-blend)*out[col] + blend*near[col].fillna(out[col])
    return out
