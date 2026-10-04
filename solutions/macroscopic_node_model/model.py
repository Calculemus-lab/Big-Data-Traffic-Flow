"""Implement the state approach here; add helper modules as needed."""
from solutions import baseline

from .network import build_network
from .visualize import draw_topology


def state(ctx):
    lwr = build_network(ctx)
    if lwr is not None:
        draw_topology(lwr, ctx)
    out = baseline.state(ctx)
    # Replace or improve the baseline. Parameters are in ctx.params.
    return out
