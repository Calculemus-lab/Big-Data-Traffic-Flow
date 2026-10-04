"""Implement the state approach here; add helper modules as needed."""
from solutions import baseline

from .visualize import draw_topology


def state(ctx):
    draw_topology(ctx)
    out = baseline.state(ctx)
    # Replace or improve the baseline. Parameters are in ctx.params.
    return out
