"""Fully convolutional imputer over a (time, corridor position) grid."""

from __future__ import annotations

import torch
from torch import nn

# (time, corridor) dilations per block. "base": about +-44 steps (3.7 h) and +-17 links
# of context; "wide" adds three blocks for about +-94 steps (7.8 h) and +-35 links.
DILATIONS = ((1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (1, 1), (4, 2), (8, 4))
CONTEXTS = {"base": DILATIONS, "wide": (*DILATIONS, (32, 8), (2, 2), (16, 8))}


class Block(nn.Module):
    """Residual dilated 3x3 convolution followed by a 1x1 mix."""

    def __init__(self, width: int, dilation: tuple[int, int]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(torch.relu(self.conv(x)))


class Imputer(nn.Module):
    """Map (batch, 7, T, S) inputs to (batch, 2, T, S) normalized residuals.

    Fully convolutional, so it runs on grids of any length.
    """

    def __init__(
        self,
        inputs: int,
        width: int = 32,
        dilations: tuple[tuple[int, int], ...] = DILATIONS,
    ) -> None:
        super().__init__()
        self.dilations = dilations
        self.inp = nn.Conv2d(inputs, width, 1)
        self.blocks = nn.Sequential(*[Block(width, d) for d in dilations])
        self.out = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.blocks(self.inp(x))))


def receptive_steps(dilations: tuple[tuple[int, int], ...] = DILATIONS) -> int:
    """Return how many time steps each output sees on either side."""
    return sum(time for time, _ in dilations)
