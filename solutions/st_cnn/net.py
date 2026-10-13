import torch
from torch import nn

# (time, corridor) dilations: about +-44 steps (3.7 h) and +-17 links of context.
DILATIONS = [(1, 1), (2, 1), (4, 2), (8, 2), (16, 4), (1, 1), (4, 2), (8, 4)]


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return x + self.mix(torch.relu(self.conv(x)))


class Imputer(nn.Module):
    """Fully convolutional over a (time, corridor position) grid, so any length works."""

    def __init__(self, inputs, width=32):
        super().__init__()
        self.inp = nn.Conv2d(inputs, width, 1)
        self.blocks = nn.Sequential(*[Block(width, d) for d in DILATIONS])
        self.out = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        return self.out(torch.relu(self.blocks(self.inp(x))))


def receptive_steps():
    return sum(t for t, _ in DILATIONS)
