"""The graph neural network that predicts queues."""

from __future__ import annotations

import torch
from torch import nn
from .grids import FORECAST_STEPS, HISTORY_STEPS

CHANNELS_PER_STEP = 3  # speed ratio, occupancy, known flag
SEQUENCE_FEATURES = HISTORY_STEPS * CHANNELS_PER_STEP


class SpatioTemporalBlock(nn.Module):
    """Causal temporal convolution, then mixing with the links ahead and behind."""

    def __init__(
        self, channels: int, ahead: torch.Tensor, behind: torch.Tensor, kernel: int = 3
    ) -> None:
        super().__init__()
        self.register_buffer("ahead", ahead)
        self.register_buffer("behind", behind)
        self.kernel = kernel
        self.temporal = nn.Conv1d(channels, channels, kernel, padding=kernel - 1)
        self.own = nn.Linear(channels, channels)
        self.from_ahead = nn.Linear(channels, channels, bias=False)
        self.from_behind = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map (examples, steps, links, channels) to the same shape."""
        examples, steps, links, channels = state.shape
        sequence = state.permute(0, 2, 3, 1).reshape(examples * links, channels, steps)
        sequence = self.temporal(sequence)[..., :steps]  # trim the end to stay causal
        convolved = torch.relu(sequence).reshape(examples, links, channels, steps)
        convolved = convolved.permute(0, 3, 1, 2)
        mixed = (
            self.own(convolved)
            + self.from_ahead(self.ahead @ convolved)
            + self.from_behind(self.behind @ convolved)
        )
        return torch.relu(self.norm(state + mixed))


class QueueSTGNN(nn.Module):
    """Spatio-temporal graph network for the six-step queue forecast."""

    def __init__(
        self,
        feature_count: int,
        ahead: torch.Tensor,
        behind: torch.Tensor,
        hidden: int = 48,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(CHANNELS_PER_STEP, hidden)
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(hidden, ahead, behind) for _ in range(layers)]
        )
        self.static = nn.Sequential(
            nn.Linear(feature_count - SEQUENCE_FEATURES, hidden), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, FORECAST_STEPS)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (examples, links, features) to raw scores (examples, links, 6)."""
        examples, links, _ = inputs.shape
        sequence = inputs[..., :SEQUENCE_FEATURES].reshape(
            examples, links, HISTORY_STEPS, CHANNELS_PER_STEP
        )
        state = self.embed(
            sequence.permute(0, 2, 1, 3)
        )  # (examples, steps, links, hidden)
        for block in self.blocks:
            state = block(state)
        summary = torch.cat(
            [
                state[:, -1],
                state.mean(dim=1),
                self.static(inputs[..., SEQUENCE_FEATURES:]),
            ],
            dim=-1,
        )
        return self.head(summary)
