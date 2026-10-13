"""Training loop for the queue GCN."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .example import Series, example_answers, example_inputs, selected_origins
from .grids import FORECAST_STEPS, HISTORY_STEPS
from .network import QueueSTGNN


def train_model(
    series: Series,
    ahead: NDArray[np.float32],
    behind: NDArray[np.float32],
    *,
    layers: int = 2,
    epochs: int = 4,
    positive_weight: float = 3.0,
    onset_weight: float = 3.0,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    seed: int = 0,
    example_limit: int | None = None,
) -> QueueSTGNN:
    """Train on every moment with a full history and forecast; log the loss per epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random = np.random.default_rng(seed)

    feature_count = 3 * HISTORY_STEPS + 3 + FORECAST_STEPS
    model = QueueSTGNN(
        feature_count, torch.from_numpy(ahead), torch.from_numpy(behind), layers=layers
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(positive_weight, device=device)
    )

    origins, ongoing = selected_origins(series)
    moment_weight = np.where(ongoing, 1.0, onset_weight).astype(np.float32)
    if example_limit is not None:  # a quick test run on a random subset
        keep = random.choice(
            len(origins), min(example_limit, len(origins)), replace=False
        )
        origins, moment_weight = origins[keep], moment_weight[keep]

    model.train()
    for epoch in range(epochs):
        order = random.permutation(len(origins))
        total_loss, batch_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            batch = origins[chosen]
            weight = torch.from_numpy(moment_weight[chosen]).to(device)[:, None, None]
            inputs = torch.from_numpy(example_inputs(series, batch)).to(device)
            queued, known = example_answers(series, batch)
            queued_tensor = torch.from_numpy(queued).to(device)
            known_tensor = torch.from_numpy(known).to(device)

            per_cell_loss = loss_function(model(inputs), queued_tensor)
            cell_weight = known_tensor * weight
            loss = (per_cell_loss * cell_weight).sum() / cell_weight.sum().clamp(min=1)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            batch_count += 1
        print(
            f"epoch {epoch + 1}/{epochs}: loss {total_loss / batch_count:.4f}",
            flush=True,
        )
    model.eval()
    return model
