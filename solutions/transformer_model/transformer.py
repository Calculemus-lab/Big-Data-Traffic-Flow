"""Encoder transformer that reconstructs masked cells of a density field.

Each layer runs classic multi-head self-attention along time, then along the
ordered detectors. A full space-time attention map is avoided because a day
has one token per detector and five-minute slot. Masked cells are replaced by
zeros plus an observation flag, and the head predicts normalized density.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

MASK_RATES = (0.2, 0.3, 0.5)


class TransformerConfig(NamedTuple):
    """Architecture stored beside a checkpoint.

    Attributes:
        d_model: Width of each token.
        n_heads: Attention heads. Must divide ``d_model``.
        n_layers: Pairs of temporal and spatial attention blocks.
        max_detectors: Largest detector axis the position table can index.
        slots: Time positions. Matches the five-minute slots in one UTC day.
    """

    d_model: int
    n_heads: int
    n_layers: int
    max_detectors: int
    slots: int


class AxisBlock(nn.Module):
    """Pre-norm multi-head self-attention and a position-wise feed-forward layer.

    Attributes:
        attention_norm: Layer norm applied before attention.
        attention: Batch-first multi-head attention along one axis.
        feedforward_norm: Layer norm applied before the feed-forward layer.
        feedforward: Two linear layers with a GELU activation between them.
    """

    def __init__(self, width: int, heads: int) -> None:
        """Create one attention block.

        Args:
            width: Token width.
            heads: Number of attention heads. Must divide ``width``.
        """
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply residual attention and a residual feed-forward layer.

        Args:
            tokens: Sequence with shape ``(batch, length, width)``.

        Returns:
            Sequence with the same shape as ``tokens``.
        """
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = tokens + attended
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class DensityTransformer(nn.Module):
    """Predict normalized density for every cell of a detector-time grid.

    Attributes:
        config: Width, depth, and position-table sizes.
        input_projection: Maps density and the observation flag to ``d_model``.
        detector_position: Learned position of each detector row.
        slot_position: Learned position of each five-minute slot.
        temporal_blocks: Attention along the time axis, one block per layer.
        spatial_blocks: Attention along the detector axis, one block per layer.
        output_norm: Layer norm before the density head.
        output_head: Linear map from ``d_model`` to one logit.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Create the density transformer.

        Args:
            config: Architecture and position-table sizes.

        Raises:
            ValueError: If the head count does not divide the model width.
        """
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.input_projection = nn.Linear(2, config.d_model)
        self.detector_position = nn.Embedding(config.max_detectors, config.d_model)
        self.slot_position = nn.Embedding(config.slots, config.d_model)
        self.temporal_blocks = nn.ModuleList(
            AxisBlock(config.d_model, config.n_heads) for _ in range(config.n_layers)
        )
        self.spatial_blocks = nn.ModuleList(
            AxisBlock(config.d_model, config.n_heads) for _ in range(config.n_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, 1)

    def forward(self, density: Tensor, observed: Tensor) -> Tensor:
        """Predict normalized density for each cell.

        Args:
            density: Observed normalized density with masked cells set to zero.
                Shape ``(batch, detectors, slots)``.
            observed: Observation flag with the same shape as ``density``.
                Values are 1 where the density is visible and 0 where it is masked.

        Returns:
            Normalized density in ``(0, 1)`` with the same shape as ``density``.

        Raises:
            ValueError: If the grid is larger than the position tables.
        """
        _batch, detectors, slots = density.shape
        if detectors > self.config.max_detectors or slots != self.config.slots:
            raise ValueError(
                f"Grid {(detectors, slots)} does not match position tables "
                f"{(self.config.max_detectors, self.config.slots)}"
            )
        features = torch.stack((density, observed), dim=-1)
        hidden = self.input_projection(features)
        detector_index = torch.arange(detectors, device=density.device)
        slot_index = torch.arange(slots, device=density.device)
        hidden = hidden + self.detector_position(detector_index)[None, :, None, :]
        hidden = hidden + self.slot_position(slot_index)[None, None, :, :]
        for temporal, spatial in zip(
            self.temporal_blocks, self.spatial_blocks, strict=True
        ):
            if not isinstance(temporal, AxisBlock) or not isinstance(
                spatial, AxisBlock
            ):
                raise TypeError("Attention block has the wrong type")
            hidden = _attend_time(temporal, hidden)
            hidden = _attend_space(spatial, hidden)
        logits = self.output_head(self.output_norm(hidden)).squeeze(-1)
        return logits.sigmoid()


def resolve_device(requested: str) -> torch.device:
    """Select a torch device, using CUDA when it is requested and available.

    Args:
        requested: ``cuda``, ``cpu``, or another torch device string.

    Returns:
        The requested device, or CPU when a CUDA device was requested but is
        unavailable.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def masked_density_loss(
    model: DensityTransformer,
    clean: Tensor,
    observed: Tensor,
    generator: np.random.Generator,
) -> Tensor:
    """Hide a random fraction of observed cells and score those predictions.

    Args:
        model: Density transformer being trained.
        clean: Normalized density with shape ``(batch, detectors, slots)``.
            Unobserved cells are zero.
        observed: Boolean observation mask with the same shape as ``clean``.
        generator: Generator that draws the mask rate and the hidden cells.

    Returns:
        Mean squared error on cells that were observed and then hidden. The
        value is zero when the draw hides nothing.
    """
    rate = float(generator.choice(MASK_RATES))
    keep = torch.as_tensor(
        generator.random(tuple(observed.shape)) >= rate,
        device=observed.device,
    )
    visible = observed & keep
    prediction = model(clean * visible.to(clean.dtype), visible.to(clean.dtype))
    hidden = observed & ~visible
    hidden_count = hidden.sum()
    if int(hidden_count) == 0:
        return prediction.sum() * 0
    squared_error = (prediction - clean).square()
    return (squared_error * hidden.to(clean.dtype)).sum() / hidden_count


def save_checkpoint(path: Path, model: DensityTransformer) -> None:
    """Write model weights and a JSON architecture sidecar.

    Args:
        path: ``.pt`` destination. The sidecar replaces the suffix with
            ``.json``.
        model: Trained density transformer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(
        json.dumps(model.config._asdict()), encoding="utf-8"
    )


def load_checkpoint(path: Path, device: torch.device) -> DensityTransformer:
    """Load a density transformer onto ``device``.

    Args:
        path: ``.pt`` weights. A sibling ``.json`` stores the architecture.
        device: Device that stores the loaded weights.

    Returns:
        Model in evaluation mode.

    Raises:
        TypeError: If the sidecar is not a JSON object of integers.
    """
    payload: object = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint config must be a JSON object")
    config = TransformerConfig(
        d_model=_json_int(payload, "d_model"),
        n_heads=_json_int(payload, "n_heads"),
        n_layers=_json_int(payload, "n_layers"),
        max_detectors=_json_int(payload, "max_detectors"),
        slots=_json_int(payload, "slots"),
    )
    model = DensityTransformer(config).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_normalized_density(
    model: DensityTransformer,
    density: NDArray[np.float32],
    observed: NDArray[np.bool_],
    device: torch.device,
) -> NDArray[np.float32]:
    """Fill one batch of density grids.

    Observed cells keep their input density. Masked cells take the transformer
    prediction.

    Args:
        model: Density transformer in evaluation mode.
        density: Normalized density shaped ``(batch, detectors, slots)``.
        observed: Observation mask with the same shape as ``density``.
        device: Device used for the forward pass.

    Returns:
        Completed normalized density on CPU, same shape as ``density``.
    """
    density_tensor = torch.as_tensor(density, device=device)
    observed_tensor = torch.as_tensor(observed, device=device)
    with torch.inference_mode():
        predicted = model(density_tensor, observed_tensor.to(density_tensor.dtype))
        completed = torch.where(observed_tensor, density_tensor, predicted)
    return completed.detach().cpu().numpy().astype(np.float32)


def _attend_time(block: AxisBlock, hidden: Tensor) -> Tensor:
    """Run attention independently on each detector's time series.

    Args:
        block: Temporal attention block.
        hidden: Tokens shaped ``(batch, detectors, slots, width)``.

    Returns:
        Tokens with the same shape as ``hidden``.
    """
    batch, detectors, slots, width = hidden.shape
    sequence = hidden.reshape(batch * detectors, slots, width)
    updated = block(sequence)
    return updated.reshape(batch, detectors, slots, width)


def _attend_space(block: AxisBlock, hidden: Tensor) -> Tensor:
    """Run attention independently on each time slot's detector sequence.

    Args:
        block: Spatial attention block.
        hidden: Tokens shaped ``(batch, detectors, slots, width)``.

    Returns:
        Tokens with the same shape as ``hidden``.
    """
    batch, detectors, slots, width = hidden.shape
    sequence = (
        hidden.transpose(1, 2).contiguous().reshape(batch * slots, detectors, width)
    )
    updated = block(sequence)
    return updated.reshape(batch, slots, detectors, width).transpose(1, 2).contiguous()


def _json_int(payload: dict[object, object], name: str) -> int:
    """Read one integer field from a checkpoint config.

    Args:
        payload: JSON object.
        name: Field name.

    Returns:
        The integer value.

    Raises:
        TypeError: If the field is missing or not an integer.
    """
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Checkpoint config field {name} must be an integer")
    return value
