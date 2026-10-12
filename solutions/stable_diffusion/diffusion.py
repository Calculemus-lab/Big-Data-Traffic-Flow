"""Conditional denoising diffusion for a two-channel traffic field.

The network follows the stable-diffusion training loop: corrupt a clean field
with a cosine noise schedule, predict the added noise with a U-Net, and draw
samples with DDIM. Known detector cells are pinned during sampling the way
inpainting repaints observed pixels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

IN_CHANNELS = 7
OUT_CHANNELS = 2
MASK_RATES = (0.2, 0.3, 0.5)


def resolve_device(requested: str) -> torch.device:
    """Select a torch device, using CUDA when it is requested and available.

    Args:
        requested: ``cuda``, ``cpu``, or another torch device string.

    Returns:
        The requested device, or CPU when CUDA was requested but is unavailable.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def cosine_alpha_bar(timesteps: int, device: torch.device) -> Tensor:
    """Build a cosine cumulative noise schedule.

    Args:
        timesteps: Number of diffusion steps. Index 0 is nearly clean.
        device: Device that stores the schedule.

    Returns:
        ``alpha_bar`` with shape ``(timesteps,)``, decreasing toward pure noise.
    """
    positions = torch.arange(timesteps, dtype=torch.float64)
    offset = 0.008
    alpha_bar = (
        torch.cos(((positions / timesteps) + offset) / (1 + offset) * math.pi / 2) ** 2
    )
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar.to(device=device, dtype=torch.float32)


class ConditionalUNet(nn.Module):
    """U-Net that predicts noise in a speed-flow field.

    The input concatenates the noisy field, the observed field, the observation
    mask, and the weekday-slot prior.

    Attributes:
        time_dimension: Width of the sinusoidal time embedding.
    """

    def __init__(self, base_channels: int = 32) -> None:
        """Create the denoiser.

        Args:
            base_channels: Channel count at the input resolution. Later levels
                double this width.
        """
        super().__init__()
        self.time_dimension = base_channels * 4
        channels = (base_channels, base_channels * 2, base_channels * 4)
        self.input_block = ConvBlock(IN_CHANNELS, channels[0], self.time_dimension)
        self.down_one = DownBlock(channels[0], channels[1], self.time_dimension)
        self.down_two = DownBlock(channels[1], channels[2], self.time_dimension)
        self.middle = ConvBlock(channels[2], channels[2], self.time_dimension)
        self.up_two = UpBlock(
            channels[2], channels[1], channels[1], self.time_dimension
        )
        self.up_one = UpBlock(
            channels[1], channels[0], channels[0], self.time_dimension
        )
        self.output_layer = nn.Conv2d(channels[0], OUT_CHANNELS, kernel_size=1)

    def forward(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        observed: Tensor,
        mask: Tensor,
        prior: Tensor,
    ) -> Tensor:
        """Predict the noise component of ``noisy``.

        Args:
            noisy: Noisy speed and flow, shape ``(batch, 2, detectors, slots)``.
            timesteps: Integer diffusion step for each batch element.
            observed: Observed field in model units, zeros on missing cells.
            mask: Observation mask, shape ``(batch, 1, detectors, slots)``.
            prior: Weekday-slot prior in model units.

        Returns:
            Noise estimate with the same shape as ``noisy``.
        """
        features = torch.cat((noisy, observed, mask, prior), dim=1)
        features, height, width = _pad_to_multiple(features, multiple=8)
        time_embedding = sinusoidal_embedding(timesteps, self.time_dimension)
        full_resolution = self.input_block(features, time_embedding)
        half_resolution = self.down_one(full_resolution, time_embedding)
        quarter_resolution = self.down_two(half_resolution, time_embedding)
        hidden = self.middle(quarter_resolution, time_embedding)
        hidden = self.up_two(hidden, half_resolution, time_embedding)
        hidden = self.up_one(hidden, full_resolution, time_embedding)
        return self.output_layer(hidden)[:, :, :height, :width]


def diffusion_loss(
    model: ConditionalUNet,
    clean: Tensor,
    observed_values: Tensor,
    mask: Tensor,
    prior: Tensor,
    alpha_bar: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Compute the noise-prediction loss with an R1/R2/R3-style random mask.

    Args:
        model: Denoiser being trained.
        clean: Complete speed-flow target in model units.
        observed_values: Measured values in model units.
        mask: Cells that were actually measured.
        prior: Weekday-slot conditioning field.
        alpha_bar: Cosine schedule indexed by diffusion step.
        generator: Random generator used for the step, noise, and mask.

    Returns:
        Mean squared error between predicted and sampled noise.
    """
    batch = clean.shape[0]
    device = clean.device
    steps = torch.randint(
        0,
        alpha_bar.shape[0],
        (batch,),
        device=device,
        generator=generator,
    )
    noise = torch.randn(clean.shape, device=device, generator=generator)
    scale = alpha_bar[steps][:, None, None, None]
    noisy = scale.sqrt() * clean + (1.0 - scale).sqrt() * noise
    rate_index = int(
        torch.randint(
            0, len(MASK_RATES), (1,), device=device, generator=generator
        ).item()
    )
    drop_rate = MASK_RATES[rate_index]
    retained = mask * (
        torch.rand(mask.shape, device=device, generator=generator) >= drop_rate
    )
    empty = retained.flatten(1).sum(dim=1) == 0
    retained = torch.where(empty[:, None, None, None], mask, retained)
    conditioned = observed_values * retained
    predicted = model(noisy, steps, conditioned, retained, prior)
    return torch.mean((predicted - noise) ** 2)


def sample_field(
    model: ConditionalUNet,
    observed_values: Tensor,
    mask: Tensor,
    prior: Tensor,
    alpha_bar: Tensor,
    sample_steps: int,
    generator: torch.Generator,
) -> Tensor:
    """Draw a speed-flow field and pin cells that were observed.

    Args:
        model: Trained denoiser in evaluation mode.
        observed_values: Measured field in model units, zeros where missing.
        mask: Ones on measured cells.
        prior: Weekday-slot prior in model units.
        alpha_bar: Cosine schedule used at training time.
        sample_steps: Number of DDIM steps, clipped to the schedule length.
        generator: Random generator for the initial noise and repaint noise.

    Returns:
        Denoised field in model units, shape ``(batch, 2, detectors, slots)``.
    """
    model.eval()
    step_count = min(sample_steps, int(alpha_bar.shape[0]))
    times = (
        torch.linspace(
            alpha_bar.shape[0] - 1,
            0,
            step_count,
            device=alpha_bar.device,
        )
        .round()
        .to(dtype=torch.long)
    )
    field = torch.randn(
        observed_values.shape,
        device=observed_values.device,
        generator=generator,
    )
    for index, time_value in enumerate(times.tolist()):
        current = int(time_value)
        timesteps = torch.full(
            (field.shape[0],),
            current,
            device=field.device,
            dtype=torch.long,
        )
        predicted_noise = model(field, timesteps, observed_values, mask, prior)
        alpha = alpha_bar[current]
        clean = (field - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp(
            min=1e-8
        )
        clean = clean.clamp(-1.0, 1.0)
        if index + 1 == len(times):
            return mask * observed_values + (1.0 - mask) * clean
        previous = int(times[index + 1].item())
        alpha_previous = alpha_bar[previous]
        field = (
            alpha_previous.sqrt() * clean
            + (1.0 - alpha_previous).sqrt() * predicted_noise
        )
        repaint_noise = torch.randn(
            field.shape, device=field.device, generator=generator
        )
        known = (
            alpha_previous.sqrt() * observed_values
            + (1.0 - alpha_previous).sqrt() * repaint_noise
        )
        field = mask * known + (1.0 - mask) * field
    return field


def save_checkpoint(
    path: Path,
    model: ConditionalUNet,
    *,
    timesteps: int,
    base_channels: int,
) -> None:
    """Write model weights and the architecture settings required to reload them.

    Args:
        path: ``.pt`` destination. A sibling ``.json`` file stores the schedule
            length and channel width.
        model: Denoiser to save.
        timesteps: Training schedule length.
        base_channels: U-Net base width.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(
        json.dumps({"timesteps": timesteps, "base_channels": base_channels})
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[ConditionalUNet, int]:
    """Load a denoiser saved by :func:`save_checkpoint`.

    Args:
        path: ``.pt`` weights path. The matching ``.json`` must sit beside it.
        device: Device that receives the weights.

    Returns:
        Evaluation-mode network and the schedule length stored with it.
    """
    config = json.loads(path.with_suffix(".json").read_text())
    model = ConditionalUNet(base_channels=int(config["base_channels"]))
    weights = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model, int(config["timesteps"])


def as_batch(array: NDArray[np.floating], device: torch.device) -> Tensor:
    """Add a batch dimension and move a numpy field onto ``device``.

    Args:
        array: Field without a batch axis.
        device: Destination device.

    Returns:
        Float tensor with a leading batch dimension of 1.
    """
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


class ConvBlock(nn.Module):
    """Two convolutions conditioned on a diffusion-time embedding."""

    def __init__(
        self, in_channels: int, out_channels: int, time_dimension: int
    ) -> None:
        """Create the block.

        Args:
            in_channels: Channels of the incoming feature map.
            out_channels: Channels produced by both convolutions.
            time_dimension: Width of the time embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time_projection = nn.Linear(time_dimension, out_channels)

    def forward(self, features: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply the convolutions and add the projected time embedding.

        Args:
            features: Incoming feature map.
            time_embedding: One embedding vector per batch element.

        Returns:
            Activated feature map with ``out_channels``.
        """
        hidden = self.norm1(self.conv1(features))
        time_shift = self.time_projection(nn.functional.silu(time_embedding))
        hidden = nn.functional.silu(hidden + time_shift[:, :, None, None])
        return nn.functional.silu(self.norm2(self.conv2(hidden)))


class DownBlock(nn.Module):
    """Stride-2 convolution followed by a conditioned convolution block."""

    def __init__(
        self, in_channels: int, out_channels: int, time_dimension: int
    ) -> None:
        """Create the downsampling block.

        Args:
            in_channels: Channels before downsampling.
            out_channels: Channels after the convolution block.
            time_dimension: Width of the time embedding.
        """
        super().__init__()
        self.pool = nn.Conv2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.block = ConvBlock(in_channels, out_channels, time_dimension)

    def forward(self, features: Tensor, time_embedding: Tensor) -> Tensor:
        """Downsample and convolve.

        Args:
            features: Feature map at the finer resolution.
            time_embedding: One embedding vector per batch element.

        Returns:
            Feature map at half the spatial size.
        """
        return self.block(self.pool(features), time_embedding)


class UpBlock(nn.Module):
    """Transposed convolution, skip concatenation, and a convolution block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dimension: int,
    ) -> None:
        """Create the upsampling block.

        Args:
            in_channels: Channels of the coarse feature map.
            skip_channels: Channels of the matching encoder map.
            out_channels: Channels after the convolution block.
            time_dimension: Width of the time embedding.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.block = ConvBlock(
            in_channels + skip_channels, out_channels, time_dimension
        )

    def forward(self, features: Tensor, skip: Tensor, time_embedding: Tensor) -> Tensor:
        """Upsample, crop to the skip map, and convolve the concatenation.

        Args:
            features: Coarse feature map.
            skip: Encoder feature map at the destination resolution.
            time_embedding: One embedding vector per batch element.

        Returns:
            Feature map aligned with ``skip``.
        """
        lifted = self.up(features)
        lifted = lifted[:, :, : skip.shape[-2], : skip.shape[-1]]
        return self.block(torch.cat((lifted, skip), dim=1), time_embedding)


def sinusoidal_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    """Encode integer diffusion steps as sinusoids.

    Args:
        timesteps: Integer step per batch element.
        dimension: Even embedding width.

    Returns:
        Embedding with shape ``(batch, dimension)``.
    """
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def _pad_to_multiple(features: Tensor, multiple: int) -> tuple[Tensor, int, int]:
    """Pad the bottom and right so both spatial sizes are divisible by ``multiple``.

    Args:
        features: Feature map shaped ``(batch, channels, height, width)``.
        multiple: Required divisor of the padded spatial size.

    Returns:
        Padded map and the original height and width.
    """
    height, width = int(features.shape[-2]), int(features.shape[-1])
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return features, height, width
    return nn.functional.pad(features, (0, pad_width, 0, pad_height)), height, width


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides ``channels``.

    Args:
        channels: Channel count of the normalized map.

    Returns:
        Largest of 8, 4, 2, and 1 that divides ``channels``.
    """
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1
