"""ConvLSTM encoder-decoder forecasters."""

from typing import List, Optional

import torch
import torch.nn as nn

from .convlstm import ConvLSTMDecoder, ConvLSTMEncoder


class ConvLSTMForecaster(nn.Module):
    """
    Forecasts the vegetation anomaly (input channel 0) `forecast_horizon` steps ahead.

    Any further input channels (concurrent or lag-shifted RZSM) are covariates. The
    decoder runs autoregressively on its own predictions, holding covariates at their
    last observed value, and is conditioned on the final encoder state at every step.
    """

    def __init__(self, input_channels: int, hidden_dims: List[int] = (64, 64),
                 kernel_sizes: List[int] = (3, 3), forecast_horizon: int = 5,
                 dropout: float = 0.2, doy_channels: int = 0):
        super().__init__()
        hidden_dims, kernel_sizes = list(hidden_dims), list(kernel_sizes)
        self.forecast_horizon = forecast_horizon
        self.doy_channels = doy_channels
        channels = input_channels + doy_channels

        self.encoder = ConvLSTMEncoder(channels, hidden_dims, kernel_sizes, dropout)
        # Single-step head. It is also built when forecast_horizon > 1 so that parameter
        # counts and seeded initialisation match the published runs.
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        if forecast_horizon > 1:
            self.decoder = ConvLSTMDecoder(channels, hidden_dims, kernel_sizes)

    def forward(self, x: torch.Tensor, doy: Optional[torch.Tensor] = None,
                future_doy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C, H, W). doy / future_doy: (B, T, 2, H, W) / (B, horizon, 2, H, W)
        sin-cos day-of-year, only used when the model was built with doy_channels > 0.
        Returns (B, horizon, 1, H, W).
        """
        encoder_input = torch.cat([x, doy], dim=2) if self.doy_channels else x
        encoded, state = self.encoder(encoder_input)

        if self.forecast_horizon == 1:
            return self.output_projection(encoded[:, -1]).unsqueeze(1)

        current = x[:, -1:, :1]
        covariates = x[:, -1:, 1:]
        predictions = []
        for step in range(self.forecast_horizon):
            parts = [current, covariates]
            if self.doy_channels:
                parts.append(future_doy[:, step:step + 1] if future_doy is not None else doy[:, -1:])
            current = self.decoder(torch.cat(parts, dim=2), state)
            predictions.append(current)
        return torch.cat(predictions, dim=1)


class DualEncoderForecaster(nn.Module):
    """
    Separate single-channel encoders for NIRv and RZSM so that each channel can have
    its own history length. Shorter histories are zero-padded at
    the oldest end and masked out of the recurrence; the two final states are summed
    per layer and decoded exactly as in ConvLSTMForecaster.
    """

    def __init__(self, hidden_dims: List[int] = (64, 64), kernel_sizes: List[int] = (3, 3),
                 forecast_horizon: int = 5, dropout: float = 0.2):
        super().__init__()
        hidden_dims, kernel_sizes = list(hidden_dims), list(kernel_sizes)
        self.forecast_horizon = forecast_horizon
        self.nirv_encoder = ConvLSTMEncoder(1, hidden_dims, kernel_sizes, dropout)
        self.rzsm_encoder = ConvLSTMEncoder(1, hidden_dims, kernel_sizes, dropout)
        self.decoder = ConvLSTMDecoder(2, hidden_dims, kernel_sizes)

    def forward(self, x: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, 2, H, W) as [NIRv, RZSM]; history_mask: (B, T, 2), 1 for real steps."""
        nirv_mask = history_mask[..., 0] if history_mask is not None else None
        rzsm_mask = history_mask[..., 1] if history_mask is not None else None
        _, nirv_state = self.nirv_encoder(x[:, :, :1], mask=nirv_mask)
        _, rzsm_state = self.rzsm_encoder(x[:, :, 1:], mask=rzsm_mask)
        state = [(hn + hr, cn + cr) for (hn, cn), (hr, cr) in zip(nirv_state, rzsm_state)]

        current, rzsm = x[:, -1:, :1], x[:, -1:, 1:]
        predictions = []
        for _ in range(self.forecast_horizon):
            current = self.decoder(torch.cat([current, rzsm], dim=2), state)
            predictions.append(current)
        return torch.cat(predictions, dim=1)
