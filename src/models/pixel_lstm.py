"""Non-spatial LSTM baseline: every pixel's time series is modelled independently."""

import torch
import torch.nn as nn


class PixelLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 64, num_layers: int = 2,
                 forecast_horizon: int = 5, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(input_channels, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> (B, horizon, 1, H, W)."""
        b, t, c, h, w = x.shape
        sequences = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        out, _ = self.lstm(sequences)
        y = self.output_projection(out[:, -1])
        return y.reshape(b, h, w, self.forecast_horizon).permute(0, 3, 1, 2).unsqueeze(2)
