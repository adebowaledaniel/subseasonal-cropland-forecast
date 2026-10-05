"""Convolutional LSTM (Shi et al., 2015) encoder and decoder."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

State = Tuple[torch.Tensor, torch.Tensor]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # One convolution computes all four gates (input, forget, cell, output).
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2)
        with torch.no_grad():
            self.conv.bias.zero_()
            self.conv.bias[hidden_dim:2 * hidden_dim].fill_(1.0)  # forget gate

    def forward(self, x: torch.Tensor, state: State) -> State:
        h, c = state
        i, f, g, o = torch.split(self.conv(torch.cat([x, h], dim=1)), self.hidden_dim, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, height: int, width: int, device) -> State:
        zeros = torch.zeros(batch, self.hidden_dim, height, width, device=device)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM over inputs of shape (B, T, C, H, W)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_sizes: List[int]):
        super().__init__()
        assert len(hidden_dims) == len(kernel_sizes)
        in_dims = [input_dim] + hidden_dims[:-1]
        self.cell_list = nn.ModuleList(
            ConvLSTMCell(i, h, k) for i, h, k in zip(in_dims, hidden_dims, kernel_sizes))

    def forward(self, x: torch.Tensor, state: Optional[List[State]] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[State]]:
        """
        mask: optional (B, T) with 1 for real and 0 for padded steps. The state is
        held fixed through padded steps, so padding has no effect on the encoding
        (feeding zeros would still move the state through the gate biases).
        """
        b, t, _, height, width = x.shape
        if state is None:
            state = [cell.init_state(b, height, width, x.device) for cell in self.cell_list]

        layer_input, last_states = x, []
        for cell, (h, c) in zip(self.cell_list, state):
            outputs = []
            for step in range(t):
                h_new, c_new = cell(layer_input[:, step], (h, c))
                if mask is None:
                    h, c = h_new, c_new
                else:
                    m = mask[:, step].to(h.dtype).view(b, 1, 1, 1)
                    h = m * h_new + (1 - m) * h
                    c = m * c_new + (1 - m) * c
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append((h, c))
        return layer_input, last_states


class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_sizes: List[int],
                 dropout: float = 0.0):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dims, kernel_sizes)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        outputs, states = self.convlstm(x, mask=mask)
        return outputs, [(self.dropout(h), c) for h, c in states]


class ConvLSTMDecoder(nn.Module):
    """One decoding step: ConvLSTM conditioned on the encoder state, then a 1x1 conv."""

    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_sizes: List[int],
                 output_dim: int = 1):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dims, kernel_sizes)
        self.output_conv = nn.Conv2d(hidden_dims[-1], output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, state: List[State]) -> torch.Tensor:
        outputs, _ = self.convlstm(x, state)
        return self.output_conv(outputs[:, -1]).unsqueeze(1)
