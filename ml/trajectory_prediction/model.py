"""Seq2Seq LSTM encoder + MLP decoder for trajectory prediction.

Architecture:
    Input:  (B, W, F_in)          W=30, F_in=5
    LSTM L1: hidden=64            → (B, W, 64)
    Dropout(0.2)
    LSTM L2: hidden=64            → take final h_N: (B, 64)
    Linear(64, 64) → ReLU
    Linear(64, H*2)               H=10
    Reshape → (B, H, 2)           (delta_east, delta_north) offsets
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 10,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, F_in)
        Returns:
            offsets: (B, H, 2) — predicted (delta_east, delta_north) at each horizon step
        """
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]              # take top layer: (B, hidden)
        out = self.head(h_last)       # (B, H*2)
        return out.view(-1, self.horizon, 2)
