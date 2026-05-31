"""PyTorch Dataset for the Seq2Seq trajectory prediction model."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Wraps pre-built (X, y) numpy window arrays as a torch Dataset.

    X shape: (N, window, F_in)
    y shape: (N, horizon, 2)  — (delta_east, delta_north) position offsets
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
