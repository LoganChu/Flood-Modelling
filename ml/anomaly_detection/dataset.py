"""PyTorch Dataset for the LSTM Autoencoder anomaly detection model."""

import numpy as np
import torch
from torch.utils.data import Dataset


class AutoencoderWindowDataset(Dataset):
    """Wraps pre-built window arrays for unsupervised autoencoder training.

    Both X and y are the same window — the autoencoder reconstructs its input.

    X shape: (N, window, F_ae)
    """

    def __init__(self, X: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        return x, x  # target = input (unsupervised reconstruction)
