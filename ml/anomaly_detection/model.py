"""LSTM Autoencoder for unsupervised anomaly detection.

Architecture (encoder → bottleneck → decoder):

    Input:  (B, W, F_ae)                   W=20, F_ae=7

    ENCODER
        LSTM(F_ae → hidden, batch_first=True)
        Take final hidden state h_N         → bottleneck: (B, hidden)

    DECODER
        Repeat h_N across W timesteps       → (B, W, hidden)
        LSTM(hidden → hidden, batch_first=True)
        Linear(hidden → F_ae) per timestep  → (B, W, F_ae)

    Loss:  MSE(output, input)   ← unsupervised; label = input sequence
    Anomaly score = per-window reconstruction MSE at inference time.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int = 7, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, F_ae) input sequence
        Returns:
            reconstruction: (B, W, F_ae) — same shape as input
        """
        _, (h_n, c_n) = self.encoder(x)
        # h_n: (1, B, hidden) — the bottleneck
        bottleneck = h_n.squeeze(0)                          # (B, hidden)
        # Repeat bottleneck as decoder input at every timestep
        decoder_input = bottleneck.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, W, hidden)
        dec_out, _ = self.decoder(decoder_input)             # (B, W, hidden)
        return self.output_layer(dec_out)                    # (B, W, F_ae)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction.

        Args:
            x: (B, W, F_ae)
        Returns:
            errors: (B,) per-window mean squared error
        """
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))  # (B,)
