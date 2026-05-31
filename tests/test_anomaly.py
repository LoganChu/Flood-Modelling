"""Unit tests for ml/anomaly_detection/ module."""

import numpy as np
import pandas as pd
import pytest
import torch

from ml.anomaly_detection.dataset import AutoencoderWindowDataset
from ml.anomaly_detection.model import LSTMAutoencoder


class TestLSTMAutoencoder:
    def test_forward_output_shape(self):
        model = LSTMAutoencoder(input_size=7, hidden_dim=32)
        x = torch.randn(4, 20, 7)
        out = model(x)
        assert out.shape == (4, 20, 7), f"Expected (4,20,7) got {out.shape}"

    def test_reconstruction_error_shape(self):
        model = LSTMAutoencoder(input_size=7, hidden_dim=32)
        x = torch.randn(8, 20, 7)
        errors = model.reconstruction_error(x)
        assert errors.shape == (8,)

    def test_reconstruction_error_non_negative(self):
        model = LSTMAutoencoder(input_size=7, hidden_dim=32)
        x = torch.randn(4, 20, 7)
        errors = model.reconstruction_error(x)
        assert (errors >= 0).all()

    def test_gradient_flows(self):
        model = LSTMAutoencoder(input_size=7, hidden_dim=16)
        x = torch.randn(2, 10, 7)
        loss = torch.nn.MSELoss()(model(x), x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_output_dtype(self):
        model = LSTMAutoencoder()
        out = model(torch.randn(2, 20, 7))
        assert out.dtype == torch.float32

    def test_single_sample(self):
        model = LSTMAutoencoder(input_size=3, hidden_dim=8)
        x = torch.randn(1, 5, 3)
        out = model(x)
        assert out.shape == (1, 5, 3)


class TestAutoencoderWindowDataset:
    def setup_method(self):
        self.X = np.random.rand(30, 20, 7).astype(np.float32)
        self.ds = AutoencoderWindowDataset(self.X)

    def test_len(self):
        assert len(self.ds) == 30

    def test_item_shape(self):
        x, y = self.ds[0]
        assert x.shape == (20, 7)
        assert y.shape == (20, 7)

    def test_target_equals_input(self):
        x, y = self.ds[7]
        torch.testing.assert_close(x, y)

    def test_dtype(self):
        x, y = self.ds[0]
        assert x.dtype == torch.float32


class TestScoreRunSynthetic:
    """Smoke test: score_run should run end-to-end on a synthetic CSV
    when the model checkpoint exists."""

    def test_score_run_requires_window_rows(self, tmp_path):
        from ml.anomaly_detection.score_run import score_csv

        n = 5   # fewer rows than window (20) → should raise
        df = pd.DataFrame(
            {
                "time_s": np.arange(n, dtype=float),
                "dt_s": np.ones(n),
                "vx_ekf": np.zeros(n),
                "vz_ekf": np.zeros(n),
                "east_raw": np.zeros(n),
                "north_raw": np.zeros(n),
                "speed_ms": np.zeros(n),
                "east_smooth": np.zeros(n),
                "north_smooth": np.zeros(n),
            }
        )
        csv_path = tmp_path / "tiny.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises((ValueError, FileNotFoundError)):
            score_csv(csv_path)
