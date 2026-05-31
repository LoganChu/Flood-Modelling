"""Unit tests for ml/trajectory_prediction/ module."""

import numpy as np
import pytest
import torch

from ml.trajectory_prediction.dataset import SlidingWindowDataset
from ml.trajectory_prediction.model import TrajectoryLSTM


class TestTrajectoryLSTM:
    def test_forward_output_shape(self):
        model = TrajectoryLSTM(input_size=5, hidden_dim=64, num_layers=2, dropout=0.2, horizon=10)
        x = torch.randn(4, 30, 5)
        out = model(x)
        assert out.shape == (4, 10, 2), f"Expected (4,10,2) got {out.shape}"

    def test_forward_single_sample(self):
        model = TrajectoryLSTM(input_size=5, hidden_dim=32, num_layers=1, dropout=0.0, horizon=5)
        x = torch.randn(1, 30, 5)
        out = model(x)
        assert out.shape == (1, 5, 2)

    def test_gradient_flows(self):
        model = TrajectoryLSTM(input_size=5, hidden_dim=16, num_layers=1, dropout=0.0, horizon=3)
        x = torch.randn(2, 10, 5)
        target = torch.zeros(2, 3, 2)
        loss = torch.nn.MSELoss()(model(x), target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_output_is_float(self):
        model = TrajectoryLSTM()
        out = model(torch.randn(2, 30, 5))
        assert out.dtype == torch.float32


class TestSlidingWindowDataset:
    def setup_method(self):
        self.X = np.random.rand(50, 30, 5).astype(np.float32)
        self.y = np.random.rand(50, 10, 2).astype(np.float32)
        self.ds = SlidingWindowDataset(self.X, self.y)

    def test_len(self):
        assert len(self.ds) == 50

    def test_item_shapes(self):
        x, y = self.ds[0]
        assert x.shape == (30, 5)
        assert y.shape == (10, 2)

    def test_item_dtype(self):
        x, y = self.ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_values_match_numpy(self):
        x, y = self.ds[5]
        np.testing.assert_allclose(x.numpy(), self.X[5], rtol=1e-6)
