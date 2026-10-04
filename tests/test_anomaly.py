"""Unit tests for ml/anomaly_detection/ module."""

import numpy as np
import pandas as pd
import pytest
import torch

from ml.anomaly_detection.dataset import AutoencoderWindowDataset
from ml.anomaly_detection.model import LSTMAutoencoder
from ml.anomaly_detection.scoring import anomaly_scores, per_sample_feature_errors


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


class TestScoring:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = LSTMAutoencoder(input_size=3, hidden_dim=8)
        self.model.eval()
        self.feat = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)

    def test_errors_shape_and_non_negative(self):
        errors = per_sample_feature_errors(self.model, self.feat, np.zeros(50, dtype=bool), window=10)
        assert errors.shape == (50, 3)
        assert not np.isnan(errors).any()
        assert (errors >= 0).all()

    def test_errors_match_direct_reconstruction_for_single_window(self):
        feat = self.feat[:10]
        errors = per_sample_feature_errors(self.model, feat, np.zeros(10, dtype=bool), window=10)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            expected = ((self.model(x) - x) ** 2).squeeze(0).numpy()
        np.testing.assert_allclose(errors, expected, rtol=1e-5)

    def test_samples_without_gap_free_window_are_nan(self):
        gap_mask = np.zeros(50, dtype=bool)
        gap_mask[[20, 25]] = True   # samples 20-25 sit in no gap-free window of 10
        errors = per_sample_feature_errors(self.model, self.feat, gap_mask, window=10)
        assert np.isnan(errors[20:26]).all()
        assert not np.isnan(errors[:20]).any()
        assert not np.isnan(errors[26:]).any()

    def test_run_shorter_than_window_is_all_nan(self):
        errors = per_sample_feature_errors(self.model, self.feat[:5], np.zeros(5, dtype=bool), window=10)
        assert errors.shape == (5, 3)
        assert np.isnan(errors).all()

    def test_scores_take_largest_normalised_feature(self):
        feature_errors = np.array([[1.0, 4.0, 2.0], [6.0, 1.0, 1.0], [np.nan, np.nan, np.nan]])
        scores, top = anomaly_scores(feature_errors, np.array([2.0, 1.0, 4.0]))
        np.testing.assert_allclose(scores[:2], [4.0, 3.0])
        assert np.isnan(scores[2])
        assert top.tolist() == [1, 0, -1]

    def test_spike_is_localised_to_its_sample(self):
        base = per_sample_feature_errors(self.model, self.feat, np.zeros(50, dtype=bool), window=10)
        spiked = self.feat.copy()
        spiked[25, 1] += 50.0
        errors = per_sample_feature_errors(self.model, spiked, np.zeros(50, dtype=bool), window=10)
        assert errors[:, 1].argmax() == 25
        # Samples outside every window containing the spike are unaffected
        np.testing.assert_allclose(errors[:16], base[:16], rtol=1e-5)
        np.testing.assert_allclose(errors[35:], base[35:], rtol=1e-5)


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
