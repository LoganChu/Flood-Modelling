"""Unit tests for ml/common/ shared infrastructure."""

import numpy as np
import pandas as pd
import pytest

from ml.common.features import derive_features, get_gap_mask
from ml.common.metrics import (
    constant_velocity_baseline,
    mae_2d,
    rmse_2d,
    rmse_per_horizon,
)
from ml.common.scaler import apply_scaler, fit_scaler
from ml.common.windows import chronological_split, make_windows


def _make_dummy_df(n: int = 100) -> pd.DataFrame:
    t = np.linspace(0, n - 1, n)
    dt = np.ones(n)
    if n > 50:
        dt[50] = 30.0  # inject a millis() gap at index 50
    df = pd.DataFrame(
        {
            "time_s": t,
            "dt_s": dt,
            "vx_ekf": np.sin(t * 0.1),
            "vz_ekf": np.cos(t * 0.1),
            "east_raw": np.cumsum(np.random.default_rng(0).normal(0, 0.5, n)),
            "north_raw": np.cumsum(np.random.default_rng(1).normal(0, 0.5, n)),
            "speed_ms": np.abs(np.sin(t * 0.1)) + 0.1,
            "east_smooth": np.cumsum(np.random.default_rng(2).normal(0, 0.3, n)),
            "north_smooth": np.cumsum(np.random.default_rng(3).normal(0, 0.3, n)),
        }
    )
    return df


class TestDeriveFeatures:
    def test_adds_required_columns(self):
        df = derive_features(_make_dummy_df())
        for col in ("ax_filt", "az_filt", "heading_deg", "delta_east", "delta_north"):
            assert col in df.columns

    def test_shape_preserved(self):
        df = _make_dummy_df()
        out = derive_features(df)
        assert len(out) == len(df)

    def test_does_not_mutate_input(self):
        df = _make_dummy_df()
        original_cols = list(df.columns)
        derive_features(df)
        assert list(df.columns) == original_cols

    def test_heading_range(self):
        df = derive_features(_make_dummy_df())
        assert df["heading_deg"].min() >= 0.0
        assert df["heading_deg"].max() < 360.0

    def test_gradient_same_length(self):
        df = derive_features(_make_dummy_df(50))
        assert len(df["ax_filt"]) == 50


class TestGetGapMask:
    def test_flags_large_dt(self):
        df = _make_dummy_df()
        mask = get_gap_mask(df, threshold_s=5.0)
        assert mask[50]   # the dt=30 row should be flagged
        assert not mask[49]

    def test_no_gaps_normal_data(self):
        df = _make_dummy_df()
        df["dt_s"] = 1.0   # uniform 1s steps
        mask = get_gap_mask(df, threshold_s=5.0)
        assert not mask.any()

    def test_length_matches_df(self):
        df = _make_dummy_df(80)
        assert len(get_gap_mask(df)) == 80


class TestMakeWindows:
    def test_output_shapes(self):
        feat = np.random.rand(200, 5).astype(np.float32)
        mask = np.zeros(200, dtype=bool)
        X, y = make_windows(feat, mask, window=30, horizon=10, stride=1)
        assert X.shape == (161, 30, 5)
        assert y.shape == (161, 10, 5)

    def test_excludes_gap_spanning_windows(self):
        feat = np.random.rand(100, 3).astype(np.float32)
        mask = np.zeros(100, dtype=bool)
        mask[50] = True  # gap at index 50
        X, y = make_windows(feat, mask, window=10, horizon=5, stride=1)
        # Any window starting in [50-14, 50] would span index 50 and should be excluded
        assert len(X) < 86  # fewer than the no-gap case

    def test_autoencoder_mode_y_equals_x(self):
        feat = np.random.rand(60, 4).astype(np.float32)
        mask = np.zeros(60, dtype=bool)
        X, y = make_windows(feat, mask, window=20, horizon=0)
        np.testing.assert_array_equal(X, y)

    def test_empty_result_on_all_gaps(self):
        feat = np.random.rand(50, 3).astype(np.float32)
        mask = np.ones(50, dtype=bool)  # all gaps
        X, y = make_windows(feat, mask, window=10, horizon=5)
        assert len(X) == 0


class TestScaler:
    def test_fit_apply_zero_mean(self):
        X = np.random.rand(100, 5).astype(np.float32) + 10.0
        scaler = fit_scaler(X)
        X_s = apply_scaler(X, scaler)
        np.testing.assert_allclose(X_s.mean(0), 0.0, atol=1e-5)

    def test_shape_preserved_3d(self):
        X = np.random.rand(50, 20, 7).astype(np.float32)
        scaler = fit_scaler(X)
        X_s = apply_scaler(X, scaler)
        assert X_s.shape == (50, 20, 7)


class TestMetrics:
    def test_rmse_2d_perfect(self):
        pred = np.ones((10, 2))
        true = np.ones((10, 2))
        assert rmse_2d(pred, true) == 0.0

    def test_rmse_per_horizon_shape(self):
        pred = np.random.rand(50, 10, 2)
        true = np.random.rand(50, 10, 2)
        out = rmse_per_horizon(pred, true)
        assert out.shape == (10,)

    def test_constant_velocity_baseline_shape(self):
        vx = np.ones(20)
        vz = np.ones(20)
        cv = constant_velocity_baseline(vx, vz, dt=1.0, horizon=10)
        assert cv.shape == (20, 10, 2)

    def test_constant_velocity_linear_growth(self):
        vx = np.array([1.0])
        vz = np.array([0.0])
        cv = constant_velocity_baseline(vx, vz, dt=1.0, horizon=5)
        expected_east = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(cv[0, :, 0], expected_east, rtol=1e-5)
