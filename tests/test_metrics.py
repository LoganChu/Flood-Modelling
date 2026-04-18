"""
test_metrics.py — Unit tests for metrics.py.

Test style mirrors test_data_loader.py / test_geo_converter.py:
- Class-based grouping by feature
- Synthetic data via helper functions
- Real CSV as optional integration tests (pytest.skip if absent)
- No mocking; direct instantiation + assertions
- pytest.approx() for float comparisons
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from drifter_vis.src.metrics import (
    DrifterMetrics,
    MetricsConfig,
    MetricsResult,
)

REAL_CSV = Path(__file__).resolve().parent.parent / "exts" / "drifter_vis" / "data" / "enoFeb16th.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geo(east: np.ndarray, north: np.ndarray):
    """Minimal geo-result stub with .enu_east and .enu_north."""
    return SimpleNamespace(enu_east=east, enu_north=north)


def _make_df(n: int, speed_ms: float = 0.5, ax: float = 0.1, az_raw: float = -8.28) -> pd.DataFrame:
    return pd.DataFrame({
        "speed_ms":    [speed_ms] * n,
        "AX":          [ax] * n,
        "AY":          [0.0] * n,
        "AZ":          [az_raw] * n,
        "dt_s":        [0.5] * n,
        "time_s":      np.arange(n) * 0.5,
        "segment_id":  [0] * n,
    })


def _make_ekf_df(
    n: int,
    x_filt: float = 0.0,
    y_filt: float = 0.0,
    vx: float = 0.5,
    vz: float = 0.0,
    ax_filt: float = 0.0,
    az_filt: float = 0.0,
) -> pd.DataFrame:
    """Synthetic EKF output DataFrame with constant values."""
    t = np.arange(n) * 0.5
    return pd.DataFrame({
        "x_filt":         [x_filt] * n,
        "y_filt":         [y_filt] * n,
        "vx_filt":        [vx] * n,
        "vz_filt":        [vz] * n,
        "ax_filt":        [ax_filt] * n,
        "az_filt":        [az_filt] * n,
        "cov_trace":      [1.0] * n,
        "innovation_norm":[0.1] * n,
        "drift_flag":     [False] * n,
    })


def _make_ekf_df_moving(n: int, speed: float = 0.3, dt: float = 0.5) -> pd.DataFrame:
    """Straight-line northward EKF output."""
    t = np.arange(n) * dt
    return pd.DataFrame({
        "x_filt":         np.zeros(n),
        "y_filt":         speed * t,
        "vx_filt":        np.zeros(n),
        "vz_filt":        [speed] * n,
        "ax_filt":        np.zeros(n),
        "az_filt":        np.zeros(n),
        "cov_trace":      np.ones(n),
        "innovation_norm":np.ones(n) * 0.1,
        "drift_flag":     [False] * n,
    })


# ---------------------------------------------------------------------------
# TestMetricsConfig
# ---------------------------------------------------------------------------

class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.rmse_report is True
        assert cfg.output_csv == "data/metrics.csv"
        assert cfg.curvature_min_speed_ms == pytest.approx(0.05)

    def test_from_yaml_empty_dict_uses_defaults(self):
        cfg = MetricsConfig.from_yaml({})
        assert cfg.rmse_report is True

    def test_from_yaml_none_uses_defaults(self):
        cfg = MetricsConfig.from_yaml(None)
        assert cfg.curvature_high_threshold == pytest.approx(0.5)

    def test_from_yaml_merges_curvature_section(self):
        # Caller merges metrics: and curvature: sections before passing
        d = {
            "output_csv": "data/test_metrics.csv",
            "min_speed_ms": 0.10,
            "high_curvature_threshold": 1.0,
        }
        cfg = MetricsConfig.from_yaml(d)
        assert cfg.output_csv == "data/test_metrics.csv"
        assert cfg.curvature_min_speed_ms == pytest.approx(0.10)
        assert cfg.curvature_high_threshold == pytest.approx(1.0)

    def test_from_yaml_partial_override(self):
        cfg = MetricsConfig.from_yaml({"velocity_error_report": False})
        assert cfg.velocity_error_report is False
        assert cfg.rmse_report is True   # other defaults preserved


# ---------------------------------------------------------------------------
# TestRMSEFormula
# ---------------------------------------------------------------------------

class TestRMSEFormula:
    def test_identical_paths_rmse_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        rmse = DrifterMetrics._compute_rmse(arr, arr, arr, arr)
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_known_3_4_5_triangle(self):
        # pred offset by (+3, +4) → per-point 2-D dist = 5 → RMSE = 5
        n = 10
        t = np.zeros(n)
        p = np.ones(n)
        rmse = DrifterMetrics._compute_rmse(t, t, 3 * p, 4 * p)
        assert rmse == pytest.approx(5.0, rel=1e-9)

    def test_rmse_nonnegative(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 5, 100)
        b = rng.normal(0, 5, 100)
        assert DrifterMetrics._compute_rmse(a, b, b, a) >= 0.0

    def test_rmse_symmetric(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 2, 50)
        b = rng.normal(0, 2, 50)
        r1 = DrifterMetrics._compute_rmse(a, b, b, a)
        r2 = DrifterMetrics._compute_rmse(b, a, a, b)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# TestEKFSmoothness
# ---------------------------------------------------------------------------

class TestEKFSmoothness:
    def test_perfectly_smooth_gives_near_zero(self):
        # Straight line: 3-pt MA equals the line → RMSE ≈ 0 (edge effects only)
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        rmse = DrifterMetrics._compute_ekf_smoothness(x, y)
        assert rmse < 0.01

    def test_noisy_signal_gives_nonzero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, 200)
        y = rng.normal(0, 1.0, 200)
        rmse = DrifterMetrics._compute_ekf_smoothness(x, y)
        assert rmse > 0.01

    def test_short_array_returns_zero(self):
        rmse = DrifterMetrics._compute_ekf_smoothness(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert rmse == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestCurvature
# ---------------------------------------------------------------------------

class TestCurvature:
    def test_straight_line_zero_curvature(self):
        n = 20
        vx = np.ones(n) * 1.0
        vz = np.zeros(n)
        ax = np.zeros(n)
        az = np.zeros(n)
        kappa = DrifterMetrics._compute_curvature(vx, vz, ax, az, min_speed=0.05)
        np.testing.assert_allclose(kappa, 0.0, atol=1e-12)

    def test_circular_trajectory_curvature(self):
        """
        For a circle of radius r at constant speed v:
            vx = -v sin(θ),  vz = v cos(θ)
            ax = -v²/r cos(θ), az = -v²/r sin(θ)
        κ = |vx·az − vz·ax| / (v²)^(3/2) = (v² / r) / v³ = 1/r
        """
        r = 5.0
        v = 1.0
        n = 360
        theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
        vx = -v * np.sin(theta)
        vz =  v * np.cos(theta)
        ax = -(v ** 2 / r) * np.cos(theta)
        az = -(v ** 2 / r) * np.sin(theta)

        kappa = DrifterMetrics._compute_curvature(vx, vz, ax, az, min_speed=0.05)
        assert float(np.mean(kappa)) == pytest.approx(1.0 / r, rel=0.02)

    def test_low_speed_returns_zero(self):
        n = 10
        vx = np.ones(n) * 0.01   # below min_speed=0.05
        vz = np.zeros(n)
        ax = np.ones(n) * 5.0
        az = np.zeros(n)
        kappa = DrifterMetrics._compute_curvature(vx, vz, ax, az, min_speed=0.05)
        np.testing.assert_allclose(kappa, 0.0, atol=1e-12)

    def test_output_shape_matches_input(self):
        n = 37
        vx = np.random.rand(n)
        vz = np.random.rand(n)
        ax = np.zeros(n)
        az = np.zeros(n)
        kappa = DrifterMetrics._compute_curvature(vx, vz, ax, az, min_speed=0.0)
        assert len(kappa) == n

    def test_curvature_nonnegative(self):
        rng = np.random.default_rng(7)
        vx = rng.normal(0, 1, 100)
        vz = rng.normal(0, 1, 100)
        ax = rng.normal(0, 0.5, 100)
        az = rng.normal(0, 0.5, 100)
        kappa = DrifterMetrics._compute_curvature(vx, vz, ax, az, min_speed=0.05)
        assert np.all(kappa >= 0.0)


# ---------------------------------------------------------------------------
# TestVelocityError
# ---------------------------------------------------------------------------

class TestVelocityError:
    def test_perfect_estimate_zero_mean_error(self):
        n = 10
        df = _make_df(n, speed_ms=0.5)
        # vx=0.5, vz=0 → |v_ekf|=0.5 = speed_ms → error=0
        ekf_df = _make_ekf_df(n, vx=0.5, vz=0.0)
        mean, std = DrifterMetrics._compute_velocity_error(df, ekf_df)
        assert mean == pytest.approx(0.0, abs=1e-9)

    def test_constant_offset_error(self):
        n = 10
        df = _make_df(n, speed_ms=0.5)
        # EKF speed = 1.0 m/s, GPS = 0.5 m/s → error = 0.5
        ekf_df = _make_ekf_df(n, vx=1.0, vz=0.0)
        mean, _ = DrifterMetrics._compute_velocity_error(df, ekf_df)
        assert mean == pytest.approx(0.5, abs=1e-9)

    def test_returns_tuple_of_two_floats(self):
        df = _make_df(5)
        ekf_df = _make_ekf_df(5)
        result = DrifterMetrics._compute_velocity_error(df, ekf_df)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# TestDriftDivergence
# ---------------------------------------------------------------------------

class TestDriftDivergence:
    def test_same_endpoint_zero_divergence(self):
        rec_e = np.array([0.0, 1.0, 2.0])
        rec_n = np.array([0.0, 1.0, 2.0])
        div = DrifterMetrics._compute_drift_divergence(rec_e, rec_n, rec_e, rec_n)
        assert div == pytest.approx(0.0, abs=1e-12)

    def test_3_4_5_triangle_divergence(self):
        rec_e = np.array([0.0, 0.0])
        rec_n = np.array([0.0, 0.0])
        sim_e = np.array([0.0, 3.0])
        sim_n = np.array([0.0, 4.0])
        div = DrifterMetrics._compute_drift_divergence(rec_e, rec_n, sim_e, sim_n)
        assert div == pytest.approx(5.0, rel=1e-9)

    def test_uses_last_point_only(self):
        # Even with large intermediate divergence, only final point matters
        rec_e = np.array([0.0, 100.0, 0.0])
        rec_n = np.array([0.0, 100.0, 0.0])
        sim_e = np.array([0.0, 200.0, 3.0])
        sim_n = np.array([0.0, 200.0, 4.0])
        div = DrifterMetrics._compute_drift_divergence(rec_e, rec_n, sim_e, sim_n)
        assert div == pytest.approx(5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# TestMetricsResult
# ---------------------------------------------------------------------------

class TestMetricsResult:
    def _make_result(self, n: int = 10) -> MetricsResult:
        return MetricsResult(
            rmse_raw_gps=2.5,
            rmse_ekf=0.1,
            rmse_physics=3.0,
            filter_improvement_ratio=25.0,
            velocity_error_mean=0.08,
            velocity_error_std=0.03,
            accel_residual_mean=0.05,
            accel_residual_std=0.02,
            curvature=np.linspace(0, 1.0, n),
            drift_divergence=4.5,
        )

    def test_to_dict_keys_present(self):
        d = self._make_result().to_dict()
        expected = [
            "rmse_raw_gps_m", "rmse_ekf_m", "rmse_physics_m",
            "filter_improvement_ratio", "velocity_error_mean_ms",
            "velocity_error_std_ms", "accel_residual_mean_ms2",
            "accel_residual_std_ms2", "curvature_mean_1pm",
            "curvature_std_1pm", "curvature_high_fraction",
            "drift_divergence_m",
        ]
        for key in expected:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["rmse_raw_gps_m"] == pytest.approx(2.5)
        assert d["velocity_error_mean_ms"] == pytest.approx(0.08)
        assert d["drift_divergence_m"] == pytest.approx(4.5)

    def test_none_physics_fields_in_dict(self):
        r = MetricsResult(
            rmse_raw_gps=1.0, rmse_ekf=0.1, rmse_physics=None,
            filter_improvement_ratio=None,
            velocity_error_mean=0.1, velocity_error_std=0.05,
            accel_residual_mean=0.02, accel_residual_std=0.01,
            curvature=np.zeros(5), drift_divergence=None,
        )
        d = r.to_dict()
        assert d["rmse_physics_m"] is None
        assert d["drift_divergence_m"] is None


# ---------------------------------------------------------------------------
# TestCSVExport
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_csv_written_and_readable(self, tmp_path):
        result = MetricsResult(
            rmse_raw_gps=2.5, rmse_ekf=0.1, rmse_physics=3.0,
            filter_improvement_ratio=25.0,
            velocity_error_mean=0.08, velocity_error_std=0.03,
            accel_residual_mean=0.05, accel_residual_std=0.02,
            curvature=np.ones(10) * 0.2,
            drift_divergence=4.5,
        )
        out = tmp_path / "metrics.csv"
        cfg = MetricsConfig(output_csv=str(out))
        m = DrifterMetrics(cfg)
        written = m.export_csv(result, str(out))
        assert written.exists()
        df = pd.read_csv(written)
        assert len(df) == 1

    def test_csv_contains_expected_columns(self, tmp_path):
        result = MetricsResult(
            rmse_raw_gps=1.0, rmse_ekf=0.05, rmse_physics=None,
            filter_improvement_ratio=20.0,
            velocity_error_mean=0.1, velocity_error_std=0.05,
            accel_residual_mean=0.02, accel_residual_std=0.01,
            curvature=np.zeros(5), drift_divergence=None,
        )
        out = tmp_path / "m.csv"
        DrifterMetrics().export_csv(result, str(out))
        cols = pd.read_csv(out).columns.tolist()
        assert "rmse_raw_gps_m" in cols
        assert "rmse_ekf_m" in cols
        assert "velocity_error_mean_ms" in cols

    def test_csv_parent_dirs_created(self, tmp_path):
        result = MetricsResult(
            rmse_raw_gps=1.0, rmse_ekf=0.1, rmse_physics=None,
            filter_improvement_ratio=10.0,
            velocity_error_mean=0.1, velocity_error_std=0.05,
            accel_residual_mean=0.02, accel_residual_std=0.01,
            curvature=np.zeros(3), drift_divergence=None,
        )
        deep = tmp_path / "deep" / "nested" / "metrics.csv"
        DrifterMetrics().export_csv(result, str(deep))
        assert deep.exists()


# ---------------------------------------------------------------------------
# TestFullCompute
# ---------------------------------------------------------------------------

class TestFullCompute:
    def test_returns_metrics_result_type(self):
        n = 20
        east = np.zeros(n)
        north = np.linspace(0, 5, n)
        geo = _make_geo(east, north)
        df = _make_df(n)
        ekf_df = _make_ekf_df_moving(n)
        result = DrifterMetrics().compute(df, geo, ekf_df)
        assert isinstance(result, MetricsResult)

    def test_no_physics_gives_none_fields(self):
        n = 15
        geo = _make_geo(np.zeros(n), np.linspace(0, 3, n))
        df = _make_df(n)
        ekf_df = _make_ekf_df_moving(n)
        result = DrifterMetrics().compute(df, geo, ekf_df)
        assert result.rmse_physics is None
        assert result.drift_divergence is None
        assert result.filter_improvement_ratio is not None

    def test_with_physics_populates_fields(self):
        n = 15
        geo = _make_geo(np.zeros(n), np.linspace(0, 3, n))
        df = _make_df(n)
        ekf_df = _make_ekf_df_moving(n)
        sim_e = np.ones(n) * 2.0
        sim_n = np.linspace(0, 3, n) + 1.0
        result = DrifterMetrics().compute(df, geo, ekf_df, sim_east=sim_e, sim_north=sim_n)
        assert result.rmse_physics is not None
        assert result.drift_divergence is not None

    def test_curvature_array_length_matches_data(self):
        n = 30
        geo = _make_geo(np.zeros(n), np.linspace(0, 6, n))
        df = _make_df(n)
        ekf_df = _make_ekf_df_moving(n)
        result = DrifterMetrics().compute(df, geo, ekf_df)
        assert len(result.curvature) == n

    def test_filter_improvement_ratio_is_positive(self):
        n = 50
        rng = np.random.default_rng(123)
        east_noise = rng.normal(0, 2.0, n)
        north_true = np.linspace(0, 10, n)
        geo = _make_geo(east_noise, north_true + rng.normal(0, 2.0, n))
        df = _make_df(n)
        # EKF path is smooth (close to true north)
        ekf_df = pd.DataFrame({
            "x_filt":         np.zeros(n),
            "y_filt":         north_true,
            "vx_filt":        np.zeros(n),
            "vz_filt":        np.ones(n) * 0.4,
            "ax_filt":        np.zeros(n),
            "az_filt":        np.zeros(n),
            "cov_trace":      np.ones(n),
            "innovation_norm":np.ones(n) * 0.1,
            "drift_flag":     [False] * n,
        })
        result = DrifterMetrics().compute(df, geo, ekf_df)
        assert result.filter_improvement_ratio > 0.0

    def test_rmse_raw_nonzero_with_noise(self):
        """GPS with noise should give nonzero RMSE vs clean EKF."""
        n = 50
        rng = np.random.default_rng(42)
        ekf_east = np.zeros(n)
        ekf_north = np.linspace(0, 10, n)
        noisy_east = ekf_east + rng.normal(0, 2.0, n)
        noisy_north = ekf_north + rng.normal(0, 2.0, n)
        geo = _make_geo(noisy_east, noisy_north)
        df = _make_df(n)
        ekf_df = pd.DataFrame({
            "x_filt":         ekf_east,
            "y_filt":         ekf_north,
            "vx_filt":        np.zeros(n),
            "vz_filt":        np.ones(n) * 0.4,
            "ax_filt":        np.zeros(n),
            "az_filt":        np.zeros(n),
            "cov_trace":      np.ones(n),
            "innovation_norm":np.ones(n) * 0.1,
            "drift_flag":     [False] * n,
        })
        result = DrifterMetrics().compute(df, geo, ekf_df)
        assert result.rmse_raw_gps > 0.5   # GPS noise should give >0.5 m RMSE


# ---------------------------------------------------------------------------
# TestRealCSVIntegration
# ---------------------------------------------------------------------------

class TestRealCSVIntegration:
    """End-to-end integration test with real data (skipped if CSV absent)."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        if not REAL_CSV.exists():
            pytest.skip("Real CSV not available")

        from drifter_vis.src.data_loader import DrifterDataLoader
        from drifter_vis.src.geo_converter import GeoConverter
        from drifter_vis.src.state_estimator import StateEstimator
        from drifter_vis.src.physics_validator import PhysicsValidator

        df = DrifterDataLoader().load(REAL_CSV)
        geo = GeoConverter().convert(df)
        ekf_df = StateEstimator().run(df, geo.enu_east, geo.enu_north)
        validator = PhysicsValidator()
        sim_east, sim_north = validator.simulate(df, geo.enu_east, geo.enu_north)
        result = DrifterMetrics().compute(
            df, geo, ekf_df, sim_east=sim_east, sim_north=sim_north
        )
        return df, geo, ekf_df, sim_east, sim_north, result

    def test_result_is_metrics_result(self, pipeline_result):
        *_, result = pipeline_result
        assert isinstance(result, MetricsResult)

    def test_rmse_raw_gps_in_expected_range(self, pipeline_result):
        *_, result = pipeline_result
        # GPS noise floor ~2-5 m CEP; RMSE vs EKF should be in this range
        assert 0.0 < result.rmse_raw_gps < 20.0

    def test_filter_improvement_ratio_positive(self, pipeline_result):
        *_, result = pipeline_result
        assert result.filter_improvement_ratio > 0.0

    def test_velocity_error_reasonable(self, pipeline_result):
        *_, result = pipeline_result
        # EKF vs GPS speed error should be < 2 m/s on a slow river
        assert result.velocity_error_mean < 2.0

    def test_physics_fields_populated(self, pipeline_result):
        *_, result = pipeline_result
        assert result.rmse_physics is not None
        assert result.drift_divergence is not None

    def test_curvature_reasonable(self, pipeline_result):
        *_, result = pipeline_result
        # Mean curvature < 2 1/m for a gentle river bend
        assert float(np.mean(result.curvature)) < 2.0

    def test_csv_export_works(self, pipeline_result, tmp_path):
        *_, result = pipeline_result
        out = tmp_path / "real_metrics.csv"
        written = DrifterMetrics().export_csv(result, str(out))
        assert written.exists()
        df = pd.read_csv(written)
        assert len(df) == 1
        assert "rmse_raw_gps_m" in df.columns
