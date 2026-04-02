"""
test_state_estimator.py — Unit tests for state_estimator.py (EKF).

Test style mirrors test_data_loader.py / test_geo_converter.py:
- Class-based grouping by feature
- Synthetic DataFrames via helper functions
- Real CSV as optional integration tests (pytest.skip if absent)
- No mocking; direct instantiation + assertions
- pytest.approx() for float comparisons
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# Make the extension root importable without Isaac Sim on sys.path
_EXT_ROOT = Path(__file__).resolve().parent.parent
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from duke.flood_modelling.drifter_vis.state_estimator import (
    EKFConfig,
    StateEstimator,
    _EKFState,
)

REAL_CSV = Path(__file__).resolve().parents[4] / "data" / "enoFeb16th.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 10,
    speed_ms: float = 0.5,
    ax: float = 0.0,
    az_raw: float = -8.28,   # raw AZ (includes gravity)
    dt: float = 0.5,
) -> pd.DataFrame:
    """Synthetic DataFrame that satisfies StateEstimator.run() column contract."""
    t = np.arange(n) * dt
    return pd.DataFrame({
        "Lat":         [36.079] * n,
        "Lon":         [-79.008] * n,
        "GPS_Alt":     [130.0] * n,
        "time_s":      t,
        "dt_s":        [dt] * n,
        "speed_ms":    [speed_ms] * n,
        "AX":          [ax] * n,
        "AY":          [0.0] * n,
        "AZ":          [az_raw] * n,
        "GX":          [0.0] * n,
        "GY":          [0.0] * n,
        "GZ":          [0.0] * n,
        "heading_deg": [0.0] * n,
        "segment_id":  [0] * n,
    })


def _make_enu_straight(
    n: int = 10,
    drift_mps: float = 0.3,
    dt: float = 0.5,
    noise_std: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Straight-line northward drift at drift_mps m/s with Gaussian GPS noise.
    Returns (enu_east, enu_north).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    true_east  = np.zeros(n)
    true_north = drift_mps * t
    east  = true_east  + rng.normal(0, noise_std, n)
    north = true_north + rng.normal(0, noise_std, n)
    return east, north


# ---------------------------------------------------------------------------
# TestEKFConfig
# ---------------------------------------------------------------------------

class TestEKFConfig:
    def test_defaults_are_sensible(self):
        cfg = EKFConfig()
        assert cfg.process_noise_pos == pytest.approx(0.01)
        assert cfg.process_noise_vel == pytest.approx(0.10)
        assert cfg.process_noise_acc == pytest.approx(1.00)
        assert cfg.measurement_noise_gps_pos == pytest.approx(4.0)
        assert cfg.measurement_noise_gps_vel == pytest.approx(0.05)
        assert cfg.mahalanobis_gate == pytest.approx(3.0)

    def test_from_yaml_overrides_one_field(self):
        cfg = EKFConfig.from_yaml({"mahalanobis_gate": 5.0})
        assert cfg.mahalanobis_gate == pytest.approx(5.0)
        # Other fields stay at defaults
        assert cfg.process_noise_pos == pytest.approx(0.01)

    def test_from_yaml_missing_keys_use_defaults(self):
        cfg = EKFConfig.from_yaml({})
        assert cfg.mahalanobis_gate == pytest.approx(EKFConfig.mahalanobis_gate)

    def test_from_yaml_none_uses_defaults(self):
        cfg = EKFConfig.from_yaml(None)
        assert cfg.process_noise_vel == pytest.approx(EKFConfig.process_noise_vel)

    def test_from_yaml_all_fields(self):
        d = {
            "process_noise_pos": 0.5,
            "process_noise_vel": 0.6,
            "process_noise_acc": 0.7,
            "measurement_noise_gps_pos": 9.0,
            "measurement_noise_gps_vel": 0.1,
            "mahalanobis_gate": 2.0,
        }
        cfg = EKFConfig.from_yaml(d)
        assert cfg.process_noise_pos == pytest.approx(0.5)
        assert cfg.measurement_noise_gps_pos == pytest.approx(9.0)
        assert cfg.mahalanobis_gate == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TestPredictStep
# ---------------------------------------------------------------------------

class TestPredictStep:
    """Direct tests of the static _predict method."""

    def _state_zero(self) -> _EKFState:
        return _EKFState(x=np.zeros(6), P=np.eye(6))

    def _state_with_velocity(self, vx: float = 1.0, vz: float = 0.0) -> _EKFState:
        x = np.zeros(6)
        x[2] = vx
        x[3] = vz
        return _EKFState(x=x, P=np.eye(6))

    def test_position_advances_with_velocity(self):
        state = self._state_with_velocity(vx=1.0, vz=0.0)
        Q = np.eye(6) * 0.0
        new = StateEstimator._predict(state, dt=0.5, ax=0.0, az=0.0, Q=Q)
        # east += vx * dt = 0.5 m
        assert new.x[0] == pytest.approx(0.5, abs=1e-9)
        assert new.x[1] == pytest.approx(0.0, abs=1e-9)

    def test_covariance_grows_without_update(self):
        state = self._state_zero()
        Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])
        trace_before = np.trace(state.P)
        new = StateEstimator._predict(state, dt=0.5, ax=0.0, az=0.0, Q=Q)
        assert np.trace(new.P) > trace_before

    def test_accel_input_changes_velocity(self):
        state = self._state_zero()
        Q = np.zeros((6, 6))
        # ax=2.0, dt=0.5 → velocity change ≈ ax*dt = 1.0 m/s
        new = StateEstimator._predict(state, dt=0.5, ax=2.0, az=0.0, Q=Q)
        assert new.x[2] == pytest.approx(1.0, abs=1e-9)   # vx

    def test_north_velocity_drives_north_position(self):
        state = self._state_with_velocity(vx=0.0, vz=0.5)
        Q = np.zeros((6, 6))
        new = StateEstimator._predict(state, dt=1.0, ax=0.0, az=0.0, Q=Q)
        assert new.x[1] == pytest.approx(0.5, abs=1e-9)   # north += vz * dt

    def test_imu_accel_overrides_previous_acc_slots(self):
        state = self._state_zero()
        state.x[4] = 99.0   # stale acceleration
        Q = np.zeros((6, 6))
        new = StateEstimator._predict(state, dt=0.5, ax=1.0, az=2.0, Q=Q)
        # The stale 99 should not propagate; vx should = 1.0 * 0.5 = 0.5
        assert new.x[2] == pytest.approx(0.5, abs=1e-9)

    def test_covariance_is_symmetric_after_predict(self):
        state = _EKFState(x=np.zeros(6), P=np.eye(6))
        Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])
        new = StateEstimator._predict(state, dt=0.5, ax=0.1, az=-0.1, Q=Q)
        assert np.allclose(new.P, new.P.T, atol=1e-12)

    def test_covariance_positive_semidefinite_after_predict(self):
        state = _EKFState(x=np.zeros(6), P=np.eye(6))
        Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])
        new = StateEstimator._predict(state, dt=0.5, ax=0.0, az=0.0, Q=Q)
        eigenvalues = np.linalg.eigvalsh(new.P)
        assert np.all(eigenvalues >= -1e-10)


# ---------------------------------------------------------------------------
# TestUpdateStep
# ---------------------------------------------------------------------------

class TestUpdateStep:
    """Direct tests of the static _update method."""

    def _state_at_origin(self) -> _EKFState:
        x = np.zeros(6)
        x[2] = 0.3   # small vx so speed row of H is non-zero
        return _EKFState(x=x, P=np.eye(6) * 4.0)

    def test_position_corrects_toward_measurement(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([5.0, 0.0, 0.3])   # east measurement at 5 m
        new_state, _, gated = StateEstimator._update(state, z, R, gate_sq=100.0)
        assert not gated
        # East should move toward 5.0
        assert new_state.x[0] > state.x[0]
        assert new_state.x[0] < 5.0

    def test_covariance_decreases_after_update(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([0.0, 0.0, 0.3])
        trace_before = np.trace(state.P)
        new_state, _, gated = StateEstimator._update(state, z, R, gate_sq=100.0)
        assert not gated
        assert np.trace(new_state.P) < trace_before

    def test_mahalanobis_gate_rejects_outlier(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        # Wild GPS measurement far from predicted state
        z = np.array([1000.0, 1000.0, 99.0])
        _, _, gated = StateEstimator._update(state, z, R, gate_sq=1.0)
        assert gated

    def test_mahalanobis_gate_passes_inlier(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        # Measurement close to predicted state
        z = np.array([0.1, 0.1, 0.3])
        _, _, gated = StateEstimator._update(state, z, R, gate_sq=100.0)
        assert not gated

    def test_state_unchanged_when_gated(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([1000.0, 1000.0, 99.0])
        new_state, _, gated = StateEstimator._update(state, z, R, gate_sq=1.0)
        assert gated
        np.testing.assert_array_equal(new_state.x, state.x)

    def test_covariance_symmetric_after_update(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([0.5, 0.5, 0.3])
        new_state, _, _ = StateEstimator._update(state, z, R, gate_sq=100.0)
        assert np.allclose(new_state.P, new_state.P.T, atol=1e-10)

    def test_covariance_positive_semidefinite_after_update(self):
        state = self._state_at_origin()
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([0.5, 0.5, 0.3])
        new_state, _, _ = StateEstimator._update(state, z, R, gate_sq=100.0)
        eigenvalues = np.linalg.eigvalsh(new_state.P)
        assert np.all(eigenvalues >= -1e-9)

    def test_near_zero_speed_does_not_crash(self):
        state = _EKFState(x=np.zeros(6), P=np.eye(6))  # vx=vz=0
        R = np.diag([4.0, 4.0, 0.05])
        z = np.array([0.0, 0.0, 0.0])
        # Should complete without ZeroDivisionError
        new_state, inn_norm, gated = StateEstimator._update(state, z, R, gate_sq=100.0)
        assert isinstance(inn_norm, float)


# ---------------------------------------------------------------------------
# TestFullRun
# ---------------------------------------------------------------------------

class TestFullRun:
    """Tests of StateEstimator.run() on synthetic data."""

    _EXPECTED_COLS = [
        "x_filt", "y_filt", "vx_filt", "vz_filt",
        "ax_filt", "az_filt", "cov_trace", "innovation_norm", "drift_flag",
    ]

    def test_output_columns_present(self):
        df = _make_df(n=5)
        east, north = _make_enu_straight(n=5)
        estimator = StateEstimator()
        ekf_df = estimator.run(df, east, north)
        for col in self._EXPECTED_COLS:
            assert col in ekf_df.columns, f"Missing column: {col}"

    def test_output_length_matches_input(self):
        n = 20
        df = _make_df(n=n)
        east, north = _make_enu_straight(n=n)
        ekf_df = StateEstimator().run(df, east, north)
        assert len(ekf_df) == n

    def test_original_columns_preserved(self):
        df = _make_df(n=10)
        east, north = _make_enu_straight(n=10)
        ekf_df = StateEstimator().run(df, east, north)
        for col in df.columns:
            assert col in ekf_df.columns

    def test_covariance_not_growing_unbounded(self):
        """On a straight noisy path the filter should converge."""
        n = 100
        df = _make_df(n=n)
        east, north = _make_enu_straight(n=n, drift_mps=0.3, noise_std=2.0)
        estimator = StateEstimator()
        ekf_df = estimator.run(df, east, north)
        # Covariance should not diverge — final trace < 1000 is a loose bound
        assert float(ekf_df["cov_trace"].iloc[-1]) < 1000.0

    def test_filtered_positions_smoother_than_raw(self):
        """EKF output should have smaller step-to-step variance than raw GPS."""
        n = 100
        df = _make_df(n=n)
        east, north = _make_enu_straight(n=n, drift_mps=0.3, noise_std=3.0)
        ekf_df = StateEstimator().run(df, east, north)

        raw_var = float(np.var(np.diff(east)) + np.var(np.diff(north)))
        ekf_var = float(
            np.var(np.diff(ekf_df["x_filt"].values)) +
            np.var(np.diff(ekf_df["y_filt"].values))
        )
        assert ekf_var < raw_var, (
            f"EKF output ({ekf_var:.3f}) should be smoother than raw GPS ({raw_var:.3f})"
        )

    def test_innovation_sequence_stored_on_estimator(self):
        df = _make_df(n=10)
        east, north = _make_enu_straight(n=10)
        estimator = StateEstimator()
        estimator.run(df, east, north)
        assert isinstance(estimator.innovation_sequence, np.ndarray)
        assert len(estimator.innovation_sequence) == 10

    def test_covariance_trace_stored_on_estimator(self):
        df = _make_df(n=10)
        east, north = _make_enu_straight(n=10)
        estimator = StateEstimator()
        estimator.run(df, east, north)
        assert isinstance(estimator.covariance_trace, np.ndarray)
        assert len(estimator.covariance_trace) == 10

    def test_drift_flag_is_bool(self):
        df = _make_df(n=10)
        east, north = _make_enu_straight(n=10)
        ekf_df = StateEstimator().run(df, east, north)
        assert ekf_df["drift_flag"].dtype in (bool, object, np.dtype("bool"))
        # All values are True or False
        assert set(ekf_df["drift_flag"].unique()).issubset({True, False})

    def test_no_nans_in_output_columns(self):
        n = 50
        df = _make_df(n=n)
        east, north = _make_enu_straight(n=n)
        ekf_df = StateEstimator().run(df, east, north)
        for col in self._EXPECTED_COLS:
            assert not ekf_df[col].isnull().any(), f"NaN in column: {col}"

    def test_first_filtered_position_matches_first_gps(self):
        """Initial state is seeded from enu_east[0], enu_north[0]."""
        east, north = _make_enu_straight(n=5, noise_std=0.0)
        df = _make_df(n=5)
        ekf_df = StateEstimator().run(df, east, north)
        assert float(ekf_df["x_filt"].iloc[0]) == pytest.approx(float(east[0]), abs=1e-9)
        assert float(ekf_df["y_filt"].iloc[0]) == pytest.approx(float(north[0]), abs=1e-9)


# ---------------------------------------------------------------------------
# TestDriftDetection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """Tests of StateEstimator._detect_drift."""

    def _df_with_speed(self, n: int, speed: float = 0.5) -> pd.DataFrame:
        return _make_df(n=n, speed_ms=speed)

    def test_straight_path_mostly_no_drift(self):
        """Uniform northward velocity → heading aligned with bulk flow → no drift."""
        n = 20
        vx = np.zeros(n)
        vz = np.ones(n) * 0.5
        df = self._df_with_speed(n)
        flags = StateEstimator._detect_drift(df, vx, vz, threshold_deg=30.0)
        assert flags.sum() == 0

    def test_90_degree_turn_flags_second_half(self):
        """
        First half: heading north (vx=0, vz=1).
        Second half: heading east (vx=1, vz=0) — 90° from median.
        Second half should be flagged.
        """
        n = 40
        vx = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        vz = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        df = self._df_with_speed(n, speed=1.0)
        flags = StateEstimator._detect_drift(df, vx, vz, threshold_deg=30.0)
        # At least some of the second half should be flagged
        second_half_flags = flags[n // 2:]
        assert second_half_flags.sum() > 0

    def test_near_zero_speed_not_flagged(self):
        """Samples with speed < min_speed_ms should never be drift-flagged."""
        n = 10
        vx = np.ones(n) * 10.0   # huge heading change...
        vz = np.zeros(n)
        df = _make_df(n=n, speed_ms=0.01)   # but speed is tiny
        flags = StateEstimator._detect_drift(df, vx, vz, threshold_deg=5.0, min_speed_ms=0.05)
        assert flags.sum() == 0

    def test_returns_bool_array_of_correct_length(self):
        n = 15
        vx = np.zeros(n)
        vz = np.ones(n) * 0.3
        df = self._df_with_speed(n)
        flags = StateEstimator._detect_drift(df, vx, vz)
        assert len(flags) == n
        assert flags.dtype == bool


# ---------------------------------------------------------------------------
# TestGravityBias
# ---------------------------------------------------------------------------

class TestGravityBias:
    def test_median_of_az_column(self):
        df = _make_df(n=10, az_raw=-8.28)
        bias = StateEstimator._compute_gravity_bias(df)
        assert bias == pytest.approx(-8.28, abs=0.01)

    def test_debias_removes_gravity_from_az(self):
        df = _make_df(n=5, az_raw=-8.28)
        _, az_d = StateEstimator._debias_accel(df, g_bias_z=-8.28)
        np.testing.assert_allclose(az_d, 0.0, atol=1e-9)

    def test_debias_clips_spikes(self):
        df = _make_df(n=3, ax=200.0, az_raw=0.0)
        ax_d, _ = StateEstimator._debias_accel(df, g_bias_z=0.0, clip=30.0)
        assert np.all(np.abs(ax_d) <= 30.0)


# ---------------------------------------------------------------------------
# TestRealCSVIntegration
# ---------------------------------------------------------------------------

class TestRealCSVIntegration:
    """End-to-end test using the real Eno River CSV (skipped if absent)."""

    @pytest.fixture(scope="class")
    def ekf_result(self):
        if not REAL_CSV.exists():
            pytest.skip("Real CSV not available")

        # Import the full pipeline (data_loader, geo_converter) from the same package
        from duke.flood_modelling.drifter_vis.data_loader import DrifterDataLoader
        from duke.flood_modelling.drifter_vis.geo_converter import GeoConverter

        loader = DrifterDataLoader()
        df = loader.load(REAL_CSV)
        geo = GeoConverter().convert(df)
        estimator = StateEstimator()
        ekf_df = estimator.run(df, geo.enu_east, geo.enu_north)
        return df, ekf_df, estimator

    def test_no_nans_in_filtered_positions(self, ekf_result):
        _, ekf_df, _ = ekf_result
        assert not ekf_df["x_filt"].isnull().any()
        assert not ekf_df["y_filt"].isnull().any()

    def test_covariance_bounded_on_real_data(self, ekf_result):
        """Covariance should not diverge unboundedly within each segment.
        The filter resets at segment boundaries so max trace per segment
        (not globally) should remain finite.  We verify the median trace
        is well below a loose bound — divergence would push medians >> 1e6.
        """
        _, ekf_df, _ = ekf_result
        median_trace = float(ekf_df["cov_trace"].median())
        assert median_trace < 1e6

    def test_filter_reduces_noise_on_real_data(self, ekf_result):
        """EKF should reduce step-to-step variance within each segment.
        We evaluate noise reduction per segment to avoid the large
        inter-segment jumps at millis() resets contaminating the metric.
        """
        df, ekf_df, _ = ekf_result
        from duke.flood_modelling.drifter_vis.geo_converter import GeoConverter
        from duke.flood_modelling.drifter_vis.data_loader import DrifterDataLoader
        loader = DrifterDataLoader()
        df2 = loader.load(REAL_CSV)
        geo = GeoConverter().convert(df2)

        seg_ids = df2["segment_id"].unique()
        raw_vars, ekf_vars = [], []
        for seg in seg_ids:
            mask = df2["segment_id"].values == seg
            if mask.sum() < 3:
                continue
            raw_vars.append(
                float(np.var(np.diff(geo.enu_east[mask]))) +
                float(np.var(np.diff(geo.enu_north[mask])))
            )
            ekf_vars.append(
                float(np.var(np.diff(ekf_df.loc[mask, "x_filt"].values))) +
                float(np.var(np.diff(ekf_df.loc[mask, "y_filt"].values)))
            )

        assert len(raw_vars) > 0, "No segments found"
        # For at least one segment, the EKF should smooth noise
        assert any(e < r for e, r in zip(ekf_vars, raw_vars)), (
            f"EKF did not reduce noise in any segment: "
            f"raw={raw_vars}  ekf={ekf_vars}"
        )

    def test_output_row_count_matches_input(self, ekf_result):
        df, ekf_df, _ = ekf_result
        assert len(ekf_df) == len(df)

    def test_drift_flag_fraction_reasonable(self, ekf_result):
        _, ekf_df, _ = ekf_result
        frac = float(ekf_df["drift_flag"].sum()) / len(ekf_df)
        # The Eno River meanders, so some drift is expected.
        # Verify the flag is not set on ALL samples (filter is not degenerate).
        assert frac < 0.65
