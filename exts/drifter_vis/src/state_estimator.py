"""
state_estimator.py — EKF-based trajectory reconstruction and state estimation.

Implements a discrete-time Extended Kalman Filter (EKF) over a 6-D state vector:

    x = [east, north, vx, vz, ax, az]

where (east, north) are ENU positions in metres, (vx, vz) are the corresponding
velocities, and (ax, az) are accelerations.  The naming follows the existing
geo_converter.py convention: east→+X_ENU, north→+Z_ENU.

Process model
-------------
Constant-velocity kinematics (acceleration treated as zero-mean process noise):

    east(k+1)  = east(k)  + vx(k)·dt
    north(k+1) = north(k) + vz(k)·dt
    vx(k+1)    = vx(k)                 (random walk)
    vz(k+1)    = vz(k)
    ax(k+1)    = 0                     (estimated but not used as driver)
    az(k+1)    = 0

The IMU is used for gravity bias estimation (pre-processing) so that
attitude and debiased accelerations are available for higher-level analysis.
Raw buoy oscillations (±10 m/s² vertical + horizontal wave forces) make the
IMU unsuitable as a direct translational process driver for a floating
platform; Q absorbs the velocity uncertainty instead.

Measurement model
-----------------
GPS provides:
    z = [east_gps, north_gps, speed_gps]

The first two rows of H are linear.  The speed row is linearised around the
predicted velocity:
    H[2] = [0, 0, vx/|v|, vz/|v|, 0, 0]   (zero when |v| < 1e-4 m/s)

Outlier rejection
-----------------
Each GPS update is gated with the Mahalanobis distance:
    d² = innovation^T · S^{-1} · innovation
Reject the update (keep predicted state) when d² > gate².

Noise and parameters are read from configs/default.yaml (ekf: section) or
fall back to the EKFConfig dataclass defaults.

Robotics relevance
------------------
Demonstrates sensor fusion, localization, EKF design, noise modelling,
and outlier rejection — core competencies for robotics perception / nav roles.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ACCEL_CLIP_MS2: float = 30.0   # default spike-rejection threshold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EKFConfig:
    """Tunable EKF parameters — all values in SI units."""

    # Process noise covariance Q (diagonal entries)
    process_noise_pos: float = 0.01   # (m²)
    process_noise_vel: float = 0.10   # ((m/s)²)
    process_noise_acc: float = 1.00   # ((m/s²)²)

    # Measurement noise covariance R
    measurement_noise_gps_pos: float = 4.0   # (m²)   ≈ 2 m CEP
    measurement_noise_gps_vel: float = 0.05  # ((m/s)²)

    # Outlier gate — reject GPS update when Mahalanobis² > gate²
    mahalanobis_gate: float = 3.0

    @classmethod
    def from_yaml(cls, ekf_dict: dict) -> "EKFConfig":
        """Populate from the ekf: sub-dict of configs/default.yaml.

        Missing keys fall back to the dataclass defaults, so a partial
        (or empty) dict is always safe to pass.
        """
        d = ekf_dict or {}
        return cls(
            process_noise_pos           = float(d.get("process_noise_pos",            cls.process_noise_pos)),
            process_noise_vel           = float(d.get("process_noise_vel",            cls.process_noise_vel)),
            process_noise_acc           = float(d.get("process_noise_acc",            cls.process_noise_acc)),
            measurement_noise_gps_pos   = float(d.get("measurement_noise_gps_pos",   cls.measurement_noise_gps_pos)),
            measurement_noise_gps_vel   = float(d.get("measurement_noise_gps_vel",   cls.measurement_noise_gps_vel)),
            mahalanobis_gate            = float(d.get("mahalanobis_gate",             cls.mahalanobis_gate)),
        )


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------

@dataclass
class _EKFState:
    """Mutable EKF state threaded through predict/update."""
    x: np.ndarray  # shape (6,) — [east, north, vx, vz, ax, az]
    P: np.ndarray  # shape (6,6) — covariance matrix


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StateEstimator:
    """
    EKF trajectory reconstruction from GPS + IMU sensor data.

    Parameters
    ----------
    config : EKFConfig (defaults used when None)

    Usage
    -----
    estimator = StateEstimator()
    ekf_df = estimator.run(df, geo.enu_east, geo.enu_north)

    After run():
        estimator.covariance_trace   — np.ndarray shape (N,): trace(P) per step
        estimator.innovation_sequence — np.ndarray shape (N,): ‖innovation‖ per step
    """

    def __init__(self, config: Optional[EKFConfig] = None) -> None:
        self._config: EKFConfig = config or EKFConfig()
        self.covariance_trace: np.ndarray = np.array([])
        self.innovation_sequence: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        enu_east: np.ndarray,
        enu_north: np.ndarray,
    ) -> pd.DataFrame:
        """
        Run the EKF over the full sensor trajectory.

        Parameters
        ----------
        df        : cleaned DataFrame from DrifterDataLoader (needs AX, AZ,
                    speed_ms, dt_s, heading_deg)
        enu_east  : ENU east positions (m), shape (N,) — from GeoConverter
        enu_north : ENU north positions (m), shape (N,) — from GeoConverter

        Returns
        -------
        df with added columns:
            x_filt, y_filt         — filtered east/north position (m)
            vx_filt, vz_filt       — filtered east/north velocity (m/s)
            ax_filt, az_filt       — filtered east/north acceleration (m/s²)
            cov_trace              — trace(P) at each step
            innovation_norm        — ‖innovation‖ before gate decision (m or m/s)
            drift_flag             — True when heading diverges > 30° from bulk flow
        """
        cfg = self._config
        n = len(df)

        # Build noise matrices once
        Q = self._build_Q(cfg)
        R = self._build_R(cfg)
        gate_sq = cfg.mahalanobis_gate ** 2

        # Debiased IMU (ENU east/north components)
        g_bias_z = self._compute_gravity_bias(df)
        ax_d, az_d = self._debias_accel(df, g_bias_z)

        dt_arr      = df["dt_s"].values.astype(float)
        speed_arr   = df["speed_ms"].values.astype(float)
        seg_arr     = df["segment_id"].values if "segment_id" in df.columns else np.zeros(n)

        # Pre-allocate output arrays
        x_filt  = np.zeros(n)
        y_filt  = np.zeros(n)
        vx_filt = np.zeros(n)
        vz_filt = np.zeros(n)
        ax_filt = np.zeros(n)
        az_filt = np.zeros(n)
        cov_trace  = np.zeros(n)
        inn_norms  = np.zeros(n)

        def _init_state(i: int) -> _EKFState:
            """Initialise (or re-initialise) the filter at row i."""
            return _EKFState(
                x=np.array([
                    float(enu_east[i]),
                    float(enu_north[i]),
                    0.0, 0.0,   # zero initial velocity
                    0.0, 0.0,   # zero initial acceleration
                ]),
                P=np.eye(6),
            )

        # Initialise state at row 0
        state = _init_state(0)

        # Record row 0
        x_filt[0]  = state.x[0]
        y_filt[0]  = state.x[1]
        vx_filt[0] = state.x[2]
        vz_filt[0] = state.x[3]
        ax_filt[0] = state.x[4]
        az_filt[0] = state.x[5]
        cov_trace[0]  = float(np.trace(state.P))
        inn_norms[0]  = 0.0

        # Main filter loop
        n_gated = 0
        n_resets = 0
        for i in range(1, n):
            dt = float(dt_arr[i])
            if dt <= 0:
                dt = 0.5  # fallback for zero-dt rows

            # Hard-reset at segment boundaries (millis() wrap-around) so
            # the filter doesn't try to bridge GPS position jumps with the
            # kinematic model — which would cause covariance divergence.
            if seg_arr[i] != seg_arr[i - 1]:
                state = _init_state(i)
                n_resets += 1
                x_filt[i]  = state.x[0]
                y_filt[i]  = state.x[1]
                vx_filt[i] = state.x[2]
                vz_filt[i] = state.x[3]
                ax_filt[i] = state.x[4]
                az_filt[i] = state.x[5]
                cov_trace[i]  = float(np.trace(state.P))
                inn_norms[i]  = 0.0
                continue

            # --- Predict ---
            # Use zero acceleration input: the IMU body-frame oscillations
            # (wave forces, bobbing) are not suitable as translational ENU
            # drivers for a floating platform.  Process noise Q absorbs the
            # velocity uncertainty; GPS measurements correct position.
            state = self._predict(state, dt, 0.0, 0.0, Q)

            # --- Measurement ---
            z = np.array([
                float(enu_east[i]),
                float(enu_north[i]),
                float(speed_arr[i]),
            ])

            state, inn_norm, gated = self._update(state, z, R, gate_sq)
            if gated:
                n_gated += 1

            # Record
            x_filt[i]  = state.x[0]
            y_filt[i]  = state.x[1]
            vx_filt[i] = state.x[2]
            vz_filt[i] = state.x[3]
            ax_filt[i] = state.x[4]
            az_filt[i] = state.x[5]
            cov_trace[i]  = float(np.trace(state.P))
            inn_norms[i]  = inn_norm

        # Drift detection over full filtered velocity sequence
        drift_flags = self._detect_drift(df, vx_filt, vz_filt)

        log.info(
            "EKF complete: %d samples  resets=%d  gated=%d (%.1f%%)  "
            "cov_trace_final=%.4f  drift_fraction=%.1f%%",
            n, n_resets, n_gated, 100.0 * n_gated / max(n, 1),
            float(cov_trace[-1]),
            100.0 * float(drift_flags.sum()) / max(n, 1),
        )

        # Store diagnostics
        self.covariance_trace = cov_trace
        self.innovation_sequence = inn_norms

        # Build output DataFrame
        out = df.copy()
        out["x_filt"]         = x_filt
        out["y_filt"]         = y_filt
        out["vx_filt"]        = vx_filt
        out["vz_filt"]        = vz_filt
        out["ax_filt"]        = ax_filt
        out["az_filt"]        = az_filt
        out["cov_trace"]      = cov_trace
        out["innovation_norm"] = inn_norms
        out["drift_flag"]     = drift_flags

        return out

    # ------------------------------------------------------------------
    # Gravity / IMU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gravity_bias(df: pd.DataFrame) -> float:
        """Estimate gravity bias as median(AZ). Mirrors geo_converter."""
        return float(np.median(df["AZ"].fillna(0.0).values.astype(float)))

    @staticmethod
    def _debias_accel(
        df: pd.DataFrame,
        g_bias_z: float,
        clip: float = _ACCEL_CLIP_MS2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove gravity bias from AZ; clip both axes to ±clip (m/s²).

        AX bias is near-zero for a horizontally floating platform (per the
        complementary filter in geo_converter.py), so AX is used as-is.

        Returns
        -------
        ax_debiased, az_debiased : shape (N,) arrays in m/s²
        """
        ax = df["AX"].fillna(0.0).values.astype(float)
        az = df["AZ"].fillna(0.0).values.astype(float) - g_bias_z
        ax = np.clip(ax, -clip, clip)
        az = np.clip(az, -clip, clip)
        return ax, az

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_Q(cfg: EKFConfig) -> np.ndarray:
        """6×6 diagonal process noise covariance."""
        p = cfg.process_noise_pos
        v = cfg.process_noise_vel
        a = cfg.process_noise_acc
        return np.diag([p, p, v, v, a, a])

    @staticmethod
    def _build_R(cfg: EKFConfig) -> np.ndarray:
        """3×3 diagonal measurement noise covariance [pos_e, pos_n, speed]."""
        gp = cfg.measurement_noise_gps_pos
        gv = cfg.measurement_noise_gps_vel
        return np.diag([gp, gp, gv])

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """
        6×6 constant-acceleration state transition matrix.

        State order: [east, north, vx, vz, ax, az]
        """
        dt2 = 0.5 * dt * dt
        F = np.eye(6)
        # position ← velocity
        F[0, 2] = dt
        F[1, 3] = dt
        # position ← acceleration
        F[0, 4] = dt2
        F[1, 5] = dt2
        # velocity ← acceleration
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    @staticmethod
    def _build_H_speed(vx: float, vz: float) -> np.ndarray:
        """
        3×6 measurement matrix linearised around predicted velocity.

        Rows: [east, north, speed]
        Speed row is linearised: dh/dvx = vx/|v|, dh/dvz = vz/|v|.
        When |v| < 1e-4 m/s the speed row is zeroed (GPS speed unreliable).
        """
        H = np.zeros((3, 6))
        H[0, 0] = 1.0   # east
        H[1, 1] = 1.0   # north
        spd = math.sqrt(vx * vx + vz * vz)
        if spd > 1e-4:
            H[2, 2] = vx / spd
            H[2, 3] = vz / spd
        return H

    # ------------------------------------------------------------------
    # Predict / Update steps
    # ------------------------------------------------------------------

    @staticmethod
    def _predict(
        state: _EKFState,
        dt: float,
        ax: float,
        az: float,
        Q: np.ndarray,
    ) -> _EKFState:
        """
        EKF prediction step.

        The IMU acceleration is injected into x[4:6] before applying the
        kinematic F matrix, so the IMU drives the velocity/position prediction.
        """
        x = state.x.copy()
        x[4] = ax
        x[5] = az

        F = StateEstimator._build_F(dt)
        x_pred = F @ x
        P_pred = F @ state.P @ F.T + Q

        return _EKFState(x=x_pred, P=P_pred)

    @staticmethod
    def _update(
        state: _EKFState,
        z: np.ndarray,
        R: np.ndarray,
        gate_sq: float,
    ) -> Tuple[_EKFState, float, bool]:
        """
        EKF measurement update with Mahalanobis gating.

        Parameters
        ----------
        state   : predicted state
        z       : measurement [east_gps, north_gps, speed_gps]
        R       : 3×3 measurement noise
        gate_sq : mahalanobis_gate²

        Returns
        -------
        (updated_state, innovation_norm, gated)
        gated=True means the GPS update was rejected.
        """
        H = StateEstimator._build_H_speed(state.x[2], state.x[3])

        # Predicted measurement (non-linear speed)
        vx, vz = state.x[2], state.x[3]
        pred_speed = math.sqrt(vx * vx + vz * vz)
        z_pred = np.array([state.x[0], state.x[1], pred_speed])

        innovation = z - z_pred
        inn_norm = float(np.linalg.norm(innovation))

        # Innovation covariance
        S = H @ state.P @ H.T + R

        # Mahalanobis gate
        try:
            mah_sq = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            mah_sq = float("inf")

        if mah_sq > gate_sq:
            return state, inn_norm, True

        # Kalman gain: K = P H^T S^{-1}
        # Computed as: solve(S^T, (P H^T)^T)^T = solve(S, P H^T)   (S symmetric)
        PH_T = state.P @ H.T   # shape (6, 3)
        try:
            K = np.linalg.solve(S.T, PH_T.T).T   # shape (6, 3)
        except np.linalg.LinAlgError:
            return state, inn_norm, True

        x_new = state.x + K @ innovation
        # Joseph form for numerical stability: (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(6) - K @ H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        return _EKFState(x=x_new, P=P_new), inn_norm, False

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_drift(
        df: pd.DataFrame,
        vx_filt: np.ndarray,
        vz_filt: np.ndarray,
        threshold_deg: float = 30.0,
        min_speed_ms: float = 0.05,
    ) -> np.ndarray:
        """
        Flag samples where heading diverges more than *threshold_deg* from
        the estimated bulk flow direction.

        The bulk flow direction is the median heading across all EKF-filtered
        velocity vectors (robust estimator using median heading).

        Near-zero-speed samples are never flagged (heading undefined).

        Returns
        -------
        drift_flag : bool array, shape (N,)
        """
        speed = df["speed_ms"].values.astype(float)
        n = len(vx_filt)

        # Median bulk flow angle (radians), ignoring near-zero velocity
        mask = speed > min_speed_ms
        if mask.sum() < 2:
            return np.zeros(n, dtype=bool)

        flow_angle = float(np.arctan2(
            np.median(vx_filt[mask]),
            np.median(vz_filt[mask]),
        ))

        threshold_rad = math.radians(threshold_deg)
        flags = np.zeros(n, dtype=bool)

        for i in range(n):
            if speed[i] < min_speed_ms:
                continue
            sample_angle = math.atan2(float(vx_filt[i]), float(vz_filt[i]))
            diff = sample_angle - flow_angle
            # Wrap to [-π, π]
            diff = math.atan2(math.sin(diff), math.cos(diff))
            if abs(diff) > threshold_rad:
                flags[i] = True

        return flags
