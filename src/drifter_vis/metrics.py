"""
metrics.py — Quantitative evaluation of trajectory reconstruction quality.

Compares three trajectory sources:
    1. Raw GPS path (from GeoConverter)
    2. EKF-filtered path (from StateEstimator)
    3. Physics-simulated path (from PhysicsValidator, optional)

All values are in SI units (metres, m/s, m/s², 1/m) unless noted.

Metrics computed
----------------
- Trajectory RMSE:
    rmse_raw_gps  — RMSE between raw GPS positions and EKF path
                    (measures GPS noise removed by EKF; units: m)
    rmse_ekf      — RMSE between EKF path and a 3-point moving average of
                    itself (measures residual jitter in the filter; units: m)
    rmse_physics  — RMSE between physics simulation and raw GPS path
                    (measures physics model accuracy; units: m)

- Filter improvement ratio:
    filter_improvement_ratio = rmse_raw_gps / rmse_ekf
    Ratio > 1 means the EKF output is smoother than the raw GPS.

- Velocity error:
    |EKF speed − GPS speed|, mean and std

- Acceleration residuals:
    |EKF acceleration − debiased IMU magnitude|, mean and std

- Path curvature:
    κ = |vx · az − vz · ax| / (vx² + vz²)^(3/2)  per sample (1/m)

- Drift divergence:
    Final Euclidean distance between physics simulation endpoint and GPS
    endpoint (measures physics model drift over the full run; units: m)

Robotics relevance
------------------
Provides the quantifiable "impact" numbers for resume bullets:
RMSE reduction, filter improvement ratio, velocity accuracy, curvature map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    """Controls which metrics are reported and where output goes."""

    rmse_report:                  bool  = True
    velocity_error_report:        bool  = True
    acceleration_residual_report: bool  = True
    drift_divergence_report:      bool  = True
    filter_improvement_report:    bool  = True

    output_csv: str = "data/metrics.csv"

    # Curvature parameters (from default.yaml curvature: section)
    curvature_min_speed_ms:    float = 0.05   # ignore below this speed
    curvature_high_threshold:  float = 0.5    # (1/m) label as high-curvature

    @classmethod
    def from_yaml(cls, merged_dict: dict) -> "MetricsConfig":
        """
        Populate from a merged dict of the metrics: and curvature: YAML sections.

        Pass ``{**cfg["metrics"], **cfg["curvature"]}`` (or either alone).
        Missing keys fall back to dataclass defaults.
        """
        d = merged_dict or {}
        return cls(
            rmse_report                  = bool(d.get("rmse_report",                  cls.rmse_report)),
            velocity_error_report        = bool(d.get("velocity_error_report",        cls.velocity_error_report)),
            acceleration_residual_report = bool(d.get("acceleration_residual_report", cls.acceleration_residual_report)),
            drift_divergence_report      = bool(d.get("drift_divergence_report",      cls.drift_divergence_report)),
            filter_improvement_report    = bool(d.get("filter_improvement_report",    cls.filter_improvement_report)),
            output_csv                   = str( d.get("output_csv",                   cls.output_csv)),
            curvature_min_speed_ms       = float(d.get("min_speed_ms",               cls.curvature_min_speed_ms)),
            curvature_high_threshold     = float(d.get("high_curvature_threshold",   cls.curvature_high_threshold)),
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    """All computed metrics for one run."""

    # Trajectory RMSE (metres)
    rmse_raw_gps: float
    rmse_ekf: float
    rmse_physics: Optional[float]

    # Filter improvement (dimensionless ratio; >1 = EKF smoother than GPS)
    filter_improvement_ratio: Optional[float]

    # Velocity error vs GPS speed (m/s)
    velocity_error_mean: float
    velocity_error_std: float

    # Acceleration residuals vs debiased IMU (m/s²)
    accel_residual_mean: float
    accel_residual_std: float

    # Per-sample path curvature (1/m) — stored as array for inspection
    curvature: np.ndarray

    # Physics simulation endpoint divergence (metres)
    drift_divergence: Optional[float]

    def to_dict(self) -> dict:
        """
        Flat dict suitable for a single-row CSV / pandas DataFrame.
        The curvature array is summarised as mean, std, and high_fraction.
        """
        kappa = self.curvature
        high = 0.0
        if len(kappa) > 0:
            # fraction of samples above curvature_high_threshold — stored
            # by the compute() caller if needed; here we just use 0.5 /m
            high = float(np.mean(kappa > 0.5))

        return {
            "rmse_raw_gps_m":            self.rmse_raw_gps,
            "rmse_ekf_m":                self.rmse_ekf,
            "rmse_physics_m":            self.rmse_physics,
            "filter_improvement_ratio":  self.filter_improvement_ratio,
            "velocity_error_mean_ms":    self.velocity_error_mean,
            "velocity_error_std_ms":     self.velocity_error_std,
            "accel_residual_mean_ms2":   self.accel_residual_mean,
            "accel_residual_std_ms2":    self.accel_residual_std,
            "curvature_mean_1pm":        float(np.mean(kappa)) if len(kappa) else float("nan"),
            "curvature_std_1pm":         float(np.std(kappa))  if len(kappa) else float("nan"),
            "curvature_high_fraction":   high,
            "drift_divergence_m":        self.drift_divergence,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DrifterMetrics:
    """
    Compute and report evaluation metrics for the drifter digital twin.

    Parameters
    ----------
    config : MetricsConfig (defaults used when None)
    """

    def __init__(self, config: Optional[MetricsConfig] = None) -> None:
        self._config = config or MetricsConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        geo,
        ekf_df: pd.DataFrame,
        sim_east: Optional[np.ndarray] = None,
        sim_north: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Compute all enabled metrics.

        Parameters
        ----------
        df      : cleaned DataFrame from DrifterDataLoader
        geo     : GeoResult from GeoConverter (has .enu_east, .enu_north)
        ekf_df  : output of StateEstimator.run() (has x_filt, y_filt,
                  vx_filt, vz_filt, ax_filt, az_filt)
        sim_east/north : PhysicsValidator output (optional)

        Returns
        -------
        MetricsResult
        """
        rec_east  = np.asarray(geo.enu_east,  dtype=float)
        rec_north = np.asarray(geo.enu_north, dtype=float)
        x_filt    = ekf_df["x_filt"].values.astype(float)
        y_filt    = ekf_df["y_filt"].values.astype(float)

        # --- RMSE ---
        rmse_raw = self._compute_rmse(x_filt, y_filt, rec_east, rec_north)
        rmse_ekf = self._compute_ekf_smoothness(x_filt, y_filt)

        rmse_physics: Optional[float] = None
        if sim_east is not None and sim_north is not None:
            rmse_physics = self._compute_rmse(
                rec_east, rec_north,
                np.asarray(sim_east, dtype=float),
                np.asarray(sim_north, dtype=float),
            )

        # --- Filter improvement ---
        filter_ratio: Optional[float] = None
        if rmse_ekf > 1e-9:
            filter_ratio = rmse_raw / rmse_ekf

        # --- Velocity error ---
        vel_mean, vel_std = self._compute_velocity_error(df, ekf_df)

        # --- Acceleration residuals ---
        g_bias_z = float(np.median(df["AZ"].fillna(0.0).values.astype(float)))
        acc_mean, acc_std = self._compute_accel_residuals(df, ekf_df, g_bias_z)

        # --- Curvature ---
        kappa = self._compute_curvature(
            ekf_df["vx_filt"].values.astype(float),
            ekf_df["vz_filt"].values.astype(float),
            ekf_df["ax_filt"].values.astype(float),
            ekf_df["az_filt"].values.astype(float),
            self._config.curvature_min_speed_ms,
        )

        # --- Drift divergence ---
        drift_div: Optional[float] = None
        if sim_east is not None and sim_north is not None:
            drift_div = self._compute_drift_divergence(
                rec_east, rec_north,
                np.asarray(sim_east, dtype=float),
                np.asarray(sim_north, dtype=float),
            )

        result = MetricsResult(
            rmse_raw_gps             = rmse_raw,
            rmse_ekf                 = rmse_ekf,
            rmse_physics             = rmse_physics,
            filter_improvement_ratio = filter_ratio,
            velocity_error_mean      = vel_mean,
            velocity_error_std       = vel_std,
            accel_residual_mean      = acc_mean,
            accel_residual_std       = acc_std,
            curvature                = kappa,
            drift_divergence         = drift_div,
        )

        log.info(
            "Metrics: RMSE_raw=%.2f m  RMSE_EKF=%.4f m  ratio=%.2f  "
            "vel_err=%.3f m/s  kappa_mean=%.4f 1/m",
            rmse_raw, rmse_ekf, filter_ratio or float("nan"),
            vel_mean, float(np.mean(kappa)),
        )
        return result

    def export_csv(
        self,
        result: MetricsResult,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Write metrics to a single-row CSV.

        The output path is resolved relative to the repository root when
        not absolute.  Parent directories are created automatically.

        Returns the resolved Path.
        """
        path_str = output_path or self._config.output_csv
        out_path = Path(path_str)
        if not out_path.is_absolute():
            # Resolve relative to repo root: this file is 7 levels deep from root
            repo_root = Path(__file__).resolve().parents[7]
            out_path = repo_root / path_str

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result.to_dict()]).to_csv(out_path, index=False)
        log.info("Metrics written to %s", out_path)
        return out_path

    def print_table(self, result: MetricsResult) -> None:
        """Print a human-readable aligned metrics table to stdout."""
        d = result.to_dict()
        kappa = result.curvature
        high_frac = float(np.mean(kappa > self._config.curvature_high_threshold)) if len(kappa) else 0.0

        def _fmt(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        lines = [
            "=" * 45,
            "  Drifter Digital Twin — Evaluation Metrics",
            "=" * 45,
            "",
            "Trajectory RMSE",
            f"  Raw GPS (vs EKF reference): {_fmt(d['rmse_raw_gps_m'])} m",
            f"  EKF internal smoothness:    {_fmt(d['rmse_ekf_m'])} m",
            f"  Physics simulation:         {_fmt(d['rmse_physics_m'])} m",
            "",
            f"Filter improvement ratio:     {_fmt(d['filter_improvement_ratio'])}×  (>1 = EKF smoother)",
            "",
            "Velocity error (EKF vs GPS speed)",
            f"  Mean:  {_fmt(d['velocity_error_mean_ms'])} m/s",
            f"  Std:   {_fmt(d['velocity_error_std_ms'])} m/s",
            "",
            "Acceleration residuals (EKF vs debiased IMU)",
            f"  Mean:  {_fmt(d['accel_residual_mean_ms2'])} m/s²",
            f"  Std:   {_fmt(d['accel_residual_std_ms2'])} m/s²",
            "",
            "Path curvature κ",
            f"  Mean:  {_fmt(d['curvature_mean_1pm'])} 1/m",
            f"  Std:   {_fmt(d['curvature_std_1pm'])} 1/m",
            f"  High-κ fraction (>{self._config.curvature_high_threshold:.2f} 1/m): {high_frac:.1%}",
            "",
            f"Drift divergence (physics endpoint): {_fmt(d['drift_divergence_m'])} m",
            "=" * 45,
        ]
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Computation helpers (all static)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rmse(
        true_east: np.ndarray,
        true_north: np.ndarray,
        pred_east: np.ndarray,
        pred_north: np.ndarray,
    ) -> float:
        """
        2-D RMSE: √(mean((Δe)² + (Δn)²)).

        Used for:
          rmse_raw_gps  — true=EKF path, pred=raw GPS (how much GPS noise EKF removes)
          rmse_physics  — true=recorded GPS, pred=physics sim (physics model accuracy)
        """
        de = pred_east  - true_east
        dn = pred_north - true_north
        return float(np.sqrt(np.mean(de ** 2 + dn ** 2)))

    @staticmethod
    def _compute_ekf_smoothness(
        x_filt: np.ndarray,
        y_filt: np.ndarray,
    ) -> float:
        """
        RMSE between EKF path and a 3-point moving average of itself.

        Measures residual high-frequency jitter remaining in the filter output.
        A well-tuned filter should produce near-zero smoothness RMSE.
        """
        if len(x_filt) < 3:
            return 0.0
        # 3-point centred moving average with edge padding so perfectly linear
        # paths stay near-zero instead of being penalized by zero-padding.
        kernel = np.ones(3, dtype=float) / 3.0
        x_ma = np.convolve(np.pad(x_filt, 1, mode="edge"), kernel, mode="valid")
        y_ma = np.convolve(np.pad(y_filt, 1, mode="edge"), kernel, mode="valid")
        return float(np.sqrt(np.mean((x_filt - x_ma) ** 2 + (y_filt - y_ma) ** 2)))

    @staticmethod
    def _compute_velocity_error(
        df: pd.DataFrame,
        ekf_df: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Mean and std of |‖v_EKF‖ − speed_GPS|.

        GPS scalar speed is compared against the EKF-estimated speed magnitude.
        """
        vx = ekf_df["vx_filt"].values.astype(float)
        vz = ekf_df["vz_filt"].values.astype(float)
        v_ekf = np.sqrt(vx ** 2 + vz ** 2)
        v_gps = df["speed_ms"].values.astype(float)
        err = np.abs(v_ekf - v_gps)
        return float(err.mean()), float(err.std())

    @staticmethod
    def _compute_accel_residuals(
        df: pd.DataFrame,
        ekf_df: pd.DataFrame,
        g_bias_z: float,
    ) -> Tuple[float, float]:
        """
        Mean and std of |‖a_EKF‖ − ‖a_IMU_debiased‖| in the horizontal plane.

        Gravity is removed from AZ via g_bias_z (= median(AZ) computed in
        StateEstimator._compute_gravity_bias, same approach).
        """
        ax_imu = df["AX"].fillna(0.0).values.astype(float)
        az_imu = df["AZ"].fillna(0.0).values.astype(float) - g_bias_z
        a_imu  = np.sqrt(ax_imu ** 2 + az_imu ** 2)

        ax_ekf = ekf_df["ax_filt"].values.astype(float)
        az_ekf = ekf_df["az_filt"].values.astype(float)
        a_ekf  = np.sqrt(ax_ekf ** 2 + az_ekf ** 2)

        residual = np.abs(a_ekf - a_imu)
        return float(residual.mean()), float(residual.std())

    @staticmethod
    def _compute_curvature(
        vx: np.ndarray,
        vz: np.ndarray,
        ax: np.ndarray,
        az: np.ndarray,
        min_speed: float,
    ) -> np.ndarray:
        """
        Per-sample path curvature κ (1/m).

            κ = |vx · az − vz · ax| / (vx² + vz²)^(3/2)

        This is the 2-D signed curvature formula applied to the EKF velocity
        and acceleration estimates.  Segments where speed < min_speed are
        set to 0 to avoid numerical blow-up near zero velocity.

        Returns
        -------
        kappa : shape (N,) array of curvature magnitudes (1/m)
        """
        speed_sq = vx ** 2 + vz ** 2
        cross    = np.abs(vx * az - vz * ax)

        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(
                speed_sq > min_speed ** 2,
                cross / np.power(np.maximum(speed_sq, 1e-12), 1.5),
                0.0,
            )
        return kappa.astype(float)

    @staticmethod
    def _compute_drift_divergence(
        rec_east: np.ndarray,
        rec_north: np.ndarray,
        sim_east: np.ndarray,
        sim_north: np.ndarray,
    ) -> float:
        """
        Euclidean distance between the final recorded and simulated positions.

        Quantifies how far the physics model has drifted from the actual
        trajectory by the end of the run — a measure of model error accumulation.
        """
        de = float(sim_east[-1])  - float(rec_east[-1])
        dn = float(sim_north[-1]) - float(rec_north[-1])
        return float(np.sqrt(de ** 2 + dn ** 2))
