#!/usr/bin/env python3
"""
Full smoothing pipeline for all four forward-pass runs.

Stages per run:
  1. EKF  — constant-velocity Kalman filter with Mahalanobis outlier gating
  2. RTS  — Rauch-Tung-Striebel backward smoother (uses all future data)
  3. SG   — Savitzky-Golay polynomial filter (removes residual jitter)

Outputs (visualizations/smoothed_forward/):
  <stem>_smoothed.csv   — original columns + east/north/Lat/Lon per stage
  <stem>_plot.png       — per-run comparison: raw vs each pipeline stage
  overlay_all_smoothed.png — all four runs on one plot (final smoothed only)

Usage:
    python visualizations/scripts/smooth_forward_runs.py [--out-dir <path>]
"""

import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.signal import savgol_filter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR   = _REPO_ROOT / "visualizations" / "smoothed_forward"

FORWARD_CSVS = [
    ("Run 1 (M30-1)",  _REPO_ROOT / "exts/drifter_vis/data/enoM30(1)/enoM30(1)_forward.csv"),
    ("Run 2 (M30-2)",  _REPO_ROOT / "exts/drifter_vis/data/enoM30(2)/enoM30(2)_forward.csv"),
    ("Run 3 (M30-4a)", _REPO_ROOT / "exts/drifter_vis/data/enoM30(4)/enoM30(4)_forward1.csv"),
    ("Run 4 (M30-4b)", _REPO_ROOT / "exts/drifter_vis/data/enoM30(4)/enoM30(4)_forward2.csv"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ── EKF tuning ───────────────────────────────────────────────────────────────
R_POS      = 4.0   # GPS position measurement noise (m²)   ≈ 2 m CEP
R_VEL      = 0.05  # GPS speed measurement noise ((m/s)²)
Q_POS      = 0.01  # process noise — position (m²)
Q_VEL      = 0.10  # process noise — velocity ((m/s)²)
MAH_GATE   = 3.0   # Mahalanobis gate threshold (σ)

# ── Savitzky-Golay tuning ────────────────────────────────────────────────────
SG_WINDOW    = 21  # samples (~21 s at 1 Hz); must be odd
SG_POLYORDER = 3

R_EARTH = 6_371_000.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coordinate helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def lla_to_enu(lat, lon, origin_lat, origin_lon):
    east  = R_EARTH * (np.deg2rad(lon) - math.radians(origin_lon)) * math.cos(math.radians(origin_lat))
    north = R_EARTH * (np.deg2rad(lat) - math.radians(origin_lat))
    return east, north


def enu_to_lla(east, north, origin_lat, origin_lon):
    lat = origin_lat + np.degrees(north / R_EARTH)
    lon = origin_lon + np.degrees(east / (R_EARTH * math.cos(math.radians(origin_lat))))
    return lat, lon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("Lat", "Lon", "GPS_Alt", "Speed", "Time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = ~((df["Lat"].fillna(0) == 0) & (df["Lon"].fillna(0) == 0))
    df = df[mask].dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    df["speed_ms"] = df["Speed"].fillna(0.0) / 2.237
    t = df["Time"].ffill().values.astype(float) / 1000.0
    t -= t[0]
    df["time_s"] = t
    dt = np.diff(t, prepend=np.nan)
    dt[0] = float(np.nanmedian(dt[1:]))
    df["dt_s"] = np.clip(dt, 1e-3, 60.0)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1 — EKF forward pass
# State: x = [east, north, vx, vz]   (4-D constant-velocity model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _H(vx: float, vz: float) -> np.ndarray:
    """3×4 measurement matrix; speed row linearised around predicted velocity."""
    H = np.zeros((3, 4))
    H[0, 0] = 1.0   # east
    H[1, 1] = 1.0   # north
    spd = math.sqrt(vx * vx + vz * vz)
    if spd > 1e-4:
        H[2, 2] = vx / spd
        H[2, 3] = vz / spd
    return H


def run_ekf(east, north, speed, dt_arr):
    """
    Forward EKF pass.

    Returns
    -------
    x_filt  (N, 4): filtered state  [east, north, vx, vz]
    P_filt  (N, 4, 4): filtered covariance
    x_pred  (N, 4): predicted state (before measurement update)
    P_pred  (N, 4, 4): predicted covariance
    F_hist  (N, 4, 4): transition matrix used at each step
    """
    n = len(east)
    Q       = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL])
    R       = np.diag([R_POS, R_POS, R_VEL])
    gate_sq = MAH_GATE ** 2

    x_filt = np.zeros((n, 4))
    P_filt = np.zeros((n, 4, 4))
    x_pred = np.zeros((n, 4))
    P_pred = np.zeros((n, 4, 4))
    F_hist = np.zeros((n, 4, 4))

    x = np.array([east[0], north[0], 0.0, 0.0])
    P = np.eye(4) * 10.0

    x_filt[0] = x;  P_filt[0] = P
    x_pred[0] = x;  P_pred[0] = P
    F_hist[0]  = np.eye(4)

    n_gated = 0
    for i in range(1, n):
        dt = max(float(dt_arr[i]), 1e-3)
        Fk = _F(dt)
        F_hist[i] = Fk

        # Predict
        xp = Fk @ x
        Pp = Fk @ P @ Fk.T + Q
        x_pred[i] = xp
        P_pred[i] = Pp

        # Linearise measurement around predicted velocity
        H  = _H(xp[2], xp[3])
        spd_pred = math.sqrt(xp[2]**2 + xp[3]**2)
        z      = np.array([east[i], north[i], speed[i]])
        z_pred = np.array([xp[0],   xp[1],    spd_pred])
        inn    = z - z_pred

        # Mahalanobis gate
        S = H @ Pp @ H.T + R
        try:
            mah_sq = float(inn @ np.linalg.solve(S, inn))
        except np.linalg.LinAlgError:
            mah_sq = float("inf")

        if mah_sq > gate_sq:
            n_gated += 1
            x, P = xp, Pp
        else:
            try:
                K = np.linalg.solve(S.T, (Pp @ H.T).T).T
            except np.linalg.LinAlgError:
                x, P = xp, Pp
                x_filt[i] = x; P_filt[i] = P
                continue
            x = xp + K @ inn
            IKH = np.eye(4) - K @ H
            P   = IKH @ Pp @ IKH.T + K @ R @ K.T  # Joseph form

        x_filt[i] = x
        P_filt[i] = P

    print(f"    EKF: {n_gated}/{n} fixes gated ({100*n_gated/n:.1f}%)")
    return x_filt, P_filt, x_pred, P_pred, F_hist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 2 — RTS backward smoother
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_rts(x_filt, P_filt, x_pred, P_pred, F_hist):
    """
    Rauch-Tung-Striebel smoother.

    Backward recursion from k = N-2 → 0:
        G_k     = P_filt[k] · F[k+1]ᵀ · P_pred[k+1]⁻¹
        x_s[k]  = x_filt[k] + G_k · (x_s[k+1] − x_pred[k+1])
        P_s[k]  = P_filt[k] + G_k · (P_s[k+1] − P_pred[k+1]) · G_kᵀ

    Returns x_smooth (N, 4).
    """
    n = len(x_filt)
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(n - 2, -1, -1):
        try:
            G = P_filt[k] @ F_hist[k + 1].T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

    return x_smooth


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 3 — Savitzky-Golay (position only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_savgol(x_smooth, window=SG_WINDOW, polyorder=SG_POLYORDER):
    """Apply SG to east and north columns of x_smooth. Window auto-shrinks if needed."""
    n = len(x_smooth)
    w = window if window <= n else (n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        return x_smooth.copy()
    out = x_smooth.copy()
    out[:, 0] = savgol_filter(x_smooth[:, 0], w, polyorder)
    out[:, 1] = savgol_filter(x_smooth[:, 1], w, polyorder)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-run processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_run(label, csv_path, out_dir, origin_lat, origin_lon):
    """Run the full pipeline for one CSV. Saves CSV + plot. Returns final east/north."""
    print(f"\n  {label}")
    df = load_csv(csv_path)
    print(f"    Loaded {len(df)} rows")

    lat   = df["Lat"].values.astype(float)
    lon   = df["Lon"].values.astype(float)
    speed = df["speed_ms"].values.astype(float)
    dt    = df["dt_s"].values.astype(float)

    east_raw, north_raw = lla_to_enu(lat, lon, origin_lat, origin_lon)

    x_filt, P_filt, x_pred, P_pred, F_hist = run_ekf(east_raw, north_raw, speed, dt)
    x_rts = run_rts(x_filt, P_filt, x_pred, P_pred, F_hist)
    x_sg  = run_savgol(x_rts)
    print(f"    RTS + SG complete")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_df = df.copy()
    out_df["east_raw"]     = east_raw
    out_df["north_raw"]    = north_raw
    out_df["east_ekf"]     = x_filt[:, 0]
    out_df["north_ekf"]    = x_filt[:, 1]
    out_df["vx_ekf"]       = x_filt[:, 2]
    out_df["vz_ekf"]       = x_filt[:, 3]
    out_df["east_rts"]     = x_rts[:, 0]
    out_df["north_rts"]    = x_rts[:, 1]
    out_df["east_smooth"]  = x_sg[:, 0]
    out_df["north_smooth"] = x_sg[:, 1]
    sm_lat, sm_lon = enu_to_lla(x_sg[:, 0], x_sg[:, 1], origin_lat, origin_lon)
    out_df["Lat_smooth"] = sm_lat
    out_df["Lon_smooth"] = sm_lon

    csv_out = out_dir / f"{csv_path.stem}_smoothed.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"    Saved: {csv_out.name}")

    # ── Per-run comparison plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(east_raw,      north_raw,      color="#cccccc", lw=1.0, alpha=0.9, label="Raw GPS",         zorder=1)
    ax.plot(x_filt[:, 0], x_filt[:, 1],   color="#aec7e8", lw=1.5, alpha=0.9, label="EKF",             zorder=2)
    ax.plot(x_rts[:, 0],  x_rts[:, 1],    color="#ff7f0e", lw=1.5, alpha=0.9, label="EKF + RTS",       zorder=3)
    ax.plot(x_sg[:, 0],   x_sg[:, 1],     color="#1f77b4", lw=2.2,            label="EKF + RTS + SG",  zorder=4)

    ax.plot(x_sg[0, 0],  x_sg[0, 1],  "go", ms=8,  zorder=5, label="Start")
    ax.plot(x_sg[-1, 0], x_sg[-1, 1], "r*", ms=12, zorder=5, label="End")

    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title(f"{label} — Smoothing Pipeline Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    png_out = out_dir / f"{csv_path.stem}_plot.png"
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {png_out.name}")

    return x_sg[:, 0], x_sg[:, 1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Overlay plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_overlay(smoothed_runs, out_dir):
    fig, ax = plt.subplots(figsize=(11, 8))
    handles = []

    for (label, east, north), color in zip(smoothed_runs, COLORS):
        ax.plot(east, north, color=color, lw=2.0, alpha=0.85)
        ax.plot(east[0],  north[0],  "o", color=color, ms=7,  zorder=5)
        ax.plot(east[-1], north[-1], "*", color=color, ms=11, zorder=5)
        handles.append(mlines.Line2D([], [], color=color, lw=2, label=label))

    handles += [
        mlines.Line2D([], [], marker="o", color="grey", linestyle="none", ms=7,  label="Start"),
        mlines.Line2D([], [], marker="*", color="grey", linestyle="none", ms=11, label="End"),
    ]
    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title("Eno River — All Four Forward Passes (EKF + RTS + SG)",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=handles, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    out = out_dir / "overlay_all_smoothed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved overlay: {out.name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Common ENU origin = first GPS point of the first available run
    first_df   = load_csv(FORWARD_CSVS[0][1])
    origin_lat = float(first_df["Lat"].iloc[0])
    origin_lon = float(first_df["Lon"].iloc[0])
    print(f"Common ENU origin: {origin_lat:.6f}°N  {origin_lon:.6f}°E")
    print(f"Output directory:  {args.out_dir}\n")

    smoothed_runs = []
    for label, csv_path in FORWARD_CSVS:
        if not csv_path.exists():
            print(f"  Warning: {csv_path.name} not found — skipping")
            continue
        east_s, north_s = process_run(label, csv_path, args.out_dir, origin_lat, origin_lon)
        smoothed_runs.append((label, east_s, north_s))

    if smoothed_runs:
        save_overlay(smoothed_runs, args.out_dir)

    print(f"\nDone. {len(smoothed_runs)} runs processed.")


if __name__ == "__main__":
    main()
