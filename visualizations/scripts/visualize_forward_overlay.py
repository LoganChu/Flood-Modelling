#!/usr/bin/env python3
"""
Overlay 2D plot of all four forward-pass runs on a shared ENU coordinate frame.

Usage:
    python visualizations/scripts/visualize_forward_overlay.py [--output <path>]
"""

import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_REPO_ROOT = Path(__file__).resolve().parents[2]

FORWARD_CSVS = [
    ("Run 1 (M30-1)",  _REPO_ROOT / "exts/drifter_vis/data/enoM30(1)/enoM30(1)_forward.csv"),
    ("Run 2 (M30-2)",  _REPO_ROOT / "exts/drifter_vis/data/enoM30(2)/enoM30(2)_forward.csv"),
    ("Run 3 (M30-4a)", _REPO_ROOT / "exts/drifter_vis/data/enoM30(4)/enoM30(4)_forward1.csv"),
    ("Run 4 (M30-4b)", _REPO_ROOT / "exts/drifter_vis/data/enoM30(4)/enoM30(4)_forward2.csv"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("Lat", "Lon", "GPS_Alt", "Speed", "Time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop GPS dropouts
    mask = ~((df["Lat"].fillna(0) == 0) & (df["Lon"].fillna(0) == 0))
    df = df[mask].dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    return df


def _lla_to_enu(lat, lon, origin_lat, origin_lon):
    """Spherical-Earth LLA → ENU (east, north) in metres."""
    R = 6_371_000.0
    east  = R * (np.deg2rad(lon) - math.radians(origin_lon)) * math.cos(math.radians(origin_lat))
    north = R * (np.deg2rad(lat) - math.radians(origin_lat))
    return east, north


def main():
    parser = argparse.ArgumentParser(description="Overlay 2D plot of four forward runs.")
    parser.add_argument("--output", type=Path,
                        default=_REPO_ROOT / "visualizations" / "forward_runs_overlay.png")
    args = parser.parse_args()

    runs = []
    for label, csv_path in FORWARD_CSVS:
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found — skipping")
            continue
        df = _load_csv(csv_path)
        runs.append((label, df))
        print(f"Loaded {label}: {len(df)} rows")

    if not runs:
        print("No CSV files found.")
        return

    # Shared ENU origin = first GPS point of the first run
    origin_lat = float(runs[0][1]["Lat"].iloc[0])
    origin_lon = float(runs[0][1]["Lon"].iloc[0])
    print(f"\nCommon ENU origin: Lat={origin_lat:.6f}, Lon={origin_lon:.6f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    legend_handles = []

    for (label, df), color in zip(runs, COLORS):
        east, north = _lla_to_enu(
            df["Lat"].values.astype(float),
            df["Lon"].values.astype(float),
            origin_lat, origin_lon,
        )
        ax.plot(east, north, color=color, linewidth=1.8, alpha=0.85)
        ax.plot(east[0],  north[0],  "o", color=color, markersize=7, zorder=5)
        ax.plot(east[-1], north[-1], "*", color=color, markersize=11, zorder=5)
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))

    legend_handles += [
        mlines.Line2D([], [], marker="o", color="grey", linestyle="none",
                      markersize=7, label="Start"),
        mlines.Line2D([], [], marker="*", color="grey", linestyle="none",
                      markersize=11, label="End"),
    ]

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title("Eno River Drifter — All Four Forward Passes", fontsize=14, fontweight="bold")
    ax.legend(handles=legend_handles, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
