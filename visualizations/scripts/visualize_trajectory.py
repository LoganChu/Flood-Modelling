#!/usr/bin/env python3
"""
2D trajectory visualization tool (no Isaac Sim required).

Usage:
    python src/visualize_trajectory.py [--csv <path>] [--output <path>]

Loads cleaned sensor data, converts to local ENU coordinates, and plots
the drifter trajectory colored by speed and time.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis

# Add extension root to path
_EXT_ROOT = Path(__file__).resolve().parent
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from drifter_vis.data_loader import DrifterDataLoader
from drifter_vis.geo_converter import GeoConverter


def visualize_trajectory(csv_path: Path, output_path: Path | None = None):
    """
    Load sensor data, clean it, convert to ENU, and plot 2D trajectories.

    Args:
        csv_path: Path to input CSV
        output_path: Optional path to save figure (e.g., 'trajectory.png')
    """
    print(f"Loading CSV: {csv_path}")

    # Load and clean sensor data
    loader = DrifterDataLoader()
    df = loader.load(csv_path)
    print(f"  Loaded {len(df)} samples after dropout removal")
    print(f"  Time range: {df['time_s'].min():.1f} to {df['time_s'].max():.1f} seconds")
    print(f"  Speed range: {df['speed_ms'].min():.2f} to {df['speed_ms'].max():.2f} m/s")

    # Convert to local ENU coordinates
    print("\nConverting to local ENU coordinate system...")
    converter = GeoConverter(use_pyproj=True)
    geo_result = converter.convert(df)
    print(f"  Origin (first GPS point): Lat={df['Lat'].iloc[0]:.6f}, Lon={df['Lon'].iloc[0]:.6f}")
    print(f"  Extent: {geo_result.enu_east.max() - geo_result.enu_east.min():.1f} m (east) x "
          f"{geo_result.enu_north.max() - geo_result.enu_north.min():.1f} m (north)")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =========================================================================
    # Plot 1: Trajectory colored by speed (left)
    # =========================================================================
    ax = axes[0]

    speed_ms = df['speed_ms'].values
    norm = Normalize(vmin=speed_ms.min(), vmax=speed_ms.max())
    cmap = ScalarMappable(norm=norm, cmap=viridis)

    # Plot segments
    for i in range(len(geo_result.enu_east) - 1):
        ax.plot(
            geo_result.enu_east[i:i+2],
            geo_result.enu_north[i:i+2],
            color=cmap.to_rgba(speed_ms[i]),
            linewidth=2,
            solid_capstyle='round'
        )

    # Start and end markers
    ax.plot(geo_result.enu_east[0], geo_result.enu_north[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(geo_result.enu_east[-1], geo_result.enu_north[-1], 'r*', markersize=15, label='End', zorder=5)

    ax.set_xlabel('East (m)', fontsize=11)
    ax.set_ylabel('North (m)', fontsize=11)
    ax.set_title('River Drifter Trajectory\n(colored by GPS speed)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper left')

    # Colorbar for speed
    cbar = plt.colorbar(cmap, ax=ax, pad=0.02)
    cbar.set_label('Speed (m/s)', fontsize=10)

    # =========================================================================
    # Plot 2: Trajectory colored by time (right)
    # =========================================================================
    ax = axes[1]

    time_s = df['time_s'].values
    norm_time = Normalize(vmin=time_s.min(), vmax=time_s.max())
    cmap_time = ScalarMappable(norm=norm_time, cmap='plasma')

    # Plot segments
    for i in range(len(geo_result.enu_east) - 1):
        ax.plot(
            geo_result.enu_east[i:i+2],
            geo_result.enu_north[i:i+2],
            color=cmap_time.to_rgba(time_s[i]),
            linewidth=2,
            solid_capstyle='round'
        )

    # Start and end markers
    ax.plot(geo_result.enu_east[0], geo_result.enu_north[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(geo_result.enu_east[-1], geo_result.enu_north[-1], 'r*', markersize=15, label='End', zorder=5)

    ax.set_xlabel('East (m)', fontsize=11)
    ax.set_ylabel('North (m)', fontsize=11)
    ax.set_title('River Drifter Trajectory\n(colored by elapsed time)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper left')

    # Colorbar for time
    cbar = plt.colorbar(cmap_time, ax=ax, pad=0.02)
    cbar.set_label('Time (seconds)', fontsize=10)

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Figure saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cleaned and converted drifter trajectory (2D)."
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'enoFeb16th.csv',
        help='Input CSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'visualizations' / 'trajectory_2d.png',
        help='Output image file (default: visualizations/trajectory_2d.png)'
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    visualize_trajectory(args.csv, args.output)


if __name__ == '__main__':
    main()
