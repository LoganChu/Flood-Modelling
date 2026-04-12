#!/usr/bin/env python3
"""
Interactive 3D trajectory visualization using Plotly.

Usage:
    python src/visualize_trajectory_3d.py [--csv <path>] [--output <path>]

Creates an interactive 3D trajectory visualization with drag, zoom, and rotation.
Requires: pip install plotly

The figure shows:
- X-axis: East displacement (m)
- Y-axis: Altitude / Vertical position (m)
- Z-axis: North displacement (m)
- Color: GPS speed (m/s)

Controls:
  - Drag to rotate
  - Scroll to zoom
  - Hover for coordinates
  - Click legend to toggle traces
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Error: plotly not installed. Install with: pip install plotly")
    sys.exit(1)

# Add extension root to path
_EXT_ROOT = Path(__file__).resolve().parent
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from drifter_vis.data_loader import DrifterDataLoader
from drifter_vis.geo_converter import GeoConverter


def visualize_3d_interactive(csv_path: Path, output_path: Path | None = None):
    """
    Create interactive 3D trajectory visualization.

    Args:
        csv_path: Path to input CSV
        output_path: Optional path to save figure as HTML (e.g., 'trajectory_3d.html')
    """
    print(f"Loading CSV: {csv_path}")

    # Load and clean sensor data
    loader = DrifterDataLoader()
    df = loader.load(csv_path)
    print(f"  Loaded {len(df)} samples after dropout removal")
    print(f"  Time range: {df['time_s'].min():.1f} to {df['time_s'].max():.1f} seconds")
    print(f"  Speed range: {df['speed_ms'].min():.2f} to {df['speed_ms'].max():.2f} m/s")
    print(f"  Altitude range: {df['GPS_Alt'].min():.2f} to {df['GPS_Alt'].max():.2f} m")

    # Convert to local ENU coordinates
    print("\nConverting to local ENU coordinate system...")
    converter = GeoConverter(use_pyproj=True)
    geo_result = converter.convert(df)
    print(f"  Origin (first GPS point): Lat={df['Lat'].iloc[0]:.6f}, Lon={df['Lon'].iloc[0]:.6f}")
    east_extent = geo_result.enu_east.max() - geo_result.enu_east.min()
    north_extent = geo_result.enu_north.max() - geo_result.enu_north.min()
    alt_extent = geo_result.usd_y.max() - geo_result.usd_y.min()
    print(f"  Extent: {east_extent:.1f} m (east) x {alt_extent:.1f} m (altitude) x {north_extent:.1f} m (north)")

    # Prepare data for plotly
    speed_ms = df['speed_ms'].values
    time_s = df['time_s'].values

    # Create figure
    fig = go.Figure()

    # Add trajectory line with color gradient (speed)
    fig.add_trace(go.Scatter3d(
        x=geo_result.enu_east,
        y=geo_result.usd_y,
        z=geo_result.enu_north,
        mode='lines',
        line=dict(
            color=speed_ms,
            colorscale='Viridis',
            showscale=True,
            cmin=speed_ms.min(),
            cmax=speed_ms.max(),
            colorbar=dict(
                title='Speed (m/s)',
                thickness=15,
                len=0.7,
            ),
            width=4,
        ),
        hovertemplate='<b>Position</b><br>East: %{x:.2f} m<br>Altitude: %{y:.2f} m<br>North: %{z:.2f} m<extra></extra>',
        name='Trajectory',
        showlegend=False,
    ))

    # Add start marker (green)
    fig.add_trace(go.Scatter3d(
        x=[geo_result.enu_east[0]],
        y=[geo_result.usd_y[0]],
        z=[geo_result.enu_north[0]],
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            symbol='circle',
            line=dict(color='darkgreen', width=2),
        ),
        name='Start',
        hovertemplate='<b>START</b><br>East: %{x:.2f} m<br>Altitude: %{y:.2f} m<br>North: %{z:.2f} m<extra></extra>',
        showlegend=True,
    ))

    # Add end marker (red diamond)
    fig.add_trace(go.Scatter3d(
        x=[geo_result.enu_east[-1]],
        y=[geo_result.usd_y[-1]],
        z=[geo_result.enu_north[-1]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond',
            line=dict(color='darkred', width=2),
        ),
        name='End',
        hovertemplate='<b>END</b><br>East: %{x:.2f} m<br>Altitude: %{y:.2f} m<br>North: %{z:.2f} m<extra></extra>',
        showlegend=True,
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '3D River Drifter Trajectory<br><sub>East, Altitude, North - colored by GPS speed</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16},
        },
        scene=dict(
            xaxis=dict(title='East (m)', backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(title='Altitude (m)', backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(title='North (m)', backgroundcolor='rgb(230, 230,230)'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
            ),
        ),
        width=1200,
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
        ),
    )

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"\n[OK] Interactive figure saved to: {output_path}")
        print(f"     Open in a web browser to interact with the 3D plot")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive 3D trajectory visualization (requires plotly)."
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
        default=Path(__file__).parent.parent / 'visualizations' / 'trajectory_3d_interactive.html',
        help='Output HTML file (default: visualizations/trajectory_3d_interactive.html)'
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    visualize_3d_interactive(args.csv, args.output)


if __name__ == '__main__':
    main()
