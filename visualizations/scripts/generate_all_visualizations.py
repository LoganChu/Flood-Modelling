#!/usr/bin/env python3
"""
Batch generation of 2D and 3D visualizations for all datasets.

Usage:
    python src/generate_all_visualizations.py [--data-dir <path>]

Automatically discovers all CSV files in the data directory and generates
both 2D PNG and 3D interactive HTML visualizations.
"""

import sys
from pathlib import Path
import subprocess

def main():
    # Get data directory
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / 'data'
    viz_dir = repo_root / 'visualizations'

    print(f"Data directory: {data_dir}")
    print(f"Visualization directory: {viz_dir}\n")

    # Find all CSV files
    csv_files = sorted(data_dir.glob('*.csv'))

    if not csv_files:
        print("No CSV files found in data directory")
        return

    print(f"Found {len(csv_files)} CSV files:\n")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file.name}")

    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70 + "\n")

    # Generate visualizations for each CSV
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension

        # Skip if already generated (enoFeb16th files are already done as examples)
        # Actually, let's regenerate everything for consistency

        print(f"\nProcessing: {csv_file.name}")
        print("-" * 70)

        # Generate 2D visualization
        output_2d = viz_dir / f"{dataset_name}_2d.png"
        print(f"  Generating 2D visualization...")
        print(f"    Output: {output_2d.name}")

        result_2d = subprocess.run(
            [sys.executable, str(repo_root / 'src' / 'visualize_trajectory.py'),
             '--csv', str(csv_file),
             '--output', str(output_2d)],
            capture_output=True,
            text=True
        )

        if result_2d.returncode != 0:
            print(f"    ERROR: {result_2d.stderr}")
        else:
            # Extract the final status line
            for line in result_2d.stdout.split('\n'):
                if 'OK' in line or 'Loaded' in line or 'Extent' in line:
                    print(f"    {line.strip()}")

        # Generate 3D visualization
        output_3d = viz_dir / f"{dataset_name}_3d.html"
        print(f"  Generating 3D visualization...")
        print(f"    Output: {output_3d.name}")

        result_3d = subprocess.run(
            [sys.executable, str(repo_root / 'src' / 'visualize_trajectory_3d.py'),
             '--csv', str(csv_file),
             '--output', str(output_3d)],
            capture_output=True,
            text=True
        )

        if result_3d.returncode != 0:
            print(f"    ERROR: {result_3d.stderr}")
        else:
            # Extract the final status line
            for line in result_3d.stdout.split('\n'):
                if 'OK' in line or 'Loaded' in line or 'Extent' in line:
                    print(f"    {line.strip()}")

    print("\n" + "=" * 70)
    print("Visualization generation complete!")
    print("=" * 70)

    # Summary
    print(f"\nGenerated visualizations:")
    viz_files = sorted(viz_dir.glob('*.png')) + sorted(viz_dir.glob('*.html'))
    for viz_file in sorted(viz_files):
        size_mb = viz_file.stat().st_size / (1024 * 1024)
        if size_mb > 1:
            print(f"  {viz_file.name:45} {size_mb:6.1f} MB")
        else:
            size_kb = viz_file.stat().st_size / 1024
            print(f"  {viz_file.name:45} {size_kb:6.1f} KB")


if __name__ == '__main__':
    main()
