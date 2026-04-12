# Trajectory Visualizations

This directory contains visualization outputs for all river drifter trajectory datasets.

## Datasets

### 1. **enoFeb16th** (Original Eno River deployment)
- **Samples**: 3,553 (GPS dropouts removed)
- **Duration**: ~36 minutes
- **Extent**: 114.2 m (E-W) × 337.0 m (N-S) × 65.7 m (altitude)
- **Speed range**: 0.00–2.45 m/s
- **Files**:
  - `enoFeb16th_2d.png` (133 KB) — Static 2D overhead view
  - `enoFeb16th_3d.html` (4.8 MB) — Interactive 3D view

### 2. **enoFeb16th_smoothed** (Pre-smoothed baseline)
- **Samples**: 3,553 (GPS dropouts removed)
- **Duration**: ~36 minutes
- **Extent**: 114.2 m (E-W) × 336.6 m (N-S) × 65.7 m (altitude)
- **Speed range**: 0.00–2.44 m/s
- **Note**: Smoothed version for comparison with EKF filter output
- **Files**:
  - `enoFeb16th_smoothed_2d.png` (132 KB)
  - `enoFeb16th_smoothed_3d.html` (4.8 MB)

### 3. **enoM30(1)** (Multi-hour deployment 1)
- **Samples**: 7,284 (GPS dropouts removed)
- **Extent**: 129.1 m (E-W) × 344.5 m (N-S) × 64.0 m (altitude)
- **Note**: Extended deployment with richer trajectory complexity
- **Files**:
  - `enoM30(1)_2d.png` (134 KB)
  - `enoM30(1)_3d.html` (5.0 MB)

### 4. **enoM30(2)** (Multi-hour deployment 2)
- **Samples**: 8,988 (GPS dropouts removed)
- **Extent**: 122.0 m (E-W) × 323.6 m (N-S) × 61.7 m (altitude)
- **Files**:
  - `enoM30(2)_2d.png` (127 KB)
  - `enoM30(2)_3d.html` (5.0 MB)

### 5. **enoM30(3)** (Multi-hour deployment 3 — DATA QUALITY ISSUE)
- **Samples**: 7,076 (GPS dropouts removed)
- **Extent**: 14,200 km (E-W) × 3,945 km (N-S) — **ANOMALOUS**
- **Note**: ⚠️ Contains anomalous GPS values; see data analysis notes
- **Files**:
  - `enoM30(3)_2d.png` (69 KB) — Shows distorted trajectory
  - `enoM30(3)_3d.html` (5.0 MB)

### 6. **enoM30(4)** (Multi-hour deployment 4)
- **Samples**: 9,532 (GPS dropouts removed)
- **Extent**: 126.3 m (E-W) × 347.0 m (N-S) × 196.3 m (altitude)
- **Note**: Largest altitude variation; possibly different deployment conditions
- **Files**:
  - `enoM30(4)_2d.png` (188 KB)
  - `enoM30(4)_3d.html` (5.1 MB)

---

## Visualization Types

### 2D Static PNG Visualizations
Shows two side-by-side overhead (plan-view) trajectories in local ENU coordinates:

- **Left plot**: Trajectory colored by GPS speed
  - Dark blue = slow water (~0 m/s)
  - Bright yellow = fast water (~2.5 m/s)
  - Reveals river current patterns
  
- **Right plot**: Trajectory colored by elapsed time
  - Dark purple = start of deployment
  - Bright yellow = end of deployment
  - Shows temporal progression

**Best for:** Reports, presentations, static documentation

### 3D Interactive HTML Visualizations
Fully interactive 3D visualization with drag, zoom, and hover capabilities:

- **Axes**: East (X), Altitude (Y), North (Z)
- **Color gradient**: GPS speed (viridis: dark blue → bright yellow)
- **Markers**:
  - Green circle = deployment start
  - Red diamond = deployment end

**Interactive controls:**
- Drag to rotate the view
- Scroll to zoom in/out
- Hover over the trajectory for exact coordinates
- Click legend items to toggle markers on/off

**Best for:** Exploration, detailed analysis, interactive inspection

**How to view:** Open the file in any web browser (Chrome, Firefox, Safari, Edge, etc.)

## Coordinate System

All visualizations use the **local ENU (East-North-Up)** coordinate frame:

- **East (X)**: Displacement from the origin in the eastward direction
- **North (Z)**: Displacement from the origin in the northward direction  
- **Altitude (Y)**: Vertical position (includes GPS altitude and sinusoidal bobbing from buoy oscillation)

The origin is set at the first valid GPS point (Eno River, North Carolina):
- Latitude: 36.078963°
- Longitude: -79.007996°

## Generation

Both visualizations are generated from the same pipeline:

1. **Data Loading**: Raw CSV ingestion with GPS dropout removal and unit normalization
2. **Geo Conversion**: WGS84 LLA → local ENU transformation with attitude estimation
3. **Visualization**: Matplotlib (2D) or Plotly (3D)

### Generate 2D Visualization
```bash
python src/visualize_trajectory.py --csv data/enoFeb16th.csv
```
Default output: `visualizations/trajectory_2d.png`

### Generate 3D Visualization
```bash
python src/visualize_trajectory_3d.py --csv data/enoFeb16th.csv
```
Default output: `visualizations/trajectory_3d_interactive.html`

Requires: `pip install plotly`

### Custom Output Location
```bash
python src/visualize_trajectory.py --csv data/enoFeb16th.csv --output my_trajectory.png
python src/visualize_trajectory_3d.py --csv data/enoFeb16th.csv --output my_trajectory.html
```

## Data Summary (Eno River Deployment)

- **Sample count**: 3,553 samples
- **Duration**: ~36 minutes (2,175 seconds)
- **Speed range**: 0.00 to 2.45 m/s
- **Altitude range**: 123.0 to 188.6 m
- **Spatial extent**:
  - East-West: 114.2 m
  - North-South: 337.0 m
  - Vertical variation: 65.7 m (including bobbing)

## Notes

- The 3D HTML file is self-contained and can be shared or emailed without needing Python or external resources
- Both visualizations respect the data cleaning and coordinate transformation pipeline (dropout removal, time normalization, etc.)
- Color gradients are consistent across both visualizations (speed-based coloring uses the "viridis" colormap)
