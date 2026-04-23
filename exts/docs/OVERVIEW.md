# Developer Architecture Guide

## What this is

An NVIDIA Isaac Sim extension that implements a robotics-grade digital twin of a GPS/IMU river
drifter. The primary engineering value is trajectory reconstruction and EKF-based state estimation
from real sensor data — not the 3-D render. The Isaac Sim USD stage is the ground-truth inspection
environment.

**Robotics domains covered:** Localization, State Estimation, Sensor Fusion, Perception.

---

## Quick start

### Script Editor

```
1. Open Isaac Sim
2. Window → Script Editor
3. Open src/run_standalone.py → click ▶ Run
4. Press Play in the timeline
```

### Extension

```toml
# ~/.nvidia-omniverse/config/omniverse.toml
[exts]
folders = ["/work/lc478/Flood-Modelling/src/exts"]
```

Restart → **River Drifter → Load Visualisation**.

---

## Module map and layer ownership

| Module | Layer | Robotics function |
|--------|-------|------------------|
| `data_loader.py` | Sensor Ingestion | Raw sensor → clean DataFrame; dropout removal, unit normalisation, segment detection |
| `geo_converter.py` | State Estimation | WGS84→ENU localization; roll/pitch/yaw from accel + GPS heading |
| `state_estimator.py` | State Estimation | EKF `[x,y,vx,vy,ax,ay]` — GPS + IMU fusion |
| `metrics.py` | Validation | RMSE, velocity error, filter improvement ratio, path curvature |
| `scene_builder.py` | Visualization | USD stage: terrain, trajectory prims, drifter mesh |
| `animator.py` | Visualization | Pre-bake USD time samples; per-frame debug-draw arrows |
| `camera_manager.py` | Visualization | Overview orbit, chase, onboard cameras |
| `ui_panel.py` | Interaction | omni.ui panel: file picker, playback, live readouts |
| `extension.py` | Orchestrator | IExt entry point; wires all layers |
| `utils.py` | Shared | Physical constants, colour maps, dependency checker |

---

## Coordinate system

```
ENU East  (+m) → USD +X
ENU Up    (+m) → USD +Y   (GPS altitude relative to origin)
ENU North (+m) → USD +Z
```

Stage: Y-up, 1 unit = 1 metre. All internal computation uses ENU metres.

### LLA → ENU conversion

Two code paths in `GeoConverter`:

| Path | Error | When used |
|------|-------|-----------|
| Spherical approx (`_lla_to_enu_spherical`) | ~3–5 m at 350 m baseline | pyproj not installed |
| ECEF via pyproj (`_lla_to_enu_pyproj`) | < 0.1 m | pyproj available |

---

## Sensor data schema

```
Lat, Lon, Sats, Speed, GPS_Alt, Dist, Time, Alt,
AX, AY, AZ, GX, GY, GZ, MX, MY, MZ, Endpoint, source_file
```

Key conventions:
- `Speed` is in **mph** — divide by 2.237 for m/s (matches MATLAB `rawToCSV.m`)
- `Time` is Arduino `millis()` in ms — may reset between concatenated segments
- `AX/AY/AZ` include gravity; AZ median ≈ −8.28 m/s² (device flat, gravity on Z)
- Segment boundary: backward jump > 1.0 s in `millis()` counter

---

## State estimation detail

### Implemented: complementary filter (geo_converter.py)

```python
# Gravity debias on AZ
g_bias_z = median(AZ)
az_debiased = AZ - g_bias_z

# Attitude
pitch = atan2(AX_debiased, sqrt(AY² + AZ_debiased²))
roll  = atan2(AY, clamp(AZ_debiased, -g, +g))
yaw   = -radians(heading_deg)   # GPS heading; 0=North, CW → USD Y-up convention
```

### Planned: EKF (state_estimator.py)

State vector: `x = [east, north, vx, vz, ax, az]` (6 × 1)

Process model (constant-acceleration kinematics):
```
east(k+1)  = east(k)  + vx(k)·dt + 0.5·ax(k)·dt²
north(k+1) = north(k) + vz(k)·dt + 0.5·az(k)·dt²
vx(k+1)    = vx(k)    + ax(k)·dt
vz(k+1)    = vz(k)    + az(k)·dt
ax(k+1)    = ax(k)                 (random walk)
az(k+1)    = az(k)
```

IMU accelerations inform the prediction step. GPS position and speed are measurement updates.
Mahalanobis-distance gate: reject GPS updates where `innovation^T S^{-1} innovation > χ²_{0.95}`.

Process noise Q and measurement noise R are tunable in `configs/default.yaml` under `ekf:`.

---

## USD baking pipeline

```
DrifterDataLoader.load()
    → DataFrame (3,553 rows × ~25 cols)
GeoConverter.convert()
    → GeoResult (ENU + USD coords + attitude arrays)
SceneBuilder.build()
    → USD stage with Cesium terrain, trajectory BasisCurves, drifter Xform
Animator.bake()
    → USD time samples for all 3,553 positions (< 1 s wall time)
    → Per-frame debug draw: velocity (blue) + acceleration (red) arrows
```

USD timecode formula: `tc = time_s × fps / speed_scale` (default 24 fps, 1.0×).

---

## Running unit tests

```bash
/work/lc478/conda_envs/isaac_sim/bin/python \
    -m pytest src/exts/duke.flood_modelling.drifter_vis/tests/ -v
```

**Expected: 33 passed.** No Isaac Sim needed — all tests run against pure Python + numpy/pandas.

Tests cover:
- GPS dropout removal (Lat=Lon=0 filter)
- Speed conversion accuracy (mph → m/s)
- Time normalisation and segment detection
- LLA→ENU accuracy (< 0.1 m with pyproj, < 5 m spherical)
- Attitude estimation (bobbing, roll/pitch/yaw)
- Full integration test with real Eno River CSV

---

## Adding a new dataset

1. Collect CSV in the same schema (see `data_loader.py` `REQUIRED_COLUMNS`)
2. Open the control panel → **Load CSV…** → select file
3. Click **Build Scene**
4. Origin is auto-derived from the first valid GPS row; all coordinates are origin-relative

---

## Optional dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `pyproj` | High-accuracy LLA→ENU (< 0.1 m) | `pip install pyproj` |
| `cesium.omniverse` | Georeferenced 3-D Tiles terrain | Omniverse Launcher |
| `omni.isaac.debug_draw` | Live velocity/acceleration arrows | Bundled with Isaac Sim |
| `scipy` | EKF linear algebra, curvature computation (Phase 3–4) | `pip install scipy` |
