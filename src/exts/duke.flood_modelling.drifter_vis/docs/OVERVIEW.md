# River Drifter Visualisation — Developer Overview

## What this is

An Isaac Sim extension that replays real GPS/IMU data from a floating river
sensor (Eno River, NC) in a photorealistic 3D scene. The drifter follows its
exact recorded path, with live velocity and acceleration arrows, timeline
scrubbing, and an optional Newton physics comparison mode.

## Quick start (Script Editor)

1. Open Isaac Sim
2. Window → Script Editor
3. Open `src/run_standalone.py`
4. Click ▶ Run
5. Press Play in the timeline

## Quick start (Extension)

1. Add `src/exts` to the Kit extension search paths:
   Edit `~/.nvidia-omniverse/config/omniverse.toml`:
   ```toml
   [exts]
   folders = ["/work/lc478/Flood-Modelling/src/exts"]
   ```
2. Restart Isaac Sim
3. Menu → River Drifter → Load Visualisation

## Module map

| File | Purpose |
|------|---------|
| `utils.py` | Constants, colour helpers, dependency checker |
| `data_loader.py` | CSV → clean DataFrame |
| `geo_converter.py` | LLA → ENU → USD coordinates + attitude |
| `scene_builder.py` | USD stage: terrain, trajectory, drifter prim, lighting |
| `animator.py` | Pre-bake USD time samples + per-frame debug arrows |
| `camera_manager.py` | Overview orbit, chase, onboard cameras |
| `ui_panel.py` | omni.ui control panel |
| `physics_validator.py` | Newton buoyancy+drag simulation |
| `extension.py` | IExt entry point, orchestrates all modules |
| `../../run_standalone.py` | Direct Script Editor runner |

## Coordinate system

```
ENU East  → USD +X
ENU Up    → USD +Y  (GPS alt relative to origin)
ENU North → USD +Z
```
Stage: Y-up, 1 unit = 1 metre.

## CSV schema

```
Lat, Lon, Sats, Speed, GPS_Alt, Dist, Time, Alt,
AX, AY, AZ, GX, GY, GZ, MX, MY, MZ, Endpoint, source_file
```

- `Speed` is in **mph** (converted to m/s by dividing by 2.237)
- `Time` is Arduino `millis()` in milliseconds
- `AX/AY/AZ` are accelerometer readings in m/s² (include gravity ~−8.28 on AZ)

## Running unit tests

```bash
/work/lc478/conda_envs/isaac_sim/bin/python \
    -m pytest src/exts/duke.flood_modelling.drifter_vis/tests/ -v
```

Tests cover: GPS dropout removal, speed conversion, time normalisation,
LLA→ENU accuracy, bobbing, attitude, and full integration with real CSV.

## Swapping in a new river dataset

1. Collect CSV in the same schema (see above)
2. Open the control panel → Load CSV → select your file
3. Click Build Scene
4. The extension auto-detects the origin from the first valid GPS row

## Optional dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `pyproj` | Higher-accuracy LLA→ENU (< 0.1 m vs ~5 m) | `pip install pyproj` |
| `cesium.omniverse` | Georeferenced 3D Tiles terrain | Omniverse Launcher |
| `omni.isaac.debug_draw` | Live velocity/acceleration arrows | Bundled with Isaac Sim |
