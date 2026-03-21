# Flood-Modelling

A research platform for real-time river monitoring and flood analysis. The project combines:

- **Embedded sensor firmware** — Arduino sketches for a GPS/IMU floating drifter (LSM9DS1 + TinyGPS++ + LoRa) that logs position, velocity, and attitude to SD card.
- **3D visualisation** — a production-ready Isaac Sim extension that replays the exact sensor path in a georeferenced photorealistic river scene, with live vector overlays, timeline scrubbing, and Newton physics validation.
- **Data processing** — MATLAB and Python pipelines for converting raw sensor logs to clean CSV and visualising trajectory/speed data.

---

## Repository layout

```
Flood-Modelling/
├── arduino/                        Embedded firmware for the river drifter
│   ├── Optimized_Code/             Core GPS + IMU logging sketch
│   ├── gyroIncludedWriting/        Adds gyroscope and LoRa radio telemetry
│   └── icuAHRS/                    Full AHRS with Madgwick filter + ZUPT
├── configs/
│   └── default.yaml                Physics and ML hyperparameters
├── data/
│   ├── enoFeb16th.csv              Raw Eno River float data (3 553 rows)
│   └── enoFeb16th_smoothed.csv     Smoothed version of the same run
├── src/
│   ├── run_standalone.py           Isaac Sim Script Editor entry point
│   └── exts/
│       └── duke.flood_modelling.drifter_vis/
│           ├── config/
│           │   └── extension.toml  Kit extension manifest
│           ├── duke/flood_modelling/drifter_vis/
│           │   ├── utils.py        Constants, colour helpers, dependency checker
│           │   ├── data_loader.py  CSV → clean pandas DataFrame
│           │   ├── geo_converter.py LLA → ENU → USD coordinates + attitude
│           │   ├── scene_builder.py USD stage: terrain, trajectory, drifter prim
│           │   ├── animator.py     Pre-bake USD time samples + debug draw arrows
│           │   ├── camera_manager.py Overview / chase / onboard cameras
│           │   ├── ui_panel.py     omni.ui control panel
│           │   ├── physics_validator.py Newton buoyancy + drag simulation
│           │   └── extension.py    IExt orchestrator + menu item
│           ├── tests/
│           │   ├── test_data_loader.py
│           │   └── test_geo_converter.py
│           └── docs/OVERVIEW.md    Developer guide
├── visualization/
│   ├── plotSpeed.m                 MATLAB 2D/3D trajectory + speed visualisation
│   └── rawToCSV.m                  Convert raw TXT sensor logs to CSV
├── requirements.txt
└── .gitignore
```

---

## CSV data schema

Both CSV files share the same column layout, produced by the Arduino firmware and `rawToCSV.m`:

| Column | Unit | Description |
|--------|------|-------------|
| `Lat`, `Lon` | degrees | GPS position |
| `Speed` | **mph** | GPS-derived speed (divide by 2.237 for m/s) |
| `GPS_Alt` | m | GPS altitude |
| `Time` | ms | Arduino `millis()` counter |
| `AX`, `AY`, `AZ` | m/s² | Accelerometer (includes gravity ≈ −8.28 on AZ) |
| `GX`, `GY`, `GZ` | rad/s | Gyroscope |
| `MX`, `MY`, `MZ` | — | Magnetometer |
| `Sats` | — | GPS satellite count |
| `Dist` | m | Cumulative distance |
| `Endpoint` | 0/1 | Run boundary marker |
| `source_file` | — | Original log filename |

---

## Isaac Sim 3D Visualisation

### Quick start — Script Editor

1. Open Isaac Sim
2. **Window → Script Editor**
3. Open `src/run_standalone.py`
4. Click **▶ Run**
5. Press **Play** in the timeline

### Quick start — Extension

Add `src/exts` to the Kit extension search paths. Edit
`~/.nvidia-omniverse/config/omniverse.toml`:

```toml
[exts]
folders = ["/work/lc478/Flood-Modelling/src/exts"]
```

Restart Isaac Sim → **River Drifter → Load Visualisation**.

### Command-line (headless)

```bash
/path/to/isaac-sim/python.sh src/run_standalone.py \
    --csv data/enoFeb16th_smoothed.csv \
    --fps 24 --speed 1.0 --physics
```

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `data/enoFeb16th_smoothed.csv` | Input CSV |
| `--fps` | `24` | USD timeline fps |
| `--speed` | `1.0` | Playback speed multiplier (0.1 – 10×) |
| `--physics` | off | Run Newton buoyancy + drag comparison |
| `--live` | off | Per-frame update instead of pre-baking |
| `--no-ui` | off | Skip the omni.ui control panel |

### Features

- **Kinematic replay** — drifter follows the exact recorded GPS path; all 3 553 positions pre-baked as USD time samples in < 1 s
- **Live overlays** — blue velocity arrow, red acceleration arrow drawn each frame via `omni.isaac.debug_draw`
- **Speed-gradient trajectory** — BasisCurves coloured dark blue (slow) → yellow (fast)
- **Three cameras** — orbital overview (pre-baked orbit), chase (follows drifter), onboard POV
- **Timeline scrubbing** — drag the frame slider to jump to any point in the run
- **Cesium terrain** — georeferenced Cesium World Terrain + Bing Maps satellite imagery if the `cesium.omniverse` extension is installed; falls back to a flat water plane otherwise
- **Newton physics validation** — forward-Euler buoyancy + drag simulation runs alongside kinematic replay; discrepancy coloured orange trajectory overlay
- **Control panel** — omni.ui window with file picker, play/pause/stop, speed slider, camera selector, live GPS/speed/altitude readouts

### Optional dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `pyproj` | Higher-accuracy LLA→ENU (< 0.1 m vs ~5 m spherical) | `pip install pyproj` |
| `cesium.omniverse` | Georeferenced 3D Tiles terrain | Omniverse Launcher |

---

## Running unit tests

Tests cover data loading, GPS dropout removal, speed conversion, time normalisation, LLA→ENU accuracy, bobbing, and attitude estimation — all runnable without Isaac Sim.

```bash
/work/lc478/conda_envs/isaac_sim/bin/python \
    -m pytest src/exts/duke.flood_modelling.drifter_vis/tests/ -v
```

Expected: **33 passed**.

---

## Swapping in a new river dataset

Any CSV with the column schema above will work:

1. Open the control panel → **Load CSV…** → select your file
2. Click **Build Scene**

The extension auto-derives the georeference origin from the first valid GPS row and re-builds the full scene.

---

## Configuration

`configs/default.yaml` holds physics and ML hyperparameters used by the simulation and any future ML training runs:

```yaml
timestep_minutes: 10
physics:
  storage_coef: 0.8
  runoff_coef: 0.3
ml:
  transfer_epochs: 5
  meta_iters: 2
```
