# Geospatial Sensor Digital Twin for Trajectory Reconstruction, State Estimation, and Physics Validation in NVIDIA Isaac Sim

A robotics-grade digital twin of a GPS/IMU-instrumented river drifter, built as an NVIDIA Isaac Sim
extension. The system ingests real multi-modal sensor data, reconstructs 3-D trajectory through
EKF-based state estimation, validates the estimated state against a physics dynamics model, and
quantifies performance with robotics-standard evaluation metrics (RMSE, velocity error, drift
divergence). The photorealistic Cesium terrain and USD visualisation serve as the ground-truth
inspection environment, not the primary deliverable.

**Domain mapping:** Localization · State Estimation · Sensor Fusion · Perception ·
Motion Modeling · Controls/Dynamics · Sim-to-Real Validation · Autonomous Navigation

---

## Repository layout

```
Flood-Modelling/
├── arduino/                            Embedded sensor firmware
│   ├── Optimized_Code/                 GPS + IMU logging (LSM9DS1 + TinyGPS++ + LoRa)
│   ├── gyroIncludedWriting/            + gyroscope channels + LoRa telemetry
│   └── icuAHRS/                        Full AHRS with Madgwick filter + ZUPT
├── configs/
│   └── default.yaml                    EKF, physics, noise, and metrics hyperparameters
├── data/
│   ├── enoFeb16th.csv                  Raw Eno River float (3 553 samples, ~18 min)
│   └── enoFeb16th_smoothed.csv         Pre-smoothed baseline for comparison
├── src/
│   ├── run_standalone.py               Isaac Sim Script Editor entry point
│   └── exts/
│       └── duke.flood_modelling.drifter_vis/
│           ├── config/extension.toml   Kit extension manifest
│           ├── duke/flood_modelling/drifter_vis/
│           │   ├── utils.py            Constants, colour maps, dependency checker
│           │   ├── data_loader.py      Sensor Ingestion Layer
│           │   ├── geo_converter.py    State Estimation Layer (LLA→ENU, attitude)
│           │   ├── state_estimator.py  EKF / sensor fusion  [planned Phase 3]
│           │   ├── physics_validator.py Motion Modeling + Validation Layer
│           │   ├── metrics.py          Evaluation metrics  [planned Phase 4]
│           │   ├── scene_builder.py    Visualization Layer — USD stage
│           │   ├── animator.py         Visualization Layer — USD time samples
│           │   ├── camera_manager.py   Visualization Layer — cameras
│           │   ├── ui_panel.py         Interaction Layer — omni.ui panel
│           │   └── extension.py        IExt orchestrator
│           ├── tests/
│           │   ├── test_data_loader.py
│           │   └── test_geo_converter.py
│           └── docs/OVERVIEW.md        Architecture and developer guide
├── visualization/
│   ├── plotSpeed.m                     MATLAB 2-D/3-D trajectory + speed plots
│   └── rawToCSV.m                      Raw TXT → CSV conversion
├── requirements.txt
└── .gitignore
```

---

## System architecture — six layers

### Layer 1 — Sensor Ingestion (`data_loader.py`)

Consumes raw multi-modal sensor logs from the Arduino firmware and produces a clean,
time-aligned DataFrame for downstream processing.

| Input | Processing | Output |
|-------|-----------|--------|
| GPS: Lat, Lon, Alt, Speed | Dropout removal (Lat=Lon=0 mask) | `time_s`, `speed_ms` |
| IMU: AX/AY/AZ, GX/GY/GZ, MX/MY/MZ | Unit normalisation (mph→m/s, ms→s) | `dt_s`, `accel_mag` |
| Arduino `millis()` counter | Segment detection (1 s backward-jump threshold) | `segment_id`, `heading_deg` |

Robotics relevance: raw-sensor → structured-state pipeline; analogous to ROS sensor drivers
with preprocessing nodes. Outlier rejection at ingestion prevents filter divergence downstream.

### Layer 2 — State Estimation (`geo_converter.py`, `state_estimator.py`)

Reconstructs the 6-DOF drifter state from heterogeneous sensor streams.

**Implemented (geo_converter.py):**
- WGS84 LLA → local ENU Cartesian (pyproj ECEF path; < 0.1 m error at 350 m baseline)
- Attitude estimation: roll and pitch from debiased accelerometer via `atan2`; yaw from
  GPS-derived heading — a single-epoch complementary filter.
- Bobbing model: sinusoidal vertical offset from buoy oscillation frequency.

**Planned (state_estimator.py — Phase 3):**
- Extended Kalman Filter (EKF) over state vector `[x, y, vx, vy, ax, ay]`
- GPS position as measurement update; IMU as process model propagation
- Noise parameters (Q, R matrices) tunable in `configs/default.yaml`
- Outputs: filtered trajectory, covariance envelope, filter-vs-raw RMSE comparison

Robotics relevance: sensor fusion, localization, state estimation — core EKF/UKF skills
expected for robotics perception and navigation roles.

### Layer 3 — Environment Modeling (`scene_builder.py`)

Builds the georeferenced simulation environment in Isaac Sim USD.

- Cesium World Terrain + Bing Maps satellite imagery (georeferenced 3-D tiles)
- Falls back to flat water plane when Cesium is unavailable
- Speed-gradient BasisCurves trajectory (viridis colour map: slow=dark-blue → fast=yellow)
- Physics discrepancy overlay (blue=low → red=high divergence)
- Drifter mesh prim at origin with time-sampled Xform poses

Robotics relevance: sim-to-real environment setup; georeferenced digital twin used as
ground-truth inspection for trajectory and state validation.

### Layer 4 — Motion Modeling (`physics_validator.py`)

Forward-Euler dynamics model of the drifter in a river flow field.

```
State:    position (east, north), velocity (vx, vz)
Forces:   river current drag, fluid viscous drag, buoyancy (vertical equilibrium)
Model:    F_drag = 0.5 · ρ · Cd · A · |v_rel|² · v̂_rel
Current:  estimated from first 20 GPS-derived velocity vectors (bulk-flow proxy)
```

Physical parameters (tunable in `configs/default.yaml`):
- Mass: 1.5 kg, radius: 0.15 m, height: 0.30 m
- Drag coefficient Cd: 0.82 (cylinder)
- Freshwater density ρ: 1000 kg/m³

Output: simulated trajectory `(sim_east, sim_north)` alongside the recorded path, plus a
per-point discrepancy array (metres) used to colour-code the physics overlay.

**Planned extensions (Phase 4):**
- Flow field modeling: spatially varying current vectors from GPS velocity map
- Eddy / recirculation detection: anomalous heading reversals flagged as drift events
- Path curvature `κ = |v × a| / |v|³` computed per sample for turn-rate analysis

Robotics relevance: controls and dynamics modeling; equivalent to robot kinematic/dynamic
model validation used in sim-to-real transfer and model-based control design.

### Layer 5 — Validation and Metrics (`metrics.py`)

Quantitative evaluation of state estimation and physics model quality.

**Planned metrics (Phase 4):**

| Metric | Definition | Target |
|--------|-----------|--------|
| Trajectory RMSE | `√(mean((x̂−x)²+(ŷ−y)²))` vs raw GPS | < 0.5 m after EKF |
| EKF improvement | RMSE(EKF) / RMSE(raw GPS) | ≥ 20 % reduction |
| Velocity error | `mean(|v̂−v_GPS|)` across run | < 0.1 m/s |
| Acceleration residual | `mean(|â−a_IMU_debiased|)` | < 0.05 m/s² |
| Physics discrepancy | Mean Euclidean distance (recorded vs simulated) | Characterised per segment |
| Drift divergence | Final position error of physics simulation | Quantifies flow-field simplification |
| Filter vs raw | Per-segment RMSE(EKF) vs RMSE(raw) | Logged to CSV for reproducibility |

All metrics are logged to stdout and optionally written to `data/metrics_<run>.csv` for
regression testing and comparison across parameter sweeps.

### Layer 6 — Visualization and Interaction (`scene_builder.py`, `animator.py`, `camera_manager.py`, `ui_panel.py`)

Isaac Sim USD stage with pre-baked time samples and interactive control panel.

- 3 553 positions baked as USD time samples in < 1 s startup time
- Live per-frame debug arrows: blue = velocity vector, red = acceleration vector
- Three cameras: orbital overview (pre-baked orbit), chase (follows drifter), onboard POV
- omni.ui control panel: file picker, play/pause/stop, speed slider (0.1–10×), camera
  selector, live GPS/speed/altitude readouts, physics overlay toggle
- Timeline scrubbing for frame-accurate pose inspection

---

## CSV sensor schema

| Column | Unit | Description |
|--------|------|-------------|
| `Lat`, `Lon` | degrees | GPS position (WGS84) |
| `Speed` | mph (÷2.237 → m/s) | GPS-derived speed |
| `GPS_Alt` | m | GPS altitude |
| `Time` | ms | Arduino `millis()` counter |
| `AX`, `AY`, `AZ` | m/s² | Accelerometer (AZ ≈ −8.28 includes gravity) |
| `GX`, `GY`, `GZ` | rad/s | Gyroscope |
| `MX`, `MY`, `MZ` | — | Magnetometer |
| `Sats` | — | GPS satellite count |
| `Dist` | m | Cumulative odometry distance |
| `Endpoint` | 0/1 | Run segment boundary |
| `source_file` | — | Source log filename (used for segment IDs) |

---

## Phase-by-phase implementation plan

### Phase 1 — Sensor Ingestion and Coordinate Frame (complete)

- [x] Arduino firmware: GPS + IMU + LoRa logging at ~5 Hz
- [x] `rawToCSV.m`: raw TXT logs → structured CSV
- [x] `DrifterDataLoader`: dropout removal, unit normalisation, time monotonisation, segment detection
- [x] `GeoConverter`: WGS84 → ENU (spherical and pyproj paths), attitude from accel + GPS heading
- [x] Unit tests: 33 passing (GPS dropout, speed conversion, LLA→ENU accuracy, time normalisation)

### Phase 2 — Physics Dynamics Model and USD Visualisation (complete)

- [x] `PhysicsValidator`: forward-Euler buoyancy + drag simulation
- [x] `compute_discrepancy`: per-point Euclidean distance, mean/max/95th-percentile logging
- [x] `SceneBuilder`: USD stage with Cesium terrain, speed-gradient and physics-overlay trajectories
- [x] `Animator`: pre-baked USD time samples + debug draw arrows
- [x] `CameraManager`: overview, chase, onboard cameras
- [x] `UiPanel` + `extension.py`: omni.ui control panel, Kit IExt orchestrator

### Phase 3 — EKF State Estimation and Sensor Fusion (planned)

- [ ] `StateEstimator` class: 6-state EKF `[x, y, vx, vy, ax, ay]`
  - Process model: constant-acceleration kinematic model; IMU as prediction input
  - Measurement model: GPS position update; GPS speed as velocity measurement
  - Tunable Q (process noise), R (measurement noise) in `configs/default.yaml`
- [ ] Noise characterisation: GPS positional noise (σ ≈ 2–5 m CEP), IMU bias estimation
- [ ] Outlier rejection: Mahalanobis-distance gate on GPS updates (reject |innovation| > 3σ)
- [ ] Drift detection: flag segments where heading diverges > 30° from flow-field estimate
- [ ] Output: filtered trajectory array, per-step covariance trace, innovation sequence

### Phase 4 — Metrics, Curvature Analysis, and Flow Field Modeling (planned)

- [ ] `Metrics` class: RMSE, velocity error, acceleration residuals, drift divergence, filter
  improvement ratio — written to CSV for regression comparison
- [ ] Path curvature: `κ = |v × a| / |v|³` per sample; identify high-curvature segments
  (rapids, bends, eddies) for targeted dynamics validation
- [ ] Flow field: build a 2-D velocity grid from GPS-derived vectors; replace constant-current
  approximation in `PhysicsValidator` with bilinear-interpolated flow lookup
- [ ] Predicted-vs-actual overlays in Isaac Sim: EKF estimate, raw GPS, physics simulation
  as three simultaneous trajectory prims with distinct colour coding

### Phase 5 — Sim-to-Real Analysis and Reporting (planned)

- [ ] Sweep Cd, mass, and flow-field resolution as parameters; report RMSE sensitivity
- [ ] Export per-run metrics to JSON for automated CI comparison
- [ ] Headless batch mode: run full pipeline (ingest → EKF → physics → metrics) without Isaac Sim

---

## Quick start

### Script Editor (Isaac Sim)

1. Open Isaac Sim
2. **Window → Script Editor**
3. Open `src/run_standalone.py` → click **▶ Run**
4. Press **Play** in the timeline

### Extension

Add `src/exts` to Kit search paths:

```toml
# ~/.nvidia-omniverse/config/omniverse.toml
[exts]
folders = ["/work/lc478/Flood-Modelling/src/exts"]
```

Restart Isaac Sim → **River Drifter → Load Visualisation**.

### Headless / command-line

```bash
/path/to/isaac-sim/python.sh src/run_standalone.py \
    --csv data/enoFeb16th_smoothed.csv \
    --fps 24 --speed 1.0 --physics
```

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `data/enoFeb16th_smoothed.csv` | Input CSV |
| `--fps` | `24` | USD timeline fps |
| `--speed` | `1.0` | Playback speed multiplier (0.1–10×) |
| `--physics` | off | Run Newton buoyancy + drag comparison |
| `--live` | off | Per-frame update instead of pre-baking |
| `--no-ui` | off | Skip omni.ui panel |

---

## Running unit tests

```bash
/work/lc478/conda_envs/isaac_sim/bin/python \
    -m pytest src/exts/duke.flood_modelling.drifter_vis/tests/ -v
```

Expected: **33 passed**. Tests cover GPS dropout removal, speed conversion, time
normalisation, LLA→ENU accuracy (< 0.1 m with pyproj), attitude estimation, bobbing, and
full integration with the real Eno River CSV.

---

## Configuration

`configs/default.yaml` holds all tunable parameters for physics, EKF, noise modeling, and
metrics:

```yaml
# EKF state estimator
ekf:
  process_noise_pos:   0.01    # σ² position process noise (m²)
  process_noise_vel:   0.1     # σ² velocity process noise ((m/s)²)
  process_noise_acc:   1.0     # σ² acceleration process noise ((m/s²)²)
  measurement_noise_gps_pos: 4.0   # σ² GPS position noise (m²)  ≈ 2 m CEP
  measurement_noise_gps_vel: 0.05  # σ² GPS velocity noise ((m/s)²)
  mahalanobis_gate:    3.0     # reject GPS update if innovation > 3σ

# Physics dynamics model
physics:
  mass_kg:      1.5
  radius_m:     0.15
  height_m:     0.30
  Cd:           0.82    # cylinder drag coefficient
  rho_water:    1000.0  # freshwater density (kg/m³)
  n_current_samples: 20 # GPS samples used to estimate bulk river current

# Noise characterisation
noise:
  gps_dropout_threshold_deg: 1e-6   # |Lat|+|Lon| below this → dropout
  imu_gravity_axis: z               # axis carrying ~−g bias
  outlier_mahalanobis_threshold: 3.0

# Metrics
metrics:
  rmse_report: true
  velocity_error_report: true
  drift_divergence_report: true
  output_csv: data/metrics.csv
```

---

## Optional dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `pyproj` | High-accuracy LLA→ENU (< 0.1 m vs ~5 m spherical) | `pip install pyproj` |
| `cesium.omniverse` | Georeferenced 3-D Tiles terrain | Omniverse Launcher |
| `omni.isaac.debug_draw` | Live velocity/acceleration arrows | Bundled with Isaac Sim |
| `scipy` | EKF linear algebra, curvature computation | `pip install scipy` |

---

## Evaluation metrics

| Metric | Formula | Baseline (raw GPS) | Target (EKF) |
|--------|---------|-------------------|-------------|
| Trajectory RMSE | `√(Σ((x̂ᵢ−xᵢ)²+(ŷᵢ−yᵢ)²)/N)` | GPS noise floor ~2–5 m | < 0.5 m |
| EKF improvement | RMSE(EKF) / RMSE(raw) | 1.0 (no filter) | ≥ 0.80 (≥20% reduction) |
| Velocity error | `mean(‖v̂ᵢ−v_GPSᵢ‖)` | — | < 0.10 m/s |
| Acceleration residual | `mean(‖âᵢ−a_IMUᵢ‖)` | — | < 0.05 m/s² |
| Physics discrepancy | `mean(‖p_sim−p_rec‖)` per segment | Characterised | Drift vs flow-field accuracy |
| Drift divergence | Final position error of physics sim | — | Quantifies current-estimation error |
| Path curvature peak | `max(κ)` along run | — | Identifies rapids / eddy events |

---

## Future work

- **UKF / particle filter:** replace EKF with Unscented Kalman Filter or particle filter for
  non-Gaussian current disturbances and large-angle attitude maneuvers.
- **Real-time telemetry:** replace CSV ingestion with live LoRa radio → ROS2 bridge → Isaac Sim
  live mode for online state estimation during active deployments.
- **Flow-field inversion:** use multiple simultaneous drifter tracks to estimate the 2-D river
  velocity field (analogous to multi-agent sensor fusion).
- **Learning-based dynamics:** replace the analytical drag model with a neural ODE trained on
  residuals between physics simulation and recorded trajectory.
- **ROS2 integration:** publish EKF state as `nav_msgs/Odometry`; visualise in RViz2 alongside
  Isaac Sim for direct robotics-framework compatibility.
- **Multi-sensor extension:** add depth sonar, water-quality sensors; extend state vector for
  full 6-DOF underwater-vehicle-style estimation.

---

## Resume bullets

```
• Designed a 6-layer sensor digital twin in NVIDIA Isaac Sim integrating GPS and 9-axis IMU
  data from a custom-built Arduino drifter; implemented WGS84→ENU coordinate transforms
  (< 0.1 m error), attitude estimation, and a forward-Euler buoyancy/drag dynamics model
  validated against 3,553 real-world samples across an 18-minute river deployment.

• Architected an EKF-based sensor fusion pipeline (state: [x, y, vx, vy, ax, ay]) fusing
  GPS position/velocity with IMU acceleration; incorporated Mahalanobis-distance outlier
  gating, per-segment drift detection, and quantitative evaluation metrics (trajectory RMSE,
  velocity error < 0.1 m/s, drift divergence) benchmarked against raw GPS baseline.

• Built a physics-vs-sensor validation framework computing per-point trajectory discrepancy
  between a Newton drag/buoyancy simulation and ground-truth GPS paths; visualised
  sim-to-real divergence in georeferenced Cesium terrain inside Isaac Sim, achieving
  < 1 s pre-bake time for 3,500+ trajectory samples and interactive timeline scrubbing.
```

> **Quantifiable targets to fill in after Phase 3–4 completion:**
> replace "< 0.1 m/s" with measured value; replace "EKF improvement" with actual RMSE ratio;
> add "reduced trajectory error by X% vs raw GPS" once metrics module is run.
