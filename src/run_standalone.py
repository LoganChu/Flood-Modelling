"""
run_standalone.py — Standalone runner for the River Drifter Visualisation.

Two usage modes
---------------

1. Isaac Sim Script Editor (recommended for development):
   - Open Isaac Sim → Window → Script Editor
   - File → Open this script
   - Click the Run (▶) button

2. Headless / command-line using Isaac Sim's bundled Python:
   /path/to/isaac-sim/python.sh src/run_standalone.py \\
       --csv data/enoFeb16th_smoothed.csv \\
       --fps 24 --speed 1.0 --physics

   Isaac Sim's Python has omni.*, pxr, etc. on sys.path already.
   Do NOT use the system Python — use the Kit-bundled interpreter.

   On a typical Linux install:
   ~/.local/share/ov/pkg/isaac-sim-4.x/python.sh

Coordinate system
-----------------
USD stage: Y-up, 1 unit = 1 metre
  ENU East  → USD +X
  ENU Up    → USD +Y  (GPS altitude relative to origin)
  ENU North → USD +Z

Time mapping
------------
CSV Time column = Arduino millis() in ms, normalised to seconds from 0.
USD timecode    = time_s × fps / speed_scale

At 24 fps, 1× speed: 527 s of data → 12,648 frames
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("run_standalone")

# ---------------------------------------------------------------------------
# Add the source root to sys.path so imports work regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # src/
_SRC_ROOT   = _SCRIPT_DIR

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_DEFAULT_CSV = str(_SCRIPT_DIR.parent / "data" / "enoFeb16th_smoothed.csv")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="River Drifter 3D Visualisation — standalone runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv", default=_DEFAULT_CSV,
        help="Path to drifter CSV (default: data/enoFeb16th_smoothed.csv)",
    )
    p.add_argument("--fps",   type=float, default=24.0,  help="USD stage fps (default 24)")
    p.add_argument("--speed", type=float, default=1.0,   help="Playback speed multiplier (default 1.0)")
    p.add_argument("--physics",  action="store_true",    help="Run Newton physics validation")
    p.add_argument("--live",     action="store_true",    help="Live-update mode (default: pre-bake)")
    p.add_argument("--no-ui",    action="store_true",    help="Skip omni.ui panel (headless mode)")
    p.add_argument("--verbose",  action="store_true",    help="Enable DEBUG logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    csv_path:     str   = _DEFAULT_CSV,
    fps:          float = 24.0,
    speed_scale:  float = 1.0,
    physics_mode: bool  = False,
    live_mode:    bool  = False,
    show_ui:      bool  = True,
) -> None:
    """
    Execute the full visualisation pipeline.

    Can be called directly from the Isaac Sim Script Editor without argparse::

        import sys
        sys.path.insert(0, "/work/lc478/Flood-Modelling/src")
        from run_standalone import run
        run()

    Parameters
    ----------
    csv_path     : absolute or relative path to the drifter CSV
    fps          : USD timeline frames per second
    speed_scale  : playback multiplier (0.1 – 10.0)
    physics_mode : also run and visualise the Newton simulation
    live_mode    : per-frame callback instead of pre-baking (slower)
    show_ui      : create the omni.ui control panel
    """
    from drifter_vis.utils import check_dependencies
    from drifter_vis.data_loader import DrifterDataLoader
    from drifter_vis.geo_converter import GeoConverter
    from drifter_vis.scene_builder import SceneBuilder
    from drifter_vis.animator import Animator
    from drifter_vis.camera_manager import CameraManager, CameraMode
    from drifter_vis.physics_validator import (
        PhysicsValidator, DrifterPhysicsParams,
    )

    deps = check_dependencies()
    log.info("Dependency check: %s", deps)

    # -----------------------------------------------------------------------
    # 1. Load CSV
    # -----------------------------------------------------------------------
    log.info("Step 1/6: Loading CSV: %s", csv_path)
    loader = DrifterDataLoader()
    df = loader.load(csv_path)
    meta = loader.metadata
    log.info(
        "Loaded %d rows | %d segment(s) | %.1f s",
        meta["rows"], meta["segments"], meta["duration_s"],
    )

    # -----------------------------------------------------------------------
    # 2. Convert coordinates
    # -----------------------------------------------------------------------
    log.info("Step 2/6: LLA → ENU → USD")
    converter = GeoConverter()
    geo = converter.convert(df)
    log.info(
        "ENU extent: E [%.1f, %.1f] m  N [%.1f, %.1f] m",
        geo.enu_east.min(), geo.enu_east.max(),
        geo.enu_north.min(), geo.enu_north.max(),
    )

    # -----------------------------------------------------------------------
    # 3. Build USD scene
    # -----------------------------------------------------------------------
    log.info("Step 3/6: Building USD scene")
    builder = SceneBuilder(
        origin_lat = meta["origin_lat"],
        origin_lon = meta["origin_lon"],
        origin_alt = meta["origin_alt"],
    )
    builder.build(
        east_arr  = geo.enu_east,
        north_arr = geo.enu_north,
        up_arr    = geo.enu_up,
        speeds    = df["speed_ms"].values,
    )

    # -----------------------------------------------------------------------
    # 4. Bake / activate animation
    # -----------------------------------------------------------------------
    log.info("Step 4/6: %s animation", "Live-update" if live_mode else "Pre-baking")
    animator = Animator(
        drifter_prim_path = SceneBuilder.DRIFTER_XFORM_PATH,
        usd_x    = geo.usd_x,
        usd_y    = geo.usd_y,
        usd_z    = geo.usd_z,
        roll     = geo.roll,
        pitch    = geo.pitch,
        yaw      = geo.yaw,
        time_s   = df["time_s"].values,
        ax_world = df["AX"].values,
        ay_world = df["AY"].values,
        az_world = df["AZ"].values,
        fps         = fps,
        speed_scale = speed_scale,
    )
    if live_mode:
        animator.activate_live_update()
    else:
        animator.bake_animation()

    # -----------------------------------------------------------------------
    # 5. Camera setup
    # -----------------------------------------------------------------------
    log.info("Step 5/6: Cameras")
    import numpy as np
    centre = (
        float(geo.usd_x.mean()),
        float(geo.usd_y.mean()),
        float(geo.usd_z.mean()),
    )
    extent = max(
        float(geo.usd_x.max() - geo.usd_x.min()),
        float(geo.usd_z.max() - geo.usd_z.min()),
    )
    cam_mgr = CameraManager(
        centre          = centre,
        orbit_radius    = extent * 0.7,
        orbit_period_s  = 120.0,
        data_duration_s = float(df["time_s"].iloc[-1]),
        fps             = fps,
    )
    cam_mgr.bake_overview_orbit()
    cam_mgr.activate(CameraMode.OVERVIEW)

    # -----------------------------------------------------------------------
    # 6. Physics validation (optional)
    # -----------------------------------------------------------------------
    if physics_mode:
        log.info("Step 6/6: Newton physics simulation")
        validator = PhysicsValidator(params=DrifterPhysicsParams())
        sim_east, sim_north = validator.simulate(df, geo.enu_east, geo.enu_north)
        discrepancy = validator.compute_discrepancy(
            geo.enu_east, geo.enu_north, sim_east, sim_north,
        )
        validator.bake_physics_trajectory(
            sim_east, sim_north, geo.enu_up,
            discrepancy, df["time_s"].values, fps=fps, speed_scale=speed_scale,
        )
    else:
        log.info("Step 6/6: Physics skipped (pass --physics to enable)")

    # -----------------------------------------------------------------------
    # 7. UI panel (optional)
    # -----------------------------------------------------------------------
    if show_ui:
        try:
            from drifter_vis.ui_panel import DrifterUIPanel
            import omni.timeline as _tl

            panel = DrifterUIPanel(
                on_play  = lambda: _tl.get_timeline_interface().play(),
                on_pause = lambda: _tl.get_timeline_interface().pause(),
                on_stop  = lambda: _tl.get_timeline_interface().stop(),
                on_speed = animator.set_speed_scale,
                on_camera = lambda m: cam_mgr.activate(
                    {"Overview": CameraMode.OVERVIEW,
                     "Chase (Follow)": CameraMode.CHASE,
                     "Onboard (POV)": CameraMode.ONBOARD}.get(m, CameraMode.OVERVIEW)
                ),
            )
            panel.set_dataframe(df)
            panel.set_total_frames(len(df))
            panel.set_status(
                f"Loaded {meta['rows']} rows | {meta['duration_s']:.1f}s | "
                f"{len(loader.segments)} segment(s)"
            )
            panel.show()
            log.info("UI panel opened")
        except Exception as exc:
            log.warning("UI panel skipped (not in Kit environment): %s", exc)

    log.info(
        "Pipeline complete — %d rows, %.1f s. "
        "Press Play in the timeline to start.",
        meta["rows"], meta["duration_s"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(
        csv_path     = args.csv,
        fps          = args.fps,
        speed_scale  = args.speed,
        physics_mode = args.physics,
        live_mode    = args.live,
        show_ui      = not args.no_ui,
    )
