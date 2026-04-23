"""
extension.py — Main omni.ext.IExt entry point for the River Drifter Visualisation.

Lifecycle
---------
on_startup()
    Registers the "River Drifter > Load Visualisation" menu item and creates
    the UI panel in a deferred state (no CSV loaded yet).

on_shutdown()
    Cleans up all UI, subscriptions, and references.

Pipeline (triggered by "Build Scene" button or on_build() callback)
---------
1. DrifterDataLoader   : load and clean CSV
2. GeoConverter        : LLA → ENU → USD coordinates + attitude
3. SceneBuilder        : build USD stage (terrain, trajectory, drifter prim)
4. Animator            : pre-bake USD time samples + register debug draw
5. CameraManager       : create orbit camera, activate overview
6. DrifterUIPanel      : supply DataFrame for live readouts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    import omni.ext
    import omni.kit.app
    import omni.kit.menu.utils
    import omni.timeline
    _OMNI_AVAILABLE = True
except ImportError:
    _OMNI_AVAILABLE = False
    log.debug("Omniverse not available — extension running in stub mode")

from .data_loader      import DrifterDataLoader
from .geo_converter    import GeoConverter
from .scene_builder    import SceneBuilder
from .animator         import Animator
from .camera_manager   import CameraManager, CameraMode
from .state_estimator  import StateEstimator
from .ui_panel         import DrifterUIPanel
from .terrain_draper   import TerrainDraper
from .utils            import (
    USD_STAGE_FPS, check_dependencies,
    TRAJECTORY_PATH, TRAJECTORY_EKF_PATH,
)

_DEFAULT_CSV = str(
    Path(__file__).resolve().parents[2] / "data" / "enoFeb16th_smoothed.csv"
)


def _get_config_path() -> Path:
    """Get the path to the CSV path config file."""
    return Path(__file__).resolve().parent / ".last_csv_path.json"


def _load_last_csv_path() -> Optional[str]:
    """Load the last CSV path from persistent storage, if it exists and is valid."""
    try:
        config_file = _get_config_path()
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                path = data.get("path")
                if path and Path(path).exists():
                    log.debug("Loaded last CSV path: %s", path)
                    return path
    except Exception as exc:
        log.debug("Failed to load last CSV path: %s", exc)
    return None


def _save_last_csv_path(path: str) -> None:
    """Save the CSV path to persistent storage."""
    try:
        config_file = _get_config_path()
        with open(config_file, 'w') as f:
            json.dump({"path": path}, f)
        log.debug("Saved CSV path: %s", path)
    except Exception as exc:
        log.warning("Failed to save CSV path: %s", exc)


class DrifterVisExtension(omni.ext.IExt if _OMNI_AVAILABLE else object):
    """
    Isaac Sim extension: River Drifter 3D Visualisation.

    Registered in extension.toml as duke.flood_modelling.drifter_vis.
    """

    def on_startup(self, ext_id: str) -> None:
        log.info("DrifterVisExtension starting up (ext_id=%s)", ext_id)
        self._ext_id = ext_id
        self._panel:     Optional[DrifterUIPanel]   = None
        self._loader:    Optional[DrifterDataLoader] = None
        self._builder:   Optional[SceneBuilder]     = None
        self._animator:  Optional[Animator]          = None
        self._cam_mgr:   Optional[CameraManager]     = None
        self._draper:    Optional[TerrainDraper]     = None
        # Load last CSV path if available, otherwise use default
        self._csv_path: str = _load_last_csv_path() or _DEFAULT_CSV
        self._ekf_mode:     bool = False
        self._prebake_mode: bool = True

        deps = check_dependencies()
        log.info("Dependencies: %s", deps)

        self._register_menu()
        self._create_panel()

    def on_shutdown(self) -> None:
        log.info("DrifterVisExtension shutting down")
        if self._draper:
            self._draper.stop()
            self._draper = None
        if self._animator:
            self._animator.deactivate()
        if self._panel:
            self._panel.destroy()
        self._deregister_menu()

    # ------------------------------------------------------------------
    # Menu registration
    # ------------------------------------------------------------------

    def _register_menu(self) -> None:
        if not _OMNI_AVAILABLE:
            return
        try:
            self._menu_entry = omni.kit.menu.utils.add_menu_items(
                [
                    omni.kit.menu.utils.MenuItemDescription(
                        name="Load Visualisation",
                        onclick_fn=self._on_menu_clicked,
                    )
                ],
                name="River Drifter",
            )
        except Exception as exc:
            log.warning("Menu registration failed: %s", exc)
            self._menu_entry = None

    def _deregister_menu(self) -> None:
        if not _OMNI_AVAILABLE:
            return
        try:
            if self._menu_entry:
                omni.kit.menu.utils.remove_menu_items(self._menu_entry, "River Drifter")
        except Exception:
            pass

    def _on_menu_clicked(self) -> None:
        if self._panel:
            self._panel.show()

    # ------------------------------------------------------------------
    # UI panel creation
    # ------------------------------------------------------------------

    def _create_panel(self) -> None:
        self._panel = DrifterUIPanel(
            on_load_csv    = self._on_load_csv,
            on_build       = self._build_pipeline,
            on_play        = self._on_play,
            on_pause       = self._on_pause,
            on_stop        = self._on_stop,
            on_speed       = self._on_speed_changed,
            on_seek        = self._on_seek,
            on_camera      = self._on_camera_changed,
            on_ekf_toggled = self._on_ekf_toggled,
            on_raycast     = self._on_raycast,
        )
        self._panel.show()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _on_load_csv(self, path: str) -> None:
        if path:
            self._csv_path = path
            _save_last_csv_path(path)
        self._panel.set_status(f"CSV selected: {Path(self._csv_path).name}")

    def _build_pipeline(self) -> None:
        """Run the full 7-step build pipeline."""
        panel = self._panel

        # Step 1: Load data
        panel.set_status("Step 1/6: Loading CSV…")
        self._loader = DrifterDataLoader()
        try:
            df = self._loader.load(self._csv_path)
        except Exception as exc:
            panel.set_status(f"Error loading CSV: {exc}")
            log.error("CSV load failed: %s", exc, exc_info=True)
            return

        meta = self._loader.metadata
        panel.set_dataframe(df)

        # Step 2: Convert coordinates
        panel.set_status("Step 2/6: Converting coordinates…")
        converter = GeoConverter()
        geo = converter.convert(df)

        # Step 2b: EKF state estimation (optional)
        ekf_east = ekf_north = None
        if self._ekf_mode:
            panel.set_status("Step 2b/6: Running EKF state estimator…")
            estimator = StateEstimator()
            ekf_df = estimator.run(df, geo.enu_east, geo.enu_north)
            ekf_east  = ekf_df["x_filt"].values
            ekf_north = ekf_df["y_filt"].values

        # Step 3: Build USD scene
        panel.set_status("Step 3/6: Building USD scene…")
        # Clear selection so PhysX manipulator doesn't hold stale null-prim references
        # after RemovePrim / DefinePrim calls inside SceneBuilder.build().
        if _OMNI_AVAILABLE:
            omni.usd.get_context().get_selection().clear_selected_prim_paths()
        self._builder = SceneBuilder(
            origin_lat = meta["origin_lat"],
            origin_lon = meta["origin_lon"],
            origin_alt = meta["origin_alt"],
        )
        self._builder.build(
            east_arr  = geo.enu_east,
            north_arr = geo.enu_north,
            speeds    = df["speed_ms"].values,
            ekf_east  = ekf_east,
            ekf_north = ekf_north,
        )

        # Step 4: Animation
        panel.set_status("Step 4/6: Baking animation…")
        self._animator = Animator(
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
            fps          = USD_STAGE_FPS,
            speed_scale  = 1.0,
        )
        if self._prebake_mode:
            self._animator.bake_animation()
        else:
            self._animator.activate_live_update()

        # Step 5: Cameras
        panel.set_status("Step 5/6: Setting up cameras…")
        centre = (
            float(geo.usd_x.mean()),
            0.0,
            float(geo.usd_z.mean()),
        )
        extent = max(
            float(geo.usd_x.max() - geo.usd_x.min()),
            float(geo.usd_z.max() - geo.usd_z.min()),
        )
        self._cam_mgr = CameraManager(
            centre         = centre,
            orbit_radius   = extent * 0.7,
            orbit_period_s = 120.0,
            data_duration_s = float(df["time_s"].iloc[-1]),
            fps             = USD_STAGE_FPS,
        )
        self._cam_mgr.bake_overview_orbit()
        self._cam_mgr.activate(CameraMode.OVERVIEW)

        # Step 4b: Terrain draping (post-bake, deferred via frame counter)
        panel.set_status("Step 4b/6: Scheduling terrain drape…")
        curve_paths = [(TRAJECTORY_PATH, 0.0)]
        if self._ekf_mode:
            curve_paths.append((TRAJECTORY_EKF_PATH, 0.6))

        if self._draper is not None:
            self._draper.stop()

        deps = check_dependencies()
        self._draper = TerrainDraper(
            east_arr   = geo.enu_east,
            north_arr  = geo.enu_north,
            animator   = self._animator,
            curve_paths= curve_paths,
            bob_y_arr  = geo.bob_y,
            enabled    = deps.get("cesium", False),
        )
        self._draper.start()

        panel.set_total_frames(len(df))
        panel.set_status(
            f"Ready — {meta['rows']} rows, {meta['duration_s']:.1f}s, "
            f"{len(self._loader.segments)} segment(s)"
        )

    # ------------------------------------------------------------------
    # Playback callbacks
    # ------------------------------------------------------------------

    def _on_play(self) -> None:
        if _OMNI_AVAILABLE:
            omni.timeline.get_timeline_interface().play()

    def _on_pause(self) -> None:
        if _OMNI_AVAILABLE:
            omni.timeline.get_timeline_interface().pause()

    def _on_stop(self) -> None:
        if _OMNI_AVAILABLE:
            omni.timeline.get_timeline_interface().stop()

    def _on_speed_changed(self, scale: float) -> None:
        if self._animator:
            self._animator.set_speed_scale(scale)
            if self._prebake_mode:
                self._panel.set_status(f"Re-baking at {scale:.1f}×…")
                self._animator.bake_animation()
                self._panel.set_status(f"Speed set to {scale:.1f}×")

    def _on_seek(self, frame: int) -> None:
        if not _OMNI_AVAILABLE or self._animator is None:
            return
        if frame < len(self._animator._time_s):
            t = float(self._animator._time_s[frame]) / self._animator._speed_scale
            omni.timeline.get_timeline_interface().set_current_time(t)

    def _on_camera_changed(self, mode_name: str) -> None:
        if not self._cam_mgr:
            return
        mode_map = {
            "Overview":      CameraMode.OVERVIEW,
            "Third Person":  CameraMode.THIRD_PERSON,
        }
        self._cam_mgr.activate(mode_map.get(mode_name, CameraMode.OVERVIEW))

    def _on_ekf_toggled(self, enabled: bool) -> None:
        self._ekf_mode = enabled
        log.info("EKF filtering: %s", "ON" if enabled else "OFF")
        if self._loader is not None:
            self._build_pipeline()

    def _on_raycast(self) -> None:
        if self._draper is None:
            self._panel.set_status("Build the scene first before raycasting")
            return
        submitted = self._draper.run_drape_pass()
        if submitted:
            self._panel.set_status("Raycasting terrain — results apply on next frame")
        else:
            self._panel.set_status("Raycast unavailable — Cesium not loaded")
