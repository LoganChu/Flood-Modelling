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
7. PhysicsValidator    : (optional) simulate and bake comparison trajectory
"""

from __future__ import annotations

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
from .physics_validator import PhysicsValidator, DrifterPhysicsParams
from .ui_panel         import DrifterUIPanel
from .utils            import USD_STAGE_FPS, check_dependencies

_DEFAULT_CSV = str(
    Path(__file__).resolve().parents[2] / "data" / "enoFeb16th_smoothed.csv"
)


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
        self._validator: Optional[PhysicsValidator]  = None
        self._csv_path: str = _DEFAULT_CSV
        self._physics_mode: bool = False
        self._prebake_mode: bool = True

        deps = check_dependencies()
        log.info("Dependencies: %s", deps)

        self._register_menu()
        self._create_panel()

    def on_shutdown(self) -> None:
        log.info("DrifterVisExtension shutting down")
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
            on_load_csv        = self._on_load_csv,
            on_build           = self._build_pipeline,
            on_play            = self._on_play,
            on_pause           = self._on_pause,
            on_stop            = self._on_stop,
            on_speed           = self._on_speed_changed,
            on_seek            = self._on_seek,
            on_camera          = self._on_camera_changed,
            on_physics_toggled = self._on_physics_toggled,
        )
        self._panel.show()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _on_load_csv(self, path: str) -> None:
        if path:
            self._csv_path = path
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

        # Step 3: Build USD scene
        panel.set_status("Step 3/6: Building USD scene…")
        self._builder = SceneBuilder(
            origin_lat = meta["origin_lat"],
            origin_lon = meta["origin_lon"],
            origin_alt = meta["origin_alt"],
        )
        self._builder.build(
            east_arr  = geo.enu_east,
            north_arr = geo.enu_north,
            up_arr    = geo.enu_up,
            speeds    = df["speed_ms"].values,
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
            float(geo.usd_y.mean()),
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

        # Step 6: Physics validation (optional)
        if self._physics_mode:
            panel.set_status("Step 6/6: Running physics simulation…")
            self._validator = PhysicsValidator(params=DrifterPhysicsParams())
            sim_east, sim_north = self._validator.simulate(
                df, geo.enu_east, geo.enu_north,
            )
            discrepancy = self._validator.compute_discrepancy(
                geo.enu_east, geo.enu_north, sim_east, sim_north,
            )
            self._validator.bake_physics_trajectory(
                sim_east, sim_north, geo.enu_up,
                discrepancy, df["time_s"].values, fps=USD_STAGE_FPS,
            )
        else:
            panel.set_status("Step 6/6: Physics skipped (enable via checkbox)")

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
            "Overview":       CameraMode.OVERVIEW,
            "Chase (Follow)": CameraMode.CHASE,
            "Onboard (POV)":  CameraMode.ONBOARD,
        }
        self._cam_mgr.activate(mode_map.get(mode_name, CameraMode.OVERVIEW))

    def _on_physics_toggled(self, enabled: bool) -> None:
        self._physics_mode = enabled
        log.info("Physics validation: %s", "ON" if enabled else "OFF")
        if enabled and self._loader is not None:
            # Trigger a rebuild to include physics trajectory
            self._build_pipeline()
