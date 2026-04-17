"""
animator.py — Apply time-sampled animation to the drifter Xform and draw
              live velocity/acceleration arrows each frame.

Two modes
---------
Pre-bake (default, recommended)
    Writes USD time samples for all 3554 data points at startup (~< 1 s).
    After baking, playback is entirely GPU-driven — zero Python overhead.
    The stage can be saved as .usd and re-opened without re-running the pipeline.

Live update (debugging)
    An omni.kit.app update-event subscription fires each frame, looks up the
    current timeline position, and moves the drifter.  Slower but useful for
    iterating on arrow appearance.

Debug draw arrows are always live (omni.isaac.debug_draw cannot be baked into
USD time samples).  They update in the per-frame callback regardless of mode.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

from .utils import (
    DRIFTER_XFORM_PATH, ARROW_SCALE, ACCEL_ARROW_SCALE,
    COLOUR_VELOCITY, COLOUR_ACCEL, USD_STAGE_FPS,
)

log = logging.getLogger(__name__)

try:
    from pxr import Gf, Usd, UsdGeom, Vt
    import omni.usd
    import omni.timeline
    import omni.kit.app
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False
    log.debug("USD/Omniverse not available — Animator in stub mode")

try:
    from omni.isaac.debug_draw import _debug_draw
    _DEBUG_DRAW_AVAILABLE = True
except ImportError:
    _DEBUG_DRAW_AVAILABLE = False
    log.debug("omni.isaac.debug_draw not available — arrows disabled")


class Animator:
    """
    Drive the drifter Xform with sensor data and draw debug arrows.

    Parameters
    ----------
    drifter_prim_path : USD path to the animated Xform (default DRIFTER_XFORM_PATH)
    usd_x/y/z        : USD stage coordinates (N,) arrays
    roll/pitch/yaw   : attitude angles in radians (N,) arrays
    time_s           : monotone timestamps in seconds (N,) array
    ax_world/ay_world/az_world : acceleration in world frame (N,) for arrows
    fps              : USD stage fps
    speed_scale      : playback speed multiplier (1.0 = real time)
    """

    def __init__(
        self,
        drifter_prim_path: str = DRIFTER_XFORM_PATH,
        usd_x: Optional[np.ndarray] = None,
        usd_y: Optional[np.ndarray] = None,
        usd_z: Optional[np.ndarray] = None,
        roll:  Optional[np.ndarray] = None,
        pitch: Optional[np.ndarray] = None,
        yaw:   Optional[np.ndarray] = None,
        time_s: Optional[np.ndarray] = None,
        ax_world: Optional[np.ndarray] = None,
        ay_world: Optional[np.ndarray] = None,
        az_world: Optional[np.ndarray] = None,
        fps: float = USD_STAGE_FPS,
        speed_scale: float = 1.0,
    ) -> None:
        self._path = drifter_prim_path
        self._usd_x = np.asarray(usd_x) if usd_x is not None else np.array([])
        self._usd_y = np.asarray(usd_y) if usd_y is not None else np.array([])
        self._usd_z = np.asarray(usd_z) if usd_z is not None else np.array([])
        self._roll  = np.asarray(roll)  if roll  is not None else np.zeros(len(self._usd_x))
        self._pitch = np.asarray(pitch) if pitch is not None else np.zeros(len(self._usd_x))
        self._yaw   = np.asarray(yaw)   if yaw   is not None else np.zeros(len(self._usd_x))
        self._time_s = np.asarray(time_s) if time_s is not None else np.array([])
        self._ax = np.asarray(ax_world) if ax_world is not None else np.zeros(len(self._usd_x))
        self._ay = np.asarray(ay_world) if ay_world is not None else np.zeros(len(self._usd_x))
        self._az = np.asarray(az_world) if az_world is not None else np.zeros(len(self._usd_x))
        self._fps = fps
        self._speed_scale = speed_scale
        self._sub = None           # update-event subscription (live mode)
        self._draw = None          # debug_draw interface
        self._last_row: int = 0    # cached row index for live mode

        if _DEBUG_DRAW_AVAILABLE:
            self._draw = _debug_draw.acquire_debug_draw_interface()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bake_animation(self) -> None:
        """
        Write all USD time samples to the drifter Xform.

        Called once at scene-build time.  Re-call after changing speed_scale.
        """
        if not _USD_AVAILABLE:
            log.warning("USD not available — bake_animation() is a no-op")
            return
        if len(self._time_s) == 0:
            log.warning("No data — bake_animation() skipped")
            return

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._path)
        if not prim.IsValid():
            log.error("Drifter prim not found at %s", self._path)
            return

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()

        translate_op  = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

        n = len(self._time_s)
        log.info("Baking %d animation frames at %.1f fps (speed=%.1f×)…", n, self._fps, self._speed_scale)

        for i in range(n):
            tc = self._time_s[i] * self._fps / self._speed_scale
            translate_op.Set(
                Gf.Vec3d(float(self._usd_x[i]), float(self._usd_y[i]), float(self._usd_z[i])),
                time=tc,
            )

        # Set stage timeline range
        last_tc = float(self._time_s[-1]) * self._fps / self._speed_scale
        stage.SetStartTimeCode(0.0)
        stage.SetEndTimeCode(last_tc)
        stage.SetTimeCodesPerSecond(self._fps)

        log.info("Animation baked: 0 → %.1f timecodes (%.1f s)", last_tc, float(self._time_s[-1]))

        # Always register live debug-draw callback (can't bake arrows into USD)
        self._register_draw_callback()

    def activate_live_update(self) -> None:
        """
        Use per-frame update callback instead of baked time samples.

        Slower but useful for debugging the arrow appearance.
        """
        if not _USD_AVAILABLE:
            log.warning("USD not available — activate_live_update() is a no-op")
            return
        self._register_draw_callback()
        log.info("Live update mode active")

    def set_speed_scale(self, scale: float) -> None:
        """Change playback speed (call bake_animation() again to apply in pre-bake mode)."""
        self._speed_scale = max(0.1, min(10.0, scale))

    def deactivate(self) -> None:
        """Unsubscribe from the update stream."""
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None

    # ------------------------------------------------------------------
    # Internal — live update / debug draw
    # ------------------------------------------------------------------

    def _register_draw_callback(self) -> None:
        if not _USD_AVAILABLE:
            return
        app = omni.kit.app.get_app()
        self._sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="drifter_vis_update"
        )

    def _on_update(self, event) -> None:
        """Per-frame callback: move drifter (live mode) + draw debug arrows."""
        if len(self._time_s) == 0:
            return

        timeline = omni.timeline.get_timeline_interface()
        current_tc = timeline.get_current_time() * self._fps

        # Map timecode → data row
        data_time = current_tc / self._fps * self._speed_scale
        row = int(np.searchsorted(self._time_s, data_time, side="right")) - 1
        row = max(0, min(row, len(self._time_s) - 1))
        self._last_row = row

        # Live transform update (only needed in live mode, not pre-bake)
        # In pre-bake mode the USD handles positioning; we still draw arrows.
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._path)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            # Only update if no time-sampled ops exist (live mode guard)
            ops = xformable.GetOrderedXformOps()
            if not ops or not ops[0].GetTimeSamples():
                self._set_transform(xformable, row, Usd.TimeCode.Default())

        self._draw_arrows(row)

    def _set_transform(self, xformable, row: int, time) -> None:
        xformable.ClearXformOpOrder()
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(Gf.Vec3d(float(self._usd_x[row]), float(self._usd_y[row]), float(self._usd_z[row])), time)

    def _draw_arrows(self, row: int) -> None:
        """Draw velocity (blue) and acceleration (red) arrows via debug_draw."""
        if self._draw is None:
            return

        self._draw.clear_lines()

        px = float(self._usd_x[row])
        py = float(self._usd_y[row])
        pz = float(self._usd_z[row])

        # ---- Velocity arrow (blue) — direction from yaw, length = speed ----
        yaw_r = float(self._yaw[row])
        # USD Y-up: East=+X, North=+Z, yaw 0 = North
        # velocity direction: vx = sin(-yaw) = -sin(yaw), vz = cos(yaw) (toward North)
        # (yaw = -heading_deg in radians, where 0=North, 90=East)
        # But we store yaw = -heading_rad, so heading 0 → yaw 0 → pointing +Z (North)
        heading_rad = -yaw_r
        vx = math.sin(heading_rad)
        vz = math.cos(heading_rad)

        # Infer speed from position delta if possible
        speed = 0.0
        if row > 0:
            dx = float(self._usd_x[row]) - float(self._usd_x[row-1])
            dz = float(self._usd_z[row]) - float(self._usd_z[row-1])
            dt = float(self._time_s[row]) - float(self._time_s[row-1])
            if dt > 0:
                speed = math.sqrt(dx*dx + dz*dz) / dt

        arrow_len = speed * ARROW_SCALE
        ex = px + vx * arrow_len
        ey = py
        ez = pz + vz * arrow_len

        r, g, b = COLOUR_VELOCITY
        self._draw.draw_lines(
            [(px, py, pz)], [(ex, ey, ez)],
            [(r, g, b, 1.0)], [3.0],
        )

        # ---- Acceleration arrow (red) — raw sensor axes ----
        ax = float(self._ax[row]) * ACCEL_ARROW_SCALE
        ay = float(self._ay[row]) * ACCEL_ARROW_SCALE
        az_ = float(self._az[row]) * ACCEL_ARROW_SCALE
        r2, g2, b2 = COLOUR_ACCEL
        self._draw.draw_lines(
            [(px, py, pz)],
            [(px + ax, py + ay, pz + az_)],
            [(r2, g2, b2, 1.0)], [3.0],
        )
