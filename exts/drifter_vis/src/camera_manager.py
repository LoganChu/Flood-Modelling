"""
camera_manager.py — Manage two viewports for the River Drifter Visualisation.

Cameras
-------
OVERVIEW      /World/Cameras/OverviewCamera
    Orbits the full path from above.  The orbit is pre-baked as USD time
    samples on the camera Xform so it plays even without a Python process.

THIRD_PERSON  /World/Drifter/DrifterXform/ThirdPersonCamera
    Child prim of the drifter — inherits its animated transform and follows
    automatically.  Local offset: 8 m behind, 3 m above, pitched −20° down
    so the drifter stays centred in frame above the terrain surface.

Camera switching uses omni.kit.viewport.utility to set the active camera in
the primary viewport.  If viewport utilities are unavailable (Script Editor
or headless), a warning is logged and the call is a no-op.
"""

from __future__ import annotations

import enum
import logging
import math
from typing import Tuple

import numpy as np

from .utils import DRIFTER_XFORM_PATH, CAMERAS_PATH, USD_STAGE_FPS

log = logging.getLogger(__name__)

try:
    from pxr import Gf, Usd, UsdGeom, Vt
    import omni.usd
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False

try:
    import omni.kit.viewport.utility as _vp_util
    _VIEWPORT_AVAILABLE = True
except ImportError:
    _VIEWPORT_AVAILABLE = False
    log.debug("omni.kit.viewport.utility not available — camera switching disabled")


class CameraMode(enum.Enum):
    OVERVIEW      = "Overview"
    THIRD_PERSON  = "Third Person"


class CameraManager:
    """
    Create and manage the three scene cameras.

    Parameters
    ----------
    overview_cam_path     : USD path to the overview camera
    third_person_cam_path : USD path to the third-person camera (child of drifter)
    centre                : (x, y, z) world-space centroid of the drifter path
    orbit_radius          : orbital radius for the overview camera (m)
    orbit_period_s        : time in seconds for one full orbit
    data_duration_s       : total duration of the drifter data (s)
    fps                   : USD stage fps
    """

    def __init__(
        self,
        overview_cam_path:      str = f"{CAMERAS_PATH}/OverviewCamera",
        third_person_cam_path:  str = f"{DRIFTER_XFORM_PATH}/ThirdPersonCamera",
        centre:                 Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orbit_radius:           float = 200.0,
        orbit_period_s:         float = 120.0,
        data_duration_s:        float = 527.0,
        fps:                    float = USD_STAGE_FPS,
    ) -> None:
        self._overview_path      = overview_cam_path
        self._third_person_path  = third_person_cam_path
        self._centre             = centre
        self._orbit_radius  = orbit_radius
        self._orbit_period  = orbit_period_s
        self._duration      = data_duration_s
        self._fps           = fps
        self._active_mode: CameraMode = CameraMode.OVERVIEW

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bake_overview_orbit(self) -> None:
        """
        Pre-bake a circular orbit animation onto the overview camera Xform.

        The orbit circles above the path centroid at a fixed height and radius.
        The animation loops seamlessly over the full data duration.
        """
        if not _USD_AVAILABLE:
            log.warning("USD not available — bake_overview_orbit() is a no-op")
            return

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._overview_path)
        if not prim.IsValid():
            log.warning("Overview camera prim not found: %s", self._overview_path)
            return

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        rotate_op    = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)

        cx, cy, cz = self._centre
        height = self._orbit_radius * 0.6
        num_samples = max(48, int(self._duration * self._fps / self._orbit_period))
        # Ensure full orbit completes — repeat if data is shorter than one orbit
        total_tc = self._duration * self._fps

        log.debug(
            "Baking overview orbit: radius=%.1fm height=%.1fm %d samples",
            self._orbit_radius, height, num_samples,
        )

        for i in range(num_samples + 1):
            frac = i / num_samples
            tc = frac * total_tc
            # Camera orbits in XZ plane (Y-up), directly above centroid
            angle = 2.0 * math.pi * frac
            ox = cx
            oz = cz
            oy = cy + height

            translate_op.Set(Gf.Vec3d(ox, oy, oz), time=tc)

            # X = -90° → camera looks straight down (-Y); Y rotates the image slowly
            rotate_op.Set(Gf.Vec3f(-90.0, math.degrees(angle), 0.0), time=tc)

        log.info("Overview orbit baked (%d samples, duration %.1f s)", num_samples, self._duration)

    def activate(self, mode: CameraMode) -> None:
        """
        Set *mode* as the active camera in the primary viewport.

        Parameters
        ----------
        mode : CameraMode enum value
        """
        self._active_mode = mode
        path = self._path_for_mode(mode)

        if not _VIEWPORT_AVAILABLE:
            log.warning(
                "Viewport utilities unavailable — camera switch to %s is a no-op", mode.value
            )
            return

        try:
            vp = _vp_util.get_active_viewport()
            if vp is not None:
                vp.set_active_camera(path)
                log.info("Active camera: %s (%s)", mode.value, path)
            else:
                log.warning("No active viewport found")
        except Exception as exc:
            log.warning("Camera switch failed: %s", exc)

    @property
    def active_mode(self) -> CameraMode:
        return self._active_mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for_mode(self, mode: CameraMode) -> str:
        return {
            CameraMode.OVERVIEW:     self._overview_path,
            CameraMode.THIRD_PERSON: self._third_person_path,
        }[mode]
