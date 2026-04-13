"""
terrain_draper.py — Drape trajectory curves and drifter onto Cesium terrain via PhysX raycasting.

After the USD scene is built with flat trajectories (Y=0), this module waits for Cesium
3D tiles to load, then fires downward raycasts to query terrain height at each point,
and updates the BasisCurves and Animator with the correct heights.

Design:
  - Warmup period (120 frames, ~2s): let Cesium tiles begin streaming
  - Pass 1–3: raycast heights, drape curves, re-bake animator
  - Each pass checks if heights changed; if not, skip re-bake and stop
  - After MAX_PASSES, unsubscribe from the update stream (zero ongoing cost)

Graceful degradation:
  - Cesium not installed → start() is a no-op
  - PhysX unavailable → raycasts fail → heights stay at 0.0
  - Tiles not loaded → raycasts miss → heights stay at 0.0 for this pass, retry next interval
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from .utils import (
    TERRAIN_CAST_ORIGIN_Y,
    TERRAIN_ABOVE_OFFSET_M,
    TERRAIN_DRAPE_WARMUP_FRAMES,
    TERRAIN_DRAPE_UPDATE_INTERVAL,
    TERRAIN_DRAPE_MAX_PASSES,
    TRAJECTORY_PATH,
)

log = logging.getLogger(__name__)

# Guard imports so this module can be imported outside Isaac Sim (unit tests, linting)
try:
    import carb
    import omni.usd
    import omni.kit.app
    from pxr import Gf, Vt, UsdGeom
    from omni.physx import get_physx_scene_query_interface
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False
    log.debug("USD/Omniverse not available — TerrainDraper in stub mode")


class TerrainDraper:
    """
    Drape BasisCurves trajectories and drifter Xform animation onto Cesium terrain.

    Uses PhysX scene query (downward raycasts) to sample terrain heights at each
    point, then updates curve Y values and re-bakes the animator.

    Parameters
    ----------
    east_arr : np.ndarray
        ENU east coordinates of points to drape (shape (N,))
    north_arr : np.ndarray
        ENU north coordinates of points to drape (shape (N,))
    animator : Animator
        Reference to the live animator; will be re-baked after draping
    curve_paths : list[tuple[str, float]]
        List of (prim_path, extra_y_offset) pairs. Each pair identifies a
        BasisCurves prim and the vertical offset to apply (for visual separation).
        Example: [(TRAJECTORY_PATH, 0.0), (TRAJECTORY_PHYSICS_PATH, 0.3), ...]
    bob_y_arr : Optional[np.ndarray]
        Optional sinusoidal bob offset array. If provided, the animator's Y is set to
        terrain_heights + base_offset + bob_y_arr. If None, bob is not restored.
    enabled : bool
        If False, start() is a no-op (Cesium/PhysX not available).
    builder : Optional[SceneBuilder]
        Reference to SceneBuilder instance; used to re-enable terrain colliders as
        Cesium tiles load (deferred initialization).
    """

    def __init__(
        self,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        animator: Optional[object] = None,
        curve_paths: Optional[list] = None,
        bob_y_arr: Optional[np.ndarray] = None,
        enabled: bool = True,
        builder: Optional[object] = None,
    ) -> None:
        self._east_arr = np.asarray(east_arr)
        self._north_arr = np.asarray(north_arr)
        self._animator = animator
        self._curve_paths = curve_paths or []
        self._bob_y_arr = bob_y_arr
        self._builder = builder
        self._enabled = enabled and _USD_AVAILABLE

        self._frame_count: int = 0
        self._pass_count: int = 0
        self._last_heights: Optional[np.ndarray] = None
        self._sub: Optional[object] = None
        self._colliders_enabled: bool = False

        log.debug(
            "TerrainDraper init: %d points, %d curves, bob=%s, enabled=%s",
            len(self._east_arr),
            len(self._curve_paths),
            "yes" if self._bob_y_arr is not None else "no",
            self._enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to the update event stream; start draping on next warmup period."""
        if not self._enabled:
            log.info("TerrainDraper: disabled (Cesium/PhysX not available)")
            return
        if self._sub is not None:
            log.warning("TerrainDraper: already started")
            return

        try:
            app = omni.kit.app.get_app()
            self._sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="drifter_terrain_drape"
            )
            log.info("TerrainDraper: started, warmup=%d frames", TERRAIN_DRAPE_WARMUP_FRAMES)
        except Exception as exc:
            log.warning("TerrainDraper: failed to start: %s", exc)

    def stop(self) -> None:
        """Unsubscribe from the update stream."""
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception as exc:
                log.debug("TerrainDraper: unsubscribe error: %s", exc)
            self._sub = None
            log.debug("TerrainDraper: stopped")

    # ------------------------------------------------------------------
    # Internal — per-frame callback
    # ------------------------------------------------------------------

    def _on_update(self, event) -> None:
        """Per-frame update callback: count frames, trigger drape passes."""
        self._frame_count += 1

        # Warmup: wait for tiles to begin streaming
        if self._frame_count < TERRAIN_DRAPE_WARMUP_FRAMES:
            return

        # Drape on specific frames during the warmup and interval
        time_since_warmup = self._frame_count - TERRAIN_DRAPE_WARMUP_FRAMES
        if time_since_warmup % TERRAIN_DRAPE_UPDATE_INTERVAL != 0:
            return

        # Stop after max passes
        if self._pass_count >= TERRAIN_DRAPE_MAX_PASSES:
            log.info("TerrainDraper: max passes (%d) reached, stopping", TERRAIN_DRAPE_MAX_PASSES)
            self.stop()
            return

        # On first drape pass, attempt to enable terrain colliders (Cesium tiles may now exist)
        if self._pass_count == 0 and self._builder and not self._colliders_enabled:
            log.debug("TerrainDraper: attempting to enable terrain colliders...")
            if self._builder.enable_terrain_colliders():
                self._colliders_enabled = True
                log.info("TerrainDraper: terrain colliders ENABLED successfully")
            else:
                log.warning("TerrainDraper: enable_terrain_colliders() returned False (prim may not exist)")
        elif self._pass_count == 0:
            log.debug("TerrainDraper: skipping collider init (builder=%s, colliders_enabled=%s)",
                     self._builder is not None, self._colliders_enabled)

        # Query terrain heights
        heights = self.query_terrain_heights()
        if heights is None:
            return

        # Skip re-bake if heights unchanged (tiles did not load new data)
        if self._last_heights is not None and np.allclose(
            heights, self._last_heights, atol=0.01
        ):
            self._pass_count += 1
            log.info(
                "TerrainDraper: heights unchanged at pass %d, skipping apply (tiles stable)",
                self._pass_count,
            )
            return

        # Heights differ — drape the curves and animator
        self._last_heights = heights.copy()
        log.info(
            "TerrainDraper: pass %d — heights changed, applying to curves/animator",
            self._pass_count + 1,
        )

        try:
            stage = omni.usd.get_context().get_stage()
            self.apply_to_curves(heights, stage)
            self.apply_to_animator(heights)
            self._pass_count += 1
            log.info("TerrainDraper: drape pass %d complete", self._pass_count)
        except Exception as exc:
            log.error("TerrainDraper: drape pass failed: %s", exc, exc_info=True)
            self._pass_count += 1

    # ------------------------------------------------------------------
    # Height query via PhysX raycasting
    # ------------------------------------------------------------------

    def query_terrain_heights(self) -> Optional[np.ndarray]:
        """
        Fire downward raycasts to sample Cesium terrain height at each point.

        Returns an array of terrain Y values (shape (N,)). Missed raycasts → 0.0.
        """
        try:
            interface = get_physx_scene_query_interface()
        except Exception as exc:
            log.warning("TerrainDraper: PhysX query interface unavailable: %s", exc)
            return None

        n = len(self._east_arr)
        heights = np.zeros(n, dtype=float)
        miss_count = 0
        hit_count = 0
        height_samples = []

        for i in range(n):
            east = float(self._east_arr[i])
            north = float(self._north_arr[i])
            try:
                hit = interface.raycast_closest(
                    carb.Float3(east, TERRAIN_CAST_ORIGIN_Y, north),
                    carb.Float3(0.0, -1.0, 0.0),
                    2000.0,
                    bothSides=True,
                )
                if hit["hit"]:
                    heights[i] = float(hit["position"][1])
                    hit_count += 1
                    # Sample first few hit heights for logging
                    if i < 5 or i == n - 1:
                        height_samples.append(f"pt{i}={heights[i]:.2f}")
                else:
                    heights[i] = 0.0
                    miss_count += 1
            except Exception as exc:
                heights[i] = 0.0
                miss_count += 1
                log.debug(
                    "TerrainDraper: raycast miss at i=%d (%.1f, %.1f): %s",
                    i,
                    east,
                    north,
                    exc,
                )

        if miss_count > 0:
            log.warning(
                "TerrainDraper: %d/%d raycast misses (tiles not yet fully loaded)",
                miss_count,
                n,
            )
        else:
            log.info(
                "TerrainDraper: %d/%d raycasts HIT — heights sample: %s",
                hit_count,
                n,
                ", ".join(height_samples),
            )
        return heights

    # ------------------------------------------------------------------
    # Apply draping to curves and animator
    # ------------------------------------------------------------------

    def apply_to_curves(self, heights: np.ndarray, stage) -> None:
        """
        Update all BasisCurves prims with terrain-draped Y values.

        For each curve, the Y value is set to:
            terrain_height + TERRAIN_ABOVE_OFFSET_M + extra_offset
        where extra_offset is specific to each curve (0.0 for GPS, 0.3 for physics, etc.)
        """
        for prim_path, extra_offset in self._curve_paths:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    log.debug("TerrainDraper: prim not found: %s", prim_path)
                    continue

                curves = UsdGeom.BasisCurves(prim)
                existing_pts = curves.GetPointsAttr().Get()
                if existing_pts is None or len(existing_pts) != len(heights):
                    log.warning(
                        "TerrainDraper: point count mismatch at %s (%s vs %d)",
                        prim_path,
                        len(existing_pts) if existing_pts else "None",
                        len(heights),
                    )
                    continue

                # Rebuild point array: keep X/Z, replace Y with terrain height
                new_pts = [
                    Gf.Vec3f(
                        float(existing_pts[i][0]),
                        float(heights[i]) + TERRAIN_ABOVE_OFFSET_M + extra_offset,
                        float(existing_pts[i][2]),
                    )
                    for i in range(len(existing_pts))
                ]
                curves.GetPointsAttr().Set(Vt.Vec3fArray(new_pts))
                log.debug(
                    "TerrainDraper: updated %d points at %s (Y = terrain + %.2f m)",
                    len(new_pts),
                    prim_path,
                    extra_offset,
                )
            except Exception as exc:
                log.error("TerrainDraper: failed to drape %s: %s", prim_path, exc, exc_info=True)

    def apply_to_animator(self, heights: np.ndarray) -> None:
        """
        Update the animator's internal Y array with terrain heights and re-bake.

        The animator's _usd_y is set to:
            terrain_height + TERRAIN_ABOVE_OFFSET_M + bob_y_arr (if provided)
        This restores the sinusoidal bobbing effect onto the drifter animation.
        """
        if self._animator is None:
            return
        if len(heights) != len(self._animator._usd_y):
            log.warning(
                "TerrainDraper: animator usd_y length mismatch (%d vs %d)",
                len(heights),
                len(self._animator._usd_y),
            )
            return

        try:
            # Terrain height + base offset + optional bob
            new_y = heights + TERRAIN_ABOVE_OFFSET_M
            if self._bob_y_arr is not None and len(self._bob_y_arr) == len(heights):
                new_y = new_y + self._bob_y_arr

            self._animator._usd_y = new_y
            self._animator.bake_animation()
            log.info(
                "TerrainDraper: re-baked animator with terrain heights (bob=%s)",
                "yes" if self._bob_y_arr is not None else "no",
            )
        except Exception as exc:
            log.error("TerrainDraper: animator update failed: %s", exc, exc_info=True)
