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
    log.info("USD/Omniverse not available — TerrainDraper in stub mode")


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

        log.info(
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
            log.info("TerrainDraper: already started")
            return

        try:
            app = omni.kit.app.get_app()
            self._sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="drifter_terrain_drape"
            )
            log.info("TerrainDraper: started, warmup=%d frames", TERRAIN_DRAPE_WARMUP_FRAMES)
        except Exception as exc:
            log.info("TerrainDraper: failed to start: %s", exc)

    def stop(self) -> None:
        """Unsubscribe from the update stream."""
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception as exc:
                log.info("TerrainDraper: unsubscribe error: %s", exc)
            self._sub = None
            log.info("TerrainDraper: stopped")

    def run_drape_pass(self) -> bool:
        """
        Run a single drape pass immediately, bypassing the warmup timer.

        Enables terrain colliders on first call if not already done.
        Returns True if heights were queried and applied, False on failure.
        """
        if not self._enabled:
            log.info("TerrainDraper: disabled, skipping manual pass")
            return False

        if not self._colliders_enabled and self._builder:
            log.info("TerrainDraper: enabling terrain colliders for manual pass...")
            if self._builder.enable_terrain_colliders():
                self._colliders_enabled = True
                log.info("TerrainDraper: terrain colliders enabled")

        heights = self.query_terrain_heights()
        if heights is None:
            log.info("TerrainDraper: manual pass failed — query returned None")
            return False

        non_zero = int(np.count_nonzero(heights))
        log.info("TerrainDraper: manual pass — non_zero=%d/%d", non_zero, len(heights))

        try:
            stage = omni.usd.get_context().get_stage()
            self.apply_to_curves(heights, stage)
            self.apply_to_animator(heights)
            self._last_heights = heights.copy()
            self._pass_count += 1
            log.info("TerrainDraper: manual drape pass %d complete", self._pass_count)
            return True
        except Exception as exc:
            log.info("TerrainDraper: manual drape pass failed: %s", exc)
            return False

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
            log.info("TerrainDraper: attempting to enable terrain colliders...")
            if self._builder.enable_terrain_colliders():
                self._colliders_enabled = True
                log.info("TerrainDraper: terrain colliders ENABLED successfully")
            else:
                log.info("TerrainDraper: enable_terrain_colliders() returned False (prim may not exist)")
        elif self._pass_count == 0:
            log.info("TerrainDraper: skipping collider init (builder=%s, colliders_enabled=%s)",
                     self._builder is not None, self._colliders_enabled)

        # Query terrain heights
        heights = self.query_terrain_heights()
        if heights is None:
            log.info("TerrainDraper: heights query returned None at pass %d", self._pass_count + 1)
            return

        # Log height statistics
        non_zero_count = np.count_nonzero(heights)
        height_min = float(np.min(heights))
        height_max = float(np.max(heights))
        height_mean = float(np.mean(heights))
        log.info(
            "TerrainDraper: pass %d query results - min=%.2f, max=%.2f, mean=%.2f, non_zero=%d/%d",
            self._pass_count + 1,
            height_min,
            height_max,
            height_mean,
            non_zero_count,
            len(heights),
        )

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
        else:
            if self._last_heights is not None:
                diff = np.abs(heights - self._last_heights)
                log.info(
                    "TerrainDraper: heights CHANGED - max diff=%.3f, mean diff=%.3f",
                    float(np.max(diff)),
                    float(np.mean(diff)),
                )
            else:
                log.info("TerrainDraper: first pass, no previous heights to compare")

        # Heights differ — drape the curves and animator
        self._last_heights = heights.copy()
        log.info(
            "TerrainDraper: pass %d - heights changed, applying to curves/animator",
            self._pass_count + 1,
        )

        try:
            stage = omni.usd.get_context().get_stage()
            self.apply_to_curves(heights, stage)
            self.apply_to_animator(heights)
            self._pass_count += 1
            log.info("TerrainDraper: drape pass %d complete", self._pass_count)
        except Exception as exc:
            log.info("TerrainDraper: drape pass failed: %s", exc)
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
            log.info("TerrainDraper: PhysX query interface acquired successfully")
        except Exception as exc:
            log.info("TerrainDraper: PhysX query interface unavailable: %s", exc)
            return None

        log.info("TerrainDraper: colliders_enabled=%s - if False, raycasts WILL MISS all geometry", self._colliders_enabled)

        n = len(self._east_arr)
        log.info("TerrainDraper: query_terrain_heights START - casting %d rays", n)

        heights = np.zeros(n, dtype=float)
        miss_count = 0
        hit_count = 0
        height_samples = []

        # Log cast origin settings
        log.info("TerrainDraper: raycast config - origin_y=%.2f, max_distance=2000.0, bothSides=True",
                 TERRAIN_CAST_ORIGIN_Y)
        log.info("TerrainDraper: raycast direction vector = (0.0, -1.0, 0.0) - straight down on Y axis")

        # Log first few coordinate ranges
        if n > 0:
            east_min, east_max = float(np.min(self._east_arr)), float(np.max(self._east_arr))
            north_min, north_max = float(np.min(self._north_arr)), float(np.max(self._north_arr))
            log.info("TerrainDraper: trajectory coordinate ranges - east=[%.2f, %.2f], north=[%.2f, %.2f]",
                    east_min, east_max, north_min, north_max)
            log.info("TerrainDraper: raycasts will be cast from Y=%.2f downward for %.1f units (to Y=%.2f)",
                    TERRAIN_CAST_ORIGIN_Y, 2000.0, TERRAIN_CAST_ORIGIN_Y - 2000.0)

        for i in range(n):
            east = float(self._east_arr[i])
            north = float(self._north_arr[i])
            try:
                ray_origin = carb.Float3(east, TERRAIN_CAST_ORIGIN_Y, north)
                ray_direction = carb.Float3(0.0, -1.0, 0.0)

                hit = interface.raycast_closest(
                    ray_origin,
                    ray_direction,
                    2000.0,
                    bothSides=True,
                )
                if hit["hit"]:
                    hit_pos = hit["position"]
                    heights[i] = float(hit_pos[1])
                    hit_count += 1
                    # Sample first few hit heights for logging
                    if i < 5 or i == n - 1:
                        height_samples.append(f"pt{i}={heights[i]:.2f}")
                        if i < 3:
                            log.info("TerrainDraper: raycast HIT at i=%d - ray origin=(%.1f, %.1f, %.1f), hit position=(%.1f, %.1f, %.1f), Y height=%.2f",
                                    i, east, TERRAIN_CAST_ORIGIN_Y, north,
                                    float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2]),
                                    heights[i])
                else:
                    heights[i] = 0.0
                    miss_count += 1
                    if i < 3:
                        log.info("TerrainDraper: raycast MISS at i=%d - ray origin=(%.1f, %.1f, %.1f), checking direction (0, -1, 0) = straight down",
                                i, east, TERRAIN_CAST_ORIGIN_Y, north)
            except Exception as exc:
                heights[i] = 0.0
                miss_count += 1
                if i < 3:
                    log.info(
                        "TerrainDraper: raycast exception at i=%d (%.1f, %.1f): %s",
                        i,
                        east,
                        north,
                        exc,
                    )

        log.info(
            "TerrainDraper: query_terrain_heights COMPLETE - hits=%d, misses=%d, hit_rate=%.1f%%",
            hit_count,
            miss_count,
            (hit_count / n * 100) if n > 0 else 0,
        )

        if miss_count > 0:
            log.info(
                "TerrainDraper: %d/%d raycast misses (tiles not yet fully loaded)",
                miss_count,
                n,
            )
        else:
            log.info(
                "TerrainDraper: all raycasts HIT - heights sample: %s",
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
        log.info("TerrainDraper: apply_to_curves START - processing %d curve paths", len(self._curve_paths))
        for prim_path, extra_offset in self._curve_paths:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    log.info("TerrainDraper: prim not found/invalid: %s", prim_path)
                    continue

                curves = UsdGeom.BasisCurves(prim)
                existing_pts = curves.GetPointsAttr().Get()
                if existing_pts is None or len(existing_pts) != len(heights):
                    log.info(
                        "TerrainDraper: point count mismatch at %s (%s vs %d)",
                        prim_path,
                        len(existing_pts) if existing_pts else "None",
                        len(heights),
                    )
                    continue

                # Log old Y values (sample first few)
                old_y_samples = [float(existing_pts[i][1]) for i in range(min(5, len(existing_pts)))]
                log.info("TerrainDraper: %s old Y values (first 5): %s", prim_path, old_y_samples)

                # Rebuild point array: keep X/Z, replace Y with terrain height
                new_pts = [
                    Gf.Vec3f(
                        float(existing_pts[i][0]),
                        float(heights[i]) + TERRAIN_ABOVE_OFFSET_M + extra_offset,
                        float(existing_pts[i][2]),
                    )
                    for i in range(len(existing_pts))
                ]

                # Log new Y values (sample first few)
                new_y_samples = [float(new_pts[i][1]) for i in range(min(5, len(new_pts)))]
                log.info(
                    "TerrainDraper: %s new Y values (first 5): %s, offset=%.2f",
                    prim_path,
                    new_y_samples,
                    extra_offset,
                )

                curves.GetPointsAttr().Set(Vt.Vec3fArray(new_pts))
                log.info(
                    "TerrainDraper: updated %d points at %s (Y = terrain + %.2f m)",
                    len(new_pts),
                    prim_path,
                    extra_offset,
                )
            except Exception as exc:
                log.info("TerrainDraper: failed to drape %s: %s", prim_path, exc)

        log.info("TerrainDraper: apply_to_curves COMPLETE")

    def apply_to_animator(self, heights: np.ndarray) -> None:
        """
        Update the animator's internal Y array with terrain heights and re-bake.

        The animator's _usd_y is set to:
            terrain_height + TERRAIN_ABOVE_OFFSET_M + bob_y_arr (if provided)
        This restores the sinusoidal bobbing effect onto the drifter animation.
        """
        log.info("TerrainDraper: apply_to_animator START")

        if self._animator is None:
            log.info("TerrainDraper: animator is None, skipping")
            return

        if len(heights) != len(self._animator._usd_y):
            log.info(
                "TerrainDraper: animator usd_y length mismatch (%d vs %d)",
                len(heights),
                len(self._animator._usd_y),
            )
            return

        try:
            # Log old Y values (sample first few)
            old_y = self._animator._usd_y
            old_y_samples = [float(old_y[i]) for i in range(min(5, len(old_y)))]
            log.info("TerrainDraper: animator old Y values (first 5): %s", old_y_samples)

            # Terrain height + base offset + optional bob
            new_y = heights + TERRAIN_ABOVE_OFFSET_M
            if self._bob_y_arr is not None and len(self._bob_y_arr) == len(heights):
                new_y = new_y + self._bob_y_arr
                log.info("TerrainDraper: bob array applied (length=%d)", len(self._bob_y_arr))
            else:
                log.info("TerrainDraper: no bob array applied (bob_y_arr=%s)",
                        "None" if self._bob_y_arr is None else f"length mismatch ({len(self._bob_y_arr)} vs {len(heights)})")

            # Log new Y values (sample first few)
            new_y_samples = [float(new_y[i]) for i in range(min(5, len(new_y)))]
            log.info("TerrainDraper: animator new Y values (first 5): %s", new_y_samples)
            log.info("TerrainDraper: new_y stats - min=%.2f, max=%.2f, mean=%.2f",
                    float(np.min(new_y)), float(np.max(new_y)), float(np.mean(new_y)))

            self._animator._usd_y = new_y
            log.info("TerrainDraper: assigned new_y to animator._usd_y")

            self._animator.bake_animation()
            log.info(
                "TerrainDraper: re-baked animator with terrain heights (bob=%s)",
                "yes" if self._bob_y_arr is not None else "no",
            )
            log.info("TerrainDraper: apply_to_animator COMPLETE")
        except Exception as exc:
            log.info("TerrainDraper: animator update failed: %s", exc)
