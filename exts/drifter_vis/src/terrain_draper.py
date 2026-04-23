"""
terrain_draper.py — Drape trajectory curves and drifter onto Cesium terrain via RTX raycasting.

After the USD scene is built with flat trajectories (Y=0), this module waits for Cesium
3D tiles to load, then fires downward RTX raycasts to query terrain height at each point,
and updates the BasisCurves and Animator with the correct heights.

RTX raycasting (omni.kit.raycast.query) queries the renderer's BVH directly, so it hits
dynamically-streamed Cesium 3D Tile meshes the moment they appear on screen — no physics
collider cooking required.

Design:
  - Warmup period (120 frames, ~2s): let Cesium tiles begin streaming into the renderer
  - Submit phase: batch-submit N downward rays via RTX sequence
  - Collect phase: poll get_latest_result each frame until results arrive (timeout ~2.5s)
  - Pass 1–3: apply heights, re-bake animator; stop when heights stabilise
  - After MAX_PASSES, clean up RTX sequence and unsubscribe (zero ongoing cost)

Graceful degradation:
  - Cesium not installed → start() is a no-op
  - omni.kit.raycast.query unavailable → start() is a no-op
  - Tiles not yet rendered → raycasts miss (height=0.0), retry next pass interval
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .utils import (
    TERRAIN_CAST_ORIGIN_Y,
    TERRAIN_ABOVE_OFFSET_M,
    TERRAIN_DRAPE_UPDATE_INTERVAL,
    DRIFTER_PATH,
    DRIFTER_XFORM_PATH,
)

log = logging.getLogger(__name__)

# Guard imports so this module can be imported outside Isaac Sim (unit tests, linting)
try:
    import omni.usd
    import omni.kit.app
    import omni.kit.raycast.query as _rtx_mod
    from pxr import Gf, Vt, UsdGeom
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False
    log.info("USD/Omniverse not available — TerrainDraper in stub mode")


class TerrainDraper:
    """
    Drape BasisCurves trajectories and drifter Xform animation onto Cesium terrain.

    Uses RTX raycasting (omni.kit.raycast.query) to sample terrain heights at each
    point, then updates curve Y values and re-bakes the animator. RTX raycasting
    works directly on the renderer's BVH — no physics colliders needed.

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
        Example: [(TRAJECTORY_PATH, 0.0), (TRAJECTORY_EKF_PATH, 0.3), ...]
    bob_y_arr : Optional[np.ndarray]
        Optional sinusoidal bob offset array. If provided, the animator's Y is set to
        terrain_heights + base_offset + bob_y_arr. If None, bob is not restored.
    enabled : bool
        If False, start() is a no-op (Cesium/RTX not available).
    """

    def __init__(
        self,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        animator: Optional[object] = None,
        curve_paths: Optional[list] = None,
        bob_y_arr: Optional[np.ndarray] = None,
        enabled: bool = True,
    ) -> None:
        self._east_arr = np.asarray(east_arr)
        self._north_arr = np.asarray(north_arr)
        self._animator = animator
        self._curve_paths = curve_paths or []
        self._bob_y_arr = bob_y_arr
        self._enabled = enabled and _USD_AVAILABLE

        self._frame_count: int = 0
        self._pass_count: int = 0
        self._last_heights: Optional[np.ndarray] = None
        self._sub: Optional[object] = None
        self._rtx_seq: Optional[int] = None
        self._rtx_submitted: bool = False
        self._rtx_submit_frame: int = 0
        self._hidden_prims: list = []
        self._pending_hide: bool = False

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
        """Subscribe to the update event stream; start draping after warmup."""
        if not self._enabled:
            log.info("TerrainDraper: disabled (Cesium/RTX not available)")
            return
        if self._sub is not None:
            log.info("TerrainDraper: already started")
            return

        try:
            app = omni.kit.app.get_app()
            self._sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="drifter_terrain_drape"
            )
            log.info("TerrainDraper: started (manual raycast mode)")
        except Exception as exc:
            log.info("TerrainDraper: failed to start: %s", exc)

    def stop(self) -> None:
        """Unsubscribe from the update stream and clean up the RTX sequence."""
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception as exc:
                log.info("TerrainDraper: unsubscribe error: %s", exc)
            self._sub = None
        self._pending_hide = False
        self._restore_draped_prims()
        self._cleanup_rtx_sequence()
        log.info("TerrainDraper: stopped")

    def run_drape_pass(self) -> bool:
        """
        Submit an RTX raycast pass immediately, bypassing the warmup timer.

        Results are collected and applied asynchronously by the update loop on
        the next frame. Returns True if submission succeeded, False if disabled.
        """
        if not self._enabled:
            log.info("TerrainDraper: disabled, skipping manual pass")
            return False

        if self._sub is None:
            self.start()

        self._hide_draped_prims()
        self._pending_hide = True
        log.info("TerrainDraper: manual RTX raycast queued (rays submit next frame after BVH update)")
        return True

    # ------------------------------------------------------------------
    # Internal — per-frame callback (submit → collect state machine)
    # ------------------------------------------------------------------

    def _on_update(self, event) -> None:
        """Per-frame callback: collect pending RTX results from a manual raycast pass."""
        self._frame_count += 1

        # Pending hide: RTX BVH has updated since last frame's hide — safe to submit now
        if self._pending_hide:
            self._pending_hide = False
            self._submit_rtx_rays()
            return

        # Collect phase: poll for results if rays are in flight
        if self._rtx_submitted:
            frames_waiting = self._frame_count - self._rtx_submit_frame
            if frames_waiting > TERRAIN_DRAPE_UPDATE_INTERVAL // 2:
                log.info(
                    "TerrainDraper: RTX collect timeout after %d frames",
                    frames_waiting,
                )
                self._rtx_submitted = False
                self._restore_draped_prims()
            else:
                heights = self._collect_rtx_results()
                if heights is not None:
                    self._rtx_submitted = False
                    self._apply_heights(heights)
                    self._restore_draped_prims()

    # ------------------------------------------------------------------
    # RTX raycast helpers
    # ------------------------------------------------------------------

    def _ensure_rtx_sequence(self) -> Optional[int]:
        """Create the RTX raycast sequence if not already created. Returns seq_id or None."""
        if self._rtx_seq is not None:
            return self._rtx_seq
        try:
            iface = _rtx_mod.acquire_raycast_query_interface()
            self._rtx_seq = iface.add_raycast_sequence()
            log.info("TerrainDraper: RTX sequence created (id=%s)", self._rtx_seq)
            return self._rtx_seq
        except Exception as exc:
            log.info("TerrainDraper: failed to create RTX sequence: %s", exc)
            return None

    def _cleanup_rtx_sequence(self) -> None:
        """Remove the RTX sequence if it exists."""
        if self._rtx_seq is None:
            return
        try:
            iface = _rtx_mod.acquire_raycast_query_interface()
            iface.remove_raycast_sequence(self._rtx_seq)
            log.info("TerrainDraper: RTX sequence removed (id=%s)", self._rtx_seq)
        except Exception as exc:
            log.info("TerrainDraper: failed to remove RTX sequence: %s", exc)
        self._rtx_seq = None
        self._rtx_submitted = False

    def _hide_draped_prims(self) -> None:
        """Make trajectory curves and drifter invisible so rays reach terrain unobstructed."""
        try:
            stage = omni.usd.get_context().get_stage()
            paths = [p for p, _ in self._curve_paths] + [DRIFTER_PATH, DRIFTER_XFORM_PATH]
            hidden = []
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(path)
            self._hidden_prims = hidden
            log.info("TerrainDraper: hid %d prims for raycast pass", len(hidden))
        except Exception as exc:
            log.info("TerrainDraper: hide prims failed: %s", exc)

    def _restore_draped_prims(self) -> None:
        """Restore visibility of prims hidden before raycasting."""
        if not self._hidden_prims:
            return
        try:
            stage = omni.usd.get_context().get_stage()
            for path in self._hidden_prims:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeVisible()
            log.info("TerrainDraper: restored %d prims", len(self._hidden_prims))
        except Exception as exc:
            log.info("TerrainDraper: restore prims failed: %s", exc)
        self._hidden_prims = []

    def _submit_rtx_rays(self) -> None:
        """Batch-submit downward RTX raycasts (prims already hidden on previous frame)."""
        seq_id = self._ensure_rtx_sequence()
        if seq_id is None:
            self._restore_draped_prims()
            return

        n = len(self._east_arr)
        rays = [
            _rtx_mod.Ray(
                (float(self._east_arr[i]), TERRAIN_CAST_ORIGIN_Y, -float(self._north_arr[i])),
                (0.0, -1.0, 0.0),
            )
            for i in range(n)
        ]

        try:
            iface = _rtx_mod.acquire_raycast_query_interface()
            iface.submit_ray_to_raycast_sequence_array(seq_id, rays)
            self._rtx_submitted = True
            self._rtx_submit_frame = self._frame_count
            log.info(
                "TerrainDraper: submitted %d RTX rays (seq=%s, frame=%d)",
                n, seq_id, self._frame_count,
            )
            # Prims stay hidden until _on_update() receives results or times out,
            # so the BVH remains clean when the GPU evaluates the rays next frame.
        except Exception as exc:
            log.info("TerrainDraper: RTX ray submission failed: %s", exc)
            self._restore_draped_prims()

    def _collect_rtx_results(self) -> Optional[np.ndarray]:
        """
        Poll the RTX sequence for completed results.

        Returns a heights array (shape (N,)) if results are ready, None if still pending.
        Missed rays default to 0.0.
        """
        seq_id = self._rtx_seq
        if seq_id is None:
            return None

        try:
            iface = _rtx_mod.acquire_raycast_query_interface()
            err, _rays, results = iface.get_latest_result_from_raycast_sequence_array(seq_id)
        except Exception as exc:
            log.info("TerrainDraper: RTX result poll failed: %s", exc)
            return None

        if getattr(err, "value", err) or not results:
            return None

        n = len(self._east_arr)
        if len(results) != n:
            log.info("TerrainDraper: result count mismatch (%d vs %d)", len(results), n)
            return None

        heights = np.zeros(n, dtype=float)
        hit_count = 0
        target_paths: dict[str, int] = {}
        for i, r in enumerate(results):
            if r.valid:
                heights[i] = float(r.hit_position[1])
                hit_count += 1
                try:
                    path = r.get_target_usd_path()
                    target_paths[path] = target_paths.get(path, 0) + 1
                except Exception:
                    pass

        miss_count = n - hit_count
        log.info(
            "TerrainDraper: RTX results collected - hits=%d, misses=%d, hit_rate=%.1f%%",
            hit_count, miss_count, (hit_count / n * 100) if n > 0 else 0,
        )

        if hit_count == 0:
            # GPU hasn't processed rays yet (all valid=False) — keep polling
            log.info("TerrainDraper: all misses, results not ready yet — will retry")
            return None

        if target_paths:
            top = sorted(target_paths.items(), key=lambda x: -x[1])[:5]
            log.info("TerrainDraper: hit targets - %s", ", ".join(f"{p}({c})" for p, c in top))

        # Reject results that only hit our own geometry — Cesium tiles haven't
        # streamed into the RTX BVH yet.  Wait for a pass where at least one
        # hit lands on a cesium_geometry_pool prim.
        cesium_hits = sum(c for p, c in target_paths.items() if "cesium" in p.lower())
        if cesium_hits == 0:
            log.info(
                "TerrainDraper: no Cesium geometry in BVH yet (hits on %s) — will retry",
                ", ".join(target_paths.keys()),
            )
            return None
        if n > 0:
            log.info(
                "TerrainDraper: heights - min=%.2f, max=%.2f, mean=%.2f",
                float(np.min(heights)), float(np.max(heights)), float(np.mean(heights)),
            )
        return heights

    def _apply_heights(self, heights: np.ndarray) -> None:
        """Apply collected heights to curves and animator, incrementing pass count."""
        if self._last_heights is not None and np.allclose(heights, self._last_heights, atol=0.01):
            self._pass_count += 1
            log.info(
                "TerrainDraper: heights unchanged at pass %d (tiles stable)", self._pass_count
            )
            return

        if self._last_heights is not None:
            diff = np.abs(heights - self._last_heights)
            log.info(
                "TerrainDraper: heights CHANGED - max diff=%.3f, mean diff=%.3f",
                float(np.max(diff)), float(np.mean(diff)),
            )
        else:
            log.info("TerrainDraper: first pass, no previous heights to compare")

        self._last_heights = heights.copy()

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
    # Apply draping to curves and animator
    # ------------------------------------------------------------------

    def apply_to_curves(self, heights: np.ndarray, stage) -> None:
        """
        Update all BasisCurves prims with terrain-draped Y values.

        For each curve, Y = terrain_height + TERRAIN_ABOVE_OFFSET_M + extra_offset.
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

                old_y_samples = [float(existing_pts[i][1]) for i in range(min(5, len(existing_pts)))]
                log.info("TerrainDraper: %s old Y (first 5): %s", prim_path, old_y_samples)

                new_pts = [
                    Gf.Vec3f(
                        float(existing_pts[i][0]),
                        float(heights[i]) + TERRAIN_ABOVE_OFFSET_M + extra_offset,
                        float(existing_pts[i][2]),
                    )
                    for i in range(len(existing_pts))
                ]
                curves.GetPointsAttr().Set(Vt.Vec3fArray(new_pts))

                new_y_samples = [float(new_pts[i][1]) for i in range(min(5, len(new_pts)))]
                log.info(
                    "TerrainDraper: %s new Y (first 5): %s, offset=%.2f",
                    prim_path, new_y_samples, extra_offset,
                )
                log.info(
                    "TerrainDraper: updated %d points at %s (Y = terrain + %.2f m)",
                    len(new_pts), prim_path, extra_offset,
                )
            except Exception as exc:
                log.info("TerrainDraper: failed to drape %s: %s", prim_path, exc)

        log.info("TerrainDraper: apply_to_curves COMPLETE")

    def apply_to_animator(self, heights: np.ndarray) -> None:
        """
        Update the animator's internal Y array with terrain heights and re-bake.

        Sets _usd_y = terrain_height + TERRAIN_ABOVE_OFFSET_M + bob_y_arr (if provided).
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
            old_y = self._animator._usd_y
            old_y_samples = [float(old_y[i]) for i in range(min(5, len(old_y)))]
            log.info("TerrainDraper: animator old Y (first 5): %s", old_y_samples)

            new_y = heights + TERRAIN_ABOVE_OFFSET_M
            if self._bob_y_arr is not None and len(self._bob_y_arr) == len(heights):
                new_y = new_y + self._bob_y_arr
                log.info("TerrainDraper: bob array applied (length=%d)", len(self._bob_y_arr))
            else:
                log.info(
                    "TerrainDraper: no bob array applied (bob_y_arr=%s)",
                    "None" if self._bob_y_arr is None
                    else f"length mismatch ({len(self._bob_y_arr)} vs {len(heights)})",
                )

            new_y_samples = [float(new_y[i]) for i in range(min(5, len(new_y)))]
            log.info("TerrainDraper: animator new Y (first 5): %s", new_y_samples)
            log.info(
                "TerrainDraper: new_y stats - min=%.2f, max=%.2f, mean=%.2f",
                float(np.min(new_y)), float(np.max(new_y)), float(np.mean(new_y)),
            )

            self._animator._usd_y = new_y
            self._animator.bake_animation()
            log.info(
                "TerrainDraper: re-baked animator with terrain heights (bob=%s)",
                "yes" if self._bob_y_arr is not None else "no",
            )
            log.info("TerrainDraper: apply_to_animator COMPLETE")
        except Exception as exc:
            log.info("TerrainDraper: animator update failed: %s", exc)
