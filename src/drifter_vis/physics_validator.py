"""
physics_validator.py — Newton forward-Euler buoyancy + drag simulation.

Runs a simplified physics model alongside the kinematic replay so users can
compare the actual recorded path with what a free-floating object would do
if only acted on by river current drag and buoyancy.

Physics model (2-D horizontal: East/North plane)
-------------------------------------------------
State:   position (east, north), velocity (vx, vz)

Forces:
  1. River current — initialised from the first GPS velocity vector.
     Constant thereafter (first-order approximation of the Eno River flow).

  2. Fluid drag — opposes relative motion between drifter and current.
     F_drag = 0.5 * ρ * Cd * A * |v_rel|² * v̂_rel

  3. Buoyancy + gravity — cancel out in the horizontal plane (the drifter
     floats at equilibrium depth). Vertical position follows the recorded
     GPS altitude (no vertical physics).

Integration: explicit forward Euler with the recorded dt_s timesteps.

Discrepancy metric
------------------
After simulation, computes per-point Euclidean distance (metres) between
the recorded ENU path and the simulated path. This is used to colour-code
the physics trajectory BasisCurves.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Optional: USD baking
try:
    from pxr import Gf, Vt, UsdGeom
    import omni.usd
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False

from .utils import (
    TRAJECTORY_PHYSICS_PATH, RHO_WATER, G_MS2, discrepancy_to_rgb,
)


@dataclass
class DrifterPhysicsParams:
    """Physical parameters of the sensor buoy."""
    mass_kg:    float = 1.5       # estimated total mass
    radius_m:   float = 0.15      # half-diameter of the cylinder
    height_m:   float = 0.30      # cylinder height
    Cd:         float = 0.82      # drag coefficient for a cylinder
    rho_water:  float = RHO_WATER # freshwater density (kg/m³)
    g:          float = G_MS2     # gravity (m/s²)

    @property
    def frontal_area(self) -> float:
        """Projected frontal area (m²) — 2r × h cylinder face."""
        return 2.0 * self.radius_m * self.height_m

    @property
    def submerged_volume(self) -> float:
        """Submerged volume at neutral float (m³) — 50 % submerged assumption."""
        return math.pi * self.radius_m**2 * self.height_m * 0.5

    @property
    def buoyancy_force(self) -> float:
        """Net upward buoyancy minus gravity (N) — should be ~0 for equilibrium."""
        return self.rho_water * self.submerged_volume * self.g - self.mass_kg * self.g


class PhysicsValidator:
    """
    Simulate drifter motion with buoyancy and drag, compare to recorded path.

    Parameters
    ----------
    params : DrifterPhysicsParams (uses defaults if omitted)
    """

    def __init__(self, params: Optional[DrifterPhysicsParams] = None) -> None:
        self.params = params or DrifterPhysicsParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        df: pd.DataFrame,
        rec_east: np.ndarray,
        rec_north: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the forward-Euler simulation over the full data timeline.

        The river current is estimated from the first few GPS-derived
        velocity vectors (average of first 10 samples with speed > 0.1 m/s).

        Parameters
        ----------
        df        : cleaned DataFrame (needs speed_ms, heading_deg, dt_s, time_s)
        rec_east  : recorded ENU east positions (m)
        rec_north : recorded ENU north positions (m)

        Returns
        -------
        sim_east, sim_north : simulated ENU positions (m)
        """
        n = len(df)
        p = self.params
        dt_arr = df["dt_s"].values.astype(float)

        # ---- Initial conditions ----
        sim_east  = np.zeros(n)
        sim_north = np.zeros(n)
        sim_east[0]  = float(rec_east[0])
        sim_north[0] = float(rec_north[0])

        # Estimate initial velocity from first valid GPS speed+heading
        vx, vz = self._initial_velocity(df)
        log.debug("Physics IC: vx=%.3f  vz=%.3f (m/s)", vx, vz)

        # River current: constant, estimated from early GPS data
        curr_x, curr_z = self._estimate_current(df)
        log.debug("River current: cx=%.3f  cz=%.3f (m/s)", curr_x, curr_z)

        # ---- Integration loop ----
        for i in range(1, n):
            dt = float(dt_arr[i])
            if dt <= 0:
                sim_east[i]  = sim_east[i-1]
                sim_north[i] = sim_north[i-1]
                continue

            # Relative velocity w.r.t. river current
            rel_vx = vx - curr_x
            rel_vz = vz - curr_z
            rel_spd = math.sqrt(rel_vx**2 + rel_vz**2)

            # Drag force magnitude
            if rel_spd > 1e-6:
                f_drag = 0.5 * p.rho_water * p.Cd * p.frontal_area * rel_spd**2
                ax = -f_drag * rel_vx / rel_spd / p.mass_kg
                az = -f_drag * rel_vz / rel_spd / p.mass_kg
            else:
                ax = az = 0.0

            # Euler step
            vx += ax * dt
            vz += az * dt
            sim_east[i]  = sim_east[i-1]  + vx * dt
            sim_north[i] = sim_north[i-1] + vz * dt

        log.info(
            "Physics simulation complete: final pos (%.1f, %.1f) m",
            sim_east[-1], sim_north[-1],
        )
        return sim_east, sim_north

    def compute_discrepancy(
        self,
        rec_east: np.ndarray,
        rec_north: np.ndarray,
        sim_east: np.ndarray,
        sim_north: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-point Euclidean distance between recorded and simulated paths.

        Returns
        -------
        discrepancy : array of shape (N,) in metres
        """
        de = sim_east  - rec_east
        dn = sim_north - rec_north
        disc = np.sqrt(de**2 + dn**2)
        log.info(
            "Discrepancy: mean=%.2f m  max=%.2f m  95th pct=%.2f m",
            float(disc.mean()), float(disc.max()),
            float(np.percentile(disc, 95)),
        )
        return disc

    def bake_physics_trajectory(
        self,
        sim_east: np.ndarray,
        sim_north: np.ndarray,
        enu_up: np.ndarray,
        discrepancy: np.ndarray,
        time_s: np.ndarray,
        fps: float = 24.0,
        speed_scale: float = 1.0,
    ) -> None:
        """
        Pre-bake a USD BasisCurves prim for the simulated path, coloured by
        discrepancy, and a time-sampled Xform showing the simulated drifter
        position.

        The prim path is TRAJECTORY_PHYSICS_PATH (defined in scene_builder.py
        as a static curve; here we just update its point colours if it already
        exists, or skip if not).
        """
        if not _USD_AVAILABLE:
            log.warning("USD not available — bake_physics_trajectory() is a no-op")
            return

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(TRAJECTORY_PHYSICS_PATH)
        if not prim.IsValid():
            log.debug("Physics trajectory prim not found — skipped")
            return

        # Update vertex colours
        curves = UsdGeom.BasisCurves(prim)
        colours = [Gf.Vec3f(*discrepancy_to_rgb(float(d))) for d in discrepancy]
        curves.GetDisplayColorAttr().Set(Vt.Vec3fArray(colours))

        # Update point positions
        pts = [
            Gf.Vec3f(float(e), float(u) + 0.3, float(n))
            for e, u, n in zip(sim_east, enu_up, sim_north)
        ]
        curves.GetPointsAttr().Set(Vt.Vec3fArray(pts))

        log.debug("Physics trajectory updated")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_velocity(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Return initial (vx, vz) from first row with non-trivial speed."""
        for _, row in df.iterrows():
            spd = float(row["speed_ms"])
            if spd > 0.05:
                hdg = float(row["heading_deg"])
                hdg_r = math.radians(hdg)
                return spd * math.sin(hdg_r), spd * math.cos(hdg_r)
        return 0.0, 0.0

    def _estimate_current(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
    ) -> Tuple[float, float]:
        """
        Estimate mean river current from the first *n_samples* valid GPS points.

        Uses the GPS-derived velocity vector (speed + heading) as a proxy
        for the bulk river current (valid approximation: the drifter travels
        with the river between rapids / eddies).
        """
        vxs, vzs = [], []
        for _, row in df.head(n_samples).iterrows():
            spd = float(row["speed_ms"])
            if spd > 0.1:
                hdg_r = math.radians(float(row["heading_deg"]))
                vxs.append(spd * math.sin(hdg_r))
                vzs.append(spd * math.cos(hdg_r))
        if not vxs:
            return 0.0, 0.0
        return float(np.mean(vxs)), float(np.mean(vzs))
