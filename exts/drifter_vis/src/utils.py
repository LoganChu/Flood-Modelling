"""
utils.py — Shared constants, helpers, and dependency checker.

All values are in SI units (metres, seconds, radians) unless noted.
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_371_000.0   # Mean spherical Earth radius (m)
MPH_TO_MS: float = 1.0 / 2.237        # miles per hour → metres per second
G_MS2: float = 9.80665                # Standard gravity (m/s²)
RHO_WATER: float = 1000.0             # Freshwater density (kg/m³)

# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------
USD_STAGE_FPS: float = 24.0           # USD timeline frames per second
ARROW_SCALE: float = 0.5              # Arrow length per (m/s) for velocity arrows
ACCEL_ARROW_SCALE: float = 0.1        # Arrow length per (m/s²) for accel arrows

# ---------------------------------------------------------------------------
# Terrain draping (PhysX raycast)
# ---------------------------------------------------------------------------
TERRAIN_CAST_ORIGIN_Y: float      = 1000.0   # ray origin height above stage (m)
TERRAIN_ABOVE_OFFSET_M: float     = 0.05     # lift above hit surface (m)
TERRAIN_DRAPE_WARMUP_FRAMES: int  = 120      # frames before first drape (~2s at 60fps)
TERRAIN_DRAPE_UPDATE_INTERVAL: int = 300     # frames between re-drape passes
TERRAIN_DRAPE_MAX_PASSES: int     = 3        # stop after this many passes
CESIUM_TERRAIN_PATH: str = "/World/Cesium/CesiumWorldTerrain"

# ---------------------------------------------------------------------------
# Drifter bobbing
# ---------------------------------------------------------------------------
BOB_AMPLITUDE_M: float = 0.05         # ± vertical bobbing amplitude (m)
BOB_FREQ_HZ: float = 0.8              # Bobbing oscillation frequency (Hz)

# ---------------------------------------------------------------------------
# Visualisation colours (R, G, B) in 0–1 range
# ---------------------------------------------------------------------------
COLOUR_VELOCITY: Tuple[float, float, float] = (0.2, 0.5, 1.0)    # blue
COLOUR_ACCEL: Tuple[float, float, float] = (1.0, 0.2, 0.2)       # red
COLOUR_PHYSICS: Tuple[float, float, float] = (1.0, 0.6, 0.1)     # orange

# USD prim paths (canonical)
WORLD_PATH = "/World"
CESIUM_PATH = "/World/Cesium"
RIVER_PATH = "/World/River"
DRIFTER_PATH = "/World/Drifter"
DRIFTER_XFORM_PATH = "/World/Drifter/DrifterXform"
CAMERAS_PATH = "/World/Cameras"
LIGHTING_PATH = "/World/Lighting"
TRAJECTORY_PATH = "/World/River/Trajectory"
TRAJECTORY_PHYSICS_PATH = "/World/River/TrajectoryPhysics"
TRAJECTORY_EKF_PATH = "/World/River/TrajectoryEKF"
WATER_PLANE_PATH = "/World/River/WaterPlane"


# ---------------------------------------------------------------------------
# Dependency checker
# ---------------------------------------------------------------------------

def check_dependencies() -> dict:
    """
    Probe availability of optional Isaac Sim / Omniverse / Python packages.

    Returns a dict mapping package/extension names to bool.

    Example::

        deps = check_dependencies()
        if deps["cesium"]:
            # use Cesium georeferenced terrain
        else:
            # fall back to flat water plane
    """
    results: dict[str, bool] = {}

    # Python packages
    for pkg in ("numpy", "pandas", "pyproj"):
        results[pkg] = importlib.util.find_spec(pkg) is not None

    # Omniverse / Isaac Sim modules (only available inside Kit)
    for mod in (
        "omni.usd",
        "omni.timeline",
        "omni.kit.app",
        "omni.kit.viewport.utility",
        "omni.isaac.debug_draw",
        "omni.physx",
    ):
        try:
            importlib.import_module(mod)
            results[mod] = True
        except Exception:
            results[mod] = False

    # Cesium: check if the extension is loaded (not import, since lxml may fail)
    # We detect Cesium by checking if the extension manager can find it
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        ext_manager = app.get_extension_manager()
        cesium_ext_id = "cesium.omniverse"
        results["cesium.omniverse"] = ext_manager.is_extension_enabled(cesium_ext_id)
    except Exception:
        results["cesium.omniverse"] = False

    # Convenience aliases
    results["cesium"] = results.get("cesium.omniverse", False)
    results["debug_draw"] = results.get("omni.isaac.debug_draw", False)
    results["omni"] = results.get("omni.usd", False)

    log.debug("Dependency check: %s", results)
    return results


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def speed_to_rgb(speed_ms: float, max_ms: float) -> Tuple[float, float, float]:
    """
    Map a speed value to a viridis-inspired colour gradient.

    0 m/s → dark blue (0.07, 0.12, 0.47)
    max   → bright yellow (0.99, 0.91, 0.14)

    Parameters
    ----------
    speed_ms : current speed in m/s
    max_ms   : maximum speed in the dataset (used to normalise)

    Returns
    -------
    (R, G, B) tuple with components in [0, 1]
    """
    if max_ms <= 0:
        return (0.07, 0.12, 0.47)

    t = _clamp(speed_ms / max_ms, 0.0, 1.0)

    # Simple 3-stop viridis approximation
    if t < 0.5:
        s = t * 2.0
        r = 0.07 + s * (0.13 - 0.07)
        g = 0.12 + s * (0.57 - 0.12)
        b = 0.47 + s * (0.55 - 0.47)
    else:
        s = (t - 0.5) * 2.0
        r = 0.13 + s * (0.99 - 0.13)
        g = 0.57 + s * (0.91 - 0.57)
        b = 0.55 + s * (0.14 - 0.55)

    return (r, g, b)


def discrepancy_to_rgb(
    metres: float,
    clamp: float = 5.0,
) -> Tuple[float, float, float]:
    """
    Map a path discrepancy (metres) to a blue→red gradient.

    0 m   → blue  (0.2, 0.4, 1.0)
    clamp → red   (1.0, 0.1, 0.1)

    Parameters
    ----------
    metres : discrepancy in metres
    clamp  : value at which colour saturates to full red
    """
    t = _clamp(metres / max(clamp, 1e-6), 0.0, 1.0)
    r = 0.2 + t * (1.0 - 0.2)
    g = 0.4 + t * (0.1 - 0.4)
    b = 1.0 + t * (0.1 - 1.0)
    return (r, g, b)


def format_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.t"""
    s = max(0.0, seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:04.1f}"
