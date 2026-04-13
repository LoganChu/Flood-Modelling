"""
geo_converter.py — LLA → ENU → USD coordinate conversion.

Converts GPS latitude/longitude/altitude to a local East-North-Up (ENU)
Cartesian frame centred at the first valid data point, then remaps to
Isaac Sim's Y-up USD stage convention:

    ENU East  (+m) → USD  +X
    ENU Up    (+m) → USD  +Y   (GPS altitude relative to origin)
    ENU North (+m) → USD  +Z

Attitude (roll/pitch/yaw) is estimated from the GPS heading and
accelerometer readings using a lightweight complementary filter —
no full AHRS required for a near-horizontal floating buoy.

A subtle sinusoidal bobbing offset is added to USD Y to simulate
floating motion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import EARTH_RADIUS_M, BOB_AMPLITUDE_M, BOB_FREQ_HZ, G_MS2

log = logging.getLogger(__name__)

# Try pyproj for higher-accuracy conversion (< 0.1 m error at 350 m scale).
# Fall back to spherical approximation (~3–5 m error) if not available.
try:
    from pyproj import Transformer as _PyprojTransformer
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False
    log.debug("pyproj not available — using spherical LLA→ENU approximation")


@dataclass
class GeoResult:
    """Output of GeoConverter.convert()."""
    # ENU coordinates (metres from origin)
    enu_east: np.ndarray   = field(default_factory=lambda: np.array([]))
    enu_north: np.ndarray  = field(default_factory=lambda: np.array([]))
    enu_up: np.ndarray     = field(default_factory=lambda: np.array([]))

    # USD stage coordinates (Y-up, 1 unit = 1 m)
    usd_x: np.ndarray      = field(default_factory=lambda: np.array([]))  # East
    usd_y: np.ndarray      = field(default_factory=lambda: np.array([]))  # Up + bob
    usd_z: np.ndarray      = field(default_factory=lambda: np.array([]))  # North

    # Attitude angles (radians, for xformOp:rotateXYZ)
    roll: np.ndarray       = field(default_factory=lambda: np.array([]))
    pitch: np.ndarray      = field(default_factory=lambda: np.array([]))
    yaw: np.ndarray        = field(default_factory=lambda: np.array([]))

    # Bobbing offset only (metres, added to enu_up to produce usd_y)
    bob_y: np.ndarray      = field(default_factory=lambda: np.array([]))

    # Origin (WGS84)
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    origin_alt: float = 0.0


class GeoConverter:
    """
    Convert a cleaned drifter DataFrame to USD stage coordinates.

    Parameters
    ----------
    use_pyproj : bool
        Force use of pyproj (True) or spherical approximation (False).
        If None (default), use pyproj when available.
    """

    # Bobbing parameters (can be overridden for testing)
    BOB_AMPLITUDE_M: float = BOB_AMPLITUDE_M
    BOB_FREQ_HZ: float = BOB_FREQ_HZ

    def __init__(self, use_pyproj: Optional[bool] = None) -> None:
        if use_pyproj is None:
            self._use_pyproj = _PYPROJ_AVAILABLE
        else:
            self._use_pyproj = use_pyproj and _PYPROJ_AVAILABLE

        if self._use_pyproj:
            log.debug("GeoConverter: using pyproj")
        else:
            log.debug("GeoConverter: using spherical approximation")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, df: pd.DataFrame) -> GeoResult:
        """
        Convert the cleaned DataFrame to ENU + USD coordinates.

        The DataFrame must contain: Lat, Lon, GPS_Alt, time_s, AX, AY, AZ,
        heading_deg (all produced by DrifterDataLoader).

        Parameters
        ----------
        df : cleaned DataFrame from DrifterDataLoader

        Returns
        -------
        GeoResult dataclass
        """
        lat = df["Lat"].values.astype(float)
        lon = df["Lon"].values.astype(float)
        alt = df["GPS_Alt"].values.astype(float)
        time_s = df["time_s"].values.astype(float)

        origin_lat = float(lat[0])
        origin_lon = float(lon[0])
        origin_alt = float(alt[0])

        # 1. LLA → ENU
        if self._use_pyproj:
            east, north, up = self._lla_to_enu_pyproj(
                lat, lon, alt, origin_lat, origin_lon, origin_alt
            )
        else:
            east, north, up = self._lla_to_enu_spherical(
                lat, lon, alt, origin_lat, origin_lon, origin_alt
            )

        # 2. Attitude estimation
        roll, pitch, yaw = self._estimate_attitude(df, east, north)

        # 3. Bobbing offset
        bob_y = self.BOB_AMPLITUDE_M * np.sin(
            2.0 * math.pi * self.BOB_FREQ_HZ * time_s
        )

        # 4. USD stage: East→X, (flat)→Y, North→Z
        # Note: usd_y is now flat (Y=0); terrain height will be drape onto trajectories
        # via TerrainDraper if Cesium is available. bob_y is preserved in GeoResult
        # for the draper to restore bobbing onto the drifter animation.
        usd_x = east.copy()
        usd_y = np.zeros_like(east)
        usd_z = north.copy()

        return GeoResult(
            enu_east=east,
            enu_north=north,
            enu_up=up,
            usd_x=usd_x,
            usd_y=usd_y,
            usd_z=usd_z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            bob_y=bob_y,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_alt=origin_alt,
        )

    # ------------------------------------------------------------------
    # LLA → ENU implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _lla_to_enu_spherical(
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        origin_lat: float,
        origin_lon: float,
        origin_alt: float,
    ):
        """
        Spherical-Earth approximation.  Matches MATLAB plotSpeed.m lines 47-51:

            R = 6371000;
            lat0 = deg2rad(lat(1));  lon0 = deg2rad(lon(1));
            x = R * (deg2rad(lon) - lon0) .* cos(lat0);   % Easting
            y = R * (deg2rad(lat) - lat0);                 % Northing

        Error: ~3–5 m over 350 m baseline (< 1.5 %, acceptable for vis).
        """
        R = EARTH_RADIUS_M
        lat0 = math.radians(origin_lat)
        lon0 = math.radians(origin_lon)

        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)

        east  = R * (lon_r - lon0) * math.cos(lat0)
        north = R * (lat_r - lat0)
        up    = alt - origin_alt

        return east, north, up

    @staticmethod
    def _lla_to_enu_pyproj(
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        origin_lat: float,
        origin_lon: float,
        origin_alt: float,
    ):
        """
        High-accuracy conversion using pyproj (WGS84 → local ENU).
        Error: < 0.1 m at 350 m baseline.
        """
        from pyproj import Transformer

        # WGS84 geodetic → ECEF
        t_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        x_ecef, y_ecef, z_ecef = t_ecef.transform(lon, lat, alt)
        ox, oy, oz = t_ecef.transform(origin_lon, origin_lat, origin_alt)

        # ECEF delta → ENU rotation
        lat0_r = math.radians(origin_lat)
        lon0_r = math.radians(origin_lon)
        sl = math.sin(lat0_r); cl = math.cos(lat0_r)
        sl0 = math.sin(lon0_r); cl0 = math.cos(lon0_r)

        dx = x_ecef - ox
        dy = y_ecef - oy
        dz = z_ecef - oz

        east  = -sl0 * dx + cl0 * dy
        north = -sl * cl0 * dx - sl * sl0 * dy + cl * dz
        up    =  cl * cl0 * dx + cl * sl0 * dy + sl * dz

        return east, north, up

    # ------------------------------------------------------------------
    # Attitude estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_attitude(
        df: pd.DataFrame,
        east: np.ndarray,
        north: np.ndarray,
    ):
        """
        Estimate roll, pitch, yaw (radians) for each data row.

        Yaw   : derived from GPS heading_deg (0 = North, clockwise)
                → in USD: yaw = -heading_deg (Y-up, Z=North convention)
        Pitch : atan2(AX_debiased, sqrt(AY²+AZ_debiased²))
        Roll  : atan2(AY, AZ_debiased)

        The device rests roughly flat, so AZ ≈ −g ≈ −8.28 m/s².
        We use the median AZ as the gravity reference to debias.
        """
        ax = df["AX"].fillna(0).values.astype(float)
        ay = df["AY"].fillna(0).values.astype(float)
        az = df["AZ"].fillna(0).values.astype(float)

        # Estimate gravity bias from the median AZ (device mostly flat)
        g_bias_z = float(np.median(az))      # ≈ −8.28 for this dataset
        az_debiased = az - g_bias_z
        ax_debiased = ax  # AX/AY bias is near-zero for horizontal float

        # Pitch and roll (small angles expected for floating buoy)
        pitch = np.arctan2(ax_debiased, np.sqrt(ay**2 + az_debiased**2))
        roll  = np.arctan2(ay, np.clip(az_debiased, -G_MS2, G_MS2))

        # Yaw from GPS heading (degrees → radians, rotate to USD convention)
        heading_deg = df["heading_deg"].fillna(0).values.astype(float)
        # In USD Y-up with North=+Z, East=+X:
        # heading 0° (North) → yaw 0 (looking along +Z)
        # heading 90° (East) → yaw -90° (looking along +X)
        yaw = -np.deg2rad(heading_deg)

        return roll, pitch, yaw
