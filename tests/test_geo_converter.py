"""
Unit tests for GeoConverter.
Run without Isaac Sim:  pytest tests/ -v
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drifter_vis.src.geo_converter import GeoConverter, GeoResult
from drifter_vis.src.data_loader import DrifterDataLoader

REAL_CSV = Path(__file__).resolve().parent.parent / "exts" / "drifter_vis" / "data" / "enoFeb16th.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(lats, lons, alts, times_ms=None, speeds=None):
    n = len(lats)
    if times_ms is None:
        times_ms = [i * 200 for i in range(n)]
    if speeds is None:
        speeds = [0.5] * n
    return pd.DataFrame({
        "Lat": lats, "Lon": lons, "GPS_Alt": alts,
        "Time": times_ms,
        "time_s": [t / 1000.0 for t in times_ms],
        "dt_s": [0.2] * n,
        "speed_ms": speeds,
        "heading_deg": [0.0] * n,   # North
        "AX": [0.0] * n,
        "AY": [0.0] * n,
        "AZ": [-9.81] * n,
        "GX": [0.0] * n,
        "GY": [0.0] * n,
        "GZ": [0.0] * n,
        "segment_id": [0] * n,
        "accel_mag": [9.81] * n,
        "source_file": ["test.csv"] * n,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOriginIsZero:
    def test_first_point_east_north_zero(self):
        df = _make_df([36.079], [-79.008], [130.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        assert abs(result.enu_east[0]) < 1e-6
        assert abs(result.enu_north[0]) < 1e-6

    def test_usd_x_z_zero_at_origin(self):
        df = _make_df([36.079], [-79.008], [130.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        assert abs(result.usd_x[0]) < 1e-6
        assert abs(result.usd_z[0]) < 1e-6


class TestDirectionalDisplacement:
    def test_east_displacement(self):
        """Moving east (increasing lon) → positive enu_east / usd_x."""
        df = _make_df([36.079, 36.079], [-79.008, -79.007], [130.0, 130.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.enu_east[1] > 0.0
        assert result.usd_x[1] > 0.0

    def test_north_displacement(self):
        """Moving north (increasing lat) → positive enu_north, negative usd_z (North=−Z)."""
        df = _make_df([36.079, 36.080], [-79.008, -79.008], [130.0, 130.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.enu_north[1] > 0.0
        assert result.usd_z[1] < 0.0

    def test_usd_y_is_flat_baseline(self):
        """usd_y is always 0.0 — terrain draping sets actual heights at runtime."""
        df = _make_df(
            [36.079, 36.079], [-79.008, -79.008],
            [130.0, 135.0],
        )
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.usd_y[0] == pytest.approx(0.0)
        assert result.usd_y[1] == pytest.approx(0.0)
        # Altitude delta is preserved in enu_up, not usd_y
        assert result.enu_up[1] - result.enu_up[0] == pytest.approx(5.0, abs=0.01)


class TestSphericalScale:
    # The spherical formula uses R=6,371,000 m, giving:
    # 1° lat = π/180 × R = 111,194.9 m  (differs from WGS84 111,320 m by ~0.11%)
    _R = 6_371_000.0
    _DEG_TO_M = math.pi / 180.0 * _R   # 111,194.9 m / degree

    def test_one_degree_latitude(self):
        """1° latitude → π/180 × R metres (spherical Earth)."""
        df = _make_df([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        expected = self._DEG_TO_M
        assert abs(result.enu_north[1] - expected) / expected < 1e-9

    def test_one_degree_longitude_at_equator(self):
        """1° longitude at equator → π/180 × R metres (cos 0° = 1)."""
        df = _make_df([0.0, 0.0], [0.0, 1.0], [0.0, 0.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        expected = self._DEG_TO_M
        assert abs(result.enu_east[1] - expected) / expected < 1e-9

    def test_longitude_shrinks_at_latitude(self):
        """1° longitude at 36°N → π/180 × R × cos(36°)."""
        df = _make_df([36.0, 36.0], [0.0, 1.0], [0.0, 0.0])
        result = GeoConverter(use_pyproj=False).convert(df)
        expected = self._DEG_TO_M * math.cos(math.radians(36.0))
        assert abs(result.enu_east[1] - expected) / expected < 1e-9


class TestBobbing:
    def test_amplitude_bounded(self):
        """Bobbing offset must stay within ±BOB_AMPLITUDE_M."""
        n = 200
        conv = GeoConverter(use_pyproj=False)
        df = _make_df(
            [36.079] * n, [-79.008] * n, [130.0] * n,
            times_ms=list(range(0, n * 200, 200)),
        )
        result = conv.convert(df)
        assert np.all(np.abs(result.bob_y) <= conv.BOB_AMPLITUDE_M + 1e-9)

    def test_bob_zero_at_t0(self):
        """sin(0) = 0 so bob_y[0] should be 0."""
        df = _make_df([36.079], [-79.008], [130.0], times_ms=[0])
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.bob_y[0] == pytest.approx(0.0, abs=1e-9)


class TestAttitude:
    def test_yaw_north_is_zero(self):
        """heading_deg=0 (North) → yaw=0."""
        df = _make_df([36.079], [-79.008], [130.0])
        df["heading_deg"] = 0.0
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.yaw[0] == pytest.approx(0.0, abs=1e-6)

    def test_yaw_east_is_minus_90(self):
        """heading_deg=90 (East) → yaw = -π/2."""
        df = _make_df([36.079], [-79.008], [130.0])
        df["heading_deg"] = 90.0
        result = GeoConverter(use_pyproj=False).convert(df)
        assert result.yaw[0] == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_flat_float_roll_near_zero(self):
        """Device laying flat: AX=0, AY=0, AZ=-g → roll≈0, pitch≈0."""
        df = _make_df([36.079], [-79.008], [130.0])
        df["AX"] = 0.0
        df["AY"] = 0.0
        df["AZ"] = -9.81
        result = GeoConverter(use_pyproj=False).convert(df)
        assert abs(result.roll[0]) < 0.1     # < ~5.7°
        assert abs(result.pitch[0]) < 0.1


class TestRealData:
    """Integration test using actual sensor CSV."""

    @pytest.fixture
    def result(self):
        if not REAL_CSV.exists():
            pytest.skip("Real CSV not available")
        df = DrifterDataLoader().load(REAL_CSV)
        return GeoConverter(use_pyproj=False).convert(df)

    def test_path_extent_reasonable(self, result):
        """Eno River stretch: expect <1 km bounding box."""
        east_range  = result.enu_east.max()  - result.enu_east.min()
        north_range = result.enu_north.max() - result.enu_north.min()
        assert east_range < 2000,  f"East range too large: {east_range:.1f} m"
        assert north_range < 2000, f"North range too large: {north_range:.1f} m"

    def test_no_nans(self, result):
        for name in ("usd_x", "usd_y", "usd_z", "roll", "pitch", "yaw"):
            arr = getattr(result, name)
            assert not np.any(np.isnan(arr)), f"{name} contains NaN"

    def test_origin_at_zero(self, result):
        assert abs(result.enu_east[0]) < 1e-6
        assert abs(result.enu_north[0]) < 1e-6
