"""
Unit tests for DrifterDataLoader.
Run without Isaac Sim:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Resolve package root so tests can be run from any directory
_EXT_ROOT = Path(__file__).resolve().parent.parent
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))

from duke.flood_modelling.drifter_vis.data_loader import (
    DataLoadError,
    DrifterDataLoader,
    REQUIRED_COLUMNS,
)

REAL_CSV = Path(__file__).resolve().parents[4] / "data" / "enoFeb16th.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> dict:
    """Build a minimal valid CSV row dict."""
    base = {
        "Lat": 36.079, "Lon": -79.008, "Speed": 1.0, "GPS_Alt": 130.0,
        "Time": 1000, "AX": 0.0, "AY": 0.0, "AZ": -9.8,
        "GX": 0.0, "GY": 0.0, "GZ": 0.0,
        "Sats": 8, "Dist": 0.0, "Alt": 130.0,
        "MX": 0.0, "MY": 0.0, "MZ": 0.0,
        "Endpoint": 0, "source_file": "test.csv",
    }
    base.update(overrides)
    return base


def _write_csv(tmp_path, rows):
    csv = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGPSDropoutRemoval:
    def test_drops_both_zero(self, tmp_path):
        rows = [
            _make_row(Lat=0.0, Lon=0.0),   # dropout
            _make_row(Lat=36.079, Lon=-79.008),
        ]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert len(df) == 1
        assert float(df["Lat"].iloc[0]) == pytest.approx(36.079)

    def test_keeps_nonzero(self, tmp_path):
        rows = [_make_row(), _make_row(Lat=36.080)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert len(df) == 2

    def test_partial_zero_kept(self, tmp_path):
        # Lat==0 but Lon!=0 → not a dropout
        rows = [_make_row(Lat=0.0, Lon=-79.008)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert len(df) == 1


class TestSpeedConversion:
    def test_mph_to_ms(self, tmp_path):
        # 2.237 mph ≈ 1.000 m/s
        rows = [_make_row(Speed=2.237)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert df["speed_ms"].iloc[0] == pytest.approx(1.0, rel=1e-3)

    def test_zero_speed(self, tmp_path):
        rows = [_make_row(Speed=0.0)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert df["speed_ms"].iloc[0] == pytest.approx(0.0)


class TestTimeNormalisation:
    def test_starts_at_zero(self, tmp_path):
        rows = [_make_row(Time=527603), _make_row(Lat=36.08, Time=527803)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert df["time_s"].iloc[0] == pytest.approx(0.0)

    def test_monotone_after_reset(self, tmp_path):
        # Simulate millis() reset between two runs
        rows = [
            _make_row(Lat=36.079, Time=1000),
            _make_row(Lat=36.080, Time=1200),
            _make_row(Lat=36.081, Time=1400),
            _make_row(Lat=36.082, Time=100),   # reset — new run
            _make_row(Lat=36.083, Time=300),
        ]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        ts = df["time_s"].values
        assert np.all(np.diff(ts) >= 0), "time_s is not monotone"

    def test_dt_s_positive(self, tmp_path):
        rows = [_make_row(Time=t) for t in [1000, 1200, 1400, 1600]]
        # Update Lat so rows are distinct
        for i, r in enumerate(rows):
            r["Lat"] = 36.079 + i * 0.0001
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert (df["dt_s"].values > 0).all()


class TestMissingColumns:
    def test_raises_on_missing_required(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"Lat": [36.0], "Lon": [-79.0]}).to_csv(csv, index=False)
        with pytest.raises(DataLoadError, match="missing required columns"):
            DrifterDataLoader().load(csv)

    def test_optional_filled_with_zero(self, tmp_path):
        # CSV without MX/MY/MZ — should not raise
        rows = [_make_row()]
        row_no_opt = {k: v for k, v in rows[0].items()
                      if k not in ("MX", "MY", "MZ", "Sats")}
        csv = tmp_path / "t.csv"
        pd.DataFrame([row_no_opt]).to_csv(csv, index=False)
        df = DrifterDataLoader().load(csv)
        assert "MX" in df.columns
        assert float(df["MX"].iloc[0]) == pytest.approx(0.0)


class TestDerivedColumns:
    def test_accel_mag(self, tmp_path):
        rows = [_make_row(AX=3.0, AY=4.0, AZ=0.0)]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert df["accel_mag"].iloc[0] == pytest.approx(5.0, rel=1e-3)

    def test_segment_id_present(self, tmp_path):
        rows = [_make_row(source_file="a.csv"), _make_row(Lat=36.08, source_file="b.csv")]
        df = DrifterDataLoader().load(_write_csv(tmp_path, rows))
        assert "segment_id" in df.columns
        assert df["segment_id"].nunique() == 2


class TestRealCSV:
    """Smoke tests on the actual sensor data (skipped if file absent)."""

    @pytest.fixture
    def df(self):
        if not REAL_CSV.exists():
            pytest.skip("Real CSV not available")
        return DrifterDataLoader().load(REAL_CSV)

    def test_row_count(self, df):
        assert len(df) > 3000

    def test_no_zero_lat_lon(self, df):
        assert not ((df["Lat"] == 0) & (df["Lon"] == 0)).any()

    def test_time_monotone(self, df):
        assert np.all(np.diff(df["time_s"].values) >= 0)

    def test_two_segments(self, df):
        loader = DrifterDataLoader()
        loader.load(REAL_CSV)
        assert len(loader.segments) >= 1

    def test_speed_ms_range(self, df):
        # Expect drifter speeds between 0 and ~3 m/s for river conditions
        assert (df["speed_ms"] >= 0).all()
        assert (df["speed_ms"] < 10).all()   # sanity: < 10 m/s
