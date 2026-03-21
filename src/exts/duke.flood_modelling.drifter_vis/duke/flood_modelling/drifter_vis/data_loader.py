"""
data_loader.py — Load and clean a river drifter CSV into a pandas DataFrame.

CSV column schema (as produced by the Arduino + rawToCSV.m pipeline):
    Lat, Lon, Sats, Speed, GPS_Alt, Dist, Time, Alt,
    AX, AY, AZ, GX, GY, GZ, MX, MY, MZ, Endpoint, source_file

Key transformations applied:
- Drop GPS dropouts (Lat == 0 AND Lon == 0)
- Speed mph → m/s
- Time (Arduino millis()) → monotone time_s starting at 0
- Derived columns: dt_s, accel_mag, heading_deg, speed_ms
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns that must exist in the CSV (subset of full schema)
REQUIRED_COLUMNS: List[str] = [
    "Lat", "Lon", "Speed", "GPS_Alt", "Time",
    "AX", "AY", "AZ", "GX", "GY", "GZ",
]

# Optional columns — filled with 0 if absent
OPTIONAL_COLUMNS: List[str] = [
    "Sats", "Dist", "Alt", "MX", "MY", "MZ", "Endpoint",
]

# Any backward jump larger than this (seconds) is treated as a millis() reset
_SEGMENT_JUMP_S: float = 1.0


class DataLoadError(ValueError):
    """Raised for unrecoverable problems with the input CSV."""


class DrifterDataLoader:
    """
    Load a drifter CSV and return a clean, enriched DataFrame.

    Attributes
    ----------
    segments : list of (start_idx, end_idx) tuples (inclusive) per source file
    metadata : summary dict with keys: rows, segments, duration_s,
               origin_lat, origin_lon, origin_alt
    """

    def __init__(self) -> None:
        self.segments: List[Tuple[int, int]] = []
        self.metadata: Dict = {}
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load, clean, and enrich the CSV at *path*.

        Parameters
        ----------
        path : path to the CSV file

        Returns
        -------
        Cleaned DataFrame with additional derived columns.

        Raises
        ------
        DataLoadError  if required columns are missing or file is unreadable
        FileNotFoundError  if *path* does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        log.info("Loading CSV: %s", path)
        raw = self._read_csv(path)
        df = self._validate(raw)
        df = self._filter_gps_dropouts(df)
        df = self._fill_optional(df)
        df = self._convert_units(df)
        df = self._normalise_time(df)
        df = self._enrich(df)

        self._df = df
        self._build_segment_index(df)
        self._build_metadata(df)

        log.info(
            "Loaded %d rows | %d segment(s) | %.1f s",
            len(df), len(self.segments), self.metadata["duration_s"],
        )
        return df

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception as exc:
            raise DataLoadError(f"Cannot read CSV: {exc}") from exc

        # Strip whitespace from column names and string values
        df.columns = [c.strip() for c in df.columns]
        return df

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataLoadError(
                f"CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        # Coerce numeric columns
        numeric_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _filter_gps_dropouts(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where both Lat and Lon are zero (GPS not locked)."""
        n_before = len(df)
        mask = ~((df["Lat"].fillna(0) == 0) & (df["Lon"].fillna(0) == 0))
        df = df[mask].copy().reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            log.debug("Dropped %d GPS dropout rows", dropped)
        # Also drop rows where Lat or Lon is NaN
        df = df.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
        return df

    @staticmethod
    def _fill_optional(df: pd.DataFrame) -> pd.DataFrame:
        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    @staticmethod
    def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
        """Speed: mph → m/s (matches MATLAB: spd = spd./2.237)."""
        df["speed_ms"] = df["Speed"].fillna(0.0) / 2.237
        return df

    @staticmethod
    def _normalise_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Arduino millis() Time column → monotone time_s (seconds from 0).

        The sensor may record two independent float runs concatenated in one
        CSV, each with its own millis() counter that resets.  We detect
        backward jumps > _SEGMENT_JUMP_S seconds and add a cumulative offset
        so that time_s is strictly non-decreasing across the whole dataset.
        """
        t_raw = df["Time"].ffill().values.astype(float)

        # Convert ms → s
        t_s = t_raw / 1000.0

        # Normalise first sample to 0
        t_s = t_s - t_s[0]

        # Find wrap-around / backward jumps and add cumulative offset
        offsets = np.zeros(len(t_s))
        cumulative_offset = 0.0
        for i in range(1, len(t_s)):
            dt = t_s[i] - t_s[i - 1] + cumulative_offset
            prev_abs = t_s[i - 1] + cumulative_offset
            if (t_s[i] + cumulative_offset) < prev_abs - _SEGMENT_JUMP_S:
                # Backward jump — treat as a new segment starting from prev+1s
                cumulative_offset = prev_abs + 1.0 - t_s[i]
                log.debug(
                    "Time reset detected at row %d; adding offset %.3fs",
                    i, cumulative_offset,
                )
            offsets[i] = cumulative_offset

        df["time_s"] = t_s + offsets
        return df

    @staticmethod
    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns."""
        n = len(df)

        # dt_s (time step per row)
        dt = np.diff(df["time_s"].values, prepend=np.nan)
        dt[0] = float(np.nanmedian(dt[1:]))  # fill first row
        dt = np.clip(dt, 1e-3, 60.0)         # guard against zeros / huge gaps
        df["dt_s"] = dt

        # Acceleration magnitude
        ax = df["AX"].fillna(0).values
        ay = df["AY"].fillna(0).values
        az = df["AZ"].fillna(0).values
        df["accel_mag"] = np.sqrt(ax**2 + ay**2 + az**2)

        # Heading from sequential GPS positions (degrees, 0=North, CW)
        lat_rad = np.deg2rad(df["Lat"].values)
        lon_rad = np.deg2rad(df["Lon"].values)
        dlat = np.diff(lat_rad, prepend=lat_rad[0])
        dlon = np.diff(lon_rad, prepend=lon_rad[0])
        heading = np.degrees(np.arctan2(dlon, dlat)) % 360.0
        # Smooth heading by forward-filling zero-movement rows
        heading[df["speed_ms"].values < 0.05] = np.nan
        # Fill NaN with previous valid heading
        s = pd.Series(heading)
        heading = s.ffill().fillna(0.0).values
        df["heading_deg"] = heading

        # Segment id from source_file column
        if "source_file" in df.columns:
            df["segment_id"] = pd.Categorical(df["source_file"].fillna("")).codes
        else:
            df["segment_id"] = 0

        return df

    def _build_segment_index(self, df: pd.DataFrame) -> None:
        """Populate self.segments as list of (start, end) index ranges."""
        if "segment_id" not in df.columns:
            self.segments = [(0, len(df) - 1)]
            return
        self.segments = []
        for seg_id, grp in df.groupby("segment_id", sort=True):
            idx = grp.index
            self.segments.append((int(idx[0]), int(idx[-1])))

    def _build_metadata(self, df: pd.DataFrame) -> None:
        self.metadata = {
            "rows": len(df),
            "segments": len(self.segments),
            "duration_s": float(df["time_s"].iloc[-1]),
            "origin_lat": float(df["Lat"].iloc[0]),
            "origin_lon": float(df["Lon"].iloc[0]),
            "origin_alt": float(df["GPS_Alt"].iloc[0]),
        }
