"""Feature engineering for ML modules.

Derives ax_filt, az_filt, heading_deg (for Seq2Seq) and delta_east/north
(for anomaly detection) from the smoothed EKF output columns.
"""

import numpy as np
import pandas as pd


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns to a smoothed-run DataFrame.

    Adds:
        ax_filt     — east acceleration via np.gradient(vx_ekf, time_s)
        az_filt     — north acceleration via np.gradient(vz_ekf, time_s)
        heading_deg — bearing from velocity vector, 0°=East (CCW), range [0, 360)
        delta_east  — first difference of east_raw (m/step)
        delta_north — first difference of north_raw (m/step)

    Returns a new DataFrame; does not modify the input.
    """
    out = df.copy()
    time_s = out["time_s"].values.astype(float)
    vx = out["vx_ekf"].values.astype(float)
    vz = out["vz_ekf"].values.astype(float)

    out["ax_filt"] = np.gradient(vx, time_s)
    out["az_filt"] = np.gradient(vz, time_s)
    out["heading_deg"] = np.degrees(np.arctan2(vx, vz)) % 360.0

    east_raw = out["east_raw"].values.astype(float)
    north_raw = out["north_raw"].values.astype(float)
    out["delta_east"] = np.diff(east_raw, prepend=east_raw[0])
    out["delta_north"] = np.diff(north_raw, prepend=north_raw[0])

    return out


def get_gap_mask(df: pd.DataFrame, threshold_s: float = 5.0) -> np.ndarray:
    """Return boolean array marking millis() wrap-around boundaries.

    A True at index i means a time-gap of >threshold_s precedes sample i.
    Sliding windows that would span any True index are excluded.

    Args:
        df: DataFrame with a dt_s column.
        threshold_s: dt_s above this value is treated as a gap.
    """
    dt = df["dt_s"].values.astype(float)
    mask = np.zeros(len(df), dtype=bool)
    mask[dt > threshold_s] = True
    return mask
