"""Minimal preprocessing utilities for the sample pipeline.

These are small, pure-python placeholders intended to show interfaces.
"""
from typing import List, Tuple


def resample_rainfall(timeseries: List[Tuple[str, float]], target_minutes: int) -> List[Tuple[str, float]]:
    """Naive resampling: aggregates rainfall points into fixed-length bins.

    timeseries: list of (timestamp_str, value) sorted by time. This stub does not parse timestamps,
    it groups every N points into a single bin (use real resampling for production).
    """
    if target_minutes <= 0:
        raise ValueError("target_minutes must be > 0")

    out = []
    bin_sum = 0.0
    count = 0
    for i, (ts, val) in enumerate(timeseries):
        bin_sum += val
        count += 1
        if count >= target_minutes:
            out.append((ts, bin_sum))
            bin_sum = 0.0
            count = 0
    if count > 0:
        out.append((timeseries[-1][0], bin_sum))
    return out


def hydrologic_bias_correction(radar_series: List[float], factor: float = 1.0) -> List[float]:
    """Apply a simple multiplicative bias correction to radar rainfall.

    This placeholder is intentionally simple. Replace with statistical bias correction
    (quantile mapping, gauge adjustment) for real applications.
    """
    return [float(x) * float(factor) for x in radar_series]
