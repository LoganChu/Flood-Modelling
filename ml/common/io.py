"""Load all four smoothed forward-pass CSVs into a list of dicts."""

from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOOTHED_DIR = _REPO_ROOT / "visualizations" / "smoothed_forward"

_RUNS = [
    ("run1", "enoM30(1)_forward_smoothed.csv"),
    ("run2", "enoM30(2)_forward_smoothed.csv"),
    ("run3", "enoM30(4)_forward1_smoothed.csv"),
    ("run4", "enoM30(4)_forward2_smoothed.csv"),
]


def load_smoothed_runs(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return list of {run_id, df} dicts for all smoothed runs.

    Args:
        data_dir: Override the default smoothed_forward directory.
    """
    base = data_dir if data_dir is not None else _SMOOTHED_DIR
    runs = []
    for run_id, filename in _RUNS:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Smoothed CSV not found: {path}\n"
                "Run visualizations/scripts/smooth_forward_runs.py first."
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        numeric_cols = [
            "Lat", "Lon", "Speed", "Time", "Alt",
            "AX", "AY", "AZ", "GX", "GY", "GZ",
            "speed_ms", "time_s", "dt_s",
            "east_raw", "north_raw",
            "east_ekf", "north_ekf", "vx_ekf", "vz_ekf",
            "east_rts", "north_rts",
            "east_smooth", "north_smooth",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        runs.append({"run_id": run_id, "df": df})
    return runs
