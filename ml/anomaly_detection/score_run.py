#!/usr/bin/env python3
"""Inference: score a smoothed CSV for anomalies using the trained autoencoder.

Loads model, scaler, and threshold from ml/checkpoints/.
Outputs a new CSV with anomaly_score and anomaly_flag columns appended.

Usage:
    python ml/anomaly_detection/score_run.py <smoothed_csv> [--out <output_csv>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ml.anomaly_detection.model import LSTMAutoencoder
from ml.common.features import derive_features, get_gap_mask
from ml.common.scaler import load_scaler, apply_scaler


def _load_cfg() -> dict:
    with open(_REPO_ROOT / "configs" / "ml.yaml") as f:
        return yaml.safe_load(f)["anomaly"]


def score_csv(csv_path: Path, out_path: Path | None = None) -> pd.DataFrame:
    """Score a smoothed CSV and return the DataFrame with anomaly columns added."""
    cfg = _load_cfg()
    feature_cols: list[str] = cfg["features"]
    window: int = cfg["window"]
    gap_thr: float = cfg["gap_threshold_s"]

    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"
    ckpt = torch.load(ckpt_dir / "anomaly_model.pt", weights_only=True)
    model_cfg = ckpt["cfg"]
    model = LSTMAutoencoder(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = load_scaler(ckpt_dir / "anomaly_scaler.joblib")
    threshold = float(np.load(ckpt_dir / "anomaly_threshold.npy"))

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ["time_s", "dt_s", "vx_ekf", "vz_ekf", "east_raw", "north_raw", "speed_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = derive_features(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    n = len(df)
    if n < window:
        raise ValueError(
            f"CSV has only {n} rows but window size is {window}. "
            "Need at least {window} rows to score."
        )

    gap_mask = get_gap_mask(df, threshold_s=gap_thr)
    feat = df[feature_cols].values.astype(np.float32)
    feat_scaled = apply_scaler(feat, scaler)

    # Build windows and track which sample indices each window covers
    windows_list = []
    window_indices = []  # list of (start, end) pairs in original sample space
    for start in range(0, n - window + 1):
        end = start + window
        if gap_mask[start:end].any():
            continue
        windows_list.append(feat_scaled[start:end])
        window_indices.append((start, end))

    if not windows_list:
        df["anomaly_score"] = np.nan
        df["anomaly_flag"] = 0
        if out_path:
            df.to_csv(out_path, index=False)
        return df

    X = torch.from_numpy(np.array(windows_list, dtype=np.float32))
    per_window_errors = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            errors = model.reconstruction_error(X[i : i + batch_size])
            per_window_errors.append(errors.numpy())
    per_window_errors = np.concatenate(per_window_errors)

    # Reduce to per-sample score = mean error across all windows containing that sample
    sample_scores = np.full(n, np.nan)
    sample_counts = np.zeros(n, dtype=int)
    for (start, end), err in zip(window_indices, per_window_errors):
        for idx in range(start, end):
            if np.isnan(sample_scores[idx]):
                sample_scores[idx] = 0.0
            sample_scores[idx] += err
            sample_counts[idx] += 1

    valid = sample_counts > 0
    sample_scores[valid] /= sample_counts[valid]

    df["anomaly_score"] = sample_scores
    df["anomaly_flag"] = (sample_scores > threshold).astype(int)
    df.loc[np.isnan(sample_scores), "anomaly_flag"] = 0

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved scored CSV: {out_path}")

    n_flagged = int(df["anomaly_flag"].sum())
    print(f"Flagged {n_flagged}/{n} samples ({100*n_flagged/n:.2f}%) above threshold={threshold:.6f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = args.csv_path.parent / (args.csv_path.stem + "_scored.csv")
    score_csv(args.csv_path, out)


if __name__ == "__main__":
    main()
