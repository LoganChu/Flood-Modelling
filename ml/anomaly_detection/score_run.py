#!/usr/bin/env python3
"""Inference: score a smoothed CSV for anomalies using the trained autoencoder.

Loads model, scaler, score normaliser, and threshold from ml/checkpoints/.
Outputs a new CSV with anomaly_score, anomaly_flag, and anomaly_feature (the
feature responsible for the score) columns appended.

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
from ml.anomaly_detection.scoring import anomaly_scores, per_sample_feature_errors
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
    feature_norm_path = ckpt_dir / "anomaly_feature_norm.npy"
    if not feature_norm_path.exists():
        raise FileNotFoundError(
            f"{feature_norm_path} not found. "
            "Run ml/anomaly_detection/train.py --calibrate-only to create it."
        )
    feature_norm = np.load(feature_norm_path)

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
            f"Need at least {window} rows to score."
        )

    gap_mask = get_gap_mask(df, threshold_s=gap_thr)
    feat = df[feature_cols].values.astype(np.float32)
    feat_scaled = apply_scaler(feat, scaler)

    # Per-sample score = largest per-feature reconstruction error, relative to that
    # feature's typical error. Samples no gap-free window covers stay NaN / unflagged.
    feature_errors = per_sample_feature_errors(model, feat_scaled, gap_mask, window)
    sample_scores, top_feature = anomaly_scores(feature_errors, feature_norm)

    df["anomaly_score"] = sample_scores
    df["anomaly_flag"] = (sample_scores > threshold).astype(int)
    df.loc[np.isnan(sample_scores), "anomaly_flag"] = 0
    df["anomaly_feature"] = [feature_cols[i] if i >= 0 else "" for i in top_feature]

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
