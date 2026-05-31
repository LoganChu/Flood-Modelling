#!/usr/bin/env python3
"""Unsupervised LSTM Autoencoder training for anomaly detection.

All four runs are used for training (no anomaly labels; all runs are 'normal').
After training, the 99th-percentile reconstruction error on training data
becomes the anomaly threshold.

Usage:
    python ml/anomaly_detection/train.py [--max-epochs N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ml.anomaly_detection.dataset import AutoencoderWindowDataset
from ml.anomaly_detection.model import LSTMAutoencoder
from ml.common.features import derive_features, get_gap_mask
from ml.common.io import load_smoothed_runs
from ml.common.scaler import apply_scaler, fit_scaler, save_scaler
from ml.common.windows import chronological_split, make_windows


def _load_cfg() -> dict:
    with open(_REPO_ROOT / "configs" / "ml.yaml") as f:
        return yaml.safe_load(f)["anomaly"]


def _build_all_windows(
    runs: list[dict],
    feature_cols: list[str],
    window: int,
    gap_threshold: float,
) -> np.ndarray:
    """Build reconstruction windows from all runs combined."""
    X_parts = []
    for run in runs:
        df = derive_features(run["df"])
        gap_mask = get_gap_mask(df, threshold_s=gap_threshold)
        feat = df[feature_cols].values.astype(np.float32)
        X_r, _ = make_windows(feat, gap_mask, window=window, horizon=0, stride=1)
        if len(X_r):
            X_parts.append(X_r)
    return np.concatenate(X_parts) if X_parts else np.empty((0, window, len(feature_cols)), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_cfg()
    max_epochs = args.max_epochs if args.max_epochs is not None else cfg["max_epochs"]
    feature_cols: list[str] = cfg["features"]
    window: int = cfg["window"]
    gap_thr: float = cfg["gap_threshold_s"]
    hidden_dim: int = cfg["hidden_dim"]
    threshold_pctile: float = cfg["threshold_pctile"]
    batch_size: int = cfg["batch_size"]
    lr: float = cfg["lr"]
    val_frac: float = cfg["val_fraction"]
    patience: int = cfg["early_stop_patience"]
    lr_patience: int = cfg["lr_patience"]
    lr_factor: float = cfg["lr_factor"]

    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs = load_smoothed_runs()
    X_all = _build_all_windows(runs, feature_cols, window, gap_thr)
    print(f"Total windows: {len(X_all)}")

    # Fit scaler on all data (no train/test split for anomaly detection)
    scaler = fit_scaler(X_all)
    X_scaled = apply_scaler(X_all, scaler)

    # Chronological validation split
    X_train, _, X_val, _ = chronological_split(
        X_scaled, X_scaled, val_fraction=val_frac, gap_buffer=window
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    train_loader = DataLoader(AutoencoderWindowDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AutoencoderWindowDataset(X_val), batch_size=batch_size)

    model = LSTMAutoencoder(input_size=len(feature_cols), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_factor)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            recon = model(X_b)
            loss = criterion(recon, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_losses.append(criterion(model(X_b), y_b).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch} (val_loss={best_val_loss:.6f})")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3}: val_loss={val_loss:.6f}")

    # Load best weights and compute threshold from full training set
    model.load_state_dict(best_state)
    model.eval()

    print("\nComputing anomaly threshold from training data...")
    all_errors = []
    with torch.no_grad():
        full_loader = DataLoader(AutoencoderWindowDataset(X_scaled), batch_size=512)
        for X_b, _ in full_loader:
            errors = model.reconstruction_error(X_b)
            all_errors.append(errors.numpy())
    all_errors = np.concatenate(all_errors)
    threshold = float(np.percentile(all_errors, threshold_pctile))
    print(f"Threshold @ {threshold_pctile}th pctile = {threshold:.6f}")

    # Save model, scaler, threshold
    torch.save(
        {
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "cfg": {"input_size": len(feature_cols), "hidden_dim": hidden_dim},
        },
        ckpt_dir / "anomaly_model.pt",
    )
    save_scaler(scaler, ckpt_dir / "anomaly_scaler.joblib")
    np.save(ckpt_dir / "anomaly_threshold.npy", np.array(threshold))
    print(f"Saved checkpoints to {ckpt_dir}")


if __name__ == "__main__":
    main()
