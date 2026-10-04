#!/usr/bin/env python3
"""Unsupervised LSTM Autoencoder training for anomaly detection.

All four runs are used for training (no anomaly labels; all runs are 'normal').
After training, the model is calibrated on the same runs: the median
reconstruction error of each feature becomes the score normaliser, and the
configured percentile of the per-sample anomaly score becomes the threshold
(see ml/anomaly_detection/scoring.py).

Usage:
    python ml/anomaly_detection/train.py [--max-epochs N] [--device cuda|cpu]
    python ml/anomaly_detection/train.py --calibrate-only   # reuse saved model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ml.anomaly_detection.dataset import AutoencoderWindowDataset
from ml.anomaly_detection.model import LSTMAutoencoder
from ml.anomaly_detection.scoring import anomaly_scores, per_sample_feature_errors
from ml.common.features import derive_features, get_gap_mask
from ml.common.io import load_smoothed_runs
from ml.common.scaler import apply_scaler, fit_scaler, load_scaler, save_scaler
from ml.common.windows import blocked_split, make_windows


def _load_cfg() -> dict:
    with open(_REPO_ROOT / "configs" / "ml.yaml") as f:
        return yaml.safe_load(f)["anomaly"]


def _load_run_features(
    runs: list[dict],
    feature_cols: list[str],
    gap_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (features, gap_mask) for each run."""
    run_features = []
    for run in runs:
        df = derive_features(run["df"])
        gap_mask = get_gap_mask(df, threshold_s=gap_threshold)
        feat = df[feature_cols].values.astype(np.float32)
        run_features.append((feat, gap_mask))
    return run_features


def _build_run_windows(
    run_features: list[tuple[np.ndarray, np.ndarray]],
    window: int,
) -> list[np.ndarray]:
    """Build reconstruction windows for each run."""
    X_runs = []
    for feat, gap_mask in run_features:
        X_r, _ = make_windows(feat, gap_mask, window=window, horizon=0, stride=1)
        if len(X_r):
            X_runs.append(X_r)
    return X_runs


def _train_model(
    X_runs_scaled: list[np.ndarray],
    cfg: dict,
    max_epochs: int,
    device: torch.device,
) -> tuple[LSTMAutoencoder, dict]:
    """Train the autoencoder; return it with its best weights and a training summary."""
    window: int = cfg["window"]

    # Validation = blocks spread through every run, buffered against window overlap
    X_train, _, X_val, _ = blocked_split(
        X_runs_scaled,
        X_runs_scaled,
        val_fraction=cfg["val_fraction"],
        n_blocks=cfg["val_blocks_per_run"],
        gap_buffer=window,
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    batch_size: int = cfg["batch_size"]
    train_loader = DataLoader(AutoencoderWindowDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AutoencoderWindowDataset(X_val), batch_size=batch_size)

    model = LSTMAutoencoder(input_size=X_train.shape[-1], hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg["lr_patience"], factor=cfg["lr_factor"])
    criterion = nn.MSELoss()
    patience: int = cfg["early_stop_patience"]

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
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
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_losses.append(criterion(model(X_b), y_b).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val_loss:.6f})")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3}: val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    summary = {"val_loss": best_val_loss, "best_epoch": best_epoch, "epochs_run": epoch}
    return model, summary


def calibrate(
    model: LSTMAutoencoder,
    run_features: list[tuple[np.ndarray, np.ndarray]],
    scaler: StandardScaler,
    window: int,
    threshold_pctile: float,
) -> tuple[np.ndarray, float]:
    """Derive the per-feature error normaliser and the anomaly threshold.

    Returns:
        feature_norm: (F,) median per-sample squared error of each feature.
        threshold:    threshold_pctile-th percentile of the per-sample anomaly score.
    """
    error_parts = []
    for feat, gap_mask in run_features:
        errors = per_sample_feature_errors(model, apply_scaler(feat, scaler), gap_mask, window)
        error_parts.append(errors[~np.isnan(errors).any(axis=1)])
    all_errors = np.concatenate(error_parts)

    feature_norm = np.median(all_errors, axis=0)
    scores, _ = anomaly_scores(all_errors, feature_norm)
    threshold = float(np.percentile(scores, threshold_pctile))
    return feature_norm, threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Skip training; recompute the score normaliser and threshold for the saved model",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    cfg = _load_cfg()
    max_epochs = args.max_epochs if args.max_epochs is not None else cfg["max_epochs"]
    feature_cols: list[str] = cfg["features"]
    window: int = cfg["window"]
    threshold_pctile: float = cfg["threshold_pctile"]

    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runs = load_smoothed_runs()
    run_features = _load_run_features(runs, feature_cols, cfg["gap_threshold_s"])

    if args.calibrate_only:
        ckpt = torch.load(ckpt_dir / "anomaly_model.pt", weights_only=True)
        model = LSTMAutoencoder(**ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        scaler = load_scaler(ckpt_dir / "anomaly_scaler.joblib")
    else:
        X_runs = _build_run_windows(run_features, window)
        print(f"Total windows: {sum(len(X_r) for X_r in X_runs)}")

        # Fit scaler on all data (no train/test split for anomaly detection)
        scaler = fit_scaler(np.concatenate(X_runs))
        X_runs_scaled = [apply_scaler(X_r, scaler) for X_r in X_runs]

        model, summary = _train_model(X_runs_scaled, cfg, max_epochs, device)

        torch.save(
            {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                **summary,
                "cfg": {"input_size": len(feature_cols), "hidden_dim": cfg["hidden_dim"]},
            },
            ckpt_dir / "anomaly_model.pt",
        )
        save_scaler(scaler, ckpt_dir / "anomaly_scaler.joblib")

    print("\nCalibrating anomaly score on training data...")
    feature_norm, threshold = calibrate(model, run_features, scaler, window, threshold_pctile)
    for name, norm in zip(feature_cols, feature_norm):
        print(f"  median error {name:<12} = {norm:.6f}")
    print(f"Threshold @ {threshold_pctile}th pctile = {threshold:.6f}")

    np.save(ckpt_dir / "anomaly_feature_norm.npy", feature_norm)
    np.save(ckpt_dir / "anomaly_threshold.npy", np.array(threshold))
    print(f"Saved checkpoints to {ckpt_dir}")


if __name__ == "__main__":
    main()
