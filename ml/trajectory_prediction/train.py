#!/usr/bin/env python3
"""4-fold Leave-One-Out CV training for the trajectory prediction LSTM.

Usage:
    python ml/trajectory_prediction/train.py [--max-epochs N] [--folds N]
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

from ml.common.features import derive_features, get_gap_mask
from ml.common.io import load_smoothed_runs
from ml.common.scaler import apply_scaler, fit_scaler, save_scaler
from ml.common.windows import chronological_split, make_windows
from ml.trajectory_prediction.dataset import SlidingWindowDataset
from ml.trajectory_prediction.model import TrajectoryLSTM


def _load_cfg() -> dict:
    cfg_path = _REPO_ROOT / "configs" / "ml.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["trajectory"]


def _build_windows_for_run(
    run: dict,
    feature_cols: list[str],
    window: int,
    horizon: int,
    gap_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    df = derive_features(run["df"])
    gap_mask = get_gap_mask(df, threshold_s=gap_threshold)
    feat = df[feature_cols].values.astype(np.float32)
    pos = df[["east_smooth", "north_smooth"]].values.astype(np.float32)

    # Build position-offset targets from the smoothed positions
    # y[i, h] = pos[window+i+h] - pos[window+i-1]  (offset from last input step)
    n = len(feat)
    span = window + horizon
    X_list, y_list = [], []
    for start in range(0, n - span + 1):
        end = start + span
        if gap_mask[start:end].any():
            continue
        x_win = feat[start : start + window]
        last_pos = pos[start + window - 1]
        y_win = pos[start + window : end] - last_pos   # (H, 2) offsets
        X_list.append(x_win)
        y_list.append(y_win)

    if not X_list:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), \
               np.empty((0, horizon, 2), dtype=np.float32)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_fold(
    fold_idx: int,
    train_runs: list[dict],
    test_run: dict,
    cfg: dict,
    ckpt_dir: Path,
    max_epochs: int,
) -> dict:
    feature_cols: list[str] = cfg["features"]
    window: int = cfg["window"]
    horizon: int = cfg["horizon"]
    gap_thr: float = cfg["gap_threshold_s"]
    val_frac: float = cfg["val_fraction"]
    hidden_dim: int = cfg["hidden_dim"]
    num_layers: int = cfg["num_layers"]
    dropout: float = cfg["dropout"]
    batch_size: int = cfg["batch_size"]
    lr: float = cfg["lr"]
    patience: int = cfg["early_stop_patience"]
    lr_patience: int = cfg["lr_patience"]
    lr_factor: float = cfg["lr_factor"]

    print(f"\n=== Fold {fold_idx}: holdout = {test_run['run_id']} ===")

    # Build training windows from all training runs
    X_parts, y_parts = [], []
    for run in train_runs:
        X_r, y_r = _build_windows_for_run(run, feature_cols, window, horizon, gap_thr)
        if len(X_r):
            X_parts.append(X_r)
            y_parts.append(y_r)
    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    print(f"  Training windows: {len(X_all)}")

    # Fit scaler on training features (flat view)
    scaler = fit_scaler(X_all)
    X_all = apply_scaler(X_all, scaler)

    # Chronological val split using last val_frac of the concatenated training windows.
    # gap_buffer = 3 * window to avoid leakage from overlapping windows at the boundary.
    X_train, y_train, X_val, y_val = chronological_split(
        X_all, y_all, val_fraction=val_frac, gap_buffer=3 * window
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    train_loader = DataLoader(
        SlidingWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SlidingWindowDataset(X_val, y_val), batch_size=batch_size
    )

    model = TrajectoryLSTM(
        input_size=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
    )
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
            loss = criterion(model(X_b), y_b)
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
                print(f"  Early stop at epoch {epoch} (val_loss={best_val_loss:.6f})")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3}: val_loss={val_loss:.6f}")

    # Save checkpoint and scaler
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"trajectory_loo_fold{fold_idx}.pt"
    torch.save(
        {
            "fold": fold_idx,
            "holdout_run": test_run["run_id"],
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "cfg": {
                "input_size": len(feature_cols),
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "horizon": horizon,
            },
        },
        ckpt_path,
    )
    save_scaler(scaler, ckpt_dir / f"trajectory_scaler_fold{fold_idx}.joblib")
    print(f"  Saved: {ckpt_path.name}  best_val_loss={best_val_loss:.6f}")
    return {"fold": fold_idx, "best_val_loss": best_val_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=4, help="Number of LOO folds to run (1-4)")
    args = parser.parse_args()

    cfg = _load_cfg()
    max_epochs = args.max_epochs if args.max_epochs is not None else cfg["max_epochs"]
    ckpt_dir = _REPO_ROOT / cfg.get("checkpoints_dir", "ml/checkpoints").replace(
        "ml/checkpoints",
        str(_REPO_ROOT / "ml" / "checkpoints"),
    )
    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"

    runs = load_smoothed_runs()
    results = []
    for fold_idx in range(min(args.folds, len(runs))):
        test_run = runs[fold_idx]
        train_runs = [r for i, r in enumerate(runs) if i != fold_idx]
        result = train_fold(fold_idx, train_runs, test_run, cfg, ckpt_dir, max_epochs)
        results.append(result)

    print("\n=== LOO-CV Summary ===")
    for r in results:
        print(f"  Fold {r['fold']}: best_val_loss={r['best_val_loss']:.6f}")
    best_fold = min(results, key=lambda r: r["best_val_loss"])
    print(f"  Best fold: {best_fold['fold']} (val_loss={best_fold['best_val_loss']:.6f})")


if __name__ == "__main__":
    main()
