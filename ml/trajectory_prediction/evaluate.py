#!/usr/bin/env python3
"""Evaluate trajectory prediction: RMSE vs constant-velocity baseline.

Usage:
    python ml/trajectory_prediction/evaluate.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ml.common.features import derive_features, get_gap_mask
from ml.common.io import load_smoothed_runs
from ml.common.metrics import constant_velocity_baseline, rmse_per_horizon
from ml.common.scaler import load_scaler
from ml.trajectory_prediction.model import TrajectoryLSTM
from ml.trajectory_prediction.train import _build_windows_for_run


def _load_cfg() -> dict:
    with open(_REPO_ROOT / "configs" / "ml.yaml") as f:
        return yaml.safe_load(f)["trajectory"]


def _load_fold_model(fold_idx: int, ckpt_dir: Path) -> tuple:
    ckpt_path = ckpt_dir / f"trajectory_loo_fold{fold_idx}.pt"
    if not ckpt_path.exists():
        return None, None
    ckpt = torch.load(ckpt_path, weights_only=True)
    model_cfg = ckpt["cfg"]
    model = TrajectoryLSTM(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = load_scaler(ckpt_dir / f"trajectory_scaler_fold{fold_idx}.joblib")
    return model, scaler


def evaluate_fold(
    fold_idx: int,
    model: TrajectoryLSTM,
    scaler,
    test_run: dict,
    cfg: dict,
) -> dict:
    from ml.common.scaler import apply_scaler

    feature_cols = cfg["features"]
    window = cfg["window"]
    horizon = cfg["horizon"]
    gap_thr = cfg["gap_threshold_s"]

    X_test, y_test = _build_windows_for_run(
        test_run, feature_cols, window, horizon, gap_thr
    )
    if len(X_test) == 0:
        print(f"  Fold {fold_idx}: no valid test windows")
        return {}

    X_scaled = apply_scaler(X_test, scaler)
    X_tensor = torch.from_numpy(X_scaled).float()

    with torch.no_grad():
        pred = model(X_tensor).numpy()   # (N, H, 2)

    # Constant-velocity baseline: use last vx_ekf, vz_ekf from each window
    df = derive_features(test_run["df"])
    gap_mask = get_gap_mask(df, threshold_s=gap_thr)
    feat_raw = df[feature_cols].values.astype(np.float32)
    n = len(feat_raw)
    vx_last_list, vz_last_list, dt_list = [], [], []
    for start in range(0, n - window - horizon + 1):
        end = start + window + horizon
        if gap_mask[start:end].any():
            continue
        vx_last_list.append(feat_raw[start + window - 1, feature_cols.index("vx_ekf")])
        vz_last_list.append(feat_raw[start + window - 1, feature_cols.index("vz_ekf")])
        dt_list.append(df["dt_s"].values[start + window])
    vx_last = np.array(vx_last_list, dtype=np.float32)
    vz_last = np.array(vz_last_list, dtype=np.float32)
    dt_median = float(np.median(dt_list)) if dt_list else 1.0

    cv_pred = constant_velocity_baseline(vx_last, vz_last, dt_median, horizon)

    lstm_rmse = rmse_per_horizon(pred, y_test)     # (H,)
    cv_rmse = rmse_per_horizon(cv_pred, y_test)    # (H,)

    print(f"  Fold {fold_idx} (holdout={test_run['run_id']}): "
          f"LSTM h=10 RMSE={lstm_rmse[-1]:.3f}m  CV h=10 RMSE={cv_rmse[-1]:.3f}m")
    return {"fold": fold_idx, "lstm_rmse": lstm_rmse, "cv_rmse": cv_rmse,
            "pred": pred, "true": y_test, "run_id": test_run["run_id"]}


def main() -> None:
    cfg = _load_cfg()
    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"
    out_dir = _REPO_ROOT / "visualizations" / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_smoothed_runs()
    fold_results = []
    for fold_idx, test_run in enumerate(runs):
        model, scaler = _load_fold_model(fold_idx, ckpt_dir)
        if model is None:
            print(f"  Fold {fold_idx}: checkpoint not found — skipping")
            continue
        result = evaluate_fold(fold_idx, model, scaler, test_run, cfg)
        if result:
            fold_results.append(result)

    if not fold_results:
        print("No fold results found. Run train.py first.")
        return

    # ── RMSE-vs-horizon plot ──────────────────────────────────────────────────
    horizon = cfg["horizon"]
    steps = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    lstm_mat = np.array([r["lstm_rmse"] for r in fold_results])
    cv_mat = np.array([r["cv_rmse"] for r in fold_results])

    ax.plot(steps, lstm_mat.mean(0), color="#1f77b4", lw=2.2, label="LSTM (mean)")
    ax.fill_between(steps, lstm_mat.min(0), lstm_mat.max(0), alpha=0.2, color="#1f77b4")
    ax.plot(steps, cv_mat.mean(0), color="#d62728", lw=2.2, linestyle="--", label="CV baseline (mean)")
    ax.fill_between(steps, cv_mat.min(0), cv_mat.max(0), alpha=0.15, color="#d62728")

    ax.set_xlabel("Horizon step", fontsize=11)
    ax.set_ylabel("2D RMSE (m)", fontsize=11)
    ax.set_title("Trajectory Prediction: LSTM vs Constant-Velocity Baseline", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rmse_plot = out_dir / "trajectory_rmse_vs_horizon.png"
    plt.savefig(rmse_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {rmse_plot.name}")

    # ── Per-fold RMSE table as CSV ────────────────────────────────────────────
    import pandas as pd
    rows = []
    for r in fold_results:
        row = {"run": r["run_id"]}
        for h in range(horizon):
            row[f"lstm_h{h+1}"] = round(float(r["lstm_rmse"][h]), 4)
            row[f"cv_h{h+1}"] = round(float(r["cv_rmse"][h]), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    table_path = out_dir / "trajectory_rmse_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved: {table_path.name}")

    # ── Trajectory overlay for best and worst folds ───────────────────────────
    final_rmse = [r["lstm_rmse"][-1] for r in fold_results]
    for label, fold_r in [("best", fold_results[int(np.argmin(final_rmse))]),
                           ("worst", fold_results[int(np.argmax(final_rmse))])]:
        _plot_trajectory_overlay(fold_r, label, cfg, out_dir)


def _plot_trajectory_overlay(result: dict, label: str, cfg: dict, out_dir: Path) -> None:
    pred = result["pred"]    # (N, H, 2)
    true = result["true"]    # (N, H, 2)
    n_show = min(5, len(pred))
    idxs = np.linspace(0, len(pred) - 1, n_show, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, idx in enumerate(idxs):
        color = plt.cm.tab10(i)
        ax.plot(true[idx, :, 0], true[idx, :, 1], "o-", color=color, lw=1.5, label=f"True #{idx}" if i == 0 else "_")
        ax.plot(pred[idx, :, 0], pred[idx, :, 1], "s--", color=color, lw=1.2, alpha=0.7, label="LSTM" if i == 0 else "_")

    ax.set_xlabel("Δeast (m)", fontsize=10)
    ax.set_ylabel("Δnorth (m)", fontsize=10)
    ax.set_title(
        f"Trajectory Overlay — {label} fold (holdout={result['run_id']})\n"
        f"h=10 RMSE: {result['lstm_rmse'][-1]:.3f} m",
        fontsize=11,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="-", label="Ground truth"),
        plt.Line2D([0], [0], marker="s", color="grey", linestyle="--", label="LSTM prediction"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"trajectory_overlay_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
