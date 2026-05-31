#!/usr/bin/env python3
"""Evaluate the anomaly detection model on all four runs.

Scores each run, checks known millis() gap indices are flagged,
and produces anomaly score plots + trajectory color overlays.

Usage:
    python ml/anomaly_detection/evaluate.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ml.anomaly_detection.score_run import score_csv
from ml.common.io import load_smoothed_runs, _SMOOTHED_DIR

# Known millis() wrap-around gap indices (approx.) verified in smoothed CSVs
_KNOWN_GAPS = {
    "run1": [1331],
    "run2": [],
    "run3": [2312, 2334, 2341],
    "run4": [],
}

_SMOOTHED_FILES = {
    "run1": "enoM30(1)_forward_smoothed.csv",
    "run2": "enoM30(2)_forward_smoothed.csv",
    "run3": "enoM30(4)_forward1_smoothed.csv",
    "run4": "enoM30(4)_forward2_smoothed.csv",
}


def main() -> None:
    out_dir = _REPO_ROOT / "visualizations" / "ml_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = _REPO_ROOT / "ml" / "checkpoints"
    threshold = float(np.load(ckpt_dir / "anomaly_threshold.npy"))

    all_scores = []
    sanity_results = {}

    for run_id, filename in _SMOOTHED_FILES.items():
        csv_path = _SMOOTHED_DIR / filename
        if not csv_path.exists():
            print(f"  {run_id}: CSV not found — skipping")
            continue

        print(f"\n{run_id}:")
        df = score_csv(csv_path)
        all_scores.append(df["anomaly_score"].dropna().values)

        # Sanity check: known gap indices should be flagged
        known = _KNOWN_GAPS.get(run_id, [])
        for idx in known:
            if idx < len(df):
                flagged = bool(df["anomaly_flag"].iloc[idx])
                score = df["anomaly_score"].iloc[idx]
                sanity_results[f"{run_id}[{idx}]"] = {
                    "flagged": flagged,
                    "score": score,
                    "threshold": threshold,
                }
                status = "PASS" if flagged else "FAIL"
                print(f"  [{status}] Gap index {idx}: score={score:.6f} threshold={threshold:.6f}")

        _plot_run_scores(df, run_id, threshold, known, out_dir)
        _plot_trajectory_heatmap(df, run_id, out_dir)

    # Global histogram of reconstruction errors
    if all_scores:
        _plot_error_histogram(np.concatenate(all_scores), threshold, out_dir)

    print("\n=== Sanity Summary ===")
    for key, v in sanity_results.items():
        status = "PASS" if v["flagged"] else "FAIL"
        print(f"  [{status}] {key}: score={v['score']:.6f}")


def _plot_run_scores(
    df: pd.DataFrame, run_id: str, threshold: float, known_gaps: list[int], out_dir: Path
) -> None:
    scores = df["anomaly_score"].values
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(scores, lw=0.8, color="#1f77b4", label="Anomaly score")
    ax.axhline(threshold, color="#d62728", lw=1.5, linestyle="--", label=f"Threshold ({threshold:.4f})")
    for idx in known_gaps:
        if idx < len(scores):
            ax.axvline(idx, color="#ff7f0e", lw=1.5, linestyle=":", label="Known gap" if idx == known_gaps[0] else "_")
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Reconstruction MSE", fontsize=10)
    ax.set_title(f"Anomaly Score — {run_id}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"anomaly_scores_{run_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def _plot_trajectory_heatmap(df: pd.DataFrame, run_id: str, out_dir: Path) -> None:
    scores = df["anomaly_score"].values
    east = df["east_smooth"].values if "east_smooth" in df.columns else df["east_raw"].values
    north = df["north_smooth"].values if "north_smooth" in df.columns else df["north_raw"].values

    valid = ~np.isnan(scores)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(east[valid], north[valid], c=scores[valid], cmap="viridis",
                    s=8, alpha=0.8, vmin=np.nanpercentile(scores, 1), vmax=np.nanpercentile(scores, 99))
    plt.colorbar(sc, ax=ax, label="Reconstruction MSE")
    ax.set_xlabel("East (m)", fontsize=10)
    ax.set_ylabel("North (m)", fontsize=10)
    ax.set_title(f"Anomaly Score Heatmap — {run_id}", fontsize=11)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"anomaly_heatmap_{run_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def _plot_error_histogram(all_scores: np.ndarray, threshold: float, out_dir: Path) -> None:
    pct95 = float(np.percentile(all_scores, 95))
    pct99 = float(np.percentile(all_scores, 99))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_scores, bins=100, color="#1f77b4", alpha=0.75, log=True)
    ax.axvline(pct95, color="#ff7f0e", lw=1.5, linestyle="--", label=f"95th pctile ({pct95:.4f})")
    ax.axvline(pct99, color="#d62728", lw=1.5, linestyle="--", label=f"99th pctile / threshold ({pct99:.4f})")
    ax.set_xlabel("Reconstruction MSE", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.set_title("Distribution of Per-Sample Reconstruction Errors", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "anomaly_error_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
