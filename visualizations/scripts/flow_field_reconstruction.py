#!/usr/bin/env python3
"""
2D river flow-field reconstruction from EKF-smoothed drifter data.

Reads four pre-computed EKF/RTS smoothed CSVs, fits independent Gaussian
Process regressors for vx (eastward) and vz (northward) velocity, evaluates
on a regular grid, and produces quiver, streamplot, uncertainty, and CSV
outputs.

Usage:
    python visualizations/scripts/flow_field_reconstruction.py [options]

Options:
    --data-dir       Path to smoothed_forward directory
    --out-dir        Output directory (created if absent)
    --n-train        Soft cap on GP training points (default 800)
    --grid-size      Evaluation grid resolution (default 40)
    --speed-threshold  Min speed filter in m/s (default 0.05)
    --n-restarts     GP hyperparameter optimisation restarts (default 3)
    --seed           Random seed (default 42)

Outputs (written to --out-dir):
    flow_field_quiver.png        Speed heatmap + velocity arrows + drifter tracks
    flow_field_streamplot.png    Flow streamlines coloured by speed
    flow_field_uncertainty.png   GP posterior std dev for vx and vz
    flow_field_grid.csv          40x40 grid: east_m, north_m, vx/vz pred/std

No Isaac Sim dependency.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except ImportError:
    print(
        "Error: scikit-learn not installed.\n"
        "Install with: pip install scikit-learn"
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOOTHED_DIR = _REPO_ROOT / "visualizations" / "smoothed_forward"
_OUT_DIR = _REPO_ROOT / "visualizations" / "flow_field"

SMOOTHED_CSVS: list[tuple[Path, str, str]] = [
    (_SMOOTHED_DIR / "enoM30(1)_forward_smoothed.csv",  "Run 1 (M30-1)",  "#1f77b4"),
    (_SMOOTHED_DIR / "enoM30(2)_forward_smoothed.csv",  "Run 2 (M30-2)",  "#ff7f0e"),
    (_SMOOTHED_DIR / "enoM30(4)_forward1_smoothed.csv", "Run 3 (M30-4a)", "#2ca02c"),
    (_SMOOTHED_DIR / "enoM30(4)_forward2_smoothed.csv", "Run 4 (M30-4b)", "#d62728"),
]

# ── Defaults ─────────────────────────────────────────────────────────────────

SPEED_THRESHOLD = 0.05   # m/s — below this GPS noise dominates velocity
CELL_SIZE_M = 2.0        # metres — spatial subsampling grid cell size
N_TRAIN_TARGET = 800     # soft cap on GP training points
GRID_SIZE = 40           # evaluation grid is GRID_SIZE × GRID_SIZE
MARGIN_FRAC = 0.10       # fractional bounding-box margin on each side
N_RESTARTS = 3           # GP kernel optimisation restarts
RANDOM_SEED = 42


# ── Data loading & filtering ─────────────────────────────────────────────────

def load_smoothed_runs(
    csv_specs: list[tuple[Path, str, str]] | None = None,
) -> pd.DataFrame:
    """Load and concatenate all smoothed EKF output CSVs.

    Parameters
    ----------
    csv_specs:
        List of (path, label, color) tuples. Defaults to SMOOTHED_CSVS.

    Returns
    -------
    Concatenated DataFrame with added ``run_label`` and ``run_color`` columns.
    Key velocity columns: ``vx_ekf`` (east), ``vz_ekf`` (north), both in m/s.
    Position columns: ``east_rts``, ``north_rts`` (RTS-smoothed, ENU metres).
    """
    if csv_specs is None:
        csv_specs = SMOOTHED_CSVS

    frames: list[pd.DataFrame] = []
    for path, label, color in csv_specs:
        if not path.exists():
            log.warning("  [skip] not found: %s", path)
            continue
        df = pd.read_csv(path, low_memory=False)
        df["run_label"] = label
        df["run_color"] = color
        frames.append(df)
        log.info("  Loaded %-20s  %d rows", label, len(df))

    if not frames:
        raise RuntimeError(
            f"No smoothed CSVs found in {_SMOOTHED_DIR}.\n"
            "Run visualizations/scripts/smooth_forward_runs.py first."
        )

    return pd.concat(frames, ignore_index=True)


def filter_low_speed(
    df: pd.DataFrame,
    threshold: float = SPEED_THRESHOLD,
) -> pd.DataFrame:
    """Remove samples where EKF speed magnitude is below threshold.

    At low speeds the GPS positional noise dominates the velocity estimate,
    making vx_ekf / vz_ekf unreliable as training targets.
    """
    speed = np.hypot(df["vx_ekf"].values, df["vz_ekf"].values)
    df = df.copy()
    df["speed_ekf"] = speed
    filtered = df[speed >= threshold].reset_index(drop=True)
    log.info(
        "  Speed filter (>= %.2f m/s): %d / %d samples kept",
        threshold, len(filtered), len(df),
    )
    return filtered


# ── Spatial subsampling ───────────────────────────────────────────────────────

def spatial_subsample(
    df: pd.DataFrame,
    n_target: int = N_TRAIN_TARGET,
    cell_m: float = CELL_SIZE_M,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Grid-based spatial subsampling for well-distributed GP training points.

    Algorithm:
    1. Bin RTS positions into cell_m × cell_m grid cells.
    2. Randomly select one sample per occupied cell.
    3. If occupied cells > n_target, randomly drop cells to n_target.

    Uses east_rts / north_rts (RTS-smoothed, cleaner than EKF positions).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    east = df["east_rts"].values
    north = df["north_rts"].values

    cell_e = np.floor(east / cell_m).astype(int)
    cell_n = np.floor(north / cell_m).astype(int)
    cell_id = cell_e * 1_000_003 + cell_n  # hash unique per (ce, cn)

    df = df.copy()
    df["_cell"] = cell_id

    # One representative sample per occupied cell
    one_per_cell = df.groupby("_cell").sample(n=1, random_state=int(rng.integers(2**31)))

    n_cells = len(one_per_cell)
    if n_cells > n_target:
        one_per_cell = one_per_cell.sample(n=n_target, random_state=int(rng.integers(2**31)))

    result = one_per_cell.drop(columns=["_cell"]).reset_index(drop=True)
    log.info(
        "  Spatial subsample: %d occupied cells → %d training points",
        n_cells, len(result),
    )
    return result


# ── Gaussian Process ──────────────────────────────────────────────────────────

def build_gp_kernel() -> Any:
    """Construct the GP kernel for river velocity field reconstruction.

    ConstantKernel * Matern(nu=1.5) + WhiteKernel

    Matern(nu=1.5) — once-differentiable covariance; appropriate for turbulent
    surface velocity fields that are smooth but not infinitely differentiable.
    WhiteKernel absorbs GPS measurement noise and sub-grid velocity fluctuations.
    normalize_y=True is set on the GPR (not the kernel) to handle the nonzero
    mean of vz_ekf (river flows south → negative mean).
    """
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 100.0))
        * Matern(length_scale=30.0, length_scale_bounds=(2.0, 500.0), nu=1.5)
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0))
    )


def fit_gp_models(
    train_df: pd.DataFrame,
    n_restarts: int = N_RESTARTS,
    seed: int = RANDOM_SEED,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Fit independent GP regressors for vx and vz velocity components.

    Input features: [east_rts, north_rts] in metres.
    Targets: vx_ekf (east, m/s) and vz_ekf (north, m/s).

    Returns
    -------
    gpr_vx, gpr_vz : fitted GaussianProcessRegressor instances
    """
    X_train = train_df[["east_rts", "north_rts"]].values
    y_vx = train_df["vx_ekf"].values
    y_vz = train_df["vz_ekf"].values

    def _make_gpr() -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=build_gp_kernel(),
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
        )

    log.info("  Fitting GP for vx (east velocity) …")
    gpr_vx = _make_gpr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr_vx.fit(X_train, y_vx)
    log.info("    kernel: %s", gpr_vx.kernel_)

    log.info("  Fitting GP for vz (north velocity) …")
    gpr_vz = _make_gpr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr_vz.fit(X_train, y_vz)
    log.info("    kernel: %s", gpr_vz.kernel_)

    return gpr_vx, gpr_vz


# ── Grid construction & prediction ───────────────────────────────────────────

def build_eval_grid(
    df: pd.DataFrame,
    grid_size: int = GRID_SIZE,
    margin: float = MARGIN_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a regular evaluation grid over the observed river domain.

    Returns
    -------
    east_1d  : shape (grid_size,)  — grid east axis (m)
    north_1d : shape (grid_size,)  — grid north axis (m)
    X_grid   : shape (grid_size**2, 2)  — flattened (east, north) pairs
               Row layout: meshgrid(east_1d, north_1d) with indexing='xy',
               so that reshaping flat predictions to (grid_size, grid_size)
               gives [ny, nx] layout compatible with pcolormesh/streamplot.
    """
    e_min, e_max = df["east_rts"].min(), df["east_rts"].max()
    n_min, n_max = df["north_rts"].min(), df["north_rts"].max()

    e_pad = (e_max - e_min) * margin
    n_pad = (n_max - n_min) * margin

    east_1d = np.linspace(e_min - e_pad, e_max + e_pad, grid_size)
    north_1d = np.linspace(n_min - n_pad, n_max + n_pad, grid_size)

    E_2d, N_2d = np.meshgrid(east_1d, north_1d)  # both (ny, nx)
    X_grid = np.column_stack([E_2d.ravel(), N_2d.ravel()])

    log.info(
        "  Grid: %.0f–%.0f m east, %.0f–%.0f m north, %dx%d",
        east_1d[0], east_1d[-1], north_1d[0], north_1d[-1], grid_size, grid_size,
    )
    return east_1d, north_1d, X_grid


def predict_on_grid(
    gpr_vx: GaussianProcessRegressor,
    gpr_vz: GaussianProcessRegressor,
    X_grid: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> dict[str, np.ndarray]:
    """Evaluate fitted GPs on the evaluation grid.

    Returns dict of (grid_size, grid_size) arrays in [ny, nx] layout:
        vx_pred, vx_std  — eastward velocity mean and posterior std
        vz_pred, vz_std  — northward velocity mean and posterior std
        speed_pred       — speed magnitude sqrt(vx² + vz²)
        speed_std        — first-order propagated speed uncertainty
    """
    log.info("  Predicting on %dx%d grid …", grid_size, grid_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vx_flat, vx_std_flat = gpr_vx.predict(X_grid, return_std=True)
        vz_flat, vz_std_flat = gpr_vz.predict(X_grid, return_std=True)

    ny = nx = grid_size
    vx_pred = vx_flat.reshape(ny, nx)
    vz_pred = vz_flat.reshape(ny, nx)
    vx_std = vx_std_flat.reshape(ny, nx)
    vz_std = vz_std_flat.reshape(ny, nx)

    speed_pred = np.hypot(vx_pred, vz_pred)
    # First-order uncertainty: σ_s = sqrt((vx·σvx)² + (vz·σvz)²) / max(s, ε)
    eps = 1e-6
    speed_std = np.sqrt(
        (vx_pred * vx_std) ** 2 + (vz_pred * vz_std) ** 2
    ) / np.maximum(speed_pred, eps)

    log.info(
        "  Predicted speed: mean=%.3f m/s, max=%.3f m/s",
        speed_pred.mean(), speed_pred.max(),
    )
    return {
        "vx_pred": vx_pred,
        "vz_pred": vz_pred,
        "vx_std": vx_std,
        "vz_std": vz_std,
        "speed_pred": speed_pred,
        "speed_std": speed_std,
    }


# ── CSV output ────────────────────────────────────────────────────────────────

def save_grid_csv(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
    fields: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Save predicted flow-field grid to CSV.

    Output columns (grid_size² rows):
        east_m, north_m, vx_pred, vz_pred, vx_std, vz_std, speed_pred, speed_std
    All velocity values in m/s; positions in metres (shared ENU origin).
    """
    E_2d, N_2d = np.meshgrid(east_1d, north_1d)
    rows = pd.DataFrame({
        "east_m":     E_2d.ravel(),
        "north_m":    N_2d.ravel(),
        "vx_pred":    fields["vx_pred"].ravel(),
        "vz_pred":    fields["vz_pred"].ravel(),
        "vx_std":     fields["vx_std"].ravel(),
        "vz_std":     fields["vz_std"].ravel(),
        "speed_pred": fields["speed_pred"].ravel(),
        "speed_std":  fields["speed_std"].ravel(),
    })
    rows.to_csv(out_path, index=False, float_format="%.6f")
    log.info("  Saved grid CSV: %s  (%d rows)", out_path.name, len(rows))


# ── Plots ─────────────────────────────────────────────────────────────────────

def _overlay_tracks(
    ax: plt.Axes,
    raw_df: pd.DataFrame,
    alpha: float = 0.45,
    lw: float = 0.9,
) -> list[mlines.Line2D]:
    """Overlay per-run RTS drifter tracks. Returns legend handles."""
    handles = []
    for label in raw_df["run_label"].unique():
        run = raw_df[raw_df["run_label"] == label]
        color = run["run_color"].iloc[0]
        ax.plot(run["east_rts"], run["north_rts"],
                color=color, lw=lw, alpha=alpha, zorder=4)
        ax.plot(run["east_rts"].iloc[0], run["north_rts"].iloc[0],
                "o", color=color, ms=5, zorder=6)
        ax.plot(run["east_rts"].iloc[-1], run["north_rts"].iloc[-1],
                "*", color=color, ms=9, zorder=6)
        handles.append(mlines.Line2D([], [], color=color, lw=2, label=label))
    return handles


def plot_quiver(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
    fields: dict[str, np.ndarray],
    raw_df: pd.DataFrame,
    out_path: Path,
    quiver_step: int = 2,
) -> None:
    """Speed heatmap + velocity arrows + drifter tracks."""
    vmax = float(np.percentile(fields["speed_pred"], 98))

    fig, ax = plt.subplots(figsize=(7, 18))

    pcm = ax.pcolormesh(
        east_1d, north_1d, fields["speed_pred"],
        cmap="viridis", vmin=0, vmax=vmax, shading="auto",
    )
    plt.colorbar(pcm, ax=ax, label="Predicted Speed (m/s)", pad=0.02)

    # Velocity arrows on subsampled grid
    E_2d, N_2d = np.meshgrid(east_1d, north_1d)
    s = quiver_step
    ax.quiver(
        E_2d[::s, ::s], N_2d[::s, ::s],
        fields["vx_pred"][::s, ::s], fields["vz_pred"][::s, ::s],
        color="white", alpha=0.85, units="xy", scale=3.0, width=0.4,
        zorder=5,
    )

    handles = _overlay_tracks(ax, raw_df)
    handles += [
        mlines.Line2D([], [], marker="o", color="grey", ls="none", ms=5, label="Start"),
        mlines.Line2D([], [], marker="*", color="grey", ls="none", ms=9, label="End"),
    ]

    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title(
        "Eno River — Reconstructed 2D Flow Field\n"
        "Background: GP-predicted speed  |  Arrows: velocity vectors",
        fontsize=12, fontweight="bold",
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.15, color="white")
    ax.axis("equal")
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_streamplot(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
    fields: dict[str, np.ndarray],
    raw_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Flow streamlines coloured by speed + drifter tracks."""
    from matplotlib.colors import Normalize

    vmax = float(np.percentile(fields["speed_pred"], 98))
    norm = Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 18))

    strm = ax.streamplot(
        east_1d, north_1d,
        fields["vx_pred"], fields["vz_pred"],
        color=fields["speed_pred"],
        cmap="viridis",
        norm=norm,
        density=1.5,
        linewidth=1.3,
        arrowsize=1.1,
    )
    plt.colorbar(strm.lines, ax=ax, label="Speed (m/s)", pad=0.02)

    _overlay_tracks(ax, raw_df)

    ax.set_xlabel("East (m)", fontsize=11)
    ax.set_ylabel("North (m)", fontsize=11)
    ax.set_title(
        "Eno River — GP Flow-Field Streamlines",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.15)
    ax.axis("equal")
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


def plot_uncertainty(
    east_1d: np.ndarray,
    north_1d: np.ndarray,
    fields: dict[str, np.ndarray],
    train_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Two-panel GP posterior std dev map with training point overlay."""
    vmax = float(np.percentile(
        np.concatenate([fields["vx_std"].ravel(), fields["vz_std"].ravel()]), 95
    ))

    fig, axes = plt.subplots(1, 2, figsize=(13, 18))

    for ax, std_arr, label in zip(
        axes,
        [fields["vx_std"], fields["vz_std"]],
        ["σ_vx (m/s)", "σ_vz (m/s)"],
    ):
        pcm = ax.pcolormesh(
            east_1d, north_1d, std_arr,
            cmap="plasma", vmin=0, vmax=vmax, shading="auto",
        )
        plt.colorbar(pcm, ax=ax, label=label, pad=0.02)
        ax.scatter(
            train_df["east_rts"], train_df["north_rts"],
            s=1.5, color="white", alpha=0.4, zorder=5,
        )
        ax.set_xlabel("East (m)", fontsize=11)
        ax.set_ylabel("North (m)", fontsize=11)
        ax.grid(True, alpha=0.15, color="white")
        ax.axis("equal")

    axes[0].set_title("Eastward Velocity (vx) Uncertainty", fontsize=11)
    axes[1].set_title("Northward Velocity (vz) Uncertainty", fontsize=11)
    fig.suptitle(
        "GP Posterior Std Dev — High σ indicates sparse data coverage\n"
        "White dots = GP training points",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", out_path.name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct 2D river flow field via Gaussian Process regression."
    )
    p.add_argument("--data-dir", type=Path, default=_SMOOTHED_DIR,
                   help="Directory containing smoothed forward-run CSVs")
    p.add_argument("--out-dir", type=Path, default=_OUT_DIR,
                   help="Output directory for plots and CSV")
    p.add_argument("--n-train", type=int, default=N_TRAIN_TARGET,
                   help="Soft cap on GP training points (default %(default)s)")
    p.add_argument("--grid-size", type=int, default=GRID_SIZE,
                   help="Evaluation grid resolution (default %(default)s)")
    p.add_argument("--speed-threshold", type=float, default=SPEED_THRESHOLD,
                   help="Min speed filter in m/s (default %(default)s)")
    p.add_argument("--n-restarts", type=int, default=N_RESTARTS,
                   help="GP optimiser restarts (default %(default)s)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Random seed (default %(default)s)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Remap CSV paths if a custom data-dir was given
    if args.data_dir != _SMOOTHED_DIR:
        csv_specs = [
            (args.data_dir / p.name, label, color)
            for p, label, color in SMOOTHED_CSVS
        ]
    else:
        csv_specs = SMOOTHED_CSVS

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("\n[1/6] Loading smoothed CSVs …")
    raw_df = load_smoothed_runs(csv_specs)

    # ── 2. Filter ─────────────────────────────────────────────────────────────
    log.info("\n[2/6] Filtering low-speed samples …")
    filtered_df = filter_low_speed(raw_df, threshold=args.speed_threshold)

    # ── 3. Subsample ──────────────────────────────────────────────────────────
    log.info("\n[3/6] Spatial subsampling …")
    rng = np.random.default_rng(args.seed)
    train_df = spatial_subsample(filtered_df, n_target=args.n_train, rng=rng)

    # ── 4. Fit GPs ────────────────────────────────────────────────────────────
    log.info("\n[4/6] Fitting Gaussian Process models …")
    gpr_vx, gpr_vz = fit_gp_models(train_df, n_restarts=args.n_restarts, seed=args.seed)

    # Sanity check: warn if kernel length_scale hit a bound
    for name, gpr in (("vx", gpr_vx), ("vz", gpr_vz)):
        ls = gpr.kernel_.get_params().get("k1__k2__length_scale", None)
        if ls is not None and (ls < 3.0 or ls > 490.0):
            log.warning("  [warn] %s GP length_scale=%.1f m may be at bound", name, ls)

    # ── 5. Predict on grid ────────────────────────────────────────────────────
    log.info("\n[5/6] Predicting on evaluation grid …")
    east_1d, north_1d, X_grid = build_eval_grid(
        filtered_df, grid_size=args.grid_size, margin=MARGIN_FRAC,
    )
    fields = predict_on_grid(gpr_vx, gpr_vz, X_grid, grid_size=args.grid_size)

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    log.info("\n[6/6] Saving outputs to %s …", args.out_dir)

    save_grid_csv(east_1d, north_1d, fields,
                  out_path=args.out_dir / "flow_field_grid.csv")

    plot_quiver(east_1d, north_1d, fields, raw_df,
                out_path=args.out_dir / "flow_field_quiver.png")

    plot_streamplot(east_1d, north_1d, fields, raw_df,
                    out_path=args.out_dir / "flow_field_streamplot.png")

    plot_uncertainty(east_1d, north_1d, fields, train_df,
                     out_path=args.out_dir / "flow_field_uncertainty.png")

    log.info("\n[Done] All outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
