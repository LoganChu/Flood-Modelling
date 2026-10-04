"""Sliding-window dataset construction with time-gap boundary exclusion."""

import numpy as np


def make_windows(
    features: np.ndarray,
    gap_mask: np.ndarray,
    window: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows, excluding any that span a gap boundary.

    Args:
        features:  (N, F) array of feature values.
        gap_mask:  (N,) boolean array; True at index i means a time-gap
                   precedes sample i.  Windows containing a True index
                   are dropped.
        window:    Number of past timesteps in each input window.
        horizon:   Number of future timesteps to predict (0 for autoencoder).
        stride:    Step between consecutive windows.

    Returns:
        X: (M, window, F) input windows.
        y: (M, horizon, F) target windows if horizon > 0, else same as X
           (autoencoder — target = input).
    """
    n, f = features.shape
    span = window + horizon  # total span of one sample

    X_list = []
    y_list = []

    for start in range(0, n - span + 1, stride):
        end = start + span
        if gap_mask[start:end].any():
            continue
        x_win = features[start : start + window]
        if horizon > 0:
            y_win = features[start + window : end]
        else:
            y_win = x_win  # autoencoder: reconstruct input
        X_list.append(x_win)
        y_list.append(y_win)

    if not X_list:
        return np.empty((0, window, f)), np.empty((0, max(horizon, window), f))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def blocked_split_indices(
    n: int,
    val_fraction: float,
    n_blocks: int,
    gap_buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/val window indices for one run, with validation spread through it.

    The run's n windows are divided into n_blocks equal segments and one
    contiguous validation block is taken from the middle of each, so
    validation samples the whole run rather than a single stretch. gap_buffer
    windows on both sides of every block are dropped to prevent leakage from
    overlapping windows.

    Args:
        n:              Number of windows in the run (in chronological order).
        val_fraction:   Fraction of the run's windows to use for validation.
        n_blocks:       Number of validation blocks.
        gap_buffer:     Number of windows to drop on each side of a block.

    Returns:
        train_idx, val_idx: sorted index arrays into the run's windows.
    """
    is_val = np.zeros(n, dtype=bool)
    is_buffer = np.zeros(n, dtype=bool)
    block_len = max(1, int(round(n * val_fraction / n_blocks)))
    edges = np.linspace(0, n, n_blocks + 1)
    for k in range(n_blocks):
        centre = (edges[k] + edges[k + 1]) / 2
        start = max(0, int(round(centre - block_len / 2)))
        end = min(n, start + block_len)
        is_val[start:end] = True
        is_buffer[max(0, start - gap_buffer) : start] = True
        is_buffer[end : min(n, end + gap_buffer)] = True
    is_buffer &= ~is_val
    idx = np.arange(n)
    return idx[~is_val & ~is_buffer], idx[is_val]


def blocked_split(
    X_runs: list[np.ndarray],
    y_runs: list[np.ndarray],
    val_fraction: float,
    n_blocks: int,
    gap_buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split each run's windows with blocked_split_indices and concatenate.

    Args:
        X_runs, y_runs: Per-run window arrays from make_windows.
        val_fraction:   Fraction of each run's windows to use for validation.
        n_blocks:       Number of validation blocks per run.
        gap_buffer:     Number of windows to drop on each side of a block.
    """
    X_train, y_train, X_val, y_val = [], [], [], []
    for X_r, y_r in zip(X_runs, y_runs):
        train_idx, val_idx = blocked_split_indices(len(X_r), val_fraction, n_blocks, gap_buffer)
        X_train.append(X_r[train_idx])
        y_train.append(y_r[train_idx])
        X_val.append(X_r[val_idx])
        y_val.append(y_r[val_idx])
    return (
        np.concatenate(X_train),
        np.concatenate(y_train),
        np.concatenate(X_val),
        np.concatenate(y_val),
    )


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    gap_buffer: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split windows into train/val keeping chronological order.

    The last val_fraction of windows become validation; a gap_buffer of
    windows at the boundary is excluded to prevent leakage from overlapping
    windows.

    Args:
        X, y:           Window arrays from make_windows.
        val_fraction:   Fraction of windows to use for validation.
        gap_buffer:     Number of windows to drop at the train/val boundary.
    """
    n = len(X)
    val_start = int(n * (1.0 - val_fraction))
    train_end = max(0, val_start - gap_buffer)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[val_start:]
    y_val = y[val_start:]
    return X_train, y_train, X_val, y_val
