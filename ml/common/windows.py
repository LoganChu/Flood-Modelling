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
