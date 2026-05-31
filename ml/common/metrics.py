"""Evaluation metrics and baselines for ML modules."""

import numpy as np


def rmse_2d(pred: np.ndarray, true: np.ndarray) -> float:
    """2D Euclidean RMSE in metres.

    Args:
        pred: (N, 2) predicted (delta_east, delta_north) offsets.
        true: (N, 2) ground-truth offsets.
    Returns:
        Scalar RMSE across all N samples.
    """
    diff = pred - true  # (N, 2)
    return float(np.sqrt(np.mean(diff[:, 0] ** 2 + diff[:, 1] ** 2)))


def mae_2d(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean 2D Euclidean absolute error in metres."""
    diff = pred - true
    return float(np.mean(np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)))


def rmse_per_horizon(
    pred: np.ndarray, true: np.ndarray
) -> np.ndarray:
    """Per-horizon 2D RMSE.

    Args:
        pred: (N, H, 2) predicted position offsets.
        true: (N, H, 2) ground-truth position offsets.
    Returns:
        (H,) array of RMSE values, one per horizon step.
    """
    diff = pred - true  # (N, H, 2)
    return np.sqrt(np.mean(diff[..., 0] ** 2 + diff[..., 1] ** 2, axis=0))


def constant_velocity_baseline(
    vx_last: np.ndarray,
    vz_last: np.ndarray,
    dt: float,
    horizon: int,
) -> np.ndarray:
    """Predict position offsets using constant velocity from the last window step.

    Args:
        vx_last: (N,) east velocity at the last input timestep (m/s).
        vz_last: (N,) north velocity at the last input timestep (m/s).
        dt:      Timestep duration (seconds); assumed uniform.
        horizon: Number of prediction steps.
    Returns:
        (N, H, 2) cumulative position offsets (delta_east, delta_north).
    """
    n = len(vx_last)
    steps = np.arange(1, horizon + 1, dtype=np.float32)  # (H,)
    delta_east = (vx_last[:, None] * dt) * steps[None, :]   # (N, H)
    delta_north = (vz_last[:, None] * dt) * steps[None, :]  # (N, H)
    return np.stack([delta_east, delta_north], axis=-1)      # (N, H, 2)
