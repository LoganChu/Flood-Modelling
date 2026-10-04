"""Per-sample anomaly scoring from windowed reconstructions.

Each sample is covered by up to `window` overlapping windows. Its error for a
feature is the squared reconstruction error at the sample's own timestep,
averaged over the windows that cover it. The anomaly score is the largest
per-feature error after dividing each feature by its typical (median) error,
so a score of 100 means "100x the usual error for that feature".
"""

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from ml.anomaly_detection.model import LSTMAutoencoder


def per_sample_feature_errors(
    model: LSTMAutoencoder,
    feat_scaled: np.ndarray,
    gap_mask: np.ndarray,
    window: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Squared reconstruction error per sample and feature.

    Args:
        model:       Trained autoencoder (evaluated on its current device).
        feat_scaled: (N, F) scaled features for one run.
        gap_mask:    (N,) boolean; windows containing a True index are skipped.
        window:      Reconstruction window length.

    Returns:
        errors: (N, F). Rows are NaN for samples no valid window covers.
    """
    n, f = feat_scaled.shape
    errors = np.full((n, f), np.nan)
    if n < window:
        return errors

    spans_gap = np.convolve(gap_mask.astype(int), np.ones(window, dtype=int))[window - 1 : n] > 0
    starts = np.where(~spans_gap)[0]
    if len(starts) == 0:
        return errors

    windows = sliding_window_view(feat_scaled, window, axis=0).transpose(0, 2, 1)[starts]
    X = torch.from_numpy(np.ascontiguousarray(windows)).float()
    device = next(model.parameters()).device

    sq_parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_b = X[i : i + batch_size].to(device)
            sq_parts.append(((model(x_b) - x_b) ** 2).cpu().numpy())
    sq = np.concatenate(sq_parts)  # (M, window, F)

    # Sample index of every (window, timestep) cell, for averaging over covering windows
    sample_idx = (starts[:, None] + np.arange(window)[None, :]).ravel()
    counts = np.bincount(sample_idx, minlength=n)
    covered = counts > 0
    for j in range(f):
        sums = np.bincount(sample_idx, weights=sq[:, :, j].ravel(), minlength=n)
        errors[covered, j] = sums[covered] / counts[covered]
    return errors


def anomaly_scores(
    feature_errors: np.ndarray, feature_norm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-feature errors to one score per sample.

    Args:
        feature_errors: (N, F) output of per_sample_feature_errors.
        feature_norm:   (F,) typical (median) error of each feature.

    Returns:
        scores:      (N,) largest normalised feature error; NaN where unscored.
        top_feature: (N,) index of the feature giving that score; -1 where unscored.
    """
    normed = feature_errors / feature_norm
    scored = ~np.isnan(normed).any(axis=1)
    scores = np.full(len(normed), np.nan)
    top_feature = np.full(len(normed), -1, dtype=int)
    scores[scored] = normed[scored].max(axis=1)
    top_feature[scored] = normed[scored].argmax(axis=1)
    return scores, top_feature
