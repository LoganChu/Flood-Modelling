"""Thin wrappers around sklearn StandardScaler for ML pipelines."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit and return a StandardScaler on X (shape N×F or N×W×F, flattened)."""
    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to X, preserving shape (N×F or N×W×F)."""
    shape = X.shape
    flat = X.reshape(-1, shape[-1])
    return scaler.transform(flat).reshape(shape)


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    return joblib.load(path)
