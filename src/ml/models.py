"""Minimal ML model placeholders.

These are intentionally tiny: they provide simple fit/predict interfaces that can later be
replaced with PyTorch or TensorFlow implementations.
"""
from typing import List, Any


class SimpleMLModel:
    """A naive model that predicts the historical mean of the target for any input.

    Purpose: provide a predictable API for transfer/meta-learning scaffolding.
    """

    def __init__(self):
        self._trained = False
        self._mean = 0.0

    def fit(self, X: List[Any], y: List[float], epochs: int = 1):
        if len(y) == 0:
            raise ValueError("y must not be empty")
        self._mean = sum(y) / float(len(y))
        self._trained = True
        return self

    def predict(self, X: List[Any]) -> List[float]:
        if not self._trained:
            # if untrained, be conservative and return zeros
            return [0.0 for _ in X]
        return [self._mean for _ in X]


def load_pretrained(path: str = None) -> SimpleMLModel:
    # stub: normally you'd load weights from disk
    m = SimpleMLModel()
    # pretend it's pretrained by setting a plausible mean
    m._mean = 1.0
    m._trained = True
    return m
