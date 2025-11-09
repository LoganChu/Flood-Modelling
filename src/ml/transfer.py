"""Transfer learning helper (placeholder).

This module demonstrates the API you'd use to fine-tune a pretrained model on a new local dataset.
In production you'd use torch or tensorflow code here; this file keeps things dependency-free.
"""
from typing import Any, List


def fine_tune(pretrained_model: Any, local_X: List[Any], local_y: List[float], epochs: int = 5):
    """Fine-tune an object that implements fit/predict. Returns the updated model.

    If `pretrained_model` does not expose `fit`, this simply returns it.
    """
    if pretrained_model is None:
        raise ValueError("pretrained_model must not be None")

    fit = getattr(pretrained_model, "fit", None)
    if callable(fit):
        pretrained_model.fit(local_X, local_y, epochs=epochs)
        return pretrained_model
    # nothing to do
    return pretrained_model
