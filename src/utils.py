"""Utility helpers for the sample project."""
from typing import Any, Dict


def load_config(path: str = None) -> Dict[str, Any]:
    """Load a YAML config if PyYAML is available; otherwise return a small default.

    This function is intentionally robust: it won't fail hard if PyYAML isn't installed.
    """
    default = {
        "timestep_minutes": 10,
        "physics": {"storage_coef": 0.8, "runoff_coef": 0.3},
        "ml": {"transfer_epochs": 5, "meta_iters": 2},
    }
    if path is None:
        return default
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cfg if cfg else default
    except Exception:
        # fallback
        return default
