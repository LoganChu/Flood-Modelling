"""Simple orchestrator demonstrating how pieces fit together.

Run as a script to execute a tiny end-to-end flow using placeholders.
"""
from typing import List

from .preprocessing import resample_rainfall, hydrologic_bias_correction
from .physics_model import PhysicsModel
from .ml.models import load_pretrained
from .ml.transfer import fine_tune
from .ml.meta_learning import meta_train
from .utils import load_config


def small_demo():
    cfg = load_config()
    # fake rainfall timeseries: (timestamp, mm)
    raw_ts = [(f"t{i}", v) for i, v in enumerate([0, 0.1, 0.5, 2.0, 3.0, 1.5, 0.2, 0])]
    # resample placeholder
    res_ts = resample_rainfall(raw_ts, target_minutes=1)
    rain_values = [v for (_, v) in res_ts]

    # bias-correct radar (placeholder)
    corrected = hydrologic_bias_correction(rain_values, factor=1.05)

    # physics model simulate
    phys = PhysicsModel(storage_coef=cfg["physics"]["storage_coef"], runoff_coef=cfg["physics"]["runoff_coef"])
    hydro = phys.simulate(corrected, dt_minutes=cfg["timestep_minutes"])

    # ML: load a pretrained model and fine-tune on small local data (placeholder)
    pretrained = load_pretrained()
    # pretend local supervised data: X can be timestamps; y is hydro observed
    local_X = list(range(len(hydro)))
    local_y = hydro
    fine_tuned = fine_tune(pretrained, local_X, local_y, epochs=cfg["ml"]["transfer_epochs"])

    # meta-learning placeholder across toy tasks
    tasks = [ (local_X, local_y), (local_X, [h*0.8 for h in local_y]) ]
    meta = meta_train(tasks, meta_iters=cfg["ml"]["meta_iters"])

    # Nicely formatted output
    print("Demo results:")
    print("  rainfall:", corrected)
    print("  hydro:", hydro)
    print("Fine-tuned model predict sample:", fine_tuned.predict(local_X[:3]))
    print("Meta-model base value:", getattr(meta, "_base", None))
    return {"rain": corrected, "hydro": hydro}


if __name__ == "__main__":
    small_demo()
