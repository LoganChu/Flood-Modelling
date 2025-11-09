"""A tiny smoke test to validate imports and the simple physics model."""
from flash_flood.src.physics_model import PhysicsModel


def test_physics_model_simulate_length():
    model = PhysicsModel(storage_coef=0.7, runoff_coef=0.4)
    rainfall = [0, 0.2, 1.5, 3.0, 2.0]
    hydro = model.simulate(rainfall, dt_minutes=10)
    assert isinstance(hydro, list)
    assert len(hydro) == len(rainfall)
    # ensure non-negative
    assert all(h >= 0 for h in hydro)
