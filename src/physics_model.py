"""Simple physics-based model placeholder.

This is NOT a real hydrologic solver. It provides an interface and a tiny linear-reservoir
example hydrograph generator so the rest of the scaffold can run.
"""
from typing import List


class PhysicsModel:
    """A minimal physics model interface.

    Methods
    - simulate(rainfall_series, dt_minutes): returns list of discharge values (same length as rainfall)
    """

    def __init__(self, storage_coef: float = 0.8, runoff_coef: float = 0.3):
        # storage_coef: how much previous discharge persists [0..1]
        # runoff_coef: fraction of rainfall converted to runoff
        self.storage_coef = float(storage_coef)
        self.runoff_coef = float(runoff_coef)

    def simulate(self, rainfall_series: List[float], dt_minutes: int = 10) -> List[float]:
        """Simulate a hydrograph from rainfall using a linear reservoir-type update.

        rainfall_series: list of rainfall depths (mm) per timestep
        dt_minutes: time step length (informational)
        returns: discharge_like series (arbitrary units)
        """
        if dt_minutes <= 0:
            raise ValueError("dt_minutes must be positive")

        q = 0.0
        hydro = []
        for p in rainfall_series:
            # convert rainfall input into an immediate increment of 'runoff'
            inflow = self.runoff_coef * float(p)
            q = self.storage_coef * q + inflow
            hydro.append(q)
        return hydro


def example_usage():
    model = PhysicsModel(storage_coef=0.7, runoff_coef=0.4)
    rainfall = [0, 0.2, 1.5, 3.0, 2.0, 0.5, 0, 0]
    hydro = model.simulate(rainfall, dt_minutes=10)
    return hydro
