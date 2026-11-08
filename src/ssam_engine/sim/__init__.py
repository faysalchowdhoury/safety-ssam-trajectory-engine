"""Synthetic signalized-intersection microsimulator.

Produces Vissim-format vehicle records so that the full safety pipeline can be
exercised and benchmarked without a Vissim licence. It is a research test bed,
not a replacement for a calibrated Vissim model.
"""

from .engine import IntersectionSimulator, SimulationOutput
from .scenario import SyntheticScenario
from .signals import MovementControl, PhaseTiming, RingBarrierPlan

__all__ = [
    "IntersectionSimulator",
    "MovementControl",
    "PhaseTiming",
    "RingBarrierPlan",
    "SimulationOutput",
    "SyntheticScenario",
]
