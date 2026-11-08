"""Scenario definition for the synthetic intersection simulator."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from ..core.config import StrictModel, load_model_file
from .layout import MOVEMENTS, IntersectionGeometry
from .signals import MovementControl, RingBarrierPlan


class DriverPopulation(StrictModel):
    """Driver/vehicle heterogeneity (IDM car-following + gap acceptance)."""

    car_share: float = Field(0.92, ge=0.0, le=1.0)
    car_length_m: tuple[float, float] = (4.2, 4.9)
    car_width_m: float = 1.8
    heavy_length_m: tuple[float, float] = (8.0, 10.5)
    heavy_width_m: float = 2.4
    speed_factor_mean: float = 1.0
    speed_factor_sd: float = Field(0.07, ge=0.0)
    headway_mean_s: float = Field(1.1, gt=0.3)
    headway_sd_s: float = Field(0.2, ge=0.0)
    critical_gap_mean_s: float = Field(4.5, gt=0.5)
    critical_gap_sd_s: float = Field(1.0, ge=0.0)
    critical_gap_min_s: float = Field(2.2, gt=0.0)
    critical_gap_max_s: float = Field(8.0, gt=0.0)
    car_accel_mps2: float = 2.2
    car_decel_mps2: float = 2.8
    heavy_accel_mps2: float = 1.0
    heavy_decel_mps2: float = 2.0
    standstill_gap_m: float = Field(2.0, gt=0.0)
    yellow_reaction_s: float = Field(1.0, ge=0.0)
    yellow_stop_decel_mps2: float = Field(3.0, gt=0.0, description="ITE kinematic stop decision decel")


class SyntheticScenario(StrictModel):
    """One analysis scenario (geometry, demand, signal timing, drivers)."""

    name: str
    description: str = ""
    duration_s: float = Field(1200.0, gt=0.0)
    warmup_s: float = Field(300.0, ge=0.0)
    dt_s: float = Field(0.1, gt=0.0, le=0.5)
    approach_speed_mps: float = Field(15.6, gt=0.0)
    left_turn_speed_mps: float = Field(8.5, gt=0.0)
    geometry: IntersectionGeometry = Field(default_factory=IntersectionGeometry)
    volumes_vph: dict[str, float]
    signal: RingBarrierPlan
    control: dict[str, MovementControl]
    drivers: DriverPopulation = Field(default_factory=DriverPopulation)

    @model_validator(mode="after")
    def _check(self) -> "SyntheticScenario":
        expected = set(MOVEMENTS)
        if set(self.volumes_vph) != expected:
            raise ValueError(f"volumes_vph must define exactly {sorted(expected)}")
        if set(self.control) != expected:
            raise ValueError(f"control must define exactly {sorted(expected)}")
        if any(v < 0 for v in self.volumes_vph.values()):
            raise ValueError("volumes must be non-negative")
        if self.warmup_s >= self.duration_s:
            raise ValueError("warmup_s must be shorter than duration_s")
        for mv, ctl in self.control.items():
            if ctl.mode != "protected" and not mv.endswith("L"):
                raise ValueError(f"{mv}: only left turns may be permissive")
            for ph in (ctl.phase, ctl.permissive_phase):
                if ph is not None and ph not in self.signal.timings:
                    raise ValueError(f"{mv}: phase {ph} is not timed in the signal plan")
        return self

    @property
    def analysis_s(self) -> float:
        return self.duration_s - self.warmup_s

    @classmethod
    def from_file(cls, path: str | Path) -> "SyntheticScenario":
        return load_model_file(path, cls)
