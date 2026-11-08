"""Typed engine configuration (Pydantic v2).

Defaults reproduce the FHWA SSAM default thresholds (FHWA-HRT-08-051):

* maximum TTC = 1.5 s
* maximum PET = 5.0 s
* rear-end angle < 30 degrees, crossing angle > 85 degrees, otherwise lane-change
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import ConfigurationError

LengthUnit = Literal["m", "ft"]
SpeedUnit = Literal["kmh", "mph", "mps", "fps"]
AccelUnit = Literal["mps2", "fps2"]
SpatialBackend = Literal["kdtree", "rtree", "grid", "brute"]


class StrictModel(BaseModel):
    """Immutable model that rejects unknown keys (typos in YAML fail loudly)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class UnitsConfig(StrictModel):
    """Units used in the *input* trajectory file."""

    length: LengthUnit = "m"
    speed: SpeedUnit = "kmh"
    acceleration: AccelUnit = "mps2"


class ConflictThresholds(StrictModel):
    """Conflict identification and classification thresholds."""

    max_ttc_s: float = Field(1.5, gt=0.0, le=10.0, description="SSAM maximum time-to-collision")
    max_pet_s: float = Field(5.0, gt=0.0, le=30.0, description="SSAM maximum post-encroachment time")
    rear_end_angle_deg: float = Field(30.0, ge=0.0, lt=90.0)
    crossing_angle_deg: float = Field(85.0, gt=0.0, le=180.0)
    severe_ttc_s: float = Field(0.5, gt=0.0, description="TTC at or below which a conflict is 'severe'")
    min_max_speed_mps: float = Field(
        0.0,
        ge=0.0,
        description="Drop events whose MaxS is below this value (practitioner filter; 0 disables)",
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> "ConflictThresholds":
        if self.crossing_angle_deg <= self.rear_end_angle_deg:
            raise ValueError("crossing_angle_deg must be greater than rear_end_angle_deg")
        if self.severe_ttc_s > self.max_ttc_s:
            raise ValueError("severe_ttc_s must not exceed max_ttc_s")
        return self


class DetectorConfig(StrictModel):
    """Numerical settings of the detector."""

    spatial_backend: SpatialBackend = "kdtree"
    ttc_resolution_s: float = Field(0.05, gt=0.0, le=0.5)
    ttc_refine_iterations: int = Field(5, ge=0, le=20)
    event_gap_tolerance_s: float = Field(0.2, ge=0.0)
    default_vehicle_width_m: float = Field(1.8, gt=0.0)
    mass_model: Literal["equal", "footprint"] = "footprint"
    stationary_speed_mps: float = Field(0.1, ge=0.0)
    pair_chunk_size: int = Field(20_000, ge=100)
    encroachment_buffer_m: float = Field(
        1.0,
        ge=0.0,
        le=5.0,
        description="Footprint inflation used only to verify that both vehicles really shared space",
    )
    pet_search_padding_s: float = Field(1.0, ge=0.0)
    path_lateral_tolerance_m: float = Field(
        1.0,
        gt=0.0,
        le=5.0,
        description="Max lateral deviation from the projected ray for a vehicle to count as 'on course'",
    )
    require_path_verification: bool = True


class EngineConfig(StrictModel):
    """Top-level engine configuration."""

    units: UnitsConfig = Field(default_factory=UnitsConfig)
    thresholds: ConflictThresholds = Field(default_factory=ConflictThresholds)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "EngineConfig":
        """Load from a YAML or JSON file."""
        return load_model_file(path, cls)


ModelT = TypeVar("ModelT", bound=BaseModel)


def read_mapping_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML/JSON file into a dict, raising :class:`ConfigurationError` on failure."""
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError("configuration file not found", context={"path": str(p)})
    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot parse configuration: {exc}", context={"path": str(p)}) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError("configuration root must be a mapping", context={"path": str(p)})
    return data


def load_model_file(path: str | Path, model: type[ModelT]) -> ModelT:
    """Parse and validate a configuration file into ``model``."""
    data = read_mapping_file(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            f"invalid {model.__name__}: {exc.error_count()} error(s)\n{exc}",
            context={"path": str(path)},
        ) from exc
