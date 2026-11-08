"""Domain models shared by the detector, pipeline and API."""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class ConflictType(str, Enum):
    """SSAM conflict classes (by heading angle at minimum TTC)."""

    REAR_END = "rear_end"
    CROSSING = "crossing"
    LANE_CHANGE = "lane_change"


EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "vid_a",
    "vid_b",
    "first_vid",
    "second_vid",
    "t_start",
    "t_end",
    "t_min_ttc",
    "ttc",
    "pet",
    "x",
    "y",
    "angle_deg",
    "conflict_type",
    "max_s",
    "delta_s",
    "dr",
    "max_d",
    "max_delta_v",
    "link_a",
    "link_b",
    "severe",
    "collision",
    "path_verified",
)


class DetectionStats(BaseModel):
    """Counters and stage timings produced by one detector run."""

    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_vehicles: int
    n_time_steps: int
    time_step_s: float
    n_candidate_pairs: int
    n_closest_approach_pass: int
    n_ttc_hits: int
    n_raw_events: int
    n_speed_filtered: int
    n_unverified: int
    n_pet_filtered: int
    n_events: int
    n_collisions: int
    stage_seconds: dict[str, float] = Field(default_factory=dict)

    @property
    def runtime_s(self) -> float:
        """Total wall-clock time of all stages."""
        return float(sum(self.stage_seconds.values()))
