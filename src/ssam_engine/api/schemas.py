"""Request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..core.models import ConflictType


class ConflictRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: int
    vid_a: int
    vid_b: int
    conflict_type: ConflictType
    t_min_ttc: float = Field(description="simulation second of minimum TTC")
    ttc: float = Field(description="minimum time-to-collision (s)")
    pet: float | None = Field(description="post-encroachment time (s); null if the second vehicle never encroached")
    x: float
    y: float
    angle_deg: float
    max_s: float
    delta_s: float
    dr: float
    max_d: float
    max_delta_v: float
    severe: bool
    collision: bool


class RunSummary(BaseModel):
    run_id: str
    source_name: str
    created_at: datetime
    rows: int
    vehicles: int
    events: int
    by_type: dict[str, int]
    severe: int
    collisions: int
    raw_events: int
    unverified_removed: int
    detector_runtime_s: float
    index_backend: str


class ConflictQueryResponse(BaseModel):
    run_id: str
    count: int
    truncated: bool
    conflicts: list[ConflictRecord]


class HeatmapCell(BaseModel):
    x: float
    y: float
    count: int
    severity: float


class HeatmapResponse(BaseModel):
    run_id: str
    cell_m: float
    cells: list[HeatmapCell]


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_backend: str
    runs_loaded: int
