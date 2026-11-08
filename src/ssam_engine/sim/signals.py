"""NEMA TS2 dual-ring, ring-barrier fixed-time controller.

Standard NEMA numbering (eight-phase, right-hand traffic) used throughout::

    Ring 1:  phi1 WBL | phi2 EBT  ||  phi3 NBL | phi4 SBT
    Ring 2:  phi5 EBL | phi6 WBT  ||  phi7 SBL | phi8 NBT
                      barrier 1          barrier 2

Phases on the same side of a barrier may time concurrently across rings; the
two rings must reach each barrier at the same time (equal barrier sums).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from pydantic import Field, model_validator

from ..core.config import StrictModel

SignalState = Literal["G", "Y", "R"]
_BARRIER_PHASES: tuple[frozenset[int], frozenset[int]] = (frozenset({1, 2, 5, 6}), frozenset({3, 4, 7, 8}))


class PhaseTiming(StrictModel):
    """Phase split components (seconds)."""

    green_s: float = Field(gt=0.0, le=180.0)
    yellow_s: float = Field(ge=3.0, le=6.0, description="MUTCD yellow change interval: 3-6 s")
    red_clearance_s: float = Field(ge=0.0, le=6.0)

    @property
    def split_s(self) -> float:
        return self.green_s + self.yellow_s + self.red_clearance_s


class MovementControl(StrictModel):
    """How a movement is served.

    * ``protected``: runs only on ``phase`` (green arrow / ball).
    * ``permissive``: runs on ``phase`` but must yield (left turns only).
    * ``protected_permissive``: protected on ``phase`` (e.g. phi5 green arrow), then
      permissive on ``permissive_phase`` (e.g. phi2, flashing yellow arrow per MUTCD).
    """

    phase: int = Field(ge=1, le=8)
    mode: Literal["protected", "permissive", "protected_permissive"] = "protected"
    permissive_phase: int | None = Field(None, ge=1, le=8)

    @model_validator(mode="after")
    def _check(self) -> "MovementControl":
        if (self.mode == "protected_permissive") != (self.permissive_phase is not None):
            raise ValueError("permissive_phase is required for (and only for) protected_permissive mode")
        return self


@dataclass(frozen=True)
class _Interval:
    phase: int
    start: float
    green_end: float
    yellow_end: float
    end: float


class RingBarrierPlan(StrictModel):
    """Fixed-time dual-ring plan.

    Attributes:
        ring1: phase sequence per barrier for ring 1, e.g. ``[[1, 2], [3, 4]]``.
        ring2: phase sequence per barrier for ring 2, e.g. ``[[5, 6], [7, 8]]``.
        timings: split components per phase number.
        offset_s: cycle offset relative to simulation time zero.
    """

    ring1: list[list[int]]
    ring2: list[list[int]]
    timings: dict[int, PhaseTiming]
    offset_s: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> "RingBarrierPlan":
        if len(self.ring1) != 2 or len(self.ring2) != 2:
            raise ValueError("each ring must list exactly two barrier groups")
        for ring, allowed in ((self.ring1, {1, 2, 3, 4}), (self.ring2, {5, 6, 7, 8})):
            for b, group in enumerate(ring):
                if not group:
                    raise ValueError("barrier groups must contain at least one phase")
                for ph in group:
                    if ph not in allowed:
                        raise ValueError(f"phase {ph} is not valid in this ring")
                    if ph not in _BARRIER_PHASES[b]:
                        raise ValueError(f"phase {ph} cannot run in barrier {b + 1}")
                    if ph not in self.timings:
                        raise ValueError(f"phase {ph} has no timing")
        for b in range(2):
            s1 = sum(self.timings[p].split_s for p in self.ring1[b])
            s2 = sum(self.timings[p].split_s for p in self.ring2[b])
            if abs(s1 - s2) > 1e-6:
                raise ValueError(f"barrier {b + 1} ring sums differ ({s1:.2f} s vs {s2:.2f} s)")
        return self

    @cached_property
    def cycle_s(self) -> float:
        return sum(sum(self.timings[p].split_s for p in self.ring1[b]) for b in range(2))

    @cached_property
    def _schedule(self) -> dict[int, _Interval]:
        out: dict[int, _Interval] = {}
        barrier_start = 0.0
        for b in range(2):
            barrier_len = sum(self.timings[p].split_s for p in self.ring1[b])
            for ring in (self.ring1, self.ring2):
                t0 = barrier_start
                for ph in ring[b]:
                    tm = self.timings[ph]
                    out[ph] = _Interval(ph, t0, t0 + tm.green_s, t0 + tm.green_s + tm.yellow_s, t0 + tm.split_s)
                    t0 += tm.split_s
            barrier_start += barrier_len
        return out

    def phase_state(self, phase: int, t: float) -> SignalState:
        """Signal indication of ``phase`` at simulation time ``t``."""
        iv = self._schedule.get(phase)
        if iv is None:
            return "R"
        tc = (t - self.offset_s) % self.cycle_s
        if iv.start <= tc < iv.green_end:
            return "G"
        if iv.green_end <= tc < iv.yellow_end:
            return "Y"
        return "R"

    def cycle_index(self, t: float) -> int:
        return int((t - self.offset_s) // self.cycle_s)

    def last_yellow_onset(self, phase: int, t: float) -> float:
        """Absolute time of the most recent yellow onset of ``phase`` at or before ``t``.

        Used to key yellow-interval stop/go decisions so that a decision made in
        one change interval is never reused in a later one.
        """
        iv = self._schedule.get(phase)
        if iv is None:
            return float("-inf")
        k = self.cycle_index(t)
        onset = self.offset_s + k * self.cycle_s + iv.green_end
        if onset > t + 1e-9:
            onset -= self.cycle_s
        return round(onset, 6)

    def effective_green_ratio(self, phase: int, lost_time_s: float = 4.0) -> float:
        """HCM-style g/C with ``g = G + Y + AR - lost time``."""
        tm = self.timings.get(phase)
        if tm is None:
            return 0.0
        return max(tm.split_s - lost_time_s, 0.0) / self.cycle_s
