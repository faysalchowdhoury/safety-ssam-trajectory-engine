"""Time-stepped microsimulation of a signalized four-leg intersection.

Behavioural model (deliberately transparent so results can be reasoned about):

* **Car following** - Intelligent Driver Model (Treiber et al., 2000) with
  per-driver desired speed, time headway, acceleration and comfortable decel.
* **Signal response** - kinematic yellow-onset stop/go decision
  (go if distance < ``v * t_r + v^2 / (2 b_stop)``), stop on red.
* **Permissive left turns** - wait at a yield point 4 m past the stop line and
  accept a gap when every opposing through vehicle needs more than the driver's
  critical gap ``t_c`` (normal, truncated) to reach the conflict point.
* **Intersection-box priority** - a vehicle that has *committed* to the box
  (crossed its stop line, or accepted a gap) has priority over vehicles that
  committed later; those brake for its conflict point if their predicted
  occupancy windows overlap. This is what produces evasive braking and TTC
  conflicts when short gaps are accepted.

Output is written in Vissim vehicle-record layout (``.fzp``), metric units,
speeds in km/h, at the simulation resolution.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import pandas as pd

from ..core.exceptions import SimulationError
from ..core.hcm import geh
from ..core.logging import get_logger
from ..io.fzp import write_fzp
from .layout import MOVEMENTS, OPPOSING, MovementPath, build_movement_paths, find_conflicts
from .scenario import SyntheticScenario

log = get_logger(__name__)

CONFLICT_HALF_ZONE_M: Final[float] = 1.8
YIELD_POINT_OFFSET_M: Final[float] = 4.0
GAP_DECISION_DISTANCE_M: Final[float] = 15.0
REACTION_DISTANCE_M: Final[float] = 80.0
OCCUPANCY_BUFFER_S: Final[float] = 0.3
MAX_DECEL_MPS2: Final[float] = 9.0
STOP_LINE_GAP_M: Final[float] = 1.0
SPEED_LOOKAHEAD_M: Final[float] = 20.0
VEHTYPE_CAR: Final[int] = 100
VEHTYPE_HGV: Final[int] = 200


class _Vehicle:
    __slots__ = (
        "vid", "s", "v", "a", "length", "width", "vtype", "v0f", "T", "amax", "b", "s0", "tc",
        "committed", "commit_key", "decision_cycle", "decision_go", "arrival", "entry", "accepted_gap",
        "crossed_stop",
    )

    def __init__(self, vid: int) -> None:
        self.vid = vid
        self.s = 0.0
        self.v = 0.0
        self.a = 0.0
        self.committed = False
        self.commit_key: tuple[float, int] = (math.inf, vid)
        self.decision_cycle = float("nan")  # yellow-onset key of the last stop/go decision
        self.decision_go = False
        self.accepted_gap = math.nan
        self.crossed_stop = False


@dataclass
class _Movement:
    name: str
    path: MovementPath
    phase: int
    permissive_phase: int | None
    mode: str
    permissive: bool
    vehicles: list[_Vehicle]
    queue: deque[float]
    arrivals: np.ndarray
    next_arrival: int
    conflicts: list[tuple["_Movement", float, float]]
    opposing: tuple["_Movement", float, float] | None
    served: int = 0
    pending: _Vehicle | None = None


@dataclass(frozen=True)
class SimulationOutput:
    """Files and summary produced by one simulation run."""

    fzp_path: Path
    vehicles_path: Path
    movements_path: Path
    n_vehicles: int
    n_records: int
    runtime_s: float


def _idm(v: float, v0: float, gap: float | None, dv: float, veh: _Vehicle, s0: float) -> float:
    free = 1.0 - (v / max(v0, 0.1)) ** 4
    if gap is None:
        return veh.amax * free
    gap = max(gap, 0.05)
    s_star = s0 + max(0.0, v * veh.T + v * dv / (2.0 * math.sqrt(veh.amax * veh.b)))
    return veh.amax * (free - (s_star / gap) ** 2)


def _time_to_cover(d: float, v: float, a: float) -> float:
    """Time to travel ``d`` metres from speed ``v`` accelerating at ``a``."""
    if d <= 0.0:
        return 0.0
    if a <= 1e-6:
        return d / max(v, 0.1)
    return (-v + math.sqrt(v * v + 2.0 * a * d)) / a


class IntersectionSimulator:
    """Simulate one scenario with one random seed."""

    def __init__(self, scenario: SyntheticScenario, seed: int) -> None:
        self.sc = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.paths = build_movement_paths(scenario.geometry, scenario.approach_speed_mps, scenario.left_turn_speed_mps)
        conflicts = find_conflicts(self.paths)
        self.mv: dict[str, _Movement] = {}
        for name in MOVEMENTS:
            ctl = scenario.control[name]
            self.mv[name] = _Movement(
                name=name, path=self.paths[name], phase=ctl.phase, permissive_phase=ctl.permissive_phase,
                mode=ctl.mode, permissive=ctl.mode in ("permissive", "protected_permissive"),
                vehicles=[], queue=deque(), arrivals=self._arrivals(scenario.volumes_vph[name]),
                next_arrival=0, conflicts=[], opposing=None,
            )
        for name, lst in conflicts.items():
            m = self.mv[name]
            m.conflicts = [(self.mv[o], s_self, s_other) for o, s_self, s_other in lst]
            if m.permissive:
                opp = f"{OPPOSING[m.path.approach]}T"
                for o, s_self, s_other in m.conflicts:
                    if o.name == opp:
                        m.opposing = (o, s_self, s_other)
                if m.opposing is None:
                    raise SimulationError("permissive left has no opposing-through conflict", context={"movement": name})
        self._next_vid = 1
        self._finished: list[dict[str, object]] = []

    # ---------------------------------------------------------------- set-up
    def _arrivals(self, vph: float) -> np.ndarray:
        if vph <= 0:
            return np.empty(0)
        mean_h = 3600.0 / vph
        n = int(vph * self.sc.duration_s / 3600.0 * 1.6) + 20
        times = np.cumsum(self.rng.exponential(mean_h, size=n))
        while times[-1] < self.sc.duration_s:
            times = np.concatenate([times, times[-1] + np.cumsum(self.rng.exponential(mean_h, size=n))])
        return times[times < self.sc.duration_s]

    def _new_vehicle(self, arrival: float) -> _Vehicle:
        d = self.sc.drivers
        r = self.rng
        veh = _Vehicle(self._next_vid)
        self._next_vid += 1
        if r.random() < d.car_share:
            veh.vtype, veh.width = VEHTYPE_CAR, d.car_width_m
            veh.length = float(r.uniform(*d.car_length_m))
            veh.amax, veh.b = d.car_accel_mps2, d.car_decel_mps2
        else:
            veh.vtype, veh.width = VEHTYPE_HGV, d.heavy_width_m
            veh.length = float(r.uniform(*d.heavy_length_m))
            veh.amax, veh.b = d.heavy_accel_mps2, d.heavy_decel_mps2
        veh.v0f = float(np.clip(r.normal(d.speed_factor_mean, d.speed_factor_sd), 0.8, 1.25))
        veh.T = float(np.clip(r.normal(d.headway_mean_s, d.headway_sd_s), 0.6, 2.5))
        veh.tc = float(np.clip(r.normal(d.critical_gap_mean_s, d.critical_gap_sd_s), d.critical_gap_min_s, d.critical_gap_max_s))
        veh.s0 = d.standstill_gap_m
        veh.arrival = arrival
        return veh

    # ------------------------------------------------------------- behaviour
    def _state(self, m: _Movement, t: float) -> str:
        """Indication seen by the movement: G (protected), P (permissive green), Y or R."""
        plan = self.sc.signal
        st = plan.phase_state(m.phase, t)
        if m.mode == "permissive":
            return "P" if st == "G" else st
        if m.mode == "protected_permissive" and st == "R":
            assert m.permissive_phase is not None
            ps = plan.phase_state(m.permissive_phase, t)
            return "P" if ps == "G" else ps
        return st

    def _yellow_key(self, m: _Movement, t: float) -> float:
        """Onset time of the most recent change interval that applies to this movement."""
        plan = self.sc.signal
        key = plan.last_yellow_onset(m.phase, t)
        if m.permissive_phase is not None:
            key = max(key, plan.last_yellow_onset(m.permissive_phase, t))
        return key

    def _will_stop(self, veh: _Vehicle, m: _Movement, st: str, t: float) -> bool:
        """Stop/go decision for the current change interval.

        The decision is made once, at the first step of yellow the driver sees, and is
        keyed to that yellow onset. On red, a driver who decided 'go' during the
        preceding yellow keeps going; everyone else stops.
        """
        key = self._yellow_key(m, t)
        if veh.decision_cycle == key:
            return not veh.decision_go
        if st == "R":
            return True
        d = self.sc.drivers
        dist = m.path.stop_s - veh.s
        go = dist < veh.v * d.yellow_reaction_s + veh.v * veh.v / (2.0 * d.yellow_stop_decel_mps2)
        veh.decision_cycle, veh.decision_go = key, go
        return not go

    def _opposing_will_stop(self, o: _Vehicle, om: _Movement, ost: str, t: float) -> bool:
        if o.committed or o.s >= om.path.stop_s or ost not in ("Y", "R"):
            return False
        if o.decision_cycle == self._yellow_key(om, t):
            return not o.decision_go
        return ost == "R"

    def _gap_ok(self, m: _Movement, veh: _Vehicle, states: dict[str, str], t: float) -> bool:
        assert m.opposing is not None
        om, _s_self, s_other = m.opposing
        ost = states[om.name]
        tmin = math.inf
        for o in om.vehicles:
            if o.s - o.length > s_other + CONFLICT_HALF_ZONE_M:
                continue
            if self._opposing_will_stop(o, om, ost, t):
                continue
            d = s_other - CONFLICT_HALF_ZONE_M - o.s
            tmin = min(tmin, _time_to_cover(d, o.v, o.amax))
        if tmin > veh.tc:
            veh.accepted_gap = tmin if math.isfinite(tmin) else math.nan
            return True
        return False

    def _accel(
        self, m: _Movement, k: int, veh: _Vehicle, st: str, t: float, cyc: int,
        states: dict[str, str], occupants: dict[str, list[_Vehicle]],
    ) -> float:
        p = m.path
        v0 = veh.v0f * p.vmax_at(veh.s + SPEED_LOOKAHEAD_M)
        acc = _idm(veh.v, v0, None, 0.0, veh, veh.s0)
        if k > 0:
            lead = m.vehicles[k - 1]
            acc = min(acc, _idm(veh.v, v0, lead.s - lead.length - veh.s, veh.v - lead.v, veh, veh.s0))

        obstacles: list[float] = []
        if not veh.committed and st == "G" and veh.s >= p.stop_s:
            # waiting in the box when the protected arrow comes up: go now, with priority
            veh.committed = True
            veh.commit_key = (t, veh.vid)
        if not veh.committed:
            stopping = False
            if veh.s < p.stop_s and st in ("Y", "R"):
                stopping = self._will_stop(veh, m, st, t)
                if stopping:
                    obstacles.append(p.stop_s)
            if not stopping and m.permissive and st != "G":
                ys = p.stop_s + YIELD_POINT_OFFSET_M
                head = k == 0 or m.vehicles[k - 1].committed
                if head and ys - veh.s <= GAP_DECISION_DISTANCE_M and self._gap_ok(m, veh, states, t):
                    veh.committed = True
                    veh.commit_key = (t, veh.vid)
                else:
                    obstacles.append(ys)

        for om, s_self, s_other in m.conflicts:
            zone_start = s_self - CONFLICT_HALF_ZONE_M
            if veh.s >= zone_start or zone_start - veh.s > REACTION_DISTANCE_M:
                continue
            ti_a = (zone_start - veh.s) / max(veh.v, 0.5)
            to_a = (s_self + CONFLICT_HALF_ZONE_M + veh.length - veh.s) / max(veh.v, 0.5)
            for o in occupants[om.name]:
                if veh.committed and o.commit_key >= veh.commit_key:
                    continue
                if o.s - o.length > s_other + CONFLICT_HALF_ZONE_M:
                    continue
                ti_o = _time_to_cover(s_other - CONFLICT_HALF_ZONE_M - o.s, o.v, o.amax)
                to_o = _time_to_cover(s_other + CONFLICT_HALF_ZONE_M + o.length - o.s, o.v, o.amax)
                if ti_a < to_o + OCCUPANCY_BUFFER_S and ti_o < to_a + OCCUPANCY_BUFFER_S:
                    obstacles.append(zone_start - 0.5)
                    break

        for obs in obstacles:
            gap = obs - veh.s
            if gap > -0.5:
                acc = min(acc, _idm(veh.v, v0, max(gap, 0.05), veh.v, veh, STOP_LINE_GAP_M))
        return max(-MAX_DECEL_MPS2, min(acc, veh.amax))

    # ------------------------------------------------------------------- run
    def run(self, out_dir: str | Path, stem: str | None = None, *, compress: bool = True) -> SimulationOutput:
        """Simulate and write ``<stem>.fzp[.gz]``, ``<stem>_vehicles.csv`` and ``<stem>_movements.csv``."""
        tick = perf_counter()
        sc = self.sc
        dt = sc.dt_s
        n_steps = int(round(sc.duration_s / dt))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{sc.name}_seed{self.seed}"
        cols: dict[str, list[np.ndarray]] = {k: [] for k in (
            "SIMSEC", "NO", "LANE\\LINK\\NO", "LANE\\INDEX", "SPEED", "ACCELERATION",
            "COORDFRONTX", "COORDFRONTY", "COORDREARX", "COORDREARY", "LENGTH", "WIDTH", "VEHTYPE")}
        movements = list(self.mv.values())
        plan = sc.signal

        for i in range(n_steps):
            t = i * dt
            t_next = (i + 1) * dt
            cyc = plan.cycle_index(t)
            states = {m.name: self._state(m, t) for m in movements}
            self._spawn(movements, t)
            occupants = {
                m.name: [o for o in m.vehicles if o.committed and o.s - o.length < m.path.box_exit_s + CONFLICT_HALF_ZONE_M]
                for m in movements
            }
            updates: list[tuple[_Movement, _Vehicle, float]] = []
            for m in movements:
                st = states[m.name]
                for k, veh in enumerate(m.vehicles):
                    updates.append((m, veh, self._accel(m, k, veh, st, t, cyc, states, occupants)))
            for m, veh, acc in updates:
                self._integrate(m, veh, acc, dt, t_next)
            self._exit(movements, t_next)
            if t_next > sc.warmup_s + 1e-9:
                self._record(movements, t_next, cols)

        records = pd.DataFrame({k: (np.concatenate(v) if v else np.empty(0)) for k, v in cols.items()})
        if records.empty:
            raise SimulationError("simulation produced no trajectory records", context={"scenario": sc.name})
        for c in ("NO", "LANE\\LINK\\NO", "LANE\\INDEX", "VEHTYPE"):
            records[c] = records[c].astype(np.int64)
        fzp_path = out / f"{stem}.fzp{'.gz' if compress else ''}"
        write_fzp(records, fzp_path, comment=f"synthetic scenario={sc.name} seed={self.seed}")

        vehicles = pd.DataFrame(self._finished)
        vehicles_path = out / f"{stem}_vehicles.csv"
        vehicles.to_csv(vehicles_path, index=False)
        movements_df = self._movement_table()
        movements_path = out / f"{stem}_movements.csv"
        movements_df.to_csv(movements_path, index=False)

        runtime = perf_counter() - tick
        log.info("sim.finished", extra={"scenario": sc.name, "seed": self.seed, "records": len(records),
                                        "vehicles": len(vehicles), "runtime_s": round(runtime, 2)})
        return SimulationOutput(fzp_path, vehicles_path, movements_path, len(vehicles), len(records), runtime)

    def _spawn(self, movements: list[_Movement], t: float) -> None:
        for m in movements:
            while m.next_arrival < m.arrivals.size and m.arrivals[m.next_arrival] <= t:
                m.queue.append(float(m.arrivals[m.next_arrival]))
                m.next_arrival += 1
            if not m.queue:
                continue
            if m.pending is None:  # draw attributes once, when the vehicle reaches the entry
                m.pending = self._new_vehicle(m.queue[0])
            veh = m.pending
            v0 = veh.v0f * m.path.vmax_at(0.0)
            if m.vehicles:
                last = m.vehicles[-1]
                space = last.s - last.length
                v_entry = min(v0, last.v)
                if space < veh.s0 + v_entry * veh.T + 0.5:
                    continue  # entry blocked: vehicle waits in the virtual entry queue
            else:
                v_entry = v0
            m.queue.popleft()
            m.pending = None
            veh.v = v_entry
            veh.entry = t
            m.vehicles.append(veh)

    def _integrate(self, m: _Movement, veh: _Vehicle, acc: float, dt: float, t_next: float) -> None:
        v_new = veh.v + acc * dt
        if v_new < 0.0:
            ds = -veh.v * veh.v / (2.0 * acc) if acc < 0 else 0.0
            v_new = 0.0
        else:
            ds = veh.v * dt + 0.5 * acc * dt * dt
        old_s = veh.s
        veh.s += max(ds, 0.0)
        veh.v, veh.a = v_new, acc
        if not veh.crossed_stop and old_s < m.path.stop_s <= veh.s:
            veh.crossed_stop = True
            if not veh.committed:
                veh.committed = True
                veh.commit_key = (t_next, veh.vid)
            if self.sc.warmup_s < t_next <= self.sc.duration_s:
                m.served += 1

    def _exit(self, movements: list[_Movement], t_next: float) -> None:
        for m in movements:
            end = m.path.line.length
            while m.vehicles and m.vehicles[0].s - m.vehicles[0].length > end:
                veh = m.vehicles.pop(0)
                ff = m.path.free_flow_time(veh.v0f, extra_m=veh.length)
                self._finished.append(
                    {
                        "vid": veh.vid, "movement": m.name, "vtype": veh.vtype, "arrival_s": round(veh.arrival, 3),
                        "entry_s": round(veh.entry, 3), "exit_s": round(t_next, 3),
                        "travel_time_s": round(t_next - veh.arrival, 3), "free_flow_time_s": round(ff, 3),
                        "delay_s": round(t_next - veh.arrival - ff, 3), "critical_gap_s": round(veh.tc, 3),
                        "accepted_gap_s": round(veh.accepted_gap, 3) if math.isfinite(veh.accepted_gap) else None,
                    }
                )

    def _record(self, movements: list[_Movement], t: float, cols: dict[str, list[np.ndarray]]) -> None:
        for m in movements:
            if not m.vehicles:
                continue
            vs = m.vehicles
            s = np.fromiter((v.s for v in vs), float, len(vs))
            L = np.fromiter((v.length for v in vs), float, len(vs))
            front = m.path.line.point(s)
            rear = m.path.line.point(s - L)
            n = len(vs)
            cols["SIMSEC"].append(np.full(n, round(t, 3)))
            cols["NO"].append(np.fromiter((v.vid for v in vs), np.int64, n))
            cols["LANE\\LINK\\NO"].append(np.full(n, m.path.link_no, dtype=np.int64))
            cols["LANE\\INDEX"].append(np.ones(n, dtype=np.int64))
            cols["SPEED"].append(np.fromiter((v.v for v in vs), float, n) * 3.6)
            cols["ACCELERATION"].append(np.fromiter((v.a for v in vs), float, n))
            cols["COORDFRONTX"].append(front[:, 0])
            cols["COORDFRONTY"].append(front[:, 1])
            cols["COORDREARX"].append(rear[:, 0])
            cols["COORDREARY"].append(rear[:, 1])
            cols["LENGTH"].append(L)
            cols["WIDTH"].append(np.fromiter((v.width for v in vs), float, n))
            cols["VEHTYPE"].append(np.fromiter((v.vtype for v in vs), np.int64, n))

    def _movement_table(self) -> pd.DataFrame:
        hours = self.sc.analysis_s / 3600.0
        rows = []
        for name, m in self.mv.items():
            demand = self.sc.volumes_vph[name]
            served = m.served / hours
            rows.append(
                {
                    "movement": name, "phase": m.phase, "mode": m.mode,
                    "demand_vph": demand, "served_vph": round(served, 1), "geh": round(float(geh(served, demand)), 3),
                }
            )
        return pd.DataFrame(rows)
