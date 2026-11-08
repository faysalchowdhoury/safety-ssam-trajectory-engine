"""SSAM-equivalent conflict detector.

Pipeline (all heavy stages vectorised with NumPy):

1. **Frame split** - rows are sorted by time; each simulation step is a frame.
2. **Candidate pairs** - a spatial index returns pairs within reach
   ``r_i + r_j`` (``r = v * maxTTC + half_diagonal``).
3. **Closest-approach filter** - exact bounding-circle test over the horizon.
4. **TTC** - sampled oriented-box SAT on linear projections + bisection.
5. **Event grouping** - consecutive frames of the same pair with TTC <= maxTTC
   form one conflict event (SSAM semantics).
6. **Event metrics** - minimum TTC, PET, MaxS, DeltaS, DR, MaxD, MaxDeltaV,
   conflict angle and type.
7. **Path verification** - linear projection over-predicts conflicts for
   turning vehicles (a known SSAM artefact). An event is *verified* if either
   (a) the two vehicles' actual footprints (inflated by ``encroachment_buffer_m``)
   occupy common space within maxPET of each other, or (b) both vehicles stay
   within ``path_lateral_tolerance_m`` of their projected rays for the TTC
   horizon - i.e. the projection was physically right and the collision was
   avoided only by braking (the classic stop-short near miss, which has no PET).
   Events failing both tests are projection artefacts and are removed.

PET definition used here: the minimum ``|t_a - t_b|`` over all sample pairs at
which vehicle ``a``'s footprint at ``t_a`` overlaps vehicle ``b``'s footprint at
``t_b``. This equals the time between the first vehicle leaving and the second
vehicle entering the shared space; it is NaN when the second vehicle never
encroached (for example, it stopped short of the leader).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..spatial.pairs import make_pair_finder
from .config import EngineConfig
from .geometry import FloatArray, dot2, heading_angle_deg, heading_vectors, obb_overlap
from .logging import get_logger
from .models import EVENT_COLUMNS, ConflictType, DetectionStats
from .schema import infer_time_step, validate_trajectories
from .ttc import closest_approach_mask, compute_ttc

log = get_logger(__name__)

_MAX_WINDOW_SAMPLES = 1_200


class _Encroachment(NamedTuple):
    dt: float
    a_first: bool
    point: tuple[float, float]


@dataclass(frozen=True)
class DetectionResult:
    """Output of :meth:`ConflictDetector.detect`."""

    events: pd.DataFrame
    stats: DetectionStats


def empty_events() -> pd.DataFrame:
    """An empty, correctly-typed event table."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in EVENT_COLUMNS}).astype(
        {
            "event_id": "int64",
            "vid_a": "int64",
            "vid_b": "int64",
            "first_vid": "int64",
            "second_vid": "int64",
            "link_a": "int64",
            "link_b": "int64",
            "conflict_type": "object",
            "severe": "bool",
            "collision": "bool",
            "path_verified": "bool",
        }
    )


class ConflictDetector:
    """Detect and characterise traffic conflicts from vehicle trajectories."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._finder = make_pair_finder(self.config.detector.spatial_backend)

    # ------------------------------------------------------------------ public
    def detect(self, trajectories: pd.DataFrame) -> DetectionResult:
        """Run the full detection pipeline on a canonical trajectory table."""
        th = self.config.thresholds
        dc = self.config.detector
        timings: dict[str, float] = {}
        tick = perf_counter()

        df = validate_trajectories(trajectories, default_width=dc.default_vehicle_width_m)
        n = len(df)
        timings["validate"] = perf_counter() - tick
        if n == 0:
            return DetectionResult(empty_events(), self._stats(df, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, timings))

        t = df["t"].to_numpy(np.float64)
        vid = df["vid"].to_numpy(np.int64)
        xy = df[["x", "y"]].to_numpy(np.float64)
        u = heading_vectors(df["heading"].to_numpy(np.float64))
        spd = df["speed"].to_numpy(np.float64)
        acc = df["accel"].to_numpy(np.float64)
        length = df["length"].to_numpy(np.float64)
        width = df["width"].to_numpy(np.float64)
        link = df["link"].to_numpy(np.int64)
        vel = u * spd[:, None]
        half = np.column_stack([0.5 * length, 0.5 * width])
        hdiag = np.hypot(half[:, 0], half[:, 1])
        dt = infer_time_step(t)
        step = np.rint(t / dt).astype(np.int64)
        horizon = th.max_ttc_s

        # --- stage 2: candidate pairs per frame
        tick = perf_counter()
        radius = spd * horizon + hdiag
        change = np.flatnonzero(np.diff(step)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        parts_i: list[np.ndarray] = []
        parts_j: list[np.ndarray] = []
        for s0, e0 in zip(starts.tolist(), ends.tolist()):
            if e0 - s0 < 2:
                continue
            pairs = self._finder.candidate_pairs(xy[s0:e0], radius[s0:e0])
            if pairs.size:
                parts_i.append(pairs[:, 0] + s0)
                parts_j.append(pairs[:, 1] + s0)
        I = np.concatenate(parts_i) if parts_i else np.empty(0, np.int64)
        J = np.concatenate(parts_j) if parts_j else np.empty(0, np.int64)
        n_candidates = int(I.size)
        timings["candidate_pairs"] = perf_counter() - tick

        # --- stage 3: stationary + closest approach filter
        tick = perf_counter()
        moving = (spd[I] > dc.stationary_speed_mps) | (spd[J] > dc.stationary_speed_mps)
        I, J = I[moving], J[moving]
        mask = closest_approach_mask(xy[I], vel[I], xy[J], vel[J], hdiag[I] + hdiag[J], horizon)
        I, J = I[mask], J[mask]
        n_ca = int(I.size)
        timings["closest_approach"] = perf_counter() - tick

        # --- stage 4: TTC
        tick = perf_counter()
        ttc = np.full(I.size, np.inf)
        chunk = dc.pair_chunk_size
        for c0 in range(0, I.size, chunk):
            ii, jj = I[c0 : c0 + chunk], J[c0 : c0 + chunk]
            ttc[c0 : c0 + chunk] = compute_ttc(
                xy[ii], vel[ii], u[ii], half[ii], xy[jj], vel[jj], u[jj], half[jj],
                horizon=horizon, resolution=dc.ttc_resolution_s, refine_iterations=dc.ttc_refine_iterations,
            )
        hit = ttc <= horizon + 1e-9
        I, J, ttc = I[hit], J[hit], ttc[hit]
        n_hits = int(I.size)
        timings["ttc"] = perf_counter() - tick

        # --- stage 5: event grouping
        tick = perf_counter()
        swap = vid[I] > vid[J]
        ra = np.where(swap, J, I)
        rb = np.where(swap, I, J)
        va, vb, st = vid[ra], vid[rb], step[ra]
        order = np.lexsort((st, vb, va))
        ra, rb, ttc, va, vb, st = ra[order], rb[order], ttc[order], va[order], vb[order], st[order]
        allowed = int(round(dc.event_gap_tolerance_s / dt))
        if ra.size:
            new = np.concatenate(
                [[True], (va[1:] != va[:-1]) | (vb[1:] != vb[:-1]) | (st[1:] - st[:-1] > 1 + allowed)]
            )
            eid = np.cumsum(new) - 1
        else:
            eid = np.empty(0, np.int64)
        n_raw = int(eid.max() + 1) if eid.size else 0
        timings["grouping"] = perf_counter() - tick

        # --- stage 6/7: per-event metrics + verification
        tick = perf_counter()
        events = self._event_metrics(
            eid, ra, rb, ttc, t, vid, xy, u, vel, spd, acc, half, link, length, width, dt
        )
        timings["event_metrics"] = perf_counter() - tick

        n_speed = n_unver = n_pet = 0
        if not events.empty:
            speed_drop = events["max_s"] < th.min_max_speed_mps
            n_speed = int(speed_drop.sum())
            events = events[~speed_drop]
            if dc.require_path_verification:
                unver = ~events["path_verified"]
                n_unver = int(unver.sum())
                events = events[~unver]
            pet_drop = events["pet"].notna() & (events["pet"] > th.max_pet_s)
            n_pet = int(pet_drop.sum())
            events = events[~pet_drop].reset_index(drop=True)
            events["event_id"] = np.arange(len(events), dtype=np.int64)

        n_coll = int(events["collision"].sum()) if not events.empty else 0
        stats = self._stats(df, dt, n_candidates, n_ca, n_hits, n_raw, n_speed, n_unver, n_pet, n_coll, timings,
                            n_events=len(events), n_steps=int(starts.size))
        log.info(
            "detector.finished",
            extra={
                "rows": n, "events": len(events), "raw_events": n_raw, "unverified": n_unver,
                "runtime_s": round(stats.runtime_s, 3), "backend": dc.spatial_backend,
            },
        )
        return DetectionResult(events if not events.empty else empty_events(), stats)

    # ----------------------------------------------------------------- helpers
    def _stats(
        self, df: pd.DataFrame, dt: float, n_candidates: int, n_ca: int, n_hits: int, n_raw: int,
        n_speed: int, n_unver: int, n_pet: int, n_coll: int, timings: dict[str, float],
        *, n_events: int = 0, n_steps: int = 0,
    ) -> DetectionStats:
        return DetectionStats(
            n_rows=len(df), n_vehicles=int(df["vid"].nunique()) if len(df) else 0, n_time_steps=n_steps,
            time_step_s=dt, n_candidate_pairs=n_candidates, n_closest_approach_pass=n_ca, n_ttc_hits=n_hits,
            n_raw_events=n_raw, n_speed_filtered=n_speed, n_unverified=n_unver, n_pet_filtered=n_pet,
            n_events=n_events, n_collisions=n_coll, stage_seconds={k: round(v, 6) for k, v in timings.items()},
        )

    def _classify(self, angle: float) -> str:
        th = self.config.thresholds
        if angle < th.rear_end_angle_deg:
            return ConflictType.REAR_END.value
        if angle > th.crossing_angle_deg:
            return ConflictType.CROSSING.value
        return ConflictType.LANE_CHANGE.value

    def _event_metrics(
        self,
        eid: np.ndarray, ra: np.ndarray, rb: np.ndarray, ttc: np.ndarray, t: FloatArray, vid: np.ndarray,
        xy: FloatArray, u: FloatArray, vel: FloatArray, spd: FloatArray, acc: FloatArray, half: FloatArray,
        link: np.ndarray, length: FloatArray, width: FloatArray, dt: float,
    ) -> pd.DataFrame:
        th = self.config.thresholds
        dc = self.config.detector
        if eid.size == 0:
            return empty_events()

        bounds = np.flatnonzero(np.concatenate([[True], eid[1:] != eid[:-1], [True]]))
        g_start, g_end = bounds[:-1], bounds[1:]
        tracks: dict[int, np.ndarray] = pd.Series(np.arange(vid.size)).groupby(vid).indices  # type: ignore[assignment]
        track_t = {k: t[v] for k, v in tracks.items()}

        rows: list[dict[str, object]] = []
        for e, (gs, ge) in enumerate(zip(g_start.tolist(), g_end.tolist())):
            k = gs + int(np.argmin(ttc[gs:ge]))
            a, b = int(ra[k]), int(rb[k])
            ttc_e = float(ttc[k])
            t_start, t_end, t_min = float(t[ra[gs]]), float(t[ra[ge - 1]]), float(t[a])
            va, vb = int(vid[a]), int(vid[b])

            angle = float(heading_angle_deg(u[a], u[b]))
            dv = vel[a] - vel[b]
            delta_s = float(np.sqrt(dot2(dv, dv)))
            contact = 0.5 * ((xy[a] + vel[a] * ttc_e) + (xy[b] + vel[b] * ttc_e))

            lo = t_start - th.max_pet_s - dc.pet_search_padding_s
            hi = t_end + th.max_ttc_s + th.max_pet_s + dc.pet_search_padding_s
            sa = _window(tracks[va], track_t[va], lo, hi)
            sb = _window(tracks[vb], track_t[vb], lo, hi)
            exact = _encroachment(sa, sb, t, xy, u, half, 0.0)
            buffered = exact if exact is not None else _encroachment(sa, sb, t, xy, u, half, dc.encroachment_buffer_m)
            encroached = buffered is not None and buffered.dt <= th.max_pet_s + 1e-9
            pet = exact.dt if exact is not None else float("nan")

            horizon_rows_a = _window(tracks[va], track_t[va], t_min, t_min + ttc_e + dt)
            horizon_rows_b = _window(tracks[vb], track_t[vb], t_min, t_min + ttc_e + dt)
            on_course = _on_course(horizon_rows_a, xy, xy[a], u[a], dc.path_lateral_tolerance_m) and _on_course(
                horizon_rows_b, xy, xy[b], u[b], dc.path_lateral_tolerance_m
            )
            verified = encroached or on_course

            if buffered is not None:
                a_first = buffered.a_first
            else:
                follow_a = _window(tracks[va], track_t[va], t_min, t_min + ttc_e + th.max_pet_s)
                follow_b = _window(tracks[vb], track_t[vb], t_min, t_min + ttc_e + th.max_pet_s)
                pass_a = _pass_time(follow_a, t, xy, contact, u[a])
                pass_b = _pass_time(follow_b, t, xy, contact, u[b])
                if pass_a != pass_b:
                    a_first = pass_a < pass_b
                elif angle < th.rear_end_angle_deg:
                    a_first = bool(dot2(xy[b] - xy[a], u[a]) < 0.0)  # b behind a -> a leads
                else:
                    a_first = bool(acc[follow_a].mean() >= acc[follow_b].mean())  # harder braker is second
            first_vid, second_vid = (va, vb) if a_first else (vb, va)
            second_rows = tracks[second_vid]
            sec = _window(second_rows, track_t[second_vid], t_start, t_end + th.max_ttc_s)
            if sec.size:
                dr = float(acc[sec[0]])
                max_d = float(min(acc[sec].min(), dr))
            else:
                dr = max_d = 0.0
            spd_rows = np.concatenate([_window(tracks[va], track_t[va], t_start, t_end),
                                       _window(tracks[vb], track_t[vb], t_start, t_end)])
            max_s = float(spd[spd_rows].max()) if spd_rows.size else float(max(spd[a], spd[b]))

            if dc.mass_model == "footprint":
                m_a, m_b = float(length[a] * width[a]), float(length[b] * width[b])
            else:
                m_a = m_b = 1.0
            max_delta_v = max(m_b / (m_a + m_b), m_a / (m_a + m_b)) * delta_s

            point = (exact or buffered)
            px, py = point.point if point is not None else (float(contact[0]), float(contact[1]))
            collision = ttc_e <= 1e-9 or (exact is not None and exact.dt <= 0.5 * dt)
            rows.append(
                {
                    "event_id": e, "vid_a": va, "vid_b": vb, "first_vid": first_vid, "second_vid": second_vid,
                    "t_start": t_start, "t_end": t_end, "t_min_ttc": t_min, "ttc": ttc_e, "pet": pet,
                    "x": px, "y": py, "angle_deg": angle, "conflict_type": self._classify(angle),
                    "max_s": max_s, "delta_s": delta_s, "dr": dr, "max_d": max_d, "max_delta_v": max_delta_v,
                    "link_a": int(link[a]), "link_b": int(link[b]), "severe": ttc_e <= th.severe_ttc_s,
                    "collision": bool(collision), "path_verified": bool(verified),
                }
            )
        return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _window(rows: np.ndarray, times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rows of one vehicle with ``lo <= t <= hi`` (rows are time-ordered)."""
    i0 = int(np.searchsorted(times, lo - 1e-9, side="left"))
    i1 = int(np.searchsorted(times, hi + 1e-9, side="right"))
    if i1 - i0 > _MAX_WINDOW_SAMPLES:
        mid = (i0 + i1) // 2
        i0, i1 = mid - _MAX_WINDOW_SAMPLES // 2, mid + _MAX_WINDOW_SAMPLES // 2
    return rows[i0:i1]


def _on_course(rows: np.ndarray, xy: FloatArray, p0: FloatArray, u0: FloatArray, tol: float) -> bool:
    """True if every sample stays within ``tol`` metres laterally of the ray ``p0 + s * u0``."""
    if rows.size == 0:
        return False
    d = xy[rows] - p0
    lateral = np.abs(d[:, 0] * u0[1] - d[:, 1] * u0[0])
    return bool(lateral.max() <= tol)


def _pass_time(rows: np.ndarray, t: FloatArray, xy: FloatArray, point: FloatArray, u0: FloatArray) -> float:
    """First time the vehicle centre moves past ``point`` along heading ``u0`` (inf if never)."""
    if rows.size == 0:
        return float("inf")
    s = (xy[rows] - point) @ u0
    past = np.flatnonzero(s > 0.0)
    return float(t[rows[past[0]]]) if past.size else float("inf")


def _encroachment(
    sa: np.ndarray, sb: np.ndarray, t: FloatArray, xy: FloatArray, u: FloatArray, half: FloatArray, margin: float
) -> _Encroachment | None:
    """Minimum time separation at which the two footprints occupied common space."""
    if sa.size == 0 or sb.size == 0:
        return None
    ov = obb_overlap(
        xy[sa][:, None, :], u[sa][:, None, :], half[sa][:, None, :] + margin,
        xy[sb][None, :, :], u[sb][None, :, :], half[sb][None, :, :] + margin,
    )
    if not ov.any():
        return None
    ia, ib = np.nonzero(ov)
    dts = t[sb][ib] - t[sa][ia]
    k = int(np.argmin(np.abs(dts)))
    p = 0.5 * (xy[sa[ia[k]]] + xy[sb[ib[k]]])
    return _Encroachment(float(abs(dts[k])), bool(dts[k] >= 0.0), (float(p[0]), float(p[1])))
