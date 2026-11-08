"""Four-leg intersection geometry: lane centrelines, stop lines and conflict points.

Coordinate frame: origin at the intersection centre, +x east, +y north,
right-hand traffic. Each approach has an exclusive left-turn lane (inner) and a
through lane (outer); each leg has two matching receiving lanes. Paths for NB,
WB and SB are the EB paths rotated by 90, 180 and 270 degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ..core.config import StrictModel

FloatArray = NDArray[np.float64]

MOVEMENTS: Final[tuple[str, ...]] = ("EBT", "EBL", "NBT", "NBL", "WBT", "WBL", "SBT", "SBL")
ROTATION_DEG: Final[dict[str, float]] = {"EB": 0.0, "NB": 90.0, "WB": 180.0, "SB": 270.0}
OPPOSING: Final[dict[str, str]] = {"EB": "WB", "WB": "EB", "NB": "SB", "SB": "NB"}


class IntersectionGeometry(StrictModel):
    lane_width_m: float = Field(3.6, gt=2.5, le=4.5)
    box_half_m: float = Field(12.0, gt=5.0, le=40.0, description="stop line offset from centre")
    approach_length_m: float = Field(200.0, ge=50.0)
    exit_length_m: float = Field(120.0, ge=30.0)
    resolution_m: float = Field(0.5, gt=0.05, le=2.0)


@dataclass(frozen=True)
class Polyline:
    """Arc-length parameterised polyline with linear extrapolation at both ends."""

    xy: FloatArray
    s: FloatArray

    @classmethod
    def from_points(cls, pts: FloatArray) -> "Polyline":
        pts = np.asarray(pts, dtype=np.float64)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        keep = np.concatenate([[True], seg > 1e-9])
        pts = pts[keep]
        seg = np.hypot(*np.diff(pts, axis=0).T)
        return cls(pts, np.concatenate([[0.0], np.cumsum(seg)]))

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def point(self, s: FloatArray | float) -> FloatArray:
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        x = np.interp(s_arr, self.s, self.xy[:, 0])
        y = np.interp(s_arr, self.s, self.xy[:, 1])
        d0 = (self.xy[1] - self.xy[0]) / (self.s[1] - self.s[0])
        d1 = (self.xy[-1] - self.xy[-2]) / (self.s[-1] - self.s[-2])
        before = s_arr < 0.0
        after = s_arr > self.s[-1]
        x = np.where(before, self.xy[0, 0] + s_arr * d0[0], x)
        y = np.where(before, self.xy[0, 1] + s_arr * d0[1], y)
        x = np.where(after, self.xy[-1, 0] + (s_arr - self.s[-1]) * d1[0], x)
        y = np.where(after, self.xy[-1, 1] + (s_arr - self.s[-1]) * d1[1], y)
        return np.column_stack([x, y])


@dataclass(frozen=True)
class MovementPath:
    """One movement's centreline and control points."""

    name: str
    approach: str
    turn: str
    link_no: int
    line: Polyline
    stop_s: float
    box_exit_s: float
    vmax_s: FloatArray
    vmax_v: FloatArray

    def vmax_at(self, s: float) -> float:
        return float(np.interp(s, self.vmax_s, self.vmax_v))

    def free_flow_time(self, speed_factor: float, extra_m: float = 0.0) -> float:
        """Travel time at desired speed from s=0 to ``length + extra_m``."""
        s = np.linspace(0.0, self.line.length + extra_m, 2_000)
        v = np.interp(s, self.vmax_s, self.vmax_v) * speed_factor
        return float(np.trapezoid(1.0 / v, s) if hasattr(np, "trapezoid") else np.trapz(1.0 / v, s))


def _line(p0: tuple[float, float], p1: tuple[float, float], res: float) -> FloatArray:
    n = max(2, int(math.ceil(math.dist(p0, p1) / res)) + 1)
    return np.column_stack([np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)])


def _arc(center: tuple[float, float], r: float, a0: float, a1: float, res: float) -> FloatArray:
    n = max(3, int(math.ceil(r * abs(a1 - a0) / res)) + 1)
    ang = np.linspace(a0, a1, n)
    return np.column_stack([center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)])


def _rotate(pts: FloatArray, deg: float) -> FloatArray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.column_stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]])


def build_movement_paths(
    geometry: IntersectionGeometry, approach_speed_mps: float, left_turn_speed_mps: float
) -> dict[str, MovementPath]:
    """Build all eight movement paths."""
    w, h = geometry.lane_width_m, geometry.box_half_m
    app, ext, res = geometry.approach_length_m, geometry.exit_length_m, geometry.resolution_m
    x0 = -h - app

    eb_through = _line((x0, -1.5 * w), (h + ext, -1.5 * w), res)
    r = h + 0.5 * w
    eb_left = np.vstack(
        [
            _line((x0, -0.5 * w), (-h, -0.5 * w), res),
            _arc((-h, h), r, -math.pi / 2.0, 0.0, res)[1:],
            _line((0.5 * w, h), (0.5 * w, h + ext), res)[1:],
        ]
    )
    templates = {"T": (eb_through, 2.0 * h), "L": (eb_left, r * math.pi / 2.0)}

    paths: dict[str, MovementPath] = {}
    for name in MOVEMENTS:
        approach, turn = name[:2], name[2]
        pts, box_len = templates[turn]
        line = Polyline.from_points(_rotate(pts, ROTATION_DEG[approach]))
        stop_s = app
        box_exit = app + box_len
        va, vt = approach_speed_mps, left_turn_speed_mps
        if turn == "T":
            ks, kv = np.array([0.0, line.length]), np.array([va, va])
        else:
            ks = np.array([0.0, stop_s - 35.0, stop_s, box_exit, box_exit + 35.0, line.length])
            kv = np.array([va, va, vt, vt, va, va])
        paths[name] = MovementPath(name, approach, turn, MOVEMENTS.index(name) + 1, line, stop_s, box_exit, ks, kv)
    return paths


def _cross(a: FloatArray, b: FloatArray) -> FloatArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def first_intersection(
    pa: MovementPath, pb: MovementPath, margin_m: float = 1.0
) -> tuple[float, float] | None:
    """Arc-length positions ``(s_a, s_b)`` where the two paths cross inside the box."""

    def clip(p: MovementPath) -> tuple[FloatArray, FloatArray]:
        keep = (p.line.s >= p.stop_s - margin_m) & (p.line.s <= p.box_exit_s + margin_m)
        return p.line.xy[keep], p.line.s[keep]

    xa, sa = clip(pa)
    xb, sb = clip(pb)
    if len(xa) < 2 or len(xb) < 2:
        return None
    a0, d1 = xa[:-1], np.diff(xa, axis=0)
    b0, d2 = xb[:-1], np.diff(xb, axis=0)
    denom = _cross(d1[:, None, :], d2[None, :, :])
    rr = b0[None, :, :] - a0[:, None, :]
    ok_den = np.abs(denom) > 1e-12
    safe = np.where(ok_den, denom, 1.0)
    ta = _cross(rr, d2[None, :, :]) / safe
    tb = _cross(rr, d1[:, None, :]) / safe
    ok = ok_den & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
    if not ok.any():
        return None
    ia, ib = np.nonzero(ok)
    k = int(np.argmin(sa[ia] + ta[ia, ib] * np.diff(sa)[ia]))
    i, j = int(ia[k]), int(ib[k])
    s_a = float(sa[i] + ta[i, j] * (sa[i + 1] - sa[i]))
    s_b = float(sb[j] + tb[i, j] * (sb[j + 1] - sb[j]))
    return s_a, s_b


def find_conflicts(paths: dict[str, MovementPath]) -> dict[str, list[tuple[str, float, float]]]:
    """All crossing points ``{movement: [(other, s_self, s_other), ...]}`` between different approaches."""
    out: dict[str, list[tuple[str, float, float]]] = {m: [] for m in paths}
    names = list(paths)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if paths[a].approach == paths[b].approach:
                continue
            hit = first_intersection(paths[a], paths[b])
            if hit is not None:
                out[a].append((b, hit[0], hit[1]))
                out[b].append((a, hit[1], hit[0]))
    return out
