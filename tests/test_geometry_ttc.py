from __future__ import annotations

import numpy as np
import pytest

from ssam_engine.core.geometry import heading_angle_deg, heading_vectors, obb_overlap
from ssam_engine.core.ttc import closest_approach_mask, compute_ttc

H = np.array([[2.25, 0.9]])
EAST = np.array([[1.0, 0.0]])
NORTH = np.array([[0.0, 1.0]])


def _ttc(p1, v1, u1, p2, v2, u2, horizon=3.0):
    return compute_ttc(np.array([p1], float), np.array([v1], float), np.array([u1], float), H,
                       np.array([p2], float), np.array([v2], float), np.array([u2], float), H,
                       horizon=horizon, resolution=0.05, refine_iterations=8)[0]


def test_obb_overlap_basic_cases() -> None:
    assert obb_overlap(np.array([[0, 0.0]]), EAST, H, np.array([[4.0, 0]]), EAST, H)[0]
    assert not obb_overlap(np.array([[0, 0.0]]), EAST, H, np.array([[4.6, 0]]), EAST, H)[0]
    assert not obb_overlap(np.array([[0, 0.0]]), EAST, H, np.array([[0, 3.6]]), EAST, H)[0]
    # rotated 45 degrees: corner reaches further than the axis-aligned half width
    diag = heading_vectors(np.array([np.pi / 4]))
    assert obb_overlap(np.array([[0, 0.0]]), EAST, H, np.array([[0, 3.0]]), diag, H)[0]


def test_head_on_ttc_matches_closed_form() -> None:
    # centres 40 m apart, closing at 20 m/s, contact when centres are 4.5 m apart
    ttc = _ttc((0, 0), (10, 0), (1, 0), (40, 0), (-10, 0), (-1, 0))
    assert ttc == pytest.approx((40 - 4.5) / 20, abs=1e-3)


def test_rear_end_ttc_against_stopped_leader() -> None:
    ttc = _ttc((0, 0), (10, 0), (1, 0), (20, 0), (0, 0), (1, 0))
    assert ttc == pytest.approx((20 - 4.5) / 10, abs=1e-3)


def test_crossing_ttc_perpendicular() -> None:
    # A heading east reaches x = -(2.25 + 0.9) relative to B's centreline when its front meets B's side
    # B one second behind: A has cleared before B arrives, so constant-velocity projection never collides
    assert np.isinf(_ttc((-20, 0), (10, 0), (1, 0), (0, -30), (0, 10), (0, 1)))
    ttc2 = _ttc((-20, 0), (10, 0), (1, 0), (0, -20), (0, 10), (0, 1))
    assert ttc2 == pytest.approx((20 - 2.25 - 0.9) / 10, abs=1e-3)


def test_parallel_lanes_never_collide() -> None:
    assert np.isinf(_ttc((0, 0), (12, 0), (1, 0), (2, 3.6), (8, 0), (1, 0)))


def test_overlap_now_is_ttc_zero() -> None:
    assert _ttc((0, 0), (5, 0), (1, 0), (3, 0), (0, 0), (1, 0)) == 0.0


def test_closest_approach_filter_has_no_false_negatives() -> None:
    rng = np.random.default_rng(0)
    n = 4_000
    p1 = rng.uniform(-30, 30, (n, 2))
    p2 = rng.uniform(-30, 30, (n, 2))
    h1 = heading_vectors(rng.uniform(-np.pi, np.pi, n))
    h2 = heading_vectors(rng.uniform(-np.pi, np.pi, n))
    v1 = h1 * rng.uniform(0, 20, n)[:, None]
    v2 = h2 * rng.uniform(0, 20, n)[:, None]
    half = np.tile(H, (n, 1))
    full = compute_ttc(p1, v1, h1, half, p2, v2, h2, half, horizon=1.5, resolution=0.05)
    reach = np.full(n, 2 * np.hypot(*H[0]))
    mask = closest_approach_mask(p1, v1, p2, v2, reach, 1.5)
    assert np.all(mask[np.isfinite(full)])
    assert mask.mean() < 1.0


def test_heading_angle() -> None:
    assert heading_angle_deg(EAST, NORTH)[0] == pytest.approx(90.0)
    assert heading_angle_deg(EAST, -EAST)[0] == pytest.approx(180.0)
