from __future__ import annotations

import numpy as np
import pytest

from ssam_engine.core.exceptions import ConfigurationError
from ssam_engine.spatial.pairs import available_backends, make_pair_finder
from ssam_engine.spatial.points import GridPointIndex, build_point_index


def _key(pairs: np.ndarray, n: int) -> np.ndarray:
    return np.sort(pairs[:, 0] * n + pairs[:, 1])


@pytest.mark.parametrize("n", [2, 17, 300, 1500])
def test_all_backends_match_brute_force(n: int) -> None:
    rng = np.random.default_rng(n)
    xy = rng.uniform(0, 400, (n, 2))
    radius = rng.uniform(1, 25, n)
    ref = _key(make_pair_finder("brute").candidate_pairs(xy, radius), n)
    for name in available_backends():
        got = make_pair_finder(name).candidate_pairs(xy, radius)
        assert np.all(got[:, 0] < got[:, 1]), name
        assert np.array_equal(_key(got, n), ref), name


def test_single_vehicle_frame_returns_no_pairs() -> None:
    for name in available_backends():
        assert make_pair_finder(name).candidate_pairs(np.zeros((1, 2)), np.ones(1)).shape == (0, 2)


def test_unknown_backend() -> None:
    with pytest.raises(ConfigurationError):
        make_pair_finder("quadtree")


def test_rtree_backend_available() -> None:
    assert "rtree" in available_backends()


def test_point_indexes_agree() -> None:
    rng = np.random.default_rng(5)
    xy = rng.uniform(-50, 50, (2_000, 2))
    grid = GridPointIndex(xy, cell_m=7.0)
    best = build_point_index(xy)
    for _ in range(50):
        x0, y0 = rng.uniform(-60, 40, 2)
        w, h = rng.uniform(0.5, 40, 2)
        brute = np.flatnonzero((xy[:, 0] >= x0) & (xy[:, 0] <= x0 + w) & (xy[:, 1] >= y0) & (xy[:, 1] <= y0 + h))
        assert np.array_equal(grid.query_bbox(x0, y0, x0 + w, y0 + h), brute)
        assert np.array_equal(best.query_bbox(x0, y0, x0 + w, y0 + h), brute)


def test_empty_point_index() -> None:
    assert build_point_index(np.empty((0, 2))).query_bbox(0, 0, 1, 1).size == 0
