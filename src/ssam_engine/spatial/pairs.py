"""Candidate-pair generation per simulation time step.

Each vehicle ``i`` has a *reach* radius ``r_i = speed_i * max_ttc + half_diagonal_i``.
Two vehicles can only produce TTC <= max_ttc if ``|p_i - p_j| <= r_i + r_j``. The
backends below all return exactly that set of pairs (``i < j``); they differ only
in how they avoid the O(n^2) all-pairs test:

=========  =====================================  ==========================
backend    structure                              build / query (n vehicles)
=========  =====================================  ==========================
brute      dense distance matrix                  O(n^2) / -
grid       uniform spatial hash, cell = 2*r_max   O(n) / O(n * k)
kdtree     scipy cKDTree ``query_pairs``          O(n log n) / O(n log n + k)
rtree      libspatialindex R*-tree (bulk loaded)  O(n log n) / O(log n + k)
=========  =====================================  ==========================

``k`` is the number of neighbours per vehicle, which is bounded by traffic density.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..core.exceptions import ConfigurationError, SpatialIndexUnavailableError

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

_EMPTY = np.empty((0, 2), dtype=np.int64)
_HALF_NEIGHBOURS: tuple[tuple[int, int], ...] = ((1, -1), (1, 0), (1, 1), (0, 1))


def _filter_pairs(xy: FloatArray, radius: FloatArray, pairs: IntArray) -> IntArray:
    """Normalise to ``i < j`` and keep pairs whose reach circles intersect."""
    if pairs.size == 0:
        return _EMPTY
    i = np.minimum(pairs[:, 0], pairs[:, 1])
    j = np.maximum(pairs[:, 0], pairs[:, 1])
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1])
    keep = d <= radius[i] + radius[j]
    return np.column_stack([i[keep], j[keep]]).astype(np.int64, copy=False)


class PairFinder(ABC):
    """Abstract candidate-pair generator."""

    name: ClassVar[str] = "abstract"

    def candidate_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        """Return unique pairs ``(i, j)``, ``i < j``, with ``|p_i - p_j| <= r_i + r_j``."""
        if xy.shape[0] < 2:
            return _EMPTY
        return _filter_pairs(xy, radius, self._raw_pairs(xy, radius))

    @abstractmethod
    def _raw_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        """Return a superset of the qualifying pairs (each unordered pair at most once)."""


class BruteForcePairFinder(PairFinder):
    """Dense all-pairs reference implementation (testing / tiny frames)."""

    name = "brute"

    def _raw_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        n = xy.shape[0]
        iu, ju = np.triu_indices(n, 1)
        d = np.hypot(xy[ju, 0] - xy[iu, 0], xy[ju, 1] - xy[iu, 1])
        keep = d <= radius[iu] + radius[ju]
        return np.column_stack([iu[keep], ju[keep]]).astype(np.int64)


class KDTreePairFinder(PairFinder):
    """scipy ``cKDTree.query_pairs`` with the global maximum reach, then exact filter."""

    name = "kdtree"

    def _raw_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        tree = cKDTree(xy)
        pairs = tree.query_pairs(r=2.0 * float(radius.max()), output_type="ndarray")
        return pairs.astype(np.int64, copy=False)


class GridPairFinder(PairFinder):
    """Uniform spatial hash. Cell size ``2 * r_max`` guarantees neighbours lie in adjacent cells."""

    name = "grid"

    def __init__(self, min_cell_m: float = 1.0) -> None:
        self.min_cell_m = min_cell_m

    def _raw_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        cell = max(2.0 * float(radius.max()), self.min_cell_m)
        keys = np.floor(xy / cell).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=uniq.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        buckets: dict[tuple[int, int], IntArray] = {
            (int(uniq[c, 0]), int(uniq[c, 1])): order[starts[c] : starts[c] + counts[c]]
            for c in range(uniq.shape[0])
        }
        chunks: list[IntArray] = []
        for (kx, ky), members in buckets.items():
            m = members.size
            if m > 1:
                a, b = np.triu_indices(m, 1)
                chunks.append(np.column_stack([members[a], members[b]]))
            for dx, dy in _HALF_NEIGHBOURS:
                other = buckets.get((kx + dx, ky + dy))
                if other is not None:
                    chunks.append(np.column_stack([np.repeat(members, other.size), np.tile(other, m)]))
        return np.concatenate(chunks).astype(np.int64) if chunks else _EMPTY


class RTreePairFinder(PairFinder):
    """libspatialindex R*-tree, bulk-loaded each frame, queried with the vectorised API."""

    name = "rtree"

    def __init__(self, leaf_capacity: int = 32) -> None:
        try:
            from rtree import index
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SpatialIndexUnavailableError(
                "rtree backend requested but the 'rtree' package is not installed"
            ) from exc
        self._index: Any = index
        props = index.Property()
        props.dimension = 2
        props.variant = index.RT_Star
        props.leaf_capacity = leaf_capacity
        self._props = props

    def _raw_pairs(self, xy: FloatArray, radius: FloatArray) -> IntArray:
        n = xy.shape[0]
        lo = xy - radius[:, None]
        hi = xy + radius[:, None]
        boxes = np.column_stack([lo, hi])
        tree = self._index.Index(
            ((i, (float(b[0]), float(b[1]), float(b[2]), float(b[3])), None) for i, b in enumerate(boxes)),
            properties=self._props,
        )
        if hasattr(tree, "intersection_v"):
            ids, counts = tree.intersection_v(np.ascontiguousarray(lo), np.ascontiguousarray(hi))
            src = np.repeat(np.arange(n, dtype=np.int64), counts)
            ids = np.asarray(ids, dtype=np.int64)
        else:  # pragma: no cover - rtree < 1.1
            parts_src, parts_ids = [], []
            for i, b in enumerate(boxes):
                hits = np.fromiter(tree.intersection(tuple(b)), dtype=np.int64)
                parts_src.append(np.full(hits.size, i, dtype=np.int64))
                parts_ids.append(hits)
            src = np.concatenate(parts_src)
            ids = np.concatenate(parts_ids)
        keep = ids > src
        return np.column_stack([src[keep], ids[keep]])


_BACKENDS: dict[str, type[PairFinder]] = {
    "brute": BruteForcePairFinder,
    "grid": GridPairFinder,
    "kdtree": KDTreePairFinder,
    "rtree": RTreePairFinder,
}


def available_backends() -> list[str]:
    """Backends that can be constructed in this environment."""
    names = []
    for name in _BACKENDS:
        try:
            make_pair_finder(name)
        except SpatialIndexUnavailableError:
            continue
        names.append(name)
    return names


def make_pair_finder(name: str) -> PairFinder:
    """Construct a backend by name."""
    try:
        cls = _BACKENDS[name]
    except KeyError as exc:
        raise ConfigurationError("unknown spatial backend", context={"backend": name}) from exc
    return cls()
