"""Static point indexes for bounding-box queries over conflict locations (API layer)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class PointIndex(ABC):
    """Index over a fixed set of 2-D points."""

    backend: str = "abstract"

    def __init__(self, xy: FloatArray) -> None:
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    @abstractmethod
    def query_bbox(self, min_x: float, min_y: float, max_x: float, max_y: float) -> IntArray:
        """Indices of points inside the closed box, in ascending order."""


class GridPointIndex(PointIndex):
    """Uniform bucket grid; O(cells + hits) per query."""

    backend = "grid"

    def __init__(self, xy: FloatArray, cell_m: float = 10.0) -> None:
        super().__init__(xy)
        self.cell = float(cell_m)
        self._buckets: dict[tuple[int, int], IntArray] = {}
        if len(self):
            keys = np.floor(self.xy / self.cell).astype(np.int64)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            order = np.argsort(inv, kind="stable")
            counts = np.bincount(inv, minlength=uniq.shape[0])
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            for c in range(uniq.shape[0]):
                self._buckets[(int(uniq[c, 0]), int(uniq[c, 1]))] = order[starts[c] : starts[c] + counts[c]]

    def query_bbox(self, min_x: float, min_y: float, max_x: float, max_y: float) -> IntArray:
        if not len(self):
            return np.empty(0, dtype=np.int64)
        kx0, ky0 = int(np.floor(min_x / self.cell)), int(np.floor(min_y / self.cell))
        kx1, ky1 = int(np.floor(max_x / self.cell)), int(np.floor(max_y / self.cell))
        n_cells = (kx1 - kx0 + 1) * (ky1 - ky0 + 1)
        if n_cells > len(self._buckets):
            items = [v for (kx, ky), v in self._buckets.items() if kx0 <= kx <= kx1 and ky0 <= ky <= ky1]
        else:
            items = [
                self._buckets[(kx, ky)]
                for kx in range(kx0, kx1 + 1)
                for ky in range(ky0, ky1 + 1)
                if (kx, ky) in self._buckets
            ]
        if not items:
            return np.empty(0, dtype=np.int64)
        cand = np.concatenate(items)
        p = self.xy[cand]
        inside = (p[:, 0] >= min_x) & (p[:, 0] <= max_x) & (p[:, 1] >= min_y) & (p[:, 1] <= max_y)
        return np.sort(cand[inside]).astype(np.int64)


class RTreePointIndex(PointIndex):
    """libspatialindex R*-tree (bulk loaded)."""

    backend = "rtree"

    def __init__(self, xy: FloatArray) -> None:
        super().__init__(xy)
        from rtree import index

        props = index.Property()
        props.dimension = 2
        props.variant = index.RT_Star
        self._tree: Any = None
        if len(self):
            self._tree = index.Index(
                ((i, (float(x), float(y), float(x), float(y)), None) for i, (x, y) in enumerate(self.xy)),
                properties=props,
            )

    def query_bbox(self, min_x: float, min_y: float, max_x: float, max_y: float) -> IntArray:
        if self._tree is None:
            return np.empty(0, dtype=np.int64)
        hits = np.fromiter(self._tree.intersection((min_x, min_y, max_x, max_y)), dtype=np.int64)
        return np.sort(hits)


def build_point_index(xy: FloatArray, backend: Literal["auto", "rtree", "grid"] = "auto") -> PointIndex:
    """Build an R*-tree when available (``auto``/``rtree``), otherwise a grid index."""
    if backend in ("auto", "rtree"):
        try:
            return RTreePointIndex(xy)
        except ImportError:
            if backend == "rtree":
                raise
    return GridPointIndex(xy)
