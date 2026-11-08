"""Vectorised oriented-bounding-box (OBB) geometry.

Vehicles are modelled as rectangles (length x width) centred on ``c`` with unit
heading ``u``. Overlap uses the Separating Axis Theorem on the four candidate
axes (both boxes' longitudinal and lateral axes). All functions broadcast over
leading dimensions so that *P pairs x K look-ahead samples* are tested in a
single NumPy call with no Python loop.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def dot2(a: FloatArray, b: FloatArray) -> FloatArray:
    """Row-wise 2-D dot product over the last axis."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def perp(u: FloatArray) -> FloatArray:
    """Rotate vectors by +90 degrees."""
    return np.stack([-u[..., 1], u[..., 0]], axis=-1)


def heading_vectors(heading: FloatArray) -> FloatArray:
    """Unit vectors (n, 2) from headings in radians."""
    return np.column_stack([np.cos(heading), np.sin(heading)])


def obb_overlap(
    c1: FloatArray,
    u1: FloatArray,
    h1: FloatArray,
    c2: FloatArray,
    u2: FloatArray,
    h2: FloatArray,
) -> BoolArray:
    """Return True where two oriented rectangles overlap (touching counts as overlap).

    Args:
        c1, c2: centres, shape ``(..., 2)``.
        u1, u2: unit headings, shape ``(..., 2)``.
        h1, h2: half extents ``(half_length, half_width)``, shape ``(..., 2)``.

    All arguments must be mutually broadcastable.
    """
    n1 = perp(u1)
    n2 = perp(u2)
    d = c2 - c1
    shape = np.broadcast_shapes(c1.shape, c2.shape, u1.shape, u2.shape, h1.shape, h2.shape)[:-1]
    separated = np.zeros(shape, dtype=bool)
    for axis in (u1, n1, u2, n2):
        dist = np.abs(dot2(d, axis))
        r1 = h1[..., 0] * np.abs(dot2(u1, axis)) + h1[..., 1] * np.abs(dot2(n1, axis))
        r2 = h2[..., 0] * np.abs(dot2(u2, axis)) + h2[..., 1] * np.abs(dot2(n2, axis))
        separated |= dist > (r1 + r2 + 1e-12)
    return ~separated


def heading_angle_deg(u1: FloatArray, u2: FloatArray) -> FloatArray:
    """Unsigned angle between heading vectors in degrees, in [0, 180]."""
    cos = np.clip(dot2(u1, u2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))
