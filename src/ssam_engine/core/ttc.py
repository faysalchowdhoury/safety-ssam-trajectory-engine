"""Time-to-collision kernels.

Following SSAM, each vehicle is projected forward along its current heading at
its current speed. TTC is the first look-ahead time at which the two projected
footprints overlap. The search is done in two vectorised stages:

1. **Closest-approach filter** (exact, O(P)): bounding circles of radius equal to
   the half-diagonal move linearly, so the minimum centre distance over
   ``[0, horizon]`` has a closed form. Pairs whose circles never touch cannot
   collide and are discarded.
2. **Sampled SAT + bisection** (O(P*K)): survivors are tested on a uniform grid of
   look-ahead times, then the first overlapping interval is refined by bisection
   to ``resolution / 2**iterations``.
"""

from __future__ import annotations

import numpy as np

from .geometry import BoolArray, FloatArray, dot2, obb_overlap


def closest_approach_mask(
    p1: FloatArray,
    v1: FloatArray,
    p2: FloatArray,
    v2: FloatArray,
    reach: FloatArray,
    horizon: float,
) -> BoolArray:
    """True where the minimum centre distance within ``[0, horizon]`` is <= ``reach``."""
    r = p2 - p1
    w = v2 - v1
    ww = dot2(w, w)
    moving = ww > 1e-12
    tau = np.where(moving, -dot2(r, w) / np.where(moving, ww, 1.0), 0.0)
    tau = np.clip(tau, 0.0, horizon)
    closest = r + w * tau[..., None]
    return np.sqrt(dot2(closest, closest)) <= reach


def compute_ttc(
    p1: FloatArray,
    v1: FloatArray,
    u1: FloatArray,
    h1: FloatArray,
    p2: FloatArray,
    v2: FloatArray,
    u2: FloatArray,
    h2: FloatArray,
    *,
    horizon: float,
    resolution: float,
    refine_iterations: int = 5,
) -> FloatArray:
    """Vectorised TTC for ``P`` vehicle pairs.

    Args:
        p1, p2: centres (P, 2) in metres.
        v1, v2: velocity vectors (P, 2) in m/s.
        u1, u2: unit headings (P, 2).
        h1, h2: half extents (P, 2).
        horizon: maximum look-ahead (s), normally the SSAM max TTC.
        resolution: look-ahead sampling step (s).
        refine_iterations: bisection steps applied to the first overlapping interval.

    Returns:
        TTC in seconds, ``np.inf`` where no overlap occurs within the horizon and
        ``0.0`` where footprints already overlap.
    """
    n = p1.shape[0]
    out = np.full(n, np.inf)
    if n == 0:
        return out
    k = int(np.floor(horizon / resolution + 1e-9)) + 1
    taus = np.arange(k, dtype=np.float64) * resolution
    if taus[-1] < horizon - 1e-9:
        taus = np.append(taus, horizon)

    c1 = p1[:, None, :] + v1[:, None, :] * taus[None, :, None]
    c2 = p2[:, None, :] + v2[:, None, :] * taus[None, :, None]
    overlap = obb_overlap(c1, u1[:, None, :], h1[:, None, :], c2, u2[:, None, :], h2[:, None, :])
    has = overlap.any(axis=1)
    if not has.any():
        return out

    idx = np.flatnonzero(has)
    first = np.argmax(overlap[idx], axis=1)
    hi = taus[first].copy()
    need = first > 0
    if refine_iterations > 0 and need.any():
        ii = idx[need]
        hi_r = taus[first[need]].copy()
        lo_r = taus[first[need] - 1].copy()
        for _ in range(refine_iterations):
            mid = 0.5 * (lo_r + hi_r)
            o = obb_overlap(
                p1[ii] + v1[ii] * mid[:, None],
                u1[ii],
                h1[ii],
                p2[ii] + v2[ii] * mid[:, None],
                u2[ii],
                h2[ii],
            )
            hi_r = np.where(o, mid, hi_r)
            lo_r = np.where(o, lo_r, mid)
        hi[need] = hi_r
    out[idx] = hi
    return out
