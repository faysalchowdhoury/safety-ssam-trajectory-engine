"""Multi-seed stochastic statistics (FHWA Traffic Analysis Toolbox Vol. III practice)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats


def summarize(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean, sample SD, Student-t confidence half-width, min, max and n."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci": math.nan, "min": math.nan, "max": math.nan}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    ci = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * sd / math.sqrt(n)) if n > 1 else math.nan
    return {"n": n, "mean": mean, "sd": sd, "ci": ci, "min": float(x.min()), "max": float(x.max())}


def required_replications(
    values: Sequence[float], tolerance: float = 0.15, confidence: float = 0.95, max_runs: int = 200
) -> int:
    """Minimum number of seeds so the CI half-width is within ``tolerance * mean``.

    Iterates ``N >= (t_{alpha/2, N-1} * s / (tolerance * mean))^2`` (FHWA TAT Vol. III, eq. 2).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return max_runs
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd == 0.0:
        return int(x.size)
    if mean == 0.0:
        return max_runs
    err = tolerance * abs(mean)
    for n in range(2, max_runs + 1):
        t = stats.t.ppf(0.5 + confidence / 2.0, n - 1)
        if n >= (t * sd / err) ** 2:
            return n
    return max_runs


def welch_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's unequal-variance t-test ``(t, p)``; NaN when undefined."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.size < 2 or xb.size < 2 or (xa.std() == 0 and xb.std() == 0):
        return math.nan, math.nan
    res = stats.ttest_ind(xa, xb, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def ks_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov ``(D, p)`` on distributions (e.g. pooled TTC)."""
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    if xa.size == 0 or xb.size == 0:
        return math.nan, math.nan
    res = stats.ks_2samp(xa, xb)
    return float(res.statistic), float(res.pvalue)


def pct_change(base: float, alt: float) -> float:
    if base == 0 or not math.isfinite(base) or not math.isfinite(alt):
        return math.nan
    return 100.0 * (alt - base) / base
