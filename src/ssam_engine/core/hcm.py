"""HCM 7th Edition level-of-service lookups and the FHWA GEH statistic."""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np

# HCM 7th Ed. Exhibit 19-8 (signalized) and Exhibit 20-2 (TWSC/AWSC), control delay in s/veh.
SIGNALIZED_LOS: Final[tuple[tuple[float, str], ...]] = (
    (10.0, "A"),
    (20.0, "B"),
    (35.0, "C"),
    (55.0, "D"),
    (80.0, "E"),
)
UNSIGNALIZED_LOS: Final[tuple[tuple[float, str], ...]] = (
    (10.0, "A"),
    (15.0, "B"),
    (25.0, "C"),
    (35.0, "D"),
    (50.0, "E"),
)


def los_from_delay(
    delay_s: float,
    control: Literal["signalized", "unsignalized"] = "signalized",
    v_c_ratio: float | None = None,
) -> str:
    """Return the HCM LOS letter for a control delay.

    A lane group / movement with ``v/c > 1.0`` is LOS F regardless of delay (HCM 7).
    Returns ``"n/a"`` for NaN delay.
    """
    if delay_s is None or (isinstance(delay_s, float) and math.isnan(delay_s)):
        return "n/a"
    if v_c_ratio is not None and v_c_ratio > 1.0:
        return "F"
    table = SIGNALIZED_LOS if control == "signalized" else UNSIGNALIZED_LOS
    for upper, letter in table:
        if delay_s <= upper:
            return letter
    return "F"


def geh(model_vph: float | np.ndarray, count_vph: float | np.ndarray) -> float | np.ndarray:
    """GEH statistic: ``sqrt(2 (M - C)^2 / (M + C))`` with hourly flows.

    FHWA microsimulation guidance targets GEH < 5 for at least 85 % of links.
    Zero-flow pairs return 0.
    """
    m = np.asarray(model_vph, dtype=np.float64)
    c = np.asarray(count_vph, dtype=np.float64)
    denom = m + c
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(denom > 0, np.sqrt(2.0 * (m - c) ** 2 / np.where(denom > 0, denom, 1.0)), 0.0)
    return float(val) if val.ndim == 0 else val


def geh_pass_rate(model_vph: np.ndarray, count_vph: np.ndarray, threshold: float = 5.0) -> float:
    """Share (0-1) of links/movements with GEH below ``threshold``."""
    values = np.atleast_1d(geh(model_vph, count_vph))
    return float(np.mean(values < threshold)) if values.size else float("nan")
