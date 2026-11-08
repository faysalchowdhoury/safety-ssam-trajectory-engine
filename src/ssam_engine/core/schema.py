"""Canonical trajectory schema and validation.

All readers convert their input to this SI-unit, long-format table:

======== ======= ===============================================
column   dtype   meaning
======== ======= ===============================================
t        float   simulation time (s)
vid      int     vehicle number
x, y     float   footprint centre (m)
heading  float   heading angle (rad, counter-clockwise from +x)
speed    float   speed (m/s), non-negative
accel    float   longitudinal acceleration (m/s^2)
length   float   vehicle length (m)
width    float   vehicle width (m)
link     int     Vissim link number (-1 if unknown)
lane     int     lane index (-1 if unknown)
vtype    int     vehicle type (-1 if unknown)
======== ======= ===============================================
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from .exceptions import TrajectoryFormatError

REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("t", "vid", "x", "y", "heading", "speed", "length")
OPTIONAL_DEFAULTS: Final[dict[str, float | int]] = {"link": -1, "lane": -1, "vtype": -1}
TRAJECTORY_COLUMNS: Final[tuple[str, ...]] = (
    "t",
    "vid",
    "x",
    "y",
    "heading",
    "speed",
    "accel",
    "length",
    "width",
    "link",
    "lane",
    "vtype",
)


def derive_acceleration(df: pd.DataFrame) -> pd.Series:
    """Finite-difference acceleration per vehicle (first sample back-filled)."""
    ordered = df.sort_values(["vid", "t"], kind="mergesort")
    grouped = ordered.groupby("vid", sort=False)
    dv = grouped["speed"].diff()
    dt = grouped["t"].diff()
    acc = (dv / dt).replace([np.inf, -np.inf], np.nan)
    acc = acc.groupby(ordered["vid"], sort=False).bfill().fillna(0.0)
    return acc.reindex(df.index)


def validate_trajectories(df: pd.DataFrame, *, default_width: float = 1.8) -> pd.DataFrame:
    """Validate and normalise a trajectory table.

    Returns a new frame sorted by ``(t, vid)`` with a fresh RangeIndex and every
    column of :data:`TRAJECTORY_COLUMNS` present.

    Raises:
        TrajectoryFormatError: missing columns, non-finite values, negative
            speeds/dimensions or duplicate ``(t, vid)`` samples.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TrajectoryFormatError("trajectory table is missing columns", context={"missing": missing})

    out = df.copy()
    if "width" not in out.columns:
        out["width"] = default_width
    out["width"] = out["width"].fillna(default_width)
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(np.int64)

    numeric = ["t", "x", "y", "heading", "speed", "length", "width"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
    out["vid"] = pd.to_numeric(out["vid"], errors="coerce")
    bad = out[numeric + ["vid"]].isna().any(axis=1) | ~np.isfinite(out[numeric].to_numpy()).all(axis=1)
    if bad.any():
        raise TrajectoryFormatError(
            "trajectory table contains missing or non-finite values",
            context={"bad_rows": int(bad.sum())},
        )
    out["vid"] = out["vid"].astype(np.int64)

    if (out["speed"] < 0).any():
        raise TrajectoryFormatError("negative speeds found", context={"rows": int((out["speed"] < 0).sum())})
    if ((out["length"] <= 0) | (out["width"] <= 0)).any():
        raise TrajectoryFormatError("vehicle length and width must be positive")

    dup = out.duplicated(subset=["t", "vid"])
    if dup.any():
        raise TrajectoryFormatError("duplicate (t, vid) samples", context={"duplicates": int(dup.sum())})

    if "accel" not in out.columns or out["accel"].isna().all():
        out["accel"] = derive_acceleration(out)
    else:
        out["accel"] = pd.to_numeric(out["accel"], errors="coerce").astype(np.float64)
        if out["accel"].isna().any():
            out["accel"] = out["accel"].fillna(derive_acceleration(out))

    out = out.sort_values(["t", "vid"], kind="mergesort").reset_index(drop=True)
    return out[list(TRAJECTORY_COLUMNS)]


def infer_time_step(t: np.ndarray) -> float:
    """Infer the simulation resolution (s) from sample times."""
    uniq = np.unique(np.round(t, 6))
    if uniq.size < 2:
        return 0.1
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 1e-9]
    return float(np.median(diffs)) if diffs.size else 0.1
