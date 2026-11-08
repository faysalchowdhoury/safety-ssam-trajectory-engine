"""Synthetic trajectory fixtures with analytically known conflicts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssam_engine.io.fzp import write_fzp


def track(
    vid: int,
    t0: float,
    t1: float,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    *,
    dt: float = 0.1,
    length: float = 4.5,
    width: float = 1.8,
    accel: float = 0.0,
    link: int = 1,
) -> pd.DataFrame:
    """Constant-velocity (or constant-acceleration along heading) vehicle track."""
    t = np.round(np.arange(t0, t1 + dt / 2, dt), 6)
    tau = t - t0
    speed0 = float(np.hypot(vx, vy))
    heading = float(np.arctan2(vy, vx)) if speed0 > 0 else 0.0
    ux, uy = np.cos(heading), np.sin(heading)
    if accel == 0.0:
        dist = speed0 * tau
        spd = np.full_like(tau, speed0)
    else:
        t_stop = speed0 / -accel if accel < 0 else np.inf
        tc = np.minimum(tau, t_stop)
        dist = speed0 * tc + 0.5 * accel * tc**2
        spd = np.maximum(speed0 + accel * tc, 0.0)
    acc = np.where(spd > 0, accel, 0.0)
    return pd.DataFrame(
        {
            "t": t, "vid": vid, "x": x0 + ux * dist, "y": y0 + uy * dist, "heading": heading, "speed": spd,
            "accel": acc, "length": length, "width": width, "link": link, "lane": 1, "vtype": 100,
        }
    )


def path_track(vid: int, t: np.ndarray, xy: np.ndarray, *, length: float = 4.5, width: float = 1.8) -> pd.DataFrame:
    """Track from explicit positions; heading and speed from finite differences."""
    d = np.gradient(xy, t, axis=0)
    return pd.DataFrame(
        {
            "t": np.round(t, 6), "vid": vid, "x": xy[:, 0], "y": xy[:, 1], "heading": np.arctan2(d[:, 1], d[:, 0]),
            "speed": np.hypot(d[:, 0], d[:, 1]), "length": length, "width": width,
        }
    )


@pytest.fixture
def make_track() -> Callable[..., pd.DataFrame]:
    return track


def crossing_frames() -> pd.DataFrame:
    """A (eastbound, 10 m/s) and B (northbound) start on an exact collision course.

    B brakes at -8 m/s^2 from t = 3.0 s to 4.0 s, then creeps at 2 m/s, so A clears the
    crossing just before B's front arrives: a severe near miss with a short PET.
    """
    t = np.round(np.arange(0.0, 6.0 + 1e-9, 0.1), 6)
    a = track(1, 0.0, 6.0, -40.0, 0.0, 10.0, 0.0)
    s = np.clip(t - 3.0, 0.0, 1.0)
    y = np.where(t < 3.0, -40.0 + 10.0 * t, -10.0 + 10.0 * s - 4.0 * s**2)
    y = np.where(t > 4.0, -4.0 + 2.0 * (t - 4.0), y)
    v = np.where(t < 3.0, 10.0, np.where(t <= 4.0, 10.0 - 8.0 * s, 2.0))
    acc = np.where((t >= 3.0 - 1e-9) & (t < 4.0 - 1e-9), -8.0, 0.0)
    b = pd.DataFrame(
        {"t": t, "vid": 2, "x": 0.0, "y": y, "heading": np.pi / 2, "speed": v, "accel": acc, "length": 4.5,
         "width": 1.8, "link": 2, "lane": 1, "vtype": 100}
    )
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture
def crossing_near_miss() -> pd.DataFrame:
    return crossing_frames()


@pytest.fixture
def rear_end_brake() -> pd.DataFrame:
    """Leader brakes at -6 m/s^2; follower 7.5 m behind brakes at -4.5 m/s^2 and stops ~2 m short."""
    lead = track(1, 0.0, 6.0, 20.0, 0.0, 14.0, 0.0, accel=-6.0)
    foll = track(2, 0.0, 6.0, 8.0, 0.0, 14.0, 0.0, accel=-4.5)
    return pd.concat([lead, foll], ignore_index=True)


@pytest.fixture
def parallel_lanes() -> pd.DataFrame:
    a = track(1, 0.0, 5.0, 0.0, 0.0, 12.0, 0.0)
    b = track(2, 0.0, 5.0, 1.0, 3.6, 13.0, 0.0)
    return pd.concat([a, b], ignore_index=True)


def to_vissim_records(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical frame -> Vissim attribute columns (metres, km/h)."""
    ux, uy = np.cos(df["heading"]), np.sin(df["heading"])
    half = df["length"] / 2
    return pd.DataFrame(
        {
            "SIMSEC": df["t"], "NO": df["vid"].astype(int), "LANE\\LINK\\NO": df.get("link", 1),
            "LANE\\INDEX": df.get("lane", 1), "SPEED": df["speed"] * 3.6,
            "ACCELERATION": df.get("accel", 0.0), "COORDFRONTX": df["x"] + ux * half,
            "COORDFRONTY": df["y"] + uy * half, "COORDREARX": df["x"] - ux * half,
            "COORDREARY": df["y"] - uy * half, "LENGTH": df["length"], "WIDTH": df["width"],
            "VEHTYPE": df.get("vtype", 100),
        }
    )


@pytest.fixture
def fzp_file(tmp_path: Path, crossing_near_miss: pd.DataFrame, rear_end_brake: pd.DataFrame) -> Path:
    rear = rear_end_brake.copy()
    rear["vid"] += 10
    rear["y"] += 30.0
    df = pd.concat([crossing_near_miss, rear], ignore_index=True)
    return write_fzp(to_vissim_records(df), tmp_path / "sample.fzp")


@pytest.fixture
def records_factory() -> Callable[[Sequence[pd.DataFrame]], pd.DataFrame]:
    return lambda frames: to_vissim_records(pd.concat(list(frames), ignore_index=True))
