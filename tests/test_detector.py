from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssam_engine.core.config import ConflictThresholds, DetectorConfig, EngineConfig
from ssam_engine.core.detector import ConflictDetector
from ssam_engine.core.exceptions import TrajectoryFormatError

from .conftest import path_track, track


def test_crossing_near_miss_is_one_crossing_event(crossing_near_miss: pd.DataFrame) -> None:
    res = ConflictDetector().detect(crossing_near_miss)
    ev = res.events
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["conflict_type"] == "crossing"
    assert e["angle_deg"] == pytest.approx(90.0, abs=1e-6)
    assert e["first_vid"] == 1 and e["second_vid"] == 2
    # analytic: A leaves the shared square at t = 4.315 s, B enters at 4.425 s; 0.1 s sampling -> 0.2 s
    assert e["pet"] == pytest.approx(0.2, abs=1e-6)
    # analytic minimum TTC ~0.39 s while B is braking (well inside the 1.5 s threshold, below 0.5 s "severe")
    assert 0.3 < e["ttc"] < 0.5 and bool(e["severe"])
    assert bool(e["path_verified"]) and not bool(e["collision"])
    vb = crossing_near_miss[(crossing_near_miss["vid"] == 2) & np.isclose(crossing_near_miss["t"], e["t_min_ttc"])]
    assert e["delta_s"] == pytest.approx(np.hypot(10.0, float(vb["speed"].iloc[0])), rel=1e-6)
    assert e["max_delta_v"] == pytest.approx(0.5 * e["delta_s"], rel=1e-6)
    assert e["max_d"] == pytest.approx(-8.0)
    assert -10.0 < e["x"] < 10.0 and -10.0 < e["y"] < 10.0


def test_rear_end_braking_leader(rear_end_brake: pd.DataFrame) -> None:
    ev = ConflictDetector().detect(rear_end_brake).events
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["conflict_type"] == "rear_end"
    assert e["first_vid"] == 1 and e["second_vid"] == 2
    # the follower stops on pavement the leader occupied 0.6 s earlier -> finite PET
    assert e["pet"] == pytest.approx(0.6, abs=1e-6)
    assert e["max_d"] == pytest.approx(-4.5)
    assert e["ttc"] < 1.5 and not bool(e["collision"])


def test_stop_short_behind_stationary_queue_has_no_pet() -> None:
    leader = track(1, 0.0, 6.0, 36.0, 0.0, 0.0, 0.0)
    follower = track(2, 0.0, 6.0, 10.0, 0.0, 14.0, 0.0, accel=-5.0)
    ev = ConflictDetector().detect(pd.concat([leader, follower], ignore_index=True)).events
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["conflict_type"] == "rear_end" and e["second_vid"] == 2
    assert np.isnan(e["pet"])  # the follower never reaches the stopped leader's footprint
    assert bool(e["path_verified"])
    # analytic TTC = gap / closing speed; minimum at t ~ 2 s: (21.5 - 18) / 4 = 0.875 s
    assert e["ttc"] == pytest.approx(0.875, abs=0.02)
    assert e["max_d"] == pytest.approx(-5.0)


def test_parallel_lanes_produce_no_events(parallel_lanes: pd.DataFrame) -> None:
    res = ConflictDetector().detect(parallel_lanes)
    assert res.events.empty
    assert res.stats.n_raw_events == 0


def test_projection_artifact_is_removed_by_path_verification() -> None:
    # A heads north toward a parked car, then turns east 8 m before it: linear projection says TTC 0.75 s,
    # but A never comes near the car.
    t = np.round(np.arange(0.0, 4.0001, 0.1), 6)
    xy = np.where(t[:, None] <= 0.4, np.column_stack([0 * t, 10 * t]), np.column_stack([10 * (t - 0.4), 0 * t + 4.0]))
    a = path_track(1, t, xy.astype(float))
    a.loc[a["t"] <= 0.4, "heading"] = np.pi / 2
    a.loc[a["t"] > 0.4, "heading"] = 0.0
    a["speed"] = 10.0
    b = track(2, 0.0, 4.0, 0.0, 12.0, 0.0, 0.0)
    b["heading"] = 0.0
    df = pd.concat([a, b], ignore_index=True)

    strict = ConflictDetector().detect(df)
    assert strict.events.empty and strict.stats.n_unverified == 1

    lenient_cfg = EngineConfig(detector=DetectorConfig(require_path_verification=False))
    lenient = ConflictDetector(lenient_cfg).detect(df)
    assert len(lenient.events) == 1 and not bool(lenient.events.iloc[0]["path_verified"])


def test_backends_give_identical_events(crossing_near_miss: pd.DataFrame, rear_end_brake: pd.DataFrame) -> None:
    rear = rear_end_brake.copy()
    rear["vid"] += 10
    df = pd.concat([crossing_near_miss, rear], ignore_index=True)
    outs = []
    for backend in ("brute", "grid", "kdtree", "rtree"):
        cfg = EngineConfig(detector=DetectorConfig(spatial_backend=backend))
        outs.append(ConflictDetector(cfg).detect(df).events[["vid_a", "vid_b", "ttc", "pet"]])
    for other in outs[1:]:
        pd.testing.assert_frame_equal(outs[0], other)


def test_ttc_threshold_controls_detection(crossing_near_miss: pd.DataFrame) -> None:
    cfg = EngineConfig(thresholds=ConflictThresholds(max_ttc_s=0.2, severe_ttc_s=0.1))
    assert ConflictDetector(cfg).detect(crossing_near_miss).events.empty


def test_speed_filter(rear_end_brake: pd.DataFrame) -> None:
    cfg = EngineConfig(thresholds=ConflictThresholds(min_max_speed_mps=30.0))
    res = ConflictDetector(cfg).detect(rear_end_brake)
    assert res.events.empty and res.stats.n_speed_filtered == 1


def test_invalid_input_raises() -> None:
    bad = track(1, 0, 1, 0, 0, 1, 0)
    with pytest.raises(TrajectoryFormatError):
        ConflictDetector().detect(bad.drop(columns=["heading"]))
    dup = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
    with pytest.raises(TrajectoryFormatError):
        ConflictDetector().detect(dup)


def test_empty_input() -> None:
    empty = track(1, 0, 1, 0, 0, 1, 0).iloc[0:0]
    res = ConflictDetector().detect(empty)
    assert res.events.empty and res.stats.n_rows == 0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ConflictThresholds(rear_end_angle_deg=60, crossing_angle_deg=50)
    with pytest.raises(ValueError):
        EngineConfig.model_validate({"thresholds": {"max_ttc": 1.5}})  # typo -> extra key rejected
