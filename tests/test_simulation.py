from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ssam_engine.io.fzp import read_fzp
from ssam_engine.sim.layout import IntersectionGeometry, build_movement_paths, find_conflicts
from ssam_engine.sim.scenario import SyntheticScenario
from ssam_engine.sim.signals import RingBarrierPlan
from ssam_engine.sim.engine import IntersectionSimulator

ROOT = Path(__file__).resolve().parents[1]


def _plan(**over) -> dict:
    base = {
        "ring1": [[1, 2], [3, 4]],
        "ring2": [[5, 6], [7, 8]],
        "timings": {p: {"green_s": 10 if p % 2 else 30, "yellow_s": 3 if p % 2 else 4,
                        "red_clearance_s": 1 if p % 2 else 2} for p in range(1, 9)},
    }
    base.update(over)
    return base


def test_ring_barrier_states() -> None:
    plan = RingBarrierPlan.model_validate(_plan())
    assert plan.cycle_s == pytest.approx(2 * (14 + 36))
    assert plan.phase_state(1, 0.0) == "G"
    assert plan.phase_state(1, 10.5) == "Y"
    assert plan.phase_state(1, 13.5) == "R"
    assert plan.phase_state(2, 14.0) == "G"
    assert plan.phase_state(4, 50.0 + 14.0) == "G"
    assert plan.phase_state(2, 100.0) == "R"  # next cycle starts with phase 1
    assert plan.phase_state(2, 114.0) == "G"


def test_yellow_onset_key_changes_every_change_interval() -> None:
    plan = RingBarrierPlan.model_validate(_plan())
    # phase 3 greens at 50-60 s, yellow from 60 s; a red seen at t=20 (before green) belongs to the
    # *previous* cycle's change interval, so it must not be reused at this cycle's yellow.
    assert plan.last_yellow_onset(3, 20.0) == pytest.approx(60.0 - 100.0)
    assert plan.last_yellow_onset(3, 61.0) == pytest.approx(60.0)
    assert plan.last_yellow_onset(3, 99.0) == pytest.approx(60.0)
    assert plan.last_yellow_onset(3, 161.0) == pytest.approx(160.0)


def test_ring_barrier_rejects_unequal_barriers() -> None:
    p = _plan()
    p["timings"][1] = {"green_s": 20, "yellow_s": 3, "red_clearance_s": 1}
    with pytest.raises(ValidationError):
        RingBarrierPlan.model_validate(p)
    with pytest.raises(ValidationError):
        RingBarrierPlan.model_validate(_plan(ring1=[[3, 2], [1, 4]]))


def test_fya_left_turn_indications() -> None:
    sc = SyntheticScenario.from_file(ROOT / "configs" / "scenarios" / "alternative_fya.yaml")
    sim = IntersectionSimulator(sc, 1)
    ebl = sim.mv["EBL"]
    # phi5 arrow 0-8 G, 8-11 Y, 11-12 all-red; phi2 (FYA) green 12-56, yellow 56-60, red after
    expected = {0.5: "G", 9.0: "Y", 11.5: "R", 20.0: "P", 57.0: "Y", 61.0: "R", 110.5: "G"}
    for t, st in expected.items():
        assert sim._state(ebl, t) == st, t
    assert sim.mv["EBL"].opposing is not None and sim.mv["EBL"].opposing[0].name == "WBT"


def test_geometry_conflicts_are_physical() -> None:
    paths = build_movement_paths(IntersectionGeometry(), 15.6, 8.5)
    conflicts = find_conflicts(paths)
    names = {o for o, _, _ in conflicts["EBL"]}
    assert "WBT" in names and "WBL" not in names and "NBT" not in names
    s_self, s_other = next((a, b) for o, a, b in conflicts["EBL"] if o == "WBT")
    assert paths["EBL"].line.point(s_self)[0] == pytest.approx(paths["WBT"].line.point(s_other)[0], abs=1e-6)
    assert paths["EBT"].line.length == pytest.approx(200 + 24 + 120)


def test_short_simulation_runs_and_is_reproducible(tmp_path: Path) -> None:
    sc = SyntheticScenario.from_file(ROOT / "configs" / "scenarios" / "baseline_permissive.yaml")
    sc = sc.model_copy(update={"duration_s": 260.0, "warmup_s": 60.0})
    o1 = IntersectionSimulator(sc, 5).run(tmp_path / "a", "s")
    o2 = IntersectionSimulator(sc, 5).run(tmp_path / "b", "s")
    d1, d2 = read_fzp(o1.fzp_path), read_fzp(o2.fzp_path)
    pd.testing.assert_frame_equal(d1, d2)
    assert d1["t"].min() > 60.0 and d1["t"].max() <= 260.0 + 1e-6
    assert (d1["speed"] <= 15.6 * 1.25 + 1e-6).all()
    veh = pd.read_csv(o1.vehicles_path)
    assert (veh["delay_s"] > -1.0).all()
    mov = pd.read_csv(o1.movements_path)
    assert set(mov["movement"]) == {"EBT", "EBL", "NBT", "NBL", "WBT", "WBL", "SBT", "SBL"}


def test_no_vehicle_that_arrived_on_red_enters(tmp_path: Path) -> None:
    """Red entries may only come from drivers who chose 'go' at yellow onset (dilemma zone).

    Such drivers clear the stop line within the first moments of red; a vehicle that was
    far upstream when the light turned red must never enter.
    """
    sc = SyntheticScenario.from_file(ROOT / "configs" / "scenarios" / "alternative_protected.yaml")
    sc = sc.model_copy(update={"duration_s": 400.0, "warmup_s": 0.0})
    out = IntersectionSimulator(sc, 9).run(tmp_path, "r", compress=False)
    df = read_fzp(out.fzp_path)
    paths = build_movement_paths(sc.geometry, sc.approach_speed_mps, sc.left_turn_speed_mps)
    inv = {p.link_no: p for p in paths.values()}
    violations = 0
    for vid, g in df.groupby("vid"):
        p = inv[int(g["link"].iloc[0])]
        phase = sc.control[p.name].phase
        # signed distance of the front bumper past the stop line along the approach direction
        front = g[["x", "y"]].to_numpy() + 0.5 * g["length"].to_numpy()[:, None] * np.column_stack(
            [np.cos(g["heading"]), np.sin(g["heading"])]
        )
        stop = paths[p.name].line.point(paths[p.name].stop_s)[0]
        d0 = paths[p.name].line.point(paths[p.name].stop_s - 1.0)[0]
        u = (stop - d0) / np.linalg.norm(stop - d0)
        past = (front - stop) @ u > 0
        if past.any() and (~past).any():
            t_cross = g["t"].to_numpy()[np.argmax(past)]
            state = sc.signal.phase_state(phase, t_cross - 0.1)
            if state == "R":
                red_onset = sc.signal.last_yellow_onset(phase, t_cross) + sc.signal.timings[phase].yellow_s
                if t_cross - red_onset > 1.5:
                    violations += 1
    assert len(df) > 0
    assert violations == 0
