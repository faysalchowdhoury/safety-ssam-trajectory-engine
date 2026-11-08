from __future__ import annotations

import math

import numpy as np
import pytest

from ssam_engine.core.hcm import geh, geh_pass_rate, los_from_delay
from ssam_engine.services.statistics import ks_test, pct_change, required_replications, summarize, welch_test


@pytest.mark.parametrize(
    "delay,expected", [(0, "A"), (10, "A"), (10.01, "B"), (35, "C"), (55, "D"), (80, "E"), (80.1, "F")]
)
def test_signalized_los_boundaries(delay: float, expected: str) -> None:
    assert los_from_delay(delay) == expected


def test_los_vc_override_and_unsignalized() -> None:
    assert los_from_delay(5.0, v_c_ratio=1.05) == "F"
    assert los_from_delay(14.0, control="unsignalized") == "B"
    assert los_from_delay(float("nan")) == "n/a"


def test_geh_known_values() -> None:
    assert geh(100, 100) == 0.0
    assert geh(1000, 900) == pytest.approx(math.sqrt(2 * 100**2 / 1900))
    assert geh(0, 0) == 0.0
    assert geh_pass_rate(np.array([100, 1000, 500]), np.array([100, 800, 500])) == pytest.approx(2 / 3)


def test_summary_and_ci() -> None:
    s = summarize([10, 12, 14, 16, 18])
    assert s["mean"] == 14 and s["n"] == 5
    assert s["ci"] == pytest.approx(2.776 * np.std([10, 12, 14, 16, 18], ddof=1) / math.sqrt(5), rel=1e-3)


def test_required_replications_monotonic() -> None:
    tight = required_replications([100, 101, 99, 100, 102], tolerance=0.05)
    loose = required_replications([60, 140, 90, 120, 80], tolerance=0.05)
    assert tight <= 5 < loose


def test_tests_detect_difference() -> None:
    t, p = welch_test([10, 11, 12, 10, 11], [20, 21, 19, 22, 20])
    assert p < 0.001
    d, p2 = ks_test(np.linspace(0, 1, 200), np.linspace(0.5, 1.5, 200))
    assert d > 0.4 and p2 < 0.001
    assert pct_change(50, 25) == -50.0
    assert math.isnan(pct_change(0, 3))
