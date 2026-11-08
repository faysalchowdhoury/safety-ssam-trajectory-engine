from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssam_engine.core.config import UnitsConfig
from ssam_engine.core.exceptions import TrajectoryFormatError
from ssam_engine.io.fzp import read_fzp, read_fzp_header, write_fzp

from .conftest import to_vissim_records


def test_roundtrip_metric(tmp_path: Path, crossing_near_miss: pd.DataFrame) -> None:
    path = write_fzp(to_vissim_records(crossing_near_miss), tmp_path / "a.fzp")
    df = read_fzp(path)
    ref = crossing_near_miss.sort_values(["t", "vid"]).reset_index(drop=True)
    assert len(df) == len(ref)
    np.testing.assert_allclose(df["x"], ref["x"], atol=2e-3)
    np.testing.assert_allclose(df["speed"], ref["speed"], atol=2e-3)
    np.testing.assert_allclose(np.cos(df["heading"]), np.cos(ref["heading"]), atol=1e-6)


def test_gzip_roundtrip(tmp_path: Path, crossing_near_miss: pd.DataFrame) -> None:
    path = write_fzp(to_vissim_records(crossing_near_miss), tmp_path / "a.fzp.gz")
    assert len(read_fzp(path)) == len(crossing_near_miss)


def test_imperial_units(tmp_path: Path) -> None:
    text = (
        "$VISION\n* Comment\n$VEHICLE:SIMSEC;NO;SPEED;COORDFRONTX;COORDFRONTY;COORDREARX;COORDREARY;LENGTH;\n"
        "0.10;1;22.37;16.40;0.00;1.64;0.00;14.76;\n"
        "0.20;1;22.37;19.68;0.00;4.92;0.00;14.76;\n"
    )
    p = tmp_path / "ft.fzp"
    p.write_text(text)
    df = read_fzp(p, UnitsConfig(length="ft", speed="mph"))
    assert df["speed"].iloc[0] == pytest.approx(10.0, abs=0.01)
    assert df["length"].iloc[0] == pytest.approx(4.4988, abs=1e-3)
    assert df["width"].iloc[0] == pytest.approx(1.8)


def test_combined_coordinate_columns(tmp_path: Path) -> None:
    text = (
        "$VEHICLE:SIMSEC;NO;SPEED;COORDFRONT;COORDREAR\n"
        "0.1;7;36.0;5.0 2.0 0.0;0.5 2.0 0.0\n"
        "0.2;7;36.0;6.0 2.0 0.0;1.5 2.0 0.0\n"
    )
    p = tmp_path / "c.fzp"
    p.write_text(text)
    df = read_fzp(p)
    assert df["x"].iloc[0] == pytest.approx(2.75)
    assert df["speed"].iloc[0] == pytest.approx(10.0)
    assert (df["accel"] == 0).all()


def test_header_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.fzp"
    p.write_text("* no header here\n1;2;3\n")
    with pytest.raises(TrajectoryFormatError):
        read_fzp_header(p)
    p2 = tmp_path / "missing.fzp"
    p2.write_text("$VEHICLE:SIMSEC;NO;SPEED\n0.1;1;10\n")
    with pytest.raises(TrajectoryFormatError):
        read_fzp(p2)
    with pytest.raises(TrajectoryFormatError):
        read_fzp(tmp_path / "does_not_exist.fzp")
