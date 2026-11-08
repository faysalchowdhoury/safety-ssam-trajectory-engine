"""PTV Vissim vehicle-record (``.fzp``) reader and writer.

A vehicle record is a semicolon-separated attribute file. Comment lines start
with ``*``, metadata lines with ``$``, and the column header is the line that
starts with ``$VEHICLE:``, for example::

    $VEHICLE:SIMSEC;NO;LANE\\LINK\\NO;LANE\\INDEX;SPEED;ACCELERATION;COORDFRONTX;COORDFRONTY;COORDREARX;COORDREARY;LENGTH;WIDTH;VEHTYPE

Configure these attributes in Vissim under *Evaluation > Configuration > Direct
Output > Vehicle record > More...*. Newer Vissim versions can also export the
combined ``COORDFRONT`` / ``COORDREAR`` attributes (``"x y z"``); both layouts are
supported. Files ending in ``.gz`` are read and written transparently.
"""

from __future__ import annotations

import gzip
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Final

import numpy as np
import pandas as pd

from ..core.config import UnitsConfig
from ..core.exceptions import TrajectoryFormatError
from ..core.logging import get_logger
from ..core.schema import validate_trajectories
from ..core.units import ACCEL_TO_MPS2, LENGTH_TO_M, SPEED_TO_MPS

log = get_logger(__name__)

HEADER_TOKEN: Final[str] = "$VEHICLE:"
MAX_HEADER_LINES: Final[int] = 5_000

ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "t": ("SIMSEC", "SIMSECOND"),
    "vid": ("NO", "VEHNO", "VEHICLE\\NO"),
    "front_x": ("COORDFRONTX",),
    "front_y": ("COORDFRONTY",),
    "rear_x": ("COORDREARX",),
    "rear_y": ("COORDREARY",),
    "front": ("COORDFRONT",),
    "rear": ("COORDREAR",),
    "speed": ("SPEED",),
    "accel": ("ACCELERATION", "ACCEL"),
    "length": ("LENGTH",),
    "width": ("WIDTH",),
    "link": ("LANE\\LINK\\NO", "LANE\\LINK", "LINK"),
    "lane": ("LANE\\INDEX", "LANEINDEX"),
    "vtype": ("VEHTYPE", "VEHTYPE\\NO"),
}

ATTRIBUTE_LEGEND: Final[dict[str, str]] = {
    "SIMSEC": "Simulation second",
    "NO": "Number",
    "LANE\\LINK\\NO": "Lane\\Link\\Number",
    "LANE\\INDEX": "Lane\\Index",
    "SPEED": "Speed [km/h]",
    "ACCELERATION": "Acceleration [m/s2]",
    "COORDFRONTX": "Coordinate front x [m]",
    "COORDFRONTY": "Coordinate front y [m]",
    "COORDREARX": "Coordinate rear x [m]",
    "COORDREARY": "Coordinate rear y [m]",
    "LENGTH": "Length [m]",
    "WIDTH": "Width [m]",
    "VEHTYPE": "Vehicle type",
}


def _open_text(path: Path, mode: str = "rt") -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode, encoding="utf-8", errors="replace", newline="")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8", errors="replace", newline="")


def read_fzp_header(path: str | Path) -> tuple[int, list[str]]:
    """Locate the ``$VEHICLE:`` header.

    Returns:
        ``(line_index, columns)`` with upper-case column names.
    """
    p = Path(path)
    if not p.is_file():
        raise TrajectoryFormatError("vehicle record not found", context={"path": str(p)})
    with _open_text(p) as fh:
        for idx, line in enumerate(fh):
            if idx > MAX_HEADER_LINES:
                break
            stripped = line.lstrip("﻿").strip()
            if stripped.upper().startswith(HEADER_TOKEN):
                cols = [c.strip().upper() for c in stripped.split(":", 1)[1].split(";")]
                while cols and cols[-1] == "":
                    cols.pop()
                if not cols:
                    break
                return idx, cols
    raise TrajectoryFormatError(
        "no '$VEHICLE:' header found; is this a Vissim vehicle record?", context={"path": str(p)}
    )


def _pick(columns: list[str], field: str) -> str | None:
    for alias in ALIASES[field]:
        if alias in columns:
            return alias
    return None


def _split_coord(series: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    parts = series.astype(str).str.strip().str.split(r"\s+", expand=True)
    if parts.shape[1] < 2:
        raise TrajectoryFormatError(f"{name} must contain 'x y [z]' values")
    x = pd.to_numeric(parts[0], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(parts[1], errors="coerce").to_numpy(np.float64)
    return x, y


def read_fzp(
    path: str | Path,
    units: UnitsConfig | None = None,
    *,
    decimal: str = ".",
    default_width_m: float = 1.8,
) -> pd.DataFrame:
    """Read a Vissim ``.fzp`` vehicle record into the canonical SI trajectory schema.

    Args:
        path: ``.fzp`` or ``.fzp.gz`` file.
        units: units of the file (defaults: metres, km/h, m/s^2).
        decimal: decimal separator used by the exporting machine's locale.
        default_width_m: width used when the record has no ``WIDTH`` column.

    Raises:
        TrajectoryFormatError: on missing header/columns or unparsable values.
    """
    units = units or UnitsConfig()
    p = Path(path)
    header_idx, columns = read_fzp_header(p)
    try:
        raw = pd.read_csv(
            p,
            sep=";",
            skiprows=header_idx + 1,
            header=None,
            usecols=list(range(len(columns))),
            skipinitialspace=True,
            decimal=decimal,
            compression="infer",
            comment=None,
            engine="c",
            low_memory=False,
        )
    except (pd.errors.ParserError, ValueError, OSError) as exc:
        raise TrajectoryFormatError(f"cannot parse vehicle record body: {exc}", context={"path": str(p)}) from exc
    raw.columns = columns

    raw = raw[~raw[columns[0]].astype(str).str.startswith(("*", "$"))]
    if raw.empty:
        raise TrajectoryFormatError("vehicle record contains no data rows", context={"path": str(p)})

    def num(col: str) -> np.ndarray:
        return pd.to_numeric(raw[col], errors="coerce").to_numpy(np.float64)

    required = {"t", "vid", "speed"}
    picked = {f: _pick(columns, f) for f in ALIASES}
    missing = [f for f in required if picked[f] is None]
    if missing:
        raise TrajectoryFormatError(
            "vehicle record lacks required attributes",
            context={"missing": [ALIASES[f][0] for f in missing], "columns": columns},
        )

    if picked["front_x"] and picked["front_y"] and picked["rear_x"] and picked["rear_y"]:
        fx, fy = num(picked["front_x"]), num(picked["front_y"])  # type: ignore[arg-type]
        rx, ry = num(picked["rear_x"]), num(picked["rear_y"])  # type: ignore[arg-type]
    elif picked["front"] and picked["rear"]:
        fx, fy = _split_coord(raw[picked["front"]], "COORDFRONT")
        rx, ry = _split_coord(raw[picked["rear"]], "COORDREAR")
    else:
        raise TrajectoryFormatError(
            "vehicle record needs COORDFRONTX/Y + COORDREARX/Y (or COORDFRONT + COORDREAR)",
            context={"columns": columns},
        )

    lf = LENGTH_TO_M[units.length]
    fx, fy, rx, ry = fx * lf, fy * lf, rx * lf, ry * lf
    dx, dy = fx - rx, fy - ry
    chord = np.hypot(dx, dy)
    heading = np.where(chord > 1e-6, np.arctan2(dy, dx), np.nan)
    length = num(picked["length"]) * lf if picked["length"] else chord
    width = num(picked["width"]) * lf if picked["width"] else np.full(len(raw), default_width_m)

    out = pd.DataFrame(
        {
            "t": num(picked["t"]),  # type: ignore[arg-type]
            "vid": num(picked["vid"]),  # type: ignore[arg-type]
            "x": 0.5 * (fx + rx),
            "y": 0.5 * (fy + ry),
            "heading": heading,
            "speed": num(picked["speed"]) * SPEED_TO_MPS[units.speed],  # type: ignore[arg-type]
            "length": np.where(np.isfinite(length) & (length > 0), length, np.maximum(chord, 0.5)),
            "width": width,
        }
    )
    if picked["accel"]:
        out["accel"] = num(picked["accel"]) * ACCEL_TO_MPS2[units.acceleration]
    for field in ("link", "lane", "vtype"):
        col = picked[field]
        if col:
            out[field] = pd.to_numeric(raw[col], errors="coerce").fillna(-1).to_numpy(np.int64)

    if out["heading"].isna().any():
        out = _fill_heading_from_motion(out)
    out["speed"] = out["speed"].clip(lower=0.0)

    df = validate_trajectories(out, default_width=default_width_m)
    log.info(
        "fzp.read",
        extra={"path": str(p), "rows": len(df), "vehicles": int(df["vid"].nunique()), "columns": len(columns)},
    )
    return df


def _fill_heading_from_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Replace undefined headings (zero-length chord) with the direction of motion."""
    ordered = df.sort_values(["vid", "t"], kind="mergesort")
    g = ordered.groupby("vid", sort=False)
    motion = np.arctan2(g["y"].diff().to_numpy(), g["x"].diff().to_numpy())
    motion = pd.Series(np.where(np.isfinite(motion), motion, np.nan), index=ordered.index)
    motion = motion.groupby(ordered["vid"], sort=False).bfill().groupby(ordered["vid"], sort=False).ffill()
    filled = ordered["heading"].fillna(motion).fillna(0.0)
    df = df.copy()
    df["heading"] = filled.reindex(df.index)
    return df


def write_fzp(
    records: pd.DataFrame,
    path: str | Path,
    *,
    comment: str = "",
    application: str = "safety-ssam-trajectory-engine",
    float_format: str = "%.3f",
) -> Path:
    """Write a Vissim-style vehicle record.

    Args:
        records: frame whose column names are Vissim attribute names
            (e.g. ``SIMSEC``, ``NO``, ``COORDFRONTX`` ...), already in file units.
        path: output path; ``.gz`` suffix enables gzip compression.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [str(c).upper() for c in records.columns]
    header = [
        "$VISION",
        f"* File: {p.name}",
        f"* Comment: {comment}",
        f"* Date: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"* Application: {application}",
        "*",
        "* Table: Vehicles In Network",
        "*",
    ]
    header += [f"* {c}: {ATTRIBUTE_LEGEND.get(c, c)}" for c in cols]
    header += ["*", HEADER_TOKEN + ";".join(cols)]
    buf = io.StringIO()
    records.to_csv(buf, sep=";", header=False, index=False, float_format=float_format, lineterminator="\n")
    with _open_text(p, "wt") as fh:
        fh.write("\n".join(header) + "\n")
        fh.write(buf.getvalue())
    return p
