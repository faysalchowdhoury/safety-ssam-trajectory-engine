"""Unit conversion factors to SI (metres, metres/second, metres/second^2).

Vissim writes coordinates in the network's length unit (m or ft) and speeds in
km/h or mph depending on the project's unit settings.
"""

from __future__ import annotations

from typing import Final

LENGTH_TO_M: Final[dict[str, float]] = {"m": 1.0, "ft": 0.3048}
SPEED_TO_MPS: Final[dict[str, float]] = {
    "kmh": 1.0 / 3.6,
    "mph": 0.44704,
    "mps": 1.0,
    "fps": 0.3048,
}
ACCEL_TO_MPS2: Final[dict[str, float]] = {"mps2": 1.0, "fps2": 0.3048}

MPS_TO_KMH: Final[float] = 3.6
MPS_TO_MPH: Final[float] = 1.0 / 0.44704
