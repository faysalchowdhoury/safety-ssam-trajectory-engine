"""Multi-seed PTV Vissim batch runner over the COM interface (``win32com.client``).

For each random seed the runner loads the network, applies optional Wiedemann
driving-behaviour overrides, enables the vehicle record, runs the simulation in
Quick Mode and moves the resulting ``.fzp`` to ``<out_dir>/<stem>_seed<N>.fzp``.

Requirements: Windows, a licensed PTV Vissim installation registered for COM, and
``pywin32``. The vehicle-record *attribute list* is stored in the ``.inpx`` - set
it once in the GUI (Evaluation > Configuration > Direct Output > Vehicle record >
More...) to at least: SIMSEC, NO, LANE\\LINK\\NO, LANE\\INDEX, SPEED, ACCELERATION,
COORDFRONTX, COORDFRONTY, COORDREARX, COORDREARY, LENGTH, WIDTH, VEHTYPE.

COM attribute names differ slightly between Vissim releases; the names used
here are collected in :data:`ATTR` so they can be adjusted in one place after
checking *Help > COM API* for your version. This module is not exercised by the
automated test suite because it needs a Vissim licence.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from ..core.exceptions import VissimAutomationError
from ..core.logging import get_logger

log = get_logger(__name__)

ATTR: Final[dict[str, str]] = {
    "seed": "RandSeed",
    "period": "SimPeriod",
    "resolution": "SimRes",
    "max_speed": "UseMaxSimSpeed",
    "veh_rec_write": "VehRecWriteFile",
    "veh_rec_from": "VehRecFromTime",
    "veh_rec_to": "VehRecToTime",
    "quick_mode": "QuickMode",
}


@dataclass(frozen=True)
class VissimRunSpec:
    """Batch definition.

    Attributes:
        network: path to the ``.inpx``.
        seeds: random seeds (FHWA guidance: enough seeds to reach the target CI; >= 5 typical).
        period_s: simulation period including warm-up.
        warmup_s: vehicle record starts at this simulation second.
        resolution: time steps per simulation second (10 recommended for SSAM-type analysis).
        driving_behaviors: ``{behaviour_no: {attribute: value}}``, e.g.
            ``{1: {"W99cc0": 1.5, "W99cc1Distr": 2}}``.
        prog_id: COM ProgID, e.g. ``"Vissim.Vissim"`` or ``"Vissim.Vissim.2024"``.
    """

    network: Path
    seeds: tuple[int, ...]
    period_s: float = 4500.0
    warmup_s: float = 900.0
    resolution: int = 10
    driving_behaviors: dict[int, dict[str, Any]] = field(default_factory=dict)
    prog_id: str = "Vissim.Vissim"


class VissimBatchRunner:
    """Context-managed Vissim COM session."""

    def __init__(self, spec: VissimRunSpec) -> None:
        if sys.platform != "win32":
            raise VissimAutomationError("Vissim COM automation requires Windows")
        if not spec.network.is_file():
            raise VissimAutomationError("network file not found", context={"network": str(spec.network)})
        if spec.warmup_s >= spec.period_s:
            raise VissimAutomationError("warm-up must be shorter than the simulation period")
        self.spec = spec
        self._vissim: Any = None

    def __enter__(self) -> "VissimBatchRunner":
        try:
            import win32com.client as com
        except ImportError as exc:
            raise VissimAutomationError("pywin32 is not installed: pip install pywin32") from exc
        try:
            self._vissim = com.Dispatch(self.spec.prog_id)
        except Exception as exc:  # noqa: BLE001 - pywintypes.com_error is not importable everywhere
            raise VissimAutomationError(f"cannot start Vissim via COM: {exc}", context={"prog_id": self.spec.prog_id}) from exc
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        if self._vissim is not None:
            try:
                self._vissim.Exit()
            except Exception:  # noqa: BLE001
                log.warning("vissim.exit_failed")
            self._vissim = None

    def _set(self, obj: Any, attr: str, value: Any) -> None:
        try:
            obj.SetAttValue(attr, value)
        except Exception as exc:  # noqa: BLE001
            raise VissimAutomationError(f"SetAttValue({attr!r}) failed: {exc}", context={"value": value}) from exc

    def run(self, out_dir: str | Path, stem: str | None = None) -> list[Path]:
        """Run every seed and return the collected vehicle-record paths."""
        if self._vissim is None:
            raise VissimAutomationError("use VissimBatchRunner as a context manager")
        spec = self.spec
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or spec.network.stem
        net_dir = spec.network.parent
        collected: list[Path] = []
        for seed in spec.seeds:
            started = time.time()
            vis = self._vissim
            try:
                vis.LoadNet(str(spec.network.resolve()), False)
            except Exception as exc:  # noqa: BLE001
                raise VissimAutomationError(f"LoadNet failed: {exc}", context={"network": str(spec.network)}) from exc
            for no, attrs in spec.driving_behaviors.items():
                try:
                    behaviour = vis.Net.DrivingBehaviors.ItemByKey(int(no))
                except Exception as exc:  # noqa: BLE001
                    raise VissimAutomationError("driving behaviour not found", context={"no": no}) from exc
                for attr, value in attrs.items():
                    self._set(behaviour, attr, value)
            sim = vis.Simulation
            self._set(sim, ATTR["seed"], int(seed))
            self._set(sim, ATTR["period"], float(spec.period_s))
            self._set(sim, ATTR["resolution"], int(spec.resolution))
            self._set(sim, ATTR["max_speed"], True)
            ev = vis.Evaluation
            self._set(ev, ATTR["veh_rec_write"], True)
            self._set(ev, ATTR["veh_rec_from"], float(spec.warmup_s))
            self._set(ev, ATTR["veh_rec_to"], float(spec.period_s))
            try:
                vis.Graphics.CurrentNetworkWindow.SetAttValue(ATTR["quick_mode"], 1)
            except Exception:  # noqa: BLE001 - headless sessions have no network window
                pass
            log.info("vissim.run_start", extra={"seed": seed, "network": spec.network.name})
            try:
                sim.RunContinuous()
            except Exception as exc:  # noqa: BLE001
                raise VissimAutomationError(f"simulation failed: {exc}", context={"seed": seed}) from exc
            produced = sorted(
                (p for p in net_dir.glob("*.fzp") if p.stat().st_mtime >= started - 1.0),
                key=lambda p: p.stat().st_mtime,
            )
            if not produced:
                raise VissimAutomationError(
                    "no .fzp written; enable the vehicle record and its attributes in the .inpx",
                    context={"seed": seed, "dir": str(net_dir)},
                )
            target = out / f"{stem}_seed{seed}.fzp"
            shutil.move(str(produced[-1]), target)
            collected.append(target)
            log.info("vissim.run_done", extra={"seed": seed, "fzp": str(target), "wall_s": round(time.time() - started, 1)})
        return collected
