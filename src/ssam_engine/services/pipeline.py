"""Analysis pipeline: single files and multi-seed Baseline-vs-Alternative studies."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from ..core.config import EngineConfig, StrictModel, load_model_file
from ..core.detector import ConflictDetector, DetectionResult
from ..core.hcm import geh_pass_rate, los_from_delay
from ..core.logging import configure_logging, get_logger
from ..core.models import ConflictType, DetectionStats
from ..io.fzp import read_fzp
from ..sim.engine import IntersectionSimulator
from ..sim.scenario import SyntheticScenario

log = get_logger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    source: Path
    events: pd.DataFrame
    stats: DetectionStats


def analyze_file(path: str | Path, config: EngineConfig | None = None) -> AnalysisResult:
    """Read one vehicle record and detect conflicts."""
    config = config or EngineConfig()
    traj = read_fzp(path, config.units, default_width_m=config.detector.default_vehicle_width_m)
    result: DetectionResult = ConflictDetector(config).detect(traj)
    return AnalysisResult(Path(path), result.events, result.stats)


def event_summary(events: pd.DataFrame) -> dict[str, Any]:
    """Counts by SSAM type plus TTC/PET descriptors.

    Events with TTC = 0 or PET = 0 (footprints overlapped) are reported separately as
    ``collisions`` and excluded from conflict totals: in microsimulation they are model
    artefacts, which is how FHWA SSAM guidance recommends treating them.
    """
    coll = events["collision"].astype(bool) if not events.empty else pd.Series(dtype=bool)
    conf = events[~coll] if not events.empty else events
    counts = conf["conflict_type"].value_counts() if not conf.empty else pd.Series(dtype=int)
    out: dict[str, Any] = {"total": int(len(conf))}
    for ct in ConflictType:
        out[ct.value] = int(counts.get(ct.value, 0))
    out["severe"] = int(conf["severe"].astype(bool).sum()) if not conf.empty else 0
    out["collisions"] = int(coll.sum()) if not events.empty else 0
    out["mean_ttc_s"] = float(conf["ttc"].mean()) if not conf.empty else math.nan
    out["median_pet_s"] = float(conf["pet"].median()) if not conf.empty and conf["pet"].notna().any() else math.nan
    out["mean_max_delta_v_mps"] = float(conf["max_delta_v"].mean()) if not conf.empty else math.nan
    return out


# --------------------------------------------------------------------- study
class StudyScenarioRef(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    label: str
    scenario_file: Path


class StudyConfig(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seeds: list[int] = Field(min_length=2)
    engine_config: Path | None = None
    scenarios: list[StudyScenarioRef] = Field(min_length=2)
    baseline_id: str
    heatmap_cell_m: float = Field(2.0, gt=0.0)
    replication_tolerance: float = Field(0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check(self) -> "StudyConfig":
        ids = [s.id for s in self.scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("scenario ids must be unique")
        if self.baseline_id not in ids:
            raise ValueError("baseline_id must match a scenario id")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


def _seed_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Simulate + analyse one (scenario, seed). Top-level for process-pool pickling."""
    configure_logging(payload.get("log_level", "WARNING"))
    scenario = SyntheticScenario.model_validate(payload["scenario"])
    engine = EngineConfig.model_validate(payload["engine"])
    seed = int(payload["seed"])
    data_dir = Path(payload["data_dir"])
    stem = f"seed{seed}"
    fzp = data_dir / f"{stem}.fzp.gz"
    vehicles_csv = data_dir / f"{stem}_vehicles.csv"
    movements_csv = data_dir / f"{stem}_movements.csv"

    t0 = perf_counter()
    if not (payload.get("reuse", True) and fzp.is_file() and vehicles_csv.is_file() and movements_csv.is_file()):
        IntersectionSimulator(scenario, seed).run(data_dir, stem)
    sim_s = perf_counter() - t0

    analysis = analyze_file(fzp, engine)
    events = analysis.events.copy()
    events.insert(0, "seed", seed)
    events.to_csv(data_dir / f"{stem}_conflicts.csv", index=False)

    veh = pd.read_csv(vehicles_csv)
    lo, hi = scenario.warmup_s, scenario.duration_s
    veh = veh[(veh["arrival_s"] >= lo) & (veh["exit_s"] <= hi)]
    mov = pd.read_csv(movements_csv)
    per_mv = veh.groupby("movement")["delay_s"].mean().to_dict()

    metrics: dict[str, Any] = {"scenario": payload["scenario_id"], "seed": seed}
    metrics.update(event_summary(events))
    metrics["control_delay_s"] = float(veh["delay_s"].mean()) if len(veh) else math.nan
    metrics["vehicles_completed"] = int(len(veh))
    metrics["geh_pass_rate"] = geh_pass_rate(mov["served_vph"].to_numpy(), mov["demand_vph"].to_numpy())
    metrics["max_geh"] = float(mov["geh"].max())
    metrics["unverified_removed"] = analysis.stats.n_unverified
    metrics["raw_events"] = analysis.stats.n_raw_events
    metrics["trajectory_rows"] = analysis.stats.n_rows
    metrics["sim_runtime_s"] = sim_s
    metrics["detect_runtime_s"] = analysis.stats.runtime_s
    metrics["movement_delay_s"] = {k: float(v) for k, v in per_mv.items()}
    metrics["movement_served_vph"] = dict(zip(mov["movement"], mov["served_vph"].astype(float)))
    metrics["conflicts_csv"] = str(data_dir / f"{stem}_conflicts.csv")
    return metrics


@dataclass(frozen=True)
class StudyResult:
    config: StudyConfig
    out_dir: Path
    per_seed: pd.DataFrame
    events: dict[str, pd.DataFrame]
    scenarios: dict[str, SyntheticScenario]
    engine: EngineConfig


def load_study(study_path: str | Path) -> tuple[StudyConfig, dict[str, SyntheticScenario], EngineConfig]:
    """Load a study file; relative paths resolve against the study file's directory."""
    sp = Path(study_path).resolve()
    cfg = load_model_file(sp, StudyConfig)
    base = sp.parent
    scenarios = {
        s.id: SyntheticScenario.from_file(s.scenario_file if s.scenario_file.is_absolute() else base / s.scenario_file)
        for s in cfg.scenarios
    }
    if cfg.engine_config is None:
        engine = EngineConfig()
    else:
        p = cfg.engine_config if cfg.engine_config.is_absolute() else base / cfg.engine_config
        engine = EngineConfig.from_file(p)
    return cfg, scenarios, engine


def run_study(
    study_path: str | Path, out_dir: str | Path, *, workers: int = 2, reuse: bool = True, log_level: str = "WARNING"
) -> StudyResult:
    """Simulate and analyse every (scenario, seed) and collect per-seed metrics."""
    cfg, scenarios, engine = load_study(study_path)
    out = Path(out_dir) / cfg.name
    payloads = []
    for ref in cfg.scenarios:
        data_dir = out / "data" / ref.id
        data_dir.mkdir(parents=True, exist_ok=True)
        for seed in cfg.seeds:
            payloads.append(
                {
                    "scenario_id": ref.id, "scenario": scenarios[ref.id].model_dump(mode="json"),
                    "engine": engine.model_dump(mode="json"), "seed": seed, "data_dir": str(data_dir),
                    "reuse": reuse, "log_level": log_level,
                }
            )
    t0 = perf_counter()
    workers = max(1, min(workers, len(payloads), os.cpu_count() or 1))
    if workers == 1:
        results = [_seed_task(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_seed_task, payloads))
    log.info("study.finished", extra={"study": cfg.name, "runs": len(results), "wall_s": round(perf_counter() - t0, 1)})

    per_seed = pd.DataFrame(results)
    events = {
        ref.id: pd.concat([pd.read_csv(p) for p in per_seed.loc[per_seed["scenario"] == ref.id, "conflicts_csv"]],
                          ignore_index=True)
        for ref in cfg.scenarios
    }
    out.mkdir(parents=True, exist_ok=True)
    flat = per_seed.drop(columns=["movement_delay_s", "movement_served_vph"])
    flat.to_csv(out / "per_seed_metrics.csv", index=False)
    with open(out / "per_seed_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    return StudyResult(cfg, out, per_seed, events, scenarios, engine)


def movement_los_table(result: StudyResult) -> pd.DataFrame:
    """Mean control delay and HCM LOS per movement and scenario (seed-averaged)."""
    rows = []
    for ref in result.config.scenarios:
        sub = result.per_seed[result.per_seed["scenario"] == ref.id]
        delays = pd.DataFrame(list(sub["movement_delay_s"]))
        for mv in delays.columns:
            d = float(np.nanmean(delays[mv]))
            rows.append({"scenario": ref.id, "movement": mv, "delay_s": d, "los": los_from_delay(d)})
    return pd.DataFrame(rows)
