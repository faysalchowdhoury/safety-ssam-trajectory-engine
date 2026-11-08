"""Command-line interface: ``ssam-engine <command> [options]``.

Commands
--------
simulate     run the synthetic intersection simulator for one scenario/seed
analyze      detect conflicts in one or more vehicle records
study        multi-seed Baseline-vs-Alternative study with report and figures
bench-index  benchmark spatial index backends
bench-api    benchmark API query latency (in-process)
serve        start the FastAPI service
vissim-run   batch-run PTV Vissim seeds over COM (Windows + licence)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import EngineConfig
from .core.exceptions import SSAMEngineError
from .core.logging import configure_logging, get_logger

log = get_logger("ssam_engine.cli")


def _engine(path: str | None, backend: str | None) -> EngineConfig:
    cfg = EngineConfig.from_file(path) if path else EngineConfig()
    if backend:
        cfg = cfg.model_copy(update={"detector": cfg.detector.model_copy(update={"spatial_backend": backend})})
        EngineConfig.model_validate(cfg.model_dump())
    return cfg


def cmd_simulate(a: argparse.Namespace) -> int:
    from .sim.engine import IntersectionSimulator
    from .sim.scenario import SyntheticScenario

    sc = SyntheticScenario.from_file(a.scenario)
    out = IntersectionSimulator(sc, a.seed).run(a.out, a.stem, compress=not a.no_compress)
    print(json.dumps({"fzp": str(out.fzp_path), "vehicles": out.n_vehicles, "records": out.n_records,
                      "runtime_s": round(out.runtime_s, 2)}, indent=2))
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from .services.heatmap import render_heatmaps
    from .services.pipeline import analyze_file, event_summary

    cfg = _engine(a.config, a.backend)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = {}
    report = []
    for f in a.inputs:
        res = analyze_file(f, cfg)
        name = Path(f).name.split(".")[0]
        res.events.to_csv(out / f"{name}_conflicts.csv", index=False)
        summary = {"file": str(f), **event_summary(res.events), "stats": res.stats.model_dump()}
        report.append(summary)
        panels[name] = res.events
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if a.heatmap:
        render_heatmaps(panels, out / "conflict_heatmaps.png", cell_m=a.cell, max_ttc=cfg.thresholds.max_ttc_s,
                        extent_m=a.extent)
    print(json.dumps([{k: v for k, v in r.items() if k != "stats"} for r in report], indent=2, default=float))
    return 0


def cmd_study(a: argparse.Namespace) -> int:
    from .services.pipeline import run_study
    from .services.report import write_report

    result = run_study(a.study, a.out, workers=a.workers, reuse=not a.no_reuse, log_level=a.worker_log_level)
    md = write_report(result)
    print(md.read_text(encoding="utf-8"))
    return 0


def cmd_bench_index(a: argparse.Namespace) -> int:
    from .services.bench import bench_pair_finders

    df = bench_pair_finders(a.sizes, repeats=a.repeats)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    return 0


def cmd_bench_api(a: argparse.Namespace) -> int:
    from .services.bench import bench_api

    res = bench_api(Path(a.fzp), queries=a.queries)
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ssam_engine.api.app:app_from_env", factory=True, host=a.host, port=a.port, workers=a.workers,
                log_level="warning")
    return 0


def cmd_vissim(a: argparse.Namespace) -> int:
    from .vissim.com_runner import VissimBatchRunner, VissimRunSpec

    behaviours = json.loads(a.driving_behaviors) if a.driving_behaviors else {}
    spec = VissimRunSpec(
        network=Path(a.network), seeds=tuple(a.seeds), period_s=a.period, warmup_s=a.warmup,
        resolution=a.resolution, driving_behaviors={int(k): v for k, v in behaviours.items()}, prog_id=a.prog_id,
    )
    with VissimBatchRunner(spec) as runner:
        paths = runner.run(a.out)
    print(json.dumps([str(p) for p in paths], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ssam-engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--plain-logs", action="store_true", help="human-readable logs instead of JSON lines")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="run the synthetic intersection simulator")
    s.add_argument("--scenario", required=True)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default="data")
    s.add_argument("--stem")
    s.add_argument("--no-compress", action="store_true")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("analyze", help="detect conflicts in vehicle records")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--config")
    s.add_argument("--backend", choices=["kdtree", "rtree", "grid", "brute"])
    s.add_argument("--out", default="results/analyze")
    s.add_argument("--heatmap", action="store_true")
    s.add_argument("--cell", type=float, default=2.0)
    s.add_argument("--extent", type=float, default=40.0)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("study", help="multi-seed baseline vs alternative study")
    s.add_argument("study")
    s.add_argument("--out", default="results")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--no-reuse", action="store_true", help="re-simulate even if data exists")
    s.add_argument("--worker-log-level", default="WARNING")
    s.set_defaults(func=cmd_study)

    s = sub.add_parser("bench-index", help="benchmark spatial backends")
    s.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 1000, 5000])
    s.add_argument("--repeats", type=int, default=7)
    s.add_argument("--out")
    s.set_defaults(func=cmd_bench_index)

    s = sub.add_parser("bench-api", help="benchmark API query latency")
    s.add_argument("--fzp", required=True)
    s.add_argument("--queries", type=int, default=2000)
    s.add_argument("--out")
    s.set_defaults(func=cmd_bench_api)

    s = sub.add_parser("serve", help="start the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("vissim-run", help="batch-run Vissim seeds over COM")
    s.add_argument("--network", required=True)
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--period", type=float, default=4500.0)
    s.add_argument("--warmup", type=float, default=900.0)
    s.add_argument("--resolution", type=int, default=10)
    s.add_argument("--driving-behaviors", help='JSON, e.g. \'{"1": {"W99cc0": 1.5}}\'')
    s.add_argument("--prog-id", default="Vissim.Vissim")
    s.add_argument("--out", default="data/vissim")
    s.set_defaults(func=cmd_vissim)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_output=not args.plain_logs)
    try:
        return int(args.func(args))
    except SSAMEngineError as exc:
        log.error("cli.failed", extra={"error": str(exc), "type": type(exc).__name__})
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
