"""Study report: comparison tables (Markdown + JSON) and figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.hcm import los_from_delay
from ..sim.layout import build_movement_paths
from .heatmap import render_heatmaps, render_ttc_cdf
from .pipeline import StudyResult, movement_los_table
from .statistics import ks_test, pct_change, required_replications, summarize, welch_test

METRICS: tuple[tuple[str, str, str], ...] = (
    ("total", "Total conflicts (TTC <= {ttc} s) per 15 min", "{:.1f}"),
    ("crossing", "Crossing conflicts", "{:.1f}"),
    ("rear_end", "Rear-end conflicts", "{:.1f}"),
    ("lane_change", "Lane-change / angle conflicts", "{:.1f}"),
    ("severe", "Severe conflicts (TTC <= {sev} s)", "{:.1f}"),
    ("collisions", "Simulated overlaps (TTC = 0, excluded above)", "{:.1f}"),
    ("mean_ttc_s", "Mean minimum TTC (s)", "{:.2f}"),
    ("median_pet_s", "Median PET (s)", "{:.2f}"),
    ("mean_max_delta_v_mps", "Mean MaxDeltaV (m/s)", "{:.2f}"),
    ("control_delay_s", "Intersection control delay (s/veh)", "{:.1f}"),
)


def _fmt(fmt: str, v: float) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def build_comparison(result: StudyResult) -> dict[str, Any]:
    """Seed statistics per scenario and alternative-vs-baseline tests."""
    cfg = result.config
    ps = result.per_seed
    base = cfg.baseline_id
    out: dict[str, Any] = {"study": cfg.name, "seeds": cfg.seeds, "baseline": base, "scenarios": {}, "tests": {}}
    for ref in cfg.scenarios:
        sub = ps[ps["scenario"] == ref.id]
        entry: dict[str, Any] = {"label": ref.label, "metrics": {}}
        for key, _, _ in METRICS:
            entry["metrics"][key] = summarize(sub[key].astype(float).tolist())
        entry["los"] = los_from_delay(entry["metrics"]["control_delay_s"]["mean"])
        entry["geh_pass_rate_min"] = float(sub["geh_pass_rate"].min())
        entry["max_geh"] = float(sub["max_geh"].max())
        entry["required_replications_total"] = required_replications(
            sub["total"].astype(float).tolist(), tolerance=cfg.replication_tolerance
        )
        entry["unverified_removed_mean"] = float(sub["unverified_removed"].mean())
        entry["raw_events_mean"] = float(sub["raw_events"].mean())
        entry["trajectory_rows_mean"] = float(sub["trajectory_rows"].mean())
        entry["detect_runtime_s_mean"] = float(sub["detect_runtime_s"].mean())
        entry["sim_runtime_s_mean"] = float(sub["sim_runtime_s"].mean())
        out["scenarios"][ref.id] = entry

    b_rows = ps[ps["scenario"] == base]
    for ref in cfg.scenarios:
        if ref.id == base:
            continue
        a_rows = ps[ps["scenario"] == ref.id]
        tests: dict[str, Any] = {}
        for key, _, _ in METRICS:
            t, p = welch_test(b_rows[key].astype(float), a_rows[key].astype(float))
            tests[key] = {
                "pct_change": pct_change(float(b_rows[key].mean()), float(a_rows[key].mean())),
                "welch_t": t,
                "p_value": p,
            }
        d, p = ks_test(result.events[base]["ttc"], result.events[ref.id]["ttc"])
        tests["ttc_distribution_ks"] = {"D": d, "p_value": p}
        out["tests"][ref.id] = tests
    return out


def write_report(result: StudyResult) -> Path:
    """Write ``summary.json``, ``RESULTS.md``, a movement LOS table and figures; return the Markdown path."""
    cfg = result.config
    out = result.out_dir
    fig_dir = out / "figures"
    comp = build_comparison(result)
    th = result.engine.thresholds
    (out / "summary.json").write_text(json.dumps(comp, indent=2, default=float), encoding="utf-8")

    los = movement_los_table(result)
    los.to_csv(out / "movement_los.csv", index=False)

    any_sc = next(iter(result.scenarios.values()))
    paths = build_movement_paths(any_sc.geometry, any_sc.approach_speed_mps, any_sc.left_turn_speed_mps)
    lanes = [p.line.xy for p in paths.values()]
    labels = {r.id: r.label for r in cfg.scenarios}
    render_heatmaps(
        {labels[k]: v for k, v in result.events.items()}, fig_dir / "conflict_heatmaps.png",
        cell_m=cfg.heatmap_cell_m, max_ttc=th.max_ttc_s,
        per_seed_divisor={labels[k]: len(cfg.seeds) for k in result.events}, lanes=lanes,
    )
    render_ttc_cdf({labels[k]: v["ttc"].to_numpy() for k, v in result.events.items()},
                   fig_dir / "ttc_cdf.png", th.max_ttc_s)

    ids = [r.id for r in cfg.scenarios]
    base = cfg.baseline_id
    lines = [f"# Study results: `{cfg.name}`", ""]
    lines.append(
        f"Seeds: {', '.join(map(str, cfg.seeds))} ({len(cfg.seeds)} replications per scenario). "
        f"Values are seed means with 95% Student-t confidence half-widths. SSAM thresholds: "
        f"max TTC {th.max_ttc_s} s, max PET {th.max_pet_s} s, rear-end < {th.rear_end_angle_deg} deg, "
        f"crossing > {th.crossing_angle_deg} deg."
    )
    alts = [i for i in ids if i != base]
    lines += ["", "## No-Build vs. Build alternatives", ""]
    header = "| Metric | " + " | ".join(labels[i] for i in ids) + " | " + " | ".join(
        f"Change {labels[a].split(':')[0]} (p)" for a in alts
    ) + " |"
    lines += [header, "|" + "---|" * (1 + len(ids) + len(alts))]
    for key, label, fmt in METRICS:
        label = label.format(ttc=th.max_ttc_s, sev=th.severe_ttc_s)
        cells = []
        for i in ids:
            m = comp["scenarios"][i]["metrics"][key]
            cells.append(f"{_fmt(fmt, m['mean'])} ± {_fmt(fmt, m['ci'])}")
        for a in alts:
            tst = comp["tests"][a][key]
            chg = "-" if math.isnan(tst["pct_change"]) else f"{tst['pct_change']:+.0f}%"
            p = tst["p_value"]
            pv = "-" if math.isnan(p) else (f"{p:.3f}" if p >= 0.001 else "<0.001")
            cells.append(f"{chg} ({pv})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| HCM 7 intersection LOS | " + " | ".join(comp["scenarios"][i]["los"] for i in ids) + " |" + " |" * len(alts)
    )
    lines.append("")
    for a in alts:
        ks = comp["tests"][a]["ttc_distribution_ks"]
        lines.append(
            f"Two-sample KS test, pooled minimum TTC, {labels[a].split(':')[0]} vs No-Build: "
            f"D = {ks['D']:.3f}, p = {ks['p_value']:.3g}."
        )

    lines += ["", "## Model checks", "", "| Check | " + " | ".join(labels[i] for i in ids) + " |",
              "|" + "---|" * (len(ids) + 1)]
    checks = [
        ("Movements with GEH < 5 (worst seed)", lambda e: f"{100 * e['geh_pass_rate_min']:.0f}%"),
        ("Max movement GEH (all seeds)", lambda e: f"{e['max_geh']:.2f}"),
        (f"Seeds needed for ±{100 * cfg.replication_tolerance:.0f}% on total conflicts",
         lambda e: str(e["required_replications_total"])),
        ("Raw TTC events per seed (before verification)", lambda e: f"{e['raw_events_mean']:.1f}"),
        ("Projection-only events removed per seed", lambda e: f"{e['unverified_removed_mean']:.1f}"),
        ("Trajectory rows per seed", lambda e: f"{e['trajectory_rows_mean']:,.0f}"),
        ("Detector runtime per seed (s)", lambda e: f"{e['detect_runtime_s_mean']:.2f}"),
    ]
    for name, fn in checks:
        lines.append(f"| {name} | " + " | ".join(fn(comp["scenarios"][i]) for i in ids) + " |")

    lines += ["", "## Movement control delay and LOS (seed mean)", "",
              "| Movement | " + " | ".join(labels[i] for i in ids) + " |", "|" + "---|" * (len(ids) + 1)]
    for mv in sorted(los["movement"].unique()):
        cells = []
        for i in ids:
            r = los[(los["scenario"] == i) & (los["movement"] == mv)]
            cells.append("-" if r.empty else f"{r['delay_s'].iloc[0]:.1f} s ({r['los'].iloc[0]})")
        lines.append(f"| {mv} | " + " | ".join(cells) + " |")

    lines += ["", "![Conflict heatmaps](figures/conflict_heatmaps.png)", "", "![TTC CDF](figures/ttc_cdf.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
