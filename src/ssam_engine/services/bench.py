"""Micro-benchmarks for the spatial backends, the detector and the query API."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..spatial.pairs import available_backends, make_pair_finder


def _synthetic_frame(n: int, rng: np.random.Generator, density_veh_per_km2: float = 2_000.0) -> tuple[np.ndarray, np.ndarray]:
    side = 1000.0 * np.sqrt(n / density_veh_per_km2)
    xy = rng.uniform(0.0, side, size=(n, 2))
    speed = rng.uniform(0.0, 20.0, size=n)
    radius = speed * 1.5 + 2.5
    return xy, radius


def bench_pair_finders(sizes: Sequence[int], repeats: int = 5, seed: int = 7) -> pd.DataFrame:
    """Median wall time per frame for each backend at constant traffic density."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for n in sizes:
        xy, radius = _synthetic_frame(int(n), rng)
        reference: np.ndarray | None = None
        for name in available_backends():
            if name == "brute" and n > 4_000:
                continue
            finder = make_pair_finder(name)
            finder.candidate_pairs(xy, radius)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                pairs = finder.candidate_pairs(xy, radius)
                times.append(time.perf_counter() - t0)
            key = np.sort(pairs[:, 0] * n + pairs[:, 1])
            if reference is None:
                reference = key
            rows.append(
                {
                    "vehicles": int(n), "backend": name, "pairs": int(pairs.shape[0]),
                    "median_ms": 1000.0 * float(np.median(times)),
                    "matches_reference": bool(np.array_equal(reference, key)),
                }
            )
    return pd.DataFrame(rows)


def bench_api(fzp: Path, queries: int = 2_000, seed: int = 3) -> dict[str, Any]:
    """In-process ASGI latency (no network) for bbox conflict queries, cold vs warm cache."""
    from fastapi.testclient import TestClient

    from ..api.app import create_app
    from ..api.settings import ApiSettings

    app = create_app(ApiSettings(redis_url=None))
    rng = np.random.default_rng(seed)
    with TestClient(app) as client:
        with open(fzp, "rb") as fh:
            r = client.post("/v1/runs", files={"file": (fzp.name, fh, "application/octet-stream")})
        r.raise_for_status()
        run_id = r.json()["run_id"]
        boxes = []
        for _ in range(queries):
            cx, cy = rng.uniform(-25, 25, size=2)
            half = rng.uniform(3, 15)
            boxes.append((round(cx - half, 2), round(cy - half, 2), round(cx + half, 2), round(cy + half, 2)))

        def run(batch: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, list[str]]:
            lat, cache = [], []
            for b in batch:
                t0 = time.perf_counter()
                resp = client.get(
                    f"/v1/runs/{run_id}/conflicts",
                    params={"min_x": b[0], "min_y": b[1], "max_x": b[2], "max_y": b[3], "limit": 200},
                )
                lat.append(time.perf_counter() - t0)
                resp.raise_for_status()
                cache.append(resp.headers.get("x-cache", ""))
            return 1000.0 * np.asarray(lat), cache

        cold, cold_tags = run(boxes)
        warm, warm_tags = run(boxes)

    def pct(x: np.ndarray) -> dict[str, float]:
        return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)),
                "p99_ms": float(np.percentile(x, 99)), "max_ms": float(x.max())}

    return {
        "run_id": run_id, "queries": queries, "events_indexed": int(r.json()["events"]),
        "cold": pct(cold) | {"hit_rate": cold_tags.count("HIT") / len(cold_tags)},
        "warm": pct(warm) | {"hit_rate": warm_tags.count("HIT") / len(warm_tags)},
    }
