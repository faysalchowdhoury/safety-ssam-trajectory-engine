"""FastAPI application.

Endpoints
---------
``GET  /health``                         liveness + cache backend
``POST /v1/runs``                        upload a ``.fzp``/``.fzp.gz``; returns a run summary
``GET  /v1/runs/{run_id}``               run summary
``GET  /v1/runs/{run_id}/conflicts``     bbox / type / TTC filtered conflicts (R*-tree + cache)
``GET  /v1/runs/{run_id}/heatmap``       severity-weighted grid cells

Analysis is CPU-bound and runs in a worker thread so the event loop keeps serving
queries. Query responses are serialised once and cached as JSON strings, so a
cache hit is a dictionary lookup plus a socket write.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..core.config import EngineConfig
from ..core.exceptions import ConfigurationError, RunNotFoundError, SSAMEngineError, TrajectoryFormatError
from ..core.logging import configure_logging, get_logger
from ..core.models import ConflictType
from ..services.heatmap import grid_table
from ..services.pipeline import analyze_file, event_summary
from ..spatial.points import build_point_index
from .cache import build_cache
from .schemas import (
    ConflictQueryResponse,
    ConflictRecord,
    HealthResponse,
    HeatmapCell,
    HeatmapResponse,
    RunSummary,
)
from .settings import ApiSettings
from .store import RunRecord, RunStore

log = get_logger(__name__)
_ALLOWED_SUFFIXES = (".fzp", ".fzp.gz")
_CHUNK = 1 << 20


def create_app(settings: ApiSettings | None = None, engine_config: EngineConfig | None = None) -> FastAPI:
    """Application factory (used by uvicorn, tests and benchmarks)."""
    settings = settings or ApiSettings.from_env()
    if engine_config is None:
        engine_config = (
            EngineConfig.from_file(settings.engine_config_path) if settings.engine_config_path else EngineConfig()
        )
    cfg = engine_config

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cache = build_cache(settings.redis_url, settings.cache_max_entries)
        app.state.store = RunStore(settings.max_runs)
        log.info("api.started", extra={"cache": app.state.cache.backend, "version": __version__})
        yield
        await app.state.cache.close()

    app = FastAPI(title="SSAM Trajectory Engine", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = 1000.0 * (time.perf_counter() - t0)
        response.headers["x-process-time-ms"] = f"{ms:.3f}"
        if request.url.path.startswith("/v1/runs") and request.method == "POST":
            log.info("api.request", extra={"path": request.url.path, "status": response.status_code, "ms": round(ms, 2)})
        return response

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "run_not_found", "detail": str(exc)})

    @app.exception_handler(TrajectoryFormatError)
    async def _bad_traj(_: Request, exc: TrajectoryFormatError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "invalid_trajectory", "detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def _bad_cfg(_: Request, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "invalid_configuration", "detail": str(exc)})

    @app.exception_handler(SSAMEngineError)
    async def _engine(_: Request, exc: SSAMEngineError) -> JSONResponse:
        log.error("api.engine_error", extra={"detail": str(exc)})
        return JSONResponse(status_code=500, content={"error": "engine_error", "detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, cache_backend=request.app.state.cache.backend,
                              runs_loaded=len(request.app.state.store))

    @app.post("/v1/runs", response_model=RunSummary, status_code=201)
    async def create_run(request: Request, file: UploadFile = File(...)) -> RunSummary:
        name = file.filename or "upload.fzp"
        suffix = ".fzp.gz" if name.lower().endswith(".fzp.gz") else Path(name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="upload a Vissim vehicle record (.fzp or .fzp.gz)")
        limit = settings.max_upload_mb * (1 << 20)
        digest = hashlib.sha256()
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        size = 0
        try:
            with os.fdopen(fd, "wb") as tmp:
                while chunk := await file.read(_CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(status_code=413, detail=f"upload exceeds {settings.max_upload_mb} MB")
                    digest.update(chunk)
                    tmp.write(chunk)
            result = await run_in_threadpool(analyze_file, Path(tmp_name), cfg)
        finally:
            Path(tmp_name).unlink(missing_ok=True)

        events = result.events.reset_index(drop=True)
        index = build_point_index(events[["x", "y"]].to_numpy(np.float64))
        counts = event_summary(events)
        run_id = f"{digest.hexdigest()[:12]}-{uuid.uuid4().hex[:8]}"
        summary = RunSummary(
            run_id=run_id, source_name=name, created_at=datetime.now(timezone.utc), rows=result.stats.n_rows,
            vehicles=result.stats.n_vehicles, events=len(events),
            by_type={ct.value: counts[ct.value] for ct in ConflictType}, severe=counts["severe"],
            collisions=counts["collisions"],
            raw_events=result.stats.n_raw_events, unverified_removed=result.stats.n_unverified,
            detector_runtime_s=round(result.stats.runtime_s, 4), index_backend=index.backend,
        )
        evicted = request.app.state.store.put(RunRecord(summary, events, index))
        for rid in evicted:
            await request.app.state.cache.delete_prefix(f"q:{rid}:")
        return summary

    @app.get("/v1/runs/{run_id}", response_model=RunSummary)
    async def get_run(request: Request, run_id: str) -> RunSummary:
        return request.app.state.store.get(run_id).summary

    @app.get("/v1/runs/{run_id}/conflicts", response_model=ConflictQueryResponse)
    async def query_conflicts(
        request: Request,
        run_id: str,
        min_x: float | None = None,
        min_y: float | None = None,
        max_x: float | None = None,
        max_y: float | None = None,
        conflict_type: ConflictType | None = None,
        max_ttc: float | None = Query(None, gt=0.0),
        severe_only: bool = False,
        limit: int = Query(500, ge=1, le=10_000),
    ) -> Response:
        bbox = (min_x, min_y, max_x, max_y)
        if any(v is not None for v in bbox) and not all(v is not None for v in bbox):
            raise HTTPException(status_code=422, detail="provide all of min_x, min_y, max_x, max_y or none")
        if min_x is not None and (min_x > max_x or min_y > max_y):  # type: ignore[operator]
            raise HTTPException(status_code=422, detail="min must not exceed max")
        rec = request.app.state.store.get(run_id)
        key = f"q:{run_id}:{bbox}:{conflict_type}:{max_ttc}:{severe_only}:{limit}"
        cache = request.app.state.cache
        cached = await cache.get(key)
        if cached is not None:
            return Response(content=cached, media_type="application/json", headers={"x-cache": "HIT"})

        ev = rec.events
        idx = rec.index.query_bbox(*bbox) if min_x is not None else np.arange(len(ev))  # type: ignore[arg-type]
        sub = ev.iloc[idx]
        if conflict_type is not None:
            sub = sub[sub["conflict_type"] == conflict_type.value]
        if max_ttc is not None:
            sub = sub[sub["ttc"] <= max_ttc]
        if severe_only:
            sub = sub[sub["severe"]]
        sub = sub.sort_values("ttc", kind="mergesort")
        total = len(sub)
        sub = sub.head(limit)
        records = [
            ConflictRecord(
                event_id=int(r.event_id), vid_a=int(r.vid_a), vid_b=int(r.vid_b), conflict_type=r.conflict_type,
                t_min_ttc=float(r.t_min_ttc), ttc=float(r.ttc), pet=None if np.isnan(r.pet) else float(r.pet),
                x=float(r.x), y=float(r.y), angle_deg=float(r.angle_deg), max_s=float(r.max_s),
                delta_s=float(r.delta_s), dr=float(r.dr), max_d=float(r.max_d), max_delta_v=float(r.max_delta_v),
                severe=bool(r.severe), collision=bool(r.collision),
            )
            for r in sub.itertuples(index=False)
        ]
        body = ConflictQueryResponse(run_id=run_id, count=total, truncated=total > limit, conflicts=records)
        payload = body.model_dump_json()
        await cache.set(key, payload, settings.cache_ttl_s)
        return Response(content=payload, media_type="application/json", headers={"x-cache": "MISS"})

    @app.get("/v1/runs/{run_id}/heatmap", response_model=HeatmapResponse)
    async def heatmap(request: Request, run_id: str, cell_m: float = Query(2.0, gt=0.1, le=100.0)) -> HeatmapResponse:
        rec = request.app.state.store.get(run_id)
        cells = grid_table(rec.events, cell_m, cfg.thresholds.max_ttc_s)
        return HeatmapResponse(
            run_id=run_id, cell_m=cell_m,
            cells=[HeatmapCell(x=float(r.x), y=float(r.y), count=int(r.count), severity=float(r.severity))
                   for r in cells.itertuples(index=False)],
        )

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory entry point: ``uvicorn ssam_engine.api.app:app_from_env --factory``."""
    settings = ApiSettings.from_env()
    configure_logging(settings.log_level)
    return create_app(settings)
