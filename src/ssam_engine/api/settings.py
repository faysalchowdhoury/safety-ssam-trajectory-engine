"""API settings read from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ApiSettings(BaseModel):
    """Runtime settings.

    Environment variables: ``SSAM_REDIS_URL``, ``SSAM_CACHE_TTL_S``, ``SSAM_CACHE_MAX_ENTRIES``,
    ``SSAM_MAX_UPLOAD_MB``, ``SSAM_MAX_RUNS``, ``SSAM_ENGINE_CONFIG``, ``SSAM_LOG_LEVEL``.
    """

    model_config = ConfigDict(frozen=True)

    redis_url: str | None = None
    cache_ttl_s: int = Field(300, ge=1)
    cache_max_entries: int = Field(10_000, ge=10)
    max_upload_mb: int = Field(1_024, ge=1)
    max_runs: int = Field(16, ge=1)
    engine_config_path: Path | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ApiSettings":
        env = os.environ
        return cls(
            redis_url=env.get("SSAM_REDIS_URL") or None,
            cache_ttl_s=int(env.get("SSAM_CACHE_TTL_S", 300)),
            cache_max_entries=int(env.get("SSAM_CACHE_MAX_ENTRIES", 10_000)),
            max_upload_mb=int(env.get("SSAM_MAX_UPLOAD_MB", 1_024)),
            max_runs=int(env.get("SSAM_MAX_RUNS", 16)),
            engine_config_path=Path(env["SSAM_ENGINE_CONFIG"]) if env.get("SSAM_ENGINE_CONFIG") else None,
            log_level=env.get("SSAM_LOG_LEVEL", "INFO"),
        )
