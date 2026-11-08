"""Domain exceptions.

Every error raised deliberately by the engine derives from :class:`SSAMEngineError`
so callers (CLI, API) can map failures to exit codes or HTTP status codes without
catching broad built-in exceptions.
"""

from __future__ import annotations


class SSAMEngineError(Exception):
    """Base class for all engine errors.

    Args:
        message: Human-readable description.
        context: Structured key/value details that are attached to log records.
    """

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({details})"


class ConfigurationError(SSAMEngineError):
    """A configuration file or value is missing or invalid."""


class TrajectoryFormatError(SSAMEngineError):
    """A trajectory file cannot be parsed or fails schema validation."""


class SpatialIndexUnavailableError(SSAMEngineError):
    """The requested spatial index backend is not installed."""


class SimulationError(SSAMEngineError):
    """The synthetic simulator reached an invalid state."""


class VissimAutomationError(SSAMEngineError):
    """PTV Vissim COM automation failed or is unavailable on this host."""


class RunNotFoundError(SSAMEngineError):
    """An API run identifier does not exist (or was evicted)."""
