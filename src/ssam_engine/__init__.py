"""safety-ssam-trajectory-engine.

Trajectory-based surrogate safety assessment that reproduces the FHWA SSAM
conflict methodology (TTC, PET, MaxS, DeltaS, DR, MaxD, MaxDeltaV) on PTV Vissim
vehicle-record (``.fzp``) output, with spatial indexing, multi-seed statistics,
risk heatmaps and a low-latency query API.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
