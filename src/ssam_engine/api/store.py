"""Thread-safe in-memory store of analysed runs (bounded, LRU eviction)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from ..core.exceptions import RunNotFoundError
from ..spatial.points import PointIndex
from .schemas import RunSummary


@dataclass(frozen=True)
class RunRecord:
    summary: RunSummary
    events: pd.DataFrame
    index: PointIndex


class RunStore:
    def __init__(self, max_runs: int) -> None:
        self.max_runs = max_runs
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, record: RunRecord) -> list[str]:
        """Insert a run; returns ids evicted to respect ``max_runs``."""
        evicted: list[str] = []
        with self._lock:
            self._runs[record.summary.run_id] = record
            self._runs.move_to_end(record.summary.run_id)
            while len(self._runs) > self.max_runs:
                rid, _ = self._runs.popitem(last=False)
                evicted.append(rid)
        return evicted

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                raise RunNotFoundError("run not found", context={"run_id": run_id})
            self._runs.move_to_end(run_id)
            return rec

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)
