"""Severity-weighted conflict density grids and figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SEVERITY_FLOOR = 0.1


def severity_weight(ttc: np.ndarray, max_ttc: float) -> np.ndarray:
    """Linear weight: 1.0 at TTC = 0 down to ``SEVERITY_FLOOR`` at TTC = max_ttc."""
    w = 1.0 - np.clip(ttc / max_ttc, 0.0, 1.0)
    return SEVERITY_FLOOR + (1.0 - SEVERITY_FLOOR) * w


def density_grid(
    events: pd.DataFrame,
    cell_m: float,
    bounds: tuple[float, float, float, float],
    max_ttc: float,
    weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D histogram ``(H[x, y], x_edges, y_edges)`` of conflict locations."""
    x0, y0, x1, y1 = bounds
    xe = np.arange(x0, x1 + cell_m, cell_m)
    ye = np.arange(y0, y1 + cell_m, cell_m)
    if events.empty:
        return np.zeros((xe.size - 1, ye.size - 1)), xe, ye
    w = severity_weight(events["ttc"].to_numpy(float), max_ttc) if weighted else None
    h, _, _ = np.histogram2d(events["x"].to_numpy(float), events["y"].to_numpy(float), bins=[xe, ye], weights=w)
    return h, xe, ye


def grid_table(events: pd.DataFrame, cell_m: float, max_ttc: float) -> pd.DataFrame:
    """Non-empty cells as a tidy table (used by the API)."""
    if events.empty:
        return pd.DataFrame(columns=["x", "y", "count", "severity"])
    cx = np.floor(events["x"].to_numpy(float) / cell_m).astype(np.int64)
    cy = np.floor(events["y"].to_numpy(float) / cell_m).astype(np.int64)
    w = severity_weight(events["ttc"].to_numpy(float), max_ttc)
    df = pd.DataFrame({"cx": cx, "cy": cy, "w": w})
    g = df.groupby(["cx", "cy"]).agg(count=("w", "size"), severity=("w", "sum")).reset_index()
    g["x"] = (g["cx"] + 0.5) * cell_m
    g["y"] = (g["cy"] + 0.5) * cell_m
    return g[["x", "y", "count", "severity"]].sort_values("severity", ascending=False).reset_index(drop=True)


def render_heatmaps(
    panels: Mapping[str, pd.DataFrame],
    out_png: str | Path,
    *,
    cell_m: float,
    max_ttc: float,
    per_seed_divisor: Mapping[str, int] | None = None,
    lanes: Sequence[np.ndarray] | None = None,
    extent_m: float = 40.0,
) -> Path:
    """Side-by-side severity-weighted heatmaps on a shared colour scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    bounds = (-extent_m, -extent_m, extent_m, extent_m)
    grids = {}
    for label, ev in panels.items():
        h, xe, ye = density_grid(ev, cell_m, bounds, max_ttc)
        div = (per_seed_divisor or {}).get(label, 1) or 1
        grids[label] = (gaussian_filter(h / div, sigma=0.8), xe, ye)
    vmax = max((g[0].max() for g in grids.values()), default=1.0) or 1.0

    fig, axes = plt.subplots(1, len(grids), figsize=(6.0 * len(grids), 5.4), squeeze=False)
    for ax, (label, (h, xe, ye)) in zip(axes[0], grids.items()):
        if lanes:
            for line in lanes:
                ax.plot(line[:, 0], line[:, 1], color="#9aa4ad", lw=0.6, zorder=1)
        im = ax.pcolormesh(xe, ye, h.T, cmap="inferno", vmin=0, vmax=vmax, shading="flat", alpha=0.92, zorder=2)
        ev = panels[label]
        if not ev.empty:
            sev = ev[ev["severe"].astype(bool)]
            ax.scatter(sev["x"], sev["y"], s=10, facecolors="none", edgecolors="#36c5f0", lw=0.7, zorder=3,
                       label=f"severe (TTC <= {ev['ttc'][ev['severe'].astype(bool)].max():.2f} s)" if len(sev) else None)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-extent_m, extent_m)
        ax.set_ylim(-extent_m, extent_m)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if not ev.empty and ev["severe"].any():
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="severity-weighted conflicts per seed")
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def render_ttc_cdf(samples: Mapping[str, np.ndarray], out_png: str | Path, max_ttc: float) -> Path:
    """Empirical CDF of minimum TTC per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, x in samples.items():
        x = np.sort(np.asarray(x, dtype=float))
        if x.size:
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", label=f"{label} (n={x.size})")
    ax.set_xlim(0, max_ttc)
    ax.set_ylim(0, 1)
    ax.set_xlabel("minimum TTC (s)")
    ax.set_ylabel("cumulative share of conflicts")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
