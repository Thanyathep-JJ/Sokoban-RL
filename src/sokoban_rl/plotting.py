"""Training curves from ``log.csv`` files (one or several seeds)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_training(logs: list[str | Path], out: str | Path) -> Path:
    runs = [_read(Path(p)) for p in logs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "avg_return_100", "Average return (last 100 episodes)"),
                           (axes[1], "val_success", "Validation success (%)")):
        n = min(len(r["step"]) for r in runs)
        steps = runs[0]["step"][:n]
        for i, r in enumerate(runs):
            ax.plot(steps, r[key][:n], alpha=0.35, lw=1, label=f"seed {i}" if len(runs) > 1 else None)
        if len(runs) > 1:
            stack = np.stack([r[key][:n] for r in runs])
            mean, std = stack.mean(0), stack.std(0)
            ax.plot(steps, mean, lw=2, color="k", label="mean")
            ax.fill_between(steps, mean - std, mean + std, color="k", alpha=0.12)
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
