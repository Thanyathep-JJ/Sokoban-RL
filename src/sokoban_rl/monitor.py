"""Live progress bar for a running DQN training (``sokoban-rl watch``).

It only reads ``config.json`` and ``log.csv`` from the run folder, so it can be started, stopped and
restarted at any time without touching the training process. The log gets a new row every
``eval_every`` steps, so between rows the step count is estimated from the average speed.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Run:
    label: str
    total: int
    eval_every: int
    rows: list[dict]  # one dict per validation checkpoint: step, val, eps, minutes
    updated: float  # when log.csv was last written (epoch seconds)


@dataclass
class Progress:
    step: float
    total: int
    elapsed_min: float
    left_min: float | None  # None until there is a speed to extrapolate from
    finished: bool
    overdue_min: float | None  # minutes since the last checkpoint, if far later than expected

    @property
    def fraction(self) -> float:
        return min(1.0, self.step / self.total) if self.total else 0.0


def read_run(run_dir: str | Path) -> Run:
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    log = d / "log.csv"
    rows = []
    if log.exists():
        with log.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({"step": int(r["step"]), "val": float(r["val_success"]),
                                 "eps": float(r["epsilon"]), "minutes": float(r["minutes"])})
                except (KeyError, ValueError, TypeError):
                    continue  # the training may be writing this line right now
    updated = (log if log.exists() else d / "config.json").stat().st_mtime
    label = f"{d.parent.name}/{d.name}" + (" (Double DQN)" if cfg.get("double") else "")
    return Run(label, int(cfg["steps"]), int(cfg.get("eval_every", 10_000)), rows, updated)


def estimate(run: Run, now: float) -> Progress:
    since = max(0.0, now - run.updated) / 60
    if not run.rows:
        return Progress(0, run.total, since, None, False, since if since > 10 else None)
    last = run.rows[-1]
    if last["step"] >= run.total:
        return Progress(run.total, run.total, last["minutes"], 0.0, True, None)
    rate = last["step"] / last["minutes"] if last["minutes"] > 0 else 0.0  # steps per minute
    if rate <= 0:
        return Progress(last["step"], run.total, last["minutes"] + since, None, False, None)
    step = min(last["step"] + rate * since, last["step"] + run.eval_every, run.total)
    gap = run.eval_every / rate  # normal minutes between two log rows
    overdue = since if since > max(3 * gap, 3.0) else None
    return Progress(step, run.total, last["minutes"] + since, (run.total - step) / rate, False, overdue)


def fmt_duration(minutes: float) -> str:
    seconds = int(round(minutes * 60))
    hours, rest = divmod(seconds, 3600)
    mins, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def render(run: Run, now: float, bar_width: int = 30) -> list[str]:
    """Three lines per run: bar and percent, timing, then the latest results (or a warning)."""
    p = estimate(run, now)
    filled = int(bar_width * p.fraction)
    bar = "█" * filled + "░" * (bar_width - filled)
    timing = [f"step {p.step:,.0f} / {p.total:,}", f"elapsed {fmt_duration(p.elapsed_min)}"]
    if p.finished:
        timing.append("finished")
    elif p.left_min is not None:
        eta = datetime.fromtimestamp(now) + timedelta(minutes=p.left_min)
        timing += [f"left {fmt_duration(p.left_min)}", f"done around {eta:%H:%M}"]
    else:
        timing.append("waiting for the first checkpoint")
    if p.overdue_min is not None:
        third = f"no new checkpoint for {fmt_duration(p.overdue_min)}, training may have stopped"
    elif run.rows:
        last = run.rows[-1]
        best = max(r["val"] for r in run.rows)
        third = f"validation {last['val']:.0f}% (best {best:.0f}%) · exploration ε {last['eps']:.2f}"
    else:
        third = ""
    return [f"{run.label}  [{bar}] {p.fraction * 100:5.1f}%", "  " + " · ".join(timing), "  " + third]


def find_active_runs(root: str | Path = ".") -> list[Path]:
    """Run folders under models*/ whose training has not reached its last step."""
    active = []
    for cfg in sorted(Path(root).glob("models*/*/config.json")):
        try:
            if not estimate(read_run(cfg.parent), time.time()).finished:
                active.append(cfg.parent)
        except (OSError, ValueError, KeyError):
            continue
    return active


def watch(run_dirs: list[str | Path], interval: float = 1.0, once: bool = False, out=sys.stdout,
          clock=time.time, sleep=time.sleep) -> None:
    """Redraw the progress in place until every run has finished (Ctrl+C stops watching)."""
    live = not once and getattr(out, "isatty", lambda: False)()
    drawn = 0
    try:
        while True:
            now = clock()
            runs = [read_run(d) for d in run_dirs]
            width = shutil.get_terminal_size((100, 20)).columns
            bar = max(10, min(40, width - max(len(r.label) for r in runs) - 14))
            lines = [ln for r in runs for ln in render(r, now, bar)]
            if live and drawn:
                out.write(f"\033[{drawn}A")
            for ln in lines:
                out.write(("\033[K" if live else "") + ln + "\n")
            out.flush()
            drawn = len(lines)
            if not live or all(estimate(r, now).finished for r in runs):
                return
            sleep(interval)
    except KeyboardInterrupt:
        out.write("\n")
