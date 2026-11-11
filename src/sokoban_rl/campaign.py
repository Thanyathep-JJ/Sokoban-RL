"""Level-by-level campaign for a human player: ordering, scoring, timer results and saved progress.

Score per level: 100 for the shortest possible solution, less for longer ones
(``100 x shortest / your moves``), 0 when you give up. Stars: 3 = shortest, 2 = at most 1.5x,
1 = solved. Time is recorded too and breaks ties between equal scores. Progress is kept in a
small JSON file, so the campaign continues where you left off and keeps your best result per level.
"""

from __future__ import annotations

import json
from pathlib import Path

from sokoban_rl.game import Level


def score_for(moves: int, best: int) -> int:
    return max(1, round(100 * best / moves)) if moves > 0 else 0


def stars_for(moves: int, best: int) -> int:
    if moves <= best:
        return 3
    return 2 if moves <= 1.5 * best else 1


def level_key(level: Level, index: int) -> str:
    return level.name or f"#{index}"


def order_by_difficulty(levels: list[Level]) -> list[Level]:
    """Shortest solutions first, so the campaign gets harder as you go (stable within a length)."""
    return sorted(levels, key=lambda lv: lv.bfs_len if lv.bfs_len is not None else 0)


class Progress:
    """Best result per level, stored as JSON: {key: {moves, seconds, score, stars}}."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.best: dict[str, dict] = {}
        if self.path and self.path.exists():
            try:
                self.best = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.best = {}  # a damaged file should not stop the game

    @property
    def total_score(self) -> int:
        return sum(r["score"] for r in self.best.values())

    @property
    def total_stars(self) -> int:
        return sum(r["stars"] for r in self.best.values())

    def record(self, key: str, moves: int, seconds: float, best: int) -> tuple[dict, bool]:
        """Store a solved level. Returns (this result, whether it beat the saved best)."""
        result = {"moves": moves, "seconds": round(seconds, 1), "score": score_for(moves, best),
                  "stars": stars_for(moves, best)}
        old = self.best.get(key)
        improved = old is None or (result["score"], -result["seconds"]) > (old["score"], -old["seconds"])
        if improved:
            self.best[key] = result
            self.save()
        return result, improved

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.best, indent=1), encoding="utf-8")

    def first_unsolved(self, keys: list[str]) -> int:
        return next((i for i, k in enumerate(keys) if k not in self.best), 0)


def fmt_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
