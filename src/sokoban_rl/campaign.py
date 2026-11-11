"""Level-by-level campaign for a human player: ordering, scoring, timer results and saved progress.

Score per level: 100 for the shortest possible solution, less for longer ones
(``100 x shortest / your moves``), 0 when you give up. Stars: 3 = shortest, 2 = at most 1.5x,
1 = solved. Time is recorded too and breaks ties between equal scores. Progress is kept in a
small JSON file, so the campaign continues where you left off and keeps your best result per level.
"""

from __future__ import annotations

import json
from pathlib import Path

from sokoban_rl.game import ACTION_CHARS, Level, load_levels, move, save_levels

# Difficulty modes for human play. Each is a fixed, reproducible set of generated levels made of
# one or more parts; a part is picked from a larger pool so its solution lengths climb evenly.
# Single-box levels rarely need more than 8-9 moves, so later parts add a box.
# Hidden modes are not shown in the window's mode bar but still work with --difficulty.
DIFFICULTIES = {
    "easy": {"label": "ง่าย", "seed": 7001, "parts": [
        {"size": 6, "boxes": 1, "min_len": 3, "max_len": 9, "count": 20, "pool": 800},
    ]},
    "medium": {"label": "ปานกลาง", "seed": 7002, "parts": [
        {"size": 7, "boxes": 1, "min_len": 7, "max_len": 12, "count": 6, "pool": 1500},
        {"size": 7, "boxes": 2, "min_len": 9, "max_len": 16, "count": 14, "pool": 1500},
    ]},
    "hard": {"label": "ยาก", "seed": 7003, "hidden": True, "parts": [
        {"size": 7, "boxes": 2, "min_len": 8, "max_len": 14, "count": 20, "pool": 500},
    ]},
    "expert": {"label": "ยากมาก", "seed": 7004, "hidden": True, "parts": [
        {"size": 7, "boxes": 3, "min_len": 15, "max_len": 40, "count": 20, "pool": 600},
    ]},
}


def visible_difficulties() -> list[str]:
    return [name for name, spec in DIFFICULTIES.items() if not spec.get("hidden")]


def pushes(level: Level) -> int:
    """How many of the shortest solution's moves push a box."""
    player, boxes, n = level.player, level.boxes, 0
    for ch in level.bfs_solution or "":
        player, boxes, push = move(level, player, boxes, ACTION_CHARS.index(ch))
        n += push is not None
    return n


def difficulty_score(level: Level) -> tuple[int, int, int]:
    """Harder = more boxes, then a longer shortest solution, then more pushes in it."""
    return len(level.boxes), level.bfs_len or 0, pushes(level)


def pick_progression(pool: list[Level], count: int) -> list[Level]:
    """``count`` levels whose solution lengths climb evenly from the pool's shortest to its longest."""
    pool = sorted(pool, key=difficulty_score)
    lens = [lv.bfs_len or 0 for lv in pool]
    lo, hi = lens[0], lens[-1]
    used: set[int] = set()
    for i in range(min(count, len(pool))):
        target = lo + (hi - lo) * i / max(1, count - 1)
        j = min((j for j in range(len(pool)) if j not in used), key=lambda j: (abs(lens[j] - target), -j))
        used.add(j)
    return sorted((pool[j] for j in used), key=difficulty_score)


def _training_keys(data_dir: str | Path) -> set[str]:
    """Shapes of every level the AI trains or is tested on, so game levels can avoid them."""
    from sokoban_rl.levelgen import canonical_key

    keys: set[str] = set()
    for path in Path(data_dir).glob("s*_*.json"):
        try:
            keys |= {canonical_key(lv) for lv in load_levels(path)}
        except (OSError, ValueError):
            continue
    return keys


def difficulty_levels(name: str, data_dir: str | Path = "data") -> list[Level]:
    """The level set of one difficulty, easiest first. Generated once (seconds), then read from a cache."""
    spec = DIFFICULTIES[name]
    count = sum(part["count"] for part in spec["parts"])
    path = Path(data_dir) / f"play_{name}.json"
    if path.exists():
        try:
            levels = load_levels(path)
            if len(levels) == count:
                return levels
        except (OSError, ValueError):
            pass  # damaged or outdated cache: build it again
    from sokoban_rl.levelgen import generate_split

    exclude = _training_keys(data_dir)
    levels: list[Level] = []
    for i, part in enumerate(spec["parts"]):
        pool, _ = generate_split(part["pool"], spec["seed"] + 100 * i, size=part["size"],
                                 n_boxes=part["boxes"], min_len=part["min_len"],
                                 max_len=part["max_len"], exclude=exclude, prefix=name)
        levels += pick_progression(pool, part["count"])
    levels.sort(key=difficulty_score)
    for i, lv in enumerate(levels, start=1):
        lv.name = f"{name}_{i:02d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_levels(levels, path)
    return levels


def score_for(moves: int, best: int) -> int:
    return max(1, round(100 * best / moves)) if moves > 0 else 0


def stars_for(moves: int, best: int) -> int:
    if moves <= best:
        return 3
    return 2 if moves <= 1.5 * best else 1


def level_key(level: Level, index: int) -> str:
    return level.name or f"#{index}"


def order_by_difficulty(levels: list[Level]) -> list[Level]:
    """Easiest first (boxes, solution length, pushes), so the campaign gets harder as you go."""
    return sorted(levels, key=difficulty_score)


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

    def totals(self, keys: list[str]) -> tuple[int, int, int]:
        """(score, stars, levels solved) over these levels only, ignoring results of old level sets."""
        rows = [self.best[k] for k in keys if k in self.best]
        return sum(r["score"] for r in rows), sum(r["stars"] for r in rows), len(rows)

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
