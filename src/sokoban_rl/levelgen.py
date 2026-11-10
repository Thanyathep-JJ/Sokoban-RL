"""Level generator using reverse play: start solved, then *pull* boxes away at random.

Every produced level is solvable by construction; BFS then confirms it and measures
the optimal length, which is the final judge of difficulty.
"""

from __future__ import annotations

import numpy as np

from sokoban_rl.game import ACTIONS, Level, level_to_lines, save_levels
from sokoban_rl.solver import solve

Pos = tuple[int, int]

# (seed, count) per split. Different seeds + de-duplication keep the splits disjoint.
SPLITS = {"train": (1000, 1000), "val": (2000, 100), "test": (3000, 200)}


def _connected(cells: set[Pos]) -> bool:
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ACTIONS:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _random_level(rng: np.random.Generator, size: int, n_boxes: int) -> Level | None:
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    walls = {(r, c) for r in range(size) for c in range(size)} - set(interior)
    n_inner = int(rng.integers(0, 4))  # 0-3 inner walls
    for i in rng.choice(len(interior), size=n_inner, replace=False):
        walls.add(interior[int(i)])
    floor = [p for p in interior if p not in walls]
    if len(floor) < 2 * n_boxes + 1 or not _connected(set(floor)):
        return None

    picks = rng.choice(len(floor), size=n_boxes + 1, replace=False)
    targets = frozenset(floor[int(i)] for i in picks[:n_boxes])
    player = floor[int(picks[-1])]
    boxes = set(targets)

    for _ in range(int(rng.integers(10 * n_boxes, 40 * n_boxes + 1))):
        dr, dc = ACTIONS[int(rng.integers(4))]
        nxt = (player[0] + dr, player[1] + dc)
        if nxt in walls or nxt in boxes:
            continue
        back = (player[0] - dr, player[1] - dc)
        if back in boxes and rng.random() < 0.5:  # pull the box along
            boxes.remove(back)
            boxes.add(player)
        player = nxt

    if boxes == set(targets):
        return None
    return Level(size, size, frozenset(walls), targets, player, frozenset(boxes))


def canonical_key(level: Level) -> str:
    """Identical for a level and any of its 8 rotations/reflections."""
    grid = np.array([list(row) for row in level_to_lines(level)])
    forms = []
    for k in range(4):
        rot = np.rot90(grid, k)
        forms.append("".join(rot.ravel()))
        forms.append("".join(np.fliplr(rot).ravel()))
    return min(forms)


def generate_split(
    count: int,
    seed: int,
    size: int = 6,
    n_boxes: int = 1,
    min_len: int = 3,
    max_len: int = 15,
    exclude: set[str] | None = None,
    prefix: str = "lv",
) -> tuple[list[Level], dict]:
    """Return ``count`` unique levels with optimal length in [min_len, max_len]."""
    rng = np.random.default_rng(seed)
    keys = set(exclude or ())
    levels: list[Level] = []
    stats = {"tried": 0, "duplicates": 0}
    while len(levels) < count:
        stats["tried"] += 1
        lv = _random_level(rng, size, n_boxes)
        if lv is None:
            continue
        sol = solve(lv, max_depth=max_len)
        if sol is None or len(sol) < min_len:
            continue
        key = canonical_key(lv)
        if key in keys:
            stats["duplicates"] += 1
            continue
        keys.add(key)
        lv.name = f"{prefix}_{len(levels):04d}"
        lv.bfs_solution = sol
        levels.append(lv)
    return levels, stats


def build_stage(out_dir, size: int, n_boxes: int, stage: int, min_len: int = 3,
                max_len: int = 15, log=print) -> dict[str, list[Level]]:
    """Generate train/val/test for one stage and write ``s{stage}_{split}.json``."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    result = {}
    for split, (seed, count) in SPLITS.items():
        levels, stats = generate_split(count, seed + stage, size, n_boxes, min_len, max_len,
                                       exclude=used, prefix=split)
        used |= {canonical_key(lv) for lv in levels}
        save_levels(levels, out / f"s{stage}_{split}.json")
        hist = np.bincount([lv.bfs_len for lv in levels], minlength=max_len + 1)
        log(f"{split:5s}: {len(levels)} levels, dropped {stats['duplicates']} duplicates, "
            f"bfs_len histogram {dict(enumerate(hist.tolist()))}")
        result[split] = levels
    return result
