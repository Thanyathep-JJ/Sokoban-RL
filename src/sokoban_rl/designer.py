"""Designed 1-box levels for the human game: harder because they need thought, not just more steps.

Ideas from the Sokoban generation literature:

* Taylor & Parberry (2011): build the room from small wall pieces (corridors and corners) instead of
  single random blocks, start from the state *farthest* from solved, and rate a level by its
  **box lines** (straight runs of pushes; more direction changes = harder). Long straight pushes only
  make a level tedious, so the longest run is capped.
* Jarušek & Pelánek (2010): solution length alone predicts human difficulty poorly, so the order
  inside a mode uses box lines first.

The AI's train/val/test sets still come from ``levelgen`` (random walk), so earlier results are
unchanged; these levels are only for people (and to see how the AI copes with something new).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from sokoban_rl.game import ACTION_CHARS, ACTIONS, Level
from sokoban_rl.levelgen import _connected, canonical_key
from sokoban_rl.solver import solve

Pos = tuple[int, int]

# wall pieces as offsets: a block, short walls and corners
PIECES = (
    ((0, 0),),
    ((0, 0), (0, 1)), ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 0), (1, 1)), ((0, 1), (1, 0), (1, 1)),
)


def push_runs(level: Level, solution: str) -> list[int]:
    """Lengths of the straight push runs ("box lines"). Walking away or turning starts a new line."""
    player, boxes = level.player, set(level.boxes)
    runs: list[int] = []
    last: str | None = None
    for ch in solution:
        dr, dc = ACTIONS[ACTION_CHARS.index(ch)]
        nxt = (player[0] + dr, player[1] + dc)
        if nxt in boxes:
            boxes.remove(nxt)
            boxes.add((nxt[0] + dr, nxt[1] + dc))
            if last == ch:
                runs[-1] += 1
            else:
                runs.append(1)
            last = ch
        else:
            last = None
        player = nxt
    return runs


def box_lines(level: Level) -> int:
    return len(push_runs(level, level.bfs_solution or ""))


def rating(level: Level) -> tuple[int, int]:
    """Order inside a mode: more box lines first, then a longer shortest solution."""
    return box_lines(level), level.bfs_len or 0


def _layout(rng: np.random.Generator, size: int, walls: tuple[str, int, int]) -> set[Pos] | None:
    """Outer wall plus inner walls: ("random", lo, hi) single blocks or ("pieces", lo, hi) wall pieces."""
    kind, lo, hi = walls
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    out = {(r, c) for r in range(size) for c in range(size)} - set(interior)
    for _ in range(int(rng.integers(lo, hi + 1))):
        shape = PIECES[int(rng.integers(len(PIECES)))] if kind == "pieces" else ((0, 0),)
        r0, c0 = interior[int(rng.integers(len(interior)))]
        out |= {(r0 + dr, c0 + dc) for dr, dc in shape if 0 < r0 + dr < size - 1 and 0 < c0 + dc < size - 1}
    floor = set(interior) - out
    return out if len(floor) >= 4 and _connected(floor) else None


def pull_distances(walls: set[Pos], target: Pos, floor: list[Pos]) -> dict[tuple[Pos, Pos], int]:
    """Fewest pushes to solve each (player, box) state, found by pulling the box away from the target."""
    dist: dict[tuple[Pos, Pos], int] = {}
    q: deque = deque()
    for p in floor:
        if p != target:
            dist[(p, target)] = 0
            q.append((p, target))
    while q:  # 0-1 BFS: walking is free, each pull costs one push
        player, box = state = q.popleft()
        d = dist[state]
        for dr, dc in ACTIONS:
            nxt = (player[0] + dr, player[1] + dc)
            if nxt in walls or nxt == box:
                continue
            if (nxt, box) not in dist or dist[(nxt, box)] > d:
                dist[(nxt, box)] = d
                q.appendleft((nxt, box))
            if (player[0] - dr, player[1] - dc) == box and ((nxt, player) not in dist or dist[(nxt, player)] > d + 1):
                dist[(nxt, player)] = d + 1
                q.append((nxt, player))
    return dist


def _shape_key(level: Level) -> str:
    """Same room, target and box = same puzzle, wherever the player starts."""
    target = next(iter(level.targets))
    return canonical_key(Level(level.width, level.height, level.walls, level.targets, target, level.boxes))


def generate_designed(count: int, seed: int, size: int, walls: tuple[str, int, int], far: float,
                      min_len: int, max_len: int, min_lines: int, max_lines: int = 99, max_run: int = 3,
                      exclude: set[str] | None = None, prefix: str = "lv",
                      max_tries: int = 200_000) -> list[Level]:
    """``count`` distinct 1-box levels.

    ``far`` picks the start: 1.0 = only the states needing the most pushes, 0.6 = at least 60 % of
    that. Levels are kept when the shortest solution has ``min_len``-``max_len`` moves and
    ``min_lines``-``max_lines`` box lines, and no straight push is longer than ``max_run``.
    """
    rng = np.random.default_rng(seed)
    keys = set(exclude or ())
    levels: list[Level] = []
    for _ in range(max_tries):
        if len(levels) == count:
            return levels
        wall_set = _layout(rng, size, walls)
        if wall_set is None:
            continue
        floor = [(r, c) for r in range(size) for c in range(size) if (r, c) not in wall_set]
        target = floor[int(rng.integers(len(floor)))]
        dist = pull_distances(wall_set, target, floor)
        top = max(dist.values())
        if top < 1:
            continue
        starts = [s for s, d in dist.items() if d >= max(1, round(far * top))]
        player, box = starts[int(rng.integers(len(starts)))]
        lv = Level(size, size, frozenset(wall_set), frozenset({target}), player, frozenset({box}))
        key = _shape_key(lv)
        if key in keys:
            continue
        sol = solve(lv, max_depth=max_len)
        if sol is None or len(sol) < min_len:
            continue
        runs = push_runs(lv, sol)
        if not min_lines <= len(runs) <= max_lines or max(runs) > max_run:
            continue
        keys.add(key)
        keys.add(canonical_key(lv))
        lv.name = f"{prefix}_{len(levels):04d}"
        lv.bfs_solution = sol
        levels.append(lv)
    raise RuntimeError(f"only {len(levels)} of {count} designed levels found for a {size}x{size} board; "
                       "loosen the limits")


# Stage 7 (training data for the designed levels): a mix so the AI does not forget the easy ones.
# (share of each split, board size, generator settings; None = the random-walk generator of levelgen)
STAGE7_PARTS = (
    (0.10, 5, None),
    (0.25, 6, dict(walls=("random", 1, 3), far=0.5, min_len=5, max_len=12, min_lines=2, max_lines=3)),
    (0.25, 7, dict(walls=("pieces", 1, 2), far=0.5, min_len=7, max_len=12, min_lines=2, max_lines=2)),
    (0.40, 7, dict(walls=("pieces", 1, 3), far=0.6, min_len=9, max_len=16, min_lines=3, max_lines=4)),
)


def build_designed_stage(out_dir, stage: int = 7, parts=STAGE7_PARTS, exclude: set[str] | None = None,
                         log=print) -> dict[str, list[Level]]:
    """Write ``s{stage}_{train,val,test}.json`` from ``parts``; splits never share a puzzle."""
    from pathlib import Path

    from sokoban_rl.game import save_levels
    from sokoban_rl.levelgen import SPLITS, generate_split

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    used: set[str] = set(exclude or ())
    result = {}
    for split, (seed, count) in SPLITS.items():
        shares = [int(count * share) for share, _, _ in parts]
        shares[-1] += count - sum(shares)
        levels: list[Level] = []
        for k, ((_, size, how), n) in enumerate(zip(parts, shares, strict=True)):
            if how is None:
                part, _ = generate_split(n, seed + stage + 17 * k, size, 1, 2, 15, exclude=used, prefix=split)
            else:
                part = generate_designed(n, seed + stage + 17 * k, size, exclude=used, prefix=split, **how)
            used |= {canonical_key(lv) for lv in part} | {_shape_key(lv) for lv in part}
            levels += part
        for i, lv in enumerate(levels):
            lv.name = f"{split}_{i:04d}"
        save_levels(levels, out / f"s{stage}_{split}.json")
        log(f"{split:5s}: {len(levels)} levels (sizes {[s for _, s, _ in parts]}, counts {shares}), "
            f"box lines {np.bincount([box_lines(lv) for lv in levels]).tolist()}")
        result[split] = levels
    return result
