"""BFS oracle: shortest move sequence, using the exact same ``move`` rules as the game."""

from __future__ import annotations

from collections import deque

from sokoban_rl.game import ACTION_CHARS, Level, is_deadlock, move


def solve(level: Level, max_depth: int | None = None) -> str | None:
    """Shortest solution as a string over ``UDLR``; ``""`` if already solved.

    Returns None if unsolvable (or no solution within ``max_depth`` moves).
    Deadlocked states are pruned, matching the environment (which ends the game there).
    """
    if level.boxes == level.targets:
        return ""
    start = (level.player, level.boxes)
    seen = {start}
    queue = deque([(level.player, level.boxes, "")])
    while queue:
        player, boxes, path = queue.popleft()
        if max_depth is not None and len(path) >= max_depth:
            continue
        for a in range(4):
            np_, nb, push = move(level, player, boxes, a)
            if push is None and np_ == player:
                continue  # bumped into a wall / blocked push
            key = (np_, nb)
            if key in seen:
                continue
            if nb == level.targets:
                return path + ACTION_CHARS[a]
            if push is not None and is_deadlock(level, nb):
                continue
            seen.add(key)
            queue.append((np_, nb, path + ACTION_CHARS[a]))
    return None
