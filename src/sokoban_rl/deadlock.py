"""Deadlock detection stronger than the corner rule: dead squares and frozen boxes.

* A **dead square** is a floor cell from which a box can never reach any target, for example a
  cell along a wall with no target on that wall. Found once per level by "pulling" boxes
  backwards from every target.
* A box is **frozen** when it can move neither horizontally nor vertically: on each axis it has a
  wall next to it, dead squares on both sides, or a neighbouring box that is itself frozen
  (two boxes side by side along a wall, a 2x2 block, ...).

A box off target that sits on a dead square or is frozen means the level can no longer be
solved. Both checks are sound: they never flag a state that is still solvable.

Used by the human games for warnings. The RL environment keeps the simple corner rule in
``game.py`` so the trained models and the reported results stay comparable.
"""

from __future__ import annotations

from weakref import WeakKeyDictionary

from sokoban_rl.game import ACTIONS, Boxes, Level, Pos

_LIVE_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _floor(level: Level, p: Pos) -> bool:
    return 0 <= p[0] < level.height and 0 <= p[1] < level.width and p not in level.walls


def live_squares(level: Level) -> frozenset[Pos]:
    """Cells from which a box can still be pushed onto some target (ignoring other boxes)."""
    cached = _LIVE_CACHE.get(level)
    if cached is not None:
        return cached
    live = set(level.targets)
    stack = list(level.targets)
    while stack:
        x = stack.pop()
        for dr, dc in ACTIONS:
            # a push in direction (dr, dc) moves the box y -> x with the player standing at p
            y = (x[0] - dr, x[1] - dc)
            p = (x[0] - 2 * dr, x[1] - 2 * dc)
            if y not in live and _floor(level, y) and _floor(level, p):
                live.add(y)
                stack.append(y)
    result = frozenset(live)
    _LIVE_CACHE[level] = result
    return result


def _frozen(level: Level, boxes: Boxes, box: Pos, live: frozenset[Pos], as_wall: frozenset[Pos]) -> bool:
    as_wall = as_wall | {box}  # while checking neighbours, this box counts as a wall (avoids loops)
    for dr, dc in ((1, 0), (0, 1)):
        a = (box[0] + dr, box[1] + dc)
        b = (box[0] - dr, box[1] - dc)
        blocked = (
            a in level.walls or b in level.walls
            or (a not in live and b not in live)
            or any(n in as_wall or (n in boxes and _frozen(level, boxes, n, live, as_wall)) for n in (a, b))
        )
        if not blocked:
            return False
    return True


def dead_boxes(level: Level, boxes: Boxes) -> set[Pos]:
    """Boxes off target that can never reach a target. Empty means no deadlock was found."""
    live = live_squares(level)
    return {b for b in boxes
            if b not in level.targets and (b not in live or _frozen(level, boxes, b, live, frozenset()))}


def is_dead(level: Level, boxes: Boxes) -> bool:
    return bool(dead_boxes(level, boxes))
