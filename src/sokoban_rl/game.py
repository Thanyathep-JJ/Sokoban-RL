"""Sokoban rules, level parsing and the RL environment.

State is ``(player, boxes)`` with ``player = (row, col)`` and ``boxes`` a frozenset
of ``(row, col)``. Walls and targets are fixed per level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

# --- constants -------------------------------------------------------------------------------
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right
ACTION_CHARS = "UDLR"
ACTION_NAMES_TH = ("ขึ้น", "ลง", "ซ้าย", "ขวา")
OBS_SIZE = 7  # CNN input is always 4 x 7 x 7
MAX_STEPS = 50

R_STEP = -0.1
R_BOX_ON = 1.0
R_BOX_OFF = -1.0
R_WIN = 10.0
R_DEADLOCK = -5.0

Pos = tuple[int, int]
Boxes = frozenset[Pos]


class LevelError(ValueError):
    """Raised when a level text is malformed."""


# --- level -----------------------------------------------------------------------------------
@dataclass(eq=False)
class Level:
    height: int
    width: int
    walls: frozenset[Pos]
    targets: frozenset[Pos]
    player: Pos
    boxes: Boxes
    name: str = ""
    bfs_solution: str | None = field(default=None)

    @property
    def bfs_len(self) -> int | None:
        return None if self.bfs_solution is None else len(self.bfs_solution)

    @cached_property
    def _static_layers(self) -> tuple[np.ndarray, np.ndarray]:
        """Wall and target planes, padded with walls up to OBS_SIZE x OBS_SIZE."""
        walls = np.ones((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        walls[: self.height, : self.width] = 0.0
        for r, c in self.walls:
            walls[r, c] = 1.0
        targets = np.zeros((OBS_SIZE, OBS_SIZE), dtype=np.float32)
        for r, c in self.targets:
            targets[r, c] = 1.0
        return walls, targets


def parse_level(lines: list[str], name: str = "") -> Level:
    """Parse and validate a level drawn with ``# @ + $ * .`` and spaces."""
    rows = [ln.rstrip("\r\n") for ln in lines]
    while rows and not rows[-1].strip():
        rows.pop()
    while rows and not rows[0].strip():
        rows.pop(0)
    if not rows:
        raise LevelError("ด่านว่างเปล่า")
    height, width = len(rows), len(rows[0])
    if any(len(r) != width for r in rows):
        raise LevelError("ทุกแถวต้องยาวเท่ากัน (ใช้ช่องว่างแทนพื้น)")
    if height > OBS_SIZE or width > OBS_SIZE:
        raise LevelError(f"รองรับสูงสุด {OBS_SIZE} x {OBS_SIZE} (ได้ {height} x {width})")
    walls, targets, boxes, players = set(), set(), set(), []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            pos = (r, c)
            if ch == "#":
                walls.add(pos)
            elif ch == " ":
                pass
            elif ch == ".":
                targets.add(pos)
            elif ch == "$":
                boxes.add(pos)
            elif ch == "*":
                boxes.add(pos)
                targets.add(pos)
            elif ch == "@":
                players.append(pos)
            elif ch == "+":
                players.append(pos)
                targets.add(pos)
            else:
                raise LevelError(f"สัญลักษณ์ไม่รู้จัก {ch!r} ที่แถว {r + 1} คอลัมน์ {c + 1}")
    border = {(r, c) for r in range(height) for c in range(width)
              if r in (0, height - 1) or c in (0, width - 1)}
    if not border <= walls:
        raise LevelError("ขอบนอกของด่านต้องเป็นกำแพง (#) ทั้งหมด")
    if len(players) != 1:
        raise LevelError(f"ต้องมีผู้เล่น 1 คน (พบ {len(players)})")
    if not boxes:
        raise LevelError("ต้องมีกล่องอย่างน้อย 1 ใบ")
    if len(boxes) != len(targets):
        raise LevelError(f"จำนวนกล่อง ({len(boxes)}) ต้องเท่ากับจำนวนเป้า ({len(targets)})")
    return Level(height, width, frozenset(walls), frozenset(targets),
                 players[0], frozenset(boxes), name=name)


def load_level_file(path: str | Path) -> Level:
    path = Path(path)
    return parse_level(path.read_text(encoding="utf-8").splitlines(), name=path.stem)


def level_to_lines(level: Level, player: Pos | None = None, boxes: Boxes | None = None) -> list[str]:
    player = level.player if player is None else player
    boxes = level.boxes if boxes is None else boxes
    out = []
    for r in range(level.height):
        row = []
        for c in range(level.width):
            p = (r, c)
            on_t = p in level.targets
            if p in level.walls:
                row.append("#")
            elif p == player:
                row.append("+" if on_t else "@")
            elif p in boxes:
                row.append("*" if on_t else "$")
            else:
                row.append("." if on_t else " ")
        out.append("".join(row))
    return out


def save_levels(levels: list[Level], path: str | Path) -> None:
    data = [{"id": lv.name, "map": level_to_lines(lv), "bfs_len": lv.bfs_len,
             "bfs_solution": lv.bfs_solution} for lv in levels]
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_levels(path: str | Path) -> list[Level]:
    out = []
    for item in json.loads(Path(path).read_text(encoding="utf-8")):
        lv = parse_level(item["map"], name=item["id"])
        lv.bfs_solution = item.get("bfs_solution")
        out.append(lv)
    return out


# --- rules -----------------------------------------------------------------------------------
def move(level: Level, player: Pos, boxes: Boxes, action: int):
    """Apply one action. Returns ``(player, boxes, push)``; ``push`` is ``(src, dst)`` or None."""
    dr, dc = ACTIONS[action]
    nxt = (player[0] + dr, player[1] + dc)
    if nxt in level.walls:
        return player, boxes, None
    if nxt in boxes:
        beyond = (nxt[0] + dr, nxt[1] + dc)
        if beyond in level.walls or beyond in boxes:
            return player, boxes, None
        return nxt, (boxes - {nxt}) | {beyond}, (nxt, beyond)
    return nxt, boxes, None


def is_deadlock(level: Level, boxes: Boxes) -> bool:
    """A box on a non-target cell stuck in a wall corner can never move again."""
    walls = level.walls
    for r, c in boxes:
        if (r, c) in level.targets:
            continue
        vertical = (r - 1, c) in walls or (r + 1, c) in walls
        horizontal = (r, c - 1) in walls or (r, c + 1) in walls
        if vertical and horizontal:
            return True
    return False


def to_obs(level: Level, player: Pos, boxes: Boxes) -> np.ndarray:
    """4 planes (wall, player, box, target) of shape (4, 7, 7), float32 in {0, 1}."""
    walls, targets = level._static_layers
    obs = np.zeros((4, OBS_SIZE, OBS_SIZE), dtype=np.float32)
    obs[0] = walls
    obs[1][player] = 1.0
    for b in boxes:
        obs[2][b] = 1.0
    obs[3] = targets
    return obs


# --- environment -----------------------------------------------------------------------------
class SokobanEnv:
    """Gym-like environment. ``step`` returns ``(obs, reward, done, truncated, info)``.

    ``done`` = win or deadlock (true terminal). ``truncated`` = step limit (not terminal,
    so value bootstrapping must continue).
    """

    def __init__(self, levels: list[Level] | None = None, max_steps: int = MAX_STEPS, seed: int = 0):
        self.levels = levels or []
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.level: Level | None = None
        self.player: Pos = (0, 0)
        self.boxes: Boxes = frozenset()
        self.steps = 0

    def reset(self, level: Level | None = None) -> np.ndarray:
        if level is None:
            level = self.levels[int(self.rng.integers(len(self.levels)))]
        self.level = level
        self.player, self.boxes, self.steps = level.player, level.boxes, 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return to_obs(self.level, self.player, self.boxes)

    def state_key(self) -> tuple[Pos, Boxes]:
        return self.player, self.boxes

    def step(self, action: int):
        level = self.level
        self.player, self.boxes, push = move(level, self.player, self.boxes, action)
        self.steps += 1
        reward = R_STEP
        if push is not None:
            src, dst = push
            reward += R_BOX_ON * ((dst in level.targets) and (src not in level.targets))
            reward += R_BOX_OFF * ((src in level.targets) and (dst not in level.targets))
        solved = self.boxes == level.targets
        deadlock = (not solved) and push is not None and is_deadlock(level, self.boxes)
        if solved:
            reward += R_WIN
        elif deadlock:
            reward += R_DEADLOCK
        done = solved or deadlock
        truncated = (not done) and self.steps >= self.max_steps
        info = {"solved": solved, "deadlock": deadlock, "steps": self.steps}
        return self.observation(), reward, done, truncated, info
