"""Baseline policies. Every policy has ``reset(level)`` and ``act(env) -> int``."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from sokoban_rl.game import ACTION_CHARS, Level, SokobanEnv
from sokoban_rl.solver import solve


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, level: Level) -> None:
        pass

    def act(self, env: SokobanEnv) -> int:
        return int(self.rng.integers(4))


class BFSPolicy:
    """Follows the optimal path (the upper bound AI is compared against)."""

    name = "bfs"

    def reset(self, level: Level) -> None:
        sol = level.bfs_solution if level.bfs_solution is not None else solve(level)
        self.plan = [ACTION_CHARS.index(c) for c in (sol or "")]
        self.i = 0

    def act(self, env: SokobanEnv) -> int:
        a = self.plan[self.i] if self.i < len(self.plan) else 0
        self.i += 1
        return a


class QTablePolicy:
    name = "qtable"

    def __init__(self, q: defaultdict):
        self.q = q

    def reset(self, level: Level) -> None:
        pass

    def act(self, env: SokobanEnv) -> int:
        return int(np.argmax(self.q[env.state_key()]))  # epsilon = 0
