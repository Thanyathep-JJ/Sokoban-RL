"""Policy-agnostic evaluation: success rate, path length vs BFS, deadlock rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sokoban_rl.game import ACTION_CHARS, MAX_STEPS, Level, SokobanEnv
from sokoban_rl.solver import solve


@dataclass
class EpisodeResult:
    solved: bool
    deadlock: bool
    steps: int
    bfs_len: int | None
    actions: str


def run_episode(level: Level, policy, max_steps: int = MAX_STEPS) -> EpisodeResult:
    env = SokobanEnv(max_steps=max_steps)
    env.reset(level)
    policy.reset(level)
    actions, info = [], {"solved": False, "deadlock": False, "steps": 0}
    done = truncated = False
    while not (done or truncated):
        a = policy.act(env)
        actions.append(ACTION_CHARS[a])
        _, _, done, truncated, info = env.step(a)
    bfs_len = level.bfs_len if level.bfs_solution is not None else _len_or_none(solve(level))
    return EpisodeResult(info["solved"], info["deadlock"], info["steps"], bfs_len, "".join(actions))


def _len_or_none(sol: str | None) -> int | None:
    return None if sol is None else len(sol)


def evaluate(levels: list[Level], policy, max_steps: int = MAX_STEPS) -> dict[str, float]:
    results = [run_episode(lv, policy, max_steps) for lv in levels]
    solved = [r for r in results if r.solved]
    ratios = [r.steps / r.bfs_len for r in solved if r.bfs_len]
    n = len(results)
    return {
        "success_pct": 100.0 * len(solved) / n,
        "deadlock_pct": 100.0 * sum(r.deadlock for r in results) / n,
        "ratio_vs_bfs": float(np.mean(ratios)) if ratios else float("nan"),
    }
