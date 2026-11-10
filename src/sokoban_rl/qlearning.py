"""Stage 1: tabular Q-Learning on one fixed level."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from sokoban_rl.game import Level, SokobanEnv


def train_q(
    level: Level,
    episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.8,
    seed: int = 0,
) -> defaultdict:
    """Return the learned Q-table ``{(player, boxes): array of 4 values}``."""
    rng = np.random.default_rng(seed)
    env = SokobanEnv([level], seed=seed)
    q: defaultdict = defaultdict(lambda: np.zeros(4))
    decay_eps = max(1, int(episodes * decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        env.reset()
        done = truncated = False
        while not (done or truncated):
            s = env.state_key()
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                values = q[s]
                a = int(rng.choice(np.flatnonzero(values == values.max())))  # random tie-break
            _, r, done, truncated, _ = env.step(a)
            target = r if done else r + gamma * q[env.state_key()].max()  # timeout still bootstraps
            q[s][a] += alpha * (target - q[s][a])
    return q
