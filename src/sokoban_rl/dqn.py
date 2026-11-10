"""Stages 2-3: Deep Q-Network (small CNN, replay buffer, target network)."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sokoban_rl.evaluation import evaluate
from sokoban_rl.game import OBS_SIZE, Level, SokobanEnv, load_levels


@dataclass
class DQNConfig:
    stage: int = 2
    seed: int = 0
    steps: int = 300_000
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    learn_start: int = 2_000
    train_freq: int = 1
    target_update: int = 1_000
    eval_every: int = 10_000
    double: bool = False
    init: str | None = None  # warm-start weights (stage 2 -> stage 3)
    threads: int | None = None
    data_dir: str = "data"
    out_dir: str | None = None  # default: models/s{stage}_seed{seed}


class QNet(nn.Module):
    """4x7x7 -> Conv32 -> Conv32 -> FC128 -> 4 Q-values."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * OBS_SIZE * OBS_SIZE, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class ReplayBuffer:
    """Preallocated ring buffer; observations are binary so ``uint8`` is lossless."""

    def __init__(self, capacity: int):
        shape = (capacity, 4, OBS_SIZE, OBS_SIZE)
        self.obs = np.zeros(shape, np.uint8)
        self.next_obs = np.zeros(shape, np.uint8)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.action[i], self.reward[i], self.done[i] = action, reward, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, batch_size)
        return (
            torch.from_numpy(self.obs[idx].astype(np.float32)),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx].astype(np.float32)),
            torch.from_numpy(self.done[idx]),
        )


class DQNPolicy:
    """Greedy (epsilon = 0) policy backed by a Q-network."""

    name = "dqn"

    def __init__(self, net: QNet):
        self.net = net.eval()

    def reset(self, level: Level) -> None:
        pass

    @torch.no_grad()
    def act(self, env: SokobanEnv) -> int:
        obs = torch.from_numpy(env.observation()).unsqueeze(0)
        return int(self.net(obs).argmax(1).item())


def load_model(path: str | Path) -> QNet:
    net = QNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: DQNConfig, log=print) -> Path:
    """Train and return the output directory (contains best.pt, last.pt, log.csv, config.json)."""
    _seed_everything(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    data = Path(cfg.data_dir)
    train_levels = load_levels(data / f"s{cfg.stage}_train.json")
    val_levels = load_levels(data / f"s{cfg.stage}_val.json")
    out = Path(cfg.out_dir or f"models/s{cfg.stage}_seed{cfg.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    net, target = QNet(), QNet()
    if cfg.init:
        net.load_state_dict(torch.load(cfg.init, map_location="cpu", weights_only=True))
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    env = SokobanEnv(train_levels, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    buffer = ReplayBuffer(cfg.buffer_size)
    returns: deque[float] = deque(maxlen=100)
    ep_return, best_val, start = 0.0, -1.0, time.time()
    obs = env.reset()

    log_file = open(out / "log.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["step", "avg_return_100", "val_success", "val_deadlock", "val_ratio",
                     "epsilon", "minutes"])

    for t in range(1, cfg.steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * t / cfg.eps_decay_steps)
        if rng.random() < eps:
            action = int(rng.integers(4))
        else:
            with torch.no_grad():
                action = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax(1).item())

        next_obs, reward, done, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(done))  # truncation is NOT terminal
        ep_return += reward
        if done or truncated:
            returns.append(ep_return)
            ep_return = 0.0
            obs = env.reset()
        else:
            obs = next_obs

        if t >= cfg.learn_start and t % cfg.train_freq == 0:
            s, a, r, s2, d = buffer.sample(cfg.batch_size, rng)
            with torch.no_grad():
                if cfg.double:
                    best_a = net(s2).argmax(1, keepdim=True)
                    next_q = target(s2).gather(1, best_a).squeeze(1)
                else:
                    next_q = target(s2).max(1).values
                y = r + cfg.gamma * (1.0 - d) * next_q
            q = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

        if t % cfg.target_update == 0:
            target.load_state_dict(net.state_dict())

        if t % cfg.eval_every == 0:
            val = evaluate(val_levels, DQNPolicy(net))
            net.train()
            avg_ret = float(np.mean(returns)) if returns else float("nan")
            minutes = (time.time() - start) / 60
            writer.writerow([t, f"{avg_ret:.3f}", f"{val['success_pct']:.1f}",
                             f"{val['deadlock_pct']:.1f}", f"{val['ratio_vs_bfs']:.3f}",
                             f"{eps:.3f}", f"{minutes:.2f}"])
            log_file.flush()
            marker = ""
            if val["success_pct"] > best_val:
                best_val = val["success_pct"]
                torch.save(net.state_dict(), out / "best.pt")
                marker = "  * best"
            log(f"step {t:>7d}  return {avg_ret:6.2f}  val {val['success_pct']:5.1f}%  "
                f"deadlock {val['deadlock_pct']:5.1f}%  eps {eps:.2f}  {minutes:5.1f} min{marker}")

    log_file.close()
    torch.save(net.state_dict(), out / "last.pt")
    if best_val < 0:  # eval_every > steps: still leave a usable best.pt
        torch.save(net.state_dict(), out / "best.pt")
    return out
