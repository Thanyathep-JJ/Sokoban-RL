import pytest

from sokoban_rl.dqn import DQNConfig, DQNPolicy, QNet, ReplayBuffer, load_model, train
from sokoban_rl.evaluation import evaluate, run_episode
from sokoban_rl.game import load_levels, save_levels
from sokoban_rl.levelgen import generate_split
from sokoban_rl.policies import BFSPolicy, QTablePolicy, RandomPolicy
from sokoban_rl.qlearning import train_q
from sokoban_rl.render import save_gif


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q_learning_matches_bfs_on_example(example_level, seed):
    q = train_q(example_level, episodes=3000, seed=seed)
    res = run_episode(example_level, QTablePolicy(q))
    assert res.solved and res.actions == "URD"


def test_bfs_policy_is_perfect_and_random_is_worse():
    levels, _ = generate_split(30, seed=11)
    best = evaluate(levels, BFSPolicy())
    assert best["success_pct"] == 100 and best["ratio_vs_bfs"] == pytest.approx(1.0)
    rand = evaluate(levels, RandomPolicy(0))
    assert rand["success_pct"] < 60


def test_qnet_shape_and_buffer_roundtrip():
    import numpy as np
    import torch

    assert QNet()(torch.zeros(5, 4, 7, 7)).shape == (5, 4)
    buf = ReplayBuffer(8)
    for i in range(12):  # wraps around
        buf.add(np.ones((4, 7, 7)), i % 4, 1.0, np.zeros((4, 7, 7)), 0.0)
    assert buf.size == 8
    s, a, r, s2, d = buf.sample(4, np.random.default_rng(0))
    assert s.shape == (4, 4, 7, 7) and s.dtype == torch.float32 and s2.sum() == 0


def test_dqn_smoke_train_save_load(tmp_path):
    for split, n, seed in (("train", 40, 1), ("val", 10, 2)):
        levels, _ = generate_split(n, seed=seed)
        save_levels(levels, tmp_path / f"s2_{split}.json")
    cfg = DQNConfig(steps=400, learn_start=64, batch_size=16, eval_every=200, target_update=100,
                    data_dir=str(tmp_path), out_dir=str(tmp_path / "m"), double=True)
    out = train(cfg, log=lambda *_: None)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert len((out / "log.csv").read_text().splitlines()) == 3  # header + 2 evals
    res = run_episode(load_levels(tmp_path / "s2_val.json")[0], DQNPolicy(load_model(out / "best.pt")))
    assert res.steps <= 50


def test_gif_export(tmp_path, example_level):
    path = save_gif(example_level, "URD", tmp_path / "x.gif")
    assert path.exists() and path.stat().st_size > 0
