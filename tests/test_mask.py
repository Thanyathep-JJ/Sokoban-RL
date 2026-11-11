import numpy as np
import torch

from sokoban_rl.dqn import DQNConfig, DQNPolicy, QNet, ReplayBuffer, load_model, train, valid_moves
from sokoban_rl.evaluation import run_episode
from sokoban_rl.game import SokobanEnv, load_levels, save_levels
from sokoban_rl.levelgen import generate_split


def test_valid_moves_marks_wall_bumps(example_level):
    # player (2,2): up (1,2), down (3,2) and left (2,1) are free; right pushes the box (2,3) to (2,4)
    assert valid_moves(example_level, (2, 2), example_level.boxes).tolist() == [True, True, True, True]
    # top-left floor corner: up and left are walls, down and right are free
    assert valid_moves(example_level, (1, 1), example_level.boxes).tolist() == [False, True, False, True]


def test_valid_moves_never_all_false():
    from sokoban_rl.game import parse_level

    boxed_in = parse_level(["#####", "#@$.#", "#####"])  # box against... a free target: right is valid
    assert valid_moves(boxed_in, boxed_in.player, boxed_in.boxes).any()


def test_masked_policy_never_wastes_a_move(example_level):
    class AlwaysUp(torch.nn.Module):  # rates "up" best everywhere, even into the top wall
        def forward(self, x):
            return torch.tensor([[5.0, 0.0, 0.0, 0.0]]).repeat(len(x), 1)

    env = SokobanEnv()
    env.reset(example_level)
    env.player = (1, 2)  # directly below the top wall: "up" does nothing
    assert DQNPolicy(AlwaysUp()).act(env) == 0
    assert DQNPolicy(AlwaysUp(), mask=True).act(env) != 0


def test_buffer_keeps_the_next_state_mask():
    buf = ReplayBuffer(4)
    buf.add(np.ones((4, 7, 7)), 0, 1.0, np.zeros((4, 7, 7)), 0.0, np.array([True, False, True, False]))
    buf.add(np.ones((4, 7, 7)), 1, 1.0, np.zeros((4, 7, 7)), 0.0)  # no mask given: everything allowed
    *_, mask = buf.sample(2, np.random.default_rng(0), with_mask=True)
    assert mask.dtype == torch.bool and mask.shape == (2, 4)
    assert len(buf.sample(2, np.random.default_rng(0))) == 5  # old call keeps its old shape


def test_masked_training_smoke(tmp_path):
    for split, n, seed in (("train", 40, 1), ("val", 10, 2)):
        levels, _ = generate_split(n, seed=seed)
        save_levels(levels, tmp_path / f"s2_{split}.json")
    cfg = DQNConfig(steps=400, learn_start=64, batch_size=16, eval_every=200, target_update=100,
                    data_dir=str(tmp_path), out_dir=str(tmp_path / "m"), double=True, mask=True)
    out = train(cfg, log=lambda *_: None)
    net = load_model(out / "best.pt")
    assert isinstance(net, QNet)
    level = load_levels(tmp_path / "s2_val.json")[0]
    res = run_episode(level, DQNPolicy(net, mask=True))
    assert res.steps <= 50
    # a masked policy never produces a move that changes nothing
    from sokoban_rl.human import count_wasted

    assert count_wasted(level, res.actions) == 0
