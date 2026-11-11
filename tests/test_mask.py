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


def test_agent_factory_uses_the_big_model_only_on_5x5_floors(tmp_path):
    from types import SimpleNamespace

    from sokoban_rl.cli import _agent_factory

    small, big = tmp_path / "small" / "s2_seed0", tmp_path / "big"
    for d in (small, big):
        d.mkdir(parents=True)
        torch.save(QNet().state_dict(), d / "best.pt")

    def args(**kw):
        base = dict(agent="dqn", model=None, models_dir=str(tmp_path / "small"), seed=0,
                    big_model=str(big / "best.pt"), mask=True, avoid_repeats=True)
        return SimpleNamespace(**{**base, **kw})

    lv5 = generate_split(1, 1, size=5, n_boxes=1, min_len=3, max_len=9)[0][0]
    lv7 = generate_split(1, 1, size=7, n_boxes=1, min_len=3, max_len=9)[0][0]
    pick = _agent_factory(args())
    assert pick(lv7) is not pick(lv5)  # the 7x7 board gets its own model
    assert pick(lv5) is pick(lv5)  # models are loaded once
    same = _agent_factory(args(mask=False))
    assert same(lv7) is same(lv5)  # the big model needs masking, so without it stage 2 plays everything
    named = _agent_factory(args(model=str(small / "best.pt")))
    assert named(lv7) is named(lv5)  # an explicit --model wins
    gone = _agent_factory(args(big_model=str(tmp_path / "missing.pt")))
    assert gone(lv7) is gone(lv5)  # no big model on disk: fall back to stage 2


def test_avoid_repeats_breaks_a_back_and_forth_loop(example_level):
    class Swing(QNet):
        """Always rates left best, right second: a plain greedy player swings left-right forever."""

        def forward(self, x):
            return torch.tensor([[0.0, 0.0, 2.0, 1.0]]).repeat(x.shape[0], 1)

    plain = run_episode(example_level, DQNPolicy(Swing(), mask=True))
    guarded = run_episode(example_level, DQNPolicy(Swing(), mask=True, avoid_repeats=True))
    positions = len(set(guarded.actions))
    assert not plain.solved
    assert positions >= 2 and guarded.actions != plain.actions  # it leaves the loop and explores
