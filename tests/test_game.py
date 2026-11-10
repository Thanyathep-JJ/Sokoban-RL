import pytest

from sokoban_rl.game import (
    LevelError,
    SokobanEnv,
    is_deadlock,
    level_to_lines,
    move,
    parse_level,
    to_obs,
)

U, D, L, R = range(4)


def run(env, actions):
    total, info = 0.0, {}
    for a in actions:
        _, r, done, trunc, info = env.step(a)
        total += r
        if done or trunc:
            break
    return total, info, done, trunc


def test_wall_bump_keeps_position_and_costs_step(example_level):
    env = SokobanEnv()
    env.reset(example_level)
    env.step(U)
    env.step(U)  # row 1 -> row 0 is a wall
    assert env.player == (1, 2)
    _, r, *_ = env.step(U)
    assert r == pytest.approx(-0.1)


def test_push_box_one_cell(example_level):
    player, boxes, push = move(example_level, (2, 2), example_level.boxes, R)
    assert player == (2, 3) and boxes == frozenset({(2, 4)}) and push == ((2, 3), (2, 4))


def test_push_blocked_by_wall_or_box():
    lv = parse_level(["######", "#@$$ #", "#  . #", "#  . #", "#    #", "######"])
    # box (1,2) is blocked by box (1,3): nobody moves
    assert move(lv, (1, 1), lv.boxes, R)[:2] == ((1, 1), lv.boxes)
    lv2 = parse_level(["######", "# @$$#", "#  . #", "#  . #", "#    #", "######"])
    # box (1,4) next to the wall; pushing (1,3) right is blocked by box then wall
    assert move(lv2, (1, 2), lv2.boxes, R)[:2] == ((1, 2), lv2.boxes)


def test_example_win_sequence_and_reward(example_level):
    env = SokobanEnv()
    env.reset(example_level)
    total, info, done, trunc = run(env, [U, R, D])
    assert info["solved"] and done and not trunc
    assert total == pytest.approx(-0.1 - 0.1 + (-0.1 + 1 + 10))


def test_corner_deadlock_ends_game():
    lv = parse_level(["######", "#    #", "#$   #", "# @ .#", "#    #", "######"])
    env = SokobanEnv()
    env.reset(lv)
    # L -> (3,1); U pushes the box (2,1) into the top-left corner (1,1)
    total, info, done, _ = run(env, [L, U])
    assert done and info["deadlock"] and not info["solved"]
    assert total == pytest.approx(-0.1 + (-0.1 - 5.0))


def test_box_on_target_corner_is_not_deadlock():
    lv = parse_level(["######", "#*   #", "#  @ #", "#    #", "#    #", "######"])
    assert not is_deadlock(lv, lv.boxes)


def test_truncation_after_max_steps(example_level):
    env = SokobanEnv(max_steps=50)
    env.reset(example_level)
    for _ in range(50):
        _, _, done, trunc, _ = env.step(U)  # bump into the wall forever
    assert trunc and not done


def test_obs_shape_and_padding(example_level):
    obs = to_obs(example_level, example_level.player, example_level.boxes)
    assert obs.shape == (4, 7, 7)
    assert obs[0][6].all() and obs[0][:, 6].all()  # padded row/col are walls
    assert obs[1].sum() == 1 and obs[2].sum() == 1 and obs[3].sum() == 1
    assert obs[1][2, 2] == 1 and obs[2][2, 3] == 1 and obs[3][3, 3] == 1


def test_round_trip_text(example_level):
    assert level_to_lines(example_level) == ["######", "#    #", "# @$ #", "#  . #", "#    #", "######"]


@pytest.mark.parametrize("bad", [
    ["######", "#    #", "#  $ #", "#  . #", "#    #", "######"],           # no player
    ["######", "# @$ #", "#    #", "#    #", "#    #", "######"],           # box != target
    ["######", "     #", "# @$ #", "#  . #", "#    #", "######"],           # open border
    ["######", "#@  @#", "# $  #", "#  . #", "#    #", "######"],           # two players
    ["########", "#      #", "# @$.  #", "#      #", "#      #", "#      #", "#      #", "########"],  # too big
    ["######", "#    #", "# @$ #", "#  . #", "#   #", "######"],             # ragged
])
def test_parse_rejects_bad_levels(bad):
    with pytest.raises(LevelError):
        parse_level(bad)
