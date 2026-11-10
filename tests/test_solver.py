from sokoban_rl.game import SokobanEnv, parse_level
from sokoban_rl.solver import solve

ACTION_INDEX = {c: i for i, c in enumerate("UDLR")}


def test_example_is_urd(example_level):
    assert solve(example_level) == "URD"


def test_unsolvable_returns_none():
    # box already frozen in a non-target corner
    lv = parse_level(["######", "#$   #", "#  @ #", "#   .#", "#    #", "######"])
    assert solve(lv) is None


def test_already_solved_is_empty():
    lv = parse_level(["######", "#*   #", "#  @ #", "#    #", "#    #", "######"])
    assert solve(lv) == ""


def test_max_depth_limits_search(example_level):
    assert solve(example_level, max_depth=2) is None
    assert solve(example_level, max_depth=3) == "URD"


def test_solution_wins_in_env():
    lv = parse_level(["######", "#    #", "#@ $ #", "#  . #", "# #  #", "######"])
    sol = solve(lv)
    assert sol is not None
    env = SokobanEnv()
    env.reset(lv)
    for ch in sol:
        _, _, _, _, info = env.step(ACTION_INDEX[ch])
    assert info["solved"] and info["steps"] == len(sol)
