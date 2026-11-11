from sokoban_rl.designer import _shape_key, generate_designed, pull_distances, push_runs
from sokoban_rl.game import Level, parse_level
from sokoban_rl.solver import solve


def test_push_runs_split_on_turns_and_walks():
    lv = parse_level(["#######", "#     #", "# @$  #", "#     #", "#   . #", "#######"])
    assert push_runs(lv, "RR") == [2]  # two pushes the same way: one line
    assert push_runs(lv, "RLRR") == [1, 1]  # stepping back, then pushing again: a new line
    assert push_runs(lv, "RURDD") == [1, 2]  # walk round the box, then push down: a turn


def test_pull_distances_give_solvable_starts():
    lv = parse_level(["######", "#    #", "# #  #", "#  .@#", "# $  #", "######"])
    target = next(iter(lv.targets))
    floor = [(r, c) for r in range(6) for c in range(6) if (r, c) not in lv.walls]
    dist = pull_distances(set(lv.walls), target, floor)
    for (player, box), need in dist.items():
        if box == target:
            assert need == 0
            continue
        sol = solve(Level(6, 6, lv.walls, lv.targets, player, frozenset({box})))
        assert sol is not None  # every state found by pulling can be pushed back
        assert sum(push_runs(Level(6, 6, lv.walls, lv.targets, player, frozenset({box})), sol)) >= need


def test_designed_levels_meet_their_limits():
    levels = generate_designed(8, 5, 7, ("pieces", 2, 3), far=1.0, min_len=12, max_len=25, min_lines=3)
    keys = {_shape_key(lv) for lv in levels}
    assert len(keys) == 8  # no two share room, target and box
    for lv in levels:
        assert 12 <= lv.bfs_len <= 25
        runs = push_runs(lv, lv.bfs_solution)
        assert len(runs) >= 3 and max(runs) <= 3
        assert len(solve(lv)) == lv.bfs_len  # truly the shortest


def test_designed_levels_avoid_excluded_shapes():
    first = generate_designed(3, 9, 6, ("random", 1, 3), far=0.6, min_len=6, max_len=14, min_lines=2, max_lines=3)
    again = generate_designed(3, 9, 6, ("random", 1, 3), far=0.6, min_len=6, max_len=14, min_lines=2, max_lines=3,
                              exclude={_shape_key(lv) for lv in first})
    assert not {_shape_key(lv) for lv in first} & {_shape_key(lv) for lv in again}
