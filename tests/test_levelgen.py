from sokoban_rl.game import SokobanEnv
from sokoban_rl.levelgen import canonical_key, generate_split
from sokoban_rl.solver import solve

IDX = {c: i for i, c in enumerate("UDLR")}


def test_levels_are_solvable_in_range_and_unique():
    levels, _ = generate_split(40, seed=7, size=6, n_boxes=1, min_len=3, max_len=15)
    assert len({canonical_key(lv) for lv in levels}) == 40
    for lv in levels:
        assert 3 <= lv.bfs_len <= 15
        assert solve(lv) == lv.bfs_solution
        assert lv.height == lv.width == 6 and len(lv.boxes) == 1
        assert len(lv.walls) - 20 <= 3  # 20 border cells + at most 3 inner walls


def test_solution_replays_to_a_win():
    (lv,), _ = generate_split(1, seed=3)
    env = SokobanEnv()
    env.reset(lv)
    for ch in lv.bfs_solution:
        *_, info = env.step(IDX[ch])
    assert info["solved"]


def test_exclude_prevents_overlap_between_splits():
    a, _ = generate_split(30, seed=1)
    keys = {canonical_key(lv) for lv in a}
    b, _ = generate_split(30, seed=1, exclude=keys)  # same seed would collide without exclude
    assert not keys & {canonical_key(lv) for lv in b}


def test_two_box_levels():
    levels, _ = generate_split(10, seed=5, size=7, n_boxes=2)
    assert all(len(lv.boxes) == 2 and lv.height == 7 for lv in levels)


def test_canonical_key_ignores_rotation_and_reflection(example_level):
    from sokoban_rl.game import level_to_lines, parse_level

    rotated = ["".join(row) for row in zip(*level_to_lines(example_level)[::-1], strict=True)]
    assert canonical_key(parse_level(rotated)) == canonical_key(example_level)
