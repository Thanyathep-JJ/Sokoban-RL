import pytest

from sokoban_rl.deadlock import dead_boxes, is_dead, live_squares
from sokoban_rl.game import ACTION_CHARS, is_deadlock, move, parse_level
from sokoban_rl.levelgen import generate_split


def test_live_squares_of_example(example_level):
    # the target is (3, 3); a box can only be pushed onto it from this 2 x 2 area
    assert live_squares(example_level) == {(2, 2), (2, 3), (3, 2), (3, 3)}


def test_box_along_a_wall_without_target_is_dead(example_level):
    boxes = frozenset({(1, 3)})  # against the top wall, not a corner
    assert not is_deadlock(example_level, boxes)  # the old corner rule misses it
    assert dead_boxes(example_level, boxes) == {(1, 3)}


def test_two_by_two_block_is_frozen():
    level = parse_level(["#######", "#    @#", "# $$  #", "# $$  #", "#.... #", "#######"])
    assert all(b in live_squares(level) for b in level.boxes)  # every box alone could still move
    assert not is_deadlock(level, level.boxes)
    assert dead_boxes(level, level.boxes) == set(level.boxes)


def test_boxes_on_targets_are_never_dead():
    level = parse_level(["######", "#*   #", "#  @ #", "######"])
    assert not is_dead(level, level.boxes)


@pytest.mark.parametrize("size,n_boxes,seed", [(6, 1, 11), (7, 2, 12)])
def test_never_flags_a_solvable_state(size, n_boxes, seed):
    """Soundness: no state on any optimal solution path may be called a deadlock."""
    levels, _ = generate_split(12, seed, size=size, n_boxes=n_boxes, max_len=12)
    for lv in levels:
        player, boxes = lv.player, lv.boxes
        assert not is_dead(lv, boxes)
        for ch in lv.bfs_solution:
            player, boxes, _ = move(lv, player, boxes, ACTION_CHARS.index(ch))
            assert not is_dead(lv, boxes), (lv.name, ch)
