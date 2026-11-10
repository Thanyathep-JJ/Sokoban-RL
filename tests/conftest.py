import pytest

from sokoban_rl.game import parse_level

EXAMPLE = ["######", "#    #", "# @$ #", "#  . #", "#    #", "######"]


@pytest.fixture
def example_level():
    return parse_level(EXAMPLE, name="example")
