import io

from sokoban_rl.cli import main
from sokoban_rl.evaluation import EpisodeResult
from sokoban_rl.human import (
    HumanResult,
    Session,
    build_report,
    count_wasted,
    make_key_reader,
    play_interactive,
)


def keys(*items):
    it = iter(items)
    return lambda: next(it, "quit")


def test_session_moves_push_and_win(example_level):
    s = Session(example_level)
    assert s.press("U") == "moved"
    assert s.press("R") == "moved"
    assert not s.won
    assert s.press("D") == "pushed"
    assert s.won
    assert s.actions == "URD"


def test_wall_bump_is_not_counted_as_a_move(example_level):
    s = Session(example_level)
    assert s.press("U") == "moved"  # row 2 -> row 1; the next U hits the top wall
    assert s.press("U") == "blocked"
    assert s.press("U") == "blocked"
    assert s.actions == "U"
    assert s.bumps == 2


def test_undo_and_restart(example_level):
    s = Session(example_level)
    s.press("U")
    s.press("R")
    assert s.undo()
    assert s.actions == "U"
    assert s.undos == 1
    s.restart()
    assert s.actions == "" and s.restarts == 1
    assert (s.player, s.boxes) == (example_level.player, example_level.boxes)
    assert not s.undo()  # nothing left to undo


def test_stuck_when_box_pushed_into_corner(example_level):
    s = Session(example_level)
    for a in "RDRU":  # push the box right, then up into the top right corner
        s.press(a)
    assert s.stuck and not s.won
    s.undo()
    assert not s.stuck


def test_play_interactive_win(example_level):
    out = io.StringIO()
    res = play_interactive(example_level, keys("U", "R", "D"), out)
    assert res.outcome == "won" and res.actions == "URD"
    assert "@" in out.getvalue()


def test_give_up_needs_confirmation(example_level):
    res = play_interactive(example_level, keys("U", "giveup", "no", "R", "giveup", "yes"), io.StringIO())
    assert res.outcome == "gave_up" and res.actions == "UR"


def test_quit_and_end_of_input(example_level):
    assert play_interactive(example_level, keys("quit"), io.StringIO()).outcome == "quit"
    assert play_interactive(example_level, keys(), io.StringIO()).outcome == "quit"


def test_key_reader_from_pipe():
    read = make_key_reader(io.StringIO("wasd\nZ"))
    assert [read() for _ in range(5)] == ["U", "L", "D", "R", "undo"]
    assert read() == "quit"  # end of input


def test_count_wasted_counts_wall_bumps(example_level):
    assert count_wasted(example_level, "UUUU") == 3  # one real move up, then three wall bumps
    assert count_wasted(example_level, "URD") == 0


def test_report_covers_all_three_players(example_level):
    human = HumanResult("won", "URD", undos=1, restarts=0, bumps=0)
    agent = EpisodeResult(True, False, 3, 3, "URD")
    text = "\n".join(build_report(example_level, human, "dqn", agent, "URD"))
    assert "คุณ: ชนะ · 3 ก้าว" in text and "ย้อน 1 ครั้ง" in text
    assert "AI (dqn): ชนะ · 3 ก้าว" in text
    assert "BFS" in text and "เท่ากัน" in text

    gave_up = HumanResult("gave_up", "U", undos=0, restarts=0, bumps=0)
    lost = EpisodeResult(False, False, 50, 3, "U" * 50)
    text = "\n".join(build_report(example_level, gave_up, "dqn", lost, "URD"))
    assert "ยอมแพ้" in text and "แพ้" in text and "เดินชนกำแพงเปล่า" in text


def test_challenge_command_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("wds\n"))  # W = up, D = right, S = down
    main(["challenge", "--file", "data/example.txt", "--agent", "qtable", "--no-anim"])
    out = capsys.readouterr().out
    assert "คุณ: ชนะ · 3 ก้าว" in out
    assert "AI (qtable): ชนะ · 3 ก้าว" in out
