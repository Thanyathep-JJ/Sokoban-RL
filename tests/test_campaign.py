import os

import pytest

from sokoban_rl.campaign import (
    Progress,
    fmt_clock,
    level_key,
    order_by_difficulty,
    score_for,
    stars_for,
)
from sokoban_rl.game import parse_level
from sokoban_rl.human import KEYS


def test_score_and_stars():
    assert score_for(6, 6) == 100 and stars_for(6, 6) == 3
    assert score_for(8, 6) == 75 and stars_for(8, 6) == 2
    assert score_for(30, 6) == 20 and stars_for(30, 6) == 1
    assert score_for(0, 6) == 0


def test_progress_keeps_best_and_survives_restart(tmp_path):
    path = tmp_path / "progress.json"
    p = Progress(path)
    first, improved = p.record("lv1", 8, 20.0, 6)
    assert improved and first["score"] == 75
    _, improved = p.record("lv1", 10, 5.0, 6)  # worse score, faster: not better
    assert not improved
    _, improved = p.record("lv1", 6, 30.0, 6)
    assert improved
    again = Progress(path)  # reload from disk
    assert again.best["lv1"]["moves"] == 6 and again.total_score == 100 and again.total_stars == 3
    assert again.first_unsolved(["lv1", "lv2"]) == 1


def test_equal_score_faster_time_wins(tmp_path):
    p = Progress(tmp_path / "p.json")
    p.record("lv", 6, 30.0, 6)
    _, improved = p.record("lv", 6, 12.5, 6)
    assert improved and p.best["lv"]["seconds"] == 12.5


def test_damaged_progress_file_is_ignored(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert Progress(path).best == {}


def test_order_and_keys():
    a = parse_level(["#####", "#@$.#", "#####"], name="a")
    a.bfs_solution = "R"
    b = parse_level(["######", "#@ $.#", "######"], name="")
    b.bfs_solution = "RR"
    assert order_by_difficulty([b, a]) == [a, b]
    assert level_key(b, 4) == "#4"
    assert fmt_clock(75.9) == "1:15"


def test_thai_keyboard_layout_keys():
    assert [KEYS[c] for c in "ไฟหก"] == ["U", "L", "D", "R"]
    assert KEYS["ผ"] == "undo" and KEYS["เ"] == "giveup"


@pytest.mark.skipif(os.name != "nt" and not os.environ.get("DISPLAY"), reason="needs a display")
def test_gui_win_records_score(tmp_path, example_level):
    tk = pytest.importorskip("tkinter")
    from sokoban_rl.gui import App

    try:
        root = tk.Tk()  # only one Tk per process: a second one sometimes fails to start on Windows
    except tk.TclError as e:
        pytest.skip(f"Tk could not start: {e}")
    root.withdraw()
    try:
        progress = Progress(tmp_path / "p.json")
        app = App(root, [example_level], progress, agent=lambda lv: None, agent_name="none")
        for a in "URD":
            app.command(a)
        assert app.finished == "won"
        assert progress.best["example"]["score"] == 100
        assert "ชนะ" in app.vars["msg"].get() and "ไม่มีโมเดล" in app.vars["msg"].get()
        app.command("restart")
        app.command("D")  # (2,2) -> (3,2)
        app.command("R")  # push nothing: (3,3) is free
        assert app.session.actions == "DR" and not app.finished
        app.command("watch_best")  # ignored until the level is over
        assert app.replay_job is None
    finally:
        root.destroy()
