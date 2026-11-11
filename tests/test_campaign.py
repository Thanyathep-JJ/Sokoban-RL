import os

import pytest

from sokoban_rl.campaign import (
    DIFFICULTIES,
    Progress,
    difficulty_levels,
    difficulty_score,
    fmt_clock,
    level_key,
    order_by_difficulty,
    pick_progression,
    score_for,
    stars_for,
    visible_difficulties,
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
        other = parse_level(["######", "#@$ .#", "######"], name="other")
        modes = {"a": ("A", lambda: ([example_level], progress)), "b": ("B", lambda: ([other], Progress(None)))}
        app = App(root, modes, agent=lambda lv: None, agent_name="none")
        for a in "URD":
            app.command(a)
        assert app.finished == "won"
        assert progress.best["example"]["score"] == 100
        assert "ชนะ" in app.vars["msg"].get() and "ไม่มีโมเดล" in app.vars["ai"].get()
        app.command("restart")
        app.command("D")  # (2,2) -> (3,2)
        app.command("R")  # push nothing: (3,3) is free
        assert app.session.actions == "DR" and not app.finished
        app.command("watch_best")  # ignored until the level is over
        assert app.replay_job is None
        app.command("mode:b")  # switch difficulty
        assert app.level.name == "other" and app.session.actions == ""
        app.command("R")
        app.command("R")
        assert app.finished == "won"
        app.command("mode:a")
        assert app.level.name == "example" and progress.best["example"]["score"] == 100
    finally:
        root.destroy()


def test_difficulty_set_climbs_through_its_parts(tmp_path, monkeypatch):
    spec = {"label": "t", "seed": 5, "parts": [
        {"size": 6, "boxes": 1, "min_len": 3, "max_len": 6, "count": 4, "pool": 40},
        {"size": 7, "boxes": 2, "min_len": 5, "max_len": 9, "count": 4, "pool": 40},
    ]}
    monkeypatch.setitem(DIFFICULTIES, "test_mode", spec)
    levels = difficulty_levels("test_mode", tmp_path)
    assert len(levels) == 8 and (tmp_path / "play_test_mode.json").exists()
    assert [len(lv.boxes) for lv in levels] == [1] * 4 + [2] * 4  # the box is added half way
    assert [lv.width for lv in levels] == [6] * 4 + [7] * 4
    scores = [difficulty_score(lv) for lv in levels]
    assert scores == sorted(scores)  # never easier than the stage before
    assert levels[0].name == "test_mode_01" and levels[-1].name == "test_mode_08"
    again = difficulty_levels("test_mode", tmp_path)  # read back from the cache
    assert [lv.bfs_solution for lv in again] == [lv.bfs_solution for lv in levels]


def test_three_modes_shown_and_extras_kept():
    assert visible_difficulties() == ["easy", "medium", "hard"]
    assert {"two_boxes", "expert"} <= set(DIFFICULTIES)  # hidden, kept for later
    assert all(sum(p["count"] for p in DIFFICULTIES[n]["parts"]) == 20 for n in DIFFICULTIES)


def test_pick_progression_spreads_over_the_whole_range():
    def corridor(n):  # only the solution length matters to pick_progression
        lv = parse_level(["######", "#@$ .#", "######"])
        lv.bfs_solution = "R" * n
        return lv

    pool = [corridor(n) for n in [3] * 30 + [4, 5]] + [corridor(n) for n in range(6, 11)]
    lens = [lv.bfs_len for lv in pick_progression(pool, 4)]
    assert lens == sorted(lens) and lens[0] == 3 and lens[-1] == 10
    assert lens.count(3) == 1  # not crowded at the easy end where most levels are


def test_totals_ignore_levels_from_other_sets(tmp_path):
    p = Progress(tmp_path / "p.json")
    p.record("hard_0012", 9, 5.0, 9)  # a result from an older level set
    p.record("hard_01", 10, 5.0, 8)
    assert p.totals(["hard_01", "hard_02"]) == (80, 2, 1)


def test_one_floor_size_per_mode():
    from sokoban_rl.campaign import DIFFICULTIES as D

    assert [{p["size"] for p in D[n]["parts"]} for n in ("easy", "medium", "hard")] == [{5}, {6}, {7}]
    assert all(p["boxes"] == 1 for n in ("easy", "medium", "hard") for p in D[n]["parts"])


def test_cache_for_a_different_spec_is_rebuilt(tmp_path, monkeypatch):
    spec = {"label": "t", "seed": 5, "parts": [{"size": 6, "boxes": 1, "min_len": 3, "max_len": 6,
                                                "count": 3, "pool": 10}]}
    monkeypatch.setitem(DIFFICULTIES, "m", spec)
    assert all(lv.width == 6 for lv in difficulty_levels("m", tmp_path))
    monkeypatch.setitem(DIFFICULTIES, "m", dict(spec, parts=[dict(spec["parts"][0], size=7)]))
    assert all(lv.width == 7 for lv in difficulty_levels("m", tmp_path))  # old cache not reused


def test_generator_gives_up_instead_of_looping_forever():
    from sokoban_rl.levelgen import generate_split

    with pytest.raises(RuntimeError, match="unique levels"):
        generate_split(5000, 1, size=5, n_boxes=1, min_len=3, max_len=9, max_tries=3000)
