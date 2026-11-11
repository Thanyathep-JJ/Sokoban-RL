import io
import json
import os

from sokoban_rl.monitor import estimate, fmt_duration, read_run, render, watch

T = 1_000_000_000.0  # pretend the log was last written at this moment


def make_run(tmp_path, rows, total=100_000, double=True):
    run = tmp_path / "models_double" / "s2_seed2"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"steps": total, "eval_every": 10_000, "double": double}))
    lines = ["step,avg_return_100,val_success,val_deadlock,val_ratio,epsilon,minutes"]
    lines += [f"{s},0.5,{v},0.0,1.0,{e},{m}" for s, v, e, m in rows]
    log = run / "log.csv"
    log.write_text("\n".join(lines) + "\n")
    os.utime(log, (T, T))
    return run


def test_fmt_duration():
    assert fmt_duration(0.5) == "30s"
    assert fmt_duration(12.5) == "12m 30s"
    assert fmt_duration(75) == "1h 15m"


def test_estimate_interpolates_between_checkpoints(tmp_path):
    run = read_run(make_run(tmp_path, [(10_000, 10, 0.9, 2.0), (20_000, 20, 0.8, 4.0)]))
    p = estimate(run, T + 60)  # one minute after the last row, speed is 5,000 steps per minute
    assert p.step == 25_000 and abs(p.fraction - 0.25) < 1e-9
    assert abs(p.left_min - 15) < 1e-9 and abs(p.elapsed_min - 5) < 1e-9
    assert not p.finished and p.overdue_min is None


def test_estimate_never_runs_ahead_of_the_next_checkpoint(tmp_path):
    run = read_run(make_run(tmp_path, [(10_000, 10, 0.9, 2.0)]))
    assert estimate(run, T + 20 * 60).step == 20_000


def test_render_shows_bar_percent_and_time_left(tmp_path):
    run = read_run(make_run(tmp_path, [(10_000, 10, 0.9, 2.0), (20_000, 25, 0.8, 4.0)]))
    text = "\n".join(render(run, T + 60, bar_width=20))
    assert "25.0%" in text and "█████░░░░░░░░░░░░░░░" in text
    assert "left 15m 00s" in text and "done around" in text
    assert "validation 25% (best 25%)" in text and "Double DQN" in text


def test_finished_run(tmp_path):
    run = read_run(make_run(tmp_path, [(100_000, 80, 0.05, 20.0)]))
    p = estimate(run, T + 5)
    assert p.finished and p.fraction == 1.0
    assert "100.0%" in render(run, T + 5)[0] and "finished" in render(run, T + 5)[1]


def test_overdue_warning_when_log_stops_updating(tmp_path):
    run = read_run(make_run(tmp_path, [(10_000, 10, 0.9, 2.0), (20_000, 20, 0.8, 4.0)]))
    assert "may have stopped" in render(run, T + 3600)[2]


def test_before_first_checkpoint_and_half_written_row(tmp_path):
    path = make_run(tmp_path, [])
    assert "waiting for the first checkpoint" in render(read_run(path), T + 30)[1]
    with (path / "log.csv").open("a") as f:
        f.write("10000,0.5,12.0,0.0,1.0,0.9,2.0\n20000,0.5")  # the training is mid-write
    assert [r["step"] for r in read_run(path).rows] == [10_000]


def test_watch_once_prints_plain_text(tmp_path):
    path = make_run(tmp_path, [(10_000, 10, 0.9, 2.0)])
    out = io.StringIO()
    watch([path], once=True, out=out, clock=lambda: T + 30)
    text = out.getvalue()
    assert "models_double/s2_seed2" in text and "%" in text and "\033" not in text


def test_watch_redraws_in_place_on_a_terminal(tmp_path):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    calls = []

    def sleep(_):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt  # the user pressed Ctrl+C

    path = make_run(tmp_path, [(10_000, 10, 0.9, 2.0)])
    out = Terminal()
    watch([path], out=out, clock=lambda: T + 30, sleep=sleep)
    text = out.getvalue()
    assert text.count("\033[3A") == 1  # two draws; the second first moves back up over the three lines
    assert "\033[K" in text and len(calls) == 2
