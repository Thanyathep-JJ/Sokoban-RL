"""Command line interface: ``sokoban-rl <command>`` (or ``python -m sokoban_rl <command>``)."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from sokoban_rl.game import (
    ACTION_CHARS,
    ACTION_NAMES_TH,
    MAX_STEPS,
    level_to_lines,
    load_level_file,
    load_levels,
)

# stage 2: 6x6, 1 box; stage 3: 7x7, 2 boxes (game mode "hard"); stage 4: 7x7, 3 boxes ("expert")
# stage 5: 1 box on boards of 5x5, 6x6 and 7x7 (floors 3x3 to 5x5), used by the game's modes
STAGE_DEFAULTS = {2: {"size": 6, "boxes": 1, "max_len": 15}, 3: {"size": 7, "boxes": 2, "max_len": 15},
                  4: {"size": 7, "boxes": 3, "max_len": 25},
                  5: {"size": (5, 6, 7), "boxes": 1, "max_len": 15, "min_len": 2, "weights": (1, 4.5, 4.5)},
                  # stage 6: 1 box, 7x7 board only (5x5 floor), the size the stage 2 model is weakest on
                  6: {"size": 7, "boxes": 1, "max_len": 15}}


# --- helpers ---------------------------------------------------------------------------------
def _fmt(values: list[float], digits: int = 1) -> str:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}"


def _arrows(actions: str) -> str:
    return " → ".join(ACTION_NAMES_TH[ACTION_CHARS.index(c)] for c in actions) or "(ไม่ต้องเดิน)"


# --- commands --------------------------------------------------------------------------------
def cmd_make_levels(args) -> None:
    from sokoban_rl.levelgen import build_stage

    if args.stage == 7:  # designed levels (see designer.py), mixed with a few random-walk ones
        return _make_designed_stage(args)
    d = STAGE_DEFAULTS[args.stage]
    size, boxes = args.size or d["size"], args.boxes or d["boxes"]
    exclude = None
    if args.stage >= 3:  # keep the game's own levels out of the AI's training data
        from sokoban_rl.campaign import DIFFICULTIES, difficulty_levels
        from sokoban_rl.levelgen import canonical_key

        sizes = (size,) if isinstance(size, int) else size
        exclude = {canonical_key(lv) for name, spec in DIFFICULTIES.items()
                   if any(pt["size"] in sizes and pt["boxes"] == boxes for pt in spec["parts"])
                   for lv in difficulty_levels(name, args.data_dir)}
    t = time.time()
    build_stage(args.data_dir, size, boxes, args.stage, args.min_len or d.get("min_len", 3),
                args.max_len or d["max_len"], exclude=exclude, weights=d.get("weights"))
    print(f"เสร็จใน {time.time() - t:.1f} วินาที → {args.data_dir}/s{args.stage}_*.json")


def _make_designed_stage(args) -> None:
    from sokoban_rl.campaign import difficulty_levels, visible_difficulties
    from sokoban_rl.designer import _shape_key, build_designed_stage
    from sokoban_rl.levelgen import canonical_key

    game = [lv for name in visible_difficulties() for lv in difficulty_levels(name, args.data_dir)]
    exclude = {canonical_key(lv) for lv in game} | {_shape_key(lv) for lv in game}
    t = time.time()
    build_designed_stage(args.data_dir, 7, exclude=exclude)
    print(f"เสร็จใน {time.time() - t:.1f} วินาที → {args.data_dir}/s7_*.json")


def cmd_q_learn(args) -> None:
    from sokoban_rl.evaluation import run_episode
    from sokoban_rl.policies import QTablePolicy
    from sokoban_rl.qlearning import train_q
    from sokoban_rl.solver import solve

    level = load_level_file(args.level)
    best = solve(level)
    if best is None:
        sys.exit("ด่านนี้แก้ไม่ได้")
    print("\n".join(level_to_lines(level)) + f"\nBFS: {best} ({len(best)} ก้าว)\n")
    ok = True
    for seed in args.seeds:
        t = time.time()
        q = train_q(level, episodes=args.episodes, seed=seed)
        res = run_episode(level, QTablePolicy(q))
        good = res.solved and res.steps == len(best)
        ok &= good
        print(f"seed {seed}: {res.actions or '-'}  ชนะ={res.solved}  {res.steps} ก้าว  "
              f"{'เท่ากับ BFS' if good else 'ไม่เท่า BFS'}  ({time.time() - t:.2f} วินาที, "
              f"{len(q)} สถานะ)")
    print("ผ่านเกณฑ์ขั้นที่ 1" if ok else "ยังไม่ผ่านเกณฑ์ขั้นที่ 1")
    sys.exit(0 if ok else 1)


def cmd_train(args) -> None:
    from sokoban_rl.dqn import DQNConfig, train

    cfg = DQNConfig(stage=args.stage, seed=args.seed, steps=args.steps, double=args.double,
                    init=args.init, mask=args.mask, eps_start=args.eps_start, eps_decay_steps=args.eps_decay,
                    eval_every=args.eval_every, learn_start=args.learn_start,
                    lr=args.lr, threads=args.threads, data_dir=args.data_dir, out_dir=args.out_dir)
    out = train(cfg)
    print(f"บันทึกโมเดลที่ {out}")


def cmd_evaluate(args) -> None:
    from sokoban_rl.dqn import DQNPolicy, load_model
    from sokoban_rl.evaluation import evaluate
    from sokoban_rl.policies import BFSPolicy, RandomPolicy

    rows = []
    for split in args.splits:
        levels = load_levels(Path(args.data_dir) / f"s{args.stage}_{split}.json")
        for agent in args.agents:
            if agent == "bfs":
                policies = [BFSPolicy()]
            elif agent == "random":
                policies = [RandomPolicy(seed=s) for s in args.seeds]
            else:
                paths = [Path(args.models_dir) / f"s{args.stage}_seed{s}" / "best.pt" for s in args.seeds]
                missing = [str(p) for p in paths if not p.exists()]
                if missing:
                    sys.exit(f"ไม่พบโมเดล: {', '.join(missing)} (เทรนก่อนด้วย: sokoban-rl train)")
                policies = [DQNPolicy(load_model(p), mask=args.mask, avoid_repeats=args.avoid_repeats)
                            for p in paths]
            res = [evaluate(levels, p) for p in policies]
            rows.append((agent, split, len(levels), len(res),
                         _fmt([r["success_pct"] for r in res]),
                         _fmt([r["ratio_vs_bfs"] for r in res], 2),
                         _fmt([r["deadlock_pct"] for r in res])))
    header = ("agent", "split", "levels", "runs", "success %", "steps / BFS", "deadlock %")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    table = "\n".join(lines)
    print(table)
    out = Path(args.out_dir) / f"s{args.stage}_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n\n(mean ± population std over runs; one run = one seed)\n", encoding="utf-8")
    print(f"\nบันทึกที่ {out}")


def cmd_plot(args) -> None:
    from sokoban_rl.plotting import plot_training

    print(f"บันทึกกราฟที่ {plot_training(args.logs, args.out)}")


def _load_level(args):
    """Level from --file, or from --split (index --idx; a random one when --idx is not given)."""
    try:
        if args.file:
            return load_level_file(args.file)
        levels = load_levels(Path(args.data_dir) / f"s{args.stage}_{args.split}.json")
        if args.idx is None:
            args.idx = random.randrange(len(levels))
        return levels[args.idx]
    except (ValueError, OSError, IndexError) as e:  # LevelError is a ValueError
        sys.exit(f"อ่านด่านไม่ได้: {e}")


def _solve_or_exit(level) -> str:
    """BFS must be able to solve the level; returns the shortest solution and stores it."""
    from sokoban_rl.solver import solve

    best = solve(level)
    if best is None:
        print("\n".join(level_to_lines(level)))
        sys.exit("ด่านนี้แก้ไม่ได้ กรุณาส่งด่านใหม่")
    if not best:
        sys.exit("กล่องอยู่บนเป้าครบแล้ว ไม่ต้องเดิน กรุณาส่งด่านใหม่")
    level.bfs_solution = best
    return best


def _make_policy(args, level):
    if args.agent == "dqn":
        from sokoban_rl.dqn import DQNPolicy, load_model

        model = args.model or f"{args.models_dir}/s{args.stage}_seed{args.seed}/best.pt"
        if not Path(model).exists():
            sys.exit(f"ไม่พบโมเดล {model} (เทรนก่อนด้วย: sokoban-rl train --stage {args.stage})")
        return DQNPolicy(load_model(model), mask=args.mask, avoid_repeats=args.avoid_repeats)
    if args.agent == "qtable":
        from sokoban_rl.policies import QTablePolicy
        from sokoban_rl.qlearning import train_q

        return QTablePolicy(train_q(level, seed=args.seed))  # stage 1 learns this one level
    if args.agent == "bfs":
        from sokoban_rl.policies import BFSPolicy

        return BFSPolicy()
    from sokoban_rl.policies import RandomPolicy

    return RandomPolicy(args.seed)


def cmd_play(args) -> None:
    from sokoban_rl.evaluation import run_episode
    from sokoban_rl.render import animate_terminal, save_gif

    # 1-2) receive level, validate it
    level = _load_level(args)
    best = _solve_or_exit(level)
    if len(best) > MAX_STEPS:
        print(f"คำเตือน: ด่านนี้ต้องใช้ {len(best)} ก้าว เกินขีดจำกัด {MAX_STEPS} ก้าว")
    elif len(best) > 15:
        print("คำเตือน: ด่านนี้ยากกว่าช่วงที่ AI ถูกฝึกมา (คำตอบสั้นสุดเกิน 15 ก้าว)")

    # 3-4) convert + play
    policy = _make_policy(args, level)
    res = run_episode(level, policy)

    # 5) show result
    if not args.no_anim:
        animate_terminal(level, res.actions, delay=args.delay)
    else:
        print("\n".join(level_to_lines(level)))
    print(f"ลำดับการเดิน: {_arrows(res.actions)}")
    if res.solved:
        if res.steps == len(best):
            print(f"ชนะ · {res.steps} ก้าว · เท่ากับคำตอบที่สั้นที่สุด (BFS = {len(best)})")
        else:
            print(f"ชนะ · {res.steps} ก้าว · ยาวกว่าคำตอบที่สั้นที่สุด {res.steps - len(best)} ก้าว (BFS = {len(best)})")
    else:
        why = "กล่องติดมุม" if res.deadlock else f"เดินครบ {MAX_STEPS} ก้าว"
        print(f"แพ้ · {why} · BFS ทำได้ใน {len(best)} ก้าว")
    if args.gif:
        print(f"บันทึก GIF ที่ {save_gif(level, res.actions, args.gif)}")


def cmd_watch(args) -> None:
    from sokoban_rl.human import enable_ansi
    from sokoban_rl.monitor import find_active_runs, watch

    dirs = args.run_dirs or find_active_runs()
    if not dirs:
        sys.exit("no training in progress. Pass a run folder, e.g. models_double/s2_seed2")
    missing = [str(d) for d in dirs if not (Path(d) / "config.json").exists()]
    if missing:
        sys.exit(f"no config.json in: {', '.join(missing)} (is this a training run folder?)")
    if sys.stdout.isatty():
        enable_ansi()
    watch(dirs, interval=args.interval, once=args.once)


def _agent_factory(args):
    """level -> policy (or None when no AI is available). The DQN model is loaded once, on first use."""
    if args.agent == "none":
        return lambda level: None
    if args.agent == "qtable":
        from sokoban_rl.policies import QTablePolicy
        from sokoban_rl.qlearning import train_q

        return lambda level: QTablePolicy(train_q(level, seed=args.seed))
    # one model per box count: 1 box -> stage 2, 2 boxes -> stage 3, 3 boxes -> stage 4.
    # A single box on a 5x5 floor (7x7 board) is where the stage 2 model is weak (26-42% right), so
    # that case uses the mixed-size model trained with --mask (--big-model, 83%). It needs the masking
    # at play time, so it is only used while masking is on. If the file is missing, stage 2 is used.
    stage_for_boxes = {1: 2, 2: 3, 3: 4}
    cache: dict[tuple[int, bool], object] = {}
    warned: set[Path] = set()

    def model_path(n_boxes: int, big: bool) -> Path | None:
        if big:
            return Path(args.big_model)
        if n_boxes == 1 and args.model:
            return Path(args.model)
        stage = stage_for_boxes.get(n_boxes)
        return Path(f"{args.models_dir}/s{stage}_seed{args.seed}/best.pt") if stage else None

    def dqn(level):
        n = len(level.boxes)
        big = n == 1 and level.width >= 7 and args.mask and not args.model
        if big and not Path(args.big_model).exists():
            big = False  # not trained yet: the stage 2 model still plays it, just worse
        if (n, big) not in cache:
            path = model_path(n, big)
            if path is None or not path.exists():
                if path and path not in warned:
                    warned.add(path)
                    print(f"ไม่พบโมเดล {path} ด่านที่มีลัง {n} ใบจะเทียบกับ BFS อย่างเดียว")
                cache[n, big] = None
            else:
                from sokoban_rl.dqn import DQNPolicy, load_model

                cache[n, big] = DQNPolicy(load_model(path), mask=args.mask, avoid_repeats=args.avoid_repeats)
        return cache[n, big]

    return dqn


def cmd_gui(args) -> None:
    from sokoban_rl.campaign import DIFFICULTIES, Progress, difficulty_levels, visible_difficulties

    try:
        from sokoban_rl.gui import run_gui
    except ImportError as e:  # some Linux Pythons ship without tkinter
        sys.exit(f"เปิดหน้าต่างไม่ได้ ต้องมี tkinter: {e}")
    results = Path(args.progress_dir)

    def progress(name: str) -> Progress:
        path = results / f"progress_{name}.json"
        if args.reset and path.exists():
            path.unlink()
        return Progress(path)

    if args.file:
        level = _load_level(args)
        _solve_or_exit(level)
        modes = {"file": (level.name or "ด่านของคุณ", lambda: ([level], Progress(None)))}
        first = "file"
    else:
        shown = visible_difficulties()
        if args.difficulty not in shown:  # a hidden mode was asked for by name
            shown.append(args.difficulty)
        modes = {name: (DIFFICULTIES[name]["label"],
                        lambda n=name: (difficulty_levels(n, args.data_dir), progress(n))) for name in shown}
        first = args.difficulty
    start = args.level - 1 if args.level else None
    run_gui(modes, _agent_factory(args), args.agent, first, start)


def cmd_challenge(args) -> None:
    """You play first; then the AI plays the same level and the two are compared with BFS."""
    from sokoban_rl.evaluation import run_episode
    from sokoban_rl.human import build_report, enable_ansi, make_key_reader, play_interactive
    from sokoban_rl.render import animate_terminal

    level = _load_level(args)
    best = _solve_or_exit(level)
    policy = _make_policy(args, level)  # fail early if the model is missing, before you start playing
    if not args.file:
        level.name = f"{args.split} #{args.idx}"
    if sys.stdout.isatty():
        enable_ansi()

    human = play_interactive(level, make_key_reader())
    if human.outcome == "quit":
        print("ออกจากเกม")
        return

    agent = run_episode(level, policy)
    if not args.no_anim:
        print(f"\nตอนนี้ให้ AI ({args.agent}) เล่นด่านเดียวกัน...\n")
        animate_terminal(level, agent.actions, delay=args.delay)
    print("\n".join(build_report(level, human, args.agent, agent, best)))
    if not args.file:
        print(f"\nเล่นด่านนี้ซ้ำ: sokoban-rl challenge --split {args.split} --idx {args.idx}")


# --- parser ----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sokoban-rl", description="Sokoban RL: Q-Learning and DQN")
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-levels", help="generate train/val/test levels for a stage")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--size", type=int)
    s.add_argument("--boxes", type=int)
    s.add_argument("--min-len", type=int, help="default: 3 (2 for stage 5)")
    s.add_argument("--max-len", type=int, help="default: 15, or 25 for stage 4")
    s.set_defaults(func=cmd_make_levels)

    s = sub.add_parser("q-learn", help="stage 1: tabular Q-Learning on one level")
    s.add_argument("--level", default="data/example.txt")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--episodes", type=int, default=3000)
    s.set_defaults(func=cmd_q_learn)

    s = sub.add_parser("train", help="stage 2/3: train DQN")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=300_000)
    s.add_argument("--double", action="store_true", help="use Double DQN targets")
    s.add_argument("--mask", action="store_true",
                   help="never choose or bootstrap from a move that changes nothing (wall bumps)")
    s.add_argument("--init", help="warm-start weights, e.g. models/s2_seed0/best.pt")
    s.add_argument("--eps-start", type=float, default=1.0)
    s.add_argument("--eps-decay", type=int, default=100_000)
    s.add_argument("--eval-every", type=int, default=10_000)
    s.add_argument("--learn-start", type=int, default=2_000)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--threads", type=int)
    s.add_argument("--out-dir")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="compare random / bfs / dqn on level splits")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--splits", nargs="+", default=["val", "test"], choices=("train", "val", "test"))
    s.add_argument("--agents", nargs="+", default=["random", "bfs", "dqn"],
                   choices=("random", "bfs", "dqn"))
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--models-dir", default="models")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--mask", action="store_true", help="the AI never picks a move that changes nothing")
    s.add_argument("--avoid-repeats", action="store_true",
                   help="the AI prefers moves that lead to a position not seen yet in this game")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("plot", help="plot training curves from log.csv files")
    s.add_argument("logs", nargs="+")
    s.add_argument("--out", default="results/training.png")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("play", help="let the AI solve one level (animation + optional GIF)")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--file", help="level drawn with # @ $ . in a text file")
    src.add_argument("--split", choices=("train", "val", "test"), default="test")
    s.add_argument("--idx", type=int, default=0, help="level index within --split")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--agent", choices=("dqn", "qtable", "bfs", "random"), default="dqn")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model")
    s.add_argument("--models-dir", default="models")
    s.add_argument("--delay", type=float, default=0.3)
    s.add_argument("--no-anim", action="store_true")
    s.add_argument("--gif")
    s.add_argument("--mask", action="store_true", help="the AI never picks a move that changes nothing")
    s.add_argument("--avoid-repeats", action="store_true",
                   help="the AI prefers moves that lead to a position not seen yet in this game")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("watch", help="live progress bar for a running training (reads its log)")
    s.add_argument("run_dirs", nargs="*", help="run folders, e.g. models_double/s2_seed2 (default: all unfinished)")
    s.add_argument("--interval", type=float, default=1.0, help="seconds between redraws")
    s.add_argument("--once", action="store_true", help="print the status once and exit")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("gui", help="graphical game: difficulty modes, score, timer and saved progress")
    s.add_argument("--difficulty", choices=("easy", "medium", "hard", "two_boxes", "expert"), default="easy",
                   help="mode to start in: easy, medium or hard (keys 1-3 in the window); "
                        "two_boxes and expert are hidden extras")
    s.add_argument("--file", help="play one level drawn in a text file")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--level", type=int, help="start at this level number (1 = easiest); default: first unsolved")
    s.add_argument("--agent", choices=("dqn", "qtable", "none"), default="dqn")
    s.add_argument("--seed", type=int, default=0, help="which trained model to compare with")
    s.add_argument("--model")
    s.add_argument("--models-dir", default="models")
    s.add_argument("--progress-dir", default="results", help="where your scores are saved")
    s.add_argument("--reset", action="store_true", help="forget all saved scores")
    s.add_argument("--big-model", default="models_mask/s5_seed0/best.pt",
                   help="model for one box on a 5x5 floor (trained with --mask, so used only with masking)")
    s.add_argument("--allow-repeats", dest="avoid_repeats", action="store_false",
                   help="let the AI walk back to positions it has already been in (it loops more)")
    s.add_argument("--no-mask", dest="mask", action="store_false",
                   help="let the AI waste moves on walls (it is stronger without them)")
    s.set_defaults(func=cmd_gui, idx=None, split="test")

    s = sub.add_parser("challenge", help="play a level yourself, then compare with the AI and BFS")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--file", help="level drawn with # @ $ . in a text file")
    src.add_argument("--split", choices=("train", "val", "test"), default="test")
    s.add_argument("--idx", type=int, help="level index within --split (random if omitted)")
    s.add_argument("--stage", type=int, choices=(2, 3, 4, 5, 6, 7), default=2)
    s.add_argument("--agent", choices=("dqn", "qtable"), default="dqn")
    s.add_argument("--seed", type=int, default=0, help="which trained model / Q-table seed")
    s.add_argument("--model")
    s.add_argument("--models-dir", default="models")
    s.add_argument("--delay", type=float, default=0.3)
    s.add_argument("--no-anim", action="store_true")
    s.add_argument("--mask", action="store_true", help="the AI never picks a move that changes nothing")
    s.add_argument("--avoid-repeats", action="store_true",
                   help="the AI prefers moves that lead to a position not seen yet in this game")
    s.set_defaults(func=cmd_challenge)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Thai text on Windows consoles
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
