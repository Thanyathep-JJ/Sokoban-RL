"""Graphical Sokoban window (tkinter, ships with Python) in a retro 8-bit console style: play level
after level with a score, a timer and saved progress, then compare each level with the AI and the
shortest (BFS) answer.

Run it with ``sokoban-rl gui``. Game rules come from ``human.Session``; scoring and progress from
``campaign``; pixel art from ``sprites``; this file only draws and handles input.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox

from PIL import ImageTk

from sokoban_rl.campaign import Progress, fmt_clock, level_key, order_by_difficulty
from sokoban_rl.evaluation import EpisodeResult, run_episode
from sokoban_rl.game import Level
from sokoban_rl.human import KEYS, Session, count_wasted
from sokoban_rl.render import frames
from sokoban_rl.solver import solve
from sokoban_rl.sprites import make_tiles

CELL = 64  # 16 x 16 art scaled 4x
STEP_MS = 220
COLORS = {
    "bg": "#0c0c14", "panel": "#181826", "ink": "#f2f2f2", "muted": "#9a9cb8", "accent": "#f4d35e",
    "good": "#7ed678", "bad": "#ff7b6b", "button": "#2b2d42", "button_hi": "#3d405b",
}
FONT = "Leelawadee UI"  # Windows font with Thai glyphs; Tk falls back to another font elsewhere
MONO = "Consolas"  # blocky digits for the score board

# keys handled by the window itself (latin key, then the same key on the Thai layout)
_EXTRA = {"n": "next", "ื": "next", "p": "prev", "ย": "prev", "v": "watch_ai", "อ": "watch_ai",
          "b": "watch_best", "ิ": "watch_best"}
# number keys 1-5 pick a mode (top row of the Thai layout: ๅ / - ภ ถ)
_MODE_KEYS = {"1": 0, "ๅ": 0, "2": 1, "/": 1, "3": 2, "-": 2, "4": 3, "ภ": 3, "5": 4, "ถ": 4}
Mode = tuple[str, Callable[[], tuple[list[Level], Progress]]]  # (label, loader)
_KEYSYMS = {"Up": "U", "Down": "D", "Left": "L", "Right": "R", "BackSpace": "undo",
            "Return": "next", "KP_Enter": "next", "Escape": "stop"}


class App:
    def __init__(self, root: tk.Tk, modes: dict[str, Mode], agent: Callable[[Level], object | None],
                 agent_name: str, mode: str | None = None, start: int | None = None, cell: int = CELL):
        """``modes`` maps a key to (button label, loader returning the levels and their progress)."""
        self.root = root
        self.cell, self.board = cell, 7 * cell  # the largest supported level is 7 x 7
        self.tiles = {k: ImageTk.PhotoImage(v, master=root) for k, v in make_tiles(cell).items()}
        self.modes = modes
        self._loaded: dict[str, tuple[list[Level], list[str], Progress]] = {}
        self.agent = agent
        self.agent_name = agent_name
        self.replay_job: str | None = None
        self._build()
        self.select_mode(mode or next(iter(modes)), start)
        self._tick()

    def select_mode(self, mode: str, start: int | None = None) -> None:
        """Switch level set (difficulty); continue at its first unsolved level."""
        if mode not in self._loaded:
            self.vars["msg"].set("กำลังเตรียมด่าน...")
            self.root.configure(cursor="watch")
            self.root.update()
            levels, progress = self.modes[mode][1]()
            levels = order_by_difficulty(levels)
            for lv in levels:
                if lv.bfs_solution is None:
                    lv.bfs_solution = solve(lv) or ""
            self._loaded[mode] = (levels, [level_key(lv, i) for i, lv in enumerate(levels)], progress)
            self.root.configure(cursor="")
        self.mode = mode
        self.levels, self.keys, self.progress = self._loaded[mode]
        for key, b in self.mode_buttons.items():
            on = key == mode
            b.configure(bg=COLORS["accent"] if on else COLORS["button"], fg=COLORS["bg"] if on else COLORS["ink"])
        first = self.progress.first_unsolved(self.keys) if start is None else start
        self.load(max(0, min(first, len(self.levels) - 1)))

    # --- layout ------------------------------------------------------------------------------
    def _build(self) -> None:
        r = self.root
        r.title("Sokoban-RL")
        r.configure(bg=COLORS["bg"])
        r.resizable(False, False)
        self.canvas = tk.Canvas(r, width=self.board, height=self.board, bg=COLORS["bg"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=16, pady=16)

        panel = tk.Frame(r, bg=COLORS["panel"], padx=18, pady=16, highlightthickness=3,
                         highlightbackground=COLORS["button_hi"])
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 16), pady=16)
        self.vars = {k: tk.StringVar() for k in ("title", "par", "moves", "time", "record", "total", "msg")}

        def label(key, size=11, bold=False, color="ink", font=FONT, wrap=300):
            w = tk.Label(panel, textvariable=self.vars[key], bg=COLORS["panel"], fg=COLORS[color],
                         font=(font, size, "bold" if bold else "normal"), anchor="w", justify="left",
                         wraplength=wrap)
            w.pack(fill="x", pady=2)
            return w

        self.mode_buttons: dict[str, tk.Button] = {}
        if len(self.modes) > 1:
            bar = tk.Frame(panel, bg=COLORS["panel"])
            bar.pack(fill="x", pady=(0, 10))
            for n, (key, (text, _)) in enumerate(self.modes.items(), start=1):
                b = tk.Button(bar, text=f"{n} {text}", font=(FONT, 9, "bold"), takefocus=0, relief="flat", bd=0,
                              bg=COLORS["button"], fg=COLORS["ink"], activebackground=COLORS["button_hi"],
                              pady=4, cursor="hand2", command=lambda k=key: self.command(f"mode:{k}"))
                b.pack(side="left", expand=True, fill="x", padx=1)
                self.mode_buttons[key] = b
        label("title", 18, bold=True, color="accent", font=MONO)
        label("par", 10, color="muted")
        stats = tk.Frame(panel, bg=COLORS["panel"])
        stats.pack(fill="x", pady=(10, 6))
        for key, caption in (("moves", "MOVES"), ("time", "TIME")):
            box = tk.Frame(stats, bg=COLORS["panel"])
            box.pack(side="left", padx=(0, 28))
            tk.Label(box, text=caption, bg=COLORS["panel"], fg=COLORS["muted"], font=(MONO, 9, "bold")).pack(anchor="w")
            tk.Label(box, textvariable=self.vars[key], bg=COLORS["panel"], fg=COLORS["ink"],
                     font=(MONO, 24, "bold")).pack(anchor="w")
        label("record", 10, color="muted")
        label("total", 11, bold=True, color="accent")
        self.msg = label("msg", 11)
        self.msg.configure(height=8, anchor="nw")

        rows = [
            [("ย้อน (Z)", "undo"), ("เริ่มใหม่ (R)", "restart"), ("ยอมแพ้ (G)", "giveup")],
            [("◀ ก่อนหน้า (P)", "prev"), ("ถัดไป (N) ▶", "next")],
            [("ดู AI เล่น (V)", "watch_ai"), ("ดูคำตอบสั้นสุด (B)", "watch_best")],
        ]
        self.buttons: dict[str, tk.Button] = {}
        for row in rows:
            f = tk.Frame(panel, bg=COLORS["panel"])
            f.pack(fill="x", pady=3)
            for text, cmd in row:
                b = tk.Button(f, text=text, font=(FONT, 10, "bold"), takefocus=0, relief="flat", bd=0,
                              bg=COLORS["button"], fg=COLORS["ink"], activebackground=COLORS["button_hi"],
                              activeforeground=COLORS["accent"], disabledforeground="#55576e",
                              pady=6, cursor="hand2", command=lambda c=cmd: self.command(c))
                b.pack(side="left", expand=True, fill="x", padx=2)
                self.buttons[cmd] = b
        tk.Label(panel, text="เดิน: W A S D หรือปุ่มลูกศร", bg=COLORS["panel"], fg=COLORS["muted"],
                 font=(FONT, 9)).pack(anchor="w", pady=(10, 0))
        r.bind("<Key>", self.on_key)
        # freeze the panel at its natural size so the window does not jump when texts change
        r.update_idletasks()
        panel.configure(width=max(360, panel.winfo_reqwidth()), height=max(self.board, panel.winfo_reqheight()))
        panel.grid_propagate(False)
        panel.pack_propagate(False)

    # --- level flow --------------------------------------------------------------------------
    @property
    def level(self) -> Level:
        return self.levels[self.idx]

    @property
    def best(self) -> str:
        return self.level.bfs_solution or ""

    def load(self, idx: int) -> None:
        self._stop_replay()
        self.idx = idx
        self.session = Session(self.level)
        self.facing = "D"
        self.started: float | None = None
        self.elapsed = 0.0
        self.finished: str | None = None  # "won" or "gave_up"
        self.ai_result: EpisodeResult | None = None
        self.vars["msg"].set("ดันลังทุกใบไปไว้บนจุดสีเหลือง")
        self._refresh()

    def command(self, cmd: str) -> None:
        if self.replay_job and cmd not in ("next", "prev", "stop", "restart") and not cmd.startswith("mode:"):
            return  # ignore play keys while a replay is running
        if cmd in ("U", "D", "L", "R"):
            self.move(cmd)
        elif cmd == "undo" and not self.finished:
            if not self.session.undo():
                self.vars["msg"].set("ไม่มีก้าวให้ย้อน")
        elif cmd == "restart":
            self.load(self.idx)
        elif cmd == "giveup" and not self.finished:
            if messagebox.askyesno("ยอมแพ้?", "ยอมแพ้และดูคำตอบ?\nด่านนี้จะได้ 0 คะแนน (สถิติเดิมยังอยู่)",
                                   parent=self.root):
                self.finish("gave_up")
        elif cmd == "next" and self.idx + 1 < len(self.levels):
            self.load(self.idx + 1)
        elif cmd == "prev" and self.idx > 0:
            self.load(self.idx - 1)
        elif cmd == "watch_ai" and self.finished:
            self.watch(self.ai_actions(), "AI")
        elif cmd == "watch_best" and self.finished:
            self.watch(self.best, "BEST")
        elif cmd == "stop":
            self._stop_replay()
        elif cmd.startswith("mode:") and cmd[5:] != self.mode:
            self.select_mode(cmd[5:])
        self._refresh()

    def on_key(self, event) -> None:
        ch = (event.char or "").lower()
        cmd = _KEYSYMS.get(event.keysym) or _EXTRA.get(ch) or KEYS.get(ch, "")
        if cmd in ("quit", "yes", "no"):
            cmd = ""
        if not cmd and ch in _MODE_KEYS and _MODE_KEYS[ch] < len(self.modes):
            cmd = "mode:" + list(self.modes)[_MODE_KEYS[ch]]
        if cmd == "next" and event.keysym in ("Return", "KP_Enter") and not self.finished:
            cmd = ""  # Enter only moves on after the level is over
        if cmd:
            self.command(cmd)

    def move(self, action: str) -> None:
        if self.finished:
            return
        if self.started is None:
            self.started = time.monotonic()  # the clock starts with your first move
        self.facing = action
        result = self.session.press(action)
        if result == "blocked":
            self.vars["msg"].set("เดินไม่ได้ (ชนกำแพงหรือลัง)")
        elif self.session.won:
            self.finish("won")
        elif self.session.stuck:
            self.vars["msg"].set("ลังสีแดงติดตายแล้ว ดันไปถึงเป้าไม่ได้อีก\nกด Z เพื่อย้อน หรือ G เพื่อยอมแพ้")
        else:
            self.vars["msg"].set("")

    def finish(self, outcome: str) -> None:
        self.elapsed = time.monotonic() - self.started if self.started else 0.0
        self.finished = outcome
        best = len(self.best)
        moves = len(self.session.actions)
        lines = []
        if outcome == "won":
            old = self.progress.best.get(self.keys[self.idx])
            result, improved = self.progress.record(self.keys[self.idx], moves, self.elapsed, best)
            lines.append(f"ชนะ! {'★' * result['stars']}{'☆' * (3 - result['stars'])}  +{result['score']} คะแนน")
            lines.append(f"คุณ: {moves} ก้าว · {fmt_clock(self.elapsed)}")
            if old and improved:
                lines.append("สถิติใหม่ของด่านนี้!")
        else:
            lines.append("ยอมแพ้ · ด่านนี้ได้ 0 คะแนน")
            lines.append(f"คุณเดินไปได้ {moves} ก้าว")
        self.vars["msg"].set("\n".join(lines + ["กำลังให้ AI เล่นด่านเดียวกัน..."]))
        self.root.update_idletasks()
        lines.append(self._ai_line())
        lines.append(f"คำตอบสั้นสุด (BFS): {best} ก้าว  {self.best}")
        lines.append("")
        lines.append("V ดู AI เล่น · B ดูคำตอบ · N ด่านต่อไป · R เล่นใหม่")
        self.vars["msg"].set("\n".join(lines))

    def ai_actions(self) -> str:
        if self.ai_result is None:
            policy = self.agent(self.level)
            if policy is None:
                return ""
            self.ai_result = run_episode(self.level, policy)
        return self.ai_result.actions

    def _ai_line(self) -> str:
        try:
            actions = self.ai_actions()
        except Exception as e:  # the comparison is a bonus; a broken model must not end the game
            return f"AI เล่นไม่ได้: {e}"
        if self.ai_result is None or not actions:
            return f"AI ({self.agent_name}): ไม่มีโมเดลให้เทียบ"
        r = self.ai_result
        if r.solved:
            return f"AI ({self.agent_name}): ชนะ · {r.steps} ก้าว"
        wasted = count_wasted(self.level, actions)
        why = "ลังติดมุม" if r.deadlock else f"เดินครบ {r.steps} ก้าว ชนกำแพงเปล่า {wasted} ครั้ง"
        if len(self.level.boxes) > 1:
            why += " (AI ฝึกมากับลังใบเดียว)"
        return f"AI ({self.agent_name}): แพ้ · {why}"

    # --- replay ------------------------------------------------------------------------------
    def watch(self, actions: str, who: str) -> None:
        if not actions:
            self.vars["msg"].set(self.vars["msg"].get() + f"\n(ไม่มีการเดินของ {who} ให้ดู)")
            return
        self._stop_replay()
        steps = list(frames(self.level, actions))

        def show(i: int) -> None:
            player, boxes = steps[i]
            facing = actions[i - 1] if i else "D"
            if i + 1 < len(steps):
                self.draw(player, boxes, set(), facing, f"{who}  {i}/{len(actions)}")
                self.replay_job = self.root.after(STEP_MS, show, i + 1)
            else:
                self.replay_job = None
                self.draw(player, boxes, set(), facing, f"{who}  {len(actions)} MOVES")

        show(0)

    def _stop_replay(self) -> None:
        if self.replay_job:
            self.root.after_cancel(self.replay_job)
            self.replay_job = None

    # --- drawing -----------------------------------------------------------------------------
    def _refresh(self) -> None:
        n, key = len(self.levels), self.keys[self.idx]
        self.vars["title"].set(f"STAGE {self.idx + 1:03d}/{n:03d}")
        mode_label = self.modes[self.mode][0]
        self.vars["par"].set(f"{mode_label} · ลัง {len(self.level.boxes)} ใบ · คำตอบสั้นสุด {len(self.best)} ก้าว")
        self.vars["moves"].set(f"{len(self.session.actions):03d}")
        self._update_time()
        rec = self.progress.best.get(key)
        self.vars["record"].set(
            f"สถิติดีสุด: {'★' * rec['stars']} {rec['moves']} ก้าว · {fmt_clock(rec['seconds'])}"
            if rec else "ยังไม่เคยผ่านด่านนี้")
        score, stars, solved = self.progress.totals(self.keys)
        self.vars["total"].set(f"SCORE {score:,}  ★{stars}"
                               f"  ผ่าน {solved}/{n}")
        done = self.finished is not None
        for cmd in ("watch_ai", "watch_best"):
            self.buttons[cmd].configure(state="normal" if done else "disabled")
        for cmd in ("undo", "giveup"):
            self.buttons[cmd].configure(state="disabled" if done else "normal")
        self.buttons["prev"].configure(state="normal" if self.idx > 0 else "disabled")
        self.buttons["next"].configure(state="normal" if self.idx + 1 < n else "disabled")
        if not self.replay_job:
            banner = {"won": "STAGE CLEAR!", "gave_up": "GIVE UP"}.get(self.finished or "", "")
            self.draw(self.session.player, self.session.boxes, self.session.dead_boxes, self.facing, banner)

    def _update_time(self) -> None:
        if self.finished:
            seconds = self.elapsed
        else:
            seconds = time.monotonic() - self.started if self.started else 0.0
        self.vars["time"].set(fmt_clock(seconds))

    def _tick(self) -> None:
        self._update_time()
        self.root.after(250, self._tick)

    def _text(self, x: int, y: int, text: str, size: int, color: str) -> None:
        """Blocky text with a hard drop shadow, like an old console title screen."""
        off = max(2, size // 8)
        for dx, dy in ((off, off), (off, 0), (0, off)):
            self.canvas.create_text(x + dx, y + dy, text=text, fill="#000000", font=(MONO, size, "bold"))
        self.canvas.create_text(x, y, text=text, fill=color, font=(MONO, size, "bold"))

    def draw(self, player, boxes, dead: set, facing: str = "D", banner: str = "") -> None:
        c, lv, cell, t = self.canvas, self.level, self.cell, self.tiles
        c.delete("all")
        ox = (self.board - lv.width * cell) // 2
        oy = (self.board - lv.height * cell) // 2
        for r in range(lv.height):
            for col in range(lv.width):
                x, y = ox + col * cell, oy + r * cell
                if (r, col) in lv.walls:
                    c.create_image(x, y, image=t["wall"], anchor="nw")
                    continue
                shadow = (r - 1, col) in lv.walls
                c.create_image(x, y, image=t["floor_shadow" if shadow else "floor"], anchor="nw")
                if (r, col) in lv.targets:
                    c.create_image(x, y, image=t["target"], anchor="nw")
        for r, col in boxes:
            name = "crate_on" if (r, col) in lv.targets else "crate_dead" if (r, col) in dead else "crate"
            c.create_image(ox + col * cell, oy + r * cell, image=t[name], anchor="nw")
        pr, pc = player
        c.create_image(ox + pc * cell, oy + pr * cell, image=t[f"worker_{facing}"], anchor="nw")
        if banner:
            big = banner in ("STAGE CLEAR!", "GIVE UP")
            color = COLORS["bad"] if banner == "GIVE UP" else COLORS["accent"]
            y = self.board // 2 if big else self.board - cell // 3
            self._text(self.board // 2, y, banner, 30 if big else 14, color if big else COLORS["ink"])


def run_gui(modes: dict[str, Mode], agent: Callable[[Level], object | None], agent_name: str,
            mode: str | None = None, start: int | None = None) -> None:
    if sys.platform == "win32":
        try:  # draw sharp on scaled displays instead of letting Windows blur the window
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    scale = max(1.0, root.winfo_fpixels("1i") / 96)  # 1.25 on a 125 % display
    cell = 16 * max(4, round(4 * scale))  # a whole number of screen pixels per art pixel
    App(root, modes, agent, agent_name, mode, start, cell=cell)
    root.mainloop()
