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
from sokoban_rl.human import KEYS, Session
from sokoban_rl.render import frames
from sokoban_rl.solver import solve
from sokoban_rl.sprites import make_tiles

CELL = 64  # 16 x 16 art scaled 4x
STEP_MS = 220
COLORS = {
    "bg": "#0c0c14", "panel": "#181826", "tile": "#10101b", "line": "#2f3150", "ink": "#f2f2f2",
    "muted": "#8d90ad", "faint": "#55576e", "accent": "#f4d35e", "good": "#7ed678", "bad": "#ff7b6b",
    "button": "#2b2d42", "button_hi": "#3d405b",
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
            self._say("กำลังเตรียมด่าน...", "muted")
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

        u = self.cell / CELL  # screen pixels per layout pixel (bigger on scaled displays)
        P, T = COLORS["panel"], COLORS["tile"]
        self.panel_w = round(340 * u)
        panel = tk.Frame(r, bg=P, padx=round(18 * u), pady=round(16 * u), highlightthickness=2,
                         highlightbackground=COLORS["line"])
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 16), pady=16)
        names = ("title", "par", "moves", "target", "time", "record", "score", "stars", "msg",
                 "you", "you_x", "ai", "ai_x", "short", "you_name", "ai_name", "short_name")
        self.vars = {k: tk.StringVar() for k in names}

        def text(parent, key=None, size=10, bold=False, color="ink", font=FONT, bg=P, value="", **kw):
            return tk.Label(parent, textvariable=self.vars[key] if key else None, text=value, bg=bg,
                            fg=COLORS[color], font=(font, size, "bold" if bold else "normal"), **kw)

        def button(parent, label, cmd, primary=False, size=10):
            return tk.Button(parent, text=label, font=(FONT, size, "bold"), takefocus=0, relief="flat", bd=0,
                             bg=COLORS["accent"] if primary else COLORS["button"],
                             fg=COLORS["bg"] if primary else COLORS["ink"],
                             activebackground=COLORS["button_hi"], activeforeground=COLORS["accent"],
                             disabledforeground=COLORS["faint"], pady=round(6 * u), cursor="hand2",
                             command=lambda: self.command(cmd))

        # 1. mode tabs
        self.mode_buttons: dict[str, tk.Button] = {}
        if len(self.modes) > 1:
            bar = tk.Frame(panel, bg=P)
            bar.pack(fill="x", pady=(0, round(14 * u)))
            for n, (key, (label, _)) in enumerate(self.modes.items(), start=1):
                b = button(bar, f"{n}  {label}", f"mode:{key}", size=9)
                b.grid(row=0, column=n - 1, sticky="ew", padx=1)
                bar.columnconfigure(n - 1, weight=1, uniform="mode")
                self.mode_buttons[key] = b

        # 2. stage title with prev / next, the mode line, and a strip with one block per stage
        self.buttons: dict[str, tk.Button] = {}
        head = tk.Frame(panel, bg=P)
        head.pack(fill="x")
        text(head, "title", 18, bold=True, color="accent", font=MONO).pack(side="left")
        for label, cmd in (("▶", "next"), ("◀", "prev")):
            b = button(head, label, cmd, size=9)
            b.configure(width=3, pady=round(2 * u))
            b.pack(side="right", padx=(round(4 * u), 0))
            self.buttons[cmd] = b
        text(panel, "par", 9, color="muted", anchor="w").pack(fill="x")
        self.strip = tk.Canvas(panel, width=self.panel_w, height=round(10 * u), bg=P, highlightthickness=0)
        self.strip.pack(pady=(round(8 * u), round(14 * u)))

        # 3. three number tiles
        stats = tk.Frame(panel, bg=P)
        stats.pack(fill="x")
        for i, (key, caption) in enumerate((("moves", "ก้าว"), ("target", "สั้นสุด"), ("time", "เวลา"))):
            tile = tk.Frame(stats, bg=T, padx=round(10 * u), pady=round(6 * u))
            tile.pack(side="left", expand=True, fill="x", padx=(0 if i == 0 else round(6 * u), 0))
            text(tile, value=caption, size=9, color="muted", bg=T).pack(anchor="w")
            text(tile, key, 20, bold=True, font=MONO, bg=T).pack(anchor="w")
        text(panel, "record", 9, color="muted", anchor="w").pack(fill="x", pady=(round(6 * u), 0))

        # 4. message / result card: a headline, then you vs AI vs shortest once the level is over
        card = tk.Frame(panel, bg=T, padx=round(12 * u), pady=round(10 * u))
        card.pack(fill="x", pady=(round(12 * u), 0))
        self.msg = text(card, "msg", 11, bold=True, bg=T, anchor="nw", justify="left", height=2,
                        wraplength=self.panel_w - round(24 * u))
        self.msg.pack(fill="x")
        rows = tk.Frame(card, bg=T)
        rows.pack(fill="x", pady=(round(4 * u), 0))
        rows.columnconfigure(1, weight=1)
        for i, key in enumerate(("you", "ai", "short")):
            name = text(rows, f"{key}_name", 10, color="muted", bg=T)
            name.grid(row=i, column=0, sticky="w", columnspan=3 if key == "you" else 1)
            if key == "you":
                self.you_name = name
            value = text(rows, key, 10, bold=True, bg=T, anchor="w")
            value.grid(row=i, column=1, sticky="w", padx=(round(10 * u), 0))
            if key == "ai":
                self.ai_label = value
            if key == "you":
                self.you_label = value
            if key != "short":
                extra = text(rows, f"{key}_x", 10, bold=True, color="accent", bg=T)
                extra.grid(row=i, column=2, sticky="e")
                if key == "you":
                    self.you_extra = extra
        self._row_names = {"you": "คุณ", "ai": f"AI ({self.agent_name})", "short": "สั้นสุด"}

        # 5. totals for this mode
        totals = tk.Frame(panel, bg=P)
        totals.pack(fill="x", pady=round(12 * u))
        text(totals, "score", 14, bold=True, color="accent", font=MONO).pack(side="left")
        text(totals, "stars", 10, color="muted").pack(side="right")

        # 6. actions: what you can do now (while playing / after the level)
        actions = tk.Frame(panel, bg=P)
        actions.pack(fill="x")
        self.play_row, self.done_row = tk.Frame(actions, bg=P), tk.Frame(actions, bg=P)
        groups = ((self.play_row, (("ย้อน  Z", "undo", False), ("เริ่มใหม่  R", "restart", False),
                                   ("ยอมแพ้  G", "giveup", False))),
                  (self.done_row, (("ดู AI  V", "watch_ai", False), ("ดูคำตอบ  B", "watch_best", False),
                                   ("ด่านต่อไป ▶", "next", True))))
        for row, items in groups:
            for label, cmd, primary in items:
                b = button(row, label, cmd, primary)
                b.pack(side="left", expand=True, fill="x", padx=2)
                self.buttons["next_big" if cmd == "next" else cmd] = b
        self.play_row.pack(fill="x")
        text(panel, value="เดิน W A S D / ลูกศร  ·  เปลี่ยนโหมด 1-3", size=8, color="faint").pack(
            anchor="w", pady=(round(10 * u), 0))
        r.bind("<Key>", self.on_key)
        # freeze the panel at its natural size so the window does not jump when texts change
        r.update_idletasks()
        panel.configure(width=panel.winfo_reqwidth(), height=max(self.board, panel.winfo_reqheight()))
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
        self._say("ดันลังไปไว้บนจุดสีเหลือง")
        for key in ("you", "you_x", "ai", "ai_x", "short", "you_name", "ai_name", "short_name"):
            self.vars[key].set("")
        self.vars["you_name"].set("เมื่อจบด่าน จะเทียบผลของคุณกับ AI ตรงนี้")
        self.you_name.configure(fg=COLORS["muted"], font=(FONT, 10), padx=0)
        self.you_name.grid_configure(columnspan=3)
        self.you_label.grid_remove()  # empty cells would paint over the hint
        self.you_extra.grid_remove()
        self._refresh()

    def _say(self, message: str, color: str = "ink") -> None:
        self.vars["msg"].set(message)
        self.msg.configure(fg=COLORS[color])

    def command(self, cmd: str) -> None:
        if self.replay_job and cmd not in ("next", "prev", "stop", "restart") and not cmd.startswith("mode:"):
            return  # ignore play keys while a replay is running
        if cmd in ("U", "D", "L", "R"):
            self.move(cmd)
        elif cmd == "undo" and not self.finished:
            if not self.session.undo():
                self._say("ไม่มีก้าวให้ย้อน", "muted")
            else:
                self._say("ดันลังไปไว้บนจุดสีเหลือง")
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
            self._say("เดินไม่ได้ (ชนกำแพงหรือลัง)", "muted")
        elif self.session.won:
            self.finish("won")
        elif self.session.stuck:
            self._say("ลังสีแดงติดตายแล้ว ดันไปถึงเป้าไม่ได้อีก\nกด Z ย้อน หรือ G ยอมแพ้", "bad")
        else:
            self._say("ดันลังไปไว้บนจุดสีเหลือง")

    def finish(self, outcome: str) -> None:
        self.elapsed = time.monotonic() - self.started if self.started else 0.0
        self.finished = outcome
        best = len(self.best)
        moves = len(self.session.actions)
        v = self.vars
        for key, name in self._row_names.items():
            v[f"{key}_name"].set(name)
        if outcome == "won":
            old = self.progress.best.get(self.keys[self.idx])
            result, improved = self.progress.record(self.keys[self.idx], moves, self.elapsed, best)
            self._say("ชนะ! สถิติใหม่ของด่านนี้" if old and improved else "ชนะ!", "good")
            v["you"].set(f"{moves} ก้าว · {fmt_clock(self.elapsed)}")
            v["you_x"].set(f"{'★' * result['stars']}{'☆' * (3 - result['stars'])}  +{result['score']}")
        else:
            self._say("ยอมแพ้ · ด่านนี้ได้ 0 คะแนน", "bad")
            v["you"].set(f"เดินไป {moves} ก้าว")
        self.you_name.configure(fg=COLORS["muted"], font=(FONT, 10))
        self.you_name.grid_configure(columnspan=1)
        self.you_label.grid()
        self.you_extra.grid()
        v["ai"].set("กำลังเล่น...")
        v["short"].set(f"{best} ก้าว")
        self.root.update_idletasks()
        line, won = self._ai_line()
        v["ai"].set(line)
        v["ai_x"].set({True: "ชนะ", False: "แพ้"}.get(won, ""))
        self.ai_label.configure(fg=COLORS["bad"] if won is False else COLORS["ink"])

    def ai_actions(self) -> str:
        if self.ai_result is None:
            policy = self.agent(self.level)
            if policy is None:
                return ""
            self.ai_result = run_episode(self.level, policy)
        return self.ai_result.actions

    def _ai_line(self) -> tuple[str, bool | None]:
        """(short text, whether the AI won; None when there is no AI to compare with)."""
        try:
            actions = self.ai_actions()
        except Exception as e:  # the comparison is a bonus; a broken model must not end the game
            return f"เล่นไม่ได้: {e}", None
        if self.ai_result is None or not actions:
            return "ไม่มีโมเดลให้เทียบ", None
        r = self.ai_result
        if r.solved:
            return f"{r.steps} ก้าว", True
        if r.deadlock:
            return "ดันลังติดมุม", False
        if len(self.level.boxes) > 1:
            return "เดินหมด (ฝึกมากับลังใบเดียว)", False
        return f"เดินวนจนหมด {r.steps} ก้าว", False

    # --- replay ------------------------------------------------------------------------------
    def watch(self, actions: str, who: str) -> None:
        if not actions:
            self._say(f"ไม่มีการเดินของ {who} ให้ดู", "muted")
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
        v = self.vars
        v["title"].set(f"STAGE {self.idx + 1:02d}/{n:02d}")
        v["par"].set(f"{self.modes[self.mode][0]}  ·  ลัง {len(self.level.boxes)} ใบ")
        v["moves"].set(f"{len(self.session.actions):03d}")
        v["target"].set(f"{len(self.best):03d}")
        self._update_time()
        rec = self.progress.best.get(key)
        v["record"].set(f"สถิติด่านนี้  {'★' * rec['stars']}  {rec['moves']} ก้าว · {fmt_clock(rec['seconds'])}"
                        if rec else "ยังไม่เคยผ่านด่านนี้")
        score, stars, solved = self.progress.totals(self.keys)
        v["score"].set(f"SCORE {score:,}")
        v["stars"].set(f"★ {stars}/{3 * n}    ผ่าน {solved}/{n}")
        self._draw_strip()
        done = self.finished is not None
        (self.play_row if done else self.done_row).pack_forget()
        (self.done_row if done else self.play_row).pack(fill="x")
        self.buttons["prev"].configure(state="normal" if self.idx > 0 else "disabled")
        last = self.idx + 1 >= n
        for name in ("next", "next_big"):
            self.buttons[name].configure(state="disabled" if last else "normal")
        if not self.replay_job:
            banner = {"won": "STAGE CLEAR!", "gave_up": "GIVE UP"}.get(self.finished or "", "")
            self.draw(self.session.player, self.session.boxes, self.session.dead_boxes, self.facing, banner)

    def _draw_strip(self) -> None:
        """One block per stage: green = 3 stars, yellow = solved, grey = not yet; this stage outlined."""
        c, n = self.strip, len(self.levels)
        c.delete("all")
        gap = max(2, round(3 * self.cell / CELL))
        w = (self.panel_w - gap * (n - 1)) / n
        h = int(c["height"])
        for i, key in enumerate(self.keys):
            rec = self.progress.best.get(key)
            fill = COLORS["button"] if not rec else COLORS["good"] if rec["stars"] == 3 else COLORS["accent"]
            x = i * (w + gap)
            here = i == self.idx
            c.create_rectangle(x + 1, 1, x + w - 1, h - 1, fill=fill, width=2 if here else 0,
                               outline=COLORS["ink"] if here else "")

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
