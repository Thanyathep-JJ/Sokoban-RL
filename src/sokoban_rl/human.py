"""Play a level yourself, then compare with the AI and the BFS optimum (``sokoban-rl challenge``).

The game logic (``Session``) and the report are plain functions so they can be tested without a
terminal. Only ``make_key_reader`` touches the keyboard.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from sokoban_rl.evaluation import EpisodeResult
from sokoban_rl.game import (
    ACTION_CHARS,
    MAX_STEPS,
    Boxes,
    Level,
    Pos,
    is_deadlock,
    level_to_lines,
    move,
)

HELP = "เดิน: W A S D หรือปุ่มลูกศร · Z ย้อนกลับ · R เริ่มใหม่ · G ยอมแพ้ (ดูคำตอบ) · Q ออก"

# raw key -> command. Moves are "U" "D" "L" "R"; everything else is a word.
KEYS = {
    "w": "U", "s": "D", "a": "L", "d": "R",
    "z": "undo", "\x08": "undo", "\x7f": "undo",
    "r": "restart", "g": "giveup", "q": "quit", "\x03": "quit", "\x04": "quit",
    "y": "yes", "n": "no",
}
MOVES = frozenset(ACTION_CHARS)
_ARROWS_WINDOWS = {"H": "U", "P": "D", "K": "L", "M": "R"}
_ARROWS_ANSI = {"A": "U", "B": "D", "D": "L", "C": "R"}


# --- keyboard --------------------------------------------------------------------------------
def _read_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # arrow keys arrive as a prefix plus a code
        return _ARROWS_WINDOWS.get(msvcrt.getwch(), "")
    return KEYS.get(ch.lower(), "")


def _read_key_posix() -> str:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":
            if select.select([fd], [], [], 0.05)[0]:
                seq = os.read(fd, 2).decode(errors="ignore")
                if seq.startswith("["):
                    return _ARROWS_ANSI.get(seq[1:2], "")
            return ""
        return KEYS.get(ch.lower(), "")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _line_keys(stream) -> Iterator[str]:
    """Keys from a pipe or file: every character is one key; the end of input means quit."""
    for line in stream:
        for ch in line.strip():
            yield KEYS.get(ch.lower(), "")
    while True:
        yield "quit"


def make_key_reader(stream=None) -> Callable[[], str]:
    """One key per call, no Enter needed on a real terminal."""
    stream = sys.stdin if stream is None else stream
    if stream is sys.stdin and stream.isatty():
        return _read_key_windows if os.name == "nt" else _read_key_posix
    keys = _line_keys(stream)
    return lambda: next(keys)


def enable_ansi() -> None:
    """Old Windows consoles only understand escape codes after this no-op call."""
    if os.name == "nt":
        os.system("")


# --- game session ----------------------------------------------------------------------------
class Session:
    """A human game with undo. Bumping into a wall changes nothing and is not counted as a move."""

    def __init__(self, level: Level):
        self.level = level
        self.undos = 0
        self.restarts = 0
        self.bumps = 0
        self._begin()

    def _begin(self) -> None:
        self.player: Pos = self.level.player
        self.boxes: Boxes = self.level.boxes
        self._trail: list[tuple[Pos, Boxes, str]] = []

    @property
    def actions(self) -> str:
        return "".join(a for _, _, a in self._trail)

    @property
    def won(self) -> bool:
        return self.boxes == self.level.targets

    @property
    def stuck(self) -> bool:
        """A box sits in a corner: the level can no longer be solved without undoing."""
        return not self.won and is_deadlock(self.level, self.boxes)

    def press(self, action: str) -> str:
        """Try a move ("U", "D", "L" or "R"). Returns "moved", "pushed" or "blocked"."""
        player, boxes, push = move(self.level, self.player, self.boxes, ACTION_CHARS.index(action))
        if (player, boxes) == (self.player, self.boxes):
            self.bumps += 1
            return "blocked"
        self._trail.append((self.player, self.boxes, action))
        self.player, self.boxes = player, boxes
        return "pushed" if push else "moved"

    def undo(self) -> bool:
        if not self._trail:
            return False
        self.player, self.boxes, _ = self._trail.pop()
        self.undos += 1
        return True

    def restart(self) -> None:
        self.restarts += 1
        self._begin()


@dataclass
class HumanResult:
    outcome: str  # "won", "gave_up" or "quit"
    actions: str  # the line of moves on the board when the game ended
    undos: int
    restarts: int
    bumps: int


def _draw(out, level: Level, s: Session, note: str) -> None:
    if getattr(out, "isatty", lambda: False)():
        out.write("\033[2J\033[H")
    rows = [" ".join(line) for line in level_to_lines(level, s.player, s.boxes)]
    out.write(f"ด่าน {level.name or '-'} · เดินไปแล้ว {len(s.actions)} ก้าว\n\n")
    out.write("\n".join(rows) + "\n\n")
    out.write("@ คุณ   $ กล่อง   . เป้า   * กล่องอยู่บนเป้า   # กำแพง\n")
    out.write(HELP + "\n")
    if note:
        out.write(note + "\n")
    out.flush()


def play_interactive(level: Level, get_key: Callable[[], str], out=sys.stdout) -> HumanResult:
    """Run the game loop until the player wins, gives up (with confirmation) or quits."""
    s = Session(level)
    note = ""

    def result(outcome: str) -> HumanResult:
        return HumanResult(outcome, s.actions, s.undos, s.restarts, s.bumps)

    while True:
        _draw(out, level, s, note)
        cmd = get_key()
        note = ""
        if not cmd:  # a key with no meaning ('' would otherwise match `in ACTION_CHARS`)
            note = "ปุ่มนี้ใช้ไม่ได้"
        elif cmd in MOVES:
            if s.press(cmd) == "blocked":
                note = "เดินไม่ได้ (ชนกำแพงหรือกล่อง)"
            elif s.won:
                _draw(out, level, s, "")
                return result("won")
            elif s.stuck:
                note = "กล่องติดมุมแล้ว แก้ไม่ได้อีก กด Z เพื่อย้อน หรือ G เพื่อยอมแพ้"
        elif cmd == "undo":
            if not s.undo():
                note = "ไม่มีก้าวให้ย้อน"
        elif cmd == "restart":
            s.restart()
            note = "เริ่มใหม่แล้ว"
        elif cmd == "giveup":
            _draw(out, level, s, "ยอมแพ้และดูคำตอบ? กด Y เพื่อยืนยัน (ปุ่มอื่น = เล่นต่อ)")
            if get_key() == "yes":
                return result("gave_up")
        elif cmd == "quit":
            return result("quit")


# --- comparison ------------------------------------------------------------------------------
def count_wasted(level: Level, actions: str) -> int:
    """Actions that changed nothing, such as walking into a wall."""
    player, boxes = level.player, level.boxes
    wasted = 0
    for ch in actions:
        nxt_player, nxt_boxes, _ = move(level, player, boxes, ACTION_CHARS.index(ch))
        wasted += (nxt_player, nxt_boxes) == (player, boxes)
        player, boxes = nxt_player, nxt_boxes
    return wasted


def _short(actions: str, limit: int = 60) -> str:
    if not actions:
        return "-"
    return actions if len(actions) <= limit else actions[: limit - 1] + "…"


def build_report(level: Level, human: HumanResult, agent_name: str, agent: EpisodeResult,
                 best: str) -> list[str]:
    """Plain-text comparison of the player, the AI and the shortest (BFS) answer."""
    won = human.outcome == "won"
    extras = []
    if human.undos:
        extras.append(f"ย้อน {human.undos} ครั้ง")
    if human.restarts:
        extras.append(f"เริ่มใหม่ {human.restarts} ครั้ง")
    extra = f" ({', '.join(extras)})" if extras else ""
    if won:
        you = f"ชนะ · {len(human.actions)} ก้าว{extra}"
    else:
        you = f"ยอมแพ้ · เดินไปได้ {len(human.actions)} ก้าว{extra}"

    if agent.solved:
        ai = f"ชนะ · {agent.steps} ก้าว"
    else:
        why = "กล่องติดมุม" if agent.deadlock else f"เดินครบ {MAX_STEPS} ก้าวแล้วยังไม่จบ"
        ai = f"แพ้ · {why}"
        wasted = count_wasted(level, agent.actions)
        if wasted:
            ai += f" (เดินชนกำแพงเปล่า ๆ {wasted} จาก {agent.steps} ก้าว)"

    lines = [
        "─" * 12 + " เปรียบเทียบ " + "─" * 12,
        f"คุณ: {you}",
        f"    ลำดับ: {_short(human.actions)}",
        f"AI ({agent_name}): {ai}",
        f"    ลำดับ: {_short(agent.actions)}",
        f"คำตอบสั้นที่สุด (BFS): {len(best)} ก้าว",
        f"    ลำดับ: {_short(best)}",
        "",
    ]
    if won and agent.solved:
        diff = len(human.actions) - agent.steps
        if diff < 0:
            lines.append(f"คุณเดินสั้นกว่า AI {-diff} ก้าว")
        elif diff > 0:
            lines.append(f"AI เดินสั้นกว่าคุณ {diff} ก้าว")
        else:
            lines.append("คุณกับ AI ใช้จำนวนก้าวเท่ากัน")
    elif won:
        lines.append("คุณชนะด่านที่ AI แก้ไม่ได้")
    elif agent.solved:
        lines.append(f"AI แก้ด่านนี้ได้ใน {agent.steps} ก้าว")
    else:
        lines.append("AI ก็แก้ด่านนี้ไม่ได้ มีแต่ BFS ที่แก้ได้")
    if won and len(human.actions) == len(best):
        lines.append("คุณเดินได้สั้นที่สุดแล้ว (เท่ากับ BFS)")
    return lines
