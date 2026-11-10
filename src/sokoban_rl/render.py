"""Terminal animation and GIF export (Pillow only, no image assets)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from sokoban_rl.game import ACTION_CHARS, Level, level_to_lines, move

COLORS = {
    "floor": (245, 241, 230), "wall": (60, 64, 72), "target": (214, 69, 65),
    "box": (176, 122, 66), "box_ok": (76, 175, 80), "player": (52, 120, 220),
}


def frames(level: Level, actions: str):
    """Yield ``(player, boxes)`` for the start and after every action."""
    player, boxes = level.player, level.boxes
    yield player, boxes
    for ch in actions:
        player, boxes, _ = move(level, player, boxes, ACTION_CHARS.index(ch))
        yield player, boxes


def animate_terminal(level: Level, actions: str, delay: float = 0.3, out=sys.stdout) -> None:
    """Redraw the board in place using ANSI escape codes."""
    for i, (player, boxes) in enumerate(frames(level, actions)):
        out.write("\033[H\033[J" if i else "")
        out.write(f"ก้าวที่ {i}/{len(actions)}\n" + "\n".join(level_to_lines(level, player, boxes)) + "\n")
        out.flush()
        time.sleep(delay)


def draw_frame(level: Level, player, boxes, cell: int = 48) -> Image.Image:
    img = Image.new("RGB", (level.width * cell, level.height * cell), COLORS["floor"])
    d = ImageDraw.Draw(img)
    pad = cell // 8
    for r in range(level.height):
        for c in range(level.width):
            x0, y0 = c * cell, r * cell
            box = (x0, y0, x0 + cell - 1, y0 + cell - 1)
            if (r, c) in level.walls:
                d.rectangle(box, fill=COLORS["wall"])
            elif (r, c) in level.targets:
                m = cell // 3
                d.ellipse((x0 + m, y0 + m, x0 + cell - m, y0 + cell - m), fill=COLORS["target"])
    for r, c in boxes:
        x0, y0 = c * cell, r * cell
        color = COLORS["box_ok"] if (r, c) in level.targets else COLORS["box"]
        d.rectangle((x0 + pad, y0 + pad, x0 + cell - pad, y0 + cell - pad), fill=color)
    pr, pc = player
    d.ellipse((pc * cell + pad, pr * cell + pad, (pc + 1) * cell - pad, (pr + 1) * cell - pad),
              fill=COLORS["player"])
    return img


def save_gif(level: Level, actions: str, path: str | Path, cell: int = 48,
             step_ms: int = 400, hold_ms: int = 1500) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [draw_frame(level, p, b, cell) for p, b in frames(level, actions)]
    durations = [step_ms] * len(imgs)
    durations[-1] = hold_ms
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)
    return path
