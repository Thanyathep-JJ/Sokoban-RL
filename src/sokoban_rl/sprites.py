"""Original 16 x 16 pixel-art tiles in a retro 8-bit console style, drawn in code (no image files).

``make_tiles(cell)`` returns Pillow images scaled up with nearest-neighbour, so every art pixel
stays a crisp square. Used by the graphical game (``gui.py``).
"""

from __future__ import annotations

from PIL import Image, ImageDraw

ART = 16  # art resolution of one tile

PALETTE = {
    "k": (16, 16, 24),  # outline
    # walls
    "r": (176, 72, 56), "R": (214, 112, 82), "m": (78, 38, 40),
    # floor
    "f": (44, 46, 66), "F": (52, 55, 78), "g": (32, 33, 50), "s": (24, 25, 38),
    # target
    "t": (244, 211, 94), "T": (255, 240, 170),
    # crate (wood), crate on target (green), dead crate (red)
    "w": (229, 164, 88), "W": (250, 206, 130), "d": (170, 104, 50), "n": (90, 52, 30),
    # worker
    "G": (70, 170, 80), "H": (40, 110, 52), "S": (244, 200, 154), "B": (66, 98, 210),
    "b": (40, 62, 150), "E": (255, 255, 255), "M": (196, 84, 74), "h": (110, 70, 40),
}
CRATE_SWAPS = {
    "crate": {},
    "crate_on": {"w": (126, 214, 120), "W": (190, 245, 170), "d": (52, 150, 80), "n": (24, 80, 44)},
    "crate_dead": {"w": (226, 110, 100), "W": (250, 170, 160), "d": (160, 48, 48), "n": (90, 20, 24)},
}

# worker facing down; "." is transparent. Up and right are drawn too; left mirrors right.
WORKER = {
    "D": [
        "................",
        ".....kkkkkk.....",
        "....kGGGGGGk....",
        "...kGGGGGGGGk...",
        "...kHHHHHHHHk...",
        "...kSSSSSSSSk...",
        "...kSEkSSEkSk...",
        "...kSSSSSSSSk...",
        "....kSSMMSSk....",
        ".....kkkkkk.....",
        "...kBBEBBEBBk...",
        "..kSkBBBBBBkSk..",
        "..kSkBBBBBBkSk..",
        "...kkBBkkBBkk...",
        "....kbbk.kbbk...",
        "....kkkk.kkkk...",
    ],
    "U": [
        "................",
        ".....kkkkkk.....",
        "....kGGGGGGk....",
        "...kGGGGGGGGk...",
        "...kGGGGGGGGk...",
        "...khhhhhhhhk...",
        "...khhhhhhhhk...",
        "...khhhhhhhhk...",
        "....khhhhhhk....",
        ".....kkkkkk.....",
        "...kBBBBBBBBk...",
        "..kSkBBBBBBkSk..",
        "..kSkBBBBBBkSk..",
        "...kkBBkkBBkk...",
        "....kbbk.kbbk...",
        "....kkkk.kkkk...",
    ],
    "R": [
        "................",
        ".....kkkkkk.....",
        "....kGGGGGGk....",
        "...kGGGGGGGGk...",
        "...kHHHHHHHHHk..",
        "...khSSSSSSSSk..",
        "...khSSSSEkSSk..",
        "...khSSSSSSSSk..",
        "....khSSSSMMk...",
        ".....kkkkkkk....",
        "....kBBBBBBk....",
        "....kBBBkSSk....",
        "....kBBBkSSk....",
        "....kkBBBBkk....",
        ".....kbbkbbk....",
        ".....kkkkkkk....",
    ],
}


def _from_rows(rows: list[str], swaps: dict | None = None) -> Image.Image:
    img = Image.new("RGBA", (ART, ART), (0, 0, 0, 0))
    px = img.load()
    colors = {**PALETTE, **(swaps or {})}
    for y, row in enumerate(rows[:ART]):
        for x, ch in enumerate(row.ljust(ART, ".")[:ART]):
            if ch != ".":
                px[x, y] = colors[ch] + (255,)
    return img


def _wall() -> Image.Image:
    img = Image.new("RGBA", (ART, ART), PALETTE["r"] + (255,))
    d = ImageDraw.Draw(img)
    for row, y in enumerate(range(0, ART, 4)):  # bricks 4 px high, every other row shifted
        d.line([(0, y + 3), (ART - 1, y + 3)], fill=PALETTE["m"])
        d.line([(0, y), (ART - 1, y)], fill=PALETTE["R"])  # lit top edge of each brick row
        for x in ((0, 8) if row % 2 == 0 else (4, 12)):
            d.line([(x, y), (x, y + 2)], fill=PALETTE["m"])
            if x + 1 < ART:
                d.point((x + 1, y + 1), fill=PALETTE["R"])
    return img


def _floor(shadow: bool) -> Image.Image:
    img = Image.new("RGBA", (ART, ART), PALETTE["f"] + (255,))
    d = ImageDraw.Draw(img)
    d.line([(0, ART - 1), (ART - 1, ART - 1)], fill=PALETTE["g"])  # grout between tiles
    d.line([(ART - 1, 0), (ART - 1, ART - 1)], fill=PALETTE["g"])
    for x, y in ((3, 4), (10, 2), (6, 11), (12, 9), (2, 13)):  # a little texture
        d.point((x, y), fill=PALETTE["F"])
    if shadow:  # the wall above casts a shadow onto the floor
        d.rectangle([(0, 0), (ART - 1, 2)], fill=PALETTE["s"])
    return img


def _target() -> Image.Image:
    img = Image.new("RGBA", (ART, ART), (0, 0, 0, 0))
    px = img.load()
    c = ART // 2
    for y in range(ART):
        for x in range(ART):
            dist = abs(x - c + 0.5) + abs(y - c + 0.5)
            if 4 <= dist <= 5 and (x + y) % 2 == 0:  # dotted diamond
                px[x, y] = PALETTE["t"] + (255,)
            elif dist <= 1:
                px[x, y] = PALETTE["T"] + (255,)
    return img


def _crate(swaps: dict) -> Image.Image:
    c = {**PALETTE, **swaps}
    img = Image.new("RGBA", (ART, ART), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([(1, 1), (14, 14)], fill=c["d"], outline=c["k"])
    d.rectangle([(2, 2), (13, 13)], outline=c["w"])  # frame
    d.rectangle([(3, 3), (12, 12)], outline=c["w"])
    d.line([(2, 2), (13, 2)], fill=c["W"])  # light from above
    d.line([(2, 2), (2, 13)], fill=c["W"])
    d.line([(4, 4), (11, 11)], fill=c["w"])  # cross brace
    d.line([(4, 11), (11, 4)], fill=c["w"])
    d.line([(4, 5), (10, 11)], fill=c["n"])
    for x, y in ((2, 2), (13, 2), (2, 13), (13, 13)):  # nails
        d.point((x, y), fill=c["n"])
    d.line([(2, 14), (14, 14)], fill=c["n"])  # shade at the bottom and right
    d.line([(14, 2), (14, 14)], fill=c["n"])
    return img


def make_tiles(cell: int) -> dict[str, Image.Image]:
    """All tiles at ``cell`` x ``cell`` pixels (use a multiple of 16 for perfectly even pixels)."""
    art = {
        "wall": _wall(), "floor": _floor(False), "floor_shadow": _floor(True), "target": _target(),
        **{name: _crate(swaps) for name, swaps in CRATE_SWAPS.items()},
        "worker_D": _from_rows(WORKER["D"]), "worker_U": _from_rows(WORKER["U"]),
        "worker_R": _from_rows(WORKER["R"]),
    }
    art["worker_L"] = art["worker_R"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return {k: v.resize((cell, cell), Image.Resampling.NEAREST) for k, v in art.items()}
