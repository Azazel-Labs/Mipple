"""Draws the mipple icon (docs/img/mipple-icon.svg): an isometric mip pyramid of 4x4, 2x2 and 1x1 texels on a
dark tile, with ripples. Upper levels are true 2x2 box-filter averages of the level below.

  python scripts/make_icon.py [OUT.svg]   # default: docs/img/mipple-icon.svg and web/www/icon.svg
"""
import math
import sys
from pathlib import Path

PAL = [(0x22, 0xE4, 0xF0), (0x4F, 0x9B, 0xFF), (0xA0, 0x5C, 0xFF), (0xFF, 0x3E, 0xA5)]
# Level 0: four colors, like a BC1 block's palette, running left to right on screen.
L0 = [[PAL[min(3, (x - y + 4) // 2)] for x in range(4)] for y in range(4)]


def down(lv):
    n = len(lv) // 2
    return [[tuple(sum(lv[2 * y + j][2 * x + i][c] for i in (0, 1) for j in (0, 1)) / 4 for c in range(3))
             for x in range(n)] for y in range(n)]


L1 = down(L0)
L2 = down(L1)

S = 13.5            # cube edge in px
C30, S30 = math.cos(math.pi / 6), 0.5
OX, OY = 64, 43   # screen position of world origin (back corner of level 0, top)


def P(x, y, z):
    return (OX + (x - y) * C30 * S, OY + (x + y) * S30 * S - z * S)


def hexc(c, k=1.0):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * k))) for v in c)


def poly(pts, fill):
    return '<polygon points="%s" fill="%s"/>' % (" ".join("%.2f,%.2f" % p for p in pts), fill)


cubes = []
for z, (lv, off) in enumerate([(L0, 0), (L1, 1), (L2, 1.5)]):
    for y, row in enumerate(lv):
        for x, col in enumerate(row):
            cubes.append((z, x + y, off + x, off + y, col))
cubes.sort(key=lambda c: (c[0], c[1]))

g = 0.07  # inset: dark seams between texels
out = []
for z, _, x, y, col in cubes:
    x0, x1, y0, y1, z0, z1 = x + g, x + 1 - g, y + g, y + 1 - g, z, z + 1 - 2 * g
    top = [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]
    right = [P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0), P(x1, y0, z0)]  # +x face, lit less
    left = [P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0)]   # +y face
    out.append(poly(left, hexc(col, 0.62)))
    out.append(poly(right, hexc(col, 0.42)))
    out.append(poly(top, hexc(col)))

# Ripples: isometric circles on the ground plane around the pyramid's base.
cx, cy = P(2, 2, 0)
rings = []
for i, r in enumerate([3.1, 3.55]):
    rx, ry = r * C30 * S * math.sqrt(2), r * S30 * S * math.sqrt(2)
    rings.append('<ellipse cx="%.2f" cy="%.2f" rx="%.2f" ry="%.2f" fill="none" stroke="url(#ring)" '
                 'stroke-width="%.1f" opacity="%.2f"/>' % (cx, cy, rx, ry, 1.6 - 0.4 * i, 0.55 - 0.25 * i))

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <title>mipple</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#171a3a"/>
      <stop offset="1" stop-color="#0b0c1c"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.5" cy="0.56" r="0.5">
      <stop offset="0" stop-color="#7a5cff" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#7a5cff" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="ring" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#22e4f0"/>
      <stop offset="1" stop-color="#ff3ea5"/>
    </linearGradient>
  </defs>
  <rect width="128" height="128" rx="28" fill="url(#bg)"/>
  <rect width="128" height="128" rx="28" fill="url(#glow)"/>
  {"".join(rings)}
  <g>{"".join(out)}</g>
</svg>
'''
root = Path(__file__).resolve().parents[1]
# The README's icon and the web demo's favicon.
for dest in [Path(sys.argv[1])] if len(sys.argv) > 1 else [root / "docs/img/mipple-icon.svg", root / "web/www/icon.svg"]:
    dest.write_text(svg, encoding="utf-8", newline="\n")
