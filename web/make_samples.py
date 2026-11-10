#!/usr/bin/env python3
"""Regenerate the demo's sample images (web/www/samples/) from Poly Haven's CC0 textures.

Reads the 1k PNGs that scripts/fetch_polyhaven.py downloads into test-images/polyhaven/, and
writes 512x512 PNGs (Lanczos) plus 192x192 JPEG thumbnails for the gallery. Needs Pillow.

    python web/make_samples.py [--src test-images/polyhaven]
"""
import argparse
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "www" / "samples"
# Varied on purpose: hard edges, saturated color, fine noise, near-flat areas, smooth gradients
# and a tangent-space normal map (for BC5).
SAMPLES = [
    "blue_painted_planks_diff",
    "castle_brick_01_diff",
    "pine_bark_diff",
    "white_rough_plaster_diff",
    "wood_table_worn_diff",
    "cobblestone_floor_13_nor",
]
SIZE = 512
THUMB = 192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, default=ROOT / "test-images" / "polyhaven")
    a = ap.parse_args()
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        im = Image.open(a.src / f"{name}.png").convert("RGB")
        full = im.resize((SIZE, SIZE), Image.LANCZOS)
        full.save(OUT / f"{name}.png", optimize=True)
        full.resize((THUMB, THUMB), Image.LANCZOS).save(OUT / "thumbs" / f"{name}.jpg", quality=82, optimize=True)
        print(name, (OUT / f"{name}.png").stat().st_size)


if __name__ == "__main__":
    main()
