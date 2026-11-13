#!/usr/bin/env python3
"""ASTC LDR benchmark: mipple vs Intel ISPC Texture Compressor (fast profile) at 4x4, 6x6 and 8x8.

Both encoders write a `.astc` file; `mipple score --astc-linear` grades both the same way: RGB PSNR, luma
SSIM and Zstd-19 size. ISPC encodes for linear decoding, so mipple runs with `--linear` too (an sRGB
encode would be graded against a different decoder).

Encoders (each skipped if not found):
  --mipple     target/release/mipple (efforts from --efforts)
  --harness    tools/competitors bcn_harness built with ISPC (`ispc_astc:<W>x<H>`, fast profile)

Not on disk, so not graded: astcenc (building from GitHub is not available here).

Time: mipple's own base-encode time and the harness's block loop, best of --runs per image, summed over
the images. Both run on one thread unless --threads says otherwise.

Usage: python scripts/compete_astc.py test-images/kodak --runs 3 --efforts 0,2
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "bench-results" / "astc")
    ap.add_argument("--mipple", type=Path, default=ROOT / "target" / "release" / "mipple.exe")
    ap.add_argument("--harness", type=Path, default=ROOT / "third_party" / "build-harness" / "Release" / "bcn_harness.exe")
    ap.add_argument("--footprints", default="4x4,6x6,8x8")
    ap.add_argument("--efforts", default="0,2")
    ap.add_argument("--runs", type=int, default=3, help="best of N timings per image")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    a = ap.parse_args()

    imgs = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".tga"))
    if a.limit:
        imgs = imgs[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))

    def score(src, enc):
        r = run([str(a.mipple), "score", str(src), str(enc), "--csv", "--astc-linear"], env).strip().split(",")
        return float(r[5]), float(r[6]), int(r[9]), int(r[3]) * int(r[4])

    def grade(name, encode, secs_of):
        secs = psnr = ssim = zst = tex = 0.0
        for img in imgs:
            enc = a.out / f"{img.stem}_{name.replace(' ', '_')}.astc"
            secs += min(secs_of(run(encode(img, enc), env)) for _ in range(a.runs))
            p, q, z, n = score(img, enc)
            psnr += p
            ssim += q
            zst += z
            tex += n
        k = len(imgs)
        print(f"| {name} | {psnr / k:.3f} | {ssim / k:.5f} | {zst * 8 / tex:.3f} | {secs:.2f} |", flush=True)

    print("| footprint / encoder | PSNR dB | SSIM | Zstd-19 bpt | time s |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for fp in a.footprints.split(","):
        if a.harness.exists():
            grade(f"{fp} ISPC fast",
                  lambda s, d, fp=fp: [str(a.harness), f"ispc_astc:{fp}", str(s), str(d), str(a.threads)],
                  lambda out: float(re.search(r"secs=([0-9.]+)", out).group(1)))
        if a.mipple.exists():
            for e in a.efforts.split(","):
                grade(f"{fp} mipple effort {e}",
                      lambda s, d, fp=fp, e=e: [str(a.mipple), "encode", str(s), "-o", str(d), "-f", f"astc-{fp}",
                                                "-e", e, "--no-mips", "--linear"],
                      lambda out: float(re.search(r"time: base ([0-9.]+)s", out).group(1)))


if __name__ == "__main__":
    main()
