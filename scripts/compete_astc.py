#!/usr/bin/env python3
"""ASTC LDR benchmark: mipple vs Arm's astcenc and Intel ISPC Texture Compressor at 4x4, 6x6 and 8x8.

Every encoder writes a `.astc` file; `mipple score --astc-linear` grades them all the same way: RGB PSNR,
luma SSIM, and Zstd-19 and Deflate-9 size. ISPC encodes for linear decoding and astcenc runs with `-cl`, so
mipple runs with `--linear` too (an sRGB encode would be graded against a different decoder).

Encoders (each skipped if not found):
  --mipple     target/release/mipple (efforts from --efforts)
  --harness    tools/competitors bcn_harness built with ISPC (`ispc_astc:<W>x<H>`, fast profile)
  --astcenc    Arm's prebuilt astcenc release (third_party/dl/astcenc, the AVX2 build; presets from
               --astcenc-presets, `-j` set to --threads)

Time: mipple's own base-encode time, the harness's block loop and astcenc's "Coding time" (which excludes
loading and writing), best of --runs per image, summed over
the images. Both run on one thread unless --threads says otherwise. The encoders take turns per image,
so drift in clock speed lands on both. On a hybrid CPU, pin to one performance core with --cpu (on a
Core Ultra 7 155H, logical CPUs 0, 1 and 10-19 are P-cores; 2-9 are E-cores, which run ISPC about 1.8x
slower and mipple differently, so an unpinned run compares whatever cores each encode lands on).

Usage: python scripts/compete_astc.py test-images/kodak --runs 3 --efforts 0,2 --cpu 16
"""

import argparse
import ctypes
import os
import re
import subprocess
import sys
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
    ap.add_argument("--astcenc", type=Path,
                    default=ROOT / "third_party" / "dl" / "astcenc" / "x64" / "bin" / "astcenc-avx2.exe")
    ap.add_argument("--astcenc-presets", default="fast,medium,thorough")
    ap.add_argument("--footprints", default="4x4,6x6,8x8")
    ap.add_argument("--efforts", default="0,2")
    ap.add_argument("--runs", type=int, default=3, help="best of N timings per image")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    ap.add_argument("--cpu", type=int, default=None, help="pin this script and the encoders to this logical CPU")
    a = ap.parse_args()
    if a.cpu is not None:
        # Child processes inherit the affinity.
        if sys.platform == "win32":
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            if not k.SetProcessAffinityMask(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.c_size_t(1 << a.cpu)):
                raise SystemExit(f"could not pin to CPU {a.cpu}")
        else:
            os.sched_setaffinity(0, {a.cpu})

    imgs = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".tga"))
    if a.limit:
        imgs = imgs[: a.limit]
    # Encodes go in work/ (gitignored); only the table is printed.
    work = a.out / "work"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))

    def score(src, enc):
        r = run([str(a.mipple), "score", str(src), str(enc), "--csv", "--astc-linear"], env).strip().split(",")
        return float(r[5]), float(r[6]), int(r[9]), int(r[10]), int(r[3]) * int(r[4])

    print("| footprint / encoder | PSNR dB | SSIM | Zstd-19 bpt | Deflate-9 bpt | time s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for fp in a.footprints.split(","):
        # (name, command for (source, output), seconds from its stdout)
        encoders = []
        if a.harness.exists():
            encoders.append((f"{fp} ISPC fast",
                             lambda s, d, fp=fp: [str(a.harness), f"ispc_astc:{fp}", str(s), str(d), str(a.threads)],
                             lambda out: float(re.search(r"secs=([0-9.]+)", out).group(1))))
        if a.astcenc.exists():
            for pr in a.astcenc_presets.split(","):
                encoders.append((f"{fp} astcenc {pr}",
                                 lambda s, d, fp=fp, pr=pr: [str(a.astcenc), "-cl", str(s), str(d), fp, f"-{pr}",
                                                             "-j", str(a.threads)],
                                 lambda out: float(re.search(r"Coding time:\s+([0-9.]+)", out).group(1))))
        if a.mipple.exists():
            for e in a.efforts.split(","):
                encoders.append((f"{fp} mipple effort {e}",
                                 lambda s, d, fp=fp, e=e: [str(a.mipple), "encode", str(s), "-o", str(d), "-f",
                                                           f"astc-{fp}", "-e", e, "--no-mips", "--linear"],
                                 lambda out: float(re.search(r"time: base ([0-9.]+)s", out).group(1))))
        totals = {name: [0.0] * 6 for name, _, _ in encoders}  # secs, psnr, ssim, zstd, deflate bytes, texels
        for img in imgs:
            for name, encode, secs_of in encoders:
                enc = work / f"{img.stem}_{name.replace(' ', '_')}.astc"
                tot = totals[name]
                tot[0] += min(secs_of(run(encode(img, enc), env)) for _ in range(a.runs))
                for i, v in enumerate(score(img, enc), 1):
                    tot[i] += v
        for name, _, _ in encoders:
            secs, psnr, ssim, zst, dfl, tex = totals[name]
            k = len(imgs)
            print(f"| {name} | {psnr / k:.3f} | {ssim / k:.5f} | {zst * 8 / tex:.3f} | {dfl * 8 / tex:.3f} | {secs:.2f} |",
                  flush=True)


if __name__ == "__main__":
    main()
