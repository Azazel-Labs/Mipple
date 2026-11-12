#!/usr/bin/env python3
"""ETC2 benchmark: mipple vs AMD Compressonator (ETC2 RGB and ETC2 RGBA8) on the same images.

Every encoder writes a KTX2 (DDS has no ETC formats); `mipple score` grades all of them with the same
metrics as the BC1/BC3 benchmarks: RGB PSNR (RGBA PSNR for ETC2 RGBA), luma SSIM, and Zstd-19 size.

Encoders (each skipped if not found):
  --mipple     target/release/mipple (efforts from --efforts)
  --cmp        compressonatorcli (-fd ETC2_RGB / ETC2_RGBA; -Quality has no effect on its ETC2 path)

Not on disk, so not graded: etcpak and etc2comp (building from GitHub is not available here).

Time: mipple's own base-encode time, and process wall time for Compressonator (which includes loading
and writing; both are small next to its encode). --threads N limits every encoder to N threads.

Usage: python scripts/compete_etc.py test-images/kodak --format etc2-rgb --threads 1 --out bench-results/etc2
"""

import argparse
import csv
import io
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, env=None):
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("--format", default="etc2-rgb", choices=["etc2-rgb", "etc2-rgba"])
    ap.add_argument("--out", type=Path, default=ROOT / "bench-results" / "etc2")
    ap.add_argument("--mipple", type=Path, default=ROOT / "target" / "release" / "mipple.exe")
    ap.add_argument("--cmp", type=Path,
                    default=ROOT / "third_party" / "dl" / "cmp" / "compressonatorcli-4.6.0-win64" / "compressonatorcli.exe")
    ap.add_argument("--efforts", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    ap.add_argument("--extra", default="", help="extra mipple encode arguments, e.g. '--no-srgb'")
    a = ap.parse_args()

    imgs = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".tga"))
    if a.limit:
        imgs = imgs[: a.limit]
    out_dir = a.out / a.format
    tmp = out_dir / "ktx2"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))

    encoders = []
    if a.mipple.exists():
        for e in a.efforts.split(","):
            encoders.append((f"mipple e{e}", "mipple", e))
    if a.cmp.exists():
        encoders.append(("compressonator", "cmp", ""))

    rows = []
    for img in imgs:
        for label, kind, param in encoders:
            out = tmp / f"{img.stem}__{label.replace(' ', '_')}.ktx2"
            if kind == "mipple":
                so, wall = run([str(a.mipple), "encode", str(img), "-o", str(out), "-f", a.format, "-e", param,
                                "--no-mips", *a.extra.split()], env)
                secs = float(re.search(r"time: base ([\d.]+)s", so).group(1))
            else:
                cfmt = "ETC2_RGB" if a.format == "etc2-rgb" else "ETC2_RGBA"
                _, wall = run([str(a.cmp), "-nomipmap", "-fd", cfmt, "-NumThreads", str(a.threads), str(img),
                               str(out)])
                secs = wall
            so, _ = run([str(a.mipple), "score", str(img), str(out), "--csv", "--label", label, "--secs", f"{secs}"])
            r = next(csv.reader(io.StringIO(so.strip().splitlines()[-1])))
            rows.append({"label": label, "image": img.name, "psnr": float(r[5]), "ssim": float(r[6]),
                         "zstd_bpt": float(r[11]), "secs": secs, "wall": wall})
            print(f"{img.name:24s} {label:16s} PSNR {rows[-1]['psnr']:7.3f}  SSIM {rows[-1]['ssim']:.5f}"
                  f"  bpt {rows[-1]['zstd_bpt']:.3f}  {secs:.3f}s", flush=True)

    with open(out_dir / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[r["label"]].append(r)
    lines = [f"{a.format}, {len(imgs)} images, {a.threads} thread(s). Means per image; time is the sum.", "",
             "| encoder | PSNR dB | SSIM | Zstd-19 bpt | time s |", "|---|---:|---:|---:|---:|"]
    for label, _, _ in encoders:
        rs = agg[label]
        n = len(rs)
        lines.append(f"| {label} | {sum(r['psnr'] for r in rs) / n:.3f} | {sum(r['ssim'] for r in rs) / n:.5f}"
                     f" | {sum(r['zstd_bpt'] for r in rs) / n:.3f} | {sum(r['secs'] for r in rs):.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
