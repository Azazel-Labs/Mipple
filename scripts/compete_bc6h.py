#!/usr/bin/env python3
"""BC6H (HDR) benchmark: mipple vs Intel ISPC Texture Compressor, Microsoft texconv and AMD Compressonator.

Every encoder writes a BC6H_UF16 DDS from the same Radiance .hdr files; `mipple score` grades all of them
in linear light: mPSNR (multi-exposure PSNR, Munkberg et al. 2006, over each image's exposure range)
and log2 RMSE (about the error in stops), plus Zstd-19 size.

Encoders (each skipped if not found):
  --mipple     target/release/mipple (efforts from --efforts)
  --harness    tools/competitors bcn_harness built with ISPC (ispc_bc6h profiles)
  --texconv    third_party/bin/texconv.exe (BC6H_UF16; -singleproc with --threads 1)
  --cmp        compressonatorcli (-fd BC6H, -Quality from --cmp-quality; reads a texconv half-float DDS,
               as it does not load .hdr)

Time: mipple's own base-encode time, the harness's block loop, and process wall time for texconv and
Compressonator (which includes loading and writing). --threads N limits every encoder to N threads.

Usage: python scripts/compete_bc6h.py test-images/polyhaven-hdri --out bench-results/bc6h
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
    ap.add_argument("--out", type=Path, default=ROOT / "bench-results" / "bc6h")
    ap.add_argument("--mipple", type=Path, default=ROOT / "target" / "release" / "mipple.exe")
    ap.add_argument("--harness", type=Path, default=ROOT / "third_party" / "build-harness" / "Release" / "bcn_harness.exe")
    ap.add_argument("--texconv", type=Path, default=ROOT / "third_party" / "bin" / "texconv.exe")
    ap.add_argument("--cmp", type=Path,
                    default=ROOT / "third_party" / "dl" / "cmp" / "compressonatorcli-4.6.0-win64" / "compressonatorcli.exe")
    ap.add_argument("--efforts", default="0,2,5,9")
    ap.add_argument("--ispc", default="veryfast,fast,basic,slow,veryslow")
    ap.add_argument("--cmp-quality", default="0.1,1.0")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    a = ap.parse_args()

    imgs = sorted(a.images.glob("*.hdr"))
    if a.limit:
        imgs = imgs[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "dds"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))

    encoders = []
    if a.mipple.exists():
        for e in a.efforts.split(","):
            encoders.append((f"mipple e{e}", "mipple", e))
    if a.harness.exists():
        for p in a.ispc.split(","):
            encoders.append((f"ispc {p}", "ispc", p))
    if a.texconv.exists():
        encoders.append(("texconv", "texconv", ""))
    if a.cmp.exists() and a.texconv.exists():
        for q in a.cmp_quality.split(","):
            encoders.append((f"cmp q{q}", "cmp", q))

    rows = []
    for img in imgs:
        for label, kind, param in encoders:
            out = tmp / f"{img.stem}__{label.replace(' ', '_')}.dds"
            if kind == "mipple":
                so, wall = run([str(a.mipple), "encode", str(img), "-o", str(out), "-f", "bc6h", "-e", param,
                                "--no-mips"], env)
                secs = float(re.search(r"time: base ([\d.]+)s", so).group(1))
            elif kind == "ispc":
                so, wall = run([str(a.harness), f"ispc_bc6h:{param}", str(img), str(out), str(a.threads)])
                secs = float(re.search(r"secs=([\d.]+)", so).group(1))
            elif kind == "texconv":
                d = tmp / "texconv"
                d.mkdir(exist_ok=True)
                cmd = [str(a.texconv), "-nologo", "-y", "-f", "BC6H_UF16", "-m", "1", "-o", str(d)]
                if a.threads == 1:
                    cmd.append("-singleproc")
                _, wall = run(cmd + [str(img)])
                (d / f"{img.stem}.dds").replace(out)
                secs = wall
            else:
                # Compressonator does not read .hdr: give it the texconv half-float DDS of the same image
                # (converted once, untimed).
                src = tmp / "half" / f"{img.stem}.dds"
                if not src.exists():
                    src.parent.mkdir(exist_ok=True)
                    run([str(a.texconv), "-nologo", "-y", "-f", "R16G16B16A16_FLOAT", "-m", "1", "-o",
                         str(src.parent), str(img)])
                _, wall = run([str(a.cmp), "-fd", "BC6H", "-Quality", param, "-NumThreads", str(a.threads),
                               "-nomipmap", str(src), str(out)])
                secs = wall
            so, _ = run([str(a.mipple), "score", str(img), str(out), "--csv", "--label", label, "--secs", f"{secs}"])
            r = next(csv.reader(io.StringIO(so.strip().splitlines()[-1])))
            rows.append({"label": label, "image": img.name, "mpsnr": float(r[5]), "zstd_bpt": float(r[11]),
                         "secs": secs, "wall": wall, "log_rmse": float(r[14])})
            print(f"{img.name:40s} {label:16s} mPSNR {rows[-1]['mpsnr']:7.3f}  logRMSE {rows[-1]['log_rmse']:.5f}"
                  f"  bpt {rows[-1]['zstd_bpt']:.3f}  {secs:.3f}s", flush=True)

    with open(a.out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[r["label"]].append(r)
    lines = [f"{len(imgs)} images, {a.threads} thread(s). Means per image; time is the sum.", "",
             "| encoder | mPSNR dB | log2 RMSE | Zstd-19 bpt | time s |", "|---|---:|---:|---:|---:|"]
    for label, _, _ in encoders:
        rs = agg[label]
        n = len(rs)
        lines.append(f"| {label} | {sum(r['mpsnr'] for r in rs) / n:.3f} | {sum(r['log_rmse'] for r in rs) / n:.5f}"
                     f" | {sum(r['zstd_bpt'] for r in rs) / n:.3f} | {sum(r['secs'] for r in rs):.3f} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
