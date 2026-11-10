#!/usr/bin/env python3
"""Head-to-head BC7 benchmark: mipple vs bc7enc_rdo vs bc7f (vs bc7g once released).

Every encoder writes a DDS; `mipple score` grades all of them with the same metrics and the same
back ends (Zstd-19, Deflate-9), so the only variable is the encoder. Results:
  <out>/rows.csv     one row per (encoder config, image)
  <out>/summary.md   per-config totals, plus BD-rate of each RDO curve vs a reference curve

Usage (stdlib only):
  python3 scripts/head2head.py --corpus test-images/kodak --out bench-results/h2h \
      --mipple target/release/mipple \
      --bc7enc ~/src/bc7enc_rdo/build/bc7enc \
      --bc7f tools/bc7f/bc7f_encode
Any encoder whose path is omitted is skipped.
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

CSV_FIELDS = ["label", "image", "format", "width", "height", "psnr", "ssim", "normal_deg",
              "raw_bytes", "zstd_bytes", "deflate_bytes", "zstd_bpt", "deflate_bpt", "secs"]


def run(cmd, cwd=None):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p.stdout, dt


def configs(args):
    """(family, label, curve_x, function(src, dst) -> argv) for every encoder configuration."""
    out = []
    if args.mipple:
        for lam in args.lambdas:
            out.append(("mipple", f"mipple e{args.effort} λ={lam}", lam,
                        lambda s, d, lam=lam: [args.mipple, "encode", s, "-o", d, "-f", "bc7",
                                               "-e", str(args.effort), "-l", str(lam), "--no-mips"]))
    if args.bc7enc:
        for lam in args.bc7enc_lambdas:
            rdo = ["-z" + str(lam), f"-zc{args.bc7enc_window}"] if lam > 0 else []
            out.append(("bc7enc_rdo", f"bc7enc_rdo u4 λ={lam}", lam,
                        lambda s, d, rdo=rdo: [args.bc7enc, "-q", "-g", "-u4", *rdo, s, d]))
        for lam in args.bc7enc_lambdas:
            rdo = ["-z" + str(lam), f"-zc{args.bc7enc_window}"] if lam > 0 else []
            out.append(("bc7enc_rdo -e", f"bc7enc_rdo -C -e λ={lam}", lam,
                        lambda s, d, rdo=rdo: [args.bc7enc, "-q", "-g", "-C", "-e", "-u4", *rdo, s, d]))
    if args.bc7f:
        for preset in ["fastest", "faster", "fast", "default", "partial", "nonanalytical"]:
            out.append(("bc7f", f"bc7f {preset}", None,
                        lambda s, d, preset=preset: [args.bc7f, s, d, preset]))
    if args.bc7g:
        # bc7g is announced but unreleased; adjust the argv once its CLI exists.
        out.append(("bc7g", "bc7g", None, lambda s, d: [args.bc7g, s, d]))
    return out


def bd_rate(ref, test):
    """Bjøntegaard delta rate (%) of `test` vs `ref`; each is a list of (rate, psnr).

    Same method as `mipple check` (src/check.rs `bd_rate`): each curve is reduced to its Pareto
    front (dominated operating points dropped), log(rate) is interpolated piecewise-linearly in
    PSNR, and the mean difference over the overlapping PSNR range is returned. Negative = test needs
    fewer bits. (A least-squares cubic swings wildly on short or non-monotonic RDO sweeps.)"""
    def front(points):
        pts = sorted(((r, q) for r, q in points if r > 0), key=lambda p: (-p[1], p[0]))
        out = []
        for r, q in pts:  # highest PSNR first; keep only points smaller than everything above
            if not out or (r < out[-1][0] and q < out[-1][1]):
                out.append((r, q))
        return out[::-1]

    def ln_rate_at(f, q):
        k = min(max(sum(1 for p in f if p[1] < q), 1), len(f) - 1)
        (r0, q0), (r1, q1) = f[k - 1], f[k]
        t = min(max((q - q0) / (q1 - q0), 0.0), 1.0)
        return math.log(r0) + t * (math.log(r1) - math.log(r0))

    r, t = front(ref), front(test)
    if len(r) < 2 or len(t) < 2:
        return None
    lo, hi = max(r[0][1], t[0][1]), min(r[-1][1], t[-1][1])
    if hi - lo < 1e-6:
        return None
    xs = sorted({lo, hi} | {p[1] for p in r + t if lo < p[1] < hi})
    diff = lambda q: ln_rate_at(t, q) - ln_rate_at(r, q)
    area = sum(0.5 * (diff(a) + diff(b)) * (b - a) for a, b in zip(xs, xs[1:]))
    return (math.exp(area / (hi - lo)) - 1.0) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mipple", required=True)
    ap.add_argument("--bc7enc")
    ap.add_argument("--bc7f")
    ap.add_argument("--bc7g")
    ap.add_argument("--lambdas", default="0,0.25,0.5,1,2,4,8", help="mipple λ sweep")
    ap.add_argument("--bc7enc-lambdas", default="0,0.05,0.1,0.25,0.5,1",
                    help="bc7enc_rdo λ sweep (its λ scale is much stronger than mipple's)")
    ap.add_argument("--effort", type=int, default=5)
    ap.add_argument("--bc7enc-window", type=int, default=1024)
    ap.add_argument("--reference", default="bc7enc_rdo", help="family the BD-rates are measured against")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--include-rows", type=Path,
                    help="rows.csv from an earlier run: its families are added to the summary "
                         "(families re-run now replace them), so slow encoders need not be re-run")
    args = ap.parse_args()
    args.lambdas = [float(x) for x in args.lambdas.split(",")]
    args.bc7enc_lambdas = [float(x) for x in args.bc7enc_lambdas.split(",")]
    # Encoders run with cwd = the work dir, so resolve their paths up front.
    for name in ("mipple", "bc7enc", "bc7f", "bc7g"):
        if getattr(args, name):
            path = Path(getattr(args, name)).expanduser().resolve()
            if not path.is_file():
                if name == "mipple":
                    sys.exit(f"mipple binary not found: {path} (cargo build --release)")
                print(f"warning: --{name} {path} not found; skipping {name}", file=sys.stderr)
                setattr(args, name, None)
            else:
                setattr(args, name, str(path))

    images = sorted(p for p in args.corpus.iterdir() if p.suffix.lower() == ".png")
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"no PNGs in {args.corpus}")
    work = args.out / "work"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    cfgs = configs(args)
    print(f"{len(images)} images × {len(cfgs)} configs")
    for family, label, x, argv in cfgs:
        for img in images:
            dst = work / (re.sub(r"[^A-Za-z0-9._-]+", "_", label) + "__" + img.stem + ".dds")
            stdout, wall = run(argv(str(img.resolve()), str(dst.resolve())), cwd=work)
            m = re.search(r"secs=([0-9.]+)", stdout)  # bc7f harness reports pure encode time
            secs = float(m.group(1)) if m else wall
            line, _ = run([args.mipple, "score", str(img), str(dst), "--csv", "--label", label, "--secs", f"{secs:.4f}"])
            row = dict(zip(CSV_FIELDS, next(csv.reader([line.strip()]))))
            row["family"], row["x"] = family, x
            rows.append(row)
            dst.unlink()
        sel = [r for r in rows if r["label"] == label]
        print(f"  {label:<28} PSNR {sum(float(r['psnr']) for r in sel) / len(sel):6.2f} dB  "
              f"zstd {sum(int(r['zstd_bytes']) for r in sel) * 8 / sum(int(r['width']) * int(r['height']) for r in sel):5.3f} bpt  "
              f"time {sum(float(r['secs']) for r in sel):7.2f}s", flush=True)

    if args.include_rows:
        fresh = {r["family"] for r in rows}
        with open(args.include_rows, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["family"] not in fresh:
                    r["x"] = float(r["x"]) if r["x"] not in ("", "None") else None
                    rows.append(r)

    with open(args.out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS + ["family", "x"])
        w.writeheader()
        w.writerows(rows)

    # Per-config summary.
    agg = OrderedDict()
    for r in rows:
        a = agg.setdefault(r["label"], {"family": r["family"], "x": r["x"], "n": 0, "psnr": 0.0, "ssim": 0.0,
                                       "zstd": 0, "deflate": 0, "texels": 0, "secs": 0.0})
        a["n"] += 1
        a["psnr"] += float(r["psnr"])
        a["ssim"] += float(r["ssim"])
        a["zstd"] += int(r["zstd_bytes"])
        a["deflate"] += int(r["deflate_bytes"])
        a["texels"] += int(r["width"]) * int(r["height"])
        a["secs"] += float(r["secs"])
    lines = [f"Corpus: {len(images)} images from `{args.corpus}`. Sizes are corpus totals after each back end; "
             "PSNR/SSIM are per-image means; time is total encode seconds.\n",
             "| Encoder config | Mean PSNR (dB) | Mean SSIM | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    curves = defaultdict(lambda: {"zstd": [], "deflate": []})
    for label, a in agg.items():
        psnr = a["psnr"] / a["n"]
        zb = a["zstd"] * 8 / a["texels"]
        db = a["deflate"] * 8 / a["texels"]
        lines.append(f"| {label} | {psnr:.2f} | {a['ssim'] / a['n']:.4f} | {zb:.3f} | {db:.3f} | "
                     f"{a['secs']:.2f} | {a['texels'] / 1e6 / max(a['secs'], 1e-9):.2f} |")
        if a["x"] is not None:
            curves[a["family"]]["zstd"].append((zb, psnr))
            curves[a["family"]]["deflate"].append((db, psnr))

    if args.reference in curves:
        lines += ["", f"BD-rate vs **{args.reference}** (negative = fewer bits at equal PSNR):", "",
                  "| Family | Zstd-19 | Deflate-9 |", "| --- | --- | --- |"]
        ref = curves[args.reference]
        for fam, c in curves.items():
            if fam == args.reference:
                continue
            z = bd_rate(ref["zstd"], c["zstd"])
            d = bd_rate(ref["deflate"], c["deflate"])
            fmt = lambda v: "n/a (curves don't overlap)" if v is None else f"{v:+.1f}%"
            lines.append(f"| {fam} | {fmt(z)} | {fmt(d)} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
