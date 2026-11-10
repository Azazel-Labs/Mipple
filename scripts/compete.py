#!/usr/bin/env python3
"""Multi-format competitive benchmark: mipple vs third-party BCn encoders (BC1, BC4, BC5, BC7).

Every encoder writes a DDS; `mipple score` grades all of them with the same metrics and the same back
ends (Zstd-19, Deflate-9), so the only variable is the encoder.

Encoders (each is skipped if its path is not given or not found):
  --mipple     target/release/mipple
  --bc7enc     bc7enc_rdo built with SUPPORT_BC7E (bc7e.ispc, bc7enc.cpp, rgbcx BC1/BC4/BC5, RDO for all)
  --harness    tools/competitors bcn_harness (bc7f, Intel ISPC Texture Compressor, icbc, stb_dxt)
  --cmp        AMD compressonatorcli
  --texconv    Microsoft DirectXTex texconv

--threads N limits every encoder to N threads (mipple: RAYON_NUM_THREADS; bc7enc_rdo: OMP_NUM_THREADS;
harness: its thread argument; Compressonator: -NumThreads; texconv: -singleproc when N = 1).
--base-only skips RDO sweeps (use with --threads 1 for a per-core speed table).

Time is the encoder's own reported encode time where it prints one (mipple pipeline encode, bc7enc_rdo encoding + RDO
time, harness block loop) and process wall time otherwise (Compressonator, texconv); the wall
time is always recorded too.

On hybrid CPUs (Intel P/E cores) single-thread timings swing several-fold with the core the OS picks;
use --affinity with a performance core's mask for them.

Results in --out: rows.csv (appended as it goes; re-running resumes) and summary.md.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bdrate import bd_rate  # noqa: E402

SCORE_FIELDS = ["label", "image", "format", "width", "height", "psnr", "ssim", "normal_deg",
                "raw_bytes", "zstd_bytes", "deflate_bytes", "zstd_bpt", "deflate_bpt", "secs"]
FIELDS = SCORE_FIELDS + ["family", "x", "wall", "threads"]


class Cfg:
    def __init__(self, fmt, family, label, x, argv, env=None, out_name=None):
        self.fmt, self.family, self.label, self.x = fmt, family, label, x
        self.argv, self.env, self.out_name = argv, env or {}, out_name


# Distinct mipple effort tiers per format (efforts 0-9; BC1 efforts 8 and 9 are the same search; BC7 levels 0-2, 3-5, 6-7 and 8-9 share
# a search until those formats get finer tiers).
MIPPLE_EFFORTS = {"bc1": (0, 1, 2, 3, 4, 5, 6, 7, 8), "bc7": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "bc4": (0, 1, 2, 5, 6, 8), "bc5": (0, 1, 2, 5, 6, 8)}


def configs(a):
    T = a.threads
    out = []
    lam_m = [0.0] if a.base_only else [0.0, 0.25, 0.5, 1, 2, 4, 8]
    menv = {"RAYON_NUM_THREADS": str(T)}
    oenv = {"OMP_NUM_THREADS": str(T)}

    def mipple(fmt):
        if not a.mipple:
            return
        for e in MIPPLE_EFFORTS[fmt]:
            out.append(Cfg(fmt, f"mipple e{e}", f"mipple e{e}", None,
                           lambda s, d, e=e, fmt=fmt: [a.mipple, "encode", s, "-o", d, "-f", fmt, "-e", str(e), "--no-mips"],
                           menv))
        if not a.base_only:
            for lam in lam_m:
                out.append(Cfg(fmt, "mipple RDO", f"mipple e5 λ={lam:g}", lam,
                               lambda s, d, lam=lam, fmt=fmt: [a.mipple, "encode", s, "-o", d, "-f", fmt, "-e", "5",
                                                               "-l", str(lam), "--no-mips", *a.mipple_args.split()],
                               menv))

    def bc7enc(fmt, flag, levels, lams, extra=()):
        if not a.bc7enc:
            return
        name = {"bc1": "rgbcx", "bc4": "rgbcx", "bc5": "rgbcx", "bc7": "bc7e"}[fmt]
        for lvl_flag, lvl_label in levels:
            out.append(Cfg(fmt, f"{name} {lvl_label}", f"{name} {lvl_label}", None,
                           lambda s, d, lf=lvl_flag: [a.bc7enc, "-g", *flag, *lf, s, d], oenv))
        if not a.base_only:
            top_flag, top_label = levels[-1]
            for lam in lams:
                rdo = [f"-z{lam:g}", f"-zc{a.bc7enc_window}"] if lam > 0 else []
                out.append(Cfg(fmt, f"bc7enc_rdo {name}", f"bc7enc_rdo {name} {top_label} λ={lam:g}", lam,
                               lambda s, d, rdo=rdo, lf=top_flag: [a.bc7enc, "-g", *flag, *lf, *rdo, s, d], oenv))
            for ex_flags, ex_label in extra:
                for lam in lams:
                    rdo = [f"-z{lam:g}", f"-zc{a.bc7enc_window}"] if lam > 0 else []
                    out.append(Cfg(fmt, f"bc7enc_rdo {ex_label}", f"bc7enc_rdo {ex_label} λ={lam:g}", lam,
                                   lambda s, d, rdo=rdo, ef=ex_flags: [a.bc7enc, "-g", *flag, *ef, *rdo, s, d], oenv))

    def harness(fmt, encs):
        if not a.harness:
            return
        for e in encs:
            fam = e.split(":")[0]
            out.append(Cfg(fmt, e, e, None, lambda s, d, e=e: [a.harness, e, s, d, str(T)]))

    def cmp(fmt, cfmt, variants=(([], ""),)):
        if not a.cmp:
            return
        for extra, lab in variants:
            label = f"compressonator {cfmt}{(' ' + lab) if lab else ''}"
            out.append(Cfg(fmt, label, label, None,
                           lambda s, d, extra=extra: [a.cmp, "-nomipmap", "-fd", cfmt, "-NumThreads", str(T), *extra, s, d]))

    def texconv(fmt, tfmt, variants=(([], ""),)):
        if not a.texconv:
            return
        for extra, lab in variants:
            label = f"texconv {tfmt}{(' ' + lab) if lab else ''}"
            # texconv names the output after the input; run() moves it into place.
            out.append(Cfg(fmt, label, label, None,
                           lambda s, d, extra=extra: [a.texconv, "-nologo", "-y", "-nogpu", "--ignore-srgb", "-m", "1",
                                                      *(["-singleproc"] if T == 1 else []), "-f", tfmt, *extra,
                                                      "-o", str(Path(d).parent), s],
                           out_name="texconv"))

    rdo_lams = [0.0, 0.1, 0.25, 0.5, 1, 2, 4]
    for fmt in a.formats:
        mipple(fmt)
        if fmt == "bc7":
            bc7enc("bc7", ["-U"], [(["-u0"], "u0"), (["-u2"], "u2"), (["-u4"], "u4"), (["-u6"], "u6")],
                   [0.0, 0.05, 0.1, 0.25, 0.5, 1], extra=[(["-C", "-e", "-u4"], "bc7enc.cpp -e")])
            harness("bc7", [f"bc7f:{p}" for p in ("fastest", "faster", "fast", "default", "partial")] +
                    [f"ispc_bc7:{p}" for p in ("ultrafast", "veryfast", "fast", "basic", "slow")])
            cmp("bc7", "BC7", [(["-Quality", "0.05"], "q0.05"), (["-Quality", "0.2"], "q0.2"), (["-Quality", "0.6"], "q0.6")])
            texconv("bc7", "BC7_UNORM", [(["-bc", "q"], "quick")])
        elif fmt == "bc1":
            bc7enc("bc1", ["-1"], [(["-L0"], "L0"), (["-L6"], "L6"), (["-L12"], "L12"), (["-L18"], "L18")], rdo_lams)
            harness("bc1", ["icbc:1", "icbc:5", "icbc:8", "icbc:9", "stb_bc1", "stb_bc1hq", "ispc_bc1"])
            cmp("bc1", "BC1")
            texconv("bc1", "BC1_UNORM")
        elif fmt == "bc4":
            bc7enc("bc4", ["-4"], [(["-hl"], "hl"), ([], "hr5")], rdo_lams)
            harness("bc4", ["stb_bc4", "ispc_bc4"])
            cmp("bc4", "ATI1N")
            texconv("bc4", "BC4_UNORM")
        elif fmt == "bc5":
            bc7enc("bc5", ["-5"], [(["-hl"], "hl"), ([], "hr5")], rdo_lams)
            harness("bc5", ["stb_bc5", "ispc_bc5"])
            cmp("bc5", "ATI2N")
            texconv("bc5", "BC5_UNORM")
    return out


TIME_PATTERNS = [r"secs=([0-9.]+)", r"encode ([0-9.]+)s"]


def pin_cpus(mask):
    """Pins this process to `mask`; child processes inherit it."""
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            sys.exit(f"SetProcessAffinityMask({mask:#x}) failed")
    else:
        os.sched_setaffinity(0, {i for i in range(mask.bit_length()) if mask >> i & 1})


def run(cfg, src, dst, work):
    env = dict(os.environ, **cfg.env)
    argv = [str(x) for x in cfg.argv(str(src), str(dst))]
    t0 = time.perf_counter()
    p = subprocess.run(argv, cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(argv)}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    if cfg.out_name == "texconv":
        produced = Path(dst).parent / (Path(src).stem + ".dds")
        os.replace(produced, dst)
    secs = wall
    enc = re.search(r"Total encoding time: ([0-9.]+) secs", p.stdout)
    if enc:  # bc7enc_rdo: base encode + RDO pass, excluding PNG load, stats and file write
        rdo = re.search(r"Total RDO time: ([0-9.]+) secs", p.stdout)
        return float(enc.group(1)) + (float(rdo.group(1)) if rdo else 0.0), wall
    for pat in TIME_PATTERNS:
        m = re.search(pat, p.stdout)
        if m:
            secs = float(m.group(1))
            break
    return secs, wall


def summarize(rows, images, args):
    lines = [f"# Competitive benchmark\n",
             f"Corpus: {len(images)} images from `{args.corpus}`; threads per encoder: {args.threads}. "
             "Sizes are corpus totals after each back end; PSNR is the per-image mean over the format's channels; "
             "time is total encode seconds (see scripts/compete.py for what each encoder's time covers).\n"]
    by_fmt = defaultdict(list)
    for r in rows:
        by_fmt[r["format"]].append(r)
    for fmt in ["bc1", "bc4", "bc5", "bc7"]:
        frows = by_fmt.get(fmt)
        if not frows:
            continue
        agg = OrderedDict()
        for r in frows:
            a = agg.setdefault(r["label"], {"family": r["family"], "x": r["x"], "n": 0, "psnr": 0.0, "zstd": 0,
                                            "deflate": 0, "texels": 0, "secs": 0.0})
            a["n"] += 1
            a["psnr"] += float(r["psnr"])
            a["zstd"] += int(r["zstd_bytes"])
            a["deflate"] += int(r["deflate_bytes"])
            a["texels"] += int(r["width"]) * int(r["height"])
            a["secs"] += float(r["secs"])
        curves = defaultdict(lambda: {"zstd": [], "deflate": []})
        pts = []
        for label, a in agg.items():
            if a["n"] != len(images):
                continue  # incomplete config
            psnr = a["psnr"] / a["n"]
            zb, db = a["zstd"] * 8 / a["texels"], a["deflate"] * 8 / a["texels"]
            mps = a["texels"] / 1e6 / max(a["secs"], 1e-9)
            pts.append((label, a["family"], a["x"], psnr, zb, db, a["secs"], mps))
            if a["x"] not in (None, "", "None"):
                curves[a["family"]]["zstd"].append((zb, psnr))
                curves[a["family"]]["deflate"].append((db, psnr))
        mcurve = sorted(curves.get("mipple RDO", {"zstd": []})["zstd"], key=lambda p: p[1])

        def vs_mipple_curve(psnr, zb):
            # Size of this point relative to mipple's RDO curve at equal PSNR (log-linear interpolation).
            for (r0, p0), (r1, p1) in zip(mcurve, mcurve[1:]):
                if p0 <= psnr <= p1 and p1 > p0:
                    t = (psnr - p0) / (p1 - p0)
                    mr = math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0)))
                    return f"{(zb / mr - 1) * 100:+.1f}%"
            return "—"

        lines += [f"\n## {fmt.upper()}\n",
                  "| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for label, fam, x, psnr, zb, db, secs, mps in pts:
            lines.append(f"| {label} | {psnr:.2f} | {zb:.3f} | {db:.3f} | {secs:.2f} | {mps:.2f} | {vs_mipple_curve(psnr, zb)} |")
        # Pareto check: a competitor config is beaten when some mipple base config is at least as
        # good (PSNR) and at least as fast (Mpix/s).
        mine = [p for p in pts if p[1].startswith("mipple e")]
        theirs = [p for p in pts if not p[1].startswith("mipple")]
        if mine and theirs:
            lines += ["", "Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast "
                      "as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):", "",
                      "| Competitor | PSNR | Mpix/s | Beaten by | Gap |", "| --- | --- | --- | --- | --- |"]
            for label, fam, x, psnr, zb, db, secs, mps in theirs:
                win = [m for m in mine if m[3] >= psnr - 0.005 and m[7] >= mps]
                if win:
                    w = max(win, key=lambda m: m[7])
                    lines.append(f"| {label} | {psnr:.2f} | {mps:.2f} | {w[0]} ({w[3]:.2f} dB, {w[7]:.2f} Mpix/s) | ✓ |")
                else:
                    # Closest miss: the fastest mipple config with enough quality, else the best quality.
                    good = [m for m in mine if m[3] >= psnr - 0.005]
                    if good:
                        m = max(good, key=lambda m: m[7])
                        gap = f"{m[0]} needs {mps / m[7]:.2f}× speed"
                    else:
                        m = max(mine, key=lambda m: m[3])
                        gap = f"best is {m[0]}, {psnr - m[3]:.2f} dB short"
                    lines.append(f"| {label} | {psnr:.2f} | {mps:.2f} | ✗ | {gap} |")
        if "mipple RDO" in curves and len(curves) > 1:
            lines += ["", "BD-rate of **mipple RDO** vs each RDO family (negative = mipple needs fewer bits at equal PSNR):", "",
                      "| vs family | Zstd-19 | Deflate-9 |", "| --- | --- | --- |"]
            m = curves["mipple RDO"]
            for fam, c in curves.items():
                if fam == "mipple RDO":
                    continue
                z, d = bd_rate(c["zstd"], m["zstd"]), bd_rate(c["deflate"], m["deflate"])
                f = lambda v: "n/a (no overlap)" if v is None else f"{v:+.1f}%"
                lines.append(f"| {fam} | {f(z)} | {f(d)} |")
    return "\n".join(lines) + "\n"


def main():
    # Labels contain λ; Windows consoles default to a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--formats", default="bc1,bc4,bc5,bc7")
    ap.add_argument("--mipple", required=True)
    ap.add_argument("--mipple-args", default="", help="extra arguments for mipple's RDO runs (e.g. '--max-rms-ratio 5')")
    ap.add_argument("--bc7enc")
    ap.add_argument("--harness")
    ap.add_argument("--cmp")
    ap.add_argument("--texconv")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--bc7enc-window", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--interleave", action="store_true",
                    help="with --repeat: per image, run all configs in rotation and keep each one's fastest "
                         "time (robust to load that drifts over minutes)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each encode this many times and keep the fastest (outputs are deterministic); "
                         "makes timings robust to background load")
    ap.add_argument("--only", default="", help="comma-separated substrings; run only configs whose label matches one")
    ap.add_argument("--skip", default="", help="comma-separated substrings; skip configs whose label matches one")
    ap.add_argument("--affinity", type=lambda v: int(v, 0), default=0,
                    help="CPU mask (e.g. 0x4000) to pin this script and every encoder it runs to; "
                         "needed for meaningful single-thread timings on hybrid CPUs (P/E cores)")
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary.md from rows.csv")
    args = ap.parse_args()
    if args.affinity:
        pin_cpus(args.affinity)
    args.formats = args.formats.split(",")
    for name in ("mipple", "bc7enc", "harness", "cmp", "texconv"):
        v = getattr(args, name)
        if v:
            p = Path(v).expanduser().resolve()
            if not p.is_file():
                print(f"warning: --{name} {p} not found; skipping", file=sys.stderr)
                setattr(args, name, None)
            else:
                setattr(args, name, str(p))
    images = sorted(p.resolve() for p in args.corpus.iterdir() if p.suffix.lower() == ".png")
    if args.limit:
        images = images[: args.limit]
    work = (args.out / "work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.csv"
    rows = []
    if rows_path.exists():
        with open(rows_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["label"], r["format"], r["image"]) for r in rows}

    if not args.summary_only:
        cfgs = configs(args)
        if args.only:
            keys = args.only.split(",")
            cfgs = [c for c in cfgs if any(k in c.label for k in keys)]
        if args.skip:
            keys = args.skip.split(",")
            cfgs = [c for c in cfgs if not any(k in c.label for k in keys)]
        new_file = not rows_path.exists()
        f = open(rows_path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        print(f"{len(images)} images × {len(cfgs)} configs, threads={args.threads}", flush=True)
        def score(cfg, img, dst, secs, wall):
            p = subprocess.run([args.mipple, "score", str(img), str(dst), "--csv", "--label", cfg.label,
                                "--secs", f"{secs:.6f}"], capture_output=True, text=True, encoding="utf-8")
            if p.returncode != 0:
                print(f"  score failed for {dst}: {p.stderr}", file=sys.stderr, flush=True)
                return None
            row = dict(zip(SCORE_FIELDS, next(csv.reader([p.stdout.strip()]))))
            row.update(family=cfg.family, x="" if cfg.x is None else cfg.x, wall=f"{wall:.4f}", threads=args.threads)
            w.writerow(row)
            f.flush()
            rows.append(row)
            dst.unlink()
            return row

        def dst_for(cfg, img):
            return work / (re.sub(r"[^A-Za-z0-9._-]+", "_", f"{cfg.fmt}_{cfg.label}") + "__" + img.stem + ".dds")

        if args.interleave:
            # Per image, every config runs in rotation `repeat` times and keeps its fastest time,
            # so a load change over minutes affects all configs alike.
            failed = set()
            for img in images:
                todo = [c for c in cfgs if (c.label, c.fmt, img.name) not in done and (c.fmt, c.label) not in failed]
                best = {}
                for _ in range(args.repeat):
                    for cfg in todo:
                        if (cfg.fmt, cfg.label) in failed:
                            continue
                        try:
                            secs, wall = run(cfg, img, dst_for(cfg, img), work)
                        except RuntimeError as e:
                            print(f"  {cfg.fmt} {cfg.label}: {e}", file=sys.stderr, flush=True)
                            failed.add((cfg.fmt, cfg.label))
                            continue
                        key = (cfg.fmt, cfg.label)  # labels repeat across formats
                        b = best.get(key)
                        best[key] = (min(secs, b[0]), min(wall, b[1])) if b else (secs, wall)
                for cfg in todo:
                    key = (cfg.fmt, cfg.label)
                    if key in best and key not in failed:
                        score(cfg, img, dst_for(cfg, img), *best[key])
                print(f"  {img.name}: {len(todo)} configs", flush=True)
        else:
            for cfg in cfgs:
                t_cfg = 0.0
                sel = []
                for img in images:
                    if (cfg.label, cfg.fmt, img.name) in done:
                        continue
                    dst = dst_for(cfg, img)
                    try:
                        secs, wall = run(cfg, img, dst, work)
                        for _ in range(args.repeat - 1):
                            s2, w2 = run(cfg, img, dst, work)
                            secs, wall = min(secs, s2), min(wall, w2)
                    except RuntimeError as e:
                        print(f"  {cfg.fmt} {cfg.label}: {e}", file=sys.stderr, flush=True)
                        break
                    row = score(cfg, img, dst, secs, wall)
                    if row is None:
                        break
                    sel.append(row)
                    t_cfg += secs
                if sel:
                    print(f"  {cfg.fmt} {cfg.label:<40} PSNR {sum(float(r['psnr']) for r in sel) / len(sel):6.2f}  "
                          f"zstd {sum(int(r['zstd_bytes']) for r in sel) * 8 / sum(int(r['width']) * int(r['height']) for r in sel):5.3f} bpt  "
                          f"time {t_cfg:8.2f}s", flush=True)
        f.close()
    (args.out / "summary.md").write_text(summarize(rows, images, args), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
