#!/usr/bin/env python3
"""Single-thread Pareto scoreboard: does some mipple effort beat each competitor on both speed and quality?

Runs compete.py's base (non-RDO) configs for one format on a few images, pinned to one core at high priority,
interleaving configs over several rounds and keeping each (config, image)'s fastest time. Quality is scored once
(encoders are deterministic). For every competitor row it names the fastest mipple effort that is at least as fast
and at least as good (PSNR), or marks the row as unbeaten.

  py -3 scripts/scoreboard.py -f bc7 --images 3 --rounds 3
  py -3 scripts/scoreboard.py -f bc1 --mipple target/work/release/mipple.exe --skip compressonator,texconv
"""

import argparse
import csv
import ctypes
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import compete  # noqa: E402

TP = ROOT / "third_party"
DEFAULTS = {
    "bc7enc": TP / "bc7enc_rdo/build/Release/bc7enc.exe",
    "harness": TP / "build-harness/Release/bcn_harness.exe",
    "cmp": TP / "dl/cmp/compressonatorcli-4.6.0-win64/compressonatorcli.exe",
    "texconv": TP / "bin/texconv.exe",
}


def pin(mask):
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x80)  # HIGH_PRIORITY_CLASS, inherited
    else:
        os.sched_setaffinity(0, {i for i in range(mask.bit_length()) if mask >> i & 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--format", default="bc7")
    ap.add_argument("--corpus", type=Path, default=ROOT / "test-images/kodak")
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mipple", default=str(ROOT / "target/release/mipple.exe"))
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="compressonator,texconv")
    ap.add_argument("--affinity", type=lambda v: int(v, 0), default=0x4000)
    ap.add_argument("--csv", type=Path, help="also write rows here")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    pin(a.affinity)
    for k in ["mipple", *DEFAULTS]:
        p = Path(getattr(a, k)).resolve()
        setattr(a, k, str(p) if p.is_file() else None)
    if not a.mipple:
        sys.exit("mipple binary not found")
    a.threads, a.base_only, a.formats, a.bc7enc_window = 1, True, [a.format], 1024
    cfgs = compete.configs(a)
    if a.only:
        cfgs = [c for c in cfgs if any(k in c.label for k in a.only.split(","))]
    if a.skip:
        cfgs = [c for c in cfgs if not any(k in c.label for k in a.skip.split(","))]
    images = sorted(p.resolve() for p in a.corpus.iterdir() if p.suffix.lower() == ".png")[: a.images]
    work = Path(os.environ.get("TEMP", "/tmp")) / "mipple-scoreboard"
    work.mkdir(exist_ok=True)

    secs = defaultdict(lambda: 1e9)
    score = {}
    for r in range(a.rounds):
        for c in cfgs:
            for img in images:
                dst = work / (re.sub(r"[^A-Za-z0-9._-]+", "_", c.label) + "__" + img.stem + ".dds")
                t, _ = compete.run(c, img, dst, work)
                secs[c.label, img.name] = min(secs[c.label, img.name], t)
                if r == 0:
                    p = subprocess.run([a.mipple, "score", str(img), str(dst), "--csv"], capture_output=True,
                                       text=True, encoding="utf-8", check=True)
                    row = dict(zip(compete.SCORE_FIELDS, next(csv.reader([p.stdout.strip()]))))
                    score[c.label, img.name] = (float(row["psnr"]), int(row["zstd_bytes"]),
                                                int(row["width"]) * int(row["height"]))
                dst.unlink(missing_ok=True)
        print(f"round {r + 1}/{a.rounds} done", file=sys.stderr, flush=True)

    rows = []
    for c in cfgs:
        ps = [score[c.label, i.name] for i in images]
        rows.append(dict(label=c.label, mine=c.label.startswith("mipple"),
                         psnr=sum(p for p, _, _ in ps) / len(ps),
                         bpt=sum(z for _, z, _ in ps) * 8 / sum(n for _, _, n in ps),
                         secs=sum(secs[c.label, i.name] for i in images),
                         mpix=sum(n for _, _, n in ps) / 1e6))
    mine = [r for r in rows if r["mine"]]
    print(f"\n{a.format.upper()}: {len(images)} images, 1 thread pinned to mask {a.affinity:#x}, min of {a.rounds} rounds\n")
    print(f"| {'config':<24} | PSNR  | bpt   | ms      | Mpix/s | verdict |")
    print(f"| {'-' * 24} | ----- | ----- | ------- | ------ | ------- |")
    unbeaten = 0
    for r in sorted(rows, key=lambda r: r["secs"]):
        if r["mine"]:
            verdict = ""
        else:
            win = [m for m in mine if m["secs"] <= r["secs"] and m["psnr"] >= r["psnr"] - 0.005]
            if win:
                m = min(win, key=lambda m: m["secs"])
                verdict = f"beaten by {m['label'][7:]} ({r['secs'] / m['secs']:.2f}x, {m['psnr'] - r['psnr']:+.2f} dB)"
            else:
                unbeaten += 1
                verdict = "**UNBEATEN**"
        print(f"| {r['label']:<24} | {r['psnr']:.2f} | {r['bpt']:.3f} | {r['secs'] * 1000:7.1f} | "
              f"{r['mpix'] / r['secs']:6.2f} | {verdict} |")
    print(f"\nunbeaten competitor configs: {unbeaten}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
