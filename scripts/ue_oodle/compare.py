"""mipple vs Oodle Texture 2.9.16 as shipped in Unreal Engine 5.8, on Zstd-19.

Reads the blocks run_ue_oodle.py pulled out of UE's cook (one DDS per image, format and Oodle lambda), scores
them against the same source pixels UE encoded, runs a mipple lambda sweep on those pixels, and reports mipple's
size at Oodle's RMSE and at Oodle's SSIM (log-linear interpolation on mipple's Pareto front).

  python scripts/ue_oodle/compare.py --oodle <run_ue_oodle --out dir> --out bench-results/oodle-ue

Without the cook's dumps, --scores reuses Oodle's scores from an earlier results.json and only re-runs mipple
(BC1, BC4 and BC7, whose sources are the Kodak PNGs; BC5 needs the cook's R16G16 inputs):

  python scripts/ue_oodle/compare.py --scores bench-results/oodle-ue/results.json --formats bc7       --images kodim03 kodim05 kodim08 kodim13 kodim18 kodim24 --out <scratch dir>

Sources: BC1/BC7 are the PNGs UE imported, unaltered. BC4 is the R channel (UE widens it to 16 bits as R*257,
which is exact). BC5 goes through UE's normal-map import, which renormalizes to R16G16 before Oodle sees it;
the reference is that R16G16 input (bc5/inputs/<image>_IN.dds) in 8-bit units, and mipple encodes it rounded
to 8 bits (a slight handicap for mipple: Oodle saw the 16-bit values).
"""
import argparse
import csv
import json
import math
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SWEEP = [0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
CHANNELS = {"bc1": 3, "bc7": 3, "bc4": 1, "bc5": 2}


def load(p, ch):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)[:, :, :ch]


def rmse(a, b):
    """sqrt of the mean, over texels, of the summed squared channel errors."""
    return float(np.sqrt(((a - b) ** 2).sum(axis=2).mean()))


def ssim(a, b):
    """Mean SSIM over channels (8x8 windows, no Gaussian; enough to rank encodes)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        h, w = x.shape[0] // 8 * 8, x.shape[1] // 8 * 8
        xb = x[:h, :w].reshape(h // 8, 8, w // 8, 8)
        yb = y[:h, :w].reshape(h // 8, 8, w // 8, 8)
        mx, my = xb.mean((1, 3)), yb.mean((1, 3))
        vx, vy = xb.var((1, 3)), yb.var((1, 3))
        cov = ((xb - mx[:, None, :, None]) * (yb - my[:, None, :, None])).mean((1, 3))
        out.append((((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).mean())
    return float(np.mean(out))


def read_r16g16(path):
    d = Path(path).read_bytes()
    h, w = struct.unpack("<II", d[12:20])
    px = np.frombuffer(d[148:148 + w * h * 4], dtype="<u2").reshape(h, w, 2)
    return px.astype(np.float64) / 257.0


def decoded(exe, src, png, ch):
    subprocess.run([exe, "decode", str(src), "-o", str(png)], check=True, capture_output=True)
    return np.asarray(Image.open(png).convert("RGB")).astype(np.float64)[:, :, :ch]


def zstd_bpt(exe, ref_png, enc):
    f = subprocess.run([exe, "score", str(ref_png), str(enc), "--csv"], check=True, capture_output=True,
                       text=True).stdout.strip().split(",")
    return float(f[11])


def metrics(ref, dec):
    return rmse(ref, dec), ssim(ref, dec)


def at(curve, key, target):
    """mipple bpt at `target` of metric `key` (rmse: lower is better, ssim: higher), or None off the curve."""
    hi = key == "ssim"
    front = []
    for p in sorted(curve, key=lambda p: p["bpt"]):
        v = p[key] if hi else -p[key]
        if not front or v > front[-1][1]:
            front.append((p["bpt"], v))
    t = target if hi else -target
    for (b0, v0), (b1, v1) in zip(front, front[1:]):
        if v0 <= t <= v1 and v1 > v0:
            f = (t - v0) / (v1 - v0)
            return math.exp(math.log(b0) + f * (math.log(b1) - math.log(b0)))
    return None


def job(a, fmt, image, lams):
    """Scores Oodle's outputs and mipple's sweep for one (format, image)."""
    ch = CHANNELS[fmt]
    work = a.work / fmt / image
    work.mkdir(parents=True, exist_ok=True)
    if fmt == "bc5":
        ref = read_r16g16(a.oodle / "bc5" / "inputs" / f"{image}_IN.dds")
        src = work / "src.png"
        rg = np.clip(np.rint(ref), 0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([rg, np.zeros_like(rg[:, :, :1])]), "RGB").save(src)
    else:
        src = a.images[image]
        ref = load(src, ch)
    oodle = []
    for lam in lams if a.scores is None else []:
        dds = a.oodle / fmt / f"{image}_l{lam:g}.dds"
        r, s = metrics(ref, decoded(a.exe, dds, work / f"oodle_l{lam:g}.png", ch))
        oodle.append({"lambda": lam, "bpt": zstd_bpt(a.exe, src, dds), "rmse": r, "ssim": s})
    if a.scores is not None:
        oodle = [{k: o[k] for k in ("lambda", "bpt", "rmse", "ssim")} for o in a.scores[(fmt, image)]
                 if o["lambda"] in lams]
    mipple = []
    for lam in SWEEP:
        ktx = work / f"mipple_l{lam:g}.ktx2"
        argv = [a.exe, "encode", str(src), "-o", str(ktx), "-f", fmt, "-e", str(a.effort), "-l", f"{lam:g}",
                "--no-mips", *a.mipple_args.split()]
        if fmt in ("bc1", "bc7"):
            argv.append("--linear")  # Oodle's error is on stored values; no sRGB-aware filtering is involved
        subprocess.run(argv, check=True, capture_output=True)
        r, s = metrics(ref, decoded(a.exe, ktx, work / f"mipple_l{lam:g}.png", ch))
        mipple.append({"lambda": lam, "bpt": zstd_bpt(a.exe, src, ktx), "rmse": r, "ssim": s})
    for o in oodle:
        for key in ("rmse", "ssim"):
            b = at(mipple, key, o[key])
            o[f"mipple_bpt_at_{key}"] = b
            o[f"delta_at_{key}"] = None if b is None else (b / o["bpt"] - 1) * 100
    print(f"{fmt} {image}: " + " ".join(
        f"λ{o['lambda']:g} {fmt_pct(o['delta_at_rmse'])}/{fmt_pct(o['delta_at_ssim'])}" for o in oodle), flush=True)
    return {"format": fmt, "image": image, "oodle": oodle, "mipple": mipple}


def fmt_pct(v):
    return "  n/a " if v is None else f"{v:+5.1f}%"


def summarize(results, lams, a):
    out = ["# mipple vs Oodle Texture 2.9.16 (Unreal Engine 5.8)", "",
           "Oodle's blocks come from a normal UE 5.8.4 cook (Final encode speed, RDO on, lambda from Lossy "
           "Compression Amount; dumped with `-OodleDebugDumpFilter`, bit-exact with the cooked texture data). "
           "Both encoders are scored on the same source pixels, sizes are Zstd level 19 over the block data. "
           "Each cell is mipple's size at Oodle's quality relative to Oodle's size (negative = mipple smaller), "
           f"from a mipple -e {a.effort} lambda sweep" + (f" with `{a.mipple_args}`" if a.mipple_args else "")
           + ". Lambda 0 is Oodle without RDO.", ""]
    groups = {}
    for r in results:
        key = (r["format"], "Kodak-24" if r["image"].startswith("kodim") and r["image"] != "kodim18_original"
               else r["image"])
        groups.setdefault(key, []).append(r)
    for (fmt, name), rs in sorted(groups.items()):
        out += [f"## {fmt.upper()} {name}" + (f" ({len(rs)} images)" if len(rs) > 1 else ""), "",
                "| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |",
                "|---:|---:|---:|---:|---:|---:|"]
        for i, lam in enumerate(lams):
            os_ = [r["oodle"][i] for r in rs]
            bpt = sum(o["bpt"] for o in os_) / len(os_)

            def agg(key):
                # Total bits at matched quality over total Oodle bits (images off mipple's curve are skipped).
                pairs = [(o[f"mipple_bpt_at_{key}"], o["bpt"]) for o in os_ if o[f"mipple_bpt_at_{key}"]]
                if not pairs:
                    return "n/a"
                s = f"{(sum(m for m, _ in pairs) / sum(b for _, b in pairs) - 1) * 100:+.1f}%"
                return s if len(pairs) == len(os_) else f"{s} ({len(pairs)}/{len(os_)})"
            out.append(f"| {lam:g} | {bpt:.3f} | {sum(o['rmse'] for o in os_) / len(os_):.3f} | "
                       f"{sum(o['ssim'] for o in os_) / len(os_):.5f} | {agg('rmse')} | {agg('ssim')} |")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--oodle", type=Path, help="run_ue_oodle.py --out directory")
    src.add_argument("--scores", type=Path, help="reuse Oodle's scores from this results.json (no dumps needed)")
    p.add_argument("--out", type=Path, default=ROOT / "bench-results" / "oodle-ue")
    p.add_argument("--work", type=Path, help="scratch directory (default: <out>/work, not committed)")
    p.add_argument("--exe", default=str(ROOT / "target" / "release" / "mipple.exe"))
    p.add_argument("--effort", type=int, default=5)
    p.add_argument("--mipple-args", default="")
    p.add_argument("--formats", nargs="+", default=["bc1", "bc7", "bc4", "bc5"])
    p.add_argument("--images", nargs="+", help="only these images (default: all)")
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    a.work = a.work or a.out / "work"
    if a.scores is not None:
        prev = json.load(open(a.scores, encoding="utf-8"))
        lams = prev["settings"]["oodle_lambdas"]
        a.scores = {(r["format"], r["image"]): r["oodle"] for r in prev["results"]
                    if r["format"] in a.formats and r["format"] != "bc5"
                    and (a.images is None or r["image"] in a.images)}
        a.images = {img: ROOT / "test-images" / "kodak" / f"{img}.png" for _, img in a.scores}
        run(a, sorted(a.scores), lams)
        return
    rows = list(csv.DictReader(open(a.oodle / "manifest.csv", newline="")))
    run_info = json.load(open(a.oodle / "run_info.json"))
    a.images = {}
    for ri in run_info:
        args = ri["args"]
        i = args.index("--images") + 1
        while i < len(args) and not args[i].startswith("--"):
            a.images[Path(args[i]).stem] = ROOT / args[i]
            i += 1
    todo = {}
    for r in rows:
        if r["status"] == "ok" and r["fmt"] in a.formats and (a.images is None or r["image"] in a.images):
            todo.setdefault((r["fmt"], r["image"]), set()).add(float(r["lambda"]))
    run(a, sorted(todo), sorted(set.intersection(*todo.values())))


def run(a, keys, lams):
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda k: job(a, k[0], k[1], lams), keys))
    a.out.mkdir(parents=True, exist_ok=True)
    settings = {"oodle": "Oodle Texture 2.9.16 via Unreal Engine 5.8.4 (EncodeSpeed Final, bFinalUsesRDO)",
                "mipple_effort": a.effort, "mipple_args": a.mipple_args, "sweep": SWEEP, "oodle_lambdas": lams}
    (a.out / "results.json").write_text(json.dumps({"settings": settings, "results": results}, indent=1),
                                        encoding="utf-8", newline="\n")
    (a.out / "summary.md").write_text(summarize(results, lams, a) + "\n", encoding="utf-8", newline="\n")
    print((a.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
