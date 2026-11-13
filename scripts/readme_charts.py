#!/usr/bin/env python3
"""Regenerate the README's results charts and table from the committed benchmark rows.

    python scripts/readme_charts.py [--rows bench-results/compete-kodak24-rdo/rows.csv]

Reads the RDO sweeps that scripts/compete.py appends to rows.csv (mipple vs bc7enc_rdo, every format) and the
Oodle Texture 2.9.16 comparison from scripts/ue_oodle/compare.py (bench-results/oodle-ue/results.json: Oodle's
own blocks from a stock Unreal Engine 5.8 cook of Kodak-24), and writes:

  docs/img/bits.svg        % fewer bits than each competitor at equal quality (bc7enc_rdo and Oodle Texture)
  docs/img/oodle.svg       bits per texel vs Oodle Texture per Oodle λ, at Oodle's RMSE and at its SSIM
  docs/img/rd-curves.svg   rate-distortion curves per format vs bc7enc_rdo (Zstd-19 bits per texel vs PSNR)
  docs/img/speed.svg       encode time per λ setting vs bc7enc_rdo (Oodle is closed source: no timing)
  README.md                the block between `<!-- results:start -->` and `<!-- results:end -->`

The SVGs follow the viewer's light/dark preference. BD-rate is the same computation as compete.py's
summary (scripts/bdrate.py), so the README always agrees with docs/BENCHMARKS.md once both are
refreshed from the same rows. Re-run this after any compete or ue_oodle/compare run; it needs no encoder.
"""

import argparse
import csv
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bdrate import bd_rate  # noqa: E402

FORMATS = ["bc1", "bc4", "bc5", "bc7"]
# What each format is for, shown under the panel title.
USE = {"bc1": "RGB color", "bc4": "one channel", "bc5": "normal maps (two channels)", "bc7": "high-quality RGB(A)"}
# Display names for competitor families (fixed order = fixed color slot).
NAMES = OrderedDict([
    ("bc7enc_rdo rgbcx", "bc7enc_rdo"),
    ("bc7enc_rdo bc7e", "bc7enc_rdo (bc7e)"),
    ("bc7enc_rdo bc7enc.cpp -e", "bc7enc_rdo (-e)"),
])
MIPPLE = "mipple RDO"

FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
STYLE = """
  .bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .mute { fill: #898781; }
  .grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; } .ring { stroke: #fcfcfb; }
  .s0 { stroke: #2a78d6; } .f0 { fill: #2a78d6; }
  .s1 { stroke: #eb6834; } .f1 { fill: #eb6834; }
  .s2 { stroke: #1baf7a; } .f2 { fill: #1baf7a; }
  .s3 { stroke: #8a5cd6; } .f3 { fill: #8a5cd6; }
  .good { fill: #006300; } .wh { stroke: #52514e; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; } .ring { stroke: #1a1a19; }
    .s0 { stroke: #3987e5; } .f0 { fill: #3987e5; }
    .s1 { stroke: #d95926; } .f1 { fill: #d95926; }
    .s2 { stroke: #199e70; } .f2 { fill: #199e70; }
    .s3 { stroke: #9d78ea; } .f3 { fill: #9d78ea; }
    .good { fill: #0ca30c; } .wh { stroke: #c3c2b7; }
  }
  text { font-family: FONT; }
  .num { font-variant-numeric: tabular-nums; }
""".replace("FONT", FONT)


def load(path):
    """Per format: {family: [(label, x, psnr, zstd_bpt, deflate_bpt, secs)]} for complete RDO configs."""
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    images = {r["image"] for r in rows}
    out = {}
    for fmt in FORMATS:
        agg = OrderedDict()
        for r in rows:
            if r["format"] != fmt or r["x"] in ("", "None"):
                continue
            a = agg.setdefault(r["label"], {"family": r["family"], "x": float(r["x"]), "n": 0, "psnr": 0.0,
                                            "zstd": 0, "deflate": 0, "texels": 0, "secs": 0.0})
            a["n"] += 1
            a["psnr"] += float(r["psnr"])
            a["zstd"] += int(r["zstd_bytes"])
            a["deflate"] += int(r["deflate_bytes"])
            a["texels"] += int(r["width"]) * int(r["height"])
            a["secs"] += float(r["secs"])
        fams = defaultdict(list)
        for label, a in agg.items():
            if a["n"] != len(images):
                continue
            fams[a["family"]].append((label, a["x"], a["psnr"] / a["n"], a["zstd"] * 8 / a["texels"],
                                      a["deflate"] * 8 / a["texels"], a["secs"]))
        if MIPPLE in fams and len(fams) > 1:
            out[fmt] = {k: sorted(v, key=lambda p: p[3]) for k, v in fams.items()}
    return out, len(images), rows[0].get("threads", "?") if rows else "?"


def comparisons(data):
    """One entry per (format, competitor family): BD-rates and time ranges."""
    res = []
    for fmt, fams in data.items():
        m = fams[MIPPLE]
        for fam in NAMES:
            if fam not in fams:
                continue
            c = fams[fam]
            z = bd_rate([(p[3], p[2]) for p in c], [(p[3], p[2]) for p in m])
            d = bd_rate([(p[4], p[2]) for p in c], [(p[4], p[2]) for p in m])
            res.append({
                "fmt": fmt, "fam": fam, "name": NAMES[fam], "zstd": z, "deflate": d,
                # RDO settings only (λ > 0): the λ = 0 point is a plain base encode.
                "mt": [p[5] for p in m if p[1] > 0], "ct": [p[5] for p in c if p[1] > 0],
            })
    return res


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    t0 = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = t0
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks, step


def fmt_tick(v, step):
    dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    if step * 10 ** dec % 1 > 1e-6:
        dec += 1
    return f"{v:.{dec}f}"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_doc(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n'
            f'<style>{STYLE}</style>\n<rect class="bg" width="{w}" height="{h}" rx="12"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def legend(x, y, slots, body, rename=None):
    for fam, k in slots.items():
        name = (rename or {}).get(fam) or ("mipple" if fam == MIPPLE else NAMES[fam])
        body.append(f'<line class="s{k}" x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke-width="2" stroke-linecap="round"/>'
                    f'<circle class="f{k} ring" cx="{x + 9}" cy="{y}" r="4" stroke-width="2"/>')
        body.append(f'<text class="ink2" x="{x + 26}" y="{y + 4.5}" font-size="13">{esc(name)}</text>')
        x += 26 + 7.4 * len(name) + 22
    return x


def strongest(comps, fmt):
    """The competitor mode mipple beats by the least (the fairest single curve to plot)."""
    c = [c for c in comps if c["fmt"] == fmt and c["zstd"] is not None]
    return max(c, key=lambda c: c["zstd"]) if c else None


def rd_curves(data, comps, nimg, threads):
    W, PW, PH = 960, 450, 320
    cols = 2
    top = 96
    rows_n = math.ceil(len(data) / cols)
    H = top + rows_n * (PH + 20) + 30
    body = [f'<text class="ink" x="24" y="38" font-size="20" font-weight="600">Rate-distortion: mipple vs bc7enc_rdo</text>',
            f'<text class="ink2" x="24" y="60" font-size="13">Kodak, {nimg} images · Zstd-19 bits per texel (lower is better) vs PSNR (higher is better) · '
            f'each point is one λ setting</text>']
    legend(24, 82, {MIPPLE: 0, "bc7enc_rdo rgbcx": 1}, body, {"bc7enc_rdo rgbcx": "bc7enc_rdo (its strongest mode per format)"})
    for idx, (fmt, fams) in enumerate(data.items()):
        c = strongest(comps, fmt)
        if c is None:
            continue
        ox = 16 + (idx % cols) * (PW + 28)
        oy = top + (idx // cols) * (PH + 20)
        L, R, T, B = ox + 48, ox + PW - 12, oy + 66, oy + PH - 34
        shown = [(MIPPLE, 0, "mipple"), (c["fam"], 1, c["name"])]
        pts = [p for fam, _, _ in shown for p in fams[fam]]
        xs, ys = [p[3] for p in pts], [p[2] for p in pts]
        pad = (max(ys) - min(ys)) * 0.04
        xt, xstep = nice_ticks(min(xs), max(xs))
        yt, ystep = nice_ticks(min(ys) - pad, max(ys) + pad)
        x0, x1 = min(xt[0], min(xs)), max(xt[-1], max(xs))
        y0, y1 = min(yt[0], min(ys) - pad), max(yt[-1], max(ys) + pad)
        sx = lambda v: L + (v - x0) / (x1 - x0) * (R - L)
        sy = lambda v: B - (v - y0) / (y1 - y0) * (B - T)
        body.append(f'<text class="ink" x="{ox + 8}" y="{oy + 20}" font-size="15" font-weight="600">{fmt.upper()}'
                    f'<tspan class="ink2" font-weight="400" font-size="13">  {esc(USE[fmt])}</tspan></text>')
        body.append(f'<text class="good num" x="{ox + 8}" y="{oy + 40}" font-size="13" font-weight="600">'
                    f'{-c["zstd"]:.1f}% fewer bits than {esc(c["name"])} at equal PSNR</text>')
        for t in yt:
            body.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke-width="1"/>')
            body.append(f'<text class="mute num" x="{L - 6}" y="{sy(t) + 4:.1f}" font-size="11" text-anchor="end">{fmt_tick(t, ystep)}</text>')
        for t in xt:
            body.append(f'<text class="mute num" x="{sx(t):.1f}" y="{B + 16}" font-size="11" text-anchor="middle">{fmt_tick(t, xstep)}</text>')
        body.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke-width="1"/>')
        body.append(f'<text class="mute" x="{R}" y="{B + 30}" font-size="11" text-anchor="end">bits/texel →</text>')
        body.append(f'<text class="mute" x="{L - 6}" y="{T - 12}" font-size="11" text-anchor="end">dB</text>')
        # Competitor first, mipple on top.
        for fam, k, _ in reversed(shown):
            f = fams[fam]
            d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p[3]):.1f},{sy(p[2]):.1f}" for i, p in enumerate(f))
            body.append(f'<path class="s{k}" d="{d}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for p in f:
                body.append(f'<circle class="f{k} ring" cx="{sx(p[3]):.1f}" cy="{sy(p[2]):.1f}" r="4" stroke-width="2">'
                            f'<title>{esc(p[0])}: {p[2]:.2f} dB, {p[3]:.3f} bits/texel</title></circle>')
        # Direct labels just right of each curve's lowest-rate point, stacked so they do not collide.
        labs = sorted([sy(fams[fam][0][2]), sx(fams[fam][0][3]), name] for fam, _, name in shown)
        for i in range(1, len(labs)):
            labs[i][0] = max(labs[i][0], labs[i - 1][0] + 15)
        for y, x, name in labs:
            body.append(f'<text class="ink2" x="{x + 10:.1f}" y="{min(y, B - 6) + 4:.1f}" font-size="12">{esc(name)}</text>')
    body.append(f'<text class="mute" x="24" y="{H - 14}" font-size="11">Generated by scripts/readme_charts.py from '
                f'bench-results/compete-kodak24-rdo/rows.csv ({threads} threads per encoder). Details: docs/BENCHMARKS.md</text>')
    return svg_doc(W, H, body, "Rate-distortion curves: mipple vs bc7enc_rdo",
                   "Per format, Zstd-19 bits per texel against PSNR for mipple's λ sweep and bc7enc_rdo's strongest mode on the Kodak images.")


# Oodle λ shown (Unreal's Lossy Compression Amount Low to Highest): the RDO range, where every image's Oodle
# quality lies on mipple's λ sweep. λ 0-1 are in docs/BENCHMARKS.md.
OODLE_LAMBDAS = [10, 20, 30, 40]
OODLE_NAME = "Oodle Texture 2.9.16"


def load_oodle(path):
    """Oodle 2.9.16 via Unreal Engine 5.8 on Kodak, per format: [(λ, Oodle bpt, mipple bpt at Oodle's RMSE,
    mipple bpt at Oodle's SSIM)], Kodak means (mipple's from total bits at matched quality over Oodle's total,
    as in scripts/ue_oodle/compare.py's summary)."""
    if not path.exists():
        return []
    import json
    res = [r for r in json.load(open(path, encoding="utf-8"))["results"]
           if r["image"].startswith("kodim") and r["image"] != "kodim18_original"]
    out = []
    for fmt in FORMATS:
        rs = [r for r in res if r["format"] == fmt]
        if not rs:
            continue
        pts = []
        for lam in OODLE_LAMBDAS:
            os_ = [o for r in rs for o in r["oodle"] if o["lambda"] == lam]
            if len(os_) != len(rs):
                continue
            ob = sum(o["bpt"] for o in os_) / len(os_)
            m = []
            for key in ("rmse", "ssim"):
                pairs = [(o[f"mipple_bpt_at_{key}"], o["bpt"]) for o in os_ if o[f"mipple_bpt_at_{key}"]]
                m.append(ob * sum(a for a, _ in pairs) / sum(b for _, b in pairs) if len(pairs) == len(os_) else None)
            if None not in m:
                pts.append((lam, ob, m[0], m[1]))
        if pts:
            out.append({"fmt": fmt, "n": len(rs), "pts": pts})
    return out


def load_formats(path):
    """The BC3, BC6H, ETC2 and ASTC tables of docs/BENCHMARKS.md (bench-results/formats.json)."""
    if not path.exists():
        return {"speed": [], "rd": []}
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def size_at(curve, q, col=2):
    """mipple's bits/texel at quality q, interpolated in log-rate between consecutive λ points that bracket
    it (the smallest, if PSNR is not monotonic in λ: light RDO can raise it)."""
    sizes = []
    for a, b in zip(curve, curve[1:]):
        if min(a[1], b[1]) <= q <= max(a[1], b[1]) and a[1] != b[1]:
            t = (q - a[1]) / (b[1] - a[1])
            sizes.append(math.exp(math.log(a[col]) + t * (math.log(b[col]) - math.log(a[col]))))
    return min(sizes, default=None)


# Competitors of the formats.json rows, after bc7enc_rdo (slot 1) and Oodle (slot 2).
OTHER_SLOT = {"bc7enc_rdo": 1, "Compressonator": 3, "ISPC fast": 3}


def extra_bits(fdata):
    """% fewer bits than the competitor for formats.json's RD rows: BD-rate against a curve, or the size
    at a single competitor point's PSNR."""
    rows = []
    for r in fdata["rd"]:
        m = r["mipple"]
        dv = None
        if r["kind"] == "curve":
            v = -bd_rate([(p[2], p[1]) for p in r["other"]], [(p[2], p[1]) for p in m])
            tip = f'BD-rate on {r["corpus"]} (PSNR), Zstd-19'
            if all(len(p) > 3 for p in m + r["other"]):
                dv = -bd_rate([(p[3], p[1]) for p in r["other"]], [(p[3], p[1]) for p in m])
                tip += f"; Deflate-9 {-dv:.1f}%"
        else:
            q, b = r["other"][0][1], r["other"][0][2]
            at = size_at(m, q)
            if at is None:
                continue
            v = (1 - at / b) * 100
            tip = f'{r["corpus"]}, at {r["who"]}’s {q:.2f} dB (interpolated between λ points), Zstd-19'
            if all(len(p) > 3 for p in m + r["other"]):
                dat = size_at(m, q, 3)
                if dat is not None:
                    dv = (1 - dat / r["other"][0][3]) * 100
                    tip += f"; Deflate-9 {-dv:.1f}%"
        rows.append({"fmt": r["fmt"], "who": r["who"], "k": OTHER_SLOT.get(r["who"], 1), "v": v, "dv": dv,
                     "lo": None, "hi": None, "tip": tip})
    return rows


# Group order in the bits chart.
ORDER = ["BC1", "BC3", "BC4", "BC5", "BC7", "ETC2 RGB", "ETC2 RGBA", "ASTC 4x4", "ASTC 6x6", "ASTC 8x8"]


def bits_rows(comps, oodle, extra=()):
    """One row per competitor: % fewer bits than it at equal quality (positive = mipple smaller)."""
    rows = []
    for fmt in FORMATS:
        for c in comps:
            if c["fmt"] == fmt and c["zstd"] is not None:
                rows.append({"fmt": fmt, "who": c["name"], "k": 1, "v": -c["zstd"], "lo": None, "hi": None,
                             "tip": f'BD-rate on Kodak (PSNR), Zstd-19; Deflate-9 {-c["deflate"]:.1f}%'})
        for o in oodle:
            if o["fmt"] == fmt:
                for k, key in ((2, "RMSE"), (3, "SSIM")):
                    sv = [(1 - p[k] / p[1]) * 100 for p in o["pts"]]
                    rows.append({"fmt": fmt, "who": f"{OODLE_NAME} · equal {key}", "k": 2, "v": sum(sv) / len(sv),
                                 "lo": min(sv), "hi": max(sv),
                                 "tip": f"Kodak ({o['n']} images), mean over Oodle λ "
                                        + ", ".join(f"{p[0]:g}" for p in o["pts"]) + f" at Oodle's {key}, both Zstd-19"})
    for r in rows:
        r["fmt"] = r["fmt"].upper()
    rows += extra
    rank = {f: i for i, f in enumerate(ORDER)}
    return sorted(rows, key=lambda r: rank.get(r["fmt"], len(ORDER)))  # stable: keeps each format's row order


def bits_chart(rows, nimg):
    W, top, rh = 960, 128, 32
    groups = list(dict.fromkeys(r["fmt"] for r in rows))
    H = top + len(rows) * rh + (len(groups) - 1) * 12 + 50
    LX, LW = 330, 470
    lo = min(0.0, min((r["lo"] if r["lo"] is not None else r["v"]) for r in rows))
    hi = max((r["hi"] if r["hi"] is not None else r["v"]) for r in rows)
    xt, xstep = nice_ticks(lo, hi, 6)
    x0, x1 = min(xt[0], lo), max(xt[-1], hi)
    sx = lambda v: LX + (v - x0) / (x1 - x0) * LW
    body = ['<text class="ink" x="24" y="38" font-size="20" font-weight="600">Fewer bits than each competitor, at equal quality</text>',
            '<text class="ink2" x="24" y="60" font-size="13">Bars right of zero: mipple\'s files are that much smaller at the same error. '
            'Both encoders\' output compressed with Zstd-19.</text>',
            f'<text class="ink2" x="24" y="78" font-size="13">bc7enc_rdo: BD-rate over each λ sweep (PSNR), Kodak ({nimg} images; BC3: 12 RGBA). '
            'Oodle Texture 2.9.16: Unreal 5.8 cook, mean over λ 10-40.</text>']
    slots = {MIPPLE: 1, "o": 2}
    names = {MIPPLE: "vs bc7enc_rdo", "o": "vs Oodle Texture (whisker: range over λ)"}
    if any(r["k"] == 3 for r in rows):
        slots["c"], names["c"] = 3, "vs an encoder without RDO, at its PSNR"
    legend(24, 102, slots, body, names)
    yb_all = top + len(rows) * rh + (len(groups) - 1) * 12
    for t in xt:
        body.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 6}" y2="{yb_all}" stroke-width="1"/>')
        body.append(f'<text class="mute num" x="{sx(t):.1f}" y="{yb_all + 16}" font-size="11" text-anchor="middle">{fmt_tick(t, xstep)}%</text>')
    y = top
    prev = None
    for r in rows:
        if prev is not None and r["fmt"] != prev:
            y += 12
        if r["fmt"] != prev:
            body.append(f'<text class="ink" x="24" y="{y + rh / 2 + 5:.1f}" font-size="15" font-weight="600">{esc(r["fmt"])}</text>')
        prev = r["fmt"]
        body.append(f'<text class="ink2" x="{LX - 12}" y="{y + rh / 2 + 4:.1f}" font-size="13" text-anchor="end">{esc(r["who"])}</text>')
        bh = 14
        by = y + (rh - bh) / 2
        a, b = sorted((sx(0), sx(r["v"])))
        w = max(b - a, 2)
        body.append(f'<rect class="f{r["k"]}" x="{a:.1f}" y="{by:.1f}" width="{w:.1f}" height="{bh}" rx="3">'
                    f'<title>{esc(r["who"])}: {r["v"]:.1f}% fewer bits ({esc(r["tip"])})</title></rect>')
        end = b
        if r["lo"] is not None:
            ya = by + bh / 2
            body.append(f'<path class="wh" d="M{sx(r["lo"]):.1f},{ya - 4:.1f} v8 M{sx(r["lo"]):.1f},{ya:.1f} H{sx(r["hi"]):.1f} '
                        f'M{sx(r["hi"]):.1f},{ya - 4:.1f} v8" fill="none" stroke-width="1.5"/>')
            end = max(end, sx(r["hi"]))
        # Values always sit right of the bar and whisker, so negative bars never run into the row labels.
        rng = f'  {r["lo"]:.1f} to {r["hi"]:.1f}' if r["lo"] is not None else ""
        body.append(f'<text class="ink num" x="{end + 6:.1f}" y="{by + 11.5:.1f}" font-size="12"><tspan font-weight="600">{r["v"]:.1f}%</tspan>'
                    f'<tspan class="mute">{rng}</tspan></text>')
        y += rh
    body.append(f'<line class="axis" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top - 6}" y2="{yb_all}" stroke-width="1.5"/>')
    body.append(f'<text class="mute" x="24" y="{H - 14}" font-size="11">Generated by scripts/readme_charts.py · '
                'Oodle rows from bench-results/oodle-ue, BC3, ETC2 and ASTC from bench-results/formats.json · details in docs/BENCHMARKS.md</text>')
    return svg_doc(W, H, body, "Fewer bits than each competitor at equal quality",
                   "Per format and competitor, how much smaller mipple's Zstd-19 compressed output is at equal error.")


def oodle_chart(oodle):
    W, PW, PH, cols = 960, 450, 270, 2
    top = 100
    rows_n = math.ceil(len(oodle) / cols)
    H = top + rows_n * (PH + 16) + 30
    n = oodle[0]["n"]
    body = [f'<text class="ink" x="24" y="38" font-size="20" font-weight="600">Bits per texel vs {OODLE_NAME}, same back end</text>',
            f'<text class="ink2" x="24" y="60" font-size="13">Kodak ({n} images). Oodle\'s blocks from a stock Unreal Engine 5.8 cook; '
            'mipple at Oodle\'s RMSE and SSIM; all Zstd-19. Lower is better.</text>']
    legend(24, 82, {"r": 0, "s": 1, "o": 2}, body,
           {"r": "mipple at Oodle's RMSE", "s": "mipple at Oodle's SSIM", "o": OODLE_NAME})
    for idx, o in enumerate(oodle):
        ox = 16 + (idx % cols) * (PW + 28)
        oy = top + (idx // cols) * (PH + 16)
        L, R, T, B = ox + 40, ox + PW - 8, oy + 74, oy + PH - 30
        ymax = max(max(p[1:]) for p in o["pts"])
        yt, ystep = nice_ticks(0, ymax * 1.05, 5)
        y1 = max(yt[-1], ymax * 1.05)
        sy = lambda v: B - v / y1 * (B - T)
        body.append(f'<text class="ink" x="{ox + 8}" y="{oy + 20}" font-size="15" font-weight="600">{o["fmt"].upper()}'
                    f'<tspan class="ink2" font-weight="400" font-size="13">  {USE[o["fmt"]]}</tspan></text>')
        for tk in yt:
            body.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{sy(tk):.1f}" y2="{sy(tk):.1f}" stroke-width="1"/>')
            body.append(f'<text class="mute num" x="{L - 6}" y="{sy(tk) + 4:.1f}" font-size="11" text-anchor="end">{fmt_tick(tk, ystep)}</text>')
        body.append(f'<text class="mute" x="{L - 6}" y="{T - 8}" font-size="11" text-anchor="start">bits/texel</text>')
        gw = (R - L) / len(o["pts"])
        bw = min(24, gw * 0.24)
        for i, (lam, ob, mr, ms) in enumerate(o["pts"]):
            cx = L + gw * (i + 0.5)
            for jj, (v, k, who) in enumerate(((mr, 0, "mipple at Oodle's RMSE"), (ms, 1, "mipple at Oodle's SSIM"),
                                              (ob, 2, OODLE_NAME))):
                x = cx - 1.5 * bw - 2 + jj * (bw + 2)
                body.append(f'<path class="f{k}" d="M{x:.1f},{B} V{sy(v) + 4:.1f} a4,4 0 0 1 4,-4 h{bw - 8:.1f} a4,4 0 0 1 4,4 V{B} z">'
                            f'<title>{who}, Oodle λ {lam:g}: {v:.3f} bits/texel</title></path>')
            for line, (key, v) in enumerate((("RMSE", mr), ("SSIM", ms))):
                d = (v / ob - 1) * 100
                cls = "good" if d < 0 else "ink2"
                body.append(f'<text class="{cls} num" x="{cx:.1f}" y="{oy + 40 + line * 15}" font-size="11" font-weight="600" '
                            f'text-anchor="middle">{key} {d:+.1f}%</text>')
            body.append(f'<text class="mute num" x="{cx:.1f}" y="{B + 16}" font-size="11" text-anchor="middle">λ {lam:g}</text>')
        body.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke-width="1"/>')
    body.append(f'<text class="mute" x="24" y="{H - 14}" font-size="11">Generated by scripts/readme_charts.py from '
                'bench-results/oodle-ue/results.json · % = mipple\'s size vs Oodle\'s at equal quality (negative = smaller)</text>')
    return svg_doc(W, H, body, f"Bits per texel vs {OODLE_NAME}",
                   "Per format and Oodle λ on Kodak, mipple's Zstd-19 bits per texel at Oodle's RMSE and at its SSIM "
                   "against Oodle's own blocks under Zstd-19.")


def speed_chart(comps, nimg, threads):
    rows = [c for c in comps if c["mt"] and c["ct"]]
    W, top, rh = 960, 104, 34
    H = top + len(rows) * rh + 54
    RX, RW = 240, 520
    tmax = max(max(sum(c["mt"]) / len(c["mt"]), sum(c["ct"]) / len(c["ct"])) for c in rows)
    tt, tstep = nice_ticks(0, tmax * 1.05, 6)
    tmx = max(tt[-1], tmax)
    body = ['<text class="ink" x="24" y="38" font-size="20" font-weight="600">Encode time vs bc7enc_rdo</text>',
            f'<text class="ink2" x="24" y="60" font-size="13">Mean seconds per λ setting over the RDO sweep, Kodak ({nimg} images, '
            f'{threads} threads each), lower is faster. Oodle Texture is closed source: no timing.</text>']
    legend(24, 84, {MIPPLE: 0, "bc7enc_rdo rgbcx": 1}, body)
    yb = top + len(rows) * rh
    for t in tt:
        x = RX + t / tmx * RW
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{yb}" stroke-width="1"/>')
        body.append(f'<text class="mute num" x="{x:.1f}" y="{yb + 16}" font-size="11" text-anchor="middle">{fmt_tick(t, tstep)} s</text>')
    for i, c in enumerate(rows):
        y = top + i * rh
        label = c["fmt"].upper() + ("" if c["name"] == "bc7enc_rdo" else f'  vs {c["name"].replace("bc7enc_rdo ", "").strip("()")}')
        body.append(f'<text class="ink" x="{RX - 12}" y="{y + rh / 2 + 4:.1f}" font-size="13" text-anchor="end">{esc(label)}</text>')
        mt, ct = sum(c["mt"]) / len(c["mt"]), sum(c["ct"]) / len(c["ct"])
        for j, (val, k, who, ts) in enumerate(((mt, 0, "mipple", c["mt"]), (ct, 1, c["name"], c["ct"]))):
            bh = 11
            by = y + rh / 2 - bh - 1 + j * (bh + 2)
            bw = max(val / tmx * RW, 2)
            body.append(f'<rect class="f{k}" x="{RX}" y="{by:.1f}" width="{bw:.1f}" height="{bh}" rx="3">'
                        f'<title>{esc(who)}: {val:.1f} s mean per λ ({min(ts):.1f}-{max(ts):.1f} s)</title></rect>')
            body.append(f'<text class="ink2 num" x="{RX + bw + 6:.1f}" y="{by + 9.5:.1f}" font-size="11">{val:.1f} s</text>')
        body.append(f'<text class="good num" x="{W - 24}" y="{y + rh / 2 + 4:.1f}" font-size="13" font-weight="600" text-anchor="end">{ct / mt:.2f}× faster</text>')
    body.append(f'<line class="axis" x1="{RX}" x2="{RX}" y1="{top - 6}" y2="{yb}" stroke-width="1"/>')
    body.append(f'<text class="mute" x="24" y="{H - 14}" font-size="11">Generated by scripts/readme_charts.py · details in docs/BENCHMARKS.md</text>')
    return svg_doc(W, H, body, "Encode time vs bc7enc_rdo",
                   "Per format, mean encode time per λ setting for mipple and bc7enc_rdo on the Kodak images.")


def log_ticks(lo, hi):
    ticks = [m * 10 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1) for m in (1, 2, 5)]
    return [t for t in ticks if lo <= t <= hi]


def formats_chart(fdata):
    """Quality against encode time, one panel per newer format: mipple's effort levels as a line, every
    competitor setting as a point. Up and to the left is better."""
    panels = fdata["speed"]
    W, cols, PW, PH, top = 960, 3, 296, 250, 100
    H = top + math.ceil(len(panels) / cols) * (PH + 14) + 30
    body = ['<text class="ink" x="24" y="38" font-size="20" font-weight="600">Quality vs encode time: BC6H, ETC2, ASTC</text>',
            '<text class="ink2" x="24" y="60" font-size="13">One thread for every encoder, no RDO. Up and to the left is better. '
            'Time is the total over each corpus, log scale.</text>']
    legend(24, 82, {"m": 0, "o": 1}, body, {"m": "mipple, one point per effort level", "o": "other encoders"})
    for idx, pnl in enumerate(panels):
        ox = 16 + (idx % cols) * (PW + 14)
        oy = top + (idx // cols) * (PH + 14)
        L, R, T, B = ox + 44, ox + PW - 10, oy + 52, oy + PH - 30
        pts = pnl["mipple"] + pnl["others"]
        ts, qs = [p[2] for p in pts], [p[1] for p in pts]
        tlo, thi = min(ts) / 1.6, max(ts) * 1.6
        pad = (max(qs) - min(qs)) * 0.12
        yt, ystep = nice_ticks(min(qs) - pad, max(qs) + pad, 5)
        y0, y1 = min(yt[0], min(qs) - pad), max(yt[-1], max(qs) + pad)
        sx = lambda v: L + (math.log(v) - math.log(tlo)) / (math.log(thi) - math.log(tlo)) * (R - L)
        sy = lambda v: B - (v - y0) / (y1 - y0) * (B - T)
        body.append(f'<text class="ink" x="{ox + 8}" y="{oy + 20}" font-size="15" font-weight="600">{esc(pnl["panel"])}'
                    f'<tspan class="ink2" font-weight="400" font-size="12">  {esc(pnl["use"])}</tspan></text>')
        for t in yt:
            body.append(f'<line class="grid" x1="{L}" x2="{R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke-width="1"/>')
            body.append(f'<text class="mute num" x="{L - 6}" y="{sy(t) + 4:.1f}" font-size="11" text-anchor="end">{fmt_tick(t, ystep)}</text>')
        for t in log_ticks(tlo, thi):
            body.append(f'<text class="mute num" x="{sx(t):.1f}" y="{B + 16}" font-size="11" text-anchor="middle">{t:g}</text>')
        body.append(f'<line class="axis" x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke-width="1"/>')
        body.append(f'<text class="mute" x="{R}" y="{T - 10}" font-size="11" text-anchor="end">{esc(pnl["metric"])} ↑ · seconds →</text>')
        m = sorted(pnl["mipple"], key=lambda p: p[2])
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p[2]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(m))
        body.append(f'<path class="s0" d="{d}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for name, q, t in pnl["others"]:
            body.append(f'<circle class="f1 ring" cx="{sx(t):.1f}" cy="{sy(q):.1f}" r="4.5" stroke-width="2">'
                        f'<title>{esc(name)}: {q:.3f} dB, {t:g} s</title></circle>')
        for name, q, t in m:
            body.append(f'<circle class="f0 ring" cx="{sx(t):.1f}" cy="{sy(q):.1f}" r="4.5" stroke-width="2">'
                        f'<title>mipple {esc(name)}: {q:.3f} dB, {t:g} s</title></circle>')
        # Direct labels: competitors right of their point (except the panel's "quiet" ones, which keep only
        # their tooltip), mipple's fastest and slowest effort above theirs.
        labs = [[sy(q) + 4, sx(t) + 8, n, "ink2"] for n, q, t in pnl["others"] if n not in pnl.get("quiet", ())]
        labs += [[sy(m[0][1]) - 9, sx(m[0][2]) - 4, "mipple " + m[0][0], "ink"]]
        if len(m) > 1:
            labs += [[sy(m[-1][1]) - 9, sx(m[-1][2]) - 4, m[-1][0], "ink"]]
        placed = []  # [y, left, right]
        for y, x, n, cls in sorted(labs):
            # A label that would run past the panel's right edge ends at its point instead (below it, for a competitor).
            w, anchor = 5.6 * len(n), ""
            if x + w > R:
                x, y, anchor = x - (4 if cls == "ink2" else -8), y + (14 if cls == "ink2" else 0), ' text-anchor="end"'
            lx = x - w if anchor else x
            # Push down past any placed label it would overlap.
            for py, pl, pr in sorted(placed):
                if lx < pr + 4 and pl < lx + w + 4 and abs(y - py) < 13:
                    y = py + 13
            placed.append([y, lx, lx + w])
            body.append(f'<text class="{cls}" x="{x:.1f}" y="{min(y, B - 4):.1f}" font-size="11"{anchor}>{esc(n)}</text>')
    body.append(f'<text class="mute" x="24" y="{H - 14}" font-size="11">Generated by scripts/readme_charts.py from '
                'bench-results/formats.json (the docs/BENCHMARKS.md tables, which give each command)</text>')
    return svg_doc(W, H, body, "Quality vs encode time: BC6H, ETC2, ASTC",
                   "Per format, quality against single-thread encode time for mipple's effort levels and each competitor setting.")


def readme_block(comps, oodle, nimg, threads, extra=()):
    lines = ["<!-- results:start -->",
             "<!-- Generated by scripts/readme_charts.py; edit that script, not this block. -->",
             "",
             '<img src="docs/img/bits.svg" alt="Fewer bits than each competitor at equal quality, per format" width="100%">',
             "",
             '<img src="docs/img/oodle.svg" alt="Bits per texel vs Oodle Texture 2.9.16 on Kodak, per Oodle lambda" width="100%">',
             "",
             '<img src="docs/img/formats.svg" alt="Quality vs single-thread encode time for BC6H, ETC2 and ASTC: mipple vs ISPC, texconv and Compressonator" width="100%">',
             "",
             '<img src="docs/img/rd-curves.svg" alt="Rate-distortion curves per format: mipple vs bc7enc_rdo" width="100%">',
             "",
             '<img src="docs/img/speed.svg" alt="Encode time per lambda setting: mipple vs bc7enc_rdo" width="100%">',
             "",
             f"**bc7enc_rdo:** RDO on all {nimg} Kodak images, {threads} threads per encoder, graded identically with "
             "`mipple score`. BD-rate is the size difference at equal PSNR (negative = mipple is smaller); times are "
             "the range over each encoder's λ sweep. **Oodle Texture 2.9.16** is closed source, so it is run the way a "
             "game team runs it: a stock Unreal Engine 5.8 install imports the same Kodak images and cooks them (RDO on, "
             "Lossy Compression Amount Low to Highest = Oodle λ 10-40), and Oodle's blocks are taken from the cook and "
             "compressed with Zstd-19 like mipple's. mipple's size is measured at Oodle's RMSE and at its SSIM (positive "
             "= mipple is larger); there is no timing, since Unreal builds textures in parallel. Every caveat is in "
             "[docs/BENCHMARKS.md](docs/BENCHMARKS.md).",
             "",
             "| Format | vs | Zstd-19 | Deflate-9 | mipple time (s) | competitor time (s) |",
             "| --- | --- | --- | --- | --- | --- |"]
    # Bold marks mipple's wins (smaller files); a loss is shown plain.
    f = lambda v: "n/a" if v is None else (f"**{v:+.1f}%**" if v < 0 else f"{v:+.1f}%").replace("-", "−")
    for fmt in FORMATS:
        for c in comps:
            if c["fmt"] == fmt:
                lines.append(f'| {fmt.upper()} | {c["name"]} | {f(c["zstd"])} | {f(c["deflate"])} | '
                             f'{min(c["mt"]):.1f}-{max(c["mt"]):.1f} | {min(c["ct"]):.1f}-{max(c["ct"]):.1f} |')
        for o in oodle:
            if o["fmt"] == fmt:
                lam = f'λ {o["pts"][0][0]:g}-{o["pts"][-1][0]:g}'
                for k, key in ((2, "RMSE"), (3, "SSIM")):
                    d = sorted((p[k] / p[1] - 1) * 100 for p in o["pts"])
                    lines.append(f"| {fmt.upper()} | {OODLE_NAME} (Unreal 5.8), equal {key} | {f(d[0])} to {f(d[-1])} "
                                 f"({lam}) | n/a | n/a | n/a |")
    for r in extra:
        dv = r.get("dv")
        lines.append(f'| {r["fmt"]} | {r["who"]} | {f(-r["v"])} | {f(None if dv is None else -dv)} | n/a | n/a |')
    lines += ["", "Refresh after a new benchmark run with `python scripts/readme_charts.py` (reads "
              "`bench-results/compete-kodak24-rdo/rows.csv` and `bench-results/oodle-ue/results.json`, rewrites "
              "this block and `docs/img/*.svg`).",
              "", "<!-- results:end -->"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "bench-results/compete-kodak24-rdo/rows.csv")
    ap.add_argument("--oodle", type=Path, default=ROOT / "bench-results/oodle-ue/results.json")
    ap.add_argument("--formats", type=Path, default=ROOT / "bench-results/formats.json")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--img", type=Path, default=ROOT / "docs/img")
    a = ap.parse_args()
    data, nimg, threads = load(a.rows)
    comps = comparisons(data)
    oodle = load_oodle(a.oodle)
    fdata = load_formats(a.formats)
    extra = extra_bits(fdata)
    a.img.mkdir(parents=True, exist_ok=True)
    out = {"rd-curves.svg": rd_curves(data, comps, nimg, threads),
           "bits.svg": bits_chart(bits_rows(comps, oodle, extra), nimg),
           "speed.svg": speed_chart(comps, nimg, threads)}
    if oodle:
        out["oodle.svg"] = oodle_chart(oodle)
    if fdata["speed"]:
        out["formats.svg"] = formats_chart(fdata)
    for name, svg in out.items():
        (a.img / name).write_text(svg, encoding="utf-8", newline="\n")
    text = a.readme.read_text(encoding="utf-8")
    s, e = text.find("<!-- results:start -->"), text.find("<!-- results:end -->")
    if s < 0 or e < 0:
        sys.exit(f"{a.readme}: add <!-- results:start --> and <!-- results:end --> markers where the block goes")
    e += len("<!-- results:end -->")
    a.readme.write_text(text[:s] + readme_block(comps, oodle, nimg, threads, extra) + text[e:], encoding="utf-8", newline="\n")
    for r in bits_rows(comps, oodle, extra):
        rng = f'  ({r["lo"]:+.1f} to {r["hi"]:+.1f})' if r["lo"] is not None else ""
        print(f'{r["fmt"]} vs {r["who"]}: {r["v"]:.1f}% fewer bits{rng}')


if __name__ == "__main__":
    main()
