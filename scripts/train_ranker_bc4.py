#!/usr/bin/env python3
"""Fit the localized BC4 search ranker (src/ranker_bc4.rs) from `MIPPLE_LOCAL_DUMP` rows.

Training data: single channels of the CC0 Poly Haven textures in test-images/polyhaven
(`scripts/fetch_polyhaven.py`; never Kodak or RAD's images), since BC4 encodes channel 0:

    python scripts/train_ranker_bc4.py prep <polyhaven dir> <out dir>      # R/G/B of diffuse, R/G of normal maps
    python scripts/train_ranker_bc4.py dump <mipple exe> <out dir> train.bin  # BC4 encodes with the dump on
    python scripts/train_ranker_bc4.py train train.bin [--hidden 16] [--epochs 30]

Each group is one source block; its rows are every far block j (beyond --max-back) of the chunk,
with the RD-cost gain j's candidates would bring over the near search alone (`dump_rows` in
src/ranker_bc4.rs: u32 run, u32 group, f32 x 16 raw statistics, f32 gain, f32 near J). The
encoder narrows each block's far blocks to the M most similar by shape SAD, the MLP ranks those
and the top K get the full search, so "captured gain at K" after that prefilter is what matters.
`train` prints it for simple baselines and the fitted MLP on held-out encodes (every 5th
process), then the weights as the body of src/ranker_bc4.rs's `weights` module.
"""
import argparse
import pathlib
import subprocess
import sys
import numpy as np

NRAW = 16
ROW = np.dtype([("run", "<u4"), ("g", "<u4"), ("f", "<f4", NRAW), ("gain", "<f4"), ("near", "<f4")])
KS = (1, 2, 4, 8, 16, 24, 32, 64)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def prep(a):
    from PIL import Image

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(pathlib.Path(a.src).glob("*.png")):
        im = Image.open(f).convert("RGB")
        for k, c in enumerate("RGB" if f.stem.endswith("_diff") else "RG"):
            ch = im.getchannel(k)
            Image.merge("RGB", (ch, ch, ch)).save(out / f"{f.stem}_{c}.png")


def dump(a):
    """One process per image (the run id), two λ each, rotating over LAMBDAS."""
    imgs = sorted(pathlib.Path(a.dir).glob("*.png"))
    env = dict(**__import__("os").environ, MIPPLE_LOCAL_DUMP=a.data)
    for k, img in enumerate(imgs):
        lams = sorted({LAMBDAS[k % 5], LAMBDAS[(k + 2) % 5]})
        cmd = [str(pathlib.Path(a.exe).resolve()), "bench", str(img), "-f", "bc4", "-l", ",".join(map(str, lams)),
               "--local", "0", "--max-back", str(a.max_back)]
        print(k, img.name, lams, flush=True)
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)


def fl2(x):
    """log2(1 + x) by the float bit trick (exponent plus linear mantissa), as the encoder does."""
    v = (1.0 + np.asarray(x, np.float32)).astype(np.float32)
    return ((v.view(np.int32) - 0x3F800000) / np.float32(1 << 23)).astype(np.float32)


NPAIR = 10
# The encoder scales the pair features by multiplying with f32 reciprocals.
R10 = np.float32(1 / 10)
R12 = np.float32(1 / 12)


def features(r):
    """Must match `RankSrc1::features` and `ctx_features` (raw columns: see `RankSrc1::raw`)."""
    c = lambda k: r[:, k]
    cols = [
        np.log2(np.maximum(c(0), 1)) / 10,
        fl2(c(1)) * R10, fl2(c(2)) * R12, fl2(c(3)) * R12, fl2(c(4)) / 8,
        fl2(c(5)) / 16, fl2(c(6)) / 16, fl2(c(7)) / 16, fl2(np.minimum(c(5), c(6))) / 16,
        fl2(c(8)) / 8,
        # context
        fl2(c(11)) / 16, fl2(c(12)) / 8, fl2(c(13)) / 10,
    ]
    return np.stack(cols, 1).astype(np.float32)


def load(path):
    r = np.fromfile(path, dtype=ROW)
    key = r["run"].astype(np.int64) << 32 | r["g"].astype(np.int64)
    start = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(start) - 1
    runs = np.unique(r["run"], return_inverse=True)[1]
    return r["f"], r["gain"].astype(np.float32), gid, runs


def group_rank(gid, key):
    """Rank of each row within its group by ascending key (ties by row order, i.e. distance)."""
    order = np.lexsort((np.arange(len(key)), key, gid))
    gs = gid[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank = np.empty(len(key), np.int64)
    rank[order] = np.arange(len(gs)) - np.repeat(first, np.diff(np.r_[first, len(gs)]))
    return rank


def captured(score, gain, gid, ks=KS):
    """(captured gain, hit rate) at each k: the best gain among the top-k scores summed over groups,
    over the sum of best gains; and the share of groups with a gain whose best block is in the top k."""
    order = np.lexsort((-score, gid))
    gs, gn = gid[order], gain[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank = np.arange(len(gs)) - np.repeat(first, np.diff(np.r_[first, len(gs)]))
    best = np.maximum.reduceat(gn, first)
    tot = best.sum()
    pos = best > 0
    cap, hit = [], []
    for k in ks:
        m = np.maximum.reduceat(np.where(rank < k, gn, 0), first)
        cap.append(m.sum() / tot)
        hit.append((m[pos] >= best[pos]).mean())
    return cap, hit


class Mlp:
    def __init__(self, nin, hidden, rng):
        self.p = {
            "w1": rng.normal(0, 1 / np.sqrt(nin), (nin, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden).astype(np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        h = np.maximum(x @ self.p["w1"] + self.p["b1"], 0)
        return h @ self.p["w2"], h

    def step(self, x, ds, h, lr):
        dh = np.outer(ds, self.p["w2"]) * (h > 0)
        grads = {"w2": h.T @ ds, "w1": x.T @ dh, "b1": dh.sum(0)}
        self.t += 1
        for k, gr in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def listwise_grad(score, gain, gid, ngroups, power):
    """Softmax cross-entropy per group against targets ∝ gain^power, groups weighted by best gain."""
    mx = np.full(ngroups, -np.inf, np.float32)
    np.maximum.at(mx, gid, score)
    e = np.exp(score - mx[gid])
    z = np.bincount(gid, e, ngroups)
    p = e / z[gid]
    t = gain**power
    tz = np.bincount(gid, t, ngroups)
    best = np.zeros(ngroups, np.float32)
    np.maximum.at(best, gid, gain)
    wg = best / max(best.mean(), 1e-9)
    ok = tz[gid] > 0
    tt = np.where(ok, t / np.where(tz[gid] > 0, tz[gid], 1), 0)
    loss = -(wg[gid] * tt * np.log(p + 1e-12)).sum() / ngroups
    return (wg[gid] * np.where(ok, p - tt, 0)).astype(np.float32), loss


def emit(p, hidden, rs=None):
    """Prints the weights as Rust statics, and with `rs` also replaces them in that file."""
    fmt = lambda v: ", ".join(f"{x:.9e}" for x in np.ravel(v))
    lines = [f"    pub static RANK_W1: [[f32; {hidden}]; {p['w1'].shape[0]}] = ["]
    lines += [f"        [{fmt(row)}]," for row in p["w1"]]
    lines += ["    ];", f"    pub static RANK_B1: [f32; {hidden}] = [{fmt(p['b1'])}];",
              f"    pub static RANK_W2: [f32; {hidden}] = [{fmt(p['w2'])}];"]
    text = "\n".join(lines)
    print(text)
    if rs:
        src = pathlib.Path(rs).read_text(encoding="utf-8")
        a = src.index("    pub static RANK_W1")
        b = src.index("\n", src.index("    pub static RANK_W2"))
        pathlib.Path(rs).write_text(src[:a] + text + src[b:], encoding="utf-8", newline="\n")


def show(name, score, gain, gid):
    cap, hit = captured(score, gain, gid)
    print(f"{name:14s} cap", " ".join(f"{c:.3f}" for c in cap), "| hit", " ".join(f"{h:.3f}" for h in hit))


def train(a):
    raw, gain, gid, run = load(a.data)
    keep = raw[:, 0] > a.min_d
    raw, gain, gid, run = raw[keep], gain[keep], np.unique(gid[keep], return_inverse=True)[1], run[keep]
    nrun = run.max() + 1
    print(f"{len(raw)} rows, {gid.max() + 1} groups, {nrun} encodes; {(gain > 0).mean():.2%} of rows gain, "
          f"{(np.bincount(gid, gain > 0) > 0).mean():.1%} of groups; validation = every 5th encode")
    # Prefilter, as the encoder does: the M most similar by shape SAD, or (union) the M/2 best by
    # shape SAD plus the M/2 best by value SAD.
    srank = group_rank(gid, raw[:, 1])
    vrank = group_rank(gid, raw[:, 2])
    pres = {"shape": srank < a.m, "value": vrank < a.m,
            "union": (srank < a.m // 2) | (vrank < a.m // 2)}
    pre = pres[a.pre]
    val = (run % 5) == 4
    rng = np.random.default_rng(1)
    x = features(raw)
    vg = np.unique(gid[val], return_inverse=True)[1]
    neg = np.float32(-1e30)

    def evalv(name, s, prefilter=True, mask=None):
        s = s[val].astype(np.float32)
        if prefilter:
            s = np.where((pre if mask is None else mask)[val], s, neg)
        show(name, s, gain[val], vg)

    mlc = -np.minimum(raw[:, 5], raw[:, 6])
    for k, m in pres.items():
        evalv(f"all in {k}", np.zeros(len(raw)), mask=m)
        evalv(f"min(ls,cp) {k}", mlc, mask=m)
    evalv("min(ls,cp) all", mlc, False)

    print("K:", KS, f"(after shape-SAD top {a.m} unless noted)")
    evalv("shape SAD all", -raw[:, 1], False)
    evalv("random", rng.random(len(raw)))
    evalv("nearest d", -raw[:, 0])
    evalv("copy SSE", -raw[:, 6])
    evalv("refit resid", -raw[:, 5])
    evalv("min(ls,cp)", -np.minimum(raw[:, 5], raw[:, 6]))
    evalv("span SSE", -raw[:, 7])
    evalv("vert", -raw[:, 8])
    tr = ~val & (pre if a.train_pre else True)
    tg = np.unique(gid[tr], return_inverse=True)[1]
    xt, gt = x[tr], gain[tr]
    ng = tg.max() + 1
    gstart = np.r_[0, np.flatnonzero(tg[1:] != tg[:-1]) + 1, len(tg)]
    model = Mlp(x.shape[1], a.hidden, rng)
    for ep in range(a.epochs):
        perm = rng.permutation(ng)
        tot = 0.0
        lr = a.lr * (0.1 if ep >= a.epochs * 0.8 else 1.0)
        for b in range(0, ng, a.batch_groups):
            gs = perm[b:b + a.batch_groups]
            idx = np.concatenate([np.arange(gstart[g], gstart[g + 1]) for g in gs])
            lg = np.unique(tg[idx], return_inverse=True)[1]
            s, h = model.forward(xt[idx])
            ds, loss = listwise_grad(s, gt[idx], lg, len(gs), a.power)
            model.step(xt[idx], ds / len(gs), h, lr)
            tot += loss
        if ep % 5 == 4 or ep == a.epochs - 1:
            sv, _ = model.forward(x)
            print(f"ep {ep + 1:3d} loss {tot:.3f}", flush=True)
            evalv("mlp", sv)
    sv, _ = model.forward(x)
    evalv("mlp no prefilt", sv, False)
    np.savez(f"{a.data}.bc4h{a.hidden}.npz", **model.p)
    emit(model.p, a.hidden, a.emit_rs)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("src")
    p.add_argument("out")
    p = sub.add_parser("dump")
    p.add_argument("exe")
    p.add_argument("dir")
    p.add_argument("data")
    p.add_argument("--max-back", type=int, default=128)
    p = sub.add_parser("train")
    p.add_argument("data")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-groups", type=int, default=256)
    p.add_argument("--m", type=int, default=128, help="shape-SAD prefilter size")
    p.add_argument("--min-d", type=int, default=0, help="only rows farther back than this")
    p.add_argument("--train-pre", type=int, default=1, help="train only on prefiltered rows")
    p.add_argument("--pre", default="union", choices=["shape", "value", "union"])
    p.add_argument("--emit-rs", help="write the weights into this Rust file (src/ranker_bc4.rs)")
    a = ap.parse_args()
    {"prep": prep, "dump": dump, "train": train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
