#!/usr/bin/env python3
"""Fit the localized-search ranker from `MIPPLE_LOCAL_DUMP` rows (see `dump_rows` in src/rdo.rs).

Each group is one source block; its rows are every far block j the full search could reach, with
the RD-cost gain j's candidates would bring over the near search alone. The ranker must put the
high-gain blocks in its top K; it never decides the encoding (the search does, on the K it gets).

    python scripts/train_ranker.py train.bin [--hidden 16] [--epochs 30]

Prints the captured gain at several K for simple baselines and the fitted MLP on held-out encodes
(every 5th), then the weights as the body of src/ranker/bc5.rs's `weights` module. The encoder narrows
each block's candidates to 128 by shape SAD before the MLP ranks them, so captured gain at K = 16 is
the figure that matters.
"""
import argparse
import numpy as np

ROW = np.dtype([("g", "<u4"), ("f", "<u2", 18), ("gain", "<f4"), ("near", "<f4")])
KS = (1, 2, 4, 8, 16, 32, 64)


def load(path):
    r = np.fromfile(path, dtype=ROW)
    g = r["g"].astype(np.int64)
    # Groups are contiguous runs of one id; a process restart resets ids (a new image).
    start = np.r_[True, g[1:] != g[:-1]]
    gid = np.cumsum(start) - 1
    run = np.cumsum(np.r_[True, g[1:] < g[:-1]] & start) - 1
    return r["f"].astype(np.float32), r["gain"].astype(np.float32), gid, run


def fl2(x):
    """log2(1 + x) by the float bit trick (exponent plus linear mantissa), as the encoder does."""
    v = (1.0 + np.asarray(x, np.float32)).astype(np.float32)
    return ((v.view(np.int32) - 0x3F800000) / float(1 << 23)).astype(np.float32)


# Pair features (vary with the earlier block j), then context features (fixed for the source block;
# only the MLP's interactions use them; the encoder folds them into the first layer's bias).
NPAIR = 18


def features(f):
    d, sh, ab, mn, rg = f[:, 0], f[:, 1:3], f[:, 3:5], f[:, 5:7], f[:, 7:9]
    rs, ls, cp, bh, lam = f[:, 9:11], f[:, 11:13], f[:, 13:15], f[:, 15:17], f[:, 17]
    cols = [
        np.log2(d) / 10,
        fl2(sh[:, 0]) / 10, fl2(sh[:, 1]) / 10, fl2(sh[:, 0] + sh[:, 1]) / 10,
        fl2(ab[:, 0]) / 12, fl2(ab[:, 1]) / 12, fl2(np.minimum(ab[:, 0], ab[:, 1])) / 12,
        fl2(ab[:, 0] + ab[:, 1]) / 12,
        fl2(mn[:, 0]) / 12, fl2(mn[:, 1]) / 12,
        fl2(rg[:, 0]) / 8, fl2(rg[:, 1]) / 8,
        fl2(ls[:, 0]) / 16, fl2(ls[:, 1]) / 16, fl2(ls[:, 0] + ls[:, 1]) / 16,
        fl2(cp[:, 0]) / 16, fl2(cp[:, 1]) / 16, fl2(np.minimum(cp[:, 0], cp[:, 1])) / 16,
        fl2(bh[:, 0]) / 16, fl2(bh[:, 1]) / 16, fl2(rs[:, 0]) / 8, fl2(rs[:, 1]) / 8,
        fl2(lam) / 10,
    ]
    return np.stack(cols, 1).astype(np.float32)


def captured(score, gain, gid, ks=KS):
    """Sum over groups of the best gain among the top-k scores, over the sum of best gains."""
    order = np.lexsort((-score, gid))
    gs, gn = gid[order], gain[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank = np.arange(len(gs)) - np.repeat(first, np.diff(np.r_[first, len(gs)]))
    best = np.maximum.reduceat(gn, first)
    tot = best.sum()
    out = []
    for k in ks:
        m = np.where(rank < k, gn, 0)
        out.append(np.maximum.reduceat(m, first).sum() / tot)
    return out


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


def emit(p, hidden):
    fmt = lambda v: ", ".join(f"{x:.6e}" for x in np.ravel(v))
    print(f"    pub static RANK_W1: [[f32; {hidden}]; {p['w1'].shape[0]}] = [")
    for row in p["w1"]:
        print(f"        [{fmt(row)}],")
    print("    ];")
    print(f"    pub static RANK_B1: [f32; {hidden}] = [{fmt(p['b1'])}];")
    print(f"    pub static RANK_W2: [f32; {hidden}] = [{fmt(p['w2'])}];")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-groups", type=int, default=512)
    a = ap.parse_args()
    f, gain, gid, run = load(a.data)
    x = features(f)
    rng = np.random.default_rng(1)
    nrun = run.max() + 1
    val = (run % 5) == 4
    print(f"{len(f)} rows, {gid.max() + 1} groups, {nrun} encodes; "
          f"{(gain > 0).mean():.1%} of rows gain; validation = every 5th encode")
    vgid = np.unique(gid[val], return_inverse=True)[1]
    base = -(f[:, 1] + f[:, 2])
    print("K:", KS)
    print("shape SAD   ", " ".join(f"{c:.3f}" for c in captured(base[val], gain[val], vgid)))
    print("random      ", " ".join(f"{c:.3f}" for c in captured(rng.random(val.sum()), gain[val], vgid)))
    ls = f[:, 11:13]
    print("refit resid ", " ".join(f"{c:.3f}" for c in captured(-ls.sum(1)[val], gain[val], vgid)))
    print("resid-base  ", " ".join(f"{c:.3f}" for c in captured(
        (-np.minimum(ls[:, 0] - 4 * f[:, 15], ls[:, 1] - 4 * f[:, 16]))[val], gain[val], vgid)))
    print("nearest d   "," ".join(f"{c:.3f}" for c in captured(-f[val, 0], gain[val], vgid)))
    tr = ~val
    tg = np.unique(gid[tr], return_inverse=True)[1]
    xt, gt = x[tr], gain[tr]
    ng = tg.max() + 1
    gstart = np.r_[0, np.flatnonzero(tg[1:] != tg[:-1]) + 1, len(tg)]
    for hidden in (a.hidden,):
        model = Mlp(x.shape[1], hidden, rng)
        xx = xt
        for ep in range(a.epochs):
            perm = rng.permutation(ng)
            tot = 0.0
            for b in range(0, ng, a.batch_groups):
                gs = perm[b:b + a.batch_groups]
                idx = np.concatenate([np.arange(gstart[g], gstart[g + 1]) for g in gs])
                lg = np.unique(tg[idx], return_inverse=True)[1]
                s, h = model.forward(xx[idx])
                ds, loss = listwise_grad(s, gt[idx], lg, len(gs), a.power)
                model.step(xx[idx], ds / len(gs), h, a.lr)
                tot += loss
            if ep % 10 == 9 or ep == a.epochs - 1:
                sv, _ = model.forward(x[val])
                print(f"hidden {hidden:2d} ep {ep + 1:3d} loss {tot:.3f} ",
                      " ".join(f"{c:.3f}" for c in captured(sv, gain[val], vgid)))
        np.savez(f"{a.data}.h{hidden}.npz", **model.p)
        emit(model.p, hidden)


if __name__ == "__main__":
    main()
