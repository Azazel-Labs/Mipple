#!/usr/bin/env python3
"""Fit BC7's localized-search ranker (src/ranker_bc7.rs) from `MIPPLE_LOCAL_DUMP` rows.

Dump one file per image (group ids are only unique within a process), with the ranker off and
the exhaustive near search 64 deep (as for the shipped weights; the encoder searches 32), e.g.

    MIPPLE_LOCAL_DUMP=dump/<image>.bin mipple bench <image> -f bc7 -l 0.5,1,2,4 --max-back 64 --local 0

About one block in 256 then searches every earlier block of its chunk after the near and far
searches, and writes a row for each block within the best 64 by either prefilter key (value SAD,
refit residual; `RankSrc7::dump_row`: group, d, the ranker's features as the encoder computes
them, the RD-cost gain the block's full search brings, the better of its two prefilter ranks),
then a d = 0 row with the best gain over all blocks and their count.

    python scripts/train_ranker_bc7.py dump/*.bin [--hidden 16] [--epochs 40]

Train on Poly Haven textures only (scripts/fetch_polyhaven.py); never on the Kodak or Oodle
benchmark images. Prints the captured gain (best gain in the top K over the best gain of the
whole chunk) for the baselines and the MLP on held-out images (every 5th file), then the body of
the `weights` module.
"""
import argparse
import numpy as np

NIN, NPAIR = 19, 14
ROW = np.dtype([("g", "<u4"), ("d", "<u4"), ("f", "<f4", NIN), ("gain", "<f4"), ("pre", "<f4")])
KS = (1, 4, 8, 12, 16, 24, 32, 64)


def load(paths, m):
    xs, gains, gids, files, totals, pres, ds = [], [], [], [], [], [], []
    base = 0
    for fi, p in enumerate(paths):
        r = np.fromfile(p, dtype=ROW)
        if len(r) == 0:
            continue
        g = r["g"].astype(np.int64)
        summ = r["d"] == 0
        ug, inv = np.unique(g, return_inverse=True)
        tot = np.zeros(len(ug), np.float32)
        tot[inv[summ]] = r["gain"][summ]
        rows = ~summ
        gi = inv[rows]
        # The encoder's ranker sees the blocks within m of the best by either prefilter key.
        keep = r["pre"][rows] < m
        xs.append(r["f"][rows][keep])
        gains.append(r["gain"][rows][keep])
        pres.append(r["pre"][rows][keep])
        ds.append(r["d"][rows][keep])
        gids.append(gi[keep] + base)
        files.append(np.full(len(ug), fi))
        totals.append(tot)
        base += len(ug)
    gid = np.concatenate(gids)
    o = np.argsort(gid, kind="stable")
    return (np.concatenate(xs)[o], np.concatenate(gains)[o], gid[o], np.concatenate(files),
            np.concatenate(totals), np.concatenate(pres)[o], np.concatenate(ds)[o])


def captured(score, gain, gid, tot, ks=KS):
    """Over groups: Σ best gain among the top-k scores / Σ best gain over the whole chunk."""
    order = np.lexsort((-score, gid))
    gs, gn = gid[order], gain[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank = np.arange(len(gs)) - np.repeat(first, np.diff(np.r_[first, len(gs)]))
    t = tot[np.unique(gs)].sum()
    return [np.maximum.reduceat(np.where(rank < k, gn, 0), first).sum() / t for k in ks]


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
    p = e / np.bincount(gid, e, ngroups)[gid]
    t = gain**power
    tz = np.bincount(gid, t, ngroups)
    best = np.zeros(ngroups, np.float32)
    np.maximum.at(best, gid, gain)
    wg = best / max(best.mean(), 1e-9)
    ok = tz[gid] > 0
    tt = np.where(ok, t / np.where(ok, tz[gid], 1), 0)
    loss = -(wg[gid] * tt * np.log(p + 1e-12)).sum() / ngroups
    return (wg[gid] * np.where(ok, p - tt, 0)).astype(np.float32), loss


def emit(p, hidden):
    fmt = lambda v: ", ".join(f"{x:.6e}" for x in np.ravel(v))
    print(f"    pub static W1: [[f32; HIDDEN]; NIN] = [")
    for row in p["w1"]:
        print(f"        [{fmt(row)}],")
    print("    ];")
    print(f"    pub static B1: [f32; HIDDEN] = [{fmt(p['b1'])}];")
    print(f"    pub static W2: [f32; HIDDEN] = [{fmt(p['w2'])}];")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-groups", type=int, default=256)
    ap.add_argument("--m", type=int, default=32, help="prefilter survivors per key (LOCAL7_M)")
    ap.add_argument("--all", action="store_true", help="fit on every file (after validating)")
    a = ap.parse_args()
    x, gain, gid, gfile, tot, pre, d = load(sorted(a.data), a.m)
    gain = np.maximum(gain, 0)
    val_g = (gfile % 5) == 4
    val = val_g[gid]
    print(f"{len(a.data)} files, {len(x)} rows, {len(tot)} groups ({(tot > 0).mean():.1%} with a gain "
          f"beyond the near and far searches); validation = every 5th file")
    rng = np.random.default_rng(1)
    vt = tot
    print("captured gain on validation, K:", KS)
    show = lambda name, s: print(f"{name:14s}", " ".join(f"{c:.3f}" for c in captured(s[val], gain[val], gid[val], vt)))
    show("prefilter rank", -pre)
    show("nearest d", -d.astype(np.float32))
    show("copy SSE", -x[:, 3])
    show("refit resid", -x[:, 7])
    show("random", rng.random(len(x)).astype(np.float32))
    print(f"{'prefilter bound':14s}", f"{captured(-pre[val], gain[val], gid[val], vt, [1 << 30])[0]:.3f}",
          f"({len(x) / len(tot):.1f} blocks per group)")
    tr = np.ones(len(x), bool) if a.all else ~val
    tg = np.unique(gid[tr], return_inverse=True)[1]
    xt, gt = x[tr], gain[tr]
    ng = tg.max() + 1
    gstart = np.r_[0, np.flatnonzero(tg[1:] != tg[:-1]) + 1, len(tg)]
    order = np.argsort(tg, kind="stable")
    assert (order == np.arange(len(tg))).all()
    model = Mlp(NIN, a.hidden, rng)
    for ep in range(a.epochs):
        perm = rng.permutation(ng)
        tl = 0.0
        for b in range(0, ng, a.batch_groups):
            gs = perm[b:b + a.batch_groups]
            idx = np.concatenate([np.arange(gstart[g], gstart[g + 1]) for g in gs])
            lg = np.unique(tg[idx], return_inverse=True)[1]
            s, h = model.forward(xt[idx])
            ds, loss = listwise_grad(s, gt[idx], lg, len(gs), a.power)
            model.step(xt[idx], ds / len(gs), h, a.lr)
            tl += loss
        if ep % 10 == 9 or ep == a.epochs - 1:
            sv, _ = model.forward(x[val])
            print(f"MLP ep {ep + 1:3d} loss {tl:8.3f}",
                  " ".join(f"{c:.3f}" for c in captured(sv, gain[val], gid[val], vt)), flush=True)
    emit(model.p, a.hidden)


if __name__ == "__main__":
    main()
