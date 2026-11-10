"""Bjøntegaard delta rate, shared by compete.py and readme_charts.py."""

import math


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
