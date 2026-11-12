//! The BC6H search, eight blocks at a time: lane `l` of every vector belongs to block `l`.
//!
//! Every block in a batch runs the same steps (one region, then the partitions), but each lane
//! keeps its own modes, partitions and endpoints, so per-mode values (endpoint precision, delta
//! limits) are vectors too. Unquantizing, interpolating and finishing are exact in `f32`
//! (every intermediate is an integer below 2^24), so the half-bit palette is the format's own;
//! squared errors are summed in `f32`.
//!
//! Kernels are generic over [`SimdF32`] and fully inlined into [`encode8`]; the AVX2 entry point
//! instantiates them with the target feature enabled. No closures inside kernels: they don't
//! inherit the caller's target features (see docs/DEADENDS.md).

use super::{Bc6hBlock, HALF_MAX, HdrBlock, MODES, ONE_ORDER, Search, TWO_ORDER, W3, W4, pack, rank_partitions};
use crate::bc7::tables::{ANCHOR2, PART2};
use crate::simd::{LANES, Level, SimdF32, SimdI32, f32x8};

type Mask<V> = <V as SimdF32>::Mask;
type Int<V> = <V as SimdF32>::Int;
/// Endpoints or quantized endpoints, `[region][end][channel]`.
type Ends<V> = [[[V; 3]; 2]; 2];

/// The weight tables padded to 16 entries for [`SimdI32::lookup16`].
const W3_16: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 8 {
        t[i] = W3[i] as u8;
        i += 1;
    }
    t
};
const W4_16: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = W4[i] as u8;
        i += 1;
    }
    t
};

#[inline(always)]
fn zero<V: SimdF32>() -> V {
    V::splat(0.0)
}

#[inline(always)]
fn all<V: SimdF32>() -> Mask<V> {
    V::splat(0.0).simd_lt(V::splat(1.0))
}

#[inline(always)]
fn not<V: SimdF32>(m: Mask<V>) -> Mask<V> {
    V::splat(0.5).simd_lt(V::select(m, V::splat(0.0), V::splat(1.0)))
}

#[inline(always)]
fn abs<V: SimdF32>(x: V) -> V {
    x.simd_max(V::splat(0.0) - x)
}

/// The eight blocks' texels as half bits (`h`) and on the 16-bit interpolation scale (`t`).
struct Target<V> {
    h: [[V; 3]; 16],
    t: [[V; 3]; 16],
}

#[inline(always)]
fn target<V: SimdF32>(px: &[HdrBlock; LANES]) -> Target<V> {
    let mut h = [[zero::<V>(); 3]; 16];
    let mut t = [[zero::<V>(); 3]; 16];
    for i in 0..16 {
        for c in 0..3 {
            let mut a = [0f32; LANES];
            for l in 0..LANES {
                let v = px[l][i][c];
                a[l] = f32::from(if v & 0x8000 != 0 { 0 } else { v.min(HALF_MAX) });
            }
            h[i][c] = V::from_array(a);
            t[i][c] = h[i][c] * V::splat(64.0 / 31.0);
        }
    }
    Target { h, t }
}

/// Per-lane mode settings.
#[derive(Clone, Copy)]
struct Params<V: SimdF32> {
    /// 2^(16 - eb): unquantized values are `(q + 1/2) * scale` between the ends.
    scale: V,
    /// 2^(eb - 16).
    inv: V,
    /// 2^eb - 1.
    max: V,
    /// 16-bit endpoints (stored as is).
    is16: Mask<V>,
    /// Delta limits per channel (transformed modes).
    lo: [V; 3],
    hi: [V; 3],
    transformed: Mask<V>,
}

#[inline(always)]
fn params<V: SimdF32>(modes: &[usize; LANES]) -> Params<V> {
    let mut scale = [0f32; LANES];
    let mut inv = [0f32; LANES];
    let mut max = [0f32; LANES];
    let mut is16 = [false; LANES];
    let mut lo = [[0f32; LANES]; 3];
    let mut hi = [[0f32; LANES]; 3];
    let mut tr = [false; LANES];
    for l in 0..LANES {
        let m = MODES[modes[l]];
        let s = (1u32 << (16 - m.eb)) as f32;
        scale[l] = s;
        inv[l] = 1.0 / s;
        max[l] = ((1u32 << m.eb) - 1) as f32;
        is16[l] = m.eb >= 16;
        for c in 0..3 {
            let h = (1i32 << (m.db[c] - 1)) as f32;
            lo[c][l] = -h;
            hi[c][l] = h - 1.0;
        }
        tr[l] = m.transformed;
    }
    Params {
        scale: V::from_array(scale),
        inv: V::from_array(inv),
        max: V::from_array(max),
        is16: V::mask_from_bools(is16),
        lo: [V::from_array(lo[0]), V::from_array(lo[1]), V::from_array(lo[2])],
        hi: [V::from_array(hi[0]), V::from_array(hi[1]), V::from_array(hi[2])],
        transformed: V::mask_from_bools(tr),
    }
}

/// An endpoint component back on the 16-bit scale (unsigned).
#[inline(always)]
fn unquantize<V: SimdF32>(q: V, p: &Params<V>) -> V {
    let half = V::splat(0.5);
    let r = ((q + half) * p.scale).floor();
    let r = V::select(q.simd_lt(half), zero(), r);
    V::select(p.max.simd_lt(q + half), V::splat(65535.0), r)
}

/// The quantized value whose unquantized value is nearest `x` (`x` in 0..=65535).
#[inline(always)]
fn quantize<V: SimdF32>(x: V, p: &Params<V>) -> V {
    let one = V::splat(1.0);
    let q0 = (x * p.inv).floor().simd_clamp(zero(), p.max);
    // Between the ends the grid is even, so the floor is nearest; at the ends compare q0's
    // neighbours (ties to the lower).
    let a = (q0 - one).simd_max(zero());
    let c = (q0 + one).simd_min(p.max);
    let mut best = a;
    let mut bd = abs(unquantize(a, p) - x);
    let d = abs(unquantize(q0, p) - x);
    let lt = d.simd_lt(bd);
    best = V::select(lt, q0, best);
    bd = V::select(lt, d, bd);
    let d = abs(unquantize(c, p) - x);
    best = V::select(d.simd_lt(bd), c, best);
    let interior = one.simd_lt(q0) & q0.simd_lt(p.max - one);
    let q = V::select(interior, q0, best);
    let r16 = (x + V::splat(0.5)).floor().simd_min(V::splat(65535.0));
    V::select(p.is16, r16, q)
}

/// The palette entry with weight `w` (of 64) between unquantized ends, as half bits.
#[inline(always)]
fn entry<V: SimdF32>(u0: V, u1: V, w: V) -> V {
    let v = ((u0 * (V::splat(64.0) - w) + u1 * w + V::splat(32.0)) * V::splat(1.0 / 64.0)).floor();
    (v * V::splat(31.0 / 64.0)).floor()
}

#[inline(always)]
fn sq_err<V: SimdF32>(x: &[V; 3], u: &[[V; 3]; 2], w: V) -> V {
    let mut e = zero::<V>();
    for c in 0..3 {
        let d = x[c] - entry(u[0][c], u[1][c], w);
        e += d * d;
    }
    e
}

/// Which region each texel is in, per lane.
struct Regions<V: SimdF32> {
    two: bool,
    /// Texel `i` is in region 1.
    reg1: [Mask<V>; 16],
    /// Texel `i` is region 1's anchor.
    anchor1: [Mask<V>; 16],
    /// Region 1 as 0/1 weights.
    w1: [V; 16],
    part: V,
}

#[inline(always)]
fn one_region<V: SimdF32>() -> Regions<V> {
    let none = V::mask_from_bools([false; LANES]);
    Regions {
        two: false,
        reg1: [none; 16],
        anchor1: [none; 16],
        w1: [zero(); 16],
        part: zero(),
    }
}

#[inline(always)]
fn two_regions<V: SimdF32>(parts: &[u8; LANES]) -> Regions<V> {
    let mut reg1 = [V::mask_from_bools([false; LANES]); 16];
    let mut anchor1 = reg1;
    let mut w1 = [zero::<V>(); 16];
    let mut pf = [0f32; LANES];
    for l in 0..LANES {
        pf[l] = f32::from(parts[l]);
    }
    for i in 0..16 {
        let mut r = [false; LANES];
        let mut a = [false; LANES];
        for l in 0..LANES {
            let p = usize::from(parts[l]);
            r[l] = (PART2[p] >> i) & 1 != 0;
            a[l] = usize::from(ANCHOR2[p]) == i;
        }
        reg1[i] = V::mask_from_bools(r);
        anchor1[i] = V::mask_from_bools(a);
        w1[i] = V::select(reg1[i], V::splat(1.0), zero());
    }
    Regions {
        two: true,
        reg1,
        anchor1,
        w1,
        part: V::from_array(pf),
    }
}

/// The line through the texels weighted `w` (0 or 1): its ends on the 16-bit scale, the mean,
/// the axis and the texel count. Weighted zeros add nothing, so this is the scalar fit of the
/// region's texels.
#[inline(always)]
fn line_fit<V: SimdF32>(tg: &Target<V>, w: &[V; 16]) -> ([[V; 3]; 2], [V; 3], [V; 3], V) {
    let mut n = zero::<V>();
    let mut s = [zero::<V>(); 3];
    for i in 0..16 {
        n += w[i];
        for c in 0..3 {
            s[c] += tg.t[i][c] * w[i];
        }
    }
    let nn = n.simd_max(V::splat(1.0));
    let mut m = [zero::<V>(); 3];
    for c in 0..3 {
        m[c] = s[c] / nn;
    }
    let mut cv = [zero::<V>(); 6];
    for i in 0..16 {
        let r = (tg.t[i][0] - m[0]) * w[i];
        let g = (tg.t[i][1] - m[1]) * w[i];
        let b = (tg.t[i][2] - m[2]) * w[i];
        cv[0] += r * r;
        cv[1] += r * g;
        cv[2] += r * b;
        cv[3] += g * g;
        cv[4] += g * b;
        cv[5] += b * b;
    }
    for c in &mut cv {
        *c = *c / nn;
    }
    let axis = axis3(&cv);
    let mut lo = zero::<V>();
    let mut hi = zero::<V>();
    for i in 0..16 {
        let t = ((tg.t[i][0] - m[0]) * axis[0] + (tg.t[i][1] - m[1]) * axis[1] + (tg.t[i][2] - m[2]) * axis[2]) * w[i];
        lo = lo.simd_min(t);
        hi = hi.simd_max(t);
    }
    let mut e = [[zero::<V>(); 3]; 2];
    for c in 0..3 {
        e[0][c] = (m[c] + axis[c] * lo).simd_clamp(zero(), V::splat(65535.0));
        e[1][c] = (m[c] + axis[c] * hi).simd_clamp(zero(), V::splat(65535.0));
    }
    (e, m, axis, n)
}

/// Principal axis of a covariance (rr, rg, rb, gg, gb, bb) by power iteration, as a unit vector
/// (the gray diagonal for a flat set).
#[inline(always)]
fn axis3<V: SimdF32>(c: &[V; 6]) -> [V; 3] {
    let rows = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
    // Start from the row with the largest diagonal entry.
    let m0 = not::<V>(c[0].simd_lt(c[3])) & not::<V>(c[0].simd_lt(c[5]));
    let m1 = not::<V>(c[3].simd_lt(c[5]));
    let mut v = [zero::<V>(); 3];
    for k in 0..3 {
        v[k] = V::select(m0, rows[0][k], V::select(m1, rows[1][k], rows[2][k]));
    }
    for _ in 0..6 {
        let mut w = [zero::<V>(); 3];
        for k in 0..3 {
            w[k] = rows[k][0] * v[0] + rows[k][1] * v[1] + rows[k][2] * v[2];
        }
        let mx = abs(w[0]).simd_max(abs(w[1])).simd_max(abs(w[2]));
        // Once the product vanishes the scalar loop stops; later products vanish too.
        let live = not::<V>(mx.simd_lt(V::splat(1e-20)));
        for k in 0..3 {
            v[k] = V::select(live, w[k] / mx, v[k]);
        }
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).simd_sqrt();
    let flat = len.simd_lt(V::splat(1e-12));
    let g = V::splat(1.0 / 3f32.sqrt());
    let mut out = [zero::<V>(); 3];
    for k in 0..3 {
        out[k] = V::select(flat, g, v[k] / len);
    }
    out
}

/// Nearest palette entry per texel and the total error: each texel is projected on its region's
/// line from the first palette entry to the last, and the nearest weight and its neighbour on the
/// projection's side are compared exactly.
#[inline(always)]
fn index<V: SimdF32>(tg: &Target<V>, rg: &Regions<V>, p: &Params<V>, q: &Ends<V>) -> ([Int<V>; 16], V) {
    let (n, wt) = if rg.two { (8, &W3_16) } else { (16, &W4_16) };
    let nr = if rg.two { 2 } else { 1 };
    let mut u = [[[zero::<V>(); 3]; 2]; 2];
    let mut p0 = [[zero::<V>(); 3]; 2];
    let mut d = [[zero::<V>(); 3]; 2];
    let mut s = [zero::<V>(); 2];
    let fin = V::splat(31.0 / 64.0);
    for r in 0..nr {
        let mut dd = zero::<V>();
        for c in 0..3 {
            u[r][0][c] = unquantize(q[r][0][c], p);
            u[r][1][c] = unquantize(q[r][1][c], p);
            p0[r][c] = (u[r][0][c] * fin).floor();
            d[r][c] = (u[r][1][c] * fin).floor() - p0[r][c];
            dd += d[r][c] * d[r][c];
        }
        s[r] = V::select(zero::<V>().simd_lt(dd), V::splat(64.0) / dd, zero());
    }
    let step = V::splat((n - 1) as f32 / 64.0);
    let mut idx = [Int::<V>::splat(0); 16];
    let mut total = zero::<V>();
    for i in 0..16 {
        let (uu, pp, dv, sv) = if rg.two {
            let m = rg.reg1[i];
            let mut uu = u[0];
            let mut pp = p0[0];
            let mut dv = d[0];
            for c in 0..3 {
                uu[0][c] = V::select(m, u[1][0][c], u[0][0][c]);
                uu[1][c] = V::select(m, u[1][1][c], u[0][1][c]);
                pp[c] = V::select(m, p0[1][c], p0[0][c]);
                dv[c] = V::select(m, d[1][c], d[0][c]);
            }
            (uu, pp, dv, V::select(m, s[1], s[0]))
        } else {
            (u[0], p0[0], d[0], s[0])
        };
        let x = &tg.h[i];
        let t = ((x[0] - pp[0]) * dv[0] + (x[1] - pp[1]) * dv[1] + (x[2] - pp[2]) * dv[2]) * sv;
        // The nearest weight to round(t), ties to the lower (products are 64ths, so +0.49
        // rounds exact halves down and nothing else).
        let xr = (t.simd_max(zero()) + V::splat(0.5)).floor().simd_min(V::splat(64.0));
        let j0 = (xr * step + V::splat(0.49)).floor();
        let j0i = j0.to_int_trunc();
        let w0 = Int::<V>::lookup16(wt, j0i).to_float();
        let j1 = V::select(w0.simd_lt(t), j0 + V::splat(1.0), j0 - V::splat(1.0));
        let valid = V::splat(-0.5).simd_lt(j1) & j1.simd_lt(V::splat(n as f32 - 0.5));
        let j1i = j1.simd_clamp(zero(), V::splat(15.0)).to_int_trunc();
        let w1 = Int::<V>::lookup16(wt, j1i).to_float();
        let e0 = sq_err(x, &uu, w0);
        let e1 = V::select(valid, sq_err(x, &uu, w1), V::splat(f32::MAX));
        let take1 = e1.simd_lt(e0);
        idx[i] = Int::<V>::select(take1, j1i, j0i);
        total += e0.simd_min(e1);
    }
    (idx, total)
}

/// Least-squares ends for the indices, per region (the region's mean where its indices are all
/// equal).
#[inline(always)]
fn refit<V: SimdF32>(tg: &Target<V>, rg: &Regions<V>, idx: &[Int<V>; 16]) -> Ends<V> {
    let wt = if rg.two { &W3_16 } else { &W4_16 };
    // Sums over all texels ([0]) and over region 1 ([1]); region 0 is the difference.
    let mut aa = [zero::<V>(); 2];
    let mut ab = [zero::<V>(); 2];
    let mut bb = [zero::<V>(); 2];
    let mut ax = [[zero::<V>(); 3]; 2];
    let mut bx = [[zero::<V>(); 3]; 2];
    let mut sum = [[zero::<V>(); 3]; 2];
    let mut n1 = zero::<V>();
    for i in 0..16 {
        let t = Int::<V>::lookup16(wt, idx[i]).to_float() * V::splat(1.0 / 64.0);
        let s = V::splat(1.0) - t;
        let (ss, st, tt) = (s * s, s * t, t * t);
        aa[0] += ss;
        ab[0] += st;
        bb[0] += tt;
        for c in 0..3 {
            let x = tg.t[i][c];
            ax[0][c] += s * x;
            bx[0][c] += t * x;
            sum[0][c] += x;
        }
        if rg.two {
            let m = rg.w1[i];
            aa[1] += ss * m;
            ab[1] += st * m;
            bb[1] += tt * m;
            n1 += m;
            for c in 0..3 {
                let x = tg.t[i][c] * m;
                ax[1][c] += s * x;
                bx[1][c] += t * x;
                sum[1][c] += x;
            }
        }
    }
    let mut ends = [[[zero::<V>(); 3]; 2]; 2];
    let nr = if rg.two { 2 } else { 1 };
    for r in 0..nr {
        // Region r's sums: region 1's, or all less region 1's.
        let (a2, b2, c2) = (
            pick(r, rg.two, aa[0], aa[1]),
            pick(r, rg.two, ab[0], ab[1]),
            pick(r, rg.two, bb[0], bb[1]),
        );
        let n = pick(r, rg.two, V::splat(16.0), n1);
        let det = a2 * c2 - b2 * b2;
        let ok = V::splat(1e-6).simd_lt(abs(det));
        for c in 0..3 {
            let x0 = pick(r, rg.two, ax[0][c], ax[1][c]);
            let x1 = pick(r, rg.two, bx[0][c], bx[1][c]);
            let mean = pick(r, rg.two, sum[0][c], sum[1][c]) / n.simd_max(V::splat(1.0));
            let e0 = V::select(ok, (x0 * c2 - x1 * b2) / det, mean);
            let e1 = V::select(ok, (x1 * a2 - x0 * b2) / det, mean);
            ends[r][0][c] = e0.simd_clamp(zero(), V::splat(65535.0));
            ends[r][1][c] = e1.simd_clamp(zero(), V::splat(65535.0));
        }
    }
    ends
}

/// Region `r`'s share of a sum, from the total (`all`) and region 1's (`one`).
#[inline(always)]
fn pick<V: SimdF32>(r: usize, two: bool, all: V, one: V) -> V {
    if r == 1 {
        one
    } else if two {
        all - one
    } else {
        all
    }
}

/// One encoding attempt per lane: quantized ends (absolute), indices and error.
#[derive(Clone, Copy)]
struct Trial<V: SimdF32> {
    err: V,
    q: Ends<V>,
    idx: [Int<V>; 16],
}

#[inline(always)]
fn select_trial<V: SimdF32>(m: Mask<V>, a: &Trial<V>, b: &Trial<V>) -> Trial<V> {
    let mut t = *b;
    t.err = V::select(m, a.err, b.err);
    for r in 0..2 {
        for k in 0..2 {
            for c in 0..3 {
                t.q[r][k][c] = V::select(m, a.q[r][k][c], b.q[r][k][c]);
            }
        }
    }
    for i in 0..16 {
        t.idx[i] = Int::<V>::select(m, a.idx[i], b.idx[i]);
    }
    t
}

/// Whether `d` is within the delta limits.
#[inline(always)]
fn in_range<V: SimdF32>(d: V, lo: V, hi: V) -> Mask<V> {
    not::<V>(d.simd_lt(lo) | hi.simd_lt(d))
}

/// Quantizes `ends` for each lane's mode, indexes every texel and fixes the anchors.
#[inline(always)]
fn quantize_and_index<V: SimdF32>(tg: &Target<V>, rg: &Regions<V>, p: &Params<V>, ends: &Ends<V>) -> Trial<V> {
    let nr = if rg.two { 2 } else { 1 };
    let mut q = [[[zero::<V>(); 3]; 2]; 2];
    for r in 0..nr {
        for k in 0..2 {
            for c in 0..3 {
                q[r][k][c] = quantize(ends[r][k][c], p);
            }
        }
    }
    // Deltas from the first end, clamped to their bits.
    for (r, k) in [(0, 1), (1, 0), (1, 1)].into_iter().take(2 * nr - 1) {
        for c in 0..3 {
            let b = q[0][0][c];
            let v = b + (q[r][k][c] - b).simd_clamp(p.lo[c], p.hi[c]);
            q[r][k][c] = V::select(p.transformed, v, q[r][k][c]);
        }
    }
    let (mut idx, mut err) = index(tg, rg, p, &q);
    let n = if rg.two { 8 } else { 16 };
    let top = Int::<V>::splat(n / 2 - 1);
    let last = Int::<V>::splat(n - 1);

    // Region 0, anchored at texel 0. Swapping its ends and mirroring its indices decodes the
    // same; in a transformed mode the swap moves the base, so the deltas must still fit.
    let flag = top.simd_lt(idx[0]);
    let mut fits = not::<V>(p.transformed);
    {
        let mut ok = all::<V>();
        for c in 0..3 {
            let b = q[0][1][c];
            ok = ok & in_range(q[0][0][c] - b, p.lo[c], p.hi[c]);
            if rg.two {
                ok = ok & in_range(q[1][0][c] - b, p.lo[c], p.hi[c]);
                ok = ok & in_range(q[1][1][c] - b, p.lo[c], p.hi[c]);
            }
        }
        fits = fits | ok;
    }
    let swap = flag & fits;
    let fix = flag & not::<V>(fits);
    if V::mask_any(fix) {
        // Keep the ends; texel 0 takes its best index with the top bit clear.
        let mut u = [[zero::<V>(); 3]; 2];
        for k in 0..2 {
            for c in 0..3 {
                u[k][c] = unquantize(q[0][k][c], p);
            }
        }
        let wt = if rg.two { &W3_16 } else { &W4_16 };
        let old = sq_err(&tg.h[0], &u, Int::<V>::lookup16(wt, idx[0]).to_float());
        let mut be = V::splat(f32::MAX);
        let mut bj = Int::<V>::splat(0);
        for j in 0..n / 2 {
            let e = sq_err(&tg.h[0], &u, V::splat(wt[j as usize] as f32));
            let lt = e.simd_lt(be);
            be = V::select(lt, e, be);
            bj = Int::<V>::select(lt, Int::<V>::splat(j), bj);
        }
        idx[0] = Int::<V>::select(fix, bj, idx[0]);
        err += V::select(fix, be - old, zero());
    }
    if V::mask_any(swap) {
        for c in 0..3 {
            let (a, b) = (q[0][0][c], q[0][1][c]);
            q[0][0][c] = V::select(swap, b, a);
            q[0][1][c] = V::select(swap, a, b);
        }
        for i in 0..16 {
            let m = if rg.two { swap & not::<V>(rg.reg1[i]) } else { swap };
            idx[i] = Int::<V>::select(m, last - idx[i], idx[i]);
        }
    }
    if rg.two {
        // Region 1: its deltas fit either way round.
        let mut a = Int::<V>::splat(0);
        for i in 0..16 {
            a = Int::<V>::select(rg.anchor1[i], idx[i], a);
        }
        let swap = top.simd_lt(a);
        if V::mask_any(swap) {
            for c in 0..3 {
                let (a, b) = (q[1][0][c], q[1][1][c]);
                q[1][0][c] = V::select(swap, b, a);
                q[1][1][c] = V::select(swap, a, b);
            }
            for i in 0..16 {
                idx[i] = Int::<V>::select(swap & rg.reg1[i], last - idx[i], idx[i]);
            }
        }
    }
    Trial { err, q, idx }
}

/// The best trial so far per lane, with its mode and partition.
struct Best<V: SimdF32> {
    t: Trial<V>,
    mode: V,
    part: V,
}

#[inline(always)]
fn merge<V: SimdF32>(best: &mut Best<V>, t: &Trial<V>, m: Mask<V>, mode: V, part: V) {
    best.t = select_trial(m, t, &best.t);
    best.mode = V::select(m, mode, best.mode);
    best.part = V::select(m, part, best.part);
}

/// Tries each lane's mode from `ends`, refitting up to `refits` times, where `active`.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel state is passed by reference, not bundled"
)]
#[inline(always)]
fn try_mode<V: SimdF32>(
    tg: &Target<V>,
    rg: &Regions<V>,
    p: &Params<V>,
    ends: &Ends<V>,
    refits: usize,
    active: Mask<V>,
    mode: V,
    best: &mut Best<V>,
) {
    let mut t = quantize_and_index(tg, rg, p, ends);
    let mut going = active;
    for _ in 0..refits {
        merge(best, &t, going & t.err.simd_lt(best.t.err), mode, rg.part);
        going = going & zero::<V>().simd_lt(t.err);
        if !V::mask_any(going) {
            return;
        }
        let e = refit(tg, rg, &t.idx);
        let n = quantize_and_index(tg, rg, p, &e);
        going = going & n.err.simd_lt(t.err);
        t = select_trial(going, &n, &t);
    }
    merge(best, &t, going & t.err.simd_lt(best.t.err), mode, rg.part);
}

/// Estimated squared error (16-bit scale) of storing `ends` in mode `mi` (the same for every
/// lane): ends quantized (`q`), deltas clamped, and each region's end displacements spread over
/// its `cnt` texels as interpolation does.
#[inline(always)]
fn mode_estimate<V: SimdF32>(mi: usize, p: &Params<V>, ends: &Ends<V>, q: &Ends<V>, cnt: &[V; 2]) -> V {
    let m = MODES[mi];
    let nr = m.regions();
    let mut q = *q;
    if m.transformed {
        for (r, k) in [(0, 1), (1, 0), (1, 1)].into_iter().take(2 * nr - 1) {
            for c in 0..3 {
                let b = q[0][0][c];
                q[r][k][c] = b + (q[r][k][c] - b).simd_clamp(p.lo[c], p.hi[c]);
            }
        }
    }
    let mut e = zero::<V>();
    for r in 0..nr {
        let mut s = zero::<V>();
        for c in 0..3 {
            let a = unquantize(q[r][0][c], p) - ends[r][0][c];
            let b = unquantize(q[r][1][c], p) - ends[r][1][c];
            s += a * a + a * b + b * b;
        }
        e += cnt[r] * s / V::splat(3.0);
    }
    e
}

/// The `k` modes from `order` with the least estimated error per lane, best first (ties to the
/// earlier, more precise mode), as [`MODES`] indices.
#[inline(always)]
fn pick_modes<V: SimdF32, const N: usize>(
    order: &[usize; N],
    ends: &Ends<V>,
    cnt: &[V; 2],
    k: usize,
) -> [[usize; LANES]; N] {
    let nr = MODES[order[0]].regions();
    let mut cache = [[[[zero::<V>(); 3]; 2]; 2]; 17];
    let mut have = 0u32;
    let mut est = [zero::<V>(); N];
    for i in 0..N {
        let mi = order[i];
        let p = params::<V>(&[mi; LANES]);
        let eb = usize::from(MODES[mi].eb);
        if have & (1 << eb) == 0 {
            have |= 1 << eb;
            for r in 0..nr {
                for kk in 0..2 {
                    for c in 0..3 {
                        cache[eb][r][kk][c] = quantize(ends[r][kk][c], &p);
                    }
                }
            }
        }
        est[i] = mode_estimate(mi, &p, ends, &cache[eb], cnt);
    }
    let mut out = [[0usize; LANES]; N];
    for slot in out.iter_mut().take(k.clamp(1, N)) {
        let mut be = V::splat(f32::INFINITY);
        let mut bi = zero::<V>();
        for i in 0..N {
            let lt = est[i].simd_lt(be);
            be = V::select(lt, est[i], be);
            bi = V::select(lt, V::splat(i as f32), bi);
        }
        let bi = bi.to_array();
        for l in 0..LANES {
            slot[l] = order[bi[l] as usize];
        }
        let biv = V::from_array(bi);
        for i in 0..N {
            let chosen = abs(biv - V::splat(i as f32)).simd_lt(V::splat(0.5));
            est[i] = V::select(chosen, V::splat(f32::INFINITY), est[i]);
        }
    }
    out
}

/// Encodes eight blocks with search settings `s`.
#[inline(always)]
fn encode8<V: SimdF32>(px: &[HdrBlock; LANES], s: Search) -> [[u8; 16]; LANES] {
    let tg = target::<V>(px);
    let mut best = Best {
        t: Trial {
            err: V::splat(f32::MAX),
            q: [[[zero::<V>(); 3]; 2]; 2],
            idx: [Int::<V>::splat(0); 16],
        },
        mode: zero(),
        part: zero(),
    };
    // One region: the line through all 16 texels.
    let rg = one_region::<V>();
    let (one, mean, axis, _) = line_fit(&tg, &[V::splat(1.0); 16]);
    let ends = [one, one];
    let k1 = s.modes1.clamp(1, ONE_ORDER.len());
    let modes = pick_modes(&ONE_ORDER, &ends, &[V::splat(16.0), zero()], k1);
    for m in &modes[..k1] {
        let p = params::<V>(m);
        try_mode(&tg, &rg, &p, &ends, s.refits, all::<V>(), mode_vec::<V>(m), &mut best);
    }
    let mut active = zero::<V>().simd_lt(best.t.err);
    if V::mask_any(active) {
        // Two regions: each lane ranks the partitions by estimated fit.
        let np = s.partitions.clamp(1, 32);
        let mut parts = [[0u8; 32]; LANES];
        let mut est0 = [0f32; LANES];
        let mut tt = [[[0f32; LANES]; 3]; 16];
        let mut ma = [[0f32; LANES]; 3];
        let mut aa = [[0f32; LANES]; 3];
        for c in 0..3 {
            for i in 0..16 {
                tt[i][c] = tg.t[i][c].to_array();
            }
            ma[c] = mean[c].to_array();
            aa[c] = axis[c].to_array();
        }
        for l in 0..LANES {
            let mut t = [[0f32; 4]; 16];
            for i in 0..16 {
                t[i] = [tt[i][0][l], tt[i][1][l], tt[i][2][l], 0.0];
            }
            let (pl, el) = rank_partitions(
                &t,
                &[ma[0][l], ma[1][l], ma[2][l], 0.0],
                &[aa[0][l], aa[1][l], aa[2][l], 0.0],
                np,
            );
            parts[l] = pl;
            est0[l] = el[0];
        }
        if s.skip > 0.0 {
            // The estimate is on the 16-bit scale; errors are in half bits.
            let scale = (31.0f32 / 64.0) * (31.0 / 64.0);
            let lim = V::from_array(est0) * V::splat(s.skip * scale);
            active = active & lim.simd_lt(best.t.err);
        }
        let k2 = s.modes2.clamp(1, TWO_ORDER.len());
        for slot in 0..np {
            if !V::mask_any(active) {
                break;
            }
            let mut pl = [0u8; LANES];
            for l in 0..LANES {
                pl[l] = parts[l][slot];
            }
            let rg = two_regions::<V>(&pl);
            let mut w0 = [zero::<V>(); 16];
            for i in 0..16 {
                w0[i] = V::splat(1.0) - rg.w1[i];
            }
            let (e0, _, _, n0) = line_fit(&tg, &w0);
            let (e1, _, _, n1) = line_fit(&tg, &rg.w1);
            let ends = [e0, e1];
            let modes = pick_modes(&TWO_ORDER, &ends, &[n0, n1], k2);
            for m in &modes[..k2] {
                let p = params::<V>(m);
                try_mode(&tg, &rg, &p, &ends, s.refits, active, mode_vec::<V>(m), &mut best);
            }
        }
    }
    finish_blocks(&best)
}

#[inline(always)]
fn mode_vec<V: SimdF32>(m: &[usize; LANES]) -> V {
    let mut a = [0f32; LANES];
    for l in 0..LANES {
        a[l] = m[l] as f32;
    }
    V::from_array(a)
}

/// Packs each lane's best trial.
#[inline(always)]
fn finish_blocks<V: SimdF32>(best: &Best<V>) -> [[u8; 16]; LANES] {
    let mode = best.mode.to_array();
    let part = best.part.to_array();
    let mut q = [[[[0f32; LANES]; 3]; 2]; 2];
    for r in 0..2 {
        for k in 0..2 {
            for c in 0..3 {
                q[r][k][c] = best.t.q[r][k][c].to_array();
            }
        }
    }
    let mut idx = [[0i32; LANES]; 16];
    for i in 0..16 {
        idx[i] = best.t.idx[i].to_array();
    }
    let mut out = [[0u8; 16]; LANES];
    for l in 0..LANES {
        let mi = mode[l] as usize;
        let m = MODES[mi];
        let mut b = Bc6hBlock {
            mode: mi as u8,
            partition: if m.two_regions { part[l] as u8 } else { 0 },
            ..Default::default()
        };
        for i in 0..16 {
            b.idx[i] = idx[i][l] as u8;
        }
        let base = [q[0][0][0][l] as i32, q[0][0][1][l] as i32, q[0][0][2][l] as i32];
        for k in 0..2 * m.regions() {
            for c in 0..3 {
                let v = q[k / 2][k % 2][c][l] as i32;
                b.fields[k][c] = if k > 0 && m.transformed {
                    ((v - base[c]) & ((1 << m.db[c]) - 1)) as u16
                } else {
                    v as u16
                };
            }
        }
        out[l] = pack(&b);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode8_avx2(px: &[HdrBlock; LANES], s: Search) -> [[u8; 16]; LANES] {
    encode8::<crate::simd::avx2::F>(px, s)
}

/// Encodes up to eight blocks at effort 0..=9 (shorter batches are padded with their last block).
pub fn encode_batch(blocks: &[HdrBlock], effort: u8) -> Vec<[u8; 16]> {
    debug_assert!(!blocks.is_empty() && blocks.len() <= LANES);
    let s = super::search(effort);
    let last = blocks[blocks.len() - 1];
    let mut lanes = [last; LANES];
    lanes[..blocks.len()].copy_from_slice(blocks);
    let out = match crate::simd::level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { encode8_avx2(&lanes, s) },
        _ => encode8::<f32x8>(&lanes, s),
    };
    out[..blocks.len()].to_vec()
}

/// Encodes eight blocks with the portable backend (tests compare it with AVX2).
#[cfg(test)]
pub(super) fn encode8_generic(px: &[HdrBlock; LANES], effort: u8) -> [[u8; 16]; LANES] {
    encode8::<f32x8>(px, super::search(effort))
}
