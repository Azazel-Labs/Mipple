//! BC1 batch encoder for every effort: eight blocks per SIMD batch, one per lane.
//!
//! Every tier starts with a line fit: one power iteration of the covariance (started from its
//! largest-diagonal row), the extreme projections inset by 1/16 of their range, 565
//! quantization and indices by projection onto the quantized endpoints (Intel ISPC's method,
//! plus the inset). Effort 0 keeps one least-squares refit unconditionally; effort 1 keeps up
//! to two, each only when it lowers the error. From effort 2 ([`cfg`] lists the tiers):
//!
//! * [`cluster3`]: a 3-color cluster fit over every split of the texels sorted along the
//!   principal axis, with dark texels left to the black entry;
//! * [`polish`]: least-squares refits from the chosen indices;
//! * [`cluster_snapped`]: a 4-color cluster fit across the eight lanes. Lanes sort
//!   independently with a sorting network, every lane scores the same splits (in a learned
//!   frequency order, so prefix sums are plain vector reads), and splits are scored with
//!   565-snapped endpoints only when the closed-form least-squares bound says some lane can
//!   still win;
//! * [`refine`] (effort 4+): greedy ±1 endpoint moves under the exact error;
//! * effort 6+: a second round sorted along the best endpoints' direction, and the best few
//!   4-color splits per lane evaluated exactly.
//!
//! Each candidate is kept per lane only when its exact error is lower. Solid blocks use the
//! optimal single-color tables of the scalar encoder. All arithmetic is plain IEEE (with
//! fused multiply-adds) in a fixed order, so both backends agree bit for bit.

#[cfg(target_arch = "x86_64")]
use crate::simd::Level;
use crate::simd::{LANES, SimdF32, SimdI32, f32x8, level};
use crate::surface::Surface;

/// `floor` for non-negative lanes.
#[inline(always)]
fn floor<V: SimdF32>(v: V) -> V {
    v.floor()
}

/// Reduction of `$e` over `$i` in 0..16 with `$op` in four interleaved chains (dependency
/// depth 6 instead of 16). A macro rather than a closure: closures do not inherit the caller's
/// target features, so an un-inlined one would turn every SIMD intrinsic into a call.
macro_rules! reduce16 {
    (|$i:ident| $e:expr, |$a:ident, $b:ident| $op:expr) => {{
        let mut acc = [
            {
                let $i = 0usize;
                $e
            },
            {
                let $i = 1usize;
                $e
            },
            {
                let $i = 2usize;
                $e
            },
            {
                let $i = 3usize;
                $e
            },
        ];
        let mut $i = 4usize;
        while $i < 16 {
            let x = $e;
            let ($a, $b) = (acc[$i & 3], x);
            acc[$i & 3] = $op;
            $i += 1;
        }
        let (l, r) = {
            let ($a, $b) = (acc[0], acc[1]);
            let l = $op;
            let ($a, $b) = (acc[2], acc[3]);
            (l, $op)
        };
        let ($a, $b) = (l, r);
        $op
    }};
}

/// Sum of `$e` over `$i` in 0..16 (see [`reduce16`]).
macro_rules! sum16 {
    (|$i:ident| $e:expr) => {
        reduce16!(|$i| $e, |a, b| a + b)
    };
}

/// Texels of eight blocks: `c[channel][texel]`, one block per lane.
type Texels<V> = [[V; 16]; 3];

/// 565 levels of an endpoint and its decoded 8-bit value per channel.
struct Quant<V: SimdF32> {
    q: [V; 3],
    v: [V; 3],
}

/// Nearest 565 levels (`(x * levels / 255 + 0.5)` truncated, as the scalar `quant565`).
#[inline(always)]
fn quantize<V: SimdF32>(e: [V; 3]) -> Quant<V> {
    let (zero, full) = (V::splat(0.0), V::splat(255.0));
    let q5 = V::splat(31.0 / 255.0);
    let q6 = V::splat(63.0 / 255.0);
    let half = V::splat(0.5);
    let q = [
        floor(e[0].simd_clamp(zero, full) * q5 + half),
        floor(e[1].simd_clamp(zero, full) * q6 + half),
        floor(e[2].simd_clamp(zero, full) * q5 + half),
    ];
    // expand5(q) = 8q + q / 4, expand6(q) = 4q + q / 16 (bit replication).
    let v = [
        q[0] * V::splat(8.0) + floor(q[0] * V::splat(0.25)),
        q[1] * V::splat(4.0) + floor(q[1] * V::splat(1.0 / 16.0)),
        q[2] * V::splat(8.0) + floor(q[2] * V::splat(0.25)),
    ];
    Quant { q, v }
}

/// The packed 565 value.
#[inline(always)]
fn pack565<V: SimdF32>(q: &Quant<V>) -> V::Int {
    (q.q[0].to_int_trunc() << 11) | (q.q[1].to_int_trunc() << 5) | q.q[2].to_int_trunc()
}

/// A candidate encoding per lane: endpoints in 4-color order (c0 >= c1) and projection steps
/// (0 = c0 .. 3 = c1) per texel.
struct Fit<V: SimdF32> {
    c0: V::Int,
    c1: V::Int,
    a: [V; 3],
    b: [V; 3],
    step: [V; 16],
}

/// Orders two quantized endpoints into 4-color order and assigns steps by projection.
#[inline(always)]
fn assign<V: SimdF32>(t: &Texels<V>, e0: [V; 3], e1: [V; 3]) -> Fit<V> {
    let (q0, q1) = (quantize(e0), quantize(e1));
    let (p0, p1) = (pack565(&q0), pack565(&q1));
    let swap = p0.simd_lt(p1);
    let c0 = V::Int::select(swap, p1, p0);
    let c1 = V::Int::select(swap, p0, p1);
    let mut a = q0.v;
    let mut b = q1.v;
    let mut c = 0;
    while c < 3 {
        a[c] = V::select(swap, q1.v[c], q0.v[c]);
        b[c] = V::select(swap, q0.v[c], q1.v[c]);
        c += 1;
    }
    // step = clamp(round(3 (p - a).d / |d|^2)) with d = b - a, folded to p.k + bias.
    let zero = V::splat(0.0);
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let n2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let s = V::select(zero.simd_lt(n2), V::splat(3.0) / n2.simd_max(V::splat(1.0)), zero);
    let k = [d[0] * s, d[1] * s, d[2] * s];
    let bias = V::splat(0.5) - (a[0] * k[0] + a[1] * k[1] + a[2] * k[2]);
    let mut step = [zero; 16];
    let mut i = 0;
    while i < 16 {
        let x = t[0][i] * k[0] + t[1][i] * k[1] + t[2][i] * k[2] + bias;
        step[i] = floor(x.simd_clamp(zero, V::splat(3.0)));
        i += 1;
    }
    Fit { c0, c1, a, b, step }
}

/// Least-squares endpoints for the fit's steps (weight step / 3 toward c1). `None` lanes (a
/// singular system) keep `fallback`; returns (e0, e1, valid).
#[inline(always)]
fn refit<V: SimdF32>(t: &Texels<V>, sum: &[V; 3], f: &Fit<V>) -> ([V; 3], [V; 3], V::Mask) {
    let zero = V::splat(0.0);
    // Steps and texels are integers, so these sums are exact in any order; weights are step / 3.
    let (third, ninth) = (V::splat(1.0 / 3.0), V::splat(1.0 / 9.0));
    let st = sum16!(|i| f.step[i]) * third;
    let stt = sum16!(|i| f.step[i] * f.step[i]) * ninth;
    let stp = [
        sum16!(|i| f.step[i] * t[0][i]) * third,
        sum16!(|i| f.step[i] * t[1][i]) * third,
        sum16!(|i| f.step[i] * t[2][i]) * third,
    ];
    let s00 = V::splat(16.0) - st - st + stt;
    let s01 = st - stt;
    let det = s00 * stt - s01 * s01;
    let ok = V::splat(0.5).simd_lt(det);
    let inv = V::splat(1.0) / V::select(ok, det, V::splat(1.0));
    let mut e0 = [zero; 3];
    let mut e1 = [zero; 3];
    let mut c = 0;
    while c < 3 {
        let (s0x, s1x) = (sum[c] - stp[c], stp[c]);
        e0[c] = (stt * s0x - s01 * s1x) * inv;
        e1[c] = (s00 * s1x - s01 * s0x) * inv;
        c += 1;
    }
    (e0, e1, ok)
}

/// Squared error of a fit against the decoder palette: floor(((3 - s) a + s b) / 3) per channel.
#[inline(always)]
fn error<V: SimdF32>(t: &Texels<V>, f: &Fit<V>) -> V {
    let (three, half, third) = (V::splat(3.0), V::splat(0.5), V::splat(1.0 / 3.0));
    // Integer squared errors: exact in any summation order.
    sum16!(|i| {
        let s = f.step[i];
        let r = three - s;
        let d0 = t[0][i] - floor((r * f.a[0] + s * f.b[0] + half) * third);
        let d1 = t[1][i] - floor((r * f.a[1] + s * f.b[1] + half) * third);
        let d2 = t[2][i] - floor((r * f.a[2] + s * f.b[2] + half) * third);
        d0 * d0 + d1 * d1 + d2 * d2
    })
}

/// Per-lane select between two fits.
#[inline(always)]
fn choose<V: SimdF32>(m: V::Mask, x: Fit<V>, y: Fit<V>) -> Fit<V> {
    let mut out = y;
    out.c0 = V::Int::select(m, x.c0, out.c0);
    out.c1 = V::Int::select(m, x.c1, out.c1);
    let mut c = 0;
    while c < 3 {
        out.a[c] = V::select(m, x.a[c], out.a[c]);
        out.b[c] = V::select(m, x.b[c], out.b[c]);
        c += 1;
    }
    let mut i = 0;
    while i < 16 {
        out.step[i] = V::select(m, x.step[i], out.step[i]);
        i += 1;
    }
    out
}

/// Batcher odd-even merge sort for 16 keys (63 compare-exchanges).
const SORT16: [(u8, u8); 63] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (8, 10),
    (9, 11),
    (12, 14),
    (13, 15),
    (1, 2),
    (5, 6),
    (9, 10),
    (13, 14),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (8, 12),
    (9, 13),
    (10, 14),
    (11, 15),
    (2, 4),
    (3, 5),
    (10, 12),
    (11, 13),
    (1, 2),
    (3, 4),
    (5, 6),
    (9, 10),
    (11, 12),
    (13, 14),
    (0, 8),
    (1, 9),
    (2, 10),
    (3, 11),
    (4, 12),
    (5, 13),
    (6, 14),
    (7, 15),
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (2, 4),
    (3, 5),
    (6, 8),
    (7, 9),
    (10, 12),
    (11, 13),
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
];

/// What the batch encoder searches beyond the line fit.
#[derive(Clone, Copy, Debug)]
pub(super) struct Cfg {
    /// Least-squares refits after the line fit, each kept only if it lowers the error
    /// (0 = one refit kept unconditionally, effort 0's fast path).
    pub refits: u8,
    /// 4-color cluster-fit splits scored (in the learned order); 0 = no cluster fit.
    pub splits: usize,
    /// Score splits with the closed-form least-squares error instead of snapped endpoints.
    pub closed: bool,
    /// Also try a 3-color cluster fit with dark texels on the black entry.
    pub three: bool,
    /// Snapped scoring skips a split unless its closed-form error plus this multiple of the
    /// expected 565 snapping penalty beats some lane's best (0 = exact pruning only).
    pub prune: f32,
    /// Endpoint refinement sweeps (see [`refine`]).
    pub refine: u8,
    /// Cluster-fit rounds: later rounds sort along the best endpoints' direction.
    pub rounds: u8,
    /// Refinement move groups (2 = single-endpoint moves, 4 = also joint moves).
    pub moves: usize,
    /// Refine after every cluster round, not just the last.
    pub refine_each: bool,
    /// Least-squares refits from the chosen indices before refining (see [`polish`]).
    pub polish: u8,
    /// 4-color splits per lane evaluated exactly (1, 2 or 4 best by snapped estimate).
    pub topk: u8,
}

/// The batch configuration of each effort the batch encoder handles.
pub(super) fn cfg(effort: u8) -> Cfg {
    let base = Cfg {
        refits: 2,
        splits: 0,
        closed: false,
        three: true,
        prune: 0.0,
        refine: 0,
        rounds: 1,
        moves: 2,
        refine_each: false,
        polish: 0,
        topk: 1,
    };
    // Each tier is the cheapest search found to beat a competitor tier on Kodak (see
    // docs/BENCHMARKS.md).
    match effort {
        0 => EFFORT0,
        1 => Cfg { three: false, ..base },
        2 => Cfg {
            splits: 128,
            prune: 1.0,
            polish: 2,
            ..base
        },
        3 => Cfg {
            splits: 512,
            prune: 0.8,
            polish: 2,
            ..base
        },
        4 => Cfg {
            splits: 512,
            prune: 0.8,
            refine: 1,
            ..base
        },
        5 => Cfg {
            splits: usize::MAX,
            prune: 0.8,
            refine: 1,
            moves: 4,
            ..base
        },
        6 => Cfg {
            splits: usize::MAX,
            prune: 0.4,
            refine: 1,
            rounds: 2,
            moves: 4,
            refine_each: true,
            ..base
        },
        7 => Cfg {
            splits: usize::MAX,
            refine: 1,
            rounds: 2,
            moves: 4,
            refine_each: true,
            topk: 2,
            ..base
        },
        _ => Cfg {
            splits: usize::MAX,
            refine: 1,
            rounds: 2,
            moves: 4,
            refine_each: true,
            topk: 4,
            ..base
        },
    }
}

/// Endpoints (quantized, 4-color order unless equal), index bits and error of an encoding.
struct Coded<V: SimdF32> {
    c0: V::Int,
    c1: V::Int,
    bits: V::Int,
    err: V,
}

/// Exact index assignment: each texel takes the nearest decoder palette entry (lowest index on
/// ties). Equal endpoints decode as 3-color mode (index 3 = black).
#[inline(always)]
fn exact<V: SimdF32>(t: &Texels<V>, e0: [V; 3], e1: [V; 3]) -> Coded<V> {
    let (q0, q1) = (quantize(e0), quantize(e1));
    let (p0, p1) = (pack565(&q0), pack565(&q1));
    let swap = p0.simd_lt(p1);
    let c0 = V::Int::select(swap, p1, p0);
    let c1 = V::Int::select(swap, p0, p1);
    let same = c0.simd_lt(c1 + V::Int::splat(1));
    let (half, third) = (V::splat(0.5), V::splat(1.0 / 3.0));
    let mut pal = [[V::splat(0.0); 3]; 4];
    let mut c = 0;
    while c < 3 {
        let a = V::select(swap, q1.v[c], q0.v[c]);
        let b = V::select(swap, q0.v[c], q1.v[c]);
        pal[0][c] = a;
        pal[1][c] = b;
        pal[2][c] = V::select(same, a, floor((a + a + b + half) * third));
        pal[3][c] = V::select(same, V::splat(0.0), floor((a + b + b + half) * third));
        c += 1;
    }
    let mut err = V::splat(0.0);
    let mut bits = V::Int::splat(0);
    let mut i = 0;
    while i < 16 {
        let d = [t[0][i] - pal[0][0], t[1][i] - pal[0][1], t[2][i] - pal[0][2]];
        let mut best = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let mut code = V::Int::splat(0);
        let mut k = 1;
        while k < 4 {
            let d = [t[0][i] - pal[k][0], t[1][i] - pal[k][1], t[2][i] - pal[k][2]];
            let e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let m = e.simd_lt(best);
            best = V::select(m, e, best);
            code = V::Int::select(m, V::Int::splat(k as i32), code);
            k += 1;
        }
        err += best;
        bits = bits | (code << (2 * i as i32));
        i += 1;
    }
    Coded { c0, c1, bits, err }
}

/// Texels of eight blocks sorted along an axis (each lane independently, with a sorting
/// network) and their per-channel prefix sums (`pre[ch][n]` = sum of the first `n`).
struct Sorted<V: SimdF32> {
    c: Texels<V>,
    pre: [[V; 17]; 3],
}

#[inline(always)]
fn sort_along<V: SimdF32>(t: &Texels<V>, axis: [V; 3]) -> Sorted<V> {
    let mut c = *t;
    let mut key = [V::splat(0.0); 16];
    let mut i = 0;
    while i < 16 {
        key[i] = c[0][i] * axis[0] + c[1][i] * axis[1] + c[2][i] * axis[2];
        i += 1;
    }
    let mut n = 0;
    while n < SORT16.len() {
        let (a, b) = (SORT16[n].0 as usize, SORT16[n].1 as usize);
        let m = key[b].simd_lt(key[a]);
        let (ka, kb) = (key[a], key[b]);
        key[a] = V::select(m, kb, ka);
        key[b] = V::select(m, ka, kb);
        let mut ch = 0;
        while ch < 3 {
            let (x, y) = (c[ch][a], c[ch][b]);
            c[ch][a] = V::select(m, y, x);
            c[ch][b] = V::select(m, x, y);
            ch += 1;
        }
        n += 1;
    }
    let zero = V::splat(0.0);
    let mut pre = [[zero; 17]; 3];
    let mut ch = 0;
    while ch < 3 {
        let mut i = 0;
        while i < 16 {
            pre[ch][i + 1] = pre[ch][i] + c[ch][i];
            i += 1;
        }
        ch += 1;
    }
    Sorted { c, pre }
}

/// Least-squares endpoints of each lane's chosen 4-color split (scalar, the same formulas as
/// the snapped scorer).
#[inline(always)]
fn split_endpoints<V: SimdF32>(srt: &Sorted<V>, best_id: V::Int) -> ([V; 3], [V; 3]) {
    let table = &super::split_tables().ordered[0];
    let ids = best_id.to_array();
    let mut pa = [[[0f32; LANES]; 17]; 3];
    let mut ch = 0;
    while ch < 3 {
        let mut m = 0;
        while m < 17 {
            pa[ch][m] = srt.pre[ch][m].to_array();
            m += 1;
        }
        ch += 1;
    }
    let mut e0 = [[0f32; LANES]; 3];
    let mut e1 = [[0f32; LANES]; 3];
    let mut l = 0;
    while l < LANES {
        let id = ids[l] as usize;
        let (si, sj, sk) = (table.i[id] as usize, table.j[id] as usize, table.k[id] as usize);
        let (a2, b2, ab, inv) = (table.a2[id], table.b2[id], table.ab[id], table.inv_det[id]);
        let mut ch = 0;
        while ch < 3 {
            let u = pa[ch][si][l] + pa[ch][sj][l] + pa[ch][sk][l];
            let t = pa[ch][16][l];
            e0[ch][l] = u * ((b2 + ab) * inv / 3.0) - t * (ab * inv);
            e1[ch][l] = t * (a2 * inv) - u * ((a2 + ab) * inv / 3.0);
            ch += 1;
        }
        l += 1;
    }
    (
        [V::from_array(e0[0]), V::from_array(e0[1]), V::from_array(e0[2])],
        [V::from_array(e1[0]), V::from_array(e1[1]), V::from_array(e1[2])],
    )
}

/// 565 snap of least-squares endpoint channels: round to the nearest level, clamp the level
/// (the same as clamping the input to 0..=255 first) and expand back to 8 bits.
#[inline(always)]
fn snap<V: SimdF32>(v: V, six_bits: bool) -> V {
    let (maxq, hi, lo) = if six_bits { (63, 2, 4) } else { (31, 3, 2) };
    let q = v
        .mul_add(V::splat(maxq as f32 / 255.0), V::splat(0.5))
        .to_int_trunc()
        .simd_max(V::Int::splat(0))
        .simd_min(V::Int::splat(maxq));
    ((q << hi) | (q >> lo)).to_float()
}

/// 4-color cluster fit scoring the first `limit` splits of the learned order with
/// 565-snapped endpoints.
///
/// The closed-form least-squares error of a split (see [`Closed4`]) bounds its snapped error
/// from below, so a split is only snapped when some lane's bound, plus `prune` times the
/// expected snapping penalty, beats that lane's best so far (`prune` = 0: the same result as
/// snapping every split).
#[inline(always)]
fn cluster_snapped<V: SimdF32, const K: usize>(t: &Texels<V>, srt: &Sorted<V>, limit: usize, prune: f32) -> Coded<V> {
    let pre = &srt.pre;
    let zero = V::splat(0.0);
    let total = [pre[0][16], pre[1][16], pre[2][16]];
    let m2t = [
        total[0] * V::splat(-2.0),
        total[1] * V::splat(-2.0),
        total[2] * V::splat(-2.0),
    ];
    let tt = total[0] * total[0] + total[1] * total[1] + total[2] * total[2];
    let c4 = closed4();
    let len = limit.min(c4.k0.len());
    // The K best splits per lane, best first.
    let mut best_err = [V::splat(f32::INFINITY); K];
    let mut best_id = [V::Int::splat(0); K];
    let mut n = 0;
    while n < len {
        let (si, sj, sk) = (c4.i[n] as usize, c4.j[n] as usize, c4.k[n] as usize);
        let u = [
            pre[0][si] + pre[0][sj] + pre[0][sk],
            pre[1][si] + pre[1][sj] + pre[1][sk],
            pre[2][si] + pre[2][sj] + pre[2][sk],
        ];
        // Both errors leave out the constant sum of squared texels: the bound is -gain.
        let ut = u[0] * total[0] + u[1] * total[1] + u[2] * total[2];
        let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        let gain = V::splat(c4.k0[n]).mul_add(tt, V::splat(c4.k1[n]).mul_add(ut, V::splat(c4.k2[n]) * uu));
        let bound = V::splat(prune * c4.kp[n]) - gain;
        // A small margin keeps float rounding from pruning a split that ties.
        if V::mask_any(bound.simd_lt(best_err[K - 1] + V::splat(0.5))) {
            // ax = u / 3, bx = T - ax; a = u ka - T kat, b = T kbt - u kb. The error per
            // channel is a (a a2 - 2 ax) + b (b b2 - 2 bx) + 2 ab a b.
            let (ka, kat, kb, kbt) = (
                V::splat(c4.ka[n]),
                V::splat(c4.kat[n]),
                V::splat(c4.kb[n]),
                V::splat(c4.kbt[n]),
            );
            let (va2, vb2, vab2) = (V::splat(c4.a2[n]), V::splat(c4.b2[n]), V::splat(c4.ab2[n]));
            let mut err = zero;
            let mut ch = 0;
            while ch < 3 {
                let a = snap(u[ch] * ka - total[ch] * kat, ch == 1);
                let b = snap(total[ch] * kbt - u[ch] * kb, ch == 1);
                let nax2 = u[ch] * V::splat(-2.0 / 3.0);
                let nbx2 = m2t[ch] - nax2;
                err = a.mul_add(a.mul_add(va2, nax2), err);
                err = b.mul_add(b.mul_add(vb2, nbx2), err);
                err = (a * b).mul_add(vab2, err);
                ch += 1;
            }
            // Insert into the sorted top K.
            let (mut e, mut id) = (err, V::Int::splat(n as i32));
            let mut k = 0;
            while k < K {
                let better = e.simd_lt(best_err[k]);
                let (oe, oid) = (best_err[k], best_id[k]);
                best_err[k] = V::select(better, e, oe);
                best_id[k] = V::Int::select(better, id, oid);
                e = V::select(better, oe, e);
                id = V::Int::select(better, oid, id);
                k += 1;
            }
        }
        n += 1;
    }
    let (e0, e1) = split_endpoints(srt, best_id[0]);
    let mut out = exact(t, e0, e1);
    let mut k = 1;
    while k < K {
        let (e0, e1) = split_endpoints(srt, best_id[k]);
        keep_better(&mut out, exact(t, e0, e1));
        k += 1;
    }
    out
}

/// Per-split constants of the closed-form 4-color score, in the learned order: with
/// u = P[i] + P[j] + P[k] and T the channel totals, the error the least-squares endpoints
/// remove is `k0 |T|² + k1 (u·T) + k2 |u|²` (endpoint clamping and 565 snapping ignored).
struct Closed4 {
    i: Vec<u8>,
    j: Vec<u8>,
    k: Vec<u8>,
    k0: Vec<f32>,
    k1: Vec<f32>,
    k2: Vec<f32>,
    /// Expected snapping penalty: (a2 + b2) times the mean squared 565 rounding error summed
    /// over the channels (step² / 12 for 5, 6 and 5 bits).
    kp: Vec<f32>,
    /// Snapped scoring: a = u ka - T kat, b = T kbt - u kb; a2, b2 and 2 ab.
    ka: Vec<f32>,
    kat: Vec<f32>,
    kb: Vec<f32>,
    kbt: Vec<f32>,
    a2: Vec<f32>,
    b2: Vec<f32>,
    ab2: Vec<f32>,
}

fn closed4() -> &'static Closed4 {
    static T: std::sync::OnceLock<Closed4> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let t = &super::split_tables().ordered[0];
        let n = t.len;
        let mut c = Closed4 {
            i: Vec::with_capacity(n),
            j: Vec::with_capacity(n),
            k: Vec::with_capacity(n),
            k0: Vec::with_capacity(n),
            k1: Vec::with_capacity(n),
            k2: Vec::with_capacity(n),
            kp: Vec::with_capacity(n),
            ka: Vec::with_capacity(n),
            kat: Vec::with_capacity(n),
            kb: Vec::with_capacity(n),
            kbt: Vec::with_capacity(n),
            a2: Vec::with_capacity(n),
            b2: Vec::with_capacity(n),
            ab2: Vec::with_capacity(n),
        };
        let step2 = |levels: f32| (255.0 / levels) * (255.0 / levels) / 12.0;
        let snap_var = 2.0 * step2(31.0) + step2(63.0);
        for s in 0..n {
            // Removed error = (b2 |ax|² + a2 |bx|² - 2 ab ax·bx) / det, with ax = u / 3 and
            // bx = T - ax.
            let (a, b, cc) = (
                t.a2[s] * t.inv_det[s],
                t.b2[s] * t.inv_det[s],
                2.0 * t.ab[s] * t.inv_det[s],
            );
            c.i.push(t.i[s] as u8);
            c.j.push(t.j[s] as u8);
            c.k.push(t.k[s] as u8);
            c.k0.push(a);
            c.k1.push(-(2.0 * a + cc) / 3.0);
            c.k2.push((a + b + cc) / 9.0);
            c.kp.push((t.a2[s] + t.b2[s]) * snap_var);
            let (a2, b2, ab, inv) = (t.a2[s], t.b2[s], t.ab[s], t.inv_det[s]);
            c.ka.push((b2 + ab) * inv / 3.0);
            c.kat.push(ab * inv);
            c.kb.push((a2 + ab) * inv / 3.0);
            c.kbt.push(a2 * inv);
            c.a2.push(a2);
            c.b2.push(b2);
            c.ab2.push(2.0 * ab);
        }
        c
    })
}

/// 4-color cluster fit scoring the first `limit` splits of the learned order with the
/// closed-form least-squares error, then snapping only each lane's winner.
#[inline(always)]
fn cluster_closed<V: SimdF32>(t: &Texels<V>, srt: &Sorted<V>, limit: usize) -> Coded<V> {
    let pre = &srt.pre;
    let total = [pre[0][16], pre[1][16], pre[2][16]];
    let tt = total[0] * total[0] + total[1] * total[1] + total[2] * total[2];
    let c = closed4();
    let len = limit.min(c.k0.len());
    let mut best = V::splat(0.0);
    let mut best_id = V::Int::splat(0);
    let mut n = 0;
    while n < len {
        let (si, sj, sk) = (c.i[n] as usize, c.j[n] as usize, c.k[n] as usize);
        let u0 = pre[0][si] + pre[0][sj] + pre[0][sk];
        let u1 = pre[1][si] + pre[1][sj] + pre[1][sk];
        let u2 = pre[2][si] + pre[2][sj] + pre[2][sk];
        let ut = u0 * total[0] + u1 * total[1] + u2 * total[2];
        let uu = u0 * u0 + u1 * u1 + u2 * u2;
        let gain = V::splat(c.k0[n]) * tt + V::splat(c.k1[n]) * ut + V::splat(c.k2[n]) * uu;
        let better = best.simd_lt(gain);
        best = V::select(better, gain, best);
        best_id = V::Int::select(better, V::Int::splat(n as i32), best_id);
        n += 1;
    }
    let (e0, e1) = split_endpoints(srt, best_id);
    exact(t, e0, e1)
}

/// Texels at or below this in every channel may take 3-color mode's black entry.
const DARK: f32 = super::DARK_MAX as f32 + 0.5;

/// 3-color cluster fit over every split, with dark texels left out of the fit (they take the
/// black entry). Lanes without dark texels get the plain 3-color fit. Returns the exact
/// 3-color encoding of each lane's best split.
#[inline(always)]
fn cluster3<V: SimdF32>(t: &Texels<V>, srt: &Sorted<V>) -> Coded<V> {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let dark = V::splat(DARK);
    // Weighted prefix sums over the sorted texels: weight 0 for dark texels.
    let mut pw = [zero; 17];
    let mut px = [[zero; 17]; 3];
    let mut i = 0;
    while i < 16 {
        let c = &srt.c;
        let lit = dark.simd_lt(c[0][i].simd_max(c[1][i]).simd_max(c[2][i]));
        let w = V::select(lit, one, zero);
        pw[i + 1] = pw[i] + w;
        let mut ch = 0;
        while ch < 3 {
            px[ch][i + 1] = px[ch][i] + w * c[ch][i];
            ch += 1;
        }
        i += 1;
    }
    let (wt, tot) = (pw[16], [px[0][16], px[1][16], px[2][16]]);
    let tt = tot[0] * tot[0] + tot[1] * tot[1] + tot[2] * tot[2];
    let quarter = V::splat(0.25);
    // Best split per lane as a fraction num / den (den = det > 0), compared by
    // cross-multiplication.
    let (mut bnum, mut bden) = (zero, one);
    let (mut bi, mut bj) = (V::Int::splat(0), V::Int::splat(0));
    let mut si = 0;
    while si <= 16 {
        let mut sj = si;
        while sj <= 16 {
            // Runs: [0, si) all a, [si, sj) half and half, [sj, 16) all b. With
            // v = P[si] + P[sj]: ax = v / 2, bx = T - ax, and the error the least-squares
            // endpoints remove is (b2 |ax|² + a2 |bx|² - 2 ab ax·bx) / det
            // = ((a2 + b2 + 2 ab) |v|² / 4 - (a2 + ab) v·T + a2 |T|²) / det.
            let q = (pw[sj] - pw[si]) * quarter;
            let a2 = pw[si] + q;
            let b2 = wt - pw[sj] + q;
            let det = a2 * b2 - q * q;
            let v0 = px[0][si] + px[0][sj];
            let v1 = px[1][si] + px[1][sj];
            let v2 = px[2][si] + px[2][sj];
            let vv = v0 * v0 + v1 * v1 + v2 * v2;
            let vt = v0 * tot[0] + v1 * tot[1] + v2 * tot[2];
            let num = (a2 + b2 + q + q) * quarter * vv - (a2 + q) * vt + a2 * tt;
            let better = V::splat(0.25).simd_lt(det) & (bnum * det).simd_lt(num * bden);
            bnum = V::select(better, num, bnum);
            bden = V::select(better, det, bden);
            bi = V::Int::select(better, V::Int::splat(si as i32), bi);
            bj = V::Int::select(better, V::Int::splat(sj as i32), bj);
            sj += 1;
        }
        si += 1;
    }
    // Endpoints of each lane's split (lanes without one keep black).
    let (bi, bj) = (bi.to_array(), bj.to_array());
    let mut pa = [[[0f32; LANES]; 17]; 4];
    let mut m = 0;
    while m < 17 {
        pa[0][m] = px[0][m].to_array();
        pa[1][m] = px[1][m].to_array();
        pa[2][m] = px[2][m].to_array();
        pa[3][m] = pw[m].to_array();
        m += 1;
    }
    let mut ea = [[0f32; LANES]; 3];
    let mut eb = [[0f32; LANES]; 3];
    let mut l = 0;
    while l < LANES {
        let (i, j) = (bi[l] as usize, bj[l] as usize);
        let q = (pa[3][j][l] - pa[3][i][l]) * 0.25;
        let a2 = pa[3][i][l] + q;
        let b2 = pa[3][16][l] - pa[3][j][l] + q;
        let det = a2 * b2 - q * q;
        if det > 0.25 {
            let inv = 1.0 / det;
            let mut ch = 0;
            while ch < 3 {
                let ax = (pa[ch][i][l] + pa[ch][j][l]) * 0.5;
                let bx = pa[ch][16][l] - ax;
                ea[ch][l] = (ax * b2 - bx * q) * inv;
                eb[ch][l] = (bx * a2 - ax * q) * inv;
                ch += 1;
            }
        }
        l += 1;
    }
    exact3(
        t,
        [V::from_array(ea[0]), V::from_array(ea[1]), V::from_array(ea[2])],
        [V::from_array(eb[0]), V::from_array(eb[1]), V::from_array(eb[2])],
    )
}

/// Exact 3-color encoding: endpoints in 3-color order (c0 <= c1), palette a, b, (a + b) / 2 and
/// black; each texel takes the nearest entry (lowest index on ties).
#[inline(always)]
fn exact3<V: SimdF32>(t: &Texels<V>, e0: [V; 3], e1: [V; 3]) -> Coded<V> {
    let (q0, q1) = (quantize(e0), quantize(e1));
    let (p0, p1) = (pack565(&q0), pack565(&q1));
    let swap = p1.simd_lt(p0);
    let c0 = V::Int::select(swap, p1, p0);
    let c1 = V::Int::select(swap, p0, p1);
    let half = V::splat(0.5);
    let mut pal = [[V::splat(0.0); 3]; 4];
    let mut c = 0;
    while c < 3 {
        let a = V::select(swap, q1.v[c], q0.v[c]);
        let b = V::select(swap, q0.v[c], q1.v[c]);
        pal[0][c] = a;
        pal[1][c] = b;
        pal[2][c] = floor((a + b) * half);
        c += 1;
    }
    let mut err = V::splat(0.0);
    let mut bits = V::Int::splat(0);
    let mut i = 0;
    while i < 16 {
        let mut best = V::splat(f32::INFINITY);
        let mut code = V::Int::splat(0);
        let mut k = 0;
        while k < 4 {
            let d = [t[0][i] - pal[k][0], t[1][i] - pal[k][1], t[2][i] - pal[k][2]];
            let e = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let m = e.simd_lt(best);
            best = V::select(m, e, best);
            code = V::Int::select(m, V::Int::splat(k as i32), code);
            k += 1;
        }
        err += best;
        bits = bits | (code << (2 * i as i32));
        i += 1;
    }
    Coded { c0, c1, bits, err }
}

/// Endpoint moves tried by [`refine`] (565 level steps per channel), as in icbc's refinement.
const MOVES: [[i8; 3]; 16] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [-1, -1, 0],
    [-1, 0, -1],
    [0, -1, -1],
    [-1, 1, 0],
    [1, -1, 0],
    [0, -1, 1],
    [0, 1, -1],
];

/// 565 levels of a packed color, per channel (as floats).
#[inline(always)]
fn unpack_q<V: SimdF32>(c: V::Int) -> [V; 3] {
    [
        (c >> 11).to_float(),
        ((c >> 5) & V::Int::splat(63)).to_float(),
        (c & V::Int::splat(31)).to_float(),
    ]
}

/// Packed 565 value of float levels.
#[inline(always)]
fn pack_q<V: SimdF32>(q: &[V; 3]) -> V {
    q[0].mul_add(V::splat(2048.0), q[1].mul_add(V::splat(32.0), q[2]))
}

/// Bit-replicated 8-bit value of a 5- or 6-bit level.
#[inline(always)]
fn expand<V: SimdF32>(q: V, six: bool) -> V {
    if six {
        q.mul_add(V::splat(4.0), floor(q * V::splat(1.0 / 16.0)))
    } else {
        q.mul_add(V::splat(8.0), floor(q * V::splat(0.25)))
    }
}

/// Exact error and index bits of endpoints given as levels (already in the lane's mode order:
/// c0 > c1 for 4-color, c0 <= c1 for 3-color lanes).
#[inline(always)]
fn eval_levels<V: SimdF32>(t: &Texels<V>, q0: &[V; 3], q1: &[V; 3], three: V::Mask) -> (V, V::Int) {
    let (half, third) = (V::splat(0.5), V::splat(1.0 / 3.0));
    let mut pal = [[V::splat(0.0); 3]; 4];
    let mut c = 0;
    while c < 3 {
        let a = expand(q0[c], c == 1);
        let b = expand(q1[c], c == 1);
        pal[0][c] = a;
        pal[1][c] = b;
        pal[2][c] = V::select(three, floor((a + b) * half), floor((a + a + b + half) * third));
        pal[3][c] = V::select(three, V::splat(0.0), floor((a + b + b + half) * third));
        c += 1;
    }
    let mut err = V::splat(0.0);
    let mut bits = V::Int::splat(0);
    let mut i = 0;
    while i < 16 {
        let mut best = V::splat(f32::INFINITY);
        let mut code = V::Int::splat(0);
        let mut k = 0;
        while k < 4 {
            let d0 = t[0][i] - pal[k][0];
            let d1 = t[1][i] - pal[k][1];
            let d2 = t[2][i] - pal[k][2];
            let e = d0.mul_add(d0, d1.mul_add(d1, d2 * d2));
            let m = e.simd_lt(best);
            best = V::select(m, e, best);
            code = V::Int::select(m, V::Int::splat(k as i32), code);
            k += 1;
        }
        err += best;
        bits = bits | (code << (2 * i as i32));
        i += 1;
    }
    (err, bits)
}

/// Greedy endpoint refinement: up to `sweeps` passes over [`MOVES`] applied to each endpoint,
/// each move kept per lane when it lowers the exact error. Lanes keep their 3- or 4-color mode.
#[inline(always)]
fn refine<V: SimdF32>(t: &Texels<V>, cur: &mut Coded<V>, sweeps: u8, moves: usize) {
    // 3-color lanes are those with c0 <= c1.
    let three = cur.c0.simd_lt(cur.c1 + V::Int::splat(1));
    let mut q0 = unpack_q::<V>(cur.c0);
    let mut q1 = unpack_q::<V>(cur.c1);
    let mut err = cur.err;
    let mut bits = cur.bits;
    let hi = [V::splat(31.0), V::splat(63.0), V::splat(31.0)];
    let zero = V::splat(0.0);
    let mut sweep = 0;
    while sweep < sweeps {
        let mut improved = V::splat(0.0).simd_lt(zero);
        let mut m = 0;
        while m < 16 * moves {
            // Groups of 16: move c0, move c1, shift both, move them apart (or together), then
            // double steps of c0 and c1.
            let mv = MOVES[m % 16];
            let (sa, sb) = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0), (0.0, 2.0)][m / 16];
            let mut a = q0;
            let mut b = q1;
            let mut c = 0;
            while c < 3 {
                let d = mv[c] as f32;
                a[c] = (a[c] + V::splat(d * sa)).simd_clamp(zero, hi[c]);
                b[c] = (b[c] + V::splat(d * sb)).simd_clamp(zero, hi[c]);
                c += 1;
            }
            // Mode order: swap so c0 > c1 (4-color) or c0 <= c1 (3-color); equal endpoints
            // cannot stay in 4-color mode.
            let (pa, pb) = (pack_q(&a), pack_q(&b));
            let swap = V::select(three, pb, pa).simd_lt(V::select(three, pa, pb));
            let mut c = 0;
            while c < 3 {
                let (x, y) = (a[c], b[c]);
                a[c] = V::select(swap, y, x);
                b[c] = V::select(swap, x, y);
                c += 1;
            }
            let valid = three | pb.simd_lt(pa) | pa.simd_lt(pb);
            let (e, bt) = eval_levels(t, &a, &b, three);
            let take = valid & e.simd_lt(err);
            if V::mask_any(take) {
                improved = improved | take;
                err = V::select(take, e, err);
                bits = V::Int::select(take, bt, bits);
                let mut c = 0;
                while c < 3 {
                    q0[c] = V::select(take, a[c], q0[c]);
                    q1[c] = V::select(take, b[c], q1[c]);
                    c += 1;
                }
            }
            m += 1;
        }
        if !V::mask_any(improved) {
            break;
        }
        sweep += 1;
    }
    cur.err = err;
    cur.bits = bits;
    cur.c0 = pack_q(&q0).to_int_trunc();
    cur.c1 = pack_q(&q1).to_int_trunc();
}

/// Least-squares refits from the current indices (weights 0, 1, 1/3, 2/3 toward c1 in 4-color
/// mode; 0, 1, 1/2 and black texels left out in 3-color mode), each kept per lane when it lowers
/// the exact error.
#[inline(always)]
fn polish<V: SimdF32>(t: &Texels<V>, cur: &mut Coded<V>, iters: u8) {
    let three = cur.c0.simd_lt(cur.c1 + V::Int::splat(1));
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let hi = [V::splat(31.0), V::splat(63.0), V::splat(31.0)];
    let scale = [V::splat(31.0 / 255.0), V::splat(63.0 / 255.0), V::splat(31.0 / 255.0)];
    let mut it = 0;
    while it < iters {
        let (mut aa, mut ab, mut bb) = (zero, zero, zero);
        let mut ax = [zero; 3];
        let mut bx = [zero; 3];
        let mut i = 0;
        while i < 16 {
            let code = (cur.bits >> (2 * i as i32)) & V::Int::splat(3);
            let wb = V::select(
                three,
                V::Int::lookup16(&W3_BITS, code).to_float() * V::splat(0.5),
                V::Int::lookup16(&W4_BITS, code).to_float() * V::splat(1.0 / 3.0),
            );
            // Black texels (3-color index 3) do not constrain the endpoints.
            let used = V::select(three & V::Int::splat(2).simd_lt(code), zero, one);
            let wa = (one - wb) * used;
            let wb = wb * used;
            aa = wa.mul_add(wa, aa);
            ab = wa.mul_add(wb, ab);
            bb = wb.mul_add(wb, bb);
            let mut c = 0;
            while c < 3 {
                ax[c] = wa.mul_add(t[c][i], ax[c]);
                bx[c] = wb.mul_add(t[c][i], bx[c]);
                c += 1;
            }
            i += 1;
        }
        let det = aa * bb - ab * ab;
        let ok = V::splat(1e-3).simd_lt(det);
        let inv = one / V::select(ok, det, one);
        let mut q0 = [zero; 3];
        let mut q1 = [zero; 3];
        let mut c = 0;
        while c < 3 {
            let e0 = (bb * ax[c] - ab * bx[c]) * inv;
            let e1 = (aa * bx[c] - ab * ax[c]) * inv;
            q0[c] = floor(e0.mul_add(scale[c], V::splat(0.5)).simd_max(zero)).simd_min(hi[c]);
            q1[c] = floor(e1.mul_add(scale[c], V::splat(0.5)).simd_max(zero)).simd_min(hi[c]);
            c += 1;
        }
        let (pa, pb) = (pack_q(&q0), pack_q(&q1));
        let swap = V::select(three, pb, pa).simd_lt(V::select(three, pa, pb));
        let mut c = 0;
        while c < 3 {
            let (x, y) = (q0[c], q1[c]);
            q0[c] = V::select(swap, y, x);
            q1[c] = V::select(swap, x, y);
            c += 1;
        }
        let valid = ok & (three | pb.simd_lt(pa) | pa.simd_lt(pb));
        let (e, bt) = eval_levels(t, &q0, &q1, three);
        let take = valid & e.simd_lt(cur.err);
        if !V::mask_any(take) {
            break;
        }
        cur.err = V::select(take, e, cur.err);
        cur.bits = V::Int::select(take, bt, cur.bits);
        cur.c0 = V::Int::select(take, pack_q(&q0).to_int_trunc(), cur.c0);
        cur.c1 = V::Int::select(take, pack_q(&q1).to_int_trunc(), cur.c1);
        it += 1;
    }
}

/// Index -> weight toward c1, in thirds (4-color) and halves (3-color), for `lookup16`.
const W4_BITS: [u8; 16] = [0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const W3_BITS: [u8; 16] = [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Keeps `cand` in the lanes where it has the lower error.
#[inline(always)]
fn keep_better<V: SimdF32>(cur: &mut Coded<V>, cand: Coded<V>) {
    let take = cand.err.simd_lt(cur.err);
    cur.err = V::select(take, cand.err, cur.err);
    cur.c0 = V::Int::select(take, cand.c0, cur.c0);
    cur.c1 = V::Int::select(take, cand.c1, cur.c1);
    cur.bits = V::Int::select(take, cand.bits, cur.bits);
}

/// Palette index of each projection step: step 0 is c0 (index 0), step 3 is c1 (index 1).
const STEP_INDEX: [u8; 16] = [0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Encodes eight blocks. Returns 64 bytes in block order (lanes are in block order unless
/// `row_order`, see [`crate::simd::ROW_ORDER`]) and, per block, whether it is solid (every
/// texel equal in RGB; the caller re-encodes those with the single-color tables).
#[inline(always)]
fn encode8<V: SimdF32>(t: &Texels<V>, cfg: Cfg, row_order: bool) -> ([u8; 64], [bool; LANES]) {
    // Raw sums of integer texels and their products are exact in f32 (at most 16 * 255^2), so
    // the order of summation does not matter: four chains keep the dependency depth short.
    let sum = [sum16!(|i| t[0][i]), sum16!(|i| t[1][i]), sum16!(|i| t[2][i])];
    let inv16 = V::splat(1.0 / 16.0);
    let mean = [sum[0] * inv16, sum[1] * inv16, sum[2] * inv16];
    // Centered moments: sum(xy) - sum(x) sum(y) / 16 (exact: multiples of 1/16 below 2^20).
    let rr = sum16!(|i| t[0][i] * t[0][i]) - sum[0] * mean[0];
    let rg = sum16!(|i| t[0][i] * t[1][i]) - sum[0] * mean[1];
    let rb = sum16!(|i| t[0][i] * t[2][i]) - sum[0] * mean[2];
    let gg = sum16!(|i| t[1][i] * t[1][i]) - sum[1] * mean[1];
    let gb = sum16!(|i| t[1][i] * t[2][i]) - sum[1] * mean[2];
    let bb = sum16!(|i| t[2][i] * t[2][i]) - sum[2] * mean[2];
    // Integer texels make the covariance exact: a zero trace means a solid block.
    let solid_mask = (rr + gg + bb).simd_lt(V::splat(0.5));
    // Start from the largest-diagonal row, one power iteration.
    let mut v = [rr, rg, rb];
    let mut best = rr;
    let m = best.simd_lt(gg);
    v = [V::select(m, rg, v[0]), V::select(m, gg, v[1]), V::select(m, gb, v[2])];
    best = best.simd_max(gg);
    let m = best.simd_lt(bb);
    v = [V::select(m, rb, v[0]), V::select(m, gb, v[1]), V::select(m, bb, v[2])];
    let mut w = [
        rr * v[0] + rg * v[1] + rb * v[2],
        rg * v[0] + gg * v[1] + gb * v[2],
        rb * v[0] + gb * v[1] + bb * v[2],
    ];
    let n2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    // Degenerate covariance (solid blocks; patched below): any axis will do.
    let flat = n2.simd_lt(V::splat(1e-12));
    w = [
        V::select(flat, V::splat(1.0), w[0]),
        V::select(flat, V::splat(1.0), w[1]),
        V::select(flat, V::splat(1.0), w[2]),
    ];
    let n2 = V::select(flat, V::splat(3.0), n2);
    // Projections onto w relative to the mean (the mean's projection is subtracted once).
    macro_rules! proj {
        ($i:expr) => {
            t[0][$i] * w[0] + t[1][$i] * w[1] + t[2][$i] * w[2]
        };
    }
    let mw = mean[0] * w[0] + mean[1] * w[1] + mean[2] * w[2];
    let tmin = reduce16!(|i| proj!(i), |a, b| a.simd_min(b)) - mw;
    let tmax = reduce16!(|i| proj!(i), |a, b| a.simd_max(b)) - mw;
    let inset = (tmax - tmin) * V::splat(1.0 / 16.0);
    let (lo, hi) = ((tmin + inset) / n2, (tmax - inset) / n2);
    let e_hi = [mean[0] + w[0] * hi, mean[1] + w[1] * hi, mean[2] + w[2] * hi];
    let e_lo = [mean[0] + w[0] * lo, mean[1] + w[1] * lo, mean[2] + w[2] * lo];
    let mut fit = assign(t, e_hi, e_lo);
    let mut err = V::splat(0.0);

    if cfg.refits == 0 {
        let (e0, e1, ok) = refit(t, &sum, &fit);
        let next = assign(t, e0, e1);
        fit = choose(ok, next, fit);
        if cfg.splits > 0 || cfg.three {
            err = error(t, &fit);
        }
    } else {
        err = error(t, &fit);
        let mut it = 0;
        while it < cfg.refits {
            let (e0, e1, ok) = refit(t, &sum, &fit);
            let next = assign(t, e0, e1);
            let e = error(t, &next);
            let take = ok & e.simd_lt(err);
            if !V::mask_any(take) {
                break;
            }
            err = V::select(take, e, err);
            fit = choose(take, next, fit);
            it += 1;
        }
    }

    // Equal endpoints decode every index to c0; keep the bits 0 there.
    let same = fit.c0.simd_lt(fit.c1 + V::Int::splat(1));
    let mut bits = V::Int::splat(0);
    let mut i = 0;
    while i < 16 {
        let code = V::Int::lookup16(&STEP_INDEX, fit.step[i].to_int_trunc());
        bits = bits | (code << (2 * i as i32));
        i += 1;
    }
    bits = V::Int::select(same, V::Int::splat(0), bits);
    let (mut c0, mut c1) = (fit.c0, fit.c1);
    if cfg.splits > 0 || cfg.three {
        // A better axis for sorting: three more power iterations (normalized each time).
        let mut ax = w;
        let mut it = 0;
        while it < 3 {
            let n = [
                rr * ax[0] + rg * ax[1] + rb * ax[2],
                rg * ax[0] + gg * ax[1] + gb * ax[2],
                rb * ax[0] + gb * ax[1] + bb * ax[2],
            ];
            let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            let ok = V::splat(1e-20).simd_lt(len2);
            let inv = V::splat(1.0) / V::select(ok, len2, V::splat(1.0)).simd_sqrt();
            ax = [
                V::select(ok, n[0] * inv, ax[0]),
                V::select(ok, n[1] * inv, ax[1]),
                V::select(ok, n[2] * inv, ax[2]),
            ];
            it += 1;
        }
        let mut cur = Coded { c0, c1, bits, err };
        let mut round = 0;
        while round < cfg.rounds.max(1) {
            if round > 0 {
                // Sort along the current best endpoints' direction (lanes with equal
                // endpoints keep the principal axis).
                let (a, b) = (unpack_q::<V>(cur.c0), unpack_q::<V>(cur.c1));
                let d = [
                    expand(b[0], false) - expand(a[0], false),
                    expand(b[1], true) - expand(a[1], true),
                    expand(b[2], false) - expand(a[2], false),
                ];
                let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let ok = V::splat(0.5).simd_lt(len2);
                let mut c = 0;
                while c < 3 {
                    ax[c] = V::select(ok, d[c], ax[c]);
                    c += 1;
                }
            }
            let srt = sort_along(t, ax);
            if cfg.splits > 0 {
                let cf = if cfg.closed {
                    cluster_closed(t, &srt, cfg.splits)
                } else {
                    match cfg.topk {
                        0 | 1 => cluster_snapped::<V, 1>(t, &srt, cfg.splits, cfg.prune),
                        2 => cluster_snapped::<V, 2>(t, &srt, cfg.splits, cfg.prune),
                        _ => cluster_snapped::<V, 4>(t, &srt, cfg.splits, cfg.prune),
                    }
                };
                keep_better(&mut cur, cf);
            }
            if cfg.three {
                keep_better(&mut cur, cluster3(t, &srt));
            }
            if cfg.polish > 0 {
                polish(t, &mut cur, cfg.polish);
            }
            if cfg.refine > 0 && (cfg.refine_each || round + 1 == cfg.rounds.max(1)) {
                refine(t, &mut cur, cfg.refine, cfg.moves);
            }
            round += 1;
        }
        (c0, c1, bits) = (cur.c0, cur.c1, cur.bits);
    }
    let mut w0 = c0 | (c1 << 16);
    let mut w1 = bits;
    if row_order {
        const UNROW: [i32; LANES] = [0, 4, 1, 5, 2, 6, 3, 7];
        w0 = w0.permute(UNROW);
        w1 = w1.permute(UNROW);
    }
    let (p, q) = w0.interleave(w1);
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&p.to_le_bytes());
    out[32..].copy_from_slice(&q.to_le_bytes());
    let lanes = V::mask_to_bools(solid_mask);
    let mut solid = [false; LANES];
    let mut l = 0;
    while l < LANES {
        solid[if row_order { crate::simd::ROW_ORDER[l] } else { l }] = lanes[l];
        l += 1;
    }
    (out, solid)
}

/// Re-encodes the solid blocks of a batch (blocks `k..`) with the single-color tables.
fn patch_solid(s: &Surface, k: usize, o: &mut [u8], solid: &[bool; LANES]) {
    let w = crate::format::Format::Bc1.error_weights();
    let bw = s.blocks_x() as usize;
    for (l, ob) in o.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        if solid[l] {
            let b = s.read_block(((k + l) % bw) as u32, ((k + l) / bw) as u32);
            ob.copy_from_slice(&super::encode_solid(&b, &w).0);
        }
    }
}

/// Encodes blocks `first..` of a surface into `out` (8 bytes per block).
#[inline(always)]
fn encode_span<V: SimdF32>(s: &Surface, first: usize, out: &mut [u8], cfg: Cfg) {
    let bw = s.blocks_x() as usize;
    let (w, h) = (s.width as usize, s.height as usize);
    let mut k = first;
    for o in out.chunks_mut(LANES * 8) {
        let n = o.len() / 8;
        let (bx, by) = (k % bw, k / bw);
        if n == LANES && (bx + LANES) * 4 <= w && by * 4 + 4 <= h {
            let mut px = [V::Int::splat(0); 16];
            let mut r = 0;
            while r < 4 {
                let at = (by * 4 + r) * w + bx * 4;
                let row: &[[u8; 4]; 32] = s.pixels[at..at + 32].try_into().unwrap();
                let cols = V::Int::load_block_row(row);
                px[4 * r] = cols[0];
                px[4 * r + 1] = cols[1];
                px[4 * r + 2] = cols[2];
                px[4 * r + 3] = cols[3];
                r += 1;
            }
            let mut t = [[V::splat(0.0); 16]; 3];
            let mut i = 0;
            while i < 16 {
                t[0][i] = (px[i] & V::Int::splat(0xff)).to_float();
                t[1][i] = ((px[i] >> 8) & V::Int::splat(0xff)).to_float();
                t[2][i] = ((px[i] >> 16) & V::Int::splat(0xff)).to_float();
                i += 1;
            }
            let (enc, solid) = encode8(&t, cfg, true);
            o.copy_from_slice(&enc);
            if solid.iter().any(|&x| x) {
                patch_solid(s, k, o, &solid);
            }
        } else {
            let mut t = [[V::splat(0.0); 16]; 3];
            let mut v = [[[0f32; LANES]; 16]; 3];
            for l in 0..n {
                let i = (k + l) as u32;
                let b = s.read_block(i % bw as u32, i / bw as u32);
                for (j, px) in b.iter().enumerate() {
                    for c in 0..3 {
                        v[c][j][l] = px[c] as f32;
                    }
                }
            }
            for c in 0..3 {
                for j in 0..16 {
                    t[c][j] = V::from_array(v[c][j]);
                }
            }
            let (enc, solid) = encode8(&t, cfg, false);
            o.copy_from_slice(&enc[..o.len()]);
            patch_solid(s, k, o, &solid);
        }
        k += n;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode_span_avx2(s: &Surface, first: usize, out: &mut [u8], cfg: Cfg) {
    encode_span::<crate::simd::avx2::F>(s, first, out, cfg);
}

/// [`encode_span_avx2`] with effort 0's configuration as a constant, so the search code it
/// never runs drops out of the instantiation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode_span_avx2_e0(s: &Surface, first: usize, out: &mut [u8]) {
    encode_span::<crate::simd::avx2::F>(s, first, out, EFFORT0);
}

/// Effort 0: the line fit and one least-squares refit.
const EFFORT0: Cfg = Cfg {
    refits: 0,
    splits: 0,
    closed: false,
    three: false,
    prune: 0.0,
    refine: 0,
    rounds: 1,
    moves: 2,
    refine_each: false,
    polish: 0,
    topk: 1,
};

/// Encodes a whole surface as BC1 (see the module docs), reading texels straight from the
/// surface.
/// Returns raw 8-byte blocks in raster order.
pub fn encode_surface(surface: &Surface, effort: u8) -> Vec<u8> {
    encode_surface_cfg(surface, cfg(effort))
}

/// [`encode_surface`] with an explicit search configuration.
pub(super) fn encode_surface_cfg(surface: &Surface, cfg: Cfg) -> Vec<u8> {
    use rayon::prelude::*;
    const SPAN: usize = 512;
    let mut out = vec![0u8; surface.block_count() * 8];
    let lvl = level();
    out.par_chunks_mut(SPAN * 8).enumerate().for_each(|(ci, o)| match lvl {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 if is_effort0(&cfg) => unsafe { encode_span_avx2_e0(surface, ci * SPAN, o) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Level::Avx2 => unsafe { encode_span_avx2(surface, ci * SPAN, o, cfg) },
        _ => encode_span::<f32x8>(surface, ci * SPAN, o, cfg),
    });
    out
}

/// Whether `cfg` searches exactly what effort 0 does.
#[cfg(target_arch = "x86_64")]
fn is_effort0(cfg: &Cfg) -> bool {
    cfg.refits == 0 && cfg.splits == 0 && !cfg.three && cfg.polish == 0 && cfg.refine == 0
}

/// Encodes one block (the batch encoder on a one-block surface).
pub fn encode_block(px: &[[u8; 4]; 16], effort: u8) -> [u8; 8] {
    let mut s = Surface::new(4, 4);
    s.pixels.copy_from_slice(px);
    let raw = encode_surface(&s, effort);
    raw[..8].try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psnr(s: &Surface, raw: &[u8]) -> f64 {
        let bw = s.blocks_x();
        let mut sse = 0u64;
        let mut d = [[0u8; 4]; 16];
        for (i, b) in raw.as_chunks::<8>().0.iter().enumerate() {
            super::super::decode(b, &mut d);
            let src = s.read_block(i as u32 % bw, i as u32 / bw);
            sse += src
                .iter()
                .zip(&d)
                .map(|(a, b)| crate::surface::texel_sse(a, b, &[1, 1, 1, 0]) as u64)
                .sum::<u64>();
        }
        10.0 * (255.0f64 * 255.0 * (s.block_count() * 48) as f64 / sse.max(1) as f64).log10()
    }

    #[test]
    fn backends_agree_and_quality_is_sane() {
        // 70 x 66: runs of eight cross block rows, ragged edges.
        for img in [crate::synth::foliage(72), crate::synth::bricks(72)] {
            let mut s = Surface::new(70, 66);
            for y in 0..66 {
                for x in 0..70 {
                    s.set(x, y, img.get(x, y));
                }
            }
            for effort in 0..=9u8 {
                let fast = encode_surface(&s, effort);
                let mut generic = vec![0u8; fast.len()];
                encode_span::<f32x8>(&s, 0, &mut generic, cfg(effort));
                assert_eq!(fast, generic, "effort {effort}");
                // Within reach of the scalar PCA fit.
                let scalar: Vec<u8> = s
                    .blocks()
                    .iter()
                    .flat_map(|b| super::super::encode(b, &[1, 1, 1, 0], 0))
                    .collect();
                let (pf, ps) = (psnr(&s, &fast), psnr(&s, &scalar));
                assert!(pf > ps - 0.6, "effort {effort}: {pf:.2} vs scalar {ps:.2}");
            }
        }
    }
}

#[cfg(test)]
mod diag {
    use super::*;

    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn cluster_quality_breakdown() {
        let w = [1u32, 1, 1, 0];
        let mut surfs = Vec::new();
        for n in ["kodim01", "kodim02", "kodim03"] {
            surfs.push(Surface::load(std::path::Path::new(&format!("test-images/kodak/{n}.png"))).unwrap());
        }
        let pooled = |f: &dyn Fn(&Surface) -> Vec<u8>| -> f64 {
            let (mut sse, mut n) = (0u64, 0usize);
            for s in &surfs {
                let raw = f(s);
                let bw = s.blocks_x();
                let mut d = [[0u8; 4]; 16];
                for (i, b) in raw.as_chunks::<8>().0.iter().enumerate() {
                    super::super::decode(b, &mut d);
                    let src = s.read_block(i as u32 % bw, i as u32 / bw);
                    sse += src
                        .iter()
                        .zip(&d)
                        .map(|(a, b)| crate::surface::texel_sse(a, b, &w) as u64)
                        .sum::<u64>();
                    n += 48;
                }
            }
            10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10()
        };
        for e in 0..=4u8 {
            println!("batch e{e}: {:.3}", pooled(&|s| encode_surface(s, e)));
        }
        let c = |splits, closed, three, prune| Cfg {
            refits: 2,
            splits,
            closed,
            three,
            prune,
            refine: 0,
            rounds: 1,
            moves: 2,
            refine_each: false,
            polish: 0,
            topk: 1,
        };
        let all = usize::MAX;
        for (name, cf) in [
            ("snapped all + 3c", c(all, false, true, 0.0)),
            ("snapped all + 3c, prune 0.5", c(all, false, true, 0.5)),
            ("snapped all + 3c, prune 0.8", c(all, false, true, 0.8)),
            ("snapped all + 3c, prune 1.0", c(all, false, true, 1.0)),
            ("snapped all + 3c, prune 1.5", c(all, false, true, 1.5)),
            ("snapped 256 + 3c, prune 1.0", c(256, false, true, 1.0)),
            ("closed all + 3c", c(all, true, true, 0.0)),
            ("3c only", c(0, true, true, 0.0)),
        ] {
            let t0 = std::time::Instant::now();
            let q = pooled(&|s| encode_surface_cfg(s, cf));
            println!("{name:<32} {q:.3}  {:6.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        for lvl in 1..=3u8 {
            println!(
                "scalar level {lvl}: {:.3}",
                pooled(&|s| s
                    .blocks()
                    .iter()
                    .flat_map(|b| super::super::encode(b, &w, lvl))
                    .collect())
            );
        }
        // Scalar 4-color-only cluster fit over all splits (snapped), best estimate finalized.
        let four_only = |b: &[[u8; 4]; 16]| -> [u8; 8] {
            if b.iter().all(|t| t[..3] == b[0][..3]) {
                return super::super::encode_solid(b, &w).0;
            }
            let pts: [crate::fit::V4; 16] =
                std::array::from_fn(|i| [b[i][0] as f32, b[i][1] as f32, b[i][2] as f32, 0.0]);
            let (_, cov) = crate::fit::mean_cov(&pts, 3);
            let axis = crate::fit::principal_axis(&cov, 3);
            let sorted = super::super::sort_along(&pts, &axis).unwrap();
            let (a, bb, _) = super::super::cluster_fit(&sorted, &w, false, true, usize::MAX, None).unwrap();
            super::super::finalize(a, bb, b, &w).0
        };
        println!(
            "scalar 4c-only all splits: {:.3}",
            pooled(&|s| s.blocks().iter().flat_map(four_only).collect())
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod perf {
    use super::*;
    use crate::simd::avx2::F;

    #[target_feature(enable = "avx2,fma")]
    fn run(ts: &[Texels<F>], limit: usize, closed: bool) -> f32 {
        let mut acc = 0f32;
        let ax = [F::splat(0.57), F::splat(0.58), F::splat(0.57)];
        for t in ts {
            let srt = sort_along(t, ax);
            acc += if closed {
                cluster_closed(t, &srt, limit)
            } else {
                cluster_snapped::<F, 1>(t, &srt, limit, 0.0)
            }
            .err
            .to_array()[0];
        }
        acc
    }

    /// Throwaway timing of the cluster-fit split loop (AVX2) on one Kodak image.
    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn cluster_speed() {
        let s = Surface::load(std::path::Path::new("test-images/kodak/kodim01.png")).unwrap();
        let blocks = s.blocks();
        let ts: Vec<Texels<F>> = blocks
            .as_chunks::<LANES>()
            .0
            .iter()
            .map(|c| {
                let mut t = [[F::splat(0.0); 16]; 3];
                for ch in 0..3 {
                    for i in 0..16 {
                        t[ch][i] = F::from_array(std::array::from_fn(|l| c[l][i][ch] as f32));
                    }
                }
                t
            })
            .collect();
        for (limit, closed) in [(0usize, false), (256, false), (969, false), (256, true), (969, true)] {
            let t0 = std::time::Instant::now();
            // SAFETY: the timing test only runs on AVX2 machines.
            let acc = unsafe { run(&ts, limit, closed) };
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "limit {limit:4} closed {closed:5}: {:7.2} ms, {:6.1} ns/batch/split (chk {acc})",
                dt * 1e3,
                dt * 1e9 / ts.len() as f64 / limit.max(1) as f64
            );
        }
    }

    /// Effort-0 speed on the 24 Kodak images (fastest of N passes, one thread):
    /// `RAYON_NUM_THREADS=1 cargo test --release e0_speed -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn e0_speed() {
        let surfs: Vec<Surface> = (1..=24)
            .map(|i| Surface::load(std::path::Path::new(&format!("test-images/kodak/kodim{i:02}.png"))).unwrap())
            .collect();
        let px: f64 = surfs.iter().map(|s| (s.width * s.height) as f64).sum();
        for effort in [0u8, 1] {
            let mut best = f64::MAX;
            let mut chk = 0usize;
            for _ in 0..20 {
                let t0 = std::time::Instant::now();
                for s in &surfs {
                    chk += encode_surface(s, effort)[8] as usize;
                }
                best = best.min(t0.elapsed().as_secs_f64());
            }
            println!(
                "effort {effort}: {:.3} ms, {:.1} Mpix/s (chk {chk})",
                best * 1e3,
                px / best / 1e6
            );
        }
    }
}
