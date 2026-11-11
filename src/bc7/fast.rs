//! Fast BC7 encoder for opaque blocks: eight blocks per SIMD batch (one per lane), read straight
//! from the surface and packed in registers. It serves every effort ([`Cfg`], set by
//! `super::fast_cfg`).
//!
//! Every lane starts with mode 6 (one subset, 7-bit RGBA endpoints with a p-bit each, 4-bit
//! indices):
//!
//! * endpoints: the principal axis by power iteration, cut to 7 bits, at the extreme projections
//!   of the line through the mean;
//! * quantization: for each endpoint and p-bit the nearest levels are one rounding; the p-bit
//!   with the lower endpoint error wins (alpha included, so opaque blocks keep alpha 255 unless
//!   p = 0 is clearly better for RGB);
//! * indices: of the two palette entries around the projection, the one that decodes closer;
//! * [`Cfg::refits`] least-squares refits from those indices, each kept per lane when it lowers
//!   the error.
//!
//! Lanes with error left then try modes 1 and 3 on estimated or ranked 2-subset partitions
//! (`guess_partitions`, `rank_fits`), modes 0 and 2 on ranked 3-subset partitions
//! (`rank_fits3`) and dual-plane mode 5, with refits, ±1 endpoint refinement and (efforts 8-9)
//! bc7e-style uber trials. Blocks with alpha go to the scalar encoder.
//!
//! Every value is an integer below 2^24 in f32 (see `crate::exact`), so results do not depend on
//! rounding, summation order or fused multiply-adds: the portable and AVX2 backends and the GPU
//! kernel (`src/gpu/bc7.wgsl`, efforts 0-5) write the same blocks.

use crate::exact::{div_floor, div_floor_q, dot16, reduce16, shrink, sum16};
use crate::format::BlockOpts;
#[cfg(target_arch = "x86_64")]
use crate::simd::Level;
use crate::simd::{LANES, ROW_ORDER, SimdF32, SimdI32, f32x8, level};
use crate::surface::Surface;

/// Search configuration of the fast encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfg {
    /// Power iterations for the principal axis.
    pub power: u8,
    /// Least-squares refits, each kept per lane when it lowers the exact error.
    pub refits: u8,
    /// Also try 2-subset mode 1 (bit 0) and mode 3 (bit 1) on each lane's estimated partition.
    pub two: u8,
    /// Least-squares refits of the 2-subset encodings.
    pub two_refits: u8,
    /// Partition candidates per lane for the 2-subset modes (1-4).
    pub parts: u8,
    /// Partition ranking: the line-fit residual of all 64 partitions with a one-step λ estimate
    /// (`rank_fits`) instead of bc7f's popcount estimate.
    pub rank: bool,
    /// ±1 endpoint refinement passes of the winning 2-subset encodings (`refine_multi`).
    pub refine: u8,
    /// Mode-6 error (sum over the block) at or below which a block skips the 2-subset search.
    pub busy_err: f32,
    /// Also refine the lanes where mode 6 wins.
    pub refine6: bool,
    /// Also try 3-subset mode 0 (bit 0) and mode 2 (bit 1) on line-fit-ranked partitions.
    pub three: u8,
    /// Partition candidates per lane for the 3-subset modes.
    pub parts_three: u8,
    /// Also try dual-plane mode 5 with each color channel rotated into the alpha plane.
    pub mode5: bool,
    /// bc7e-style "uber" trials on each multi-subset winner: least-squares fits from index sets
    /// with the extremes nudged inward (`uber` >= 1) and stretched by up to `uber - 1` steps.
    pub uber: u8,
    /// Run the uber trials and refinement on every mode's best encoding before the modes compete
    /// (as bc7e does), not only on the winner; mode 6 is polished first.
    pub polish_modes: bool,
    /// With `polish_modes`: polish every partition candidate, not only each mode's best.
    pub polish_parts: bool,
    /// With `polish_parts`: only the first this many candidates of each mode (0 = all).
    pub polish_top: u8,
}

/// `floor` for non-negative lanes.
#[inline(always)]
fn floor<V: SimdF32>(v: V) -> V {
    v.floor()
}

/// Interpolation weight of index `j` for `n`-entry palettes: BC7's weight tables are exactly
/// `round(64 j / (n - 1))`.
#[inline(always)]
fn weight<V: SimdF32>(j: V, n: u32) -> V {
    floor(j.mul_add(V::splat(64.0 / (n - 1) as f32), V::splat(0.5)))
}

/// Eight blocks' texels, channel-major (RGB; alpha is 255 in a batch).
type Texels<V> = [[V; 16]; 3];

/// A quantized mode-6 endpoint pair per lane: 7-bit levels (RGBA), p-bits, 8-bit values.
#[derive(Clone, Copy)]
struct Ends<V: SimdF32> {
    q: [[V; 4]; 2],
    p: [V; 2],
    v: [[V; 4]; 2],
}

/// Nearest mode-6 levels of one endpoint (RGB from `e`, alpha 255) for the better p-bit.
#[inline(always)]
fn quant_endpoint<V: SimdF32>(e: &[V; 3]) -> ([V; 4], V, [V; 4]) {
    let zero = V::splat(0.0);
    let half = V::splat(0.5);
    let top = V::splat(127.0);
    let x = [e[0], e[1], e[2], V::splat(255.0)];
    let mut best_q = [zero; 4];
    let mut best_v = [zero; 4];
    let mut best_e = V::splat(f32::INFINITY);
    let mut best_p = zero;
    let mut p = 0;
    while p < 2 {
        let pv = V::splat(p as f32);
        let mut q = [zero; 4];
        let mut v = [zero; 4];
        let mut err = zero;
        let mut c = 0;
        while c < 4 {
            // Nearest q for 2q + p: floor((x - p) / 2 + 1/2), clamped to 0..=127.
            q[c] = floor((x[c] - pv).mul_add(half, half).simd_max(zero)).simd_min(top);
            v[c] = q[c].mul_add(V::splat(2.0), pv);
            let d = v[c] - x[c];
            err = d.mul_add(d, err);
            c += 1;
        }
        // p = 1 only when strictly better (as the scalar quantizer).
        let take = err.simd_lt(best_e);
        best_e = V::select(take, err, best_e);
        best_p = V::select(take, pv, best_p);
        let mut c = 0;
        while c < 4 {
            best_q[c] = V::select(take, q[c], best_q[c]);
            best_v[c] = V::select(take, v[c], best_v[c]);
            c += 1;
        }
        p += 1;
    }
    (best_q, best_p, best_v)
}

#[inline(always)]
fn quantize<V: SimdF32>(e0: &[V; 3], e1: &[V; 3]) -> Ends<V> {
    let (q0, p0, v0) = quant_endpoint(e0);
    let (q1, p1, v1) = quant_endpoint(e1);
    Ends {
        q: [q0, q1],
        p: [p0, p1],
        v: [v0, v1],
    }
}

/// Squared errors of texel `i` (channels `ch`) against the palette entries with weights `w0` and
/// `w1`, for endpoints `a` and `d64 = (b - a) / 64`: entry = `a + floor(w d64 + 1/2)`, which equals
/// the decoder's `((64 - w) a + w b + 32) >> 6` exactly (`64 a` is a multiple of 64 and every
/// intermediate is a multiple of 1/64).
#[inline(always)]
fn pair_err<V: SimdF32, const N: usize>(
    t: &Texels<V>,
    ch: [usize; N],
    i: usize,
    a: &[V; N],
    d64: &[V; N],
    w0: V,
    w1: V,
) -> (V, V) {
    let half = V::splat(0.5);
    let (mut e0, mut e1) = (V::splat(0.0), V::splat(0.0));
    let mut c = 0;
    while c < N {
        let xa = t[ch[c]][i] - a[c];
        let d0 = xa - floor(w0.mul_add(d64[c], half));
        let d1 = xa - floor(w1.mul_add(d64[c], half));
        e0 = d0.mul_add(d0, e0);
        e1 = d1.mul_add(d1, e1);
        c += 1;
    }
    (e0, e1)
}

/// Indices by projection onto the quantized endpoints (RGB): the better of the two palette
/// entries around the projection. Returns the indices and the block's RGB error.
#[inline(always)]
fn project<V: SimdF32>(t: &Texels<V>, e: &Ends<V>) -> ([V; 16], V) {
    let zero = V::splat(0.0);
    let d = [e.v[1][0] - e.v[0][0], e.v[1][1] - e.v[0][1], e.v[1][2] - e.v[0][2]];
    let dd = d[0].mul_add(d[0], d[1].mul_add(d[1], d[2] * d[2]));
    let top = V::splat(14.0);
    let inv64 = V::splat(1.0 / 64.0);
    let a3 = [e.v[0][0], e.v[0][1], e.v[0][2]];
    let d64 = [d[0] * inv64, d[1] * inv64, d[2] * inv64];
    let mut idx = [zero; 16];
    let mut err = zero;
    // j = floor(15 (x - a)·d / |d|²) in integers (|15 (x - a)·d| < 2^22), the numerator plus 1/2
    // ([`div_floor_q`]) in three exact fused steps; then j or j + 1, whichever decodes closer.
    let inv_dd = V::splat(1.0) / dd.simd_max(V::splat(1.0));
    let d15 = [d[0] * V::splat(15.0), d[1] * V::splat(15.0), d[2] * V::splat(15.0)];
    let xbias = V::splat(0.5) - (a3[0] * d15[0] + a3[1] * d15[1] + a3[2] * d15[2]);
    let mut i = 0;
    while i < 16 {
        let num = t[0][i].mul_add(d15[0], t[1][i].mul_add(d15[1], t[2][i].mul_add(d15[2], xbias)));
        let j = floor(num * inv_dd).simd_max(zero).simd_min(top);
        let (e0, e1) = pair_err(t, [0, 1, 2], i, &a3, &d64, weight(j, 16), weight(j + V::splat(1.0), 16));
        let up = e1.simd_lt(e0);
        idx[i] = V::select(up, j + V::splat(1.0), j);
        err += e0.simd_min(e1);
        i += 1;
    }
    (idx, err)
}

/// Least-squares RGB endpoints in 8-bit units (multiples of 1/16) for the indices `idx`
/// (`nidx`-entry palettes) of the texels in `m` (0/1 per texel), in exact integers: the line
/// t = α + β w through the members' weights w (0..=64) puts the endpoint of weight w at
/// E(w) = Σt + num (n w - Σw) / D in 1/n units, with num = n Σwt - Σw Σt and D = n Σw² - (Σw)²
/// (D cut to 13 bits, num by the same shift), clamped to 0..=255 n, then floor(16 E / n). D is
/// 4096 times the float solver's determinant; lanes with D < 5 (that determinant below 1e-3) or
/// a slope over 4096 keep `prev`. Every intermediate is an integer below 2^23.
#[inline(always)]
fn ls_ends<V: SimdF32>(
    t: &Texels<V>,
    m: &[V; 16],
    idx: &[V; 16],
    nidx: u32,
    prev: ([V; 3], [V; 3]),
) -> ([V; 3], [V; 3], V::Mask) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    // Loops, not `array::from_fn`: closures do not inherit the caller's target features.
    let mut w = [zero; 16];
    let mut i = 0;
    while i < 16 {
        w[i] = weight(idx[i], nidx) * m[i];
        i += 1;
    }
    let n = sum16!(|i| m[i]).simd_max(one);
    let sw = sum16!(|i| w[i]);
    let sww = dot16!(|i| w[i], w[i]);
    let sum = [
        dot16!(|i| m[i], t[0][i]),
        dot16!(|i| m[i], t[1][i]),
        dot16!(|i| m[i], t[2][i]),
    ];
    let swt = [
        dot16!(|i| w[i], t[0][i]),
        dot16!(|i| w[i], t[1][i]),
        dot16!(|i| w[i], t[2][i]),
    ];
    let d = n * sww - sw * sw;
    let sh = shrink(d, 12);
    let ds = floor(d * sh).simd_max(one);
    let inv = one / ds;
    let q0 = div_floor(floor((n * swt[0] - sw * sum[0]) * sh), ds, inv);
    let q1 = div_floor(floor((n * swt[1] - sw * sum[1]) * sh), ds, inv);
    let q2 = div_floor(floor((n * swt[2] - sw * sum[2]) * sh), ds, inv);
    let q = [q0, q1, q2];
    let ok = V::splat(4.5).simd_lt(d) & abs_max(&[q0.0, q1.0, q2.0]).simd_lt(V::splat(4096.5));
    let (k0, k1) = (zero - sw, n * V::splat(64.0) - sw);
    let (top, inv_n, inv16) = (n * V::splat(255.0), one / n, V::splat(1.0 / 16.0));
    let mut e0 = prev.0;
    let mut e1 = prev.1;
    let mut c = 0;
    while c < 3 {
        let (qc, rc) = q[c];
        let a = line_point(sum[c], qc, rc, k0, ds, inv, top, inv_n) * inv16;
        let b = line_point(sum[c], qc, rc, k1, ds, inv, top, inv_n) * inv16;
        e0[c] = V::select(ok, a, e0[c]);
        e1[c] = V::select(ok, b, e1[c]);
        c += 1;
    }
    (e0, e1, ok)
}

/// One channel of a line's endpoint in 1/16 units ([`ls_ends`], [`subset_ends`]):
/// E = Σt + q k + floor(r k / d) for a ratio q + r / d times k, clamped to 0..=255 n, then
/// floor(16 E / n).
#[expect(clippy::too_many_arguments, reason = "per-lane ratio, divisor and clamp")]
#[inline(always)]
fn line_point<V: SimdF32>(sc: V, q: V, r: V, k: V, d: V, inv: V, top: V, inv_n: V) -> V {
    let e = (sc + q * k + div_floor(r * k, d, inv).0).simd_clamp(V::splat(0.0), top);
    div_floor_q(e * V::splat(16.0), inv_n)
}

/// A multi-subset opaque mode of the fast encoder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Two {
    /// 3 subsets (16 partitions), 4-bit RGB, one p-bit per endpoint, 3-bit indices.
    M0,
    /// 2 subsets, 6-bit RGB, one p-bit per subset, 3-bit indices.
    M1,
    /// 3 subsets, 5-bit RGB, no p-bits, 2-bit indices.
    M2,
    /// 2 subsets, 7-bit RGB, one p-bit per endpoint, 2-bit indices.
    M3,
}

/// How a mode's endpoints carry p-bits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PBits {
    None,
    Shared,
    Endpoint,
}

impl Two {
    fn mode(self) -> u8 {
        match self {
            Two::M0 => 0,
            Two::M1 => 1,
            Two::M2 => 2,
            Two::M3 => 3,
        }
    }
    fn subsets(self) -> usize {
        match self {
            Two::M0 | Two::M2 => 3,
            _ => 2,
        }
    }
    fn color_bits(self) -> u8 {
        match self {
            Two::M0 => 4,
            Two::M1 => 6,
            Two::M2 => 5,
            Two::M3 => 7,
        }
    }
    fn pbits(self) -> PBits {
        match self {
            Two::M1 => PBits::Shared,
            Two::M2 => PBits::None,
            _ => PBits::Endpoint,
        }
    }
    fn index_bits(self) -> u8 {
        match self {
            Two::M0 | Two::M1 => 3,
            _ => 2,
        }
    }
}

/// One subset's encoding per lane: RGB levels, p-bits, 8-bit endpoint values, indices and the
/// error summed over the subset's texels.
#[derive(Clone, Copy)]
struct Sub<V: SimdF32> {
    q: [[V; 3]; 2],
    p: [V; 2],
    v: [[V; 3]; 2],
    idx: [V; 16],
    err: V,
}

/// The rounded level of an 8-bit target `x` (a multiple of 1/16) on a `full`-step scale:
/// floor(full x / 255 + 1/2), or with a p-bit floor((full x / 255 - p) / 2 + 1/2), not below 0.
/// Exact: floor((16 full x - 4080 p + 4080) / 8160) in integers below 2^20.
#[inline(always)]
fn round_level<V: SimdF32>(x: V, p: V, full: f32, has_p: bool) -> V {
    let x16 = x * V::splat(16.0) * V::splat(full);
    let (num, den) = if has_p {
        (x16 - p * V::splat(4080.0) + V::splat(4080.0), 8160.0)
    } else {
        (x16 + V::splat(2040.0), 4080.0)
    };
    div_floor_q(num, V::splat(1.0 / den)).simd_max(V::splat(0.0))
}

/// Nearest 6-bit levels (with p-bit `p`) of 8-bit targets, as 7-bit values `2q + p` expanded by
/// bit replication: tries the rounded level and its neighbours.
#[inline(always)]
fn quant6p<V: SimdF32>(x: V, p: V) -> (V, V, V) {
    let zero = V::splat(0.0);
    let c = round_level(x, p, 127.0, true);
    let (mut bq, mut bv, mut be) = (zero, zero, V::splat(f32::INFINITY));
    let mut off = -1;
    while off <= 1 {
        let q = (c + V::splat(off as f32)).simd_clamp(zero, V::splat(63.0));
        let v7 = q.mul_add(V::splat(2.0), p);
        // (v7 << 1) | (v7 >> 6)
        let v = v7.mul_add(V::splat(2.0), floor(v7 * V::splat(1.0 / 64.0)));
        let d = v - x;
        let e = d * d;
        let take = e.simd_lt(be);
        bq = V::select(take, q, bq);
        bv = V::select(take, v, bv);
        be = V::select(take, e, be);
        off += 1;
    }
    (bq, bv, be)
}

/// Nearest levels of 8-bit targets for `mode`'s color bits with p-bit `p` (`2q + p` when the mode
/// has p-bits), bit-replicated to 8 bits: the rounded level and its neighbours.
#[inline(always)]
fn quant_gen<V: SimdF32>(x: V, p: V, mode: Two) -> (V, V, V) {
    let zero = V::splat(0.0);
    let cb = mode.color_bits() as i32;
    let has_p = mode.pbits() != PBits::None;
    let n = cb + has_p as i32;
    let c = round_level(x, p, ((1 << n) - 1) as f32, has_p);
    let top = V::splat(((1 << cb) - 1) as f32);
    let (mut bq, mut bv, mut be) = (zero, zero, V::splat(f32::INFINITY));
    let mut off = -1;
    while off <= 1 {
        let q = (c + V::splat(off as f32)).simd_clamp(zero, top);
        let v = dequant_two(q, p, mode);
        let d = v - x;
        let e = d * d;
        let take = e.simd_lt(be);
        bq = V::select(take, q, bq);
        bv = V::select(take, v, bv);
        be = V::select(take, e, be);
        off += 1;
    }
    (bq, bv, be)
}

/// Quantizes a subset's endpoint pair for `mode`.
#[inline(always)]
fn quant_two<V: SimdF32>(e: &[[V; 3]; 2], mode: Two) -> ([[V; 3]; 2], [V; 2], [[V; 3]; 2]) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let mut q = [[zero; 3]; 2];
    let mut v = [[zero; 3]; 2];
    let mut p = [zero; 2];
    match mode {
        Two::M1 => {
            // One p-bit for both endpoints: the one with the lower total error.
            let mut r = [[[zero; 3]; 2]; 2];
            let mut vv = [[[zero; 3]; 2]; 2];
            let mut err = [zero; 2];
            let mut pb = 0;
            while pb < 2 {
                let pv = V::splat(pb as f32);
                let mut k = 0;
                while k < 2 {
                    let mut c = 0;
                    while c < 3 {
                        let (a, b, d) = quant6p(e[k][c], pv);
                        r[pb][k][c] = a;
                        vv[pb][k][c] = b;
                        err[pb] += d;
                        c += 1;
                    }
                    k += 1;
                }
                pb += 1;
            }
            let use1 = err[1].simd_lt(err[0]);
            let mut k = 0;
            while k < 2 {
                let mut c = 0;
                while c < 3 {
                    q[k][c] = V::select(use1, r[1][k][c], r[0][k][c]);
                    v[k][c] = V::select(use1, vv[1][k][c], vv[0][k][c]);
                    c += 1;
                }
                k += 1;
            }
            p = [V::select(use1, one, zero); 2];
        }
        Two::M2 => {
            let mut k = 0;
            while k < 2 {
                let mut c = 0;
                while c < 3 {
                    let (a, b, _) = quant_gen(e[k][c], zero, mode);
                    q[k][c] = a;
                    v[k][c] = b;
                    c += 1;
                }
                k += 1;
            }
        }
        Two::M0 => {
            // The better p-bit per endpoint.
            let mut k = 0;
            while k < 2 {
                let mut be = V::splat(f32::INFINITY);
                let mut pb = 0;
                while pb < 2 {
                    let pv = V::splat(pb as f32);
                    let mut qq = [zero; 3];
                    let mut vv = [zero; 3];
                    let mut err = zero;
                    let mut c = 0;
                    while c < 3 {
                        let (a, b, d) = quant_gen(e[k][c], pv, mode);
                        qq[c] = a;
                        vv[c] = b;
                        err += d;
                        c += 1;
                    }
                    let take = err.simd_lt(be);
                    be = V::select(take, err, be);
                    p[k] = V::select(take, pv, p[k]);
                    let mut c = 0;
                    while c < 3 {
                        q[k][c] = V::select(take, qq[c], q[k][c]);
                        v[k][c] = V::select(take, vv[c], v[k][c]);
                        c += 1;
                    }
                    pb += 1;
                }
                k += 1;
            }
        }
        Two::M3 => {
            // 8-bit values 2q + p: one rounding per p-bit, the better p-bit per endpoint.
            let (half, top) = (V::splat(0.5), V::splat(127.0));
            let mut k = 0;
            while k < 2 {
                let mut be = V::splat(f32::INFINITY);
                let mut pb = 0;
                while pb < 2 {
                    let pv = V::splat(pb as f32);
                    let mut qq = [zero; 3];
                    let mut vv = [zero; 3];
                    let mut err = zero;
                    let mut c = 0;
                    while c < 3 {
                        qq[c] = floor((e[k][c] - pv).mul_add(half, half).simd_max(zero)).simd_min(top);
                        vv[c] = qq[c].mul_add(V::splat(2.0), pv);
                        let d = vv[c] - e[k][c];
                        err = d.mul_add(d, err);
                        c += 1;
                    }
                    let take = err.simd_lt(be);
                    be = V::select(take, err, be);
                    p[k] = V::select(take, pv, p[k]);
                    let mut c = 0;
                    while c < 3 {
                        q[k][c] = V::select(take, qq[c], q[k][c]);
                        v[k][c] = V::select(take, vv[c], v[k][c]);
                        c += 1;
                    }
                    pb += 1;
                }
                k += 1;
            }
        }
    }
    (q, p, v)
}

/// Exact indices of all 16 texels for a subset's endpoints (the better of the two palette
/// entries around the projection) and the error summed over the members `m`.
#[inline(always)]
fn index_two<V: SimdF32>(t: &Texels<V>, m: &[V; 16], v: &[[V; 3]; 2], mode: Two) -> ([V; 16], V) {
    let zero = V::splat(0.0);
    let n = (1u32 << mode.index_bits()) as f32;
    let d = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
    // j = floor((n - 1) (x - a)·d / |d|²) in integers (|(n - 1) (x - a)·d| < 2^21).
    let dd1 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).simd_max(V::splat(1.0));
    let inv_dd = V::splat(1.0) / dd1;
    let steps = V::splat(n - 1.0);
    let ds = [d[0] * steps, d[1] * steps, d[2] * steps];
    // The numerator plus 1/2 ([`div_floor_q`]) in three exact fused steps per texel.
    let bias = V::splat(0.5) - (v[0][0] * ds[0] + v[0][1] * ds[1] + v[0][2] * ds[2]);
    let top = V::splat(n - 2.0);
    let inv64 = V::splat(1.0 / 64.0);
    let d64 = [d[0] * inv64, d[1] * inv64, d[2] * inv64];
    let mut idx = [zero; 16];
    let mut err = zero;
    let mut i = 0;
    while i < 16 {
        let num = t[0][i].mul_add(ds[0], t[1][i].mul_add(ds[1], t[2][i].mul_add(ds[2], bias)));
        let j = floor(num * inv_dd).simd_max(zero).simd_min(top);
        let w0 = weight(j, n as u32);
        let w1 = weight(j + V::splat(1.0), n as u32);
        let (e0, e1) = pair_err(t, [0, 1, 2], i, &v[0], &d64, w0, w1);
        let up = e1.simd_lt(e0);
        idx[i] = V::select(up, j + V::splat(1.0), j);
        err = m[i].mul_add(e0.simd_min(e1), err);
        i += 1;
    }
    (idx, err)
}

/// [`index_two`] for the first `NS` subsets at once: each texel is indexed against its own
/// subset's endpoints (picked by the membership masks `m`), so one pass instead of `NS`. The
/// per-subset errors equal `index_two`'s exactly (same operations per texel; non-members add
/// `0 · e`). Each index is the texel's own subset's, which is the only one callers read.
#[inline(always)]
fn index_multi<V: SimdF32, const NS: usize>(
    t: &Texels<V>,
    m: &[[V; 16]; 3],
    v: &[[[V; 3]; 2]; NS],
    mode: Two,
) -> ([V; 16], [V; NS]) {
    let zero = V::splat(0.0);
    let n = (1u32 << mode.index_bits()) as f32;
    let steps = V::splat(n - 1.0);
    let inv64 = V::splat(1.0 / 64.0);
    // Per subset, as in `index_two`.
    let (mut ds, mut bias, mut inv_dd, mut d64) = ([[zero; 3]; NS], [zero; NS], [zero; NS], [[zero; 3]; NS]);
    let mut s = 0;
    while s < NS {
        let vs = &v[s];
        let d = [vs[1][0] - vs[0][0], vs[1][1] - vs[0][1], vs[1][2] - vs[0][2]];
        let dd1 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).simd_max(V::splat(1.0));
        inv_dd[s] = V::splat(1.0) / dd1;
        ds[s] = [d[0] * steps, d[1] * steps, d[2] * steps];
        bias[s] = V::splat(0.5) - (vs[0][0] * ds[s][0] + vs[0][1] * ds[s][1] + vs[0][2] * ds[s][2]);
        d64[s] = [d[0] * inv64, d[1] * inv64, d[2] * inv64];
        s += 1;
    }
    let top = V::splat(n - 2.0);
    let mut idx = [zero; 16];
    let mut err = [zero; NS];
    let mut i = 0;
    while i < 16 {
        let in1 = zero.simd_lt(m[1][i]);
        let in2 = if NS == 3 { zero.simd_lt(m[2][i]) } else { in1 };
        let pick = |x: [V; NS]| {
            let y = V::select(in1, x[1], x[0]);
            if NS == 3 { V::select(in2, x[NS - 1], y) } else { y }
        };
        let ds3: [V; 3] = std::array::from_fn(|c| pick(std::array::from_fn(|s| ds[s][c])));
        let d643: [V; 3] = std::array::from_fn(|c| pick(std::array::from_fn(|s| d64[s][c])));
        let a3: [V; 3] = std::array::from_fn(|c| pick(std::array::from_fn(|s| v[s][0][c])));
        let (b, inv) = (pick(bias), pick(inv_dd));
        let num = t[0][i].mul_add(ds3[0], t[1][i].mul_add(ds3[1], t[2][i].mul_add(ds3[2], b)));
        let j = floor(num * inv).simd_max(zero).simd_min(top);
        let w0 = weight(j, n as u32);
        let w1 = weight(j + V::splat(1.0), n as u32);
        let (e0, e1) = pair_err(t, [0, 1, 2], i, &a3, &d643, w0, w1);
        let up = e1.simd_lt(e0);
        idx[i] = V::select(up, j + V::splat(1.0), j);
        let e = e0.simd_min(e1);
        let mut s = 0;
        while s < NS {
            err[s] = m[s][i].mul_add(e, err[s]);
            s += 1;
        }
        i += 1;
    }
    (idx, err)
}

/// bc7e's "uber" trials then greedy ±1 refinement (per `level` and `passes`; 0 skips) of the
/// first `NS` subsets, run in lockstep so each trial indexes every subset in one
/// [`index_multi`] pass. Subsets are independent (a subset's error depends only on its own
/// endpoints and members), and a refinement pass that changes nothing would change nothing if
/// repeated, so running passes until every subset has converged gives each subset exactly the
/// result it gets alone.
#[inline(always)]
fn polish_multi<V: SimdF32, const NS: usize>(
    t: &Texels<V>,
    m: &[[V; 16]; 3],
    subs: &mut [Sub<V>; 3],
    mode: Two,
    level: u8,
    passes: u8,
) {
    if level > 0 {
        uber_multi::<V, NS>(t, m, subs, mode, level);
    }
    if passes > 0 {
        refine_multi::<V, NS>(t, m, subs, mode, passes);
    }
}

/// [`polish_multi`] for the subset count of `mode`.
#[inline(always)]
fn polish<V: SimdF32>(t: &Texels<V>, m: &[[V; 16]; 3], subs: &mut [Sub<V>; 3], mode: Two, level: u8, passes: u8) {
    if mode.subsets() == 3 {
        polish_multi::<V, 3>(t, m, subs, mode, level, passes);
    } else {
        polish_multi::<V, 2>(t, m, subs, mode, level, passes);
    }
}

/// bc7e's "uber" trials for each subset: least-squares fits from index sets with the subset's
/// lowest index raised, its highest lowered, both, and (for `level` >= 2) the indices linearly
/// stretched from `[ly, hy]` onto the full range for ly in -Q..=1 and hy in max-1..=max+Q
/// (Q = `level` - 1). Each fit is quantized and re-indexed exactly; kept per lane when better.
#[inline(always)]
fn uber_multi<V: SimdF32, const NS: usize>(
    t: &Texels<V>,
    m: &[[V; 16]; 3],
    subs: &mut [Sub<V>; 3],
    mode: Two,
    level: u8,
) {
    let zero = V::splat(0.0);
    let maxsel = ((1u32 << mode.index_bits()) - 1) as f32;
    let top = V::splat(maxsel);
    // Lowest and highest index over each subset's members.
    let (mut lo, mut hi) = ([top; NS], [zero; NS]);
    let mut s = 0;
    while s < NS {
        let mut i = 0;
        while i < 16 {
            let member = zero.simd_lt(m[s][i]);
            lo[s] = V::select(member, lo[s].simd_min(subs[s].idx[i]), lo[s]);
            hi[s] = V::select(member, hi[s].simd_max(subs[s].idx[i]), hi[s]);
            i += 1;
        }
        s += 1;
    }
    let base: [[V; 16]; NS] = std::array::from_fn(|s| subs[s].idx);
    let prev: [[[V; 3]; 2]; NS] = std::array::from_fn(|s| subs[s].v);
    let q = level as i32 - 1;
    let nt = 3 + if level >= 2 {
        ((q + 2) * (q + 2) - 1) as usize
    } else {
        0
    };
    let mut k = 0;
    while k < nt {
        let mut oks = [zero.simd_lt(zero); NS];
        let mut qs = [[[zero; 3]; 2]; NS];
        let mut ps = [[zero; 2]; NS];
        let mut vs = [[[zero; 3]; 2]; NS];
        let mut s = 0;
        while s < NS {
            let mut idx2 = base[s];
            if k < 3 {
                let mut i = 0;
                while i < 16 {
                    let x = base[s][i];
                    let up = (x - lo[s]).simd_max(lo[s] - x).simd_lt(V::splat(0.5)) & x.simd_lt(top);
                    let down = (x - hi[s]).simd_max(hi[s] - x).simd_lt(V::splat(0.5)) & zero.simd_lt(x);
                    idx2[i] = match k {
                        0 => V::select(up, x + V::splat(1.0), x),
                        1 => V::select(down, x - V::splat(1.0), x),
                        _ => V::select(up, x + V::splat(1.0), V::select(down, x - V::splat(1.0), x)),
                    };
                    i += 1;
                }
            } else {
                // Stretch combination (ly, hy), skipping the identity (0, max).
                let mut n = k - 3;
                let side = (q + 2) as usize;
                let mut ly = (n / side) as i32 - q;
                let mut hy = (n % side) as i32 + maxsel as i32 - 1;
                if ly > 0 || (ly == 0 && hy >= maxsel as i32) {
                    // Shift past the identity.
                    n += 1;
                    ly = (n / side) as i32 - q;
                    hy = (n % side) as i32 + maxsel as i32 - 1;
                }
                let scale = V::splat(maxsel / (hy - ly) as f32);
                let l = V::splat(ly as f32);
                let mut i = 0;
                while i < 16 {
                    idx2[i] = floor(((base[s][i] - l) * scale + V::splat(0.5)).simd_max(zero)).simd_min(top);
                    i += 1;
                }
            }
            let (e, ok) = ls_two(t, &m[s], &idx2, mode, &prev[s]);
            (qs[s], ps[s], vs[s]) = quant_two(&e, mode);
            oks[s] = ok;
            s += 1;
        }
        let (idx3, err3) = index_multi(t, m, &vs, mode);
        let mut s = 0;
        while s < NS {
            let take = oks[s] & err3[s].simd_lt(subs[s].err);
            if V::mask_any(take) {
                subs[s] = merge_sub(
                    take,
                    &Sub {
                        q: qs[s],
                        p: ps[s],
                        v: vs[s],
                        idx: idx3,
                        err: err3[s],
                    },
                    &subs[s],
                );
            }
            s += 1;
        }
        k += 1;
    }
}

/// Greedy ±1 refinement of each subset's quantized endpoint channels and p-bit flips: each
/// change is kept per lane when it lowers the subset's exact error (indices re-chosen for every
/// trial; holding them fixed loses two thirds of the gain). Up to `passes` sweeps.
#[inline(always)]
fn refine_multi<V: SimdF32, const NS: usize>(
    t: &Texels<V>,
    m: &[[V; 16]; 3],
    subs: &mut [Sub<V>; 3],
    mode: Two,
    passes: u8,
) {
    let top = V::splat(((1u32 << mode.color_bits()) - 1) as f32);
    let (zero, one) = (V::splat(0.0), V::splat(1.0));
    let mut pass = 0;
    while pass < passes {
        let mut any = false;
        let mut k = 0;
        while k < 2 {
            let mut c = 0;
            while c < 3 {
                let mut dir = 0;
                while dir < 2 {
                    let d = V::splat(if dir == 0 { 1.0 } else { -1.0 });
                    let mut qs = [zero; NS];
                    let mut vs = [[[zero; 3]; 2]; NS];
                    let mut s = 0;
                    while s < NS {
                        qs[s] = (subs[s].q[k][c] + d).simd_clamp(zero, top);
                        vs[s] = subs[s].v;
                        vs[s][k][c] = dequant_two(qs[s], subs[s].p[k], mode);
                        s += 1;
                    }
                    let (idx, err) = index_multi(t, m, &vs, mode);
                    let mut s = 0;
                    while s < NS {
                        let take = err[s].simd_lt(subs[s].err);
                        if V::mask_any(take) {
                            any = true;
                            let sub = &mut subs[s];
                            sub.q[k][c] = V::select(take, qs[s], sub.q[k][c]);
                            sub.v[k][c] = V::select(take, vs[s][k][c], sub.v[k][c]);
                            sub.err = V::select(take, err[s], sub.err);
                            let mut i = 0;
                            while i < 16 {
                                sub.idx[i] = V::select(take, idx[i], sub.idx[i]);
                                i += 1;
                            }
                        }
                        s += 1;
                    }
                    dir += 1;
                }
                c += 1;
            }
            k += 1;
        }
        // p-bit flips: the shared one (mode 1), or each and both (modes 0 and 3).
        let flips: &[[bool; 2]] = match mode.pbits() {
            PBits::Shared => &[[true, true]],
            PBits::Endpoint => &[[true, false], [false, true], [true, true]],
            PBits::None => &[],
        };
        for f in flips {
            let mut nps = [[zero; 2]; NS];
            let mut vs = [[[zero; 3]; 2]; NS];
            let mut s = 0;
            while s < NS {
                nps[s] = subs[s].p;
                vs[s] = subs[s].v;
                let mut k = 0;
                while k < 2 {
                    if f[k] {
                        nps[s][k] = one - subs[s].p[k];
                        let mut c = 0;
                        while c < 3 {
                            vs[s][k][c] = dequant_two(subs[s].q[k][c], nps[s][k], mode);
                            c += 1;
                        }
                    }
                    k += 1;
                }
                s += 1;
            }
            let (idx, err) = index_multi(t, m, &vs, mode);
            let mut s = 0;
            while s < NS {
                let take = err[s].simd_lt(subs[s].err);
                if V::mask_any(take) {
                    any = true;
                    let sub = &mut subs[s];
                    let mut k = 0;
                    while k < 2 {
                        sub.p[k] = V::select(take, nps[s][k], sub.p[k]);
                        let mut c = 0;
                        while c < 3 {
                            sub.v[k][c] = V::select(take, vs[s][k][c], sub.v[k][c]);
                            c += 1;
                        }
                        k += 1;
                    }
                    sub.err = V::select(take, err[s], sub.err);
                    let mut i = 0;
                    while i < 16 {
                        sub.idx[i] = V::select(take, idx[i], sub.idx[i]);
                        i += 1;
                    }
                }
                s += 1;
            }
        }
        if !any {
            break;
        }
        pass += 1;
    }
}

/// Encodes one subset (members `m`) per lane: masked principal axis, extremes, quantization,
/// exact indices, then `refits` least-squares refits kept when they lower the error.
#[inline(always)]
fn encode_two<V: SimdF32>(t: &Texels<V>, m: &[V; 16], mode: Two, refits: u8) -> Sub<V> {
    // In exact integers, as the mode-6 line fit: n Σxy - Σx Σy over the members, three power
    // iterations, a 7-bit axis and the endpoints at the members' extreme projections.
    let one = V::splat(1.0);
    let n = sum16!(|i| m[i]).simd_max(one);
    let mut mt = [[one; 16]; 3];
    let mut i = 0;
    while i < 16 {
        mt[0][i] = m[i] * t[0][i];
        mt[1][i] = m[i] * t[1][i];
        mt[2][i] = m[i] * t[2][i];
        i += 1;
    }
    let sum = [sum16!(|i| mt[0][i]), sum16!(|i| mt[1][i]), sum16!(|i| mt[2][i])];
    let mut cov = [one; 6];
    for (k, (a, b)) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)].into_iter().enumerate() {
        cov[k] = dot16!(|i| mt[a][i], t[b][i]) * n - sum[a] * sum[b];
    }
    let v = principal_axis(cov, 3);
    let ws = {
        let s = shrink(abs_max(&v), 6);
        [floor(v[0] * s), floor(v[1] * s), floor(v[2] * s)]
    };
    let (hi, lo) = subset_ends(t, m, n, &sum, &ws);
    let inv16 = V::splat(1.0 / 16.0);
    let mut e = [
        [lo[0] * inv16, lo[1] * inv16, lo[2] * inv16],
        [hi[0] * inv16, hi[1] * inv16, hi[2] * inv16],
    ];
    let (q, p, vq) = quant_two(&e, mode);
    let (idx, err) = index_two(t, m, &vq, mode);
    let mut best = Sub { q, p, v: vq, idx, err };
    let mut r = 0;
    while r < refits {
        let (a, b, ok) = ls_ends(t, m, &best.idx, 1u32 << mode.index_bits(), (e[0], e[1]));
        let ne = [a, b];
        let (q, p, vq) = quant_two(&ne, mode);
        let (idx, err) = index_two(t, m, &vq, mode);
        let take = ok & err.simd_lt(best.err);
        if !V::mask_any(take) {
            break;
        }
        best = merge_sub(take, &Sub { q, p, v: vq, idx, err }, &best);
        e = ne;
        r += 1;
    }
    best
}

/// 8-bit value of a quantized endpoint channel `q` with p-bit `p` for `mode`: the `n`-bit value
/// (`2q + p` with p-bits) bit-replicated to 8 bits.
#[inline(always)]
fn dequant_two<V: SimdF32>(q: V, p: V, mode: Two) -> V {
    match mode {
        Two::M3 => q.mul_add(V::splat(2.0), p),
        Two::M1 => {
            let v7 = q.mul_add(V::splat(2.0), p);
            v7.mul_add(V::splat(2.0), floor(v7 * V::splat(1.0 / 64.0)))
        }
        _ => {
            let has_p = mode.pbits() != PBits::None;
            let n = mode.color_bits() as i32 + has_p as i32;
            let t = if has_p { q.mul_add(V::splat(2.0), p) } else { q };
            t.mul_add(
                V::splat((1 << (8 - n)) as f32),
                floor(t * V::splat(1.0 / (1 << (2 * n - 8)) as f32)),
            )
        }
    }
}

/// Greedy ±1 refinement of mode-6 RGB endpoint levels and p-bit flips in the `active` lanes,
/// each change kept per lane when it lowers the exact error (indices re-chosen).
#[inline(always)]
fn refine6<V: SimdF32>(t: &Texels<V>, ends: &mut Ends<V>, idx: &mut [V; 16], err: &mut V, active: V::Mask, passes: u8) {
    let (zero, top, one) = (V::splat(0.0), V::splat(127.0), V::splat(1.0));
    let mut pass = 0;
    while pass < passes {
        let mut any = false;
        // 12 level moves (endpoint x channel x direction), then 3 p-bit flips.
        let mut mv = 0;
        while mv < 15 {
            let mut cand = *ends;
            if mv < 12 {
                let (k, c, d) = (mv / 6, (mv / 2) % 3, if mv % 2 == 0 { 1.0 } else { -1.0 });
                cand.q[k][c] = (ends.q[k][c] + V::splat(d)).simd_clamp(zero, top);
                cand.v[k][c] = cand.q[k][c].mul_add(V::splat(2.0), ends.p[k]);
            } else {
                let f = [mv != 13, mv != 12];
                let mut k = 0;
                while k < 2 {
                    if f[k] {
                        cand.p[k] = one - ends.p[k];
                        let mut c = 0;
                        while c < 4 {
                            cand.v[k][c] = cand.q[k][c].mul_add(V::splat(2.0), cand.p[k]);
                            c += 1;
                        }
                    }
                    k += 1;
                }
            }
            let (cidx, cerr) = project(t, &cand);
            let take = active & cerr.simd_lt(*err);
            if V::mask_any(take) {
                any = true;
                *err = V::select(take, cerr, *err);
                let mut k = 0;
                while k < 2 {
                    let mut c = 0;
                    while c < 4 {
                        ends.q[k][c] = V::select(take, cand.q[k][c], ends.q[k][c]);
                        ends.v[k][c] = V::select(take, cand.v[k][c], ends.v[k][c]);
                        c += 1;
                    }
                    ends.p[k] = V::select(take, cand.p[k], ends.p[k]);
                    k += 1;
                }
                let mut i = 0;
                while i < 16 {
                    idx[i] = V::select(take, cidx[i], idx[i]);
                    i += 1;
                }
            }
            mv += 1;
        }
        if !any {
            break;
        }
        pass += 1;
    }
}

/// Least-squares endpoints of members `m` for indices `idx` of `mode`; lanes with a singular
/// system keep `prev`.
#[inline(always)]
fn ls_two<V: SimdF32>(
    t: &Texels<V>,
    m: &[V; 16],
    idx: &[V; 16],
    mode: Two,
    prev: &[[V; 3]; 2],
) -> ([[V; 3]; 2], V::Mask) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let n = 1u32 << mode.index_bits();
    let (mut aa, mut ab, mut bbw) = (zero, zero, zero);
    let mut ax = [zero; 3];
    let mut bx = [zero; 3];
    let inv64 = V::splat(1.0 / 64.0);
    let mut i = 0;
    while i < 16 {
        let wb = weight(idx[i], n) * inv64;
        let wa = (one - wb) * m[i];
        let wb = wb * m[i];
        aa = wa.mul_add(wa, aa);
        ab = wa.mul_add(wb, ab);
        bbw = wb.mul_add(wb, bbw);
        let mut c = 0;
        while c < 3 {
            ax[c] = wa.mul_add(t[c][i], ax[c]);
            bx[c] = wb.mul_add(t[c][i], bx[c]);
            c += 1;
        }
        i += 1;
    }
    let det = aa * bbw - ab * ab;
    let ok = V::splat(1e-3).simd_lt(det);
    let inv = one / V::select(ok, det, one);
    let full = V::splat(255.0);
    let mut e = *prev;
    let mut c = 0;
    while c < 3 {
        e[0][c] = V::select(ok, ((bbw * ax[c] - ab * bx[c]) * inv).simd_clamp(zero, full), e[0][c]);
        e[1][c] = V::select(ok, ((aa * bx[c] - ab * ax[c]) * inv).simd_clamp(zero, full), e[1][c]);
        c += 1;
    }
    (e, ok)
}

/// Polish (uber trials and refinement) skips a batch's (mode, partition) encoding unless some
/// lane's unpolished error is within this factor of the lowest error seen for that lane so far:
/// mode 6, every mode's unpolished candidates, then each polished result. Polish rarely cuts an
/// error by a third. Effort 8 on Kodak-24: 1.72× faster at -0.006 dB; BC7 RDO λ 1: 15% less CPU
/// time at -0.04% size and -0.004 dB.
const POLISH_REACH: f32 = 1.5;

/// [`uber_multi`]'s trials for mode 6 in the `active` lanes (16-level indices, 8-bit endpoints).
#[inline(always)]
fn uber6<V: SimdF32>(t: &Texels<V>, ends: &mut Ends<V>, idx: &mut [V; 16], err: &mut V, active: V::Mask, level: u8) {
    let zero = V::splat(0.0);
    let maxsel = 15.0f32;
    let top = V::splat(maxsel);
    let (mut lo, mut hi) = (top, zero);
    let mut i = 0;
    while i < 16 {
        lo = lo.simd_min(idx[i]);
        hi = hi.simd_max(idx[i]);
        i += 1;
    }
    let base = *idx;
    let prev = (
        [ends.v[0][0], ends.v[0][1], ends.v[0][2]],
        [ends.v[1][0], ends.v[1][1], ends.v[1][2]],
    );
    let q = level as i32 - 1;
    let nt = 3 + if level >= 2 {
        ((q + 2) * (q + 2) - 1) as usize
    } else {
        0
    };
    let mut k = 0;
    while k < nt {
        let mut idx2 = base;
        if k < 3 {
            let mut i = 0;
            while i < 16 {
                let x = base[i];
                let up = (x - lo).simd_max(lo - x).simd_lt(V::splat(0.5)) & x.simd_lt(top);
                let down = (x - hi).simd_max(hi - x).simd_lt(V::splat(0.5)) & zero.simd_lt(x);
                idx2[i] = match k {
                    0 => V::select(up, x + V::splat(1.0), x),
                    1 => V::select(down, x - V::splat(1.0), x),
                    _ => V::select(up, x + V::splat(1.0), V::select(down, x - V::splat(1.0), x)),
                };
                i += 1;
            }
        } else {
            let mut n = k - 3;
            let side = (q + 2) as usize;
            let mut ly = (n / side) as i32 - q;
            let mut hy = (n % side) as i32 + maxsel as i32 - 1;
            if ly > 0 || (ly == 0 && hy >= maxsel as i32) {
                n += 1;
                ly = (n / side) as i32 - q;
                hy = (n % side) as i32 + maxsel as i32 - 1;
            }
            let scale = V::splat(maxsel / (hy - ly) as f32);
            let l = V::splat(ly as f32);
            let mut i = 0;
            while i < 16 {
                idx2[i] = floor(((base[i] - l) * scale + V::splat(0.5)).simd_max(zero)).simd_min(top);
                i += 1;
            }
        }
        let (a, b, ok) = ls_ends(t, &[V::splat(1.0); 16], &idx2, 16, prev);
        let cand = quantize(&a, &b);
        let (cidx, cerr) = project(t, &cand);
        let take = active & ok & cerr.simd_lt(*err);
        if V::mask_any(take) {
            *err = V::select(take, cerr, *err);
            let mut kk = 0;
            while kk < 2 {
                let mut c = 0;
                while c < 4 {
                    ends.q[kk][c] = V::select(take, cand.q[kk][c], ends.q[kk][c]);
                    ends.v[kk][c] = V::select(take, cand.v[kk][c], ends.v[kk][c]);
                    c += 1;
                }
                ends.p[kk] = V::select(take, cand.p[kk], ends.p[kk]);
                kk += 1;
            }
            let mut i = 0;
            while i < 16 {
                idx[i] = V::select(take, cidx[i], idx[i]);
                i += 1;
            }
        }
        k += 1;
    }
}

/// A mode's best encoding per lane: its subsets (2 or 3 used) and the partition.
type TwoBest<V> = ([Sub<V>; 3], <V as SimdF32>::Int);

/// Per lane: `a` where `take` is set, else `b`.
#[inline(always)]
fn merge_sub<V: SimdF32>(take: V::Mask, a: &Sub<V>, b: &Sub<V>) -> Sub<V> {
    let mut out = *b;
    let mut k = 0;
    while k < 2 {
        let mut c = 0;
        while c < 3 {
            out.q[k][c] = V::select(take, a.q[k][c], b.q[k][c]);
            out.v[k][c] = V::select(take, a.v[k][c], b.v[k][c]);
            c += 1;
        }
        out.p[k] = V::select(take, a.p[k], b.p[k]);
        k += 1;
    }
    let mut i = 0;
    while i < 16 {
        out.idx[i] = V::select(take, a.idx[i], b.idx[i]);
        i += 1;
    }
    out.err = V::select(take, a.err, b.err);
    out
}

/// Popcount of the low 16 bits of each lane (four nibble lookups).
#[inline(always)]
fn popcount16<V: SimdF32>(x: V::Int) -> V::Int {
    const NIB: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];
    let f = V::Int::splat(15);
    V::Int::lookup16(&NIB, x & f)
        + V::Int::lookup16(&NIB, (x >> 4) & f)
        + V::Int::lookup16(&NIB, (x >> 8) & f)
        + V::Int::lookup16(&NIB, (x >> 12) & f)
}

/// Each lane's `K` 2-subset partitions nearest (in Hamming distance, either polarity) to the
/// split of its texels by the side of the mean they fall on along the principal axis (bc7f's
/// estimate), best first.
#[inline(always)]
fn guess_partitions<V: SimdF32, const K: usize>(t: &Texels<V>, mean: &[V; 3], axis: &[V; 3]) -> [V::Int; K] {
    let zero = V::splat(0.0);
    let mut bits = V::Int::splat(0);
    let mut i = 0;
    while i < 16 {
        let s = (t[0][i] - mean[0]).mul_add(
            axis[0],
            (t[1][i] - mean[1]).mul_add(axis[1], (t[2][i] - mean[2]) * axis[2]),
        );
        let side = V::Int::select(zero.simd_lt(s), V::Int::splat(1 << i), V::Int::splat(0));
        bits = bits | side;
        i += 1;
    }
    let mut best = [V::Int::splat(99); K];
    let mut best_p = [V::Int::splat(0); K];
    let sixteen = V::Int::splat(16);
    // popcount(a ^ b) = popcount(a) + popcount(b) - 2 popcount(a & b).
    let pc = popcount16::<V>(bits);
    let mut p = 0;
    while p < 64 {
        let part = super::tables::PART2[p];
        let common = popcount16::<V>(bits & V::Int::splat(part as i32));
        let d = pc + V::Int::splat(part.count_ones() as i32) - common - common;
        let mut d = d.simd_min(sixteen - d);
        let mut id = V::Int::splat(p as i32);
        let mut k = 0;
        while k < K {
            let better = d.simd_lt(best[k]);
            let (ob, op) = (best[k], best_p[k]);
            best[k] = V::Int::select(better, d, ob);
            best_p[k] = V::Int::select(better, id, op);
            d = V::Int::select(better, ob, d);
            id = V::Int::select(better, op, id);
            k += 1;
        }
        p += 1;
    }
    best_p
}

/// `tab[k][m][v]` of [`moment_tables`].
type MomentTables<V> = [[[V; 9]; 16]; 4];

/// Per-nibble subset moment tables for [`rank_fits`] and [`rank_fits3`]: `tab[k][m][v]` is moment
/// `v` (r, g, b, rr, rg, rb, gg, gb, bb) summed over the texels `4k + j` with bit `j` set in `m`,
/// and the block's totals. Texels are integers, so every sum is exact.
#[inline(always)]
fn moment_tables<V: SimdF32>(t: &Texels<V>) -> (MomentTables<V>, [V; 9]) {
    let zero = V::splat(0.0);
    // Texels and their products: 9 moments per texel (integers below 2^16).
    let mut val = [[zero; 16]; 9];
    let mut i = 0;
    while i < 16 {
        let (r, g, b) = (t[0][i], t[1][i], t[2][i]);
        val[0][i] = r;
        val[1][i] = g;
        val[2][i] = b;
        val[3][i] = r * r;
        val[4][i] = r * g;
        val[5][i] = r * b;
        val[6][i] = g * g;
        val[7][i] = g * b;
        val[8][i] = b * b;
        i += 1;
    }
    // tab[k][m][v]: moment v summed over the texels 4k + j with bit j set in m.
    let mut tab = [[[zero; 9]; 16]; 4];
    let mut k = 0;
    while k < 4 {
        let mut m = 1usize;
        while m < 16 {
            let low = m & (16 - m) & 15; // lowest set bit (m & -m)
            let j = low.trailing_zeros() as usize;
            let rest = m & !low;
            let mut v = 0;
            while v < 9 {
                tab[k][m][v] = tab[k][rest][v] + val[v][4 * k + j];
                v += 1;
            }
            m += 1;
        }
        k += 1;
    }
    let mut tot = tab[0][15];
    let mut k = 1;
    while k < 4 {
        let mut v = 0;
        while v < 9 {
            tot[v] += tab[k][15][v];
            v += 1;
        }
        k += 1;
    }
    (tab, tot)
}

/// One subset's error estimates for [`rank_fits`] and [`rank_fits3`]: `n` texels with moment sums
/// `sm` (Σr, Σg, Σb, Σrr, Σrg, Σrb, Σgg, Σgb, Σbb), from the block's 10-bit `axis`. Returns the
/// first `R` index sizes' estimates (3-bit, 2-bit) in 2^cs / 16 units and the scale 2^-cs.
#[inline(always)]
fn subset_est<V: SimdF32, const R: usize>(n: f32, sm: &[V; 9], axis: &[V; 3]) -> ([V; 2], V) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    // (a, b) of est = (a tr - b λ) / n: Δw² = 1/49 (3-bit), 1/9 (2-bit).
    const AB: [(f32, f32); 2] = [(49.0, 48.0), (9.0, 8.0)];
    let mut est = [zero; 2];
    // n Σttᵀ - Σt Σtᵀ (each product below 2^24, the difference below 2^23), cut to
    // 10 bits.
    let nv = V::splat(n);
    let ns = [zero - sm[0], zero - sm[1], zero - sm[2]];
    let c = [
        ns[0].mul_add(sm[0], sm[3] * nv),
        ns[0].mul_add(sm[1], sm[4] * nv),
        ns[0].mul_add(sm[2], sm[5] * nv),
        ns[1].mul_add(sm[1], sm[6] * nv),
        ns[1].mul_add(sm[2], sm[7] * nv),
        ns[2].mul_add(sm[2], sm[8] * nv),
    ];
    let sc = shrink(c[0].simd_max(c[3]).simd_max(c[5]), 9);
    let mut cs = [zero; 6];
    let mut j = 0;
    while j < 6 {
        cs[j] = floor(c[j] * sc);
        j += 1;
    }
    let [c00, c01, c02, c11, c12, c22] = cs;
    // One power step from the 10-bit axis (|v1| < 3 * 2^20), cut to 5 bits.
    let v1 = [
        c00.mul_add(axis[0], c01.mul_add(axis[1], c02 * axis[2])),
        c01.mul_add(axis[0], c11.mul_add(axis[1], c12 * axis[2])),
        c02.mul_add(axis[0], c12.mul_add(axis[1], c22 * axis[2])),
    ];
    let s5 = shrink(abs_max(&v1), 4);
    let v1 = [floor(v1[0] * s5), floor(v1[1] * s5), floor(v1[2] * s5)];
    let v2 = [
        c00.mul_add(v1[0], c01.mul_add(v1[1], c02 * v1[2])),
        c01.mul_add(v1[0], c11.mul_add(v1[1], c12 * v1[2])),
        c02.mul_add(v1[0], c12.mul_add(v1[1], c22 * v1[2])),
    ];
    // λ = floor(v1·v2 / |v1|²): |v1|² < 3 * 2^10 and the cut matrix's eigenvalues are
    // below 3 * 2^10, so the numerator is below 2^24 and the quotient below 2^12, the
    // division's estimate within one ([`div_floor`]); 0 when the step vanishes.
    let num = v1[0].mul_add(v2[0], v1[1].mul_add(v2[1], v1[2] * v2[2]));
    let den = v1[0].mul_add(v1[0], v1[1].mul_add(v1[1], v1[2] * v1[2])).simd_max(one);
    let lam = div_floor(num, den, one / den).0.simd_max(zero);
    let tr = c00 + c11 + c22;
    let inv_n = V::splat(1.0 / n);
    let mut r = 0;
    while r < R {
        let (a, b) = AB[r];
        // 16 x < 2^22 for x = a tr - b λ <= 49 * 3069.
        let x = lam.mul_add(V::splat(-b), tr * V::splat(a)).simd_max(zero);
        est[r] = div_floor_q(x * V::splat(16.0), inv_n);
        r += 1;
    }
    (est, sc)
}

/// Ranks all 64 2-subset partitions per lane by an estimate of their error with 3-bit and with
/// 2-bit indices: per subset, `trace(S) - λ + Δw² λ` with λ the Rayleigh quotient after one
/// power step from the block's principal axis (`Δw` = index step, as the batch encoder's
/// ranking), in exact integers so the GPU kernel ranks the same: the scatter matrix
/// n S = n Σttᵀ - Σt Σtᵀ cut to 10 bits (by its largest diagonal), the power step's vector cut to
/// 5 bits (a Rayleigh quotient's error is quadratic in the vector's), λ floored, then
/// (a tr - b λ) / n for Δw² = 1 - b / a (49/48 and 9/8) in 1/16 units floored, the two subsets
/// scaled back and summed with one rounding. Subset sums come from per-nibble subset tables: the membership
/// of a partition is the same in every lane. Returns each lane's `K` best partitions for the
/// first `R` index sizes (3-bit, then 2-bit; the rest left 0), best first.
#[inline(always)]
fn rank_fits<V: SimdF32, const K: usize, const R: usize>(t: &Texels<V>, axis: &[V; 3]) -> [[V::Int; K]; 2] {
    let zero = V::splat(0.0);
    let (tab, tot) = moment_tables(t);
    let mut best = [[V::splat(f32::INFINITY); K]; 2];
    let mut best_p = [[V::Int::splat(0); K]; 2];
    let mut p = 0;
    while p < 64 {
        let bits = super::tables::PART2[p];
        let n1 = bits.count_ones() as f32;
        let mut s1 = [zero; 9];
        let mut k = 0;
        while k < 4 {
            let m = ((bits >> (4 * k)) & 15) as usize;
            let mut v = 0;
            while v < 9 {
                s1[v] += tab[k][m][v];
                v += 1;
            }
            k += 1;
        }
        // Per subset: each index size's estimate (in 2^cs / 16 units) and the scale 2^-cs.
        let mut s0 = [zero; 9];
        let mut v = 0;
        while v < 9 {
            s0[v] = tot[v] - s1[v];
            v += 1;
        }
        let (est0, sc0) = subset_est::<V, R>(16.0 - n1, &s0, axis);
        let (est1, sc1) = subset_est::<V, R>(n1, &s1, axis);
        let est = [est0, est1];
        let scale = [sc0, sc1];
        // est0 2^cs0 + est1 2^cs1 in one rounded f32 addition: the products are exact (powers
        // of two; a scale's reciprocal is an exponent flip), so the sum is the exact sum rounded
        // once, the same under any fusing.
        let (up0, up1) = (scale[0].pow2_recip(), scale[1].pow2_recip());
        let mut r = 0;
        while r < R {
            let mut e = est[0][r] * up0 + est[1][r] * up1;
            let mut id = V::Int::splat(p as i32);
            let mut k = 0;
            while k < K {
                let better = e.simd_lt(best[r][k]);
                let (ob, op) = (best[r][k], best_p[r][k]);
                best[r][k] = V::select(better, e, ob);
                best_p[r][k] = V::Int::select(better, id, op);
                e = V::select(better, ob, e);
                id = V::Int::select(better, op, id);
                k += 1;
            }
            r += 1;
        }
        p += 1;
    }
    best_p
}

/// Subset-1 and subset-2 membership bits of each 3-subset partition.
const PART3_MASKS: [(u16, u16); 64] = {
    let mut out = [(0u16, 0u16); 64];
    let mut p = 0;
    while p < 64 {
        let code = super::tables::PART3[p];
        let (mut m1, mut m2) = (0u16, 0u16);
        let mut i = 0;
        while i < 16 {
            let sub = (code >> (2 * i)) & 3;
            if sub == 1 {
                m1 |= 1 << i;
            } else if sub == 2 {
                m2 |= 1 << i;
            }
            i += 1;
        }
        out[p] = (m1, m2);
        p += 1;
    }
    out
};

/// Ranks 3-subset partitions per lane like [`rank_fits`], in the same exact integers: mode 0's 16
/// with 3-bit indices and mode 2's 64 with 2-bit indices. The three subsets' estimates are cut to
/// the coarsest one's scale (floored) and summed as integers, so the sum is exact and the GPU
/// kernel ranks the same under any reassociation. Returns each lane's `K` best for each, best
/// first.
#[inline(always)]
fn rank_fits3<V: SimdF32, const K: usize>(t: &Texels<V>, axis: &[V; 3]) -> [[V::Int; K]; 2] {
    let zero = V::splat(0.0);
    let (tab, tot) = moment_tables(t);
    let mut best = [[V::splat(f32::INFINITY); K]; 2];
    let mut best_p = [[V::Int::splat(0); K]; 2];
    let mut p = 0;
    while p < 64 {
        let (m1, m2) = PART3_MASKS[p];
        let mut s1 = [zero; 9];
        let mut s2 = [zero; 9];
        let mut k = 0;
        while k < 4 {
            let a = ((m1 >> (4 * k)) & 15) as usize;
            let b = ((m2 >> (4 * k)) & 15) as usize;
            let mut v = 0;
            while v < 9 {
                s1[v] += tab[k][a][v];
                s2[v] += tab[k][b][v];
                v += 1;
            }
            k += 1;
        }
        let mut s0 = [zero; 9];
        let mut v = 0;
        while v < 9 {
            s0[v] = tot[v] - s1[v] - s2[v];
            v += 1;
        }
        // Every subset of a BC7 partition holds at least its anchor texel.
        let n1 = m1.count_ones() as f32;
        let n2 = m2.count_ones() as f32;
        let (e0, c0) = subset_est::<V, 2>(16.0 - n1 - n2, &s0, axis);
        let (e1, c1) = subset_est::<V, 2>(n1, &s1, axis);
        let (e2, c2) = subset_est::<V, 2>(n2, &s2, axis);
        // est_i 2^cs_i summed at the coarsest scale 2^cs (cs = max cs_i): each term
        // floor(est_i 2^(cs_i - cs)) is exact (a power-of-two scaling, then a floor), the integer
        // sum stays below 3 * 2^22, and the final scaling by 2^cs is exact.
        let fine = c0.simd_min(c1).simd_min(c2); // 2^-cs
        let up = fine.pow2_recip();
        let ups = [c0.pow2_recip() * fine, c1.pow2_recip() * fine, c2.pow2_recip() * fine];
        // Mode 0 only has the first 16 partitions.
        let mut r = usize::from(p >= 16);
        while r < 2 {
            let sum = floor(e0[r] * ups[0]) + floor(e1[r] * ups[1]) + floor(e2[r] * ups[2]);
            let mut e = sum * up;
            let mut id = V::Int::splat(p as i32);
            let mut k = 0;
            while k < K {
                let better = e.simd_lt(best[r][k]);
                let (ob, op) = (best[r][k], best_p[r][k]);
                best[r][k] = V::select(better, e, ob);
                best_p[r][k] = V::Int::select(better, id, op);
                e = V::select(better, ob, e);
                id = V::Int::select(better, op, id);
                k += 1;
            }
            r += 1;
        }
        p += 1;
    }
    best_p
}

/// Subset membership masks (as 0/1 floats) of each lane's partition for `mode` (the third is
/// all zero for 2-subset modes).
#[inline(always)]
fn part_masks<V: SimdF32>(part: V::Int, mode: Two) -> [[V; 16]; 3] {
    let parts = part.to_array();
    let one = V::Int::splat(1);
    let mut out = [[V::splat(0.0); 16]; 3];
    if mode.subsets() == 2 {
        let codes = V::Int::from_array(parts.map(|p| super::tables::PART2[p as usize] as i32));
        let mut i = 0;
        while i < 16 {
            let s1 = ((codes >> i as i32) & one).to_float();
            out[1][i] = s1;
            out[0][i] = V::splat(1.0) - s1;
            i += 1;
        }
    } else {
        let m1 = V::Int::from_array(parts.map(|p| PART3_MASKS[p as usize].0 as i32));
        let m2 = V::Int::from_array(parts.map(|p| PART3_MASKS[p as usize].1 as i32));
        let mut i = 0;
        while i < 16 {
            let a = ((m1 >> i as i32) & one).to_float();
            let b = ((m2 >> i as i32) & one).to_float();
            out[1][i] = a;
            out[2][i] = b;
            out[0][i] = V::splat(1.0) - a - b;
            i += 1;
        }
    }
    out
}

/// Packs the lanes set in `take` of a multi-subset encoding into `over`.
#[inline(always)]
fn pack_two_lanes<V: SimdF32>(
    mode: Two,
    part: V::Int,
    subs: &[Sub<V>; 3],
    take: [bool; LANES],
    over: &mut [Option<[u8; 16]>; LANES],
) {
    let ns = mode.subsets();
    let parts = part.to_array();
    let mut qa = [[[[0f32; LANES]; 3]; 2]; 3];
    let mut pa = [[[0f32; LANES]; 2]; 3];
    let mut ia = [[[0f32; LANES]; 16]; 3];
    for s in 0..ns {
        for e in 0..2 {
            for c in 0..3 {
                qa[s][e][c] = subs[s].q[e][c].to_array();
            }
            pa[s][e] = subs[s].p[e].to_array();
        }
        for i in 0..16 {
            ia[s][i] = subs[s].idx[i].to_array();
        }
    }
    for l in 0..LANES {
        if !take[l] {
            continue;
        }
        let part = parts[l] as u8;
        let mut q = [[[0u8; 3]; 2]; 3];
        let mut pb = [[0u8; 2]; 3];
        for s in 0..ns {
            for e in 0..2 {
                for c in 0..3 {
                    q[s][e][c] = qa[s][e][c][l] as u8;
                }
                pb[s][e] = pa[s][e][l] as u8;
            }
        }
        let mut ix = [0u8; 16];
        for (i, x) in ix.iter_mut().enumerate() {
            *x = ia[super::subset_of(ns as u8, part, i)][i][l] as u8;
        }
        over[l] = Some(pack_two(mode, part, &q, &pb, &ix));
    }
}

/// Scalar packing of a multi-subset block (anchor fix-up included).
fn pack_two(mode: Two, part: u8, sub: &[[[u8; 3]; 2]; 3], pb: &[[u8; 2]; 3], idx: &[u8; 16]) -> [u8; 16] {
    let ns = mode.subsets();
    let n = 1u8 << mode.index_bits();
    let anc = super::anchors(ns as u8, part);
    let mut blk = super::Bc7Block {
        mode: mode.mode(),
        partition: part,
        ..Default::default()
    };
    let mut ix = *idx;
    for s in 0..ns {
        let mut ep = [
            [sub[s][0][0], sub[s][0][1], sub[s][0][2], 0],
            [sub[s][1][0], sub[s][1][1], sub[s][1][2], 0],
        ];
        let mut p = pb[s];
        if ix[anc[s]] >= n / 2 {
            ep.swap(0, 1);
            p.swap(0, 1);
            for (i, x) in ix.iter_mut().enumerate() {
                if super::subset_of(ns as u8, part, i) == s {
                    *x = n - 1 - *x;
                }
            }
        }
        blk.ep[s] = ep;
        blk.pbits[s] = p;
    }
    blk.idx = ix;
    super::pack(&blk)
}

/// A mode-5 encoding per lane (one rotation): 7-bit color levels (the rotated channel's slot
/// holds 127, which decodes to alpha 255 after the swap), the 8-bit scalar-plane endpoints, both
/// index sets and the error.
#[derive(Clone, Copy)]
struct M5<V: SimdF32> {
    cq: [[V; 3]; 2],
    av: [V; 2],
    cidx: [V; 16],
    aidx: [V; 16],
    err: V,
}

/// Nearest 7-bit level of 8-bit targets (multiples of 1/16; bit replication `2q + q / 64`): the
/// exactly rounded level ([`round_level`]) and its neighbours. Returns (level, value).
#[inline(always)]
fn quant7<V: SimdF32>(x: V) -> (V, V) {
    let zero = V::splat(0.0);
    let c = round_level(x, zero, 127.0, false);
    let (mut bq, mut bv, mut be) = (zero, zero, V::splat(f32::INFINITY));
    let mut off = -1;
    while off <= 1 {
        let q = (c + V::splat(off as f32)).simd_clamp(zero, V::splat(127.0));
        let v = q.mul_add(V::splat(2.0), floor(q * V::splat(1.0 / 64.0)));
        let d = v - x;
        let e = d * d;
        let take = e.simd_lt(be);
        bq = V::select(take, q, bq);
        bv = V::select(take, v, bv);
        be = V::select(take, e, be);
        off += 1;
    }
    (bq, bv)
}

/// Mode 5 with channel `r` in the scalar (alpha) plane, in exact integers like the mode-1 subset
/// fits (so a GPU kernel can match it): the scalar plane's min/max (8-bit endpoints), then a
/// least-squares refit rounded to integers; the other two channels' principal axis, endpoints at
/// the extreme projections and one refit (7-bit levels). Both planes get exact 2-bit indices,
/// computed on texels with the other channels zeroed, so the 3-channel helpers serve both.
#[inline(always)]
fn encode_m5<V: SimdF32>(t: &Texels<V>, r: usize) -> M5<V> {
    let zero = V::splat(0.0);
    let half = V::splat(0.5);
    let ones = [V::splat(1.0); 16];
    let (u, w) = match r {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    // Scalar plane.
    let ts: Texels<V> = [t[r], [zero; 16], [zero; 16]];
    let (mut lo, mut hi) = (t[r][0], t[r][0]);
    let mut i = 1;
    while i < 16 {
        lo = lo.simd_min(t[r][i]);
        hi = hi.simd_max(t[r][i]);
        i += 1;
    }
    let plane = |a: V, b: V| [[a, zero, zero], [b, zero, zero]];
    let (mut aidx, mut aerr) = index_two(&ts, &ones, &plane(lo, hi), Two::M2);
    let mut av = [lo, hi];
    let (na, nb, _) = ls_ends(&ts, &ones, &aidx, 4, ([lo, zero, zero], [hi, zero, zero]));
    // Multiples of 1/16 in 0..=255: rounding is exact.
    let (na, nb) = (floor(na[0] + half), floor(nb[0] + half));
    let (cidx2, cerr2) = index_two(&ts, &ones, &plane(na, nb), Two::M2);
    let take = cerr2.simd_lt(aerr);
    av = [V::select(take, na, av[0]), V::select(take, nb, av[1])];
    aerr = V::select(take, cerr2, aerr);
    let mut i = 0;
    while i < 16 {
        aidx[i] = V::select(take, cidx2[i], aidx[i]);
        i += 1;
    }
    // Color plane (channels u, w): n Σxy - Σx Σy over all 16 texels, the 3-channel principal
    // axis with the third channel empty, a 7-bit axis and the extreme projections.
    let tc: Texels<V> = [t[u], t[w], [zero; 16]];
    let n = V::splat(16.0);
    let sum = [sum16!(|i| tc[0][i]), sum16!(|i| tc[1][i]), zero];
    let cov = [
        dot16!(|i| tc[0][i], tc[0][i]) * n - sum[0] * sum[0],
        dot16!(|i| tc[0][i], tc[1][i]) * n - sum[0] * sum[1],
        zero,
        dot16!(|i| tc[1][i], tc[1][i]) * n - sum[1] * sum[1],
        zero,
        zero,
    ];
    let v = principal_axis(cov, 3);
    let ws = {
        let s = shrink(abs_max(&v), 6);
        [floor(v[0] * s), floor(v[1] * s), floor(v[2] * s)]
    };
    let (ehi, elo) = subset_ends(&tc, &ones, n, &sum, &ws);
    let inv16 = V::splat(1.0 / 16.0);
    let mut e0 = [elo[0] * inv16, elo[1] * inv16, zero];
    let mut e1 = [ehi[0] * inv16, ehi[1] * inv16, zero];
    let mut best_q = [[zero; 2]; 2];
    let mut cidx = [zero; 16];
    let mut cerr = V::splat(f32::INFINITY);
    let mut it = 0;
    while it < 2 {
        let (q00, v00) = quant7(e0[0]);
        let (q01, v01) = quant7(e0[1]);
        let (q10, v10) = quant7(e1[0]);
        let (q11, v11) = quant7(e1[1]);
        let (ix, er) = index_two(&tc, &ones, &[[v00, v01, zero], [v10, v11, zero]], Two::M2);
        let take = er.simd_lt(cerr);
        cerr = V::select(take, er, cerr);
        let qs = [[q00, q01], [q10, q11]];
        let mut k = 0;
        while k < 2 {
            let mut c = 0;
            while c < 2 {
                best_q[k][c] = V::select(take, qs[k][c], best_q[k][c]);
                c += 1;
            }
            k += 1;
        }
        let mut i = 0;
        while i < 16 {
            cidx[i] = V::select(take, ix[i], cidx[i]);
            i += 1;
        }
        if it == 0 {
            let (a, b, _) = ls_ends(&tc, &ones, &cidx, 4, (e0, e1));
            (e0, e1) = ([a[0], a[1], zero], [b[0], b[1], zero]);
        }
        it += 1;
    }
    let seven = V::splat(127.0);
    let mut cq = [[seven; 3]; 2];
    let mut k = 0;
    while k < 2 {
        cq[k][u] = best_q[k][0];
        cq[k][w] = best_q[k][1];
        k += 1;
    }
    M5 {
        cq,
        av,
        cidx,
        aidx,
        err: aerr + cerr,
    }
}

/// Scalar packing of a mode-5 block with rotation `rot` (1-3), anchor fix-up per plane.
fn pack_m5(rot: u8, cq: &[[u8; 3]; 2], av: &[u8; 2], cidx: &[u8; 16], aidx: &[u8; 16]) -> [u8; 16] {
    let mut blk = super::Bc7Block {
        mode: 5,
        rotation: rot,
        ..Default::default()
    };
    let mut ep = [
        [cq[0][0], cq[0][1], cq[0][2], av[0]],
        [cq[1][0], cq[1][1], cq[1][2], av[1]],
    ];
    let (mut ci, mut ai) = (*cidx, *aidx);
    if ci[0] >= 2 {
        for c in 0..3 {
            let x = ep[0][c];
            ep[0][c] = ep[1][c];
            ep[1][c] = x;
        }
        for x in &mut ci {
            *x = 3 - *x;
        }
    }
    if ai[0] >= 2 {
        let x = ep[0][3];
        ep[0][3] = ep[1][3];
        ep[1][3] = x;
        for x in &mut ai {
            *x = 3 - *x;
        }
    }
    blk.ep[0] = ep;
    blk.idx = ci;
    blk.idx2 = ai;
    super::pack(&blk)
}

/// Packs eight mode-6 blocks: four 32-bit words per lane, anchor fix-up included.
#[inline(always)]
fn pack<V: SimdF32>(e: &Ends<V>, idx: &[V; 16]) -> [V::Int; 4] {
    // Anchor (texel 0) index must be < 8: otherwise swap the endpoints and invert the indices.
    let flip = V::splat(7.5).simd_lt(idx[0]);
    let fifteen = V::splat(15.0);
    let mut q = [[V::Int::splat(0); 4]; 2];
    let mut c = 0;
    while c < 4 {
        q[0][c] = V::select(flip, e.q[1][c], e.q[0][c]).to_int_trunc();
        q[1][c] = V::select(flip, e.q[0][c], e.q[1][c]).to_int_trunc();
        c += 1;
    }
    let p0 = V::select(flip, e.p[1], e.p[0]).to_int_trunc();
    let p1 = V::select(flip, e.p[0], e.p[1]).to_int_trunc();
    let mut ix = [V::Int::splat(0); 16];
    let mut i = 0;
    while i < 16 {
        ix[i] = V::select(flip, fifteen - idx[i], idx[i]).to_int_trunc();
        i += 1;
    }
    // Bits: mode 6 = bit 6; R0 7, R1 14, G0 21, G1 28, B0 35, B1 42, A0 49, A1 56; P0 63, P1 64;
    // index 0 (3 bits) at 65, index i >= 1 at 64 + 4i.
    let w0 = V::Int::splat(1 << 6)
        | (q[0][0] << 7)
        | (q[1][0] << 14)
        | (q[0][1] << 21)
        | ((q[1][1] & V::Int::splat(15)) << 28);
    let w1 = (q[1][1] >> 4) | (q[0][2] << 3) | (q[1][2] << 10) | (q[0][3] << 17) | (q[1][3] << 24) | (p0 << 31);
    let mut w2 = p1 | (ix[0] << 1);
    let mut i = 1;
    while i < 8 {
        w2 = w2 | (ix[i] << (4 * i as i32));
        i += 1;
    }
    let mut w3 = V::Int::splat(0);
    while i < 16 {
        w3 = w3 | (ix[i] << (4 * (i - 8) as i32));
        i += 1;
    }
    [w0, w1, w2, w3]
}

/// Largest absolute component.
#[inline(always)]
fn abs_max<V: SimdF32>(v: &[V; 3]) -> V {
    let z = V::splat(0.0);
    v[0].simd_max(z - v[0])
        .simd_max(v[1].simd_max(z - v[1]))
        .simd_max(v[2].simd_max(z - v[2]))
}

/// The endpoints (high, low) of an integer axis `ws` (|ws| < 2^7, not all 0) through the mean,
/// in 1/16 units (floored) clamped to 0..=4080: with p = t·ws, the extreme projections relative
/// to the mean's are τ = (16 pmax - Σp) / 16 and (16 pmin - Σp) / 16, and the endpoints are
/// mean + ws τ / |ws|². Exact integers ([`div_floor`]): |16 p - Σp| < 2^21, |ws|² < 3 * 2^14.
#[inline(always)]
fn line_ends<V: SimdF32>(t: &Texels<V>, sum: &[V; 3], ws: &[V; 3]) -> ([V; 3], [V; 3]) {
    let pmin = reduce16!(|i| t[0][i] * ws[0] + t[1][i] * ws[1] + t[2][i] * ws[2], |a, b| a
        .simd_min(b));
    let pmax = reduce16!(|i| t[0][i] * ws[0] + t[1][i] * ws[1] + t[2][i] * ws[2], |a, b| a
        .simd_max(b));
    let pm = sum[0] * ws[0] + sum[1] * ws[1] + sum[2] * ws[2];
    let n2 = ws[0] * ws[0] + ws[1] * ws[1] + ws[2] * ws[2];
    let inv = V::splat(1.0) / n2;
    let s16 = V::splat(16.0);
    let (qh, rh) = div_floor(pmax * s16 - pm, n2, inv);
    let (ql, rl) = div_floor(pmin * s16 - pm, n2, inv);
    (
        [
            axis_point(sum[0], ws[0], qh, rh, n2, inv),
            axis_point(sum[1], ws[1], qh, rh, n2, inv),
            axis_point(sum[2], ws[2], qh, rh, n2, inv),
        ],
        [
            axis_point(sum[0], ws[0], ql, rl, n2, inv),
            axis_point(sum[1], ws[1], ql, rl, n2, inv),
            axis_point(sum[2], ws[2], ql, rl, n2, inv),
        ],
    )
}

/// One channel of a [`line_ends`] endpoint: sc + floor(wc τ16 / n2) for τ16 = q n2 + r (every
/// term below 2^23), clamped to 0..=4080.
#[inline(always)]
fn axis_point<V: SimdF32>(sc: V, wc: V, q: V, r: V, n2: V, inv: V) -> V {
    (sc + wc * q + div_floor(wc * r, n2, inv).0).simd_clamp(V::splat(0.0), V::splat(4080.0))
}

/// Principal axis of a covariance scaled to integers, `c` = [rr, rg, rb, gg, gb, bb] (n Σxy - Σx Σy,
/// below 2^24): the matrix cut to 11 bits (its largest diagonal bounds every entry), then `power`
/// iterations from the largest-diagonal row (the first on ties), each step's vector cut to
/// 10 bits so every product stays below 2^24. Flat lanes get (1, 1, 1).
#[inline(always)]
fn principal_axis<V: SimdF32>(c: [V; 6], power: u8) -> [V; 3] {
    let [rr, rg, rb, gg, gb, bb] = c;
    let cs = shrink(rr.simd_max(gg).simd_max(bb), 10);
    let (srr, srg, srb, sgg, sgb, sbb) = (
        floor(rr * cs),
        floor(rg * cs),
        floor(rb * cs),
        floor(gg * cs),
        floor(gb * cs),
        floor(bb * cs),
    );
    let m = rr.simd_lt(gg);
    let mut v = [V::select(m, srg, srr), V::select(m, sgg, srg), V::select(m, sgb, srb)];
    let m = rr.simd_max(gg).simd_lt(bb);
    v = [
        V::select(m, srb, v[0]),
        V::select(m, sgb, v[1]),
        V::select(m, sbb, v[2]),
    ];
    let mut it = 0;
    while it < power {
        let w = [
            srr * v[0] + srg * v[1] + srb * v[2],
            srg * v[0] + sgg * v[1] + sgb * v[2],
            srb * v[0] + sgb * v[1] + sbb * v[2],
        ];
        let s = shrink(abs_max(&w), 9);
        v = [floor(w[0] * s), floor(w[1] * s), floor(w[2] * s)];
        it += 1;
    }
    let flat = abs_max(&v).simd_lt(V::splat(0.5));
    let one = V::splat(1.0);
    [
        V::select(flat, one, v[0]),
        V::select(flat, one, v[1]),
        V::select(flat, one, v[2]),
    ]
}

/// The endpoints (high, low) of the texels in `m` (0/1 per texel; `n` members, channel sums
/// `sum`) on the integer axis `ws` through their mean, in 1/16 units: as [`line_ends`] in
/// 1/n units (τ = n p_ext - Σp over the members), clamped to 0..=255 n, then floor(16 E / n).
#[inline(always)]
fn subset_ends<V: SimdF32>(t: &Texels<V>, m: &[V; 16], n: V, sum: &[V; 3], ws: &[V; 3]) -> ([V; 3], [V; 3]) {
    let (zero, big) = (V::splat(0.0), V::splat(1.0e7));
    let mut p = [zero; 16];
    let mut i = 0;
    while i < 16 {
        p[i] = t[0][i] * ws[0] + t[1][i] * ws[1] + t[2][i] * ws[2];
        i += 1;
    }
    let pmin = reduce16!(|i| V::select(zero.simd_lt(m[i]), p[i], big), |a, b| a.simd_min(b));
    let pmax = reduce16!(|i| V::select(zero.simd_lt(m[i]), p[i], zero - big), |a, b| a
        .simd_max(b));
    let pm = sum[0] * ws[0] + sum[1] * ws[1] + sum[2] * ws[2];
    let n2 = ws[0] * ws[0] + ws[1] * ws[1] + ws[2] * ws[2];
    let inv = V::splat(1.0) / n2;
    let (qh, rh) = div_floor(pmax * n - pm, n2, inv);
    let (ql, rl) = div_floor(pmin * n - pm, n2, inv);
    let (top, inv_n) = (n * V::splat(255.0), V::splat(1.0) / n);
    let mut hi = [zero; 3];
    let mut lo = [zero; 3];
    let mut c = 0;
    while c < 3 {
        hi[c] = line_point(sum[c], qh, rh, ws[c], n2, inv, top, inv_n);
        lo[c] = line_point(sum[c], ql, rl, ws[c], n2, inv, top, inv_n);
        c += 1;
    }
    (hi, lo)
}

/// Encodes eight opaque blocks; returns the four 32-bit words of each lane's mode-6 block and,
/// for lanes where another mode won, that block instead.
#[inline(always)]
fn encode8<V: SimdF32>(t: &Texels<V>, cfg: Cfg) -> ([V::Int; 4], [Option<[u8; 16]>; LANES]) {
    // The line fit is exact integer arithmetic (every value an integer below 2^24 in f32, see
    // `crate::exact`), so it does not depend on rounding, summation order or fused
    // multiply-adds: the GPU kernel (`src/gpu/bc7.wgsl`) computes the same endpoints.
    let sum = [sum16!(|i| t[0][i]), sum16!(|i| t[1][i]), sum16!(|i| t[2][i])];
    let inv16 = V::splat(1.0 / 16.0);
    let mean = [sum[0] * inv16, sum[1] * inv16, sum[2] * inv16];
    // Covariance scaled by 16: 16 Σxy - Σx Σy.
    let s16 = V::splat(16.0);
    let rr = sum16!(|i| t[0][i] * t[0][i]) * s16 - sum[0] * sum[0];
    let rg = sum16!(|i| t[0][i] * t[1][i]) * s16 - sum[0] * sum[1];
    let rb = sum16!(|i| t[0][i] * t[2][i]) * s16 - sum[0] * sum[2];
    let gg = sum16!(|i| t[1][i] * t[1][i]) * s16 - sum[1] * sum[1];
    let gb = sum16!(|i| t[1][i] * t[2][i]) * s16 - sum[1] * sum[2];
    let bb = sum16!(|i| t[2][i] * t[2][i]) * s16 - sum[2] * sum[2];
    let v = principal_axis([rr, rg, rb, gg, gb, bb], cfg.power);
    // The axis cut to 7 bits, and the endpoints: the points of the axis line through the mean
    // that the extreme projections project from (in 1/16 units, see [`line_ends`]).
    let ws = {
        let s = shrink(abs_max(&v), 6);
        [floor(v[0] * s), floor(v[1] * s), floor(v[2] * s)]
    };
    let (hi, lo) = line_ends(t, &sum, &ws);
    let mut e0 = [lo[0] * inv16, lo[1] * inv16, lo[2] * inv16];
    let mut e1 = [hi[0] * inv16, hi[1] * inv16, hi[2] * inv16];
    let mut ends = quantize(&e0, &e1);
    let (mut idx, mut err) = project(t, &ends);
    if cfg.refits > 0 {
        let mut r = 0;
        while r < cfg.refits {
            let (a, b, ok) = ls_ends(t, &[V::splat(1.0); 16], &idx, 16, (e0, e1));
            let cand = quantize(&a, &b);
            let (cidx, cerr) = project(t, &cand);
            let take = ok & cerr.simd_lt(err);
            if !V::mask_any(take) {
                break;
            }
            err = V::select(take, cerr, err);
            let mut k = 0;
            while k < 2 {
                let mut c = 0;
                while c < 4 {
                    ends.q[k][c] = V::select(take, cand.q[k][c], ends.q[k][c]);
                    ends.v[k][c] = V::select(take, cand.v[k][c], ends.v[k][c]);
                    c += 1;
                }
                ends.p[k] = V::select(take, cand.p[k], ends.p[k]);
                k += 1;
            }
            let mut i = 0;
            while i < 16 {
                idx[i] = V::select(take, cidx[i], idx[i]);
                i += 1;
            }
            let mut c = 0;
            while c < 3 {
                e0[c] = V::select(take, a[c], e0[c]);
                e1[c] = V::select(take, b[c], e1[c]);
                c += 1;
            }
            r += 1;
        }
    }
    let mut over = [None; LANES];
    if cfg.two != 0 {
        // Lanes worth a second subset: some error left and some spread (as the batch encoder).
        let busy =
            V::splat(cfg.busy_err).simd_lt(err) & V::splat(16.0 * 16.0 * 4.0).simd_lt(rr.simd_max(gg).simd_max(bb));
        if V::mask_any(busy) && cfg.polish_modes {
            if cfg.uber > 0 {
                uber6(t, &mut ends, &mut idx, &mut err, busy, cfg.uber);
            }
            if cfg.refine > 0 {
                refine6(t, &mut ends, &mut idx, &mut err, busy, cfg.refine);
            }
        }
        if V::mask_any(busy) {
            let k = (cfg.parts as usize).clamp(1, 4);
            // Candidates per mode: [3-bit index ranking (mode 1), 2-bit (mode 3)].
            let cands: [[V::Int; 4]; 2] = if !cfg.rank {
                let g = guess_partitions::<V, 4>(t, &mean, &v);
                [g, g]
            } else if cfg.two & 2 != 0 {
                rank_fits::<V, 4, 2>(t, &v)
            } else {
                rank_fits::<V, 4, 1>(t, &v)
            };
            let cands3: [[V::Int; 4]; 2] = if cfg.three != 0 {
                rank_fits3::<V, 4>(t, &v)
            } else {
                [[V::Int::splat(0); 4]; 2]
            };
            // Best encoding of each mode per lane over its candidates, then the winner.
            const MODES: [Two; 4] = [Two::M1, Two::M3, Two::M0, Two::M2];
            let mut best_err = err;
            let mut winner = V::Int::splat(0);
            let mut mode_best: [Option<TwoBest<V>>; 4] = [None, None, None, None];
            let slot_cfg = |slot: usize| match slot {
                0 => (cfg.two & 1 != 0, &cands[0], k),
                1 => (cfg.two & 2 != 0, &cands[1], k),
                2 => (cfg.three & 1 != 0, &cands3[0], (cfg.parts_three as usize).clamp(1, 4)),
                _ => (cfg.three & 2 != 0, &cands3[1], (cfg.parts_three as usize).clamp(1, 4)),
            };
            let sum_err = |s: &[Sub<V>; 3], ns: usize| {
                if ns == 3 {
                    s[0].err + s[1].err + s[2].err
                } else {
                    s[0].err + s[1].err
                }
            };
            // Every (mode, candidate) unpolished first, so that a polish can be skipped when no
            // lane of the batch is within POLISH_REACH of the lowest error seen for it.
            let polishing = cfg.uber > 0 && (cfg.polish_parts || cfg.polish_modes);
            let mut pre = [[None::<[Sub<V>; 3]>; 4]; 4];
            let mut bound = err;
            for (slot, mode) in MODES.into_iter().enumerate() {
                let (on, list, kk) = slot_cfg(slot);
                if !on {
                    continue;
                }
                let ns = mode.subsets();
                for (ci, part) in list.iter().take(kk).enumerate() {
                    let m = part_masks::<V>(*part, mode);
                    let s0 = encode_two(t, &m[0], mode, cfg.two_refits);
                    let s1 = encode_two(t, &m[1], mode, cfg.two_refits);
                    let s2 = if ns == 3 {
                        encode_two(t, &m[2], mode, cfg.two_refits)
                    } else {
                        s0
                    };
                    let subs = [s0, s1, s2];
                    bound = bound.simd_min(sum_err(&subs, ns));
                    pre[slot][ci] = Some(subs);
                }
            }
            let reach = V::splat(POLISH_REACH);
            let worth = |e: V, bound: V| !polishing || V::mask_any(busy & e.simd_lt(bound * reach));
            for (slot, mode) in MODES.into_iter().enumerate() {
                let (on, list, kk) = slot_cfg(slot);
                if !on {
                    continue;
                }
                let ns = mode.subsets();
                let mut mb: Option<TwoBest<V>> = None;
                for (ci, part) in list.iter().take(kk).enumerate() {
                    let mut subs = pre[slot][ci].expect("encoded above");
                    if cfg.polish_parts
                        && (cfg.polish_top == 0 || ci < cfg.polish_top as usize)
                        && worth(sum_err(&subs, ns), bound)
                    {
                        let m = part_masks::<V>(*part, mode);
                        polish(t, &m, &mut subs, mode, cfg.uber, cfg.refine);
                        bound = bound.simd_min(sum_err(&subs, ns));
                    }
                    let [s0, s1, s2] = subs;
                    let e = sum_err(&subs, ns);
                    mb = Some(match mb {
                        None => ([s0, s1, s2], *part),
                        Some((b, bp)) => {
                            let be = if ns == 3 {
                                b[0].err + b[1].err + b[2].err
                            } else {
                                b[0].err + b[1].err
                            };
                            let take = e.simd_lt(be);
                            (
                                [
                                    merge_sub(take, &s0, &b[0]),
                                    merge_sub(take, &s1, &b[1]),
                                    merge_sub(take, &s2, &b[2]),
                                ],
                                V::Int::select(take, *part, bp),
                            )
                        }
                    });
                }
                if let Some((mut b, bp)) = mb {
                    if cfg.polish_modes && (!cfg.polish_parts || cfg.polish_top != 0) && worth(sum_err(&b, ns), bound) {
                        let m = part_masks::<V>(bp, mode);
                        polish(t, &m, &mut b, mode, cfg.uber, cfg.refine);
                        bound = bound.simd_min(sum_err(&b, ns));
                    }
                    let e = if ns == 3 {
                        b[0].err + b[1].err + b[2].err
                    } else {
                        b[0].err + b[1].err
                    };
                    let take = busy & e.simd_lt(best_err);
                    best_err = V::select(take, e, best_err);
                    winner = V::Int::select(take, V::Int::splat(slot as i32 + 1), winner);
                    mode_best[slot] = Some((b, bp));
                }
            }
            for (slot, mode) in MODES.into_iter().enumerate() {
                let Some((mut b, bp)) = mode_best[slot] else {
                    continue;
                };
                let code = V::Int::splat(slot as i32 + 1);
                let lanes = winner.simd_lt(code + V::Int::splat(1)) & code.simd_lt(winner + V::Int::splat(1));
                if !V::mask_any(lanes) {
                    continue;
                }
                if cfg.refine > 0 && !cfg.polish_modes {
                    // ±1 refinement of this mode's winners only (refinement never raises the
                    // error, so they stay winners).
                    let m = part_masks::<V>(bp, mode);
                    polish(t, &m, &mut b, mode, 0, cfg.refine);
                }
                pack_two_lanes(mode, bp, &b, V::mask_to_bools(lanes), &mut over);
            }
            if cfg.mode5 {
                // Dual plane: each color channel in the scalar plane; the best rotation per lane.
                let mut bm: Option<(M5<V>, V::Int)> = None;
                for r in 0..3 {
                    let e = encode_m5(t, r);
                    bm = Some(match bm {
                        None => (e, V::Int::splat(r as i32 + 1)),
                        Some((b, br)) => {
                            let take = e.err.simd_lt(b.err);
                            let mut out = b;
                            let mut k = 0;
                            while k < 2 {
                                let mut c = 0;
                                while c < 3 {
                                    out.cq[k][c] = V::select(take, e.cq[k][c], b.cq[k][c]);
                                    c += 1;
                                }
                                out.av[k] = V::select(take, e.av[k], b.av[k]);
                                k += 1;
                            }
                            let mut i = 0;
                            while i < 16 {
                                out.cidx[i] = V::select(take, e.cidx[i], b.cidx[i]);
                                out.aidx[i] = V::select(take, e.aidx[i], b.aidx[i]);
                                i += 1;
                            }
                            out.err = V::select(take, e.err, b.err);
                            (out, V::Int::select(take, V::Int::splat(r as i32 + 1), br))
                        }
                    });
                }
                if let Some((b, br)) = bm {
                    let take = busy & b.err.simd_lt(best_err);
                    if V::mask_any(take) {
                        winner = V::Int::select(take, V::Int::splat(9), winner);
                        let rots = br.to_array();
                        let mut cqa = [[[0f32; LANES]; 3]; 2];
                        let mut ava = [[0f32; LANES]; 2];
                        let mut cia = [[0f32; LANES]; 16];
                        let mut aia = [[0f32; LANES]; 16];
                        for k in 0..2 {
                            for c in 0..3 {
                                cqa[k][c] = b.cq[k][c].to_array();
                            }
                            ava[k] = b.av[k].to_array();
                        }
                        for i in 0..16 {
                            cia[i] = b.cidx[i].to_array();
                            aia[i] = b.aidx[i].to_array();
                        }
                        for (l, tk) in V::mask_to_bools(take).iter().enumerate() {
                            if !*tk {
                                continue;
                            }
                            let mut cq = [[0u8; 3]; 2];
                            for k in 0..2 {
                                for c in 0..3 {
                                    cq[k][c] = cqa[k][c][l] as u8;
                                }
                            }
                            let av = [ava[0][l] as u8, ava[1][l] as u8];
                            let ci: [u8; 16] = std::array::from_fn(|i| cia[i][l] as u8);
                            let ai: [u8; 16] = std::array::from_fn(|i| aia[i][l] as u8);
                            over[l] = Some(pack_m5(rots[l] as u8, &cq, &av, &ci, &ai));
                        }
                    }
                }
            }
            if cfg.refine > 0 && cfg.refine6 && !cfg.polish_modes {
                let m6 = busy & winner.simd_lt(V::Int::splat(1));
                if V::mask_any(m6) {
                    refine6(t, &mut ends, &mut idx, &mut err, m6, cfg.refine);
                }
            }
        }
    }
    (pack(&ends, &idx), over)
}

/// Writes eight packed blocks (`words`, lanes in `order`) to `out` (16 bytes per block),
/// skipping lanes beyond `out`.
#[inline(always)]
fn store<V: SimdF32>(words: &[V::Int; 4], order: &[usize; LANES], out: &mut [u8]) {
    let w = [
        words[0].to_array(),
        words[1].to_array(),
        words[2].to_array(),
        words[3].to_array(),
    ];
    let n = out.len() / 16;
    for (l, &b) in order.iter().enumerate() {
        if b < n {
            let o = &mut out[16 * b..16 * b + 16];
            for k in 0..4 {
                o[4 * k..4 * k + 4].copy_from_slice(&w[k][l].to_le_bytes());
            }
        }
    }
}

/// Encodes blocks `first..` of a surface into `out` (16 bytes per block).
#[inline(always)]
fn encode_span<V: SimdF32>(s: &Surface, first: usize, out: &mut [u8], cfg: Cfg, opts: BlockOpts) {
    const IDENTITY: [usize; LANES] = [0, 1, 2, 3, 4, 5, 6, 7];
    let bw = s.blocks_x() as usize;
    let (w, h) = (s.width as usize, s.height as usize);
    let mut k = first;
    for o in out.chunks_mut(LANES * 16) {
        let n = o.len() / 16;
        let (bx, by) = (k % bw, k / bw);
        let mut t = [[V::splat(0.0); 16]; 3];
        let (order, opaque) = if n == LANES && (bx + LANES) * 4 <= w && by * 4 + 4 <= h {
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
            // Lanes whose alpha is not 255 everywhere: alpha byte below 255 anywhere.
            let mut amin = V::Int::splat(255);
            let mut i = 0;
            while i < 16 {
                t[0][i] = (px[i] & V::Int::splat(0xff)).to_float();
                t[1][i] = ((px[i] >> 8) & V::Int::splat(0xff)).to_float();
                t[2][i] = ((px[i] >> 16) & V::Int::splat(0xff)).to_float();
                amin = amin.simd_min((px[i] >> 24) & V::Int::splat(0xff));
                i += 1;
            }
            let opaque = amin.to_array().map(|a| a == 255);
            let mut op = [true; LANES];
            for l in 0..LANES {
                op[ROW_ORDER[l]] = opaque[l];
            }
            (ROW_ORDER, op)
        } else {
            let mut v = [[[0f32; LANES]; 16]; 3];
            let mut op = [true; LANES];
            for (l, o) in op.iter_mut().enumerate().take(n) {
                let i = (k + l) as u32;
                let b = s.read_block(i % bw as u32, i / bw as u32);
                *o = b.iter().all(|px| px[3] == 255);
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
            (IDENTITY, op)
        };
        let (words, over) = encode8(&t, cfg);
        store::<V>(&words, &order, o);
        for (l, ov) in over.iter().enumerate() {
            if let Some(b) = ov {
                let blk = order[l];
                if blk < n {
                    o[16 * blk..16 * blk + 16].copy_from_slice(b);
                }
            }
        }
        // Blocks with alpha: the scalar encoder.
        if opaque.iter().take(n).any(|&x| !x) {
            let wts = crate::format::Format::Bc7.error_weights();
            for (l, &op) in opaque.iter().enumerate().take(n) {
                if !op {
                    let i = (k + l) as u32;
                    let b = s.read_block(i % bw as u32, i / bw as u32);
                    o[16 * l..16 * l + 16].copy_from_slice(&super::encode(&b, &wts, opts));
                }
            }
        }
        k += n;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode_span_avx2(s: &Surface, first: usize, out: &mut [u8], cfg: Cfg, opts: BlockOpts) {
    encode_span::<crate::simd::avx2::F>(s, first, out, cfg, opts);
}

/// Encodes a whole surface with the fast search `cfg`. `opts` is passed to the scalar encoder
/// for blocks with alpha. Returns raw 16-byte blocks in raster order.
pub fn encode_surface(surface: &Surface, cfg: Cfg, opts: BlockOpts) -> Vec<u8> {
    use rayon::prelude::*;
    const SPAN: usize = 256;
    let mut out = vec![0u8; surface.block_count() * 16];
    let lvl = level();
    out.par_chunks_mut(SPAN * 16).enumerate().for_each(|(ci, o)| match lvl {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { encode_span_avx2(surface, ci * SPAN, o, cfg, opts) },
        _ => encode_span::<f32x8>(surface, ci * SPAN, o, cfg, opts),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(w: u32, h: u32) -> Surface {
        let img = crate::synth::foliage(72.max(w));
        let mut s = Surface::new(w, h);
        for y in 0..h {
            for x in 0..w {
                s.set(x, y, img.get(x, y));
            }
        }
        s
    }

    #[test]
    fn backends_agree_and_blocks_decode() {
        let s = surface(70, 66);
        for cfg in [
            Cfg {
                power: 2,
                refits: 0,
                two: 0,
                two_refits: 0,
                parts: 1,
                rank: false,
                refine: 0,
                busy_err: 0.0,
                refine6: true,
                three: 0,
                parts_three: 1,
                mode5: false,
                uber: 0,
                polish_modes: false,
                polish_parts: false,
                polish_top: 0,
            },
            Cfg {
                power: 3,
                refits: 2,
                two: 3,
                two_refits: 1,
                parts: 2,
                rank: true,
                refine: 1,
                busy_err: 0.0,
                refine6: true,
                three: 3,
                parts_three: 1,
                mode5: true,
                uber: 2,
                polish_modes: true,
                polish_parts: true,
                polish_top: 1,
            },
        ] {
            let fast = encode_surface(&s, cfg, 0u8.into());
            let mut generic = vec![0u8; fast.len()];
            encode_span::<f32x8>(&s, 0, &mut generic, cfg, 0u8.into());
            assert_eq!(fast, generic, "{cfg:?}");
            // Every block is a valid mode-6 block within reach of the source.
            let bw = s.blocks_x();
            let mut sse = 0u64;
            for (i, raw) in fast.as_chunks::<16>().0.iter().enumerate() {
                let b = super::super::unpack(raw).unwrap();
                assert!(matches!(b.mode, 0..=3 | 5 | 6), "mode {}", b.mode);
                let mut d = [[0u8; 4]; 16];
                super::super::decode(raw, &mut d);
                let src = s.read_block(i as u32 % bw, i as u32 / bw);
                sse += crate::surface::block_sse(&src, &d, &[1, 1, 1, 1]);
            }
            let psnr = 10.0 * (255.0f64 * 255.0 * (fast.len() / 16 * 64) as f64 / sse as f64).log10();
            // Within reach of the scalar encoder's effort 0 (which also tries other modes).
            let wts = crate::format::Format::Bc7.error_weights();
            let mut sse0 = 0u64;
            for b in s.blocks() {
                let raw = super::super::encode(&b, &wts, 0u8);
                let mut d = [[0u8; 4]; 16];
                super::super::decode(&raw, &mut d);
                sse0 += crate::surface::block_sse(&b, &d, &[1, 1, 1, 1]);
            }
            let psnr0 = 10.0 * (255.0f64 * 255.0 * (fast.len() / 16 * 64) as f64 / sse0 as f64).log10();
            assert!(psnr > psnr0 - 1.0, "{cfg:?}: {psnr:.2} dB vs scalar {psnr0:.2}");
        }
    }

    /// Diagnostic: mean per-image PSNR and time (configs interleaved per image, fastest of 3
    /// runs) of fast configs on Kodak 1-6, one thread.
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn fast_configs() {
        let surfs: Vec<Surface> = (1..=6)
            .map(|n| Surface::load(std::path::Path::new(&format!("test-images/kodak/kodim{n:02}.png"))).unwrap())
            .collect();
        let cfgs: Vec<(&str, Cfg)> = (0..=9u8)
            .map(|e| ("effort", super::super::fast_cfg(e.into()).unwrap()))
            .collect();
        let mut secs = vec![0f64; cfgs.len()];
        let mut psnr = vec![0f64; cfgs.len()];
        for s in &surfs {
            let mut best = vec![f64::INFINITY; cfgs.len()];
            let mut out = vec![Vec::new(); cfgs.len()];
            for _ in 0..3 {
                for (k, (_, cfg)) in cfgs.iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    out[k] = encode_surface(s, *cfg, 0u8.into());
                    best[k] = best[k].min(t0.elapsed().as_secs_f64());
                }
            }
            for k in 0..cfgs.len() {
                secs[k] += best[k];
                let bw = s.blocks_x();
                let mut d = [[0u8; 4]; 16];
                let (mut sse, mut n) = (0u64, 0u64);
                for (i, b) in out[k].as_chunks::<16>().0.iter().enumerate() {
                    super::super::decode(b, &mut d);
                    let src = s.read_block(i as u32 % bw, i as u32 / bw);
                    sse += crate::surface::block_sse(&src, &d, &[1, 1, 1, 0]);
                    n += 48;
                }
                psnr[k] += 10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10();
            }
        }
        for (k, (name, _)) in cfgs.iter().enumerate() {
            println!(
                "{name} {k} {:.3} dB {:8.1} ms",
                psnr[k] / surfs.len() as f64,
                secs[k] * 1e3
            );
        }
    }

    /// Diagnostic: per-block comparison of the fast 2-subset search with the batch encoder's
    /// effort 1 on one image.
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn compare_with_batch() {
        let s = Surface::load(std::path::Path::new("test-images/kodak/kodim01.png")).unwrap();
        let cfg = Cfg {
            power: 3,
            refits: 2,
            two: 3,
            two_refits: 2,
            parts: 4,
            rank: true,
            refine: 1,
            busy_err: 0.0,
            refine6: true,
            three: 0,
            parts_three: 1,
            mode5: false,
            uber: 0,
            polish_modes: false,
            polish_parts: false,
            polish_top: 0,
        };
        let fast = encode_surface(&s, cfg, 0u8.into());
        let w = crate::format::Format::Bc7.error_weights();
        let blocks = s.blocks();
        let batch: Vec<[u8; 16]> = blocks
            .chunks(LANES)
            .flat_map(|c| super::super::encode_batch(c, &w, 1u8.into()))
            .collect();
        let mut stats = std::collections::BTreeMap::<(u8, u8), (u32, u64, u64)>::new();
        let (mut tf, mut tb) = (0u64, 0u64);
        for (i, b) in blocks.iter().enumerate() {
            let (mut df, mut db) = ([[0u8; 4]; 16], [[0u8; 4]; 16]);
            let f = &fast[16 * i..16 * i + 16];
            super::super::decode(f, &mut df);
            super::super::decode(&batch[i], &mut db);
            let ef = crate::surface::block_sse(b, &df, &[1, 1, 1, 0]);
            let eb = crate::surface::block_sse(b, &db, &[1, 1, 1, 0]);
            tf += ef;
            tb += eb;
            let mf = super::super::unpack(f).unwrap().mode;
            let mb = super::super::unpack(&batch[i]).unwrap().mode;
            let e = stats.entry((mf, mb)).or_default();
            e.0 += 1;
            if ef > eb {
                e.1 += ef - eb;
            } else {
                e.2 += eb - ef;
            }
        }
        println!("total sse fast {tf} batch {tb}");
        for ((mf, mb), (n, worse, better)) in stats {
            println!("fast m{mf} batch m{mb}: {n:6} blocks, fast worse by {worse:8}, better by {better:8}");
        }
    }

    /// Diagnostic: which modes the scalar encoder's search level 2 picks over level 1, and the
    /// error they save (kodim01).
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn level2_modes() {
        let s = Surface::load(std::path::Path::new("test-images/kodak/kodim01.png")).unwrap();
        let w = crate::format::Format::Bc7.error_weights();
        let mut gain = [0u64; 8];
        let mut count = [0u32; 8];
        let (mut t1, mut t2) = (0u64, 0u64);
        for b in s.blocks() {
            let r1 = super::super::encode(&b, &w, 1u8);
            let r2 = super::super::encode(&b, &w, 2u8);
            let (mut d1, mut d2) = ([[0u8; 4]; 16], [[0u8; 4]; 16]);
            super::super::decode(&r1, &mut d1);
            super::super::decode(&r2, &mut d2);
            let e1 = crate::surface::block_sse(&b, &d1, &[1, 1, 1, 0]);
            let e2 = crate::surface::block_sse(&b, &d2, &[1, 1, 1, 0]);
            t1 += e1;
            t2 += e2;
            let m = super::super::unpack(&r2).unwrap().mode as usize;
            count[m] += 1;
            gain[m] += e1.saturating_sub(e2);
        }
        println!("level 1 sse {t1}, level 2 sse {t2}");
        for m in 0..8 {
            println!(
                "mode {m}: {:6} blocks at level 2, sse saved vs level 1 {:8}",
                count[m], gain[m]
            );
        }
    }
}
