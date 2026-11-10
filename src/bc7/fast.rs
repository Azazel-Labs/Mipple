//! Fast BC7 batch encoder: mode 6 only, eight opaque blocks per SIMD batch (one per lane), read
//! straight from the surface and packed in registers.
//!
//! Mode 6 has one subset, 7-bit RGBA endpoints with a p-bit each (so an endpoint channel is an
//! exact 8-bit value `2q + p`) and 4-bit indices. Per batch:
//!
//! * endpoints: the principal axis by power iteration (unnormalized; only its direction
//!   matters), at the extreme projections;
//! * quantization: for each endpoint and p-bit the nearest levels are one rounding; the p-bit
//!   with the lower endpoint error wins (alpha included, so opaque blocks keep alpha 255 unless
//!   p = 0 is clearly better for RGB);
//! * indices: projection onto the quantized endpoints, rounded to 15 steps (the mode-6 weights
//!   are within 0.5 of 64k/15);
//! * [`Cfg::refits`] least-squares refits from those indices, each kept per lane when it lowers
//!   the exact error.
//!
//! Blocks with alpha go to the scalar encoder. Every value is a small integer or an IEEE result
//! in a fixed order, so both backends agree bit for bit.

use crate::exact::{div_floor, reduce16, shrink, sum16};
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
    /// Keep both p-bits at 1 so alpha decodes to exactly 255 (otherwise each endpoint takes the
    /// p-bit with the lower RGBA endpoint error).
    pub opaque_pbits: bool,
    /// Pick each index by exact palette error between the two entries around the projection
    /// (otherwise the projection is rounded).
    pub exact: bool,
    /// Also try 2-subset mode 1 (bit 0) and mode 3 (bit 1) on each lane's estimated partition.
    pub two: u8,
    /// Least-squares refits of the 2-subset encodings.
    pub two_refits: u8,
    /// Partition candidates per lane for the 2-subset modes (1-4).
    pub parts: u8,
    /// Partition ranking: 0 = bc7f's popcount estimate, 1 = line-fit residual of all 64
    /// partitions with a one-step λ estimate ([`rank_fits`]), 2 = the batch encoder's `fits8`.
    pub rank: u8,
    /// ±1 endpoint refinement passes of the winning 2-subset encodings ([`refine_two`]).
    pub refine: u8,
    /// Mode-6 error (sum over the block) at or below which a block skips the 2-subset search.
    pub busy_err: f32,
    /// Partition candidates for mode 3 (0 = as many as `parts`).
    pub parts3: u8,
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
fn quant_endpoint<V: SimdF32>(e: &[V; 3], opaque: bool) -> ([V; 4], V, [V; 4]) {
    let zero = V::splat(0.0);
    let half = V::splat(0.5);
    let top = V::splat(127.0);
    let x = [e[0], e[1], e[2], V::splat(255.0)];
    let mut best_q = [zero; 4];
    let mut best_v = [zero; 4];
    let mut best_e = V::splat(f32::INFINITY);
    let mut best_p = zero;
    let mut p = i32::from(opaque);
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
fn quantize<V: SimdF32>(e0: &[V; 3], e1: &[V; 3], opaque: bool) -> Ends<V> {
    let (q0, p0, v0) = quant_endpoint(e0, opaque);
    let (q1, p1, v1) = quant_endpoint(e1, opaque);
    Ends {
        q: [q0, q1],
        p: [p0, p1],
        v: [v0, v1],
    }
}

/// Squared RGB error of texel `i` against the palette entry with weight `w` (opaque blocks are
/// scored on RGB; alpha only steers the p-bit choice).
#[inline(always)]
fn texel_error<V: SimdF32>(t: &Texels<V>, e: &Ends<V>, i: usize, w: V) -> V {
    let (c64, c32, inv64) = (V::splat(64.0), V::splat(32.0), V::splat(1.0 / 64.0));
    let wa = c64 - w;
    let mut err = V::splat(0.0);
    let mut c = 0;
    while c < 3 {
        let pal = floor(wa.mul_add(e.v[0][c], w.mul_add(e.v[1][c], c32)) * inv64);
        let d = t[c][i] - pal;
        err = d.mul_add(d, err);
        c += 1;
    }
    err
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

/// Indices by projection onto the quantized endpoints (RGB): rounded to the 15 steps, or (with
/// `exact`) the better of the two palette entries around the projection. Returns the indices
/// and, with `exact`, the block's RGB error (else zero).
#[inline(always)]
fn project<V: SimdF32>(t: &Texels<V>, e: &Ends<V>, exact: bool) -> ([V; 16], V) {
    let zero = V::splat(0.0);
    let d = [e.v[1][0] - e.v[0][0], e.v[1][1] - e.v[0][1], e.v[1][2] - e.v[0][2]];
    let dd = d[0].mul_add(d[0], d[1].mul_add(d[1], d[2] * d[2]));
    let scale = V::select(zero.simd_lt(dd), V::splat(15.0) / dd.simd_max(V::splat(1.0)), zero);
    let k = [d[0] * scale, d[1] * scale, d[2] * scale];
    // Rounded: floor((x - a)·k + 1/2); exact: candidates floor((x - a)·k) and the next one.
    let base = e.v[0][0].mul_add(k[0], e.v[0][1].mul_add(k[1], e.v[0][2] * k[2]));
    let bias = if exact { zero - base } else { V::splat(0.5) - base };
    let top = V::splat(if exact { 14.0 } else { 15.0 });
    let inv64 = V::splat(1.0 / 64.0);
    let a3 = [e.v[0][0], e.v[0][1], e.v[0][2]];
    let d64 = [d[0] * inv64, d[1] * inv64, d[2] * inv64];
    let mut idx = [zero; 16];
    let mut err = zero;
    // Exact: j = floor(15 (x - a)·d / |d|²) in integers (|15 (x - a)·d| < 2^22).
    let (dd1, fifteen) = (dd.simd_max(V::splat(1.0)), V::splat(15.0));
    let inv_dd = V::splat(1.0) / dd1;
    let mut i = 0;
    while i < 16 {
        let j = if exact {
            let xa = [t[0][i] - a3[0], t[1][i] - a3[1], t[2][i] - a3[2]];
            let num = (xa[0] * d[0] + xa[1] * d[1] + xa[2] * d[2]) * fifteen;
            div_floor(num, dd1, inv_dd).0.simd_max(zero).simd_min(top)
        } else {
            let s = t[0][i].mul_add(k[0], t[1][i].mul_add(k[1], t[2][i].mul_add(k[2], bias)));
            floor(s.simd_max(zero)).simd_min(top)
        };
        if exact {
            let (e0, e1) = pair_err(t, [0, 1, 2], i, &a3, &d64, weight(j, 16), weight(j + V::splat(1.0), 16));
            let up = e1.simd_lt(e0);
            idx[i] = V::select(up, j + V::splat(1.0), j);
            err += e0.simd_min(e1);
        } else {
            idx[i] = j;
        }
        i += 1;
    }
    (idx, err)
}

/// Exact RGB error of indices against the decoded palette.
#[inline(always)]
fn error<V: SimdF32>(t: &Texels<V>, e: &Ends<V>, idx: &[V; 16]) -> V {
    let mut err = V::splat(0.0);
    let mut i = 0;
    while i < 16 {
        err += texel_error(t, e, i, weight(idx[i], 16));
        i += 1;
    }
    err
}

/// Least-squares RGB endpoints for the indices; lanes with a singular system keep `prev`.
#[inline(always)]
fn ls_endpoints<V: SimdF32>(t: &Texels<V>, idx: &[V; 16], prev: ([V; 3], [V; 3])) -> ([V; 3], [V; 3], V::Mask) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let (mut aa, mut ab, mut bb) = (zero, zero, zero);
    let mut ax = [zero; 3];
    let mut bx = [zero; 3];
    let inv64 = V::splat(1.0 / 64.0);
    let mut i = 0;
    while i < 16 {
        let wb = weight(idx[i], 16) * inv64;
        let wa = one - wb;
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
    let full = V::splat(255.0);
    let mut e0 = prev.0;
    let mut e1 = prev.1;
    let mut c = 0;
    while c < 3 {
        let a = ((bb * ax[c] - ab * bx[c]) * inv).simd_clamp(zero, full);
        let b = ((aa * bx[c] - ab * ax[c]) * inv).simd_clamp(zero, full);
        e0[c] = V::select(ok, a, e0[c]);
        e1[c] = V::select(ok, b, e1[c]);
        c += 1;
    }
    (e0, e1, ok)
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

/// Nearest 6-bit levels (with p-bit `p`) of 8-bit targets, as 7-bit values `2q + p` expanded by
/// bit replication: tries the rounded level and its neighbours.
#[inline(always)]
fn quant6p<V: SimdF32>(x: V, p: V) -> (V, V, V) {
    let zero = V::splat(0.0);
    let c = floor(((x * V::splat(127.0 / 255.0) - p) * V::splat(0.5) + V::splat(0.5)).simd_max(zero));
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
    let full = ((1 << n) - 1) as f32;
    let t = x * V::splat(full / 255.0);
    let c = if has_p {
        floor(((t - p) * V::splat(0.5) + V::splat(0.5)).simd_max(zero))
    } else {
        floor((t + V::splat(0.5)).simd_max(zero))
    };
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
    let dd = d[0].mul_add(d[0], d[1].mul_add(d[1], d[2] * d[2]));
    let scale = V::select(zero.simd_lt(dd), V::splat(n - 1.0) / dd.simd_max(V::splat(1.0)), zero);
    let k = [d[0] * scale, d[1] * scale, d[2] * scale];
    let bias = zero - v[0][0].mul_add(k[0], v[0][1].mul_add(k[1], v[0][2] * k[2]));
    let top = V::splat(n - 2.0);
    let inv64 = V::splat(1.0 / 64.0);
    let d64 = [d[0] * inv64, d[1] * inv64, d[2] * inv64];
    let mut idx = [zero; 16];
    let mut err = zero;
    let mut i = 0;
    while i < 16 {
        let s = t[0][i].mul_add(k[0], t[1][i].mul_add(k[1], t[2][i].mul_add(k[2], bias)));
        let j = floor(s.simd_max(zero)).simd_min(top);
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

/// Encodes one subset (members `m`) per lane: masked principal axis, extremes, quantization,
/// exact indices, then `refits` least-squares refits kept when they lower the error.
#[inline(always)]
fn encode_two<V: SimdF32>(t: &Texels<V>, m: &[V; 16], mode: Two, refits: u8) -> Sub<V> {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let mut n = zero;
    let mut mean = [zero; 3];
    let mut i = 0;
    while i < 16 {
        n += m[i];
        let mut c = 0;
        while c < 3 {
            mean[c] = m[i].mul_add(t[c][i], mean[c]);
            c += 1;
        }
        i += 1;
    }
    let inv_n = one / n.simd_max(one);
    mean = [mean[0] * inv_n, mean[1] * inv_n, mean[2] * inv_n];
    let (mut rr, mut rg, mut rb, mut gg, mut gb, mut bb) = (zero, zero, zero, zero, zero, zero);
    let mut i = 0;
    while i < 16 {
        let (r, g, b) = (t[0][i] - mean[0], t[1][i] - mean[1], t[2][i] - mean[2]);
        let (mr, mg) = (m[i] * r, m[i] * g);
        rr = mr.mul_add(r, rr);
        rg = mr.mul_add(g, rg);
        rb = mr.mul_add(b, rb);
        gg = mg.mul_add(g, gg);
        gb = mg.mul_add(b, gb);
        bb = (m[i] * b).mul_add(b, bb);
        i += 1;
    }
    let mut v = [rr, rg, rb];
    let mut big = rr;
    let sel = big.simd_lt(gg);
    v = [
        V::select(sel, rg, v[0]),
        V::select(sel, gg, v[1]),
        V::select(sel, gb, v[2]),
    ];
    big = big.simd_max(gg);
    let sel = big.simd_lt(bb);
    v = [
        V::select(sel, rb, v[0]),
        V::select(sel, gb, v[1]),
        V::select(sel, bb, v[2]),
    ];
    let mut it = 0;
    while it < 3 {
        let w = [
            rr.mul_add(v[0], rg.mul_add(v[1], rb * v[2])),
            rg.mul_add(v[0], gg.mul_add(v[1], gb * v[2])),
            rb.mul_add(v[0], gb.mul_add(v[1], bb * v[2])),
        ];
        let mx = w[0]
            .simd_max(zero - w[0])
            .simd_max(w[1].simd_max(zero - w[1]))
            .simd_max(w[2].simd_max(zero - w[2]));
        let s = V::select(V::splat(1e-20).simd_lt(mx), one / mx, zero);
        v = [w[0] * s, w[1] * s, w[2] * s];
        it += 1;
    }
    let n2 = v[0].mul_add(v[0], v[1].mul_add(v[1], v[2] * v[2]));
    let flat = n2.simd_lt(V::splat(1e-12));
    v = [
        V::select(flat, one, v[0]),
        V::select(flat, one, v[1]),
        V::select(flat, one, v[2]),
    ];
    let n2 = V::select(flat, V::splat(3.0), n2);
    let (mut lo, mut hi) = (V::splat(f32::MAX), V::splat(f32::MIN));
    let mut i = 0;
    while i < 16 {
        let s = (t[0][i] - mean[0]).mul_add(v[0], (t[1][i] - mean[1]).mul_add(v[1], (t[2][i] - mean[2]) * v[2]));
        let member = zero.simd_lt(m[i]);
        lo = V::select(member, lo.simd_min(s), lo);
        hi = V::select(member, hi.simd_max(s), hi);
        i += 1;
    }
    let (lo, hi) = (lo.simd_min(zero) / n2, hi.simd_max(zero) / n2);
    let full = V::splat(255.0);
    let mut e = [[zero; 3]; 2];
    let mut c = 0;
    while c < 3 {
        e[0][c] = v[c].mul_add(lo, mean[c]).simd_clamp(zero, full);
        e[1][c] = v[c].mul_add(hi, mean[c]).simd_clamp(zero, full);
        c += 1;
    }
    let (q, p, vq) = quant_two(&e, mode);
    let (idx, err) = index_two(t, m, &vq, mode);
    let mut best = Sub { q, p, v: vq, idx, err };
    let mut r = 0;
    while r < refits {
        // Least-squares endpoints over the members for the current indices.
        let (mut aa, mut ab, mut bbw) = (zero, zero, zero);
        let mut ax = [zero; 3];
        let mut bx = [zero; 3];
        let inv64 = V::splat(1.0 / 64.0);
        let mut i = 0;
        while i < 16 {
            let wb = weight(best.idx[i], 1u32 << mode.index_bits()) * inv64;
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
        let mut ne = e;
        let mut c = 0;
        while c < 3 {
            ne[0][c] = V::select(ok, ((bbw * ax[c] - ab * bx[c]) * inv).simd_clamp(zero, full), e[0][c]);
            ne[1][c] = V::select(ok, ((aa * bx[c] - ab * ax[c]) * inv).simd_clamp(zero, full), e[1][c]);
            c += 1;
        }
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

/// Greedy ±1 refinement of a subset's quantized endpoint channels and p-bit flips: each change is
/// kept per lane when it lowers the subset's exact error (indices re-chosen for every trial;
/// holding them fixed loses two thirds of the gain). Up to `passes` sweeps.
#[inline(always)]
fn refine_two<V: SimdF32>(t: &Texels<V>, m: &[V; 16], sub: &mut Sub<V>, mode: Two, passes: u8) {
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
                    let q = (sub.q[k][c] + d).simd_clamp(zero, top);
                    let mut v = sub.v;
                    v[k][c] = dequant_two(q, sub.p[k], mode);
                    let (idx, err) = index_two(t, m, &v, mode);
                    let take = err.simd_lt(sub.err);
                    if V::mask_any(take) {
                        any = true;
                        sub.q[k][c] = V::select(take, q, sub.q[k][c]);
                        sub.v[k][c] = V::select(take, v[k][c], sub.v[k][c]);
                        sub.err = V::select(take, err, sub.err);
                        let mut i = 0;
                        while i < 16 {
                            sub.idx[i] = V::select(take, idx[i], sub.idx[i]);
                            i += 1;
                        }
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
            let mut np = sub.p;
            let mut v = sub.v;
            let mut k = 0;
            while k < 2 {
                if f[k] {
                    np[k] = one - sub.p[k];
                    let mut c = 0;
                    while c < 3 {
                        v[k][c] = dequant_two(sub.q[k][c], np[k], mode);
                        c += 1;
                    }
                }
                k += 1;
            }
            let (idx, err) = index_two(t, m, &v, mode);
            let take = err.simd_lt(sub.err);
            if V::mask_any(take) {
                any = true;
                let mut k = 0;
                while k < 2 {
                    sub.p[k] = V::select(take, np[k], sub.p[k]);
                    let mut c = 0;
                    while c < 3 {
                        sub.v[k][c] = V::select(take, v[k][c], sub.v[k][c]);
                        c += 1;
                    }
                    k += 1;
                }
                sub.err = V::select(take, err, sub.err);
                let mut i = 0;
                while i < 16 {
                    sub.idx[i] = V::select(take, idx[i], sub.idx[i]);
                    i += 1;
                }
            }
        }
        if !any {
            break;
        }
        pass += 1;
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
            let (cidx, cerr) = project(t, &cand, true);
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

/// bc7e's "uber" trials for one subset: least-squares fits from index sets with the subset's
/// lowest index raised, its highest lowered, both, and (for `level` >= 2) the indices linearly
/// stretched from `[ly, hy]` onto the full range for ly in -Q..=1 and hy in max-1..=max+Q
/// (Q = `level` - 1). Each fit is quantized and re-indexed exactly; kept per lane when better.
#[inline(always)]
fn uber_two<V: SimdF32>(t: &Texels<V>, m: &[V; 16], sub: &mut Sub<V>, mode: Two, level: u8) {
    let zero = V::splat(0.0);
    let maxsel = ((1u32 << mode.index_bits()) - 1) as f32;
    let top = V::splat(maxsel);
    // Lowest and highest index over the members.
    let (mut lo, mut hi) = (top, zero);
    let mut i = 0;
    while i < 16 {
        let member = zero.simd_lt(m[i]);
        lo = V::select(member, lo.simd_min(sub.idx[i]), lo);
        hi = V::select(member, hi.simd_max(sub.idx[i]), hi);
        i += 1;
    }
    let base = sub.idx;
    let prev = sub.v;
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
                idx2[i] = floor(((base[i] - l) * scale + V::splat(0.5)).simd_max(zero)).simd_min(top);
                i += 1;
            }
        }
        let (e, ok) = ls_two(t, m, &idx2, mode, &prev);
        let (q2, p2, v2) = quant_two(&e, mode);
        let (idx3, err3) = index_two(t, m, &v2, mode);
        let take = ok & err3.simd_lt(sub.err);
        if V::mask_any(take) {
            *sub = merge_sub(
                take,
                &Sub {
                    q: q2,
                    p: p2,
                    v: v2,
                    idx: idx3,
                    err: err3,
                },
                sub,
            );
        }
        k += 1;
    }
}

/// [`uber_two`]'s trials for mode 6 in the `active` lanes (16-level indices, 8-bit endpoints).
#[inline(always)]
fn uber6<V: SimdF32>(
    t: &Texels<V>,
    ends: &mut Ends<V>,
    idx: &mut [V; 16],
    err: &mut V,
    active: V::Mask,
    level: u8,
    opaque: bool,
) {
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
        let (a, b, ok) = ls_endpoints(t, &idx2, prev);
        let cand = quantize(&a, &b, opaque);
        let (cidx, cerr) = project(t, &cand, true);
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

/// Ranks all 64 2-subset partitions per lane by an estimate of their error with 3-bit and with
/// 2-bit indices: per subset, `trace(S) - λ + Δw² λ` with λ the Rayleigh quotient after one
/// power step from the block's principal axis (`Δw` = index step, as the batch encoder's
/// ranking). Subset sums come from per-nibble subset tables: the membership of a partition is
/// the same in every lane. Returns each lane's `K` best partitions for each index size, best
/// first.
#[inline(always)]
fn rank_fits<V: SimdF32, const K: usize>(t: &Texels<V>, mean: &[V; 3], axis: &[V; 3]) -> [[V::Int; K]; 2] {
    let zero = V::splat(0.0);
    // Centered texels and their products: 9 moments per texel.
    let mut val = [[zero; 16]; 9];
    let mut i = 0;
    while i < 16 {
        let (r, g, b) = (t[0][i] - mean[0], t[1][i] - mean[1], t[2][i] - mean[2]);
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
    let tot = tab[0][15];
    let tot = {
        let mut a = tot;
        let mut k = 1;
        while k < 4 {
            let mut v = 0;
            while v < 9 {
                a[v] += tab[k][15][v];
                v += 1;
            }
            k += 1;
        }
        a
    };
    // est = trace + (Δw² - 1) λ for Δw = 1/7 (3-bit) and 1/3 (2-bit).
    let qs = [V::splat(1.0 / 49.0 - 1.0), V::splat(1.0 / 9.0 - 1.0)];
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
        let mut est = [zero; 2];
        let mut sub = 0;
        while sub < 2 {
            let (n, sm) = if sub == 1 {
                (n1, s1)
            } else {
                let mut s0 = [zero; 9];
                let mut v = 0;
                while v < 9 {
                    s0[v] = tot[v] - s1[v];
                    v += 1;
                }
                (16.0 - n1, s0)
            };
            if n > 0.0 {
                let inv_n = V::splat(1.0 / n);
                // Scatter matrix: q - s sᵀ / n.
                let c00 = sm[3] - sm[0] * sm[0] * inv_n;
                let c01 = sm[4] - sm[0] * sm[1] * inv_n;
                let c02 = sm[5] - sm[0] * sm[2] * inv_n;
                let c11 = sm[6] - sm[1] * sm[1] * inv_n;
                let c12 = sm[7] - sm[1] * sm[2] * inv_n;
                let c22 = sm[8] - sm[2] * sm[2] * inv_n;
                let v1 = [
                    c00.mul_add(axis[0], c01.mul_add(axis[1], c02 * axis[2])),
                    c01.mul_add(axis[0], c11.mul_add(axis[1], c12 * axis[2])),
                    c02.mul_add(axis[0], c12.mul_add(axis[1], c22 * axis[2])),
                ];
                let v2 = [
                    c00.mul_add(v1[0], c01.mul_add(v1[1], c02 * v1[2])),
                    c01.mul_add(v1[0], c11.mul_add(v1[1], c12 * v1[2])),
                    c02.mul_add(v1[0], c12.mul_add(v1[1], c22 * v1[2])),
                ];
                let num = v1[0].mul_add(v2[0], v1[1].mul_add(v2[1], v1[2] * v2[2]));
                let den = v1[0].mul_add(v1[0], v1[1].mul_add(v1[1], v1[2] * v1[2]));
                let lam = V::select(V::splat(1e-6).simd_lt(den), num / den.simd_max(V::splat(1e-6)), zero);
                let tr = c00 + c11 + c22;
                est[0] += qs[0].mul_add(lam, tr);
                est[1] += qs[1].mul_add(lam, tr);
            }
            sub += 1;
        }
        let mut r = 0;
        while r < 2 {
            let mut e = est[r];
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

/// Ranks 3-subset partitions per lane like [`rank_fits`]: mode 0's 16 with 3-bit indices and
/// mode 2's 64 with 2-bit indices. Returns each lane's `K` best for each, best first.
#[inline(always)]
fn rank_fits3<V: SimdF32, const K: usize>(t: &Texels<V>, mean: &[V; 3], axis: &[V; 3]) -> [[V::Int; K]; 2] {
    let zero = V::splat(0.0);
    let mut val = [[zero; 16]; 9];
    let mut i = 0;
    while i < 16 {
        let (r, g, b) = (t[0][i] - mean[0], t[1][i] - mean[1], t[2][i] - mean[2]);
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
    let mut tab = [[[zero; 9]; 16]; 4];
    let mut k = 0;
    while k < 4 {
        let mut m = 1usize;
        while m < 16 {
            let low = m & (16 - m) & 15;
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
    let mut tot = [zero; 9];
    let mut k = 0;
    while k < 4 {
        let mut v = 0;
        while v < 9 {
            tot[v] += tab[k][15][v];
            v += 1;
        }
        k += 1;
    }
    let qs = [V::splat(1.0 / 49.0 - 1.0), V::splat(1.0 / 9.0 - 1.0)];
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
        let n1 = m1.count_ones() as f32;
        let n2 = m2.count_ones() as f32;
        let ns = [16.0 - n1 - n2, n1, n2];
        let subs = [s0, s1, s2];
        let mut est = [zero; 2];
        let mut sub = 0;
        while sub < 3 {
            let (n, sm) = (ns[sub], subs[sub]);
            if n > 0.0 {
                let inv_n = V::splat(1.0 / n);
                let c00 = sm[3] - sm[0] * sm[0] * inv_n;
                let c01 = sm[4] - sm[0] * sm[1] * inv_n;
                let c02 = sm[5] - sm[0] * sm[2] * inv_n;
                let c11 = sm[6] - sm[1] * sm[1] * inv_n;
                let c12 = sm[7] - sm[1] * sm[2] * inv_n;
                let c22 = sm[8] - sm[2] * sm[2] * inv_n;
                let v1 = [
                    c00.mul_add(axis[0], c01.mul_add(axis[1], c02 * axis[2])),
                    c01.mul_add(axis[0], c11.mul_add(axis[1], c12 * axis[2])),
                    c02.mul_add(axis[0], c12.mul_add(axis[1], c22 * axis[2])),
                ];
                let v2 = [
                    c00.mul_add(v1[0], c01.mul_add(v1[1], c02 * v1[2])),
                    c01.mul_add(v1[0], c11.mul_add(v1[1], c12 * v1[2])),
                    c02.mul_add(v1[0], c12.mul_add(v1[1], c22 * v1[2])),
                ];
                let num = v1[0].mul_add(v2[0], v1[1].mul_add(v2[1], v1[2] * v2[2]));
                let den = v1[0].mul_add(v1[0], v1[1].mul_add(v1[1], v1[2] * v1[2]));
                let lam = V::select(V::splat(1e-6).simd_lt(den), num / den.simd_max(V::splat(1e-6)), zero);
                let tr = c00 + c11 + c22;
                est[0] += qs[0].mul_add(lam, tr);
                est[1] += qs[1].mul_add(lam, tr);
            }
            sub += 1;
        }
        // Mode 0 only has the first 16 partitions.
        let mut r = usize::from(p >= 16);
        while r < 2 {
            let mut e = est[r];
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

/// Nearest 7-bit level of 8-bit targets (bit replication `2q + q / 64`): the rounded level and
/// its neighbours. Returns (level, value).
#[inline(always)]
fn quant7<V: SimdF32>(x: V) -> (V, V) {
    let zero = V::splat(0.0);
    let c = floor((x * V::splat(127.0 / 255.0) + V::splat(0.5)).simd_max(zero));
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

/// Exact 2-bit indices of the channels `ch` (1 or 2 of them) between endpoint values `a` and
/// `b`: the better of the two entries around the projection. Returns indices and error.
#[inline(always)]
fn index_plane<V: SimdF32, const N: usize>(t: &Texels<V>, ch: [usize; N], a: [V; N], b: [V; N]) -> ([V; 16], V) {
    let zero = V::splat(0.0);
    let mut dd = zero;
    let mut d = [zero; N];
    let mut c = 0;
    while c < N {
        d[c] = b[c] - a[c];
        dd = d[c].mul_add(d[c], dd);
        c += 1;
    }
    let scale = V::select(zero.simd_lt(dd), V::splat(3.0) / dd.simd_max(V::splat(1.0)), zero);
    let inv64 = V::splat(1.0 / 64.0);
    let mut d64 = [zero; N];
    let mut c = 0;
    while c < N {
        d64[c] = d[c] * inv64;
        c += 1;
    }
    let mut idx = [zero; 16];
    let mut err = zero;
    let mut i = 0;
    while i < 16 {
        let mut s = zero;
        let mut c = 0;
        while c < N {
            s = (t[ch[c]][i] - a[c]).mul_add(d[c], s);
            c += 1;
        }
        let j = floor((s * scale).simd_max(zero)).simd_min(V::splat(2.0));
        let w0 = weight(j, 4);
        let w1 = weight(j + V::splat(1.0), 4);
        let (e0, e1) = pair_err(t, ch, i, &a, &d64, w0, w1);
        let up = e1.simd_lt(e0);
        idx[i] = V::select(up, j + V::splat(1.0), j);
        err += e0.simd_min(e1);
        i += 1;
    }
    (idx, err)
}

/// Least-squares endpoints of channels `ch` for 2-bit indices (unclamped lanes with a singular
/// system keep `prev`).
#[inline(always)]
fn ls_plane<V: SimdF32, const N: usize>(
    t: &Texels<V>,
    ch: [usize; N],
    idx: &[V; 16],
    prev: ([V; N], [V; N]),
) -> ([V; N], [V; N]) {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let (mut aa, mut ab, mut bb) = (zero, zero, zero);
    let mut ax = [zero; N];
    let mut bx = [zero; N];
    let inv64 = V::splat(1.0 / 64.0);
    let mut i = 0;
    while i < 16 {
        let wb = weight(idx[i], 4) * inv64;
        let wa = one - wb;
        aa = wa.mul_add(wa, aa);
        ab = wa.mul_add(wb, ab);
        bb = wb.mul_add(wb, bb);
        let mut c = 0;
        while c < N {
            ax[c] = wa.mul_add(t[ch[c]][i], ax[c]);
            bx[c] = wb.mul_add(t[ch[c]][i], bx[c]);
            c += 1;
        }
        i += 1;
    }
    let det = aa * bb - ab * ab;
    let ok = V::splat(1e-3).simd_lt(det);
    let inv = one / V::select(ok, det, one);
    let full = V::splat(255.0);
    let (mut a, mut b) = prev;
    let mut c = 0;
    while c < N {
        a[c] = V::select(ok, ((bb * ax[c] - ab * bx[c]) * inv).simd_clamp(zero, full), a[c]);
        b[c] = V::select(ok, ((aa * bx[c] - ab * ax[c]) * inv).simd_clamp(zero, full), b[c]);
        c += 1;
    }
    (a, b)
}

/// Mode 5 with channel `r` in the scalar (alpha) plane: min/max then a least-squares refit for
/// the scalar plane (8-bit endpoints), principal axis extremes then a refit for the other two
/// channels (7-bit), each with exact 2-bit indices.
#[inline(always)]
fn encode_m5<V: SimdF32>(t: &Texels<V>, r: usize) -> M5<V> {
    let zero = V::splat(0.0);
    let (u, w) = match r {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    // Scalar plane.
    let (mut lo, mut hi) = (t[r][0], t[r][0]);
    let mut i = 1;
    while i < 16 {
        lo = lo.simd_min(t[r][i]);
        hi = hi.simd_max(t[r][i]);
        i += 1;
    }
    let (mut aidx, mut aerr) = index_plane(t, [r], [lo], [hi]);
    let mut av = [lo, hi];
    let (na, nb) = ls_plane(t, [r], &aidx, ([lo], [hi]));
    let (na, nb) = (floor(na[0] + V::splat(0.5)), floor(nb[0] + V::splat(0.5)));
    let (cidx2, cerr2) = index_plane(t, [r], [na], [nb]);
    let take = cerr2.simd_lt(aerr);
    av = [V::select(take, na, av[0]), V::select(take, nb, av[1])];
    aerr = V::select(take, cerr2, aerr);
    let mut i = 0;
    while i < 16 {
        aidx[i] = V::select(take, cidx2[i], aidx[i]);
        i += 1;
    }
    // Color plane (channels u, w).
    let inv16 = V::splat(1.0 / 16.0);
    let (mut mu, mut mw) = (zero, zero);
    let mut i = 0;
    while i < 16 {
        mu += t[u][i];
        mw += t[w][i];
        i += 1;
    }
    let (mu, mw) = (mu * inv16, mw * inv16);
    let (mut cuu, mut cuw, mut cww) = (zero, zero, zero);
    let mut i = 0;
    while i < 16 {
        let (du, dw) = (t[u][i] - mu, t[w][i] - mw);
        cuu = du.mul_add(du, cuu);
        cuw = du.mul_add(dw, cuw);
        cww = dw.mul_add(dw, cww);
        i += 1;
    }
    // Principal axis of the 2x2 scatter by power iteration from the larger diagonal.
    let big_u = cww.simd_lt(cuu);
    let mut ax = [V::select(big_u, cuu, cuw), V::select(big_u, cuw, cww)];
    let mut it = 0;
    while it < 3 {
        let n = [cuu.mul_add(ax[0], cuw * ax[1]), cuw.mul_add(ax[0], cww * ax[1])];
        let mx = n[0].simd_max(zero - n[0]).simd_max(n[1].simd_max(zero - n[1]));
        let s = V::select(V::splat(1e-20).simd_lt(mx), V::splat(1.0) / mx, zero);
        ax = [n[0] * s, n[1] * s];
        it += 1;
    }
    let n2 = ax[0].mul_add(ax[0], ax[1] * ax[1]);
    let flat = n2.simd_lt(V::splat(1e-12));
    ax = [
        V::select(flat, V::splat(1.0), ax[0]),
        V::select(flat, V::splat(1.0), ax[1]),
    ];
    let n2 = V::select(flat, V::splat(2.0), n2);
    let (mut plo, mut phi) = (V::splat(f32::MAX), V::splat(f32::MIN));
    let mut i = 0;
    while i < 16 {
        let s = (t[u][i] - mu).mul_add(ax[0], (t[w][i] - mw) * ax[1]);
        plo = plo.simd_min(s);
        phi = phi.simd_max(s);
        i += 1;
    }
    let full = V::splat(255.0);
    let (plo, phi) = (plo / n2, phi / n2);
    let mut e0 = [
        ax[0].mul_add(plo, mu).simd_clamp(zero, full),
        ax[1].mul_add(plo, mw).simd_clamp(zero, full),
    ];
    let mut e1 = [
        ax[0].mul_add(phi, mu).simd_clamp(zero, full),
        ax[1].mul_add(phi, mw).simd_clamp(zero, full),
    ];
    let mut best_q = [[zero; 2]; 2];
    let mut best_v = [[zero; 2]; 2];
    let mut cidx = [zero; 16];
    let mut cerr = V::splat(f32::INFINITY);
    let mut pass = 0;
    while pass < 2 {
        let (q00, v00) = quant7(e0[0]);
        let (q01, v01) = quant7(e0[1]);
        let (q10, v10) = quant7(e1[0]);
        let (q11, v11) = quant7(e1[1]);
        let (ix, er) = index_plane(t, [u, w], [v00, v01], [v10, v11]);
        let take = er.simd_lt(cerr);
        cerr = V::select(take, er, cerr);
        let qs = [[q00, q01], [q10, q11]];
        let vs = [[v00, v01], [v10, v11]];
        let mut k = 0;
        while k < 2 {
            let mut c = 0;
            while c < 2 {
                best_q[k][c] = V::select(take, qs[k][c], best_q[k][c]);
                best_v[k][c] = V::select(take, vs[k][c], best_v[k][c]);
                c += 1;
            }
            k += 1;
        }
        let mut i = 0;
        while i < 16 {
            cidx[i] = V::select(take, ix[i], cidx[i]);
            i += 1;
        }
        if pass == 0 {
            (e0, e1) = ls_plane(t, [u, w], &cidx, (e0, e1));
        }
        pass += 1;
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

/// Encodes eight opaque blocks; returns the four 32-bit words of each lane's mode-6 block and,
/// for lanes where a 2-subset mode won, that block instead.
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
    // Power iteration from the largest-diagonal row (the first on ties), on the covariance cut
    // to 11 bits (the largest diagonal bounds every entry); each step's vector is cut to 10 bits,
    // so every product stays below 2^24.
    let cs = shrink(rr.simd_max(gg).simd_max(bb), 10);
    let (srr, srg, srb, sgg, sgb, sbb) = (
        floor(rr * cs),
        floor(rg * cs),
        floor(rb * cs),
        floor(gg * cs),
        floor(gb * cs),
        floor(bb * cs),
    );
    let mut v = [srr, srg, srb];
    let mut big = rr;
    let m = big.simd_lt(gg);
    v = [
        V::select(m, srg, v[0]),
        V::select(m, sgg, v[1]),
        V::select(m, sgb, v[2]),
    ];
    big = big.simd_max(gg);
    let m = big.simd_lt(bb);
    v = [
        V::select(m, srb, v[0]),
        V::select(m, sgb, v[1]),
        V::select(m, sbb, v[2]),
    ];
    let mut it = 0;
    while it < cfg.power {
        let w = [
            srr * v[0] + srg * v[1] + srb * v[2],
            srg * v[0] + sgg * v[1] + sgb * v[2],
            srb * v[0] + sgb * v[1] + sbb * v[2],
        ];
        let s = shrink(abs_max(&w), 9);
        v = [floor(w[0] * s), floor(w[1] * s), floor(w[2] * s)];
        it += 1;
    }
    // Degenerate (flat) lanes: any axis works.
    let flat = abs_max(&v).simd_lt(V::splat(0.5));
    v = [
        V::select(flat, V::splat(1.0), v[0]),
        V::select(flat, V::splat(1.0), v[1]),
        V::select(flat, V::splat(1.0), v[2]),
    ];
    // The axis cut to 7 bits, and the endpoints: the points of the axis line through the mean
    // that the extreme projections project from (in 1/16 units, see [`line_ends`]).
    let ws = {
        let s = shrink(abs_max(&v), 6);
        [floor(v[0] * s), floor(v[1] * s), floor(v[2] * s)]
    };
    let (hi, lo) = line_ends(t, &sum, &ws);
    let mut e0 = [lo[0] * inv16, lo[1] * inv16, lo[2] * inv16];
    let mut e1 = [hi[0] * inv16, hi[1] * inv16, hi[2] * inv16];
    // Solid lanes with p-bits 1: an even channel value c is the middle of the odd endpoints
    // c - 1 and c + 1 (indices 7 and 8 decode to c); an odd one is exact.
    if cfg.opaque_pbits {
        let solid = (rr + gg + bb).simd_lt(V::splat(0.5));
        let mut c = 0;
        while c < 3 {
            let m = mean[c];
            let even = (m - floor(m * V::splat(0.5)) * V::splat(2.0)).simd_lt(V::splat(0.5));
            let fix = solid & even & V::splat(0.5).simd_lt(m);
            e0[c] = V::select(fix, m - V::splat(1.0), e0[c]);
            e1[c] = V::select(fix, m + V::splat(1.0), e1[c]);
            c += 1;
        }
    }
    let op = cfg.opaque_pbits;
    let mut ends = quantize(&e0, &e1, op);
    let (mut idx, mut err) = project(t, &ends, cfg.exact);
    if cfg.refits > 0 {
        if !cfg.exact {
            err = error(t, &ends, &idx);
        }
        let mut r = 0;
        while r < cfg.refits {
            let (a, b, ok) = ls_endpoints(t, &idx, (e0, e1));
            let cand = quantize(&a, &b, op);
            let (cidx, cerr) = project(t, &cand, cfg.exact);
            let cerr = if cfg.exact { cerr } else { error(t, &cand, &cidx) };
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
        if !cfg.exact && cfg.refits == 0 {
            err = error(t, &ends, &idx);
        }
        // Lanes worth a second subset: some error left and some spread (as the batch encoder).
        let busy =
            V::splat(cfg.busy_err).simd_lt(err) & V::splat(16.0 * 16.0 * 4.0).simd_lt(rr.simd_max(gg).simd_max(bb));
        if V::mask_any(busy) && cfg.polish_modes && cfg.exact {
            if cfg.uber > 0 {
                uber6(t, &mut ends, &mut idx, &mut err, busy, cfg.uber, cfg.opaque_pbits);
            }
            if cfg.refine > 0 {
                refine6(t, &mut ends, &mut idx, &mut err, busy, cfg.refine);
            }
        }
        if V::mask_any(busy) {
            let k = (cfg.parts as usize).clamp(1, 4);
            // Candidates per mode: [3-bit index ranking (mode 1), 2-bit (mode 3)].
            let cands: [[V::Int; 4]; 2] = match cfg.rank {
                1 => rank_fits::<V, 4>(t, &mean, &v),
                2 => {
                    let bt = super::batch::texels_from(t);
                    let mut res = [[0f32; LANES]; 64];
                    let mut lam = [[0f32; LANES]; 64];
                    super::batch::fits8(&bt, &mut res, &mut lam);
                    let mut out = [[V::Int::splat(0); 4]; 2];
                    for (r, ib) in [3u8, 2].into_iter().enumerate() {
                        let mut pp = [[0i32; LANES]; 4];
                        for l in 0..LANES {
                            let top = super::batch::top4(&res, &lam, l, ib);
                            for (n, x) in top.iter().enumerate() {
                                pp[n][l] = x.1 as i32;
                            }
                        }
                        for n in 0..4 {
                            out[r][n] = V::Int::from_array(pp[n]);
                        }
                    }
                    out
                }
                _ => {
                    let g = guess_partitions::<V, 4>(t, &mean, &v);
                    [g, g]
                }
            };
            let cands3: [[V::Int; 4]; 2] = if cfg.three != 0 {
                rank_fits3::<V, 4>(t, &mean, &v)
            } else {
                [[V::Int::splat(0); 4]; 2]
            };
            // Best encoding of each mode per lane over its candidates, then the winner.
            const MODES: [Two; 4] = [Two::M1, Two::M3, Two::M0, Two::M2];
            let mut best_err = err;
            let mut winner = V::Int::splat(0);
            let mut mode_best: [Option<TwoBest<V>>; 4] = [None, None, None, None];
            for (slot, mode) in MODES.into_iter().enumerate() {
                let (on, list, kk) = match slot {
                    0 => (cfg.two & 1 != 0, &cands[0], k),
                    1 => (
                        cfg.two & 2 != 0,
                        &cands[1],
                        if cfg.parts3 > 0 {
                            (cfg.parts3 as usize).min(4)
                        } else {
                            k
                        },
                    ),
                    2 => (cfg.three & 1 != 0, &cands3[0], (cfg.parts_three as usize).clamp(1, 4)),
                    _ => (cfg.three & 2 != 0, &cands3[1], (cfg.parts_three as usize).clamp(1, 4)),
                };
                if !on {
                    continue;
                }
                let ns = mode.subsets();
                let mut mb: Option<TwoBest<V>> = None;
                for (ci, part) in list.iter().take(kk).enumerate() {
                    let m = part_masks::<V>(*part, mode);
                    let mut s0 = encode_two(t, &m[0], mode, cfg.two_refits);
                    let mut s1 = encode_two(t, &m[1], mode, cfg.two_refits);
                    let mut s2 = if ns == 3 {
                        encode_two(t, &m[2], mode, cfg.two_refits)
                    } else {
                        s0
                    };
                    if cfg.polish_parts && (cfg.polish_top == 0 || ci < cfg.polish_top as usize) {
                        let subs = [&mut s0, &mut s1, &mut s2];
                        for (j, sb) in subs.into_iter().enumerate().take(ns) {
                            if cfg.uber > 0 {
                                uber_two(t, &m[j], sb, mode, cfg.uber);
                            }
                            if cfg.refine > 0 {
                                refine_two(t, &m[j], sb, mode, cfg.refine);
                            }
                        }
                    }
                    let e = if ns == 3 {
                        s0.err + s1.err + s2.err
                    } else {
                        s0.err + s1.err
                    };
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
                    if cfg.polish_modes && (!cfg.polish_parts || cfg.polish_top != 0) {
                        let m = part_masks::<V>(bp, mode);
                        let mut j = 0;
                        while j < ns {
                            if cfg.uber > 0 {
                                uber_two(t, &m[j], &mut b[j], mode, cfg.uber);
                            }
                            if cfg.refine > 0 {
                                refine_two(t, &m[j], &mut b[j], mode, cfg.refine);
                            }
                            j += 1;
                        }
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
                if cfg.uber > 0 && !cfg.polish_modes {
                    let m = part_masks::<V>(bp, mode);
                    let mut j = 0;
                    while j < mode.subsets() {
                        uber_two(t, &m[j], &mut b[j], mode, cfg.uber);
                        j += 1;
                    }
                }
                if cfg.refine > 0 && !cfg.polish_modes {
                    // ±1 refinement of this mode's winners only (refinement never raises the
                    // error, so they stay winners).
                    let m = part_masks::<V>(bp, mode);
                    let mut j = 0;
                    while j < mode.subsets() {
                        refine_two(t, &m[j], &mut b[j], mode, cfg.refine);
                        j += 1;
                    }
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
            if cfg.uber > 0 && cfg.exact && cfg.refine6 && !cfg.polish_modes {
                let m6 = busy & winner.simd_lt(V::Int::splat(1));
                if V::mask_any(m6) {
                    uber6(t, &mut ends, &mut idx, &mut err, m6, cfg.uber, cfg.opaque_pbits);
                }
            }
            if cfg.refine > 0 && cfg.exact && cfg.refine6 && !cfg.polish_modes {
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

/// Encodes a whole surface with the fast mode-6 search. `opts` is passed to the scalar encoder
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
                opaque_pbits: false,
                exact: false,
                two: 0,
                two_refits: 0,
                parts: 1,
                rank: 0,
                refine: 0,
                busy_err: 0.0,
                parts3: 0,
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
                opaque_pbits: true,
                exact: true,
                two: 3,
                two_refits: 1,
                parts: 2,
                rank: 1,
                refine: 1,
                busy_err: 0.0,
                parts3: 0,
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
            opaque_pbits: false,
            exact: true,
            two: 3,
            two_refits: 2,
            parts: 4,
            rank: 1,
            refine: 1,
            busy_err: 0.0,
            parts3: 0,
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
