//! Batch BC7 encoder: 8 opaque blocks per SIMD lane group, for efforts 0 and 1.
//!
//! Every step of the effort-1 search runs on 8 blocks at once, one block per lane, the way
//! ISPC-based encoders (bc7e, Intel's) work: mode 6 (PCA endpoints, p-bit-aware quantization,
//! projected index search, least-squares refits), then modes 1 and 3 on each block's best
//! partitions (each lane uses its own partition masks; lanes that cannot win are masked off and a
//! step runs only if some lane is active), then the ±1 endpoint refinement per mode group. Packing
//! is scalar. Blocks with alpha go to the scalar encoder ([`super::encode`]), which also handles
//! efforts 2 and 3 and serves as the reference.

use super::tables::PART2;
use super::{Bc7Block, ENTROPY, W2, W3, W4, anchors, pack};
use crate::format::BlockOpts;
#[cfg(target_arch = "x86_64")]
use crate::simd::Level;
use crate::simd::{LANES, SimdF32, SimdI32, f32x8};
use crate::surface::Block;

/// Interpolation weights padded for [`SimdF32::lookup17`].
const fn weight_row(w: &[u32]) -> [f32; 32] {
    let mut r = [0f32; 32];
    let mut i = 0;
    while i < w.len() {
        r[i] = w[i] as f32;
        i += 1;
    }
    r
}
const WROW2: [f32; 32] = weight_row(&W2);
const WROW3: [f32; 32] = weight_row(&W3);
const WROW4: [f32; 32] = weight_row(&W4);

/// The opaque modes the batch encoder searches.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Spec {
    /// 1 subset, 7-bit RGBA + per-endpoint p-bit, 4-bit indices.
    M6,
    /// 2 subsets, 6-bit RGB + shared p-bit, 3-bit indices.
    M1,
    /// 2 subsets, 7-bit RGB + per-endpoint p-bit, 2-bit indices.
    M3,
}

impl Spec {
    fn mode(self) -> u8 {
        match self {
            Spec::M6 => 6,
            Spec::M1 => 1,
            Spec::M3 => 3,
        }
    }
    fn color_bits(self) -> u8 {
        match self {
            Spec::M1 => 6,
            _ => 7,
        }
    }
    fn index_bits(self) -> u8 {
        match self {
            Spec::M6 => 4,
            Spec::M1 => 3,
            Spec::M3 => 2,
        }
    }
    fn weights(self) -> &'static [f32; 32] {
        match self {
            Spec::M6 => &WROW4,
            Spec::M1 => &WROW3,
            Spec::M3 => &WROW2,
        }
    }
    /// Channels with endpoints (mode 6 also stores alpha, which must decode to 255 here).
    fn channels(self) -> usize {
        if self == Spec::M6 { 4 } else { 3 }
    }
}

/// Eight blocks' texels, channel-major, plus alpha (always 255 in a batch).
struct Texels<V> {
    x: [[V; 16]; 4],
}

/// One subset's encoding for 8 lanes. Values are small integers held in f32 lanes.
#[derive(Clone, Copy)]
struct SubsetEnc<V> {
    q: [[V; 4]; 2],
    p: [V; 2],
    idx: [V; 16],
    err: V,
}

#[inline(always)]
fn floor_pos<V: SimdF32>(v: V) -> V {
    v.floor()
}

/// 8-bit value of a quantized endpoint channel with its p-bit: `bits` + 1 bits, bit-replicated.
#[inline(always)]
fn dequant<V: SimdF32>(q: V, p: V, bits: u8) -> V {
    let n = bits as i32 + 1;
    let v = (q * V::splat(2.0) + p).to_int_trunc();
    ((v << (8 - n)) | (v >> (2 * n - 8))).to_float()
}

/// Closest quantized value to `v` for p-bit `p` (the scalar `quant_channel`, lane-wise):
/// rounds, then keeps the best of the three neighbours. Returns (q, dequantized).
#[inline(always)]
fn quant<V: SimdF32>(v: V, p: V, bits: u8) -> (V, V) {
    let maxq = ((1u32 << bits) - 1) as f32;
    let levels = ((1u32 << (bits + 1)) - 1) as f32;
    let ideal = (v.simd_clamp(V::splat(0.0), V::splat(255.0)) * V::splat(levels / 255.0) - p) * V::splat(0.5);
    let c = floor_pos((ideal + V::splat(0.5)).simd_max(V::splat(0.0)));
    let mut best_q = V::splat(0.0);
    let mut best_d = V::splat(0.0);
    let mut best_e = V::splat(f32::INFINITY);
    for off in [-1.0f32, 0.0, 1.0] {
        let q = c + V::splat(off);
        let ok = V::splat(-0.5).simd_lt(q) & q.simd_lt(V::splat(maxq + 0.5));
        let qc = q.simd_clamp(V::splat(0.0), V::splat(maxq));
        let d = dequant(qc, p, bits);
        let e = (d - v) * (d - v);
        let better = ok & e.simd_lt(best_e);
        best_q = V::select(better, qc, best_q);
        best_d = V::select(better, d, best_d);
        best_e = V::select(better, e, best_e);
    }
    (best_q, best_d)
}

/// Mode 6 under entropy reduction: 64 evenly spread 7-bit levels (`(j * 127 + 31) / 63`).
#[inline(always)]
fn quant_reduced<V: SimdF32>(v: V, p: V) -> (V, V) {
    let jc = floor_pos(
        ((v.simd_clamp(V::splat(0.0), V::splat(255.0)) - p) * V::splat(0.5 * 63.0 / 127.0) + V::splat(0.5))
            .simd_max(V::splat(0.0)),
    );
    let mut best_q = V::splat(0.0);
    let mut best_d = V::splat(0.0);
    let mut best_e = V::splat(f32::INFINITY);
    for off in [-1.0f32, 0.0, 1.0] {
        let j = jc + V::splat(off);
        let ok = V::splat(-0.5).simd_lt(j) & j.simd_lt(V::splat(63.5));
        let jj = j.simd_clamp(V::splat(0.0), V::splat(63.0));
        let q = floor_pos((jj * V::splat(127.0) + V::splat(31.0)) * V::splat(1.0 / 63.0));
        let d = dequant(q, p, 7);
        let e = (d - v) * (d - v);
        let better = ok & e.simd_lt(best_e);
        best_q = V::select(better, q, best_q);
        best_d = V::select(better, d, best_d);
        best_e = V::select(better, e, best_e);
    }
    (best_q, best_d)
}

/// Quantizes one endpoint's first `nch` channels for p-bit `p`: (q, dequantized, error).
///
/// Kernels avoid closures (including `array::from_fn`): a closure does not inherit the caller's
/// `#[target_feature]`, so AVX2 intrinsics inside one turn into real function calls.
#[inline(always)]
fn quant_endpoint<V: SimdF32>(e: &[V; 4], p: V, bits: u8, nch: usize) -> ([V; 4], [V; 4], V) {
    let mut qq = [V::splat(0.0); 4];
    let mut dd = [V::splat(255.0); 4];
    let mut err = V::splat(0.0);
    for c in 0..nch {
        let (q, d) = quant(e[c], p, bits);
        qq[c] = q;
        dd[c] = d;
        err += (d - e[c]) * (d - e[c]);
    }
    (qq, dd, err)
}

/// Quantizes an endpoint pair, choosing p-bits by endpoint error (the scalar
/// `quantize_endpoints`). Returns (q, p, dequantized endpoints).
#[inline(always)]
fn quantize<V: SimdF32>(e: &[[V; 4]; 2], spec: Spec, reduce: bool) -> ([[V; 4]; 2], [V; 2], [[V; 4]; 2]) {
    let nch = spec.channels();
    let bits = spec.color_bits();
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let mut q = [[zero; 4]; 2];
    let mut d = [[V::splat(255.0); 4]; 2];
    let mut p = [zero; 2];
    if reduce && spec == Spec::M6 {
        // Opaque blocks: p-bits (1, 1) keep alpha exactly 255.
        p = [one, one];
        for k in 0..2 {
            for c in 0..4 {
                (q[k][c], d[k][c]) = quant_reduced(e[k][c], one);
            }
        }
        return (q, p, d);
    }
    match spec {
        Spec::M1 => {
            let (a0, ad0, ae0) = quant_endpoint(&e[0], zero, bits, nch);
            let (b0, bd0, be0) = quant_endpoint(&e[1], zero, bits, nch);
            let (a1, ad1, ae1) = quant_endpoint(&e[0], one, bits, nch);
            let (b1, bd1, be1) = quant_endpoint(&e[1], one, bits, nch);
            // p = 1 only if strictly better (the scalar loop keeps the first minimum).
            let use1 = (ae1 + be1).simd_lt(ae0 + be0);
            for c in 0..4 {
                q[0][c] = V::select(use1, a1[c], a0[c]);
                q[1][c] = V::select(use1, b1[c], b0[c]);
                d[0][c] = V::select(use1, ad1[c], ad0[c]);
                d[1][c] = V::select(use1, bd1[c], bd0[c]);
            }
            p = [V::select(use1, one, zero); 2];
        }
        _ => {
            for k in 0..2 {
                let (q0, d0, e0) = quant_endpoint(&e[k], zero, bits, nch);
                let (q1, d1, e1) = quant_endpoint(&e[k], one, bits, nch);
                let use1 = e1.simd_lt(e0);
                for c in 0..4 {
                    q[k][c] = V::select(use1, q1[c], q0[c]);
                    d[k][c] = V::select(use1, d1[c], d0[c]);
                }
                p[k] = V::select(use1, one, zero);
            }
        }
    }
    (q, p, d)
}

/// Palette entry `(64 - w) * e0 + w * e1 + 32 >> 6`, exact in f32.
#[inline(always)]
fn interp<V: SimdF32>(e0: V, e1: V, w: V) -> V {
    floor_pos(((V::splat(64.0) - w) * e0 + w * e1 + V::splat(32.0)) * V::splat(1.0 / 64.0))
}

/// Error of texel `i` against the palette entry with weight `w`.
#[inline(always)]
fn texel_error<V: SimdF32>(t: &Texels<V>, d: &[[V; 4]; 2], nch: usize, i: usize, w: V) -> V {
    let mut e = V::splat(0.0);
    for c in 0..nch {
        let diff = t.x[c][i] - interp(d[0][c], d[1][c], w);
        e += diff * diff;
    }
    e
}

/// Best index per texel for dequantized endpoints `d` and the error, summed over members `m`.
/// Ties go to the lowest index, like the scalar exhaustive search.
#[inline(always)]
fn search<V: SimdF32>(t: &Texels<V>, m: &[V; 16], d: &[[V; 4]; 2], spec: Spec, lowest_tie: bool) -> ([V; 16], V) {
    let nch = spec.channels();
    let wrow = spec.weights();
    let n = 1usize << spec.index_bits();
    let mut idx = [V::splat(0.0); 16];
    let mut total = V::splat(0.0);
    if n <= 8 {
        // Exhaustive: the palette is small.
        let mut pal = [[V::splat(0.0); 4]; 8];
        for k in 0..n {
            let w = V::splat(wrow[k]);
            for c in 0..nch {
                pal[k][c] = interp(d[0][c], d[1][c], w);
            }
        }
        for i in 0..16 {
            let mut best_e = V::splat(f32::INFINITY);
            let mut best_k = V::splat(0.0);
            for (k, pk) in pal.iter().enumerate().take(n) {
                let mut e = V::splat(0.0);
                for c in 0..nch {
                    let diff = t.x[c][i] - pk[c];
                    e += diff * diff;
                }
                let better = e.simd_lt(best_e);
                best_e = V::select(better, e, best_e);
                best_k = V::select(better, V::splat(k as f32), best_k);
            }
            idx[i] = best_k;
            total += m[i] * best_e;
        }
    } else {
        // 16 entries: project onto the endpoint line, compare the nearest entry and its
        // neighbours, then step down past equal-error duplicates.
        let mut dir = [V::splat(0.0); 4];
        let mut dd = V::splat(0.0);
        for c in 0..nch {
            dir[c] = d[1][c] - d[0][c];
            dd += dir[c] * dir[c];
        }
        let scale = V::select(V::splat(0.0).simd_lt(dd), V::splat((n - 1) as f32) / dd, V::splat(0.0));
        let top = V::splat((n - 1) as f32);
        for i in 0..16 {
            let mut s = V::splat(0.0);
            for c in 0..nch {
                s += (t.x[c][i] - d[0][c]) * dir[c];
            }
            let g = floor_pos((s * scale + V::splat(0.5)).simd_clamp(V::splat(0.0), top));
            let mut best_e = V::splat(f32::INFINITY);
            let mut best_k = V::splat(0.0);
            for off in [-1.0f32, 0.0, 1.0] {
                let k = (g + V::splat(off)).simd_clamp(V::splat(0.0), top);
                let e = texel_error(t, d, nch, i, V::lookup17(wrow, k.to_int_trunc()));
                let better = e.simd_lt(best_e);
                best_e = V::select(better, e, best_e);
                best_k = V::select(better, k, best_k);
            }
            // Errors are integers, so "equal" is "less than best + 0.5". This only changes the
            // index, never the error, so error-only callers skip it.
            if lowest_tie {
                loop {
                    let km1 = (best_k - V::splat(1.0)).simd_max(V::splat(0.0));
                    let e_lo = texel_error(t, d, nch, i, V::lookup17(wrow, km1.to_int_trunc()));
                    let tie = V::splat(0.0).simd_lt(best_k) & e_lo.simd_lt(best_e + V::splat(0.5));
                    if !V::mask_any(tie) {
                        break;
                    }
                    best_k = V::select(tie, km1, best_k);
                }
            }
            idx[i] = best_k;
            total += m[i] * best_e;
        }
    }
    (idx, total)
}

/// Lane-parallel principal axis of a 3x3 symmetric matrix (power iteration).
#[inline(always)]
fn principal_axis3<V: SimdF32>(m: &[[V; 3]; 3]) -> [V; 3] {
    let zero = V::splat(0.0);
    let mut v = m[0];
    let mut big = m[0][0];
    for a in 1..3 {
        let more = big.simd_lt(m[a][a]);
        for b in 0..3 {
            v[b] = V::select(more, m[a][b], v[b]);
        }
        big = big.simd_max(m[a][a]);
    }
    for _ in 0..8 {
        let w = [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ];
        let n2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        let inv = V::select(zero.simd_lt(n2), V::splat(1.0) / n2.simd_sqrt(), zero);
        v = [w[0] * inv, w[1] * inv, w[2] * inv];
    }
    // Flat subsets: any unit axis works.
    let flat = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).simd_lt(V::splat(1e-12));
    if V::mask_any(flat) {
        let u = V::splat(1.0 / 3f32.sqrt());
        v = [
            V::select(flat, u, v[0]),
            V::select(flat, u, v[1]),
            V::select(flat, u, v[2]),
        ];
    }
    v
}

/// Initial endpoints of the members `m`: extremes of their projection on the principal axis.
#[inline(always)]
fn pca_endpoints<V: SimdF32>(t: &Texels<V>, m: &[V; 16]) -> [[V; 4]; 2] {
    let zero = V::splat(0.0);
    let mut n = zero;
    let mut mean = [zero; 3];
    for i in 0..16 {
        n += m[i];
        for c in 0..3 {
            mean[c] += m[i] * t.x[c][i];
        }
    }
    let inv_n = V::select(zero.simd_lt(n), V::splat(1.0) / n, zero);
    for c in 0..3 {
        mean[c] = mean[c] * inv_n;
    }
    let mut cov = [[zero; 3]; 3];
    for i in 0..16 {
        let dx = [t.x[0][i] - mean[0], t.x[1][i] - mean[1], t.x[2][i] - mean[2]];
        for a in 0..3 {
            for b in a..3 {
                cov[a][b] += m[i] * dx[a] * dx[b];
            }
        }
    }
    for a in 0..3 {
        for b in 0..a {
            cov[a][b] = cov[b][a];
        }
    }
    let axis = principal_axis3(&cov);
    let mut lo = V::splat(f32::INFINITY);
    let mut hi = V::splat(f32::NEG_INFINITY);
    for i in 0..16 {
        let s = (t.x[0][i] - mean[0]) * axis[0] + (t.x[1][i] - mean[1]) * axis[1] + (t.x[2][i] - mean[2]) * axis[2];
        let member = zero.simd_lt(m[i]);
        lo = V::select(member, lo.simd_min(s), lo);
        hi = V::select(member, hi.simd_max(s), hi);
    }
    let (lo, hi) = (lo.simd_min(zero), hi.simd_max(zero));
    let mut e = [[V::splat(255.0); 4]; 2];
    for c in 0..3 {
        e[0][c] = (mean[c] + axis[c] * lo).simd_clamp(zero, V::splat(255.0));
        e[1][c] = (mean[c] + axis[c] * hi).simd_clamp(zero, V::splat(255.0));
    }
    e
}

/// Least-squares endpoints for the members' current indices; lanes with a singular system keep
/// `prev`.
#[inline(always)]
fn ls_refit<V: SimdF32>(t: &Texels<V>, m: &[V; 16], idx: &[V; 16], spec: Spec, prev: &[[V; 4]; 2]) -> [[V; 4]; 2] {
    let zero = V::splat(0.0);
    let (mut aa, mut ab, mut bb) = (zero, zero, zero);
    let mut ax = [zero; 3];
    let mut bx = [zero; 3];
    let wrow = spec.weights();
    for i in 0..16 {
        let tw = V::lookup17(wrow, idx[i].to_int_trunc()) * V::splat(1.0 / 64.0);
        let si = V::splat(1.0) - tw;
        let (ms, mt) = (m[i] * si, m[i] * tw);
        aa += ms * si;
        ab += ms * tw;
        bb += mt * tw;
        for c in 0..3 {
            ax[c] += ms * t.x[c][i];
            bx[c] += mt * t.x[c][i];
        }
    }
    let det = aa * bb - ab * ab;
    let ok = V::splat(1e-6).simd_lt(det);
    let inv = V::select(ok, V::splat(1.0) / det, zero);
    let mut e = *prev;
    for c in 0..3 {
        let e0 = ((bb * ax[c] - ab * bx[c]) * inv).simd_clamp(zero, V::splat(255.0));
        let e1 = ((aa * bx[c] - ab * ax[c]) * inv).simd_clamp(zero, V::splat(255.0));
        e[0][c] = V::select(ok, e0, prev[0][c]);
        e[1][c] = V::select(ok, e1, prev[1][c]);
    }
    e
}

/// Encodes one subset (members `m`) of 8 lanes: PCA endpoints, then quantize / search / refit
/// `iters` times, keeping each lane's best.
#[inline(always)]
fn encode_subset<V: SimdF32>(t: &Texels<V>, m: &[V; 16], spec: Spec, iters: u32, reduce: bool) -> SubsetEnc<V> {
    let mut e = pca_endpoints(t, m);
    let mut best: Option<SubsetEnc<V>> = None;
    for it in 0..iters.max(1) {
        let (q, p, d) = quantize(&e, spec, reduce);
        let (idx, err) = search(t, m, &d, spec, true);
        let cand = SubsetEnc { q, p, idx, err };
        best = Some(match best {
            None => cand,
            Some(b) => merge(err.simd_lt(b.err), &cand, &b),
        });
        if it + 1 < iters {
            e = ls_refit(t, m, &idx, spec, &e);
        }
    }
    best.unwrap()
}

/// Per lane: `a` where `take` is set, else `b`.
#[inline(always)]
fn merge<V: SimdF32>(take: V::Mask, a: &SubsetEnc<V>, b: &SubsetEnc<V>) -> SubsetEnc<V> {
    let mut out = *b;
    for k in 0..2 {
        for c in 0..4 {
            out.q[k][c] = V::select(take, a.q[k][c], b.q[k][c]);
        }
        out.p[k] = V::select(take, a.p[k], b.p[k]);
    }
    for i in 0..16 {
        out.idx[i] = V::select(take, a.idx[i], b.idx[i]);
    }
    out.err = V::select(take, a.err, b.err);
    out
}

/// Error of one subset for given quantized endpoints and p-bits.
#[inline(always)]
fn subset_error<V: SimdF32>(t: &Texels<V>, m: &[V; 16], q: &[[V; 4]; 2], p: &[V; 2], spec: Spec) -> V {
    let bits = spec.color_bits();
    let mut d = [[V::splat(255.0); 4]; 2];
    for k in 0..2 {
        for c in 0..spec.channels() {
            d[k][c] = dequant(q[k][c], p[k], bits);
        }
    }
    search(t, m, &d, spec, false).1
}

/// One pass of the ±1 endpoint / p-bit refinement (the scalar `refine_endpoints`) for lanes in
/// `active`; returns the refined encoding with fresh indices.
#[inline(always)]
fn refine<V: SimdF32>(
    t: &Texels<V>,
    m: &[V; 16],
    s: &SubsetEnc<V>,
    spec: Spec,
    active: V::Mask,
    passes: u32,
) -> SubsetEnc<V> {
    /// Moves tried per channel: the 4 single-endpoint steps only. The paired steps (both
    /// endpoints together or apart) cost 0.01 dB to drop and almost halve the evaluations.
    const REFINE_MOVES: usize = 4;
    const MOVES: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
    ];
    let maxq = V::splat(((1u32 << spec.color_bits()) - 1) as f32);
    let zero = V::splat(0.0);
    let mut q = s.q;
    let mut p = s.p;
    let mut cur = subset_error(t, m, &q, &p, spec);
    for _ in 0..passes {
        // RGB only: batch blocks are opaque, and a move of mode 6's alpha endpoints (which must
        // decode to 255) can only add error.
        for c in 0..3 {
            for (d0, d1) in MOVES.into_iter().take(REFINE_MOVES) {
                let a = q[0][c] + V::splat(d0);
                let b = q[1][c] + V::splat(d1);
                let ok = V::splat(-0.5).simd_lt(a)
                    & V::splat(-0.5).simd_lt(b)
                    & a.simd_lt(maxq + V::splat(0.5))
                    & b.simd_lt(maxq + V::splat(0.5));
                let mut cq = q;
                cq[0][c] = a.simd_clamp(zero, maxq);
                cq[1][c] = b.simd_clamp(zero, maxq);
                let e = subset_error(t, m, &cq, &p, spec);
                let take = active & ok & e.simd_lt(cur);
                q[0][c] = V::select(take, cq[0][c], q[0][c]);
                q[1][c] = V::select(take, cq[1][c], q[1][c]);
                cur = V::select(take, e, cur);
            }
        }
        let flips: &[[f32; 2]] = if spec == Spec::M1 {
            &[[1.0, 1.0]]
        } else {
            &[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        };
        for f in flips {
            let cp = [
                if f[0] > 0.0 { V::splat(1.0) - p[0] } else { p[0] },
                if f[1] > 0.0 { V::splat(1.0) - p[1] } else { p[1] },
            ];
            let e = subset_error(t, m, &q, &cp, spec);
            let take = active & e.simd_lt(cur);
            p = [V::select(take, cp[0], p[0]), V::select(take, cp[1], p[1])];
            cur = V::select(take, e, cur);
        }
    }
    let bits = spec.color_bits();
    let mut d = [[V::splat(255.0); 4]; 2];
    for k in 0..2 {
        for c in 0..spec.channels() {
            d[k][c] = dequant(q[k][c], p[k], bits);
        }
    }
    let (idx, err) = search(t, m, &d, spec, true);
    let refined = SubsetEnc { q, p, idx, err };
    merge(active, &refined, s)
}

/// Per-lane best candidate so far: mode, partition, up to two subsets, error and score.
#[derive(Clone, Copy)]
struct Best<V> {
    spec: [Spec; LANES],
    part: [u8; LANES],
    sub: [SubsetEnc<V>; 2],
    err: V,
    score: V,
}

/// Second-moment slots of 3 channels: products (a, b) with a <= b.
const PAIRS3: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

#[inline(always)]
const fn pair3(a: usize, b: usize) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    match (a, b) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        _ => 5,
    }
}

/// Rayleigh quotient `v·Sv / v·v` after `iters` normalized power steps from `v`: a lower bound
/// on λmax that converges quickly from a good start.
#[inline(always)]
fn power_lambda<V: SimdF32>(m: &[[V; 3]; 3], mut v: [V; 3], iters: u32) -> V {
    let zero = V::splat(0.0);
    for _ in 0..iters {
        let w = [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ];
        let n2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        let inv = V::select(zero.simd_lt(n2), V::splat(1.0) / n2.simd_sqrt(), zero);
        v = [w[0] * inv, w[1] * inv, w[2] * inv];
    }
    let w = [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ];
    let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let vsv = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    V::select(zero.simd_lt(vv), vsv / vv, zero)
}

/// (`trace(S) - λmax`, `λmax`) of a subset with `n` texels, sums `s` and second moments `q`.
/// λmax is the larger of two short power iterations, from the block's principal axis (subset
/// axes are usually close to it) and from the column with the largest diagonal entry.
#[inline(always)]
fn subset_fit<V: SimdF32>(n: f32, s: &[V; 3], q: &[V; 6], block_axis: &[V; 3]) -> (V, V) {
    let zero = V::splat(0.0);
    let inv_n = V::splat(1.0 / n);
    let mut m = [[zero; 3]; 3];
    for a in 0..3 {
        for b in a..3 {
            let v = q[pair3(a, b)] - s[a] * s[b] * inv_n;
            m[a][b] = v;
            m[b][a] = v;
        }
    }
    let trace = m[0][0] + m[1][1] + m[2][2];
    let mut col = m[0];
    let mut big = m[0][0];
    for a in 1..3 {
        let more = big.simd_lt(m[a][a]);
        for b in 0..3 {
            col[b] = V::select(more, m[a][b], col[b]);
        }
        big = big.simd_max(m[a][a]);
    }
    // Two steps per start: λ is within a few percent of the total scatter even when the top two
    // eigenvalues are close, which ranks partitions as well as 8 steps did (BD-rate within
    // ±0.05% on the check corpora) at a quarter of the cost. An overestimated residual can only
    // make the pruning skip a marginal partition, never produce a wrong block.
    let lambda = power_lambda(&m, *block_axis, 2).simd_max(power_lambda(&m, col, 2));
    ((trace - lambda).simd_max(zero), lambda.simd_max(zero))
}

/// Line fits of all 64 two-subset partitions for 8 blocks at once, one block per lane: for each
/// partition, `(residual, λ)` summed over its two subsets (see [`super::partition`]). Subset
/// membership is the same in every lane, so the moments are plain sums over member texels.
#[inline(always)]
fn fits8<V: SimdF32>(t: &Texels<V>, res: &mut [[f32; LANES]; 64], lam: &mut [[f32; LANES]; 64]) {
    let zero = V::splat(0.0);
    // Center on the block mean to keep the f32 scatter accurate.
    let mut mean = [zero; 3];
    for i in 0..16 {
        for c in 0..3 {
            mean[c] += t.x[c][i];
        }
    }
    let mut xc = [[zero; 16]; 3];
    let mut xx = [[zero; 16]; 6];
    let mut tot_s = [zero; 3];
    let mut tot_q = [zero; 6];
    for c in 0..3 {
        mean[c] = mean[c] * V::splat(1.0 / 16.0);
    }
    for i in 0..16 {
        for c in 0..3 {
            xc[c][i] = t.x[c][i] - mean[c];
            tot_s[c] += xc[c][i];
        }
        for (j, &(a, b)) in PAIRS3.iter().enumerate() {
            xx[j][i] = xc[a][i] * xc[b][i];
            tot_q[j] += xx[j][i];
        }
    }
    let mut cov = [[zero; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            cov[a][b] = tot_q[pair3(a, b)];
        }
    }
    let axis = principal_axis3(&cov);
    for p in 0..64 {
        let mask = PART2[p];
        let n1 = mask.count_ones() as f32;
        let mut s1 = [zero; 3];
        let mut q1 = [zero; 6];
        for i in 0..16 {
            if mask >> i & 1 != 0 {
                for c in 0..3 {
                    s1[c] += xc[c][i];
                }
                for j in 0..6 {
                    q1[j] += xx[j][i];
                }
            }
        }
        let mut s0 = [zero; 3];
        let mut q0 = [zero; 6];
        for c in 0..3 {
            s0[c] = tot_s[c] - s1[c];
        }
        for j in 0..6 {
            q0[j] = tot_q[j] - q1[j];
        }
        let (r1, l1) = subset_fit(n1, &s1, &q1, &axis);
        let (r0, l0) = subset_fit(16.0 - n1, &s0, &q0, &axis);
        res[p] = (r0 + r1).to_array();
        lam[p] = (l0 + l1).to_array();
    }
}

/// The 4 most promising two-subset partitions of lane `l` for `ib`-bit indices, as (bound,
/// partition): ranked by `residual + Δw² · λ` like [`super::top_partitions`], bound = residual.
fn top4(res: &[[f32; LANES]; 64], lam: &[[f32; LANES]; 64], l: usize, ib: u8) -> [(f32, u8); 4] {
    let dw = 1.0 / ((1u32 << ib) - 1) as f32;
    let mut best = [(f32::INFINITY, f32::INFINITY, 0u8); 4];
    for p in 0..64u8 {
        let (r, lm) = (res[p as usize][l], lam[p as usize][l]);
        let est = r + dw * dw * lm;
        if est < best[3].0 {
            let mut k = 3;
            while k > 0 && est < best[k - 1].0 {
                best[k] = best[k - 1];
                k -= 1;
            }
            best[k] = (est, r, p);
        }
    }
    best.map(|(_, bound, p)| (bound, p))
}

/// Subset membership masks (1.0 / 0.0 per texel) of lanes' 2-subset partitions.
#[inline(always)]
fn masks2<V: SimdF32>(parts: &[u8; LANES]) -> [[V; 16]; 2] {
    let codes = V::Int::from_array(parts.map(|p| PART2[p as usize] as i32));
    let one = V::Int::splat(1);
    let mut m = [[V::splat(0.0); 16]; 2];
    for i in 0..16 {
        let s1 = ((codes >> i as i32) & one).to_float();
        m[1][i] = s1;
        m[0][i] = V::splat(1.0) - s1;
    }
    m
}

#[inline(always)]
fn spec_lanes<V: SimdF32>(best: &Best<V>, spec: Spec) -> V::Mask {
    V::mask_from_bools(best.spec.map(|s| s == spec))
}

#[inline(always)]
fn encode8<V: SimdF32>(blocks: &[Block; LANES], opts: BlockOpts) -> [[u8; 16]; LANES] {
    let effort = opts.effort.min(1) as usize;
    let reduce = opts.reduce_entropy;
    let iters = [1u32, 2][effort];
    let mut t = Texels {
        x: [[V::splat(255.0); 16]; 4],
    };
    for c in 0..3 {
        for i in 0..16 {
            let mut lanes = [0f32; LANES];
            for l in 0..LANES {
                lanes[l] = blocks[l][i][c] as f32;
            }
            t.x[c][i] = V::from_array(lanes);
        }
    }
    let all = [V::splat(1.0); 16];
    let m6 = encode_subset(&t, &all, Spec::M6, iters, reduce);
    let m6_weight = if reduce { ENTROPY.mode6_weight_opaque } else { 1.0 };
    // Scores round like the scalar encoder's `(err * weight).round()`.
    let score6 = if reduce {
        floor_pos(m6.err * V::splat(m6_weight) + V::splat(0.5))
    } else {
        m6.err
    };
    let mut best = Best {
        spec: [Spec::M6; LANES],
        part: [0; LANES],
        sub: [m6, m6],
        err: m6.err,
        score: score6,
    };
    if effort >= 1 {
        // Low-energy blocks stay in mode 6 (as in the scalar encoder).
        let mut busy = [false; LANES];
        let mut tops = [[[(f32::INFINITY, 0u8); 4]; LANES]; 2];
        let score_arr = best.score.to_array();
        // The entropy-reduced base (RDO's starting point at λ >= 1) also keeps mode 6 when its
        // weighted score is at most 512: a multi-subset winner that only shaves a little error adds
        // bytes RDO then has to undo. On Kodak-24 this halved the base's time (on top of the alt
        // base's busy_err 384, see rdo_alt_cfg) and no λ 10-40 cell vs Oodle moved the wrong way.
        let rbusy: f32 = if reduce { 512.0 } else { 0.0 };
        for l in 0..LANES {
            busy[l] = super::max_channel_energy(&blocks[l], 3) >= 16.0 * 4.0 && score_arr[l] > rbusy;
        }
        if busy.iter().any(|&b| b) {
            let mut res = [[0f32; LANES]; 64];
            let mut lam = [[0f32; LANES]; 64];
            fits8(&t, &mut res, &mut lam);
            for l in 0..LANES {
                if busy[l] {
                    tops[0][l] = top4(&res, &lam, l, 3);
                    tops[1][l] = top4(&res, &lam, l, 2);
                }
            }
        }
        for (ti, spec) in [(0usize, Spec::M1), (1, Spec::M3)] {
            for r in 0..4 {
                let score = best.score.to_array();
                let mut parts = [0u8; LANES];
                let mut active = [false; LANES];
                for l in 0..LANES {
                    let (bound, p) = tops[ti][l][r];
                    if bound.is_finite() {
                        parts[l] = p;
                        active[l] = busy[l] && score[l] > 0.0 && ((bound * 0.9) as u64 as f32) < score[l];
                    }
                }
                if !active.iter().any(|&a| a) {
                    continue;
                }
                let m = masks2::<V>(&parts);
                let s0 = encode_subset(&t, &m[0], spec, iters, false);
                let s1 = encode_subset(&t, &m[1], spec, iters, false);
                let err = s0.err + s1.err;
                let take = V::mask_from_bools(active) & err.simd_lt(best.score);
                let take_arr = V::mask_to_bools(take);
                best.sub[0] = merge(take, &s0, &best.sub[0]);
                best.sub[1] = merge(take, &s1, &best.sub[1]);
                best.err = V::select(take, err, best.err);
                best.score = V::select(take, err, best.score);
                for l in 0..LANES {
                    if take_arr[l] {
                        best.spec[l] = spec;
                        best.part[l] = parts[l];
                    }
                }
            }
        }
        // ±1 refinement, one mode group at a time. Low-energy blocks are left alone like in the
        // scalar encoder (tiny gains, less repetitive bytes), and entropy-reduced mode 6 keeps
        // its restricted levels.
        let busy_mask = V::mask_from_bools(busy);
        for spec in [Spec::M6, Spec::M1, Spec::M3] {
            if reduce && spec == Spec::M6 {
                continue;
            }
            let lanes = spec_lanes(&best, spec) & busy_mask & V::splat(0.0).simd_lt(best.err);
            if !V::mask_any(lanes) {
                continue;
            }
            if spec == Spec::M6 {
                best.sub[0] = refine(&t, &all, &best.sub[0], spec, lanes, 1);
            } else {
                let m = masks2::<V>(&best.part);
                best.sub[0] = refine(&t, &m[0], &best.sub[0], spec, lanes, 1);
                best.sub[1] = refine(&t, &m[1], &best.sub[1], spec, lanes, 1);
            }
        }
    }
    // Pack each lane.
    let mut q = [[[[0f32; LANES]; 4]; 2]; 2];
    let mut p = [[[0f32; LANES]; 2]; 2];
    let mut idx = [[[0f32; LANES]; 16]; 2];
    for s in 0..2 {
        for k in 0..2 {
            for c in 0..4 {
                q[s][k][c] = best.sub[s].q[k][c].to_array();
            }
            p[s][k] = best.sub[s].p[k].to_array();
        }
        for i in 0..16 {
            idx[s][i] = best.sub[s].idx[i].to_array();
        }
    }
    pack_lanes(&best.spec, &best.part, &q, &p, &idx)
}

/// Scalar packing of each lane's winner, with the anchor-index fix-up.
fn pack_lanes(
    specs: &[Spec; LANES],
    parts: &[u8; LANES],
    q: &[[[[f32; LANES]; 4]; 2]; 2],
    p: &[[[f32; LANES]; 2]; 2],
    idx: &[[[f32; LANES]; 16]; 2],
) -> [[u8; 16]; LANES] {
    std::array::from_fn(|l| {
        let spec = specs[l];
        let ns = if spec == Spec::M6 { 1 } else { 2 };
        let mut blk = Bc7Block {
            mode: spec.mode(),
            partition: parts[l],
            ..Default::default()
        };
        if spec == Spec::M6 {
            blk.partition = 0;
        }
        let n = 1u8 << spec.index_bits();
        let anc = anchors(ns, blk.partition);
        for s in 0..ns as usize {
            let mut qq = [[0u8; 4]; 2];
            let mut pp = [p[s][0][l] as u8, p[s][1][l] as u8];
            for k in 0..2 {
                for c in 0..spec.channels() {
                    qq[k][c] = q[s][k][c][l] as u8;
                }
            }
            if spec != Spec::M6 {
                // Opaque 2-subset modes have no alpha field.
                qq[0][3] = 0;
                qq[1][3] = 0;
            }
            let members: Vec<usize> = (0..16)
                .filter(|&i| super::subset_of(ns, blk.partition, i) == s)
                .collect();
            let mut ix = [0u8; 16];
            for &i in &members {
                ix[i] = idx[s][i][l] as u8;
            }
            if ix[anc[s]] >= n / 2 {
                qq.swap(0, 1);
                pp.swap(0, 1);
                for &i in &members {
                    ix[i] = n - 1 - ix[i];
                }
            }
            blk.ep[s] = qq;
            blk.pbits[s] = pp;
            for &i in &members {
                blk.idx[i] = ix[i];
            }
        }
        pack(&blk)
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode8_avx2(blocks: &[Block; LANES], opts: BlockOpts) -> [[u8; 16]; LANES] {
    encode8::<crate::simd::avx2::F>(blocks, opts)
}

/// Encodes up to 8 blocks with effort 0 or 1. Opaque blocks go through the batch encoder;
/// blocks with alpha use the scalar encoder.
pub fn encode_batch(blocks: &[Block], w: &[u32; 4], opts: BlockOpts) -> Vec<[u8; 16]> {
    debug_assert!(blocks.len() <= LANES && opts.effort <= 1);
    let opaque = |b: &Block| b.iter().all(|t| t[3] == 255);
    if *w != [1, 1, 1, 1] || !blocks.iter().any(opaque) {
        return blocks.iter().map(|b| super::encode(b, w, opts)).collect();
    }
    // Pad with a copy of an opaque block; alpha lanes are re-encoded by the scalar encoder.
    let fill = *blocks.iter().find(|b| opaque(b)).unwrap();
    let lanes: [Block; LANES] = std::array::from_fn(|l| match blocks.get(l) {
        Some(b) if opaque(b) => *b,
        _ => fill,
    });
    let out = match crate::simd::level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { encode8_avx2(&lanes, opts) },
        _ => encode8::<f32x8>(&lanes, opts),
    };
    blocks
        .iter()
        .enumerate()
        .map(|(l, b)| if opaque(b) { out[l] } else { super::encode(b, w, opts) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Throwaway stage profile of the batch encoder (AVX2 path) over a Kodak image.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn profile_batch_stages() {
        use crate::simd::avx2::F;
        let path = std::env::var("MIPPLE_PROFILE_IMG").unwrap_or("test-images/kodak/kodim01.png".into());
        let img = image::open(path).unwrap().to_rgba8();
        let mut surf = crate::surface::Surface::new(img.width(), img.height());
        for (x, y, p) in img.enumerate_pixels() {
            surf.set(x, y, p.0);
        }
        let blocks = surf.blocks();
        let batches: Vec<[Block; LANES]> = blocks.as_chunks::<LANES>().0.to_vec();
        #[target_feature(enable = "avx2,fma")]
        fn texels(b: &[Block; LANES]) -> Texels<F> {
            let mut t = Texels {
                x: [[F::splat(255.0); 16]; 4],
            };
            for c in 0..3 {
                for i in 0..16 {
                    let mut l8 = [0f32; LANES];
                    for l in 0..LANES {
                        l8[l] = b[l][i][c] as f32;
                    }
                    t.x[c][i] = F::from_array(l8);
                }
            }
            t
        }
        #[target_feature(enable = "avx2,fma")]
        fn m6(b: &[Block; LANES], iters: u32) -> f32 {
            let t = texels(b);
            encode_subset(&t, &[F::splat(1.0); 16], Spec::M6, iters, false)
                .err
                .to_array()[0]
        }
        #[target_feature(enable = "avx2,fma")]
        fn two(b: &[Block; LANES], spec: Spec) -> f32 {
            let t = texels(b);
            let m = masks2::<F>(&[3, 9, 14, 20, 1, 33, 40, 61]);
            (encode_subset(&t, &m[0], spec, 2, false).err + encode_subset(&t, &m[1], spec, 2, false).err).to_array()[0]
        }
        #[target_feature(enable = "avx2,fma")]
        fn refine6(b: &[Block; LANES]) -> f32 {
            let t = texels(b);
            let all = [F::splat(1.0); 16];
            let s = encode_subset(&t, &all, Spec::M6, 1, false);
            refine(&t, &all, &s, Spec::M6, F::mask_from_bools([true; LANES]), 1)
                .err
                .to_array()[0]
        }
        let time = |name: &str, f: &mut dyn FnMut(&[Block; LANES]) -> f32| {
            let t0 = std::time::Instant::now();
            let mut acc = 0f32;
            for b in &batches {
                acc += f(b);
            }
            println!("{name:<32} {:8.1} ms  (chk {acc})", t0.elapsed().as_secs_f64() * 1e3);
        };
        let w = [1, 1, 1, 1];
        time("encode_batch e0", &mut |b| encode_batch(b, &w, 0u8.into())[0][1] as f32);
        time("encode_batch e1", &mut |b| encode_batch(b, &w, 1u8.into())[0][1] as f32);
        // SAFETY (below): the profile only runs on AVX2 machines.
        time("mode 6, 2 iters", &mut |b| unsafe { m6(b, 2) });
        #[target_feature(enable = "avx2,fma")]
        fn fits(b: &[Block; LANES]) -> f32 {
            let t = texels(b);
            let mut res = [[0f32; LANES]; 64];
            let mut lam = [[0f32; LANES]; 64];
            fits8(&t, &mut res, &mut lam);
            (0..LANES)
                .map(|l| top4(&res, &lam, l, 3)[0].1 as f32 + top4(&res, &lam, l, 2)[0].1 as f32)
                .sum()
        }
        time("fits8 + top4 (8 blocks)", &mut |b| unsafe { fits(b) });
        time("mode 1, one partition", &mut |b| unsafe { two(b, Spec::M1) });
        time("mode 3, one partition", &mut |b| unsafe { two(b, Spec::M3) });
        time("refine mode 6, 1 pass", &mut |b| unsafe { refine6(b) });
    }

    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn show_first_difference() {
        let w = [1, 1, 1, 1];
        let blocks = crate::synth::gradient(256).blocks();
        for chunk in blocks.chunks(LANES) {
            let fast = encode_batch(chunk, &w, 0u8.into());
            for (b, f) in chunk.iter().zip(&fast) {
                let s = super::super::encode(b, &w, 0u8);
                if s != *f {
                    println!("block {:?}", b.map(|t| [t[0], t[1], t[2]]));
                    println!("scalar {:?}", super::super::unpack(&s).unwrap());
                    println!("batch  {:?}", super::super::unpack(f).unwrap());
                    return;
                }
            }
        }
    }

    #[test]
    fn fits8_matches_scalar_line_fit() {
        use crate::fit::{V4, line_fit_error};
        for img in [
            crate::synth::foliage(32),
            crate::synth::roughness(32),
            crate::synth::bricks(32),
        ] {
            for chunk in img.blocks().as_chunks::<LANES>().0 {
                let b: [Block; LANES] = *chunk;
                let mut t = Texels {
                    x: [[f32x8::splat(255.0); 16]; 4],
                };
                for c in 0..3 {
                    for i in 0..16 {
                        t.x[c][i] = f32x8::from_array(std::array::from_fn(|l| b[l][i][c] as f32));
                    }
                }
                let mut res = [[0f32; LANES]; 64];
                let mut lam = [[0f32; LANES]; 64];
                fits8(&t, &mut res, &mut lam);
                for (l, px) in b.iter().enumerate() {
                    for part in 0..64u8 {
                        let mut slow = 0.0;
                        for s in 0..2 {
                            let pts: Vec<V4> = (0..16)
                                .filter(|&i| super::super::subset_of(2, part, i) == s)
                                .map(|i| px[i].map(|v| v as f32))
                                .collect();
                            slow += line_fit_error(&pts, 3);
                        }
                        let fast = res[part as usize][l];
                        // The residual is trace - λmax: when a subset is nearly collinear it is a
                        // small difference of large numbers, so judge the short power iteration by
                        // its λ error relative to the total scatter (residual + λ). Any residual
                        // overestimate only ranks partitions and loosens the pruning bound.
                        let total = slow + lam[part as usize][l];
                        assert!(
                            (fast - slow).abs() <= 1.0 + 3e-2 * total,
                            "lane {l} part {part}: {fast} vs {slow} (scatter {total})"
                        );
                    }
                }
            }
        }
    }

    /// Diagnostic: where the batch and scalar encoders disagree on the synthetic set.
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn compare_with_scalar() {
        let w = [1, 1, 1, 1];
        for effort in [0u8, 1] {
            for (name, generator) in crate::synth::set() {
                let blocks = generator(256).blocks();
                let (mut same, mut diff, mut worse, mut better) = (0, 0, 0u64, 0u64);
                let mut modes = [[0usize; 8]; 2];
                for chunk in blocks.chunks(LANES) {
                    let fast = encode_batch(chunk, &w, effort.into());
                    for (b, f) in chunk.iter().zip(&fast) {
                        let s = super::super::encode(b, &w, effort);
                        let (mut ds, mut df) = ([[0u8; 4]; 16], [[0u8; 4]; 16]);
                        super::super::decode(&s, &mut ds);
                        super::super::decode(f, &mut df);
                        let (es, ef) = (
                            crate::surface::block_sse(b, &ds, &w),
                            crate::surface::block_sse(b, &df, &w),
                        );
                        if s == *f {
                            same += 1;
                        } else {
                            diff += 1;
                            modes[0][super::super::unpack(&s).unwrap().mode as usize] += 1;
                            modes[1][super::super::unpack(f).unwrap().mode as usize] += 1;
                            if ef > es { worse += ef - es } else { better += es - ef }
                        }
                    }
                }
                println!(
                    "e{effort} {name:<22} same {same:5} diff {diff:5}  sse +{worse} -{better}  scalar modes {:?} batch modes {:?}",
                    modes[0], modes[1]
                );
            }
        }
    }
}
