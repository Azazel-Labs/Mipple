//! BC4 batch encoder: eight channels (one block per SIMD lane) per call.
//!
//! Effort 0 ([`proj8`]): min/max endpoints in 8-value mode, indices by projection onto the
//! endpoint line (Intel ISPC's method, with a threshold tuned for the decoder's truncation).
//!
//! Effort 1 ([`fast8`]): effort 0 plus one least-squares refit, kept when it lowers the error.
//!
//! Efforts 2-9 ([`exact8`]): exactly the scalar reference (`super::reference_channel`): min/max
//! 8-value fit, 6-value fit around literal 0/255, least-squares refits, then an exhaustive
//! endpoint window of radius [`RADIUS`] (0 at efforts 2-3, up to 4 at 8-9), with the same palettes and the same tie rule (lowest
//! index wins). Index assignment tests all eight palette entries per texel, so lanes can sit in
//! different modes.
//!
//! Every value is a small integer held exactly in `f32`, and the steps that are not (projection
//! thresholds, least-squares fits) are defined in exact integer arithmetic, so the AVX2 and generic
//! backends and the GPU kernel (`src/gpu/bc4.wgsl`) agree bit for bit.

#[cfg(target_arch = "x86_64")]
use crate::simd::Level;
use crate::simd::{LANES, SimdF32, SimdI32, f32x8, level};
use crate::surface::{Block, Surface};

/// Endpoint search radius of the exact encoder per effort (efforts 0-1 use [`proj8`] / [`fast8`]).
pub(crate) const RADIUS: [i32; 10] = [0, 0, 0, 0, 1, 1, 2, 3, 4, 4];

// Effort 0 rounds projections at 0.45 of a step rather than 0.5 (9/20, as `9 * span` over
// `20 * span` in `proj8`): the decoder truncates palette values, and the lower threshold tracks
// them better (+0.13 dB on Kodak).

/// `floor` for non-negative lanes.
#[inline(always)]
fn floor<V: SimdF32>(v: V) -> V {
    v.floor()
}

/// Lanes where `a > b` (8-value mode).
#[inline(always)]
fn mode8<V: SimdF32>(a: V, b: V) -> V::Mask {
    b.simd_lt(a)
}

/// Channel values of eight blocks, one block per lane.
#[inline(always)]
fn load<V: SimdF32>(v: &[[u8; 16]; LANES]) -> [V; 16] {
    let mut x = [V::splat(0.0); 16];
    let mut i = 0;
    while i < 16 {
        let mut a = [0f32; LANES];
        let mut l = 0;
        while l < LANES {
            a[l] = v[l][i] as f32;
            l += 1;
        }
        x[i] = V::from_array(a);
        i += 1;
    }
    x
}

/// The decoder's palette for endpoints (a, b), per lane (either mode).
#[inline(always)]
fn palette<V: SimdF32>(a: V, b: V) -> [V; 8] {
    let m8 = mode8(a, b);
    let half = V::splat(0.5);
    let mut p = [a, b, a, a, a, a, V::splat(0.0), V::splat(255.0)];
    // Integer division by 7 or 5 is floor((n + 0.5) / d): the remainder plus 0.5 stays below d.
    let (inv7, inv5) = (V::splat(1.0 / 7.0), V::splat(1.0 / 5.0));
    let mut k = 1;
    while k < 7 {
        let kf = k as f32;
        let p8 = floor((V::splat(7.0 - kf) * a + V::splat(kf) * b + half) * inv7);
        p[k + 1] = if k < 5 {
            let p6 = floor((V::splat(5.0 - kf) * a + V::splat(kf) * b + half) * inv5);
            V::select(m8, p8, p6)
        } else {
            V::select(m8, p8, p[k + 1])
        };
        k += 1;
    }
    p
}

/// Nearest palette entry per texel (lowest index on ties) and the summed squared error.
#[inline(always)]
fn assign<V: SimdF32>(x: &[V; 16], p: &[V; 8]) -> ([V; 16], V) {
    let mut idx = [V::splat(0.0); 16];
    let mut err = V::splat(0.0);
    let mut i = 0;
    while i < 16 {
        let d = x[i] - p[0];
        let mut best = d * d;
        let mut bi = V::splat(0.0);
        let mut k = 1;
        while k < 8 {
            let d = x[i] - p[k];
            let e = d * d;
            let m = e.simd_lt(best);
            best = V::select(m, e, best);
            bi = V::select(m, V::splat(k as f32), bi);
            k += 1;
        }
        idx[i] = bi;
        err += best;
        i += 1;
    }
    (idx, err)
}

/// Squared error only.
#[inline(always)]
fn error<V: SimdF32>(x: &[V; 16], p: &[V; 8]) -> V {
    let mut err = V::splat(0.0);
    let mut i = 0;
    while i < 16 {
        let d = x[i] - p[0];
        let mut best = d * d;
        let mut k = 1;
        while k < 8 {
            let d = x[i] - p[k];
            best = best.simd_min(d * d);
            k += 1;
        }
        err += best;
        i += 1;
    }
    err
}

/// Step toward e1 (0..=7) of each 8-value-mode palette index: 0 -> 0, 1 -> 7, k >= 2 -> k - 1.
#[inline(always)]
fn step8<V: SimdF32>(k: V) -> V {
    V::select(
        k.simd_lt(V::splat(0.5)),
        V::splat(0.0),
        V::select(k.simd_lt(V::splat(1.5)), V::splat(7.0), k - V::splat(1.0)),
    )
}

/// Least-squares endpoints per lane from each texel's step (0..=7), by [`super::ls_steps`]
/// (exact integers); both 0 where the fit is degenerate. The sums are exact in `f32`.
#[inline(always)]
fn ls8<V: SimdF32>(x: &[V; 16], steps: &[V; 16]) -> (V, V) {
    let zero = V::splat(0.0);
    let (mut j, mut jj, mut jx, mut sx) = (zero, zero, zero, zero);
    let mut i = 0;
    while i < 16 {
        j += steps[i];
        jj += steps[i] * steps[i];
        jx += steps[i] * x[i];
        sx += x[i];
        i += 1;
    }
    let (j, jj, jx, sx) = (j.to_array(), jj.to_array(), jx.to_array(), sx.to_array());
    let (mut a, mut b) = ([0f32; LANES], [0f32; LANES]);
    let mut l = 0;
    while l < LANES {
        if let Some((p, q)) = super::ls_steps(j[l] as i32, jj[l] as i32, jx[l] as i32, sx[l] as i32) {
            a[l] = p as f32;
            b[l] = q as f32;
        }
        l += 1;
    }
    (V::from_array(a), V::from_array(b))
}

/// 8-value-mode indices by projection onto the line from `a` (step 0) to `b` (step 7), and the
/// squared error against the decoder's palette. Returns palette indices, the error and each
/// texel's step (0..=7).
#[inline(always)]
fn project<V: SimdF32>(x: &[V; 16], a: V, b: V) -> ([V; 16], V, [V; 16]) {
    let zero = V::splat(0.0);
    let span = a - b;
    // Step floor(clamp((a - x) * 7 / span + 0.5, 0, 7)), exactly: (14 (a - x) + span) / (2 span)
    // in integers, clamped. The numerator is an exact integer and + 0.5 keeps the quotient at
    // least 0.5 / (2 span) from any integer, far beyond the reciprocal's rounding error.
    let inv = V::splat(1.0) / (span + span).simd_max(V::splat(1.0));
    let pos = zero.simd_lt(span);
    let step = b - a;
    let a7 = a * V::splat(7.0) + V::splat(0.5);
    let inv7 = V::splat(1.0 / 7.0);
    let mut idx = [zero; 16];
    let mut steps = [zero; 16];
    let mut err = zero;
    let mut i = 0;
    while i < 16 {
        let n = (V::splat(14.0) * (a - x[i]) + span).simd_max(zero);
        let j = V::select(pos, floor((n + V::splat(0.5)) * inv).simd_min(V::splat(7.0)), zero);
        steps[i] = j;
        // Palette value of step j: floor((7a + j(b - a)) / 7), as in `palette`.
        let d = x[i] - floor((a7 + j * step) * inv7);
        err += d * d;
        // Step 0 is index 0 (a), step 7 is index 1 (b), steps 1-6 are indices 2-7.
        let mid = j + V::splat(1.0);
        idx[i] = V::select(
            j.simd_lt(V::splat(0.5)),
            zero,
            V::select(V::splat(6.5).simd_lt(j), V::splat(1.0), mid),
        );
        i += 1;
    }
    (idx, err, steps)
}

/// Palette index of each projection step: step 0 is e0 (index 0), step 7 is e1 (index 1).
const STEP_INDEX: [u8; 16] = [0, 2, 3, 4, 5, 6, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0];

/// Effort 0: (e0, e1, index bits of texels 0-7, of texels 8-15). With min/max endpoints the
/// projection already lies in `0.45 ..= 7.45`, so it needs no clamp.
#[inline(always)]
fn proj8<V: SimdF32>(x: &[V; 16]) -> (V, V, V::Int, V::Int) {
    let (mut lo, mut hi) = (x[0], x[0]);
    let mut i = 1;
    while i < 16 {
        lo = lo.simd_min(x[i]);
        hi = hi.simd_max(x[i]);
        i += 1;
    }
    let span = hi - lo;
    // Step floor((hi - x) * 7 / span + 0.45), exactly: (140 (hi - x) + 9 span) / (20 span) in
    // integers. The numerator (+ 0.5) is exact in f32 and stays at least 0.5 / (20 span) from a
    // multiple of the denominator, far beyond the reciprocal's rounding error.
    let inv = V::splat(1.0) / (V::splat(20.0) * span).simd_max(V::splat(1.0));
    let c = V::splat(140.0) * hi + V::splat(9.0) * span + V::splat(0.5);
    let (mut b0, mut b1) = (V::Int::splat(0), V::Int::splat(0));
    let mut i = 0;
    while i < 8 {
        let j0 = (x[i].mul_add(V::splat(-140.0), c) * inv).to_int_trunc();
        let j1 = (x[i + 8].mul_add(V::splat(-140.0), c) * inv).to_int_trunc();
        b0 = b0 | (V::Int::lookup16(&STEP_INDEX, j0) << (3 * i as i32));
        b1 = b1 | (V::Int::lookup16(&STEP_INDEX, j1) << (3 * i as i32));
        i += 1;
    }
    (hi, lo, b0, b1)
}

/// Effort 1: returns (e0, e1, palette indices) per lane.
#[inline(always)]
fn fast8<V: SimdF32>(x: &[V; 16]) -> (V, V, [V; 16]) {
    let zero = V::splat(0.0);
    let (mut lo, mut hi) = (x[0], x[0]);
    let mut i = 1;
    while i < 16 {
        lo = lo.simd_min(x[i]);
        hi = hi.simd_max(x[i]);
        i += 1;
    }
    let (idx, err, steps) = project(x, hi, lo);
    // Least-squares refit from the steps (weight toward e1 = step / 7); 0, 0 when degenerate.
    let (a, b) = ls8(x, &steps);
    let valid = b.simd_lt(a) & zero.simd_lt(err);
    if !V::mask_any(valid) {
        return (hi, lo, idx);
    }
    let (idx2, err2, _) = project(x, a, b);
    let take = valid & err2.simd_lt(err);
    let mut out = idx;
    let mut i = 0;
    while i < 16 {
        out[i] = V::select(take, idx2[i], idx[i]);
        i += 1;
    }
    (V::select(take, a, hi), V::select(take, b, lo), out)
}

/// Endpoints, indices and error of the current best per lane.
struct Best<V: SimdF32> {
    a: V,
    b: V,
    idx: [V; 16],
    err: V,
}

impl<V: SimdF32> Best<V> {
    #[inline(always)]
    fn take(&mut self, m: V::Mask, a: V, b: V, idx: &[V; 16], err: V) {
        self.a = V::select(m, a, self.a);
        self.b = V::select(m, b, self.b);
        self.err = V::select(m, err, self.err);
        let mut i = 0;
        while i < 16 {
            self.idx[i] = V::select(m, idx[i], self.idx[i]);
            i += 1;
        }
    }
}

/// Efforts 1-3 (endpoint window radius `r`): returns (e0, e1, palette indices) per lane.
#[inline(always)]
fn exact8<V: SimdF32>(x: &[V; 16], r: i32) -> (V, V, [V; 16]) {
    let (zero, full) = (V::splat(0.0), V::splat(255.0));
    let (mut lo, mut hi) = (x[0], x[0]);
    // Inner (not 0 or 255) range, for the 6-value candidate.
    let (mut ilo, mut ihi) = (full, zero);
    let mut extreme = x[0].simd_lt(V::splat(0.5)) | V::splat(254.5).simd_lt(x[0]);
    let mut i = 0;
    while i < 16 {
        lo = lo.simd_min(x[i]);
        hi = hi.simd_max(x[i]);
        let inner = V::splat(0.5).simd_lt(x[i]) & x[i].simd_lt(V::splat(254.5));
        ilo = V::select(inner, x[i], full).simd_min(ilo);
        ihi = V::select(inner, x[i], zero).simd_max(ihi);
        extreme = extreme | x[i].simd_lt(V::splat(0.5)) | V::splat(254.5).simd_lt(x[i]);
        i += 1;
    }

    // Min/max in 8-value mode (or both endpoints equal for a constant channel).
    let (idx, err) = assign(x, &palette(hi, lo));
    let mut best = Best { a: hi, b: lo, idx, err };

    // 6-value mode keeps exact 0 and 255 for free; fit the inner range.
    if V::mask_any(extreme) {
        let none = ihi.simd_lt(ilo);
        let (a, b) = (V::select(none, zero, ilo), V::select(none, zero, ihi));
        let (idx, err) = assign(x, &palette(a, b));
        best.take(extreme & err.simd_lt(best.err), a, b, &idx, err);
    }

    // Least-squares refits of 8-value-mode lanes from their indices (twice, as the reference).
    let mut it = 0;
    while it < 2 {
        let mut steps = [zero; 16];
        let mut i = 0;
        while i < 16 {
            steps[i] = step8(best.idx[i]);
            i += 1;
        }
        let (a, b) = ls8(x, &steps);
        let valid = mode8(best.a, best.b) & mode8(a, b) & zero.simd_lt(best.err);
        if !V::mask_any(valid) {
            break;
        }
        let (idx, err) = assign(x, &palette(a, b));
        best.take(valid & err.simd_lt(best.err), a, b, &idx, err);
        it += 1;
    }

    // Exhaustive window around the endpoints, staying in each lane's mode.
    let busy = zero.simd_lt(best.err);
    if r > 0 && V::mask_any(busy) {
        let (b0, b1) = (best.a, best.b);
        let m8 = mode8(b0, b1);
        let m6 = b0.simd_lt(b1 + V::splat(0.5));
        let (mut ba, mut bb, mut be) = (b0, b1, best.err);
        let mut d0 = -r;
        while d0 <= r {
            let a = b0 + V::splat(d0 as f32);
            let mut d1 = -r;
            while d1 <= r {
                let b = b1 + V::splat(d1 as f32);
                let in_range = V::splat(-0.5).simd_lt(a)
                    & a.simd_lt(V::splat(255.5))
                    & V::splat(-0.5).simd_lt(b)
                    & b.simd_lt(V::splat(255.5));
                // Same mode as the start: (a > b) == (b0 > b1).
                let same = (m8 & mode8(a, b)) | (m6 & a.simd_lt(b + V::splat(0.5)));
                let ok = busy & in_range & same;
                if V::mask_any(ok) {
                    // Out-of-range lanes are masked out; clamp so the palette stays finite.
                    let p = palette(a.simd_clamp(zero, full), b.simd_clamp(zero, full));
                    let err = error(x, &p);
                    let m = ok & err.simd_lt(be);
                    ba = V::select(m, a, ba);
                    bb = V::select(m, b, bb);
                    be = V::select(m, err, be);
                }
                d1 += 1;
            }
            d0 += 1;
        }
        let (idx, _) = assign(x, &palette(ba, bb));
        return (ba, bb, idx);
    }
    (best.a, best.b, best.idx)
}

/// Index bits of palette indices: three bits per texel, texels 0-7 and 8-15 in 24 bits each.
#[inline(always)]
fn bits<V: SimdF32>(idx: &[V; 16]) -> (V::Int, V::Int) {
    let (mut lo, mut hi) = (V::Int::splat(0), V::Int::splat(0));
    let mut i = 0;
    while i < 8 {
        lo = lo | (idx[i].to_int_trunc() << (3 * i as i32));
        hi = hi | (idx[i + 8].to_int_trunc() << (3 * i as i32));
        i += 1;
    }
    (lo, hi)
}

/// Inverse of [`crate::simd::ROW_ORDER`]: lane n of the result is block n.
const UNROW: [i32; LANES] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Eight BC4 blocks (64 bytes, block order) from endpoints and index bits. `row_order`: lanes
/// are in [`crate::simd::ROW_ORDER`] rather than block order.
#[inline(always)]
fn store<V: SimdF32>(a: V, b: V, lo: V::Int, hi: V::Int, row_order: bool) -> [u8; 64] {
    // Block bytes: e0, e1, then 48 index bits little-endian: word 0 = e0 | e1 << 8 | bits 0-15,
    // word 1 = bits 16-47.
    let mut w0 = a.to_int_trunc() | (b.to_int_trunc() << 8) | (lo << 16);
    let mut w1 = (lo >> 16) | (hi << 8);
    if row_order {
        w0 = w0.permute(UNROW);
        w1 = w1.permute(UNROW);
    }
    let (p, q) = w0.interleave(w1);
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&p.to_le_bytes());
    out[32..].copy_from_slice(&q.to_le_bytes());
    out
}

/// Encodes eight channels (one per lane) at the given effort into 64 bytes in block order.
#[inline(always)]
fn encode_x<V: SimdF32>(x: &[V; 16], effort: u8, row_order: bool) -> [u8; 64] {
    let (a, b, lo, hi) = match effort {
        0 => proj8(x),
        _ => {
            let (a, b, idx) = if effort == 1 {
                fast8(x)
            } else {
                exact8(x, RADIUS[effort.min(9) as usize])
            };
            let (lo, hi) = bits(&idx);
            (a, b, lo, hi)
        }
    };
    store(a, b, lo, hi, row_order)
}

/// Encodes eight channels at the given effort.
#[inline(always)]
fn encode8<V: SimdF32>(v: &[[u8; 16]; LANES], effort: u8) -> [[u8; 8]; LANES] {
    let raw = encode_x(&load::<V>(v), effort, false);
    let mut out = [[0u8; 8]; LANES];
    for (o, r) in out.iter_mut().zip(raw.as_chunks::<8>().0) {
        o.copy_from_slice(r);
    }
    out
}

/// Encodes the blocks `first..` of a surface into `out` (`8 * channels` bytes per block). Runs of
/// eight blocks inside one block row load texels with [`SimdI32::load_block_row`]; the rest
/// gather per block.
#[inline(always)]
fn encode_span<V: SimdF32>(s: &Surface, first: usize, out: &mut [u8], (ch0, channels): (usize, usize), effort: u8) {
    let bw = s.blocks_x() as usize;
    let (w, h) = (s.width as usize, s.height as usize);
    let bb = 8 * channels;
    let mut k = first;
    for o in out.chunks_mut(LANES * bb) {
        let n = o.len() / bb;
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
            let mut c = 0;
            while c < channels {
                let mut x = [V::splat(0.0); 16];
                let mut i = 0;
                while i < 16 {
                    let v = if ch0 + c == 0 {
                        px[i]
                    } else {
                        px[i] >> (8 * (ch0 + c) as i32)
                    };
                    x[i] = (v & V::Int::splat(0xff)).to_float();
                    i += 1;
                }
                let enc = encode_x(&x, effort, true);
                put(o, &enc, c, channels);
                c += 1;
            }
        } else {
            for c in 0..channels {
                let mut v = [[0u8; 16]; LANES];
                for (l, vl) in v.iter_mut().enumerate().take(n) {
                    let i = (k + l) as u32;
                    *vl = s.read_block_channel(i % bw as u32, i / bw as u32, ch0 + c);
                }
                let enc = encode_x(&load::<V>(&v), effort, false);
                put(o, &enc, c, channels);
            }
        }
        k += n;
    }
}

/// Writes channel `c` of up to eight encoded blocks (`enc`, 8 bytes each) into `o`.
#[inline(always)]
fn put(o: &mut [u8], enc: &[u8; 64], c: usize, channels: usize) {
    if channels == 1 {
        let n = o.len();
        o.copy_from_slice(&enc[..n]);
    } else {
        for (ob, e) in o.chunks_exact_mut(8 * channels).zip(enc.as_chunks::<8>().0) {
            ob[8 * c..8 * c + 8].copy_from_slice(e);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode_span_avx2(s: &Surface, first: usize, out: &mut [u8], chans: (usize, usize), effort: u8) {
    encode_span::<crate::simd::avx2::F>(s, first, out, chans, effort);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode8_avx2(v: &[[u8; 16]; LANES], effort: u8) -> [[u8; 8]; LANES] {
    encode8::<crate::simd::avx2::F>(v, effort)
}

/// Encodes eight channels with the best instruction set available.
pub fn encode_channels8(v: &[[u8; 16]; LANES], effort: u8) -> [[u8; 8]; LANES] {
    match level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { encode8_avx2(v, effort) },
        _ => encode8::<f32x8>(v, effort),
    }
}

/// Encodes up to eight BC4 (`channels == 1`) or BC5 (`channels == 2`) blocks.
pub fn encode_batch(blocks: &[Block], channels: usize, effort: u8) -> Vec<[u8; 16]> {
    debug_assert!(blocks.len() <= LANES && (1..=2).contains(&channels));
    let mut out = vec![[0u8; 16]; blocks.len()];
    for c in 0..channels {
        let mut v = [[0u8; 16]; LANES];
        for (l, b) in blocks.iter().enumerate() {
            v[l] = super::channel(b, c);
        }
        let enc = encode_channels8(&v, effort);
        for (o, e) in out.iter_mut().zip(&enc) {
            o[8 * c..8 * c + 8].copy_from_slice(e);
        }
    }
    out
}

/// Encodes a whole surface as BC4 (`channels == 1`) or BC5 (`channels == 2`), reading texels
/// straight from the surface (no intermediate blocks). Returns the raw blocks in raster order
/// (8 or 16 bytes each).
pub fn encode_surface(surface: &Surface, channels: usize, effort: u8) -> Vec<u8> {
    encode_channels(surface, (0, channels), effort)
}

/// Encodes channel `c` of a surface as BC4 blocks (8 bytes each, raster order): BC3's alpha half
/// for `c == 3`.
pub fn encode_surface_channel(surface: &Surface, c: usize, effort: u8) -> Vec<u8> {
    encode_channels(surface, (c, 1), effort)
}

/// Encodes channels `ch0..ch0 + channels` (one BC4 block each, in order) of every block.
fn encode_channels(surface: &Surface, (ch0, channels): (usize, usize), effort: u8) -> Vec<u8> {
    use rayon::prelude::*;
    // Spans of whole SIMD batches; 512 blocks keeps rayon's per-task overhead negligible.
    const SPAN: usize = 512;
    let bb = 8 * channels;
    let mut out = vec![0u8; surface.block_count() * bb];
    let lvl = level();
    out.par_chunks_mut(SPAN * bb).enumerate().for_each(|(ci, o)| match lvl {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { encode_span_avx2(surface, ci * SPAN, o, (ch0, channels), effort) },
        _ => encode_span::<f32x8>(surface, ci * SPAN, o, (ch0, channels), effort),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<[u8; 16]> {
        let mut out = Vec::new();
        let mut s = 0x1234_5678u32;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for n in 0..4000 {
            let base = (rnd() % 256) as i32;
            let spread = [0, 1, 3, 8, 30, 120, 255][n % 7];
            let mut v = [0u8; 16];
            for t in &mut v {
                let x = base + (rnd() % (2 * spread as u32 + 1)) as i32 - spread;
                *t = x.clamp(0, 255) as u8;
            }
            if n % 11 == 0 {
                v[(rnd() % 16) as usize] = 0;
            }
            if n % 13 == 0 {
                v[(rnd() % 16) as usize] = 255;
            }
            out.push(v);
        }
        out
    }

    fn sse(v: &[u8; 16], raw: &[u8]) -> u64 {
        let d = super::super::decode_channel(raw);
        v.iter().zip(&d).map(|(a, b)| (a.abs_diff(*b) as u64).pow(2)).sum()
    }

    #[test]
    fn fast_backends_agree_and_beat_min_max() {
        let c = corpus();
        let (mut fast_err, mut minmax_err) = (0u64, 0u64);
        for chunk in c.as_chunks::<LANES>().0 {
            let v: [[u8; 16]; LANES] = *chunk;
            let fast = encode_channels8(&v, 1);
            assert_eq!(fast, encode8::<f32x8>(&v, 1));
            assert_eq!(encode_channels8(&v, 0), encode8::<f32x8>(&v, 0));
            for l in 0..LANES {
                let (lo, hi) = (*v[l].iter().min().unwrap(), *v[l].iter().max().unwrap());
                fast_err += sse(&v[l], &fast[l]);
                minmax_err += sse(&v[l], &super::super::finalize(hi, lo, &v[l]).0);
            }
        }
        assert!(fast_err < minmax_err, "fast {fast_err} vs min/max {minmax_err}");
    }

    #[test]
    fn surface_path_matches_block_path() {
        // 70 x 66: block rows of 18 (runs of eight cross rows), ragged right and bottom edges.
        let mut s = crate::surface::Surface::new(70, 66);
        for (i, p) in s.pixels.iter_mut().enumerate() {
            *p = [(i * 13 % 256) as u8, (i * 37 % 251) as u8, 0, 255];
        }
        for channels in 1..=2 {
            for effort in [0, 1, 2, 5] {
                let a = encode_surface(&s, channels, effort);
                let b: Vec<u8> = s
                    .blocks()
                    .chunks(LANES)
                    .flat_map(|c| encode_batch(c, channels, effort))
                    .flat_map(|b| b[..8 * channels].to_vec())
                    .collect();
                assert_eq!(a, b, "channels {channels} effort {effort}");
            }
        }
    }

    #[test]
    fn matches_scalar_reference() {
        let c = corpus();
        for effort in [2u8, 4, 6, 7, 8] {
            for chunk in c.as_chunks::<LANES>().0 {
                let v: [[u8; 16]; LANES] = *chunk;
                let fast = encode_channels8(&v, effort);
                let generic = encode8::<f32x8>(&v, effort);
                for l in 0..LANES {
                    let s = super::super::reference_channel(&v[l], RADIUS[effort as usize]);
                    assert_eq!(fast[l], s, "effort {effort} block {:?}", v[l]);
                    assert_eq!(generic[l], s, "generic, effort {effort} block {:?}", v[l]);
                }
            }
        }
    }
}

// Hand-run reports; the timing one calls the AVX2 kernel directly.
#[cfg(all(test, target_arch = "x86_64"))]
mod profile {
    use super::*;

    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn profile_bc4_batch() {
        let s = crate::surface::Surface::load(std::path::Path::new("test-images/kodak/kodim01.png")).unwrap();
        let t = std::time::Instant::now();
        let blocks = s.blocks();
        println!(
            "blocks(): {:.2} ms ({} blocks)",
            t.elapsed().as_secs_f64() * 1e3,
            blocks.len()
        );
        let chans: Vec<[[u8; 16]; LANES]> = blocks
            .as_chunks::<LANES>()
            .0
            .iter()
            .map(|c| std::array::from_fn(|l| super::super::channel(&c[l], 0)))
            .collect();
        for rep in 0..20 {
            let t = std::time::Instant::now();
            let mut sum = 0u32;
            for v in &chans {
                sum += encode_channels8(v, 0)[0][2] as u32;
            }
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let e = encode_surface(&s, 1, 0);
            let t2 = t.elapsed().as_secs_f64() * 1e3;
            let mut out = vec![0u8; s.block_count() * 8];
            let t = std::time::Instant::now();
            // SAFETY: the profile only runs on AVX2 machines.
            unsafe { encode_span_avx2(&s, 0, &mut out, (0, 1), 0) };
            println!(
                "rep {rep}: kernel e0 {t1:.3} ms, encode_surface e0 {t2:.3} ms, one span {:.3} ms ({sum} {} {})",
                t.elapsed().as_secs_f64() * 1e3,
                e[53],
                out[53]
            );
        }
        for effort in 0..=9u8 {
            let t = std::time::Instant::now();
            let mut sum = 0u32;
            for c in blocks.chunks(LANES) {
                sum += encode_batch(c, 1, effort)[0][2] as u32;
            }
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            for v in &chans {
                sum += encode_channels8(v, effort)[0][2] as u32;
            }
            println!(
                "effort {effort}: {t1:.2} ms, kernel only {:.2} ms ({sum})",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

#[cfg(test)]
mod variants {
    use super::super::{decode_channel, finalize, palette as pal_scalar};

    fn psnr(sse: u64, n: usize) -> f64 {
        10.0 * (255.0f64 * 255.0 * n as f64 / sse as f64).log10()
    }

    /// Projection onto the line hi -> lo with a threshold offset `bias` (in steps).
    fn project(v: &[u8; 16], hi: u8, lo: u8, bias: f32) -> [u8; 8] {
        let mut bits = 0u64;
        let scale = if hi > lo { 7.0 / (hi - lo) as f32 } else { 0.0 };
        for (i, &x) in v.iter().enumerate() {
            let j = (((hi as f32 - x as f32) * scale + 0.5 + bias).floor()).clamp(0.0, 7.0) as u64;
            let code = match j {
                0 => 0,
                7 => 1,
                j => j + 1,
            };
            bits |= code << (3 * i);
        }
        let mut b = [0u8; 8];
        b[0] = hi;
        b[1] = lo;
        b[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
        b
    }

    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn bc4_variant_quality() {
        let mut chans = Vec::new();
        for n in ["kodim01", "kodim02", "kodim03"] {
            let s = crate::surface::Surface::load(std::path::Path::new(&format!("test-images/kodak/{n}.png"))).unwrap();
            for b in s.blocks() {
                chans.push(super::super::channel(&b, 0));
            }
        }
        let n = chans.len() * 16;
        let sse = |f: &dyn Fn(&[u8; 16]) -> [u8; 8]| -> u64 {
            chans
                .iter()
                .map(|v| {
                    let d = decode_channel(&f(v));
                    v.iter()
                        .zip(&d)
                        .map(|(a, b)| (a.abs_diff(*b) as u64).pow(2))
                        .sum::<u64>()
                })
                .sum()
        };
        let mm = |v: &[u8; 16]| (*v.iter().max().unwrap(), *v.iter().min().unwrap());
        for bias in [-0.05f32, -0.1, -0.15, -0.2, -0.3] {
            let e = sse(&|v| {
                let (h, l) = mm(v);
                project(v, h, l, bias)
            });
            println!("project bias {bias}: {:.3} dB", psnr(e, n));
        }
        // Projection, then one least-squares refit kept unconditionally, then projection again.
        for bias in [0.0f32, -0.1] {
            let e = sse(&|v| {
                let (h, l) = mm(v);
                let b0 = project(v, h, l, bias);
                if h == l {
                    return b0;
                }
                let bits = u64::from_le_bytes([b0[2], b0[3], b0[4], b0[5], b0[6], b0[7], 0, 0]);
                let (mut sw, mut sww, mut swx, mut sx) = (0f32, 0f32, 0f32, 0f32);
                for (i, &x) in v.iter().enumerate() {
                    let k = (bits >> (3 * i)) & 7;
                    let w = match k {
                        0 => 0.0,
                        1 => 1.0,
                        k => (k - 1) as f32 / 7.0,
                    };
                    sw += w;
                    sww += w * w;
                    swx += w * x as f32;
                    sx += x as f32;
                }
                let s00 = 16.0 - 2.0 * sw + sww;
                let s01 = sw - sww;
                let det = s00 * sww - s01 * s01;
                if det < 1e-3 {
                    return b0;
                }
                let a = ((sww * (sx - swx) - s01 * swx) / det).clamp(0.0, 255.0).round() as u8;
                let b = ((s00 * swx - s01 * (sx - swx)) / det).clamp(0.0, 255.0).round() as u8;
                if a <= b {
                    return b0;
                }
                project(v, a, b, bias)
            });
            println!("project + refit (always) bias {bias}: {:.3} dB", psnr(e, n));
        }
        let e = sse(&|v| {
            let (h, l) = mm(v);
            finalize(h, l, v).0
        });
        println!("min/max + exact nearest: {:.3} dB", psnr(e, n));
        let _ = pal_scalar(1, 0);
        for effort in 0..=3u8 {
            let e = sse(&|v| super::super::encode_channel(v, effort));
            println!("effort {effort}: {:.3} dB", psnr(e, n));
        }
    }
}
