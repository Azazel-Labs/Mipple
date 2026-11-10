//! Learned ranker for BC7's localized search: which earlier blocks of the chunk, beyond the
//! exhaustive near search, get the full candidate search (whole-block copy, endpoints re-indexed,
//! indices with refit endpoints). Like the BC5 ranker ([`super::bc5`]) it only proposes
//! blocks; the RD search still picks the encoding from their candidates, so a bad ranking costs
//! compression, never correctness.
//!
//! Weights are fitted by `scripts/train_ranker_bc7.py` on CC0 Poly Haven textures
//! (`scripts/fetch_polyhaven.py`), never on the Kodak or Oodle benchmark images. The features are
//! computed here, once, for both the encoder and the training dump (`MIPPLE_LOCAL_DUMP`), so the
//! trainer reads them as they are.

use crate::surface::Block;

/// Pair features (vary with the earlier block).
pub const NPAIR: usize = 14;
/// Context features (fixed for the source block; folded into the first layer's bias).
pub const NCTX: usize = 5;
pub const NIN: usize = NPAIR + NCTX;
pub const HIDDEN: usize = 16;

/// What the ranker keeps of each final (decoded) block, and of the source block.
#[derive(Clone, Copy)]
pub struct RankBlock7 {
    /// Decoded values, channel-major (16 per channel, RGBA).
    vals: [u8; 64],
    /// Values scaled to 0..=63 over the block's own range per channel (the index pattern).
    shape: [u8; 64],
    raw: [u8; 16],
    sum: [u32; 4],
    range: [u32; 4],
    /// Sum of `shape` per channel.
    q1: [f32; 4],
    /// 1 / (centered sum of squares of `shape`), or 0 for a flat channel.
    inv_vq: [f32; 4],
    mode: u8,
    partition: u8,
    ns: u8,
}

/// The source block's side: its statistics, the base encoding's fields and the MLP's hidden
/// pre-activations from its bias and the block context alone.
pub struct RankSrc7 {
    b: RankBlock7,
    s1: [f32; 4],
    vs: [f32; 4],
    h0: [f32; HIDDEN],
    pub ctx: [f32; NCTX],
}

/// log2(d) / 12 for the reachable distances (64 KiB BC7 chunks hold 4096 blocks).
static LOG2_D: std::sync::LazyLock<Vec<f32>> =
    std::sync::LazyLock::new(|| (0..8192).map(|d| (d.max(1) as f32).log2() / 12.0).collect());

/// log2(1 + x) by the float bit trick (exponent plus linear mantissa).
#[inline]
fn fl2(x: f32) -> f32 {
    let v = 1.0f32 + x;
    (v.to_bits() as i32 - 0x3F80_0000) as f32 / (1u32 << 23) as f32
}

/// Mode, partition and subset count of a raw BC7 block (mode 8, i.e. reserved, for none).
fn head(raw: &[u8; 16]) -> (u8, u8, u8) {
    let mode = raw[0].trailing_zeros().min(8) as u8;
    let bits = u16::from_le_bytes([raw[0], raw[1]]) >> (mode + 1);
    match mode {
        0 => (0, (bits & 15) as u8, 3),
        1 | 3 | 7 => (mode, (bits & 63) as u8, 2),
        2 => (2, (bits & 63) as u8, 3),
        4..=6 => (mode, 0, 1),
        _ => (8, 0, 0),
    }
}

impl RankBlock7 {
    /// `b`: the block's texels (decoded, or the source); `raw`: its BC7 encoding (the base
    /// encoding for the source).
    pub fn new(b: &Block, raw: &[u8; 16]) -> Self {
        let (mode, partition, ns) = head(raw);
        let mut r = RankBlock7 {
            vals: [0; 64],
            shape: [0; 64],
            raw: *raw,
            sum: [0; 4],
            range: [0; 4],
            q1: [0.0; 4],
            inv_vq: [0.0; 4],
            mode,
            partition,
            ns,
        };
        for c in 0..4 {
            let (mut lo, mut hi, mut sum) = (255u32, 0u32, 0u32);
            for t in 0..16 {
                let v = b[t][c] as u32;
                r.vals[c * 16 + t] = v as u8;
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v;
            }
            r.sum[c] = sum;
            r.range[c] = hi - lo;
            let (mut q1, mut q2) = (0u32, 0u32);
            if hi > lo {
                for t in 0..16 {
                    let q = (b[t][c] as u32 - lo) * 63 / (hi - lo);
                    r.shape[c * 16 + t] = q as u8;
                    q1 += q;
                    q2 += q * q;
                }
            }
            r.q1[c] = q1 as f32;
            // Exact in integers: 16·Σq² − (Σq)², then / 16.
            let vq16 = 16 * q2 - q1 * q1;
            r.inv_vq[c] = if vq16 > 0 { 16.0 / vq16 as f32 } else { 0.0 };
        }
        r
    }
}

/// Sum of absolute differences of two 64-byte arrays.
#[inline]
fn sad64(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; every load is an unaligned 16-byte load within a
        // 64-byte array.
        unsafe {
            let mut acc = _mm_setzero_si128();
            for k in 0..4 {
                let x = _mm_loadu_si128(a.as_ptr().add(k * 16).cast::<__m128i>());
                let y = _mm_loadu_si128(b.as_ptr().add(k * 16).cast::<__m128i>());
                acc = _mm_add_epi64(acc, _mm_sad_epu8(x, y));
            }
            _mm_cvtsi128_si64(_mm_add_epi64(acc, _mm_unpackhi_epi64(acc, acc))) as u32
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter().zip(b).map(|(x, y)| u32::from(x.abs_diff(*y))).sum()
    }
}

/// Per channel: Σ source value · earlier shape, and value SSE. Integer sums, so exact on every
/// target.
#[inline]
fn cross(s: &RankBlock7, p: &RankBlock7) -> ([u32; 4], [u32; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; every load is an unaligned 16-byte load within a
        // 64-byte array.
        unsafe {
            let ld = |a: &[u8; 64], k: usize| _mm_loadu_si128(a.as_ptr().add(k * 16).cast::<__m128i>());
            let hsum32 = |v: __m128i| {
                let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
                let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
                _mm_cvtsi128_si32(v) as u32
            };
            let z = _mm_setzero_si128();
            let mut out = ([0u32; 4], [0u32; 4]);
            for k in 0..4 {
                let (sv, ps, pv) = (ld(&s.vals, k), ld(&p.shape, k), ld(&p.vals, k));
                let (svl, svh) = (_mm_unpacklo_epi8(sv, z), _mm_unpackhi_epi8(sv, z));
                let (psl, psh) = (_mm_unpacklo_epi8(ps, z), _mm_unpackhi_epi8(ps, z));
                let (pvl, pvh) = (_mm_unpacklo_epi8(pv, z), _mm_unpackhi_epi8(pv, z));
                out.0[k] = hsum32(_mm_add_epi32(_mm_madd_epi16(svl, psl), _mm_madd_epi16(svh, psh)));
                let (dl, dh) = (_mm_sub_epi16(svl, pvl), _mm_sub_epi16(svh, pvh));
                out.1[k] = hsum32(_mm_add_epi32(_mm_madd_epi16(dl, dl), _mm_madd_epi16(dh, dh)));
            }
            out
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = ([0u32; 4], [0u32; 4]);
        for k in 0..4 {
            for t in k * 16..k * 16 + 16 {
                out.0[k] += s.vals[t] as u32 * p.shape[t] as u32;
                let e = s.vals[t] as i32 - p.vals[t] as i32;
                out.1[k] += (e * e) as u32;
            }
        }
        out
    }
}

/// Per channel: Σ source value · earlier shape (`cross`'s first half, for the prefilter).
#[inline]
fn cross_sq(s: &RankBlock7, p: &RankBlock7) -> [u32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; every load is an unaligned 16-byte load within a
        // 64-byte array, the store is to a 4-u32 array.
        unsafe {
            let ld = |a: &[u8; 64], k: usize| _mm_loadu_si128(a.as_ptr().add(k * 16).cast::<__m128i>());
            let z = _mm_setzero_si128();
            let mut a = [z; 4];
            for (k, ak) in a.iter_mut().enumerate() {
                let (sv, ps) = (ld(&s.vals, k), ld(&p.shape, k));
                let lo = _mm_madd_epi16(_mm_unpacklo_epi8(sv, z), _mm_unpacklo_epi8(ps, z));
                let hi = _mm_madd_epi16(_mm_unpackhi_epi8(sv, z), _mm_unpackhi_epi8(ps, z));
                *ak = _mm_add_epi32(lo, hi);
            }
            // Transpose-add the four channels' partial sums into one vector.
            let s01 = _mm_add_epi32(_mm_unpacklo_epi32(a[0], a[1]), _mm_unpackhi_epi32(a[0], a[1]));
            let s23 = _mm_add_epi32(_mm_unpacklo_epi32(a[2], a[3]), _mm_unpackhi_epi32(a[2], a[3]));
            let v = _mm_add_epi32(_mm_unpacklo_epi64(s01, s23), _mm_unpackhi_epi64(s01, s23));
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), v);
            out
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::array::from_fn(|k| {
            (k * 16..k * 16 + 16)
                .map(|t| s.vals[t] as u32 * p.shape[t] as u32)
                .sum()
        })
    }
}

/// [`RankSrc7::prefilter_top`] with the key function `key`. Each list collects pairs below its
/// current `m`-th smallest, and is cut back to `m` whenever it holds `4m`.
#[inline(always)]
fn top_with(
    blocks: &[RankBlock7],
    i: usize,
    ds: std::ops::RangeInclusive<usize>,
    m: usize,
    pre: &mut Vec<(u32, usize)>,
    res: &mut Vec<(u32, usize)>,
    mut key: impl FnMut(&RankBlock7) -> (u32, u32),
) {
    let cap = 4 * m.max(1);
    let (mut tp, mut tr) = ((u32::MAX, usize::MAX), (u32::MAX, usize::MAX));
    let cut = |l: &mut Vec<(u32, usize)>, t: &mut (u32, usize)| {
        l.select_nth_unstable(m - 1);
        l.truncate(m);
        *t = l[m - 1];
    };
    for d in ds {
        let (v, r) = key(&blocks[i - d]);
        if (v, d) < tp {
            pre.push((v, d));
            if pre.len() == cap {
                cut(pre, &mut tp);
            }
        }
        if (r, d) < tr {
            res.push((r, d));
            if res.len() == cap {
                cut(res, &mut tr);
            }
        }
    }
    for l in [pre, res] {
        if m < l.len() {
            l.select_nth_unstable(m);
            l.truncate(m);
        }
    }
}

impl RankSrc7 {
    /// `s`: the source texels; `base`: the base encoding; `base_err`: its weighted SSE;
    /// `smooth`: the block's distortion multiplier; `lambda`: the RDO lambda.
    pub fn new(s: &Block, base: &[u8; 16], base_err: u64, smooth: f32, lambda: f32) -> Self {
        let b = RankBlock7::new(s, base);
        let mut s1 = [0f32; 4];
        let mut vs = [0f32; 4];
        for c in 0..4 {
            let (mut a, mut a2) = (0u32, 0u32);
            for t in 0..16 {
                let x = b.vals[c * 16 + t] as u32;
                a += x;
                a2 += x * x;
            }
            s1[c] = a as f32;
            vs[c] = (16 * a2 - a * a) as f32 / 16.0;
        }
        let rsum = b.range.iter().sum::<u32>() as f32;
        let ctx = [
            fl2((base_err / 4).min(1 << 20) as f32) / 16.0,
            fl2(rsum) / 10.0,
            fl2((lambda * 100.0).min(65535.0).floor()) / 10.0,
            b.ns as f32 / 3.0,
            (smooth - 1.0) / 4.0,
        ];
        let mut h0 = weights::B1;
        for (x, row) in ctx.iter().zip(&weights::W1[NPAIR..]) {
            for j in 0..HIDDEN {
                h0[j] += x * row[j];
            }
        }
        RankSrc7 { b, s1, vs, h0, ctx }
    }

    /// The prefilter's two keys (lower: more promising), both cheap next to the features: the
    /// value SAD, and the residual of fitting the source's values to `p`'s shape per channel by
    /// least squares (about what copying `p`'s indices with refit endpoints would leave).
    #[inline]
    pub fn prefilter(&self, p: &RankBlock7) -> (u32, u32) {
        let sq = cross_sq(&self.b, p);
        let mut r = 0f32;
        for c in 0..4 {
            r += self.resid(sq[c], p, c);
        }
        (sad64(&self.b.vals, &p.vals), r as u32)
    }

    /// The `m` lowest of each prefilter key over the earlier blocks `blocks[i - d]`, `d` in
    /// `ds`, as `(key, d)` into `pre` (value SAD) and `res` (refit residual), unordered. Exactly
    /// the `m` smallest pairs of each, as a full sort would pick (ties go to the nearer block).
    pub fn prefilter_top(
        &self,
        blocks: &[RankBlock7],
        i: usize,
        ds: std::ops::RangeInclusive<usize>,
        m: usize,
        pre: &mut Vec<(u32, usize)>,
        res: &mut Vec<(u32, usize)>,
    ) {
        pre.clear();
        res.clear();
        if m == 0 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            unsafe { self.top_avx2(blocks, i, ds, m, pre, res) };
            return;
        }
        top_with(blocks, i, ds, m, pre, res, |p| self.prefilter(p));
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn top_avx2(
        &self,
        blocks: &[RankBlock7],
        i: usize,
        ds: std::ops::RangeInclusive<usize>,
        m: usize,
        pre: &mut Vec<(u32, usize)>,
        res: &mut Vec<(u32, usize)>,
    ) {
        use std::arch::x86_64::*;
        let ld = |a: &[u8; 64], k: usize| {
            // SAFETY: an unaligned 32-byte load within a 64-byte array.
            unsafe { _mm256_loadu_si256(a.as_ptr().add(k * 32).cast::<__m256i>()) }
        };
        let (s0, s1) = (ld(&self.b.vals, 0), ld(&self.b.vals, 1));
        let ones = _mm256_set1_epi16(1);
        top_with(blocks, i, ds, m, pre, res, |p| {
            let (v0, v1) = (ld(&p.vals, 0), ld(&p.vals, 1));
            let sad = _mm256_add_epi64(_mm256_sad_epu8(s0, v0), _mm256_sad_epu8(s1, v1));
            let sad = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256::<1>(sad));
            let sad = _mm_cvtsi128_si64(_mm_add_epi64(sad, _mm_unpackhi_epi64(sad, sad))) as u32;
            // Source values (u8) times shape (0..=63, so a valid i8): pair sums stay under
            // 2·255·63 < 2^15, so the saturating multiply-add is exact.
            let c0 = _mm256_madd_epi16(_mm256_maddubs_epi16(s0, ld(&p.shape, 0)), ones);
            let c1 = _mm256_madd_epi16(_mm256_maddubs_epi16(s1, ld(&p.shape, 1)), ones);
            // c0: channel 0 in the low 128 bits, 1 in the high; c1: channels 2 and 3.
            let h = _mm256_hadd_epi32(c0, c1);
            let h = _mm256_hadd_epi32(h, h);
            let (lo, hi) = (_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h));
            let sq = [
                _mm_cvtsi128_si32(lo) as u32,
                _mm_cvtsi128_si32(hi) as u32,
                _mm_extract_epi32::<1>(lo) as u32,
                _mm_extract_epi32::<1>(hi) as u32,
            ];
            let mut r = 0f32;
            for (c, &x) in sq.iter().enumerate() {
                r += self.resid(x, p, c);
            }
            (sad, r as u32)
        });
    }

    /// Channel `c`'s least-squares residual of the source's values against `p`'s shape; `sq`:
    /// their cross sum.
    #[inline]
    fn resid(&self, sq: u32, p: &RankBlock7, c: usize) -> f32 {
        let cov = sq as f32 - self.s1[c] * p.q1[c] / 16.0;
        (self.vs[c] - cov * cov * p.inv_vq[c]).clamp(0.0, 1e7)
    }

    /// The pair features of earlier block `p`, `d` blocks back; `far`: `d` is beyond the rate
    /// model's window (its matches are found through the model's hint only).
    pub fn features(&self, p: &RankBlock7, d: usize, far: bool) -> [f32; NPAIR] {
        let s = &self.b;
        let sh = sad64(&s.shape, &p.shape) as f32;
        let ab = sad64(&s.vals, &p.vals) as f32;
        let (sq, cp) = cross(s, p);
        let (mut res_t, mut res_m, mut cp_t, mut cp_m) = (0f32, 0f32, 0u32, 0u32);
        let (mut mn, mut rg) = (0u32, 0u32);
        for c in 0..4 {
            let res = self.resid(sq[c], p, c);
            res_t += res;
            res_m = res_m.max(res);
            cp_t += cp[c];
            cp_m = cp_m.max(cp[c]);
            mn += s.sum[c].abs_diff(p.sum[c]);
            rg += s.range[c].abs_diff(p.range[c]);
        }
        let eq = s.raw.iter().zip(&p.raw).filter(|(a, b)| a == b).count() as f32;
        let same_mode = s.mode == p.mode && s.mode < 8;
        [
            LOG2_D[d.min(LOG2_D.len() - 1)],
            fl2(sh) / 12.0,
            fl2(ab) / 14.0,
            fl2((cp_t / 4) as f32) / 16.0,
            fl2((cp_m / 4) as f32) / 16.0,
            fl2(mn as f32) / 14.0,
            fl2(rg as f32) / 10.0,
            fl2(res_t.floor()) / 20.0,
            fl2(res_m.floor()) / 20.0,
            same_mode as u8 as f32,
            (same_mode && s.partition == p.partition) as u8 as f32,
            p.ns as f32 / 3.0,
            fl2(eq) / 4.0,
            far as u8 as f32,
        ]
    }

    /// The MLP's score (higher: more likely to help), with the block context.
    pub fn mlp(&self, f: &[f32; NPAIR]) -> f32 {
        use weights::{W1, W2};
        // Per hidden unit: (h0 + even inputs' terms) + odd inputs' terms, so two independent
        // add chains; the same order on every target.
        let mut h = [0f32; HIDDEN];
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            // SAFETY: SSE2 is part of x86_64; all loads and stores are unaligned and in bounds of
            // 16-float arrays.
            unsafe {
                let ld = |a: &[f32; HIDDEN], j: usize| _mm_loadu_ps(a.as_ptr().add(j * 4));
                let mut e = [ld(&self.h0, 0), ld(&self.h0, 1), ld(&self.h0, 2), ld(&self.h0, 3)];
                let mut o = [_mm_setzero_ps(); 4];
                for i in (0..NPAIR).step_by(2) {
                    let (xe, xo) = (_mm_set1_ps(f[i]), _mm_set1_ps(f[i + 1]));
                    for j in 0..4 {
                        e[j] = _mm_add_ps(e[j], _mm_mul_ps(xe, ld(&W1[i], j)));
                        o[j] = _mm_add_ps(o[j], _mm_mul_ps(xo, ld(&W1[i + 1], j)));
                    }
                }
                for j in 0..4 {
                    _mm_storeu_ps(h.as_mut_ptr().add(j * 4), _mm_add_ps(e[j], o[j]));
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for j in 0..HIDDEN {
            let (mut e, mut o) = (self.h0[j], 0f32);
            for i in (0..NPAIR).step_by(2) {
                e += f[i] * W1[i][j];
                o += f[i + 1] * W1[i + 1][j];
            }
            h[j] = e + o;
        }
        let mut acc = [0f32; 4];
        for j in 0..HIDDEN {
            acc[j % 4] += h[j].max(0.0) * W2[j];
        }
        (acc[0] + acc[2]) + (acc[1] + acc[3])
    }

    /// One training row per earlier block (`MIPPLE_LOCAL_DUMP`): u32 group, u32 d, f32 × NIN
    /// features (pair, then context), f32 gain (the RD-cost gain its full search brings over the
    /// near and far searches, ≥ 0), f32 prefilter rank (the better of its ranks by the two keys).
    #[expect(clippy::too_many_arguments, reason = "one training row, written field by field")]
    pub fn dump_row(&self, out: &mut Vec<u8>, group: u32, p: &RankBlock7, d: usize, far: bool, gain: f32, rank: u32) {
        out.extend_from_slice(&group.to_le_bytes());
        out.extend_from_slice(&(d as u32).to_le_bytes());
        for x in self.features(p, d, far).iter().chain(&self.ctx) {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&gain.to_le_bytes());
        out.extend_from_slice(&(rank as f32).to_le_bytes());
    }
}

// Generated by scripts/train_ranker_bc7.py.
#[rustfmt::skip]
#[expect(clippy::excessive_precision, reason = "weights are printed at full f32 round-trip precision")]
mod weights {
    use super::{HIDDEN, NIN};
    pub static W1: [[f32; HIDDEN]; NIN] = [
        [-5.538465e-02, -3.563323e-01, -9.816701e-02, 6.259378e-01, 5.439405e-01, 2.613242e-01, 7.251943e-01, 6.676478e-01, 6.464293e-02, 7.743532e-02, 3.420050e-01, -6.453959e-01, 6.937538e-01, -1.885664e-01, 1.767949e-01, 1.759161e-02],
        [-1.653209e-01, -2.193431e-04, 1.683370e-01, 9.006818e-02, 2.430489e-01, 5.985764e-01, 5.028167e-01, 4.976495e-01, 1.085735e-01, -1.464261e-01, 6.803323e-01, 3.061753e-01, 2.151530e-01, -1.717213e-01, 7.427922e-01, -9.609458e-02],
        [-1.091928e-01, -5.915412e-01, -3.479206e-01, 1.089332e+00, 1.017027e+00, 1.176775e+00, 1.417474e+00, 1.576444e+00, -3.392899e-01, 1.307565e+00, -5.550492e-01, -1.705612e+00, 1.353301e+00, -1.725741e-01, 2.452510e-01, -2.656564e-01],
        [-1.318630e-01, -1.209071e+00, -6.705445e-01, 1.654625e+00, 2.058485e+00, 1.646427e+00, 1.886125e+00, 1.658189e+00, -3.129475e-01, 2.620557e+00, -1.036004e+00, -3.594499e+00, 1.215937e+00, -7.387660e-01, -1.370208e-01, -7.716579e-02],
        [-8.125623e-02, -4.814844e-02, -4.306736e-01, 6.252859e-01, 4.400988e-01, 6.721684e-01, 1.214823e+00, 2.615180e-01, -3.298369e-01, 2.827109e+00, -7.022328e-01, -3.459577e+00, 8.102953e-01, -2.627533e-01, 2.206320e-01, -4.707690e-01],
        [3.836500e-01, -7.540355e-01, -1.560220e-01, 7.881952e-01, 1.085352e+00, 1.055980e+00, 5.611436e-01, 6.906369e-01, -1.414298e-01, 1.263078e+00, -1.479403e-01, -2.069014e-01, 7.192824e-01, -1.052758e-02, 1.349805e+00, -2.798874e-01],
        [-1.048372e-01, -1.436640e-01, 9.045999e-02, 3.916650e-01, 2.802483e-01, 1.096873e+00, 2.641012e-01, 4.802059e-01, 5.104346e-02, 3.730471e-01, -2.820304e-01, 1.780408e-01, 3.766234e-01, 8.165691e-04, 6.928067e-01, -2.244341e-01],
        [-4.632419e-01, -4.817337e+00, -4.248347e+00, 4.188437e+00, 5.298887e+00, 4.709980e+00, 5.273379e+00, 5.468768e+00, -5.315750e-01, -2.910757e+00, -4.892644e+00, -1.924024e+00, 4.643615e+00, 2.732382e-01, -3.629327e+00, 1.742559e-01],
        [-4.929949e-01, 1.949232e+00, 1.764014e+00, -2.190086e+00, -1.836839e+00, -1.619083e+00, -1.782494e+00, -1.723914e+00, 1.961659e-01, -2.507510e+00, 2.317916e+00, -3.437718e-01, -2.371979e+00, 1.227603e-01, -2.833515e+00, 2.836917e-01],
        [-6.253882e-02, 1.356554e-01, 1.354862e-01, 3.790343e-02, 2.824651e-01, -1.490846e-01, -6.952820e-02, 3.483275e-03, -2.448411e-01, 4.487983e-01, 4.023061e-01, -1.382023e-02, -9.041993e-03, 2.576418e-01, -2.748219e-01, -5.167850e-01],
        [-2.758188e-03, 8.759123e-02, 2.728916e-01, -1.976795e-01, -4.317891e-01, -7.078426e-02, 2.009404e-02, -7.517445e-02, -4.233570e-01, 1.237229e+00, -4.395909e-02, -2.434162e-01, 4.814570e-03, -3.023395e-01, 4.185107e-01, 1.202760e-01],
        [1.367660e-02, 5.309380e-01, 6.209053e-01, -1.118252e-01, -1.768674e-01, -5.497432e-01, -3.155305e-01, 6.445856e-02, -3.591125e-01, 4.948578e-02, 3.996381e-01, -2.257036e-01, -9.922145e-02, -2.572821e-01, -2.117731e+00, -2.222632e-01],
        [-1.197616e-01, 8.831142e-01, 1.931746e-02, -6.397043e-01, -1.947543e-01, -3.250164e-03, -3.575357e-01, -5.468274e-01, 1.487620e-01, 2.062079e-01, -1.356978e-01, -3.762234e-01, -6.172044e-01, 1.805708e-01, -1.299171e+00, 5.791505e-02],
        [-9.349871e-02, 5.033331e-01, 2.345220e-01, -5.411988e-01, -5.674241e-01, -7.355260e-01, -5.286719e-01, -4.659637e-01, -1.054919e-01, -1.085830e+00, 6.793618e-01, 8.459207e-01, -8.611197e-01, -1.771657e-01, -4.485784e-01, 9.374288e-02],
        [1.277623e-01, 1.271485e+00, 1.258810e+00, 1.461477e-01, -1.158594e-02, 5.606407e-02, -1.342820e+00, 1.815012e-01, -3.929740e-01, 5.673754e-01, 1.176642e+00, 1.984956e+00, -1.127030e-02, 4.090928e-01, 1.448572e+00, -1.382546e-01],
        [-2.736279e-01, 1.824786e+00, 7.417151e-01, -9.053679e-02, -1.126655e-03, -1.355645e-01, -9.331828e-01, -1.685245e-01, 1.229569e-02, -9.509071e-01, 1.119606e+00, 2.165437e+00, 2.621925e-01, -1.605085e-01, 4.025559e-01, 2.635371e-01],
        [-4.564600e-01, 1.167875e+00, 1.024467e+00, 2.685568e-01, 2.460014e-01, 1.517891e-01, -8.152533e-01, 3.440015e-03, -4.564140e-01, -1.946122e+00, 1.153231e+00, 3.669806e+00, -2.055618e-01, 1.550211e-01, 1.807765e+00, -1.016530e-01],
        [2.091237e-01, 8.069192e-01, 7.656958e-01, 2.481222e-01, -9.831151e-02, -3.726542e-02, 4.785714e-01, -1.719075e-01, -2.683589e-01, -8.740054e-01, 5.619356e-01, -2.563521e-01, 1.568364e-01, -1.343144e-01, -1.439139e+00, 1.317378e-03],
        [-3.355684e-01, 5.216637e-01, 2.684602e-02, 1.078082e-01, -6.332086e-03, -1.038251e-01, 3.483133e-02, -1.780876e-01, -1.873994e-01, 4.381549e-01, 3.684210e-01, -1.171703e+00, 7.431645e-02, 3.772978e-02, 7.753386e-01, 1.329785e-01],
    ];
    pub static B1: [f32; HIDDEN] = [0.000000e+00, 1.319630e+00, 8.107386e-01, 1.164996e-01, 1.152350e-01, 8.416680e-02, -1.049152e+00, 3.969902e-04, -1.343132e-01, -5.427132e-01, 1.288276e+00, 1.315263e+00, 7.622954e-02, -6.873584e-02, 1.027472e+00, 0.000000e+00];
    pub static W2: [f32; HIDDEN] = [1.613341e-01, 8.687525e-01, 2.571193e-01, -1.641255e+00, -1.738124e+00, -1.632167e+00, -1.538680e+00, -1.840001e+00, -1.619594e-01, 5.610523e+00, 6.605723e-01, 8.479020e+00, -1.957056e+00, -7.062638e-02, 2.952630e+00, 2.005919e-01];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SIMD paths must match the portable order bit for bit, so output does not depend on
    /// the target.
    #[test]
    fn mlp_matches_scalar_order() {
        use weights::{W1, W2};
        let mut x = 12345u32;
        let mut rnd = || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 24) as u8
        };
        let mut blk = || {
            let mut b: Block = [[0u8; 4]; 16];
            for t in &mut b {
                for c in t.iter_mut() {
                    *c = rnd();
                }
            }
            let raw: [u8; 16] = std::array::from_fn(|_| rnd());
            (b, raw)
        };
        let (src, base) = blk();
        let rs = RankSrc7::new(&src, &base, 12345, 1.5, 1.0);
        for d in 1..200 {
            let (b, raw) = blk();
            let p = RankBlock7::new(&b, &raw);
            let f = rs.features(&p, d, d > 100);
            // Integer kernels against plain loops.
            let (sq, cp) = cross(&rs.b, &p);
            for c in 0..4 {
                let r = c * 16..c * 16 + 16;
                let want_sq: u32 = r.clone().map(|t| rs.b.vals[t] as u32 * p.shape[t] as u32).sum();
                let want_cp: u32 = r
                    .clone()
                    .map(|t| (rs.b.vals[t] as i32 - p.vals[t] as i32).pow(2) as u32)
                    .sum();
                assert_eq!((sq[c], cp[c]), (want_sq, want_cp));
            }
            assert_eq!(cross_sq(&rs.b, &p), sq);
            let want_sad: u32 = (0..64).map(|t| rs.b.shape[t].abs_diff(p.shape[t]) as u32).sum();
            assert_eq!(sad64(&rs.b.shape, &p.shape), want_sad);
            let mut acc = [0f32; 4];
            for j in 0..HIDDEN {
                let (mut e, mut o) = (rs.h0[j], 0f32);
                for i in (0..NPAIR).step_by(2) {
                    e += f[i] * W1[i][j];
                    o += f[i + 1] * W1[i + 1][j];
                }
                acc[j % 4] += (e + o).max(0.0) * W2[j];
            }
            let want = (acc[0] + acc[2]) + (acc[1] + acc[3]);
            assert_eq!(rs.mlp(&f).to_bits(), want.to_bits(), "d = {d}");
        }
    }

    #[test]
    fn head_matches_unpack() {
        let mut x = 99u32;
        for _ in 0..2000 {
            let raw: [u8; 16] = std::array::from_fn(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x >> 24) as u8
            });
            let (mode, part, _) = head(&raw);
            if let Some(u) = crate::bc7::unpack(&raw) {
                assert_eq!((mode, part), (u.mode, u.partition));
            }
        }
    }
}
