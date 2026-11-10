//! Learned ranker for the localized BC4 search: which far earlier blocks of the chunk get the full
//! candidate search. The single-channel counterpart of [`super::bc5`]: it only proposes
//! blocks, the RD search still picks the encoding from their candidates, so a bad ranking costs
//! compression, never correctness.
//!
//! Weights are fitted by `scripts/train_ranker_bc4.py` on CC0 Poly Haven textures
//! (`scripts/fetch_polyhaven.py`), never on the Kodak or Oodle benchmark images. The trainer reads
//! the raw statistics [`RankSrc1::raw`] writes to the dump and applies the same transforms as
//! [`RankSrc1::features`].
//!
//! Inference is plain scalar f32 arithmetic in a fixed order (Rust never contracts to FMA or
//! reassociates), so scores, and the encode, are the same on every target.

use crate::surface::Block;

/// Hidden units.
const H: usize = 16;
/// Pair features (vary with the earlier block), then context features (fixed per source block,
/// folded into the first layer's bias).
pub const NPAIR: usize = 10;
pub const NCTX: usize = 3;
/// Raw statistics per pair as dumped: [`NPAIR`] pair stats... see [`RankSrc1::raw`].
pub const NRAW: usize = 16;

/// What the ranker keeps of each final block: its decoding (channel 0) and endpoints.
#[derive(Clone, Copy)]
pub struct Rank1 {
    vals: [u8; 16],
    /// Values scaled to 0..=63 over the block's own range (the index pattern).
    shape: [u8; 16],
    sum: u32,
    range: u32,
    /// Sum of `shape`, and 1 / (its centered sum of squares), or 0 for a flat block.
    q1: f32,
    inv_vq: f32,
    /// The palette's span: [lo, hi], plus 0 and 255 in 6-value mode.
    lo: u8,
    hi: u8,
    six: bool,
}

/// The source block's side: the same statistics plus the block-level context.
pub struct RankSrc1 {
    b: Rank1,
    s1: f32,
    vs: f32,
    row: usize,
    log2d: &'static [f32],
    ctx: [f32; 3],
    /// The MLP's hidden pre-activations from its bias and the block context alone.
    h0: [f32; H],
}

/// log2(d) / 10 for the reachable distances (BC4 chunks hold 2048 blocks).
static LOG2_D: std::sync::LazyLock<Vec<f32>> =
    std::sync::LazyLock::new(|| (0..4096).map(|d| (d.max(1) as f32).log2() / 10.0).collect());

/// log2(1 + x) by the float bit trick, exactly as the trainer computes it.
#[inline]
fn fl2(x: f32) -> f32 {
    let v = 1.0f32 + x;
    (v.to_bits() as i32 - 0x3F80_0000) as f32 / (1u32 << 23) as f32
}

impl Rank1 {
    /// `dec`: the decoded block (channel 0 is used); `raw`: its 8 encoded bytes.
    pub fn new(dec: &Block, raw: &[u8]) -> Self {
        let (e0, e1) = (raw[0], raw[1]);
        let mut r = Rank1 {
            vals: [0; 16],
            shape: [0; 16],
            sum: 0,
            range: 0,
            q1: 0.0,
            inv_vq: 0.0,
            lo: e0.min(e1),
            hi: e0.max(e1),
            six: e0 <= e1,
        };
        let (mut lo, mut hi, mut sum) = (255u32, 0u32, 0u32);
        for t in 0..16 {
            let v = dec[t][0] as u32;
            r.vals[t] = v as u8;
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v;
        }
        r.sum = sum;
        r.range = hi - lo;
        let (mut q1, mut q2) = (0f32, 0f32);
        if hi > lo {
            for t in 0..16 {
                let q = (dec[t][0] as u32 - lo) * 63 / (hi - lo);
                r.shape[t] = q as u8;
                q1 += q as f32;
                q2 += (q * q) as f32;
            }
        }
        r.q1 = q1;
        let vq = q2 - q1 * q1 / 16.0;
        r.inv_vq = if vq > 0.0 { 1.0 / vq } else { 0.0 };
        r
    }
}

impl RankSrc1 {
    /// `base_sse`: the base encoding's SSE; `row`: blocks per surface row (0 if unknown).
    pub fn new(s: &Block, base_sse: u64, lambda: f32, row: usize) -> Self {
        let mut raw = [0u8; 8];
        raw[0] = 255; // any 8-value pair: the source side's span is unused
        let b = Rank1::new(s, &raw);
        let (mut a, mut a2) = (0f32, 0f32);
        for t in 0..16 {
            let x = b.vals[t] as f32;
            a += x;
            a2 += x * x;
        }
        let ctx = [
            ((base_sse / 4).min(65535)) as f32,
            b.range as f32,
            (lambda * 100.0).min(65535.0).floor(),
        ];
        let mut src = RankSrc1 {
            b,
            s1: a,
            vs: a2 - a * a / 16.0,
            row,
            log2d: &LOG2_D,
            ctx,
            h0: [0.0; H],
        };
        let cf = src.ctx_features();
        let mut h0 = RANK_B1;
        for (x, w) in cf.iter().zip(&RANK_W1[NPAIR..]) {
            for j in 0..H {
                h0[j] += x * w[j];
            }
        }
        src.h0 = h0;
        src
    }

    fn ctx_features(&self) -> [f32; NCTX] {
        let c = &self.ctx;
        [fl2(c[0]) / 16.0, fl2(c[1]) / 8.0, fl2(c[2]) / 10.0]
    }

    /// Raw statistics of earlier block `p`, `d` blocks back, as the training dump stores them:
    /// [d, shape SAD, value SAD, |sum diff|, |range diff|, refit residual (least squares of the
    /// source on p's shape), copy SSE / 4, palette-span SSE / 4 (the re-index's error bound),
    /// distance from vertical alignment, p's range, p's 6-value mode, base SSE / 4, source range,
    /// 100·λ, source sum, p's sum]. Pair features use the first 9, context the 12th to 14th.
    pub fn raw(&self, p: &Rank1, d: usize) -> [f32; NRAW] {
        let s = &self.b;
        let [sh, ab, sq, cp, iv] = stats(s, p);
        let cov = sq as f32 - self.s1 * p.q1 / 16.0;
        // Non-negative after the clamp, so truncation is floor (without a libm call).
        let ls = ((self.vs - cov * cov * p.inv_vq).clamp(0.0, 65535.0) as u32) as f32;
        let vert = if self.row > 0 {
            let r = d % self.row;
            r.min(self.row - r)
        } else {
            d
        };
        [
            d as f32,
            sh as f32,
            ab as f32,
            s.sum.abs_diff(p.sum) as f32,
            s.range.abs_diff(p.range) as f32,
            ls,
            (cp / 4).min(65535) as f32,
            (iv / 4).min(65535) as f32,
            vert.min(65535) as f32,
            p.range as f32,
            p.six as u32 as f32,
            self.ctx[0],
            self.ctx[1],
            self.ctx[2],
            s.sum as f32,
            p.sum as f32,
        ]
    }

    /// The pair features (trainer columns 0..NPAIR; scales are multiplies by the f32
    /// reciprocal, as in the trainer).
    #[inline]
    pub fn features(&self, p: &Rank1, d: usize) -> [f32; NPAIR] {
        let r = self.raw(p, d);
        const R10: f32 = 1.0 / 10.0;
        const R12: f32 = 1.0 / 12.0;
        [
            self.log2d[d.min(self.log2d.len() - 1)],
            fl2(r[1]) * R10,
            fl2(r[2]) * R12,
            fl2(r[3]) * R12,
            fl2(r[4]) * 0.125,
            fl2(r[5]) * 0.0625,
            fl2(r[6]) * 0.0625,
            fl2(r[7]) * 0.0625,
            fl2(r[5].min(r[6])) * 0.0625,
            fl2(r[8]) * 0.125,
        ]
    }

    /// The MLP's score (higher: more likely to help), with the block context.
    pub fn mlp(&self, f: &[f32; NPAIR]) -> f32 {
        // Per hidden unit j: h0[j] (bias and context), each input's term in order, then
        // acc[j % 4] += relu(h[j]) · w2[j] in order of j. The units are independent, so the SSE
        // path (four units per register) keeps every unit's operation order: the same bits.
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is part of x86_64; every load reads 4 floats inside a 16-float array.
        unsafe {
            use std::arch::x86_64::*;
            let ld = |a: &[f32; H], q: usize| _mm_loadu_ps(a.as_ptr().add(q * 4));
            let mut h = [ld(&self.h0, 0), ld(&self.h0, 1), ld(&self.h0, 2), ld(&self.h0, 3)];
            for i in 0..NPAIR {
                let x = _mm_set1_ps(f[i]);
                for q in 0..4 {
                    h[q] = _mm_add_ps(h[q], _mm_mul_ps(x, ld(&RANK_W1[i], q)));
                }
            }
            let z = _mm_setzero_ps();
            let mut acc = z;
            for q in 0..4 {
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_max_ps(h[q], z), ld(&RANK_W2, q)));
            }
            let mut a = [0f32; 4];
            _mm_storeu_ps(a.as_mut_ptr(), acc);
            return (a[0] + a[2]) + (a[1] + a[3]);
        }
        #[cfg_attr(
            target_arch = "x86_64",
            expect(unreachable_code, reason = "scalar reference for non-x86_64 targets")
        )]
        let mut h = self.h0;
        for i in 0..NPAIR {
            let x = f[i];
            let w = &RANK_W1[i];
            for j in 0..H {
                h[j] += x * w[j];
            }
        }
        let mut acc = [0f32; 4];
        for j in 0..H {
            acc[j % 4] += h[j].max(0.0) * RANK_W2[j];
        }
        (acc[0] + acc[2]) + (acc[1] + acc[3])
    }

    #[inline]
    pub fn score(&self, p: &Rank1, d: usize) -> f32 {
        self.mlp(&self.features(p, d))
    }

    /// The localized search's picks among the earlier blocks `back + 1..=i` back (`ranks` holds
    /// every earlier block of the chunk): the `m / 2` most similar by shape SAD together with the
    /// `m / 2` most similar by value SAD (both one `psadbw`; histogram-selected, ties at the cut
    /// go to the nearer blocks), then the `k` the MLP scores highest, as (−score, d).
    pub fn pick<'a>(
        &self,
        ranks: &[Rank1],
        i: usize,
        back: usize,
        m: usize,
        k: usize,
        sc: &'a mut Scratch,
    ) -> &'a [(f32, usize)] {
        let Scratch { sa, va, out } = sc;
        out.clear();
        let n = i.saturating_sub(back);
        if n > m {
            sa.clear();
            va.clear();
            for d in back + 1..=i {
                let (sh, ab) = sads(&self.b, &ranks[i - d]);
                sa.push(sh as u16);
                va.push(ab as u16);
            }
            // The m / 2 smallest of each by a 256-bin histogram (shape SADs ≤ 16·63, value SADs
            // ≤ 16·255); within the bin at the cut, the nearer blocks.
            let h = m / 2;
            let (ts, ns) = cut(sa, 2, h);
            let (tv, nv) = cut(va, 4, h);
            let (mut ns, mut nv) = (ns, nv);
            for (j, (&x, &y)) in sa.iter().zip(va.iter()).enumerate() {
                let (bx, by) = ((x >> 2) as usize, (y >> 4) as usize);
                let mut keep = bx < ts || by < tv;
                if bx == ts && ns > 0 {
                    ns -= 1;
                    keep = true;
                }
                if by == tv && nv > 0 {
                    nv -= 1;
                    keep = true;
                }
                if keep {
                    out.push((0.0, back + 1 + j));
                }
            }
        } else {
            out.extend((back + 1..=i).map(|d| (0.0, d)));
        }
        for e in out.iter_mut() {
            e.0 = -self.score(&ranks[i - e.1], e.1);
        }
        if k < out.len() {
            out.select_nth_unstable_by(k, |x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            out.truncate(k);
        }
        out
    }
}

/// Reusable buffers for [`RankSrc1::pick`].
#[derive(Default)]
pub struct Scratch {
    sa: Vec<u16>,
    va: Vec<u16>,
    out: Vec<(f32, usize)>,
}

/// The histogram bin (of `x >> shift`, 256 bins) holding the `h`-th smallest of `xs`, and how
/// many of that bin's entries are needed to make `h` with the bins below it.
fn cut(xs: &[u16], shift: u32, h: usize) -> (usize, usize) {
    let mut hist = [0u32; 256];
    for &x in xs {
        hist[(x >> shift) as usize] += 1;
    }
    let mut below = 0usize;
    for (t, &c) in hist.iter().enumerate() {
        if below + c as usize >= h {
            return (t, h - below);
        }
        below += c as usize;
    }
    (256, 0)
}

/// Shape SAD and value SAD of two blocks.
#[inline]
fn sads(s: &Rank1, p: &Rank1) -> (u32, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; the loads are unaligned 16-byte loads of 16-byte arrays.
        unsafe {
            let ld = |a: &[u8; 16]| _mm_loadu_si128(a.as_ptr().cast::<__m128i>());
            let h = |v: __m128i| _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_unpackhi_epi64(v, v))) as u32;
            (
                h(_mm_sad_epu8(ld(&s.shape), ld(&p.shape))),
                h(_mm_sad_epu8(ld(&s.vals), ld(&p.vals))),
            )
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let st = stats_scalar(s, p);
        (st[0], st[1])
    }
}

/// Integer statistics of source `s` against earlier block `p`: [shape SAD, value SAD, Σ s·p.shape,
/// copy SSE, palette-span SSE]. Exact, so the SIMD and portable paths agree.
#[inline]
fn stats(s: &Rank1, p: &Rank1) -> [u32; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; the loads are unaligned 16-byte loads of 16-byte arrays.
        unsafe {
            let ld = |a: &[u8; 16]| _mm_loadu_si128(a.as_ptr().cast::<__m128i>());
            let h64 = |v: __m128i| _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_unpackhi_epi64(v, v))) as u32;
            let h32 = |v: __m128i| {
                let v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
                _mm_cvtsi128_si32(_mm_add_epi32(v, _mm_shuffle_epi32(v, 0b10_11_00_01))) as u32
            };
            let z = _mm_setzero_si128();
            // Σ a·b over 16 bytes (each product ≤ 255², pairs fit in i32).
            let dot = |a: __m128i, b: __m128i| {
                h32(_mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi8(a, z), _mm_unpacklo_epi8(b, z)),
                    _mm_madd_epi16(_mm_unpackhi_epi8(a, z), _mm_unpackhi_epi8(b, z)),
                ))
            };
            let (x, y) = (ld(&s.vals), ld(&p.vals));
            let (qs, qp) = (ld(&s.shape), ld(&p.shape));
            let e = _mm_or_si128(_mm_subs_epu8(x, y), _mm_subs_epu8(y, x));
            let mut dv = _mm_or_si128(
                _mm_subs_epu8(_mm_set1_epi8(p.lo as i8), x),
                _mm_subs_epu8(x, _mm_set1_epi8(p.hi as i8)),
            );
            if p.six {
                let edge = _mm_min_epu8(x, _mm_xor_si128(x, _mm_set1_epi8(-1)));
                dv = _mm_min_epu8(dv, edge);
            }
            [
                h64(_mm_sad_epu8(qs, qp)),
                h64(_mm_sad_epu8(x, y)),
                dot(x, qp),
                dot(e, e),
                dot(dv, dv),
            ]
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        stats_scalar(s, p)
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    expect(dead_code, reason = "scalar fallback; x86_64 uses SSE2")
)]
fn stats_scalar(s: &Rank1, p: &Rank1) -> [u32; 5] {
    let (mut sh, mut ab, mut sq, mut cp, mut iv) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for t in 0..16 {
        let (x, y) = (s.vals[t], p.vals[t]);
        sh += s.shape[t].abs_diff(p.shape[t]) as u32;
        ab += x.abs_diff(y) as u32;
        sq += x as u32 * p.shape[t] as u32;
        let e = x.abs_diff(y) as u32;
        cp += e * e;
        let mut dv = p.lo.saturating_sub(x).max(x.saturating_sub(p.hi)) as u32;
        if p.six {
            dv = dv.min(x as u32).min(255 - x as u32);
        }
        iv += dv * dv;
    }
    [sh, ab, sq, cp, iv]
}

/// Ranker training rows (`MIPPLE_LOCAL_DUMP=path` with BC4): one row per far block j = i − d of
/// a dump block: u32 run (process id), u32 group, f32 × [`NRAW`] ([`RankSrc1::raw`]), f32 gain
/// (near best J − best J with j's candidates, ≥ 0), f32 near best J.
#[expect(clippy::too_many_arguments, reason = "one training group, written row by row")]
pub fn dump_rows(
    out: &mut impl std::io::Write,
    group: u32,
    src: &RankSrc1,
    ranks: &[Rank1],
    i: usize,
    ds: &[usize],
    gains: &[f32],
    near_j: f64,
) {
    let run = std::process::id();
    let mut buf = Vec::with_capacity(ds.len() * (16 + 4 * NRAW));
    for (k, &d) in ds.iter().enumerate() {
        buf.extend_from_slice(&run.to_le_bytes());
        buf.extend_from_slice(&group.to_le_bytes());
        for x in src.raw(&ranks[i - d], d) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&gains[k].to_le_bytes());
        buf.extend_from_slice(&(near_j as f32).to_le_bytes());
    }
    out.write_all(&buf).unwrap();
    out.flush().unwrap();
}

// Generated by scripts/train_ranker_bc4.py.
#[rustfmt::skip]
#[expect(clippy::excessive_precision, reason = "weights are printed at full f32 round-trip precision")]
mod weights {
    pub static RANK_W1: [[f32; 16]; 13] = [
        [3.772209e-01, 4.028778e-01, 3.542313e-01, -9.858561e-02, -2.106291e-01, -4.521110e-01, 9.348659e-02, -1.287868e-01, 9.548050e-01, 3.771565e-01, -1.697626e-01, 1.193325e+00, -1.614115e-01, -1.839762e-01, -1.575549e-01, -1.976080e-01],
        [5.169122e-02, 3.702847e-01, 7.808737e-01, -5.656283e-01, 2.556548e-01, 1.194281e-01, -2.859590e-01, -4.389747e-01, 8.105468e-01, 2.095595e-01, -1.193886e-01, 7.118275e-01, 1.028820e-01, -8.099683e-01, -7.324725e-01, 7.752503e-02],
        [-2.783248e-01, -8.198842e-02, 2.923376e-01, 4.506375e-01, -3.906155e-01, -1.129246e-01, -8.843828e-01, -5.772408e-02, 8.618589e-03, -4.938833e-01, -4.539593e-01, 8.086607e-02, -2.917772e-01, 5.712328e-01, 4.539346e-01, -1.522982e-02],
        [-1.059437e+00, -8.159174e-01, -2.097308e-01, 3.315591e-01, -4.831970e-01, -2.166827e-02, -1.065463e+00, -6.749003e-02, -6.622749e-02, -1.312802e-01, -1.084822e-01, 1.030653e-01, -1.997083e-01, 1.779400e-01, 1.633180e-01, -2.223783e-01],
        [1.566398e-01, 4.954917e-01, 4.366623e-01, 3.390955e-01, -1.599104e-01, 1.455376e-01, 1.189048e-01, -4.184486e-01, 5.018634e-01, -5.274515e-01, 5.118986e-02, 1.303286e-01, 3.232525e-01, 2.646247e-01, 1.747243e-01, 3.149541e-01],
        [-2.287665e-01, 1.699275e+00, 2.617650e+00, -2.213065e+00, -1.791582e-02, -5.234238e-02, -9.756840e-02, 2.681578e-02, 2.012892e+00, -4.192259e-01, -1.202503e-01, 2.529260e+00, -4.735079e-02, -2.228087e+00, -1.557699e+00, 2.989695e-01],
        [-9.159388e-01, -9.513254e-02, 5.026383e-03, 2.475356e-02, 2.402660e-02, 1.024945e-01, -1.139898e+00, -5.874078e-01, 1.236855e-01, -7.967985e-01, -4.540465e-01, -3.749975e-01, -3.145476e-01, -8.743199e-02, -2.132866e-01, -2.157627e-01],
        [-1.373030e-01, 5.083910e-01, 9.833384e-01, 3.702186e-01, -1.754598e-01, -1.063376e-01, 8.388977e-02, -2.179494e-02, 2.247871e-01, -2.147613e-01, -1.233123e-01, 9.630818e-01, -1.517338e-02, 4.036185e-01, 3.937486e-01, 6.664430e-02],
        [-4.185376e-01, 2.787330e+00, 2.000732e+00, -2.544603e+00, 3.913096e-02, -5.674328e-01, -1.859563e+00, -1.398098e-01, 1.814689e+00, -1.563481e-01, -4.247542e-01, 2.210872e+00, 1.377653e-01, -2.673533e+00, -2.406290e+00, -2.156427e-01],
        [5.580269e-01, 4.622729e-01, -2.048540e-02, 2.753782e-02, -3.663985e-01, -3.092055e-01, -1.014132e-01, 9.109089e-02, -8.290990e-02, -4.397452e-01, 5.782813e-01, -1.173761e-01, -1.076089e-01, -3.034048e-02, 1.353547e-02, -2.226551e-01],
        [5.100343e-01, 5.985050e-02, -2.081546e-01, 7.023778e-01, -1.719033e-01, -2.984448e-02, 1.374159e+00, -7.253674e-03, -2.210705e-01, -1.917847e-01, -1.897036e-01, 2.192821e-01, -3.715749e-01, 7.875789e-01, 6.802166e-01, -1.664507e-01],
        [8.663591e-01, 5.212311e-01, 2.736464e-01, 6.266806e-01, -3.205975e-01, -8.776930e-02, 8.547489e-01, -2.567123e-02, -5.716393e-02, 6.390532e-03, -9.675466e-02, 3.400764e-01, -1.799395e-01, 9.231267e-01, 7.237590e-01, -4.692006e-02],
        [5.269716e-01, -3.238493e-01, 1.651881e-03, 1.919539e+00, -2.792180e-01, 2.788977e-03, 1.617025e+00, -1.474541e-02, -2.733967e-02, -6.804439e-02, 9.586396e-02, 2.075595e-01, 2.534502e-01, 2.102701e+00, 1.909257e+00, -5.326563e-01],
    ];
    pub static RANK_B1: [f32; 16] = [5.587415e-01, 1.892606e-01, 3.579977e-02, 6.095174e-01, 0.000000e+00, 0.000000e+00, 5.919887e-01, 0.000000e+00, -6.096017e-02, -9.447015e-02, -8.911925e-02, -4.077928e-01, -4.068205e-02, 7.238595e-01, 5.286990e-01, 0.000000e+00];
    pub static RANK_W2: [f32; 16] = [1.430388e+00, -1.954408e+00, -1.764935e+00, 3.028420e+00, -1.171701e-01, 5.328535e-02, 3.930232e+00, 4.907449e-01, -2.096284e+00, -2.075790e-01, 5.560260e-02, -1.743821e+00, 3.052913e-01, 3.433317e+00, 3.775352e+00, 1.635321e-01];
}
use weights::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u32) -> impl FnMut() -> u8 {
        let mut x = seed;
        move || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 24) as u8
        }
    }

    fn block(rnd: &mut impl FnMut() -> u8) -> (Block, [u8; 8]) {
        let mut b: Block = [[0u8; 4]; 16];
        let (lo, span) = (rnd() / 2, rnd() / 2);
        for t in &mut b {
            t[0] = lo + (rnd() as u32 * span as u32 / 255) as u8;
        }
        let mut raw = [0u8; 8];
        raw[0] = rnd();
        raw[1] = rnd();
        (b, raw)
    }

    /// Inference is the scalar order written out per hidden unit (bias, context, then the pair
    /// features), so scores, and the encode, do not depend on the target or on vectorization.
    #[test]
    fn mlp_matches_scalar_order() {
        let mut rnd = rng(12345);
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        for n in 0..8 {
            let (s, _) = block(&mut rnd);
            let rs = RankSrc1::new(&s, 1000 * n as u64, 0.25 * (n + 1) as f32, 192);
            let cf = rs.ctx_features();
            for d in 1..200 {
                let (b, raw) = block(&mut rnd);
                let p = Rank1::new(&b, &raw);
                let f = rs.features(&p, d);
                let mut acc = [0f32; 4];
                for j in 0..H {
                    let mut h = RANK_B1[j];
                    for i in 0..NCTX {
                        h += cf[i] * RANK_W1[NPAIR + i][j];
                    }
                    for i in 0..NPAIR {
                        h += f[i] * RANK_W1[i][j];
                    }
                    acc[j % 4] += h.max(0.0) * RANK_W2[j];
                }
                let want = (acc[0] + acc[2]) + (acc[1] + acc[3]);
                assert_eq!(rs.score(&p, d).to_bits(), want.to_bits(), "n = {n}, d = {d}");
                hash = (hash ^ want.to_bits() as u64).wrapping_mul(0x100_0000_01B3);
            }
        }
        // Pins the weights and features: retraining or changing a feature changes this.
        assert_eq!(hash, 0x959B_861D_3709_649D, "scores hash");
    }

    /// The SIMD SADs match the portable sums, and the picks are the top K of the scores among
    /// the prefiltered blocks.
    #[test]
    fn pick_matches_reference() {
        let mut rnd = rng(777);
        let ranks: Vec<Rank1> = (0..600)
            .map(|_| {
                let (b, raw) = block(&mut rnd);
                Rank1::new(&b, &raw)
            })
            .collect();
        let (s, _) = block(&mut rnd);
        let rs = RankSrc1::new(&s, 5000, 1.0, 64);
        let i = 599;
        let back = 100;
        for p in &ranks {
            assert_eq!(stats(&rs.b, p), stats_scalar(&rs.b, p));
            let st = stats_scalar(&rs.b, p);
            assert_eq!(sads(&rs.b, p), (st[0], st[1]));
        }
        let mut sc = Scratch::default();
        // Prefilter: at most m blocks, all distinct and in range.
        let ds: Vec<usize> = rs
            .pick(&ranks, i, back, 128, usize::MAX, &mut sc)
            .iter()
            .map(|e| e.1)
            .collect();
        assert!(ds.len() >= 64 && ds.len() <= 128);
        assert!(ds.windows(2).all(|w| w[0] < w[1]));
        assert!(ds.iter().all(|&d| d > back && d <= i));
        // Without the prefilter, the picks are the K best scores.
        let out = rs.pick(&ranks, i, back, usize::MAX, 16, &mut sc);
        let mut all: Vec<f32> = (back + 1..=i).map(|d| rs.score(&ranks[i - d], d)).collect();
        all.sort_by(|a, b| b.total_cmp(a));
        let mut got: Vec<f32> = out.iter().map(|e| -e.0).collect();
        got.sort_by(|a, b| b.total_cmp(a));
        assert_eq!(got, all[..16]);
    }
}
