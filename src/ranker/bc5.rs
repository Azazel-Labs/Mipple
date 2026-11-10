//! Learned ranker for the localized search (BC5): which far earlier blocks get the full candidate
//! search. It only proposes blocks; the RD search still picks the encoding from their candidates,
//! so a bad ranking costs compression, never correctness.
//!
//! Weights are fitted by `scripts/train_ranker.py` on CC0 Poly Haven textures
//! (`scripts/fetch_polyhaven.py`), never on the Kodak or Oodle benchmark images. Features must
//! match that script's `features()`.

use crate::surface::Block;

/// What the ranker keeps of each final (decoded) block, for its two weighted channels.
#[derive(Clone, Copy)]
pub struct RankBlock {
    /// Decoded values, channel-major (16 per channel).
    vals: [u8; 32],
    /// Values scaled to 0..=63 over the block's own range per channel (the index pattern).
    shape: [u8; 32],
    sum: [u32; 2],
    range: [u32; 2],
    /// Sum and centered sum of squares of `shape`, per channel.
    q1: [f32; 2],
    /// 1 / (centered sum of squares of `shape`), or 0 for a flat channel.
    inv_vq: [f32; 2],
}

/// The source block's side: the same statistics plus the block-level context.
pub struct RankSrc {
    b: RankBlock,
    s1: [f32; 2],
    vs: [f32; 2],
    /// The MLP's hidden pre-activations from its bias and the block context alone.
    h0: [f32; 16],
}

/// log2(d) / 10 for the reachable distances (BC5 chunks hold 1024 blocks).
static LOG2_D: std::sync::LazyLock<Vec<f32>> =
    std::sync::LazyLock::new(|| (0..4096).map(|d| (d.max(1) as f32).log2() / 10.0).collect());

/// log2(1 + x) by the float bit trick, exactly as the trainer computes it.
#[inline]
fn fl2(x: f32) -> f32 {
    let v = 1.0f32 + x;
    (v.to_bits() as i32 - 0x3F80_0000) as f32 / (1u32 << 23) as f32
}

impl RankBlock {
    pub fn new(b: &Block, ch: [usize; 2]) -> Self {
        let mut r = RankBlock {
            vals: [0; 32],
            shape: [0; 32],
            sum: [0; 2],
            range: [0; 2],
            q1: [0.0; 2],
            inv_vq: [0.0; 2],
        };
        for (k, &c) in ch.iter().enumerate() {
            let (mut lo, mut hi, mut sum) = (255u32, 0u32, 0u32);
            for t in 0..16 {
                let v = b[t][c] as u32;
                r.vals[k * 16 + t] = v as u8;
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v;
            }
            r.sum[k] = sum;
            r.range[k] = hi - lo;
            let (mut q1, mut q2) = (0f32, 0f32);
            if hi > lo {
                for t in 0..16 {
                    let q = (b[t][c] as u32 - lo) * 63 / (hi - lo);
                    r.shape[k * 16 + t] = q as u8;
                    q1 += q as f32;
                    q2 += (q * q) as f32;
                }
            }
            r.q1[k] = q1;
            let vq = q2 - q1 * q1 / 16.0;
            r.inv_vq[k] = if vq > 0.0 { 1.0 / vq } else { 0.0 };
        }
        r
    }
}

/// Per channel: shape SAD, value SAD, Σ source value · earlier shape, value SSE.
#[inline]
fn sums(s: &RankBlock, p: &RankBlock) -> ([u32; 2], [u32; 2], [u32; 2], [u32; 2]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of x86_64; every load is an unaligned 16-byte load within a
        // 32-byte array.
        unsafe {
            let ld = |a: &[u8; 32], k: usize| _mm_loadu_si128(a.as_ptr().add(k * 16).cast::<__m128i>());
            let hsum64 = |v: __m128i| _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_unpackhi_epi64(v, v))) as u32;
            let hsum32 = |v: __m128i| {
                let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
                let v = _mm_add_epi32(v, _mm_shuffle_epi32::<0b10_11_00_01>(v));
                _mm_cvtsi128_si32(v) as u32
            };
            let z = _mm_setzero_si128();
            let mut out = ([0u32; 2], [0u32; 2], [0u32; 2], [0u32; 2]);
            for k in 0..2 {
                let (ss, ps) = (ld(&s.shape, k), ld(&p.shape, k));
                let (sv, pv) = (ld(&s.vals, k), ld(&p.vals, k));
                out.0[k] = hsum64(_mm_sad_epu8(ss, ps));
                out.1[k] = hsum64(_mm_sad_epu8(sv, pv));
                let (svl, svh) = (_mm_unpacklo_epi8(sv, z), _mm_unpackhi_epi8(sv, z));
                let (psl, psh) = (_mm_unpacklo_epi8(ps, z), _mm_unpackhi_epi8(ps, z));
                let (pvl, pvh) = (_mm_unpacklo_epi8(pv, z), _mm_unpackhi_epi8(pv, z));
                out.2[k] = hsum32(_mm_add_epi32(_mm_madd_epi16(svl, psl), _mm_madd_epi16(svh, psh)));
                let (dl, dh) = (_mm_sub_epi16(svl, pvl), _mm_sub_epi16(svh, pvh));
                out.3[k] = hsum32(_mm_add_epi32(_mm_madd_epi16(dl, dl), _mm_madd_epi16(dh, dh)));
            }
            out
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = ([0u32; 2], [0u32; 2], [0u32; 2], [0u32; 2]);
        for k in 0..2 {
            for t in k * 16..k * 16 + 16 {
                out.0[k] += s.shape[t].abs_diff(p.shape[t]) as u32;
                out.1[k] += s.vals[t].abs_diff(p.vals[t]) as u32;
                out.2[k] += s.vals[t] as u32 * p.shape[t] as u32;
                let e = s.vals[t] as i32 - p.vals[t] as i32;
                out.3[k] += (e * e) as u32;
            }
        }
        out
    }
}

impl RankSrc {
    /// `base_half`: the base encoding's SSE per half; `lambda`: the RDO lambda.
    pub fn new(s: &Block, ch: [usize; 2], base_half: [u64; 2], lambda: f32) -> Self {
        let b = RankBlock::new(s, ch);
        let mut s1 = [0f32; 2];
        let mut vs = [0f32; 2];
        for k in 0..2 {
            let (mut a, mut a2) = (0f32, 0f32);
            for t in 0..16 {
                let x = b.vals[k * 16 + t] as f32;
                a += x;
                a2 += x * x;
            }
            s1[k] = a;
            vs[k] = a2 - a * a / 16.0;
        }
        let q = |x: u64| ((x / 4).min(65535)) as f32;
        let ctx = [
            fl2(q(base_half[0])) / 16.0,
            fl2(q(base_half[1])) / 16.0,
            fl2(b.range[0] as f32) / 8.0,
            fl2(b.range[1] as f32) / 8.0,
            fl2((lambda * 100.0).min(65535.0).floor()) / 10.0,
        ];
        let mut h0 = RANK_B1;
        for (x, row) in ctx.iter().zip(&RANK_W1[18..]) {
            for j in 0..16 {
                h0[j] += x * row[j];
            }
        }
        RankSrc { b, s1, vs, h0 }
    }

    /// The pair features of earlier block `p`, `d` blocks back (trainer columns 0..18).
    pub fn features(&self, p: &RankBlock, d: usize) -> [f32; 18] {
        let s = &self.b;
        let (sh, ab, sq, c) = sums(s, p);
        let mut ls = [0f32; 2];
        let mut cp = [0f32; 2];
        for k in 0..2 {
            let cov = sq[k] as f32 - self.s1[k] * p.q1[k] / 16.0;
            let res = self.vs[k] - cov * cov * p.inv_vq[k];
            ls[k] = res.clamp(0.0, 65535.0).floor();
            cp[k] = (c[k] / 4).min(65535) as f32;
        }
        let sh = sh.map(|x| x as f32);
        let ab = ab.map(|x| x as f32);
        let mn = [s.sum[0].abs_diff(p.sum[0]) as f32, s.sum[1].abs_diff(p.sum[1]) as f32];
        let rg = [
            s.range[0].abs_diff(p.range[0]) as f32,
            s.range[1].abs_diff(p.range[1]) as f32,
        ];
        [
            LOG2_D[d.min(LOG2_D.len() - 1)],
            fl2(sh[0]) / 10.0,
            fl2(sh[1]) / 10.0,
            fl2(sh[0] + sh[1]) / 10.0,
            fl2(ab[0]) / 12.0,
            fl2(ab[1]) / 12.0,
            fl2(ab[0].min(ab[1])) / 12.0,
            fl2(ab[0] + ab[1]) / 12.0,
            fl2(mn[0]) / 12.0,
            fl2(mn[1]) / 12.0,
            fl2(rg[0]) / 8.0,
            fl2(rg[1]) / 8.0,
            fl2(ls[0]) / 16.0,
            fl2(ls[1]) / 16.0,
            fl2(ls[0] + ls[1]) / 16.0,
            fl2(cp[0]) / 16.0,
            fl2(cp[1]) / 16.0,
            fl2(cp[0].min(cp[1])) / 16.0,
        ]
    }

    /// The MLP's score (higher: more likely to help), with the block context.
    pub fn mlp(&self, f: &[f32; 18]) -> f32 {
        // Per hidden unit: (h0 + even inputs' terms) + odd inputs' terms, so two independent
        // add chains; the same order on every target. The context terms come first (in `h0`),
        // not last as in the trainer: a rounding-level difference.
        let mut h = [0f32; 16];
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            // SAFETY: SSE2 is part of x86_64; all loads and stores are unaligned and in bounds of
            // 16-float arrays.
            unsafe {
                let ld = |a: &[f32; 16], j: usize| _mm_loadu_ps(a.as_ptr().add(j * 4));
                let mut e = [ld(&self.h0, 0), ld(&self.h0, 1), ld(&self.h0, 2), ld(&self.h0, 3)];
                let mut o = [_mm_setzero_ps(); 4];
                for i in (0..18).step_by(2) {
                    let (xe, xo) = (_mm_set1_ps(f[i]), _mm_set1_ps(f[i + 1]));
                    for j in 0..4 {
                        e[j] = _mm_add_ps(e[j], _mm_mul_ps(xe, ld(&RANK_W1[i], j)));
                        o[j] = _mm_add_ps(o[j], _mm_mul_ps(xo, ld(&RANK_W1[i + 1], j)));
                    }
                }
                for j in 0..4 {
                    _mm_storeu_ps(h.as_mut_ptr().add(j * 4), _mm_add_ps(e[j], o[j]));
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for j in 0..16 {
            let (mut e, mut o) = (self.h0[j], 0f32);
            for i in (0..18).step_by(2) {
                e += f[i] * RANK_W1[i][j];
                o += f[i + 1] * RANK_W1[i + 1][j];
            }
            h[j] = e + o;
        }
        let mut acc = [0f32; 4];
        for j in 0..16 {
            acc[j % 4] += h[j].max(0.0) * RANK_W2[j];
        }
        (acc[0] + acc[2]) + (acc[1] + acc[3])
    }
}

// Generated by scripts/train_ranker.py.
#[rustfmt::skip]
#[expect(clippy::excessive_precision, reason = "weights are printed at full f32 round-trip precision")]
mod weights {
    pub static RANK_W1: [[f32; 16]; 23] = [
        [7.663524e-01, 9.013168e-01, 5.786793e-03, 1.468319e-02, 1.709465e-01, 4.427373e-02, 2.088166e-01, -2.194162e-01, 7.959780e-01, 9.443142e-02, 3.163185e-01, 6.031459e-01, 3.418264e-01, -2.636684e-01, -4.043229e-01, 1.345289e-01],
        [9.003859e-01, 9.753768e-01, 4.743617e-01, 9.901446e-01, -9.869144e-03, -2.925375e-01, 8.262774e-01, -1.969453e-01, 6.579299e-01, -5.334285e-03, 1.116337e+00, 9.542915e-01, 9.346019e-02, 1.904203e-02, -2.570437e-01, 7.622942e-01],
        [2.033290e-01, 1.252043e-01, 5.074988e-01, 1.622281e-01, -2.898722e-01, -2.490384e-01, 2.561809e-01, -1.418923e-01, 5.744720e-01, -1.204861e-01, 1.420334e-01, 3.749103e-02, -1.434271e-03, 1.656857e-01, -1.955003e-01, 4.967421e-01],
        [2.790923e-01, 2.588058e-01, 2.661455e-01, 6.820753e-02, -9.161384e-02, 1.894789e-02, 8.196498e-02, 1.689211e-01, -2.889501e-03, -1.591943e-01, -2.244417e-01, 4.089045e-02, 1.213905e-01, 1.651998e-01, -1.712187e-01, 4.153453e-01],
        [-3.475579e-01, 2.295656e-01, -2.491636e-01, -1.237690e-01, 5.144328e-03, -8.324220e-02, 2.729180e-01, 4.477312e-02, 3.337049e-01, -3.011063e-02, 2.949861e-02, -6.873179e-02, -8.036442e-01, -2.407862e-01, -7.206579e-02, -3.354931e-02],
        [1.682405e-01, -1.469094e-01, -2.195304e-02, 7.178327e-01, -1.252071e-01, -2.054925e-01, 7.622862e-02, -1.742367e-01, 3.100902e-01, -3.129723e-01, 4.366081e-02, 3.170389e-01, -8.729902e-01, -8.715552e-02, 2.285961e-01, 4.766576e-02],
        [-4.083125e-01, -2.014011e-01, -1.447077e-01, 8.160133e-02, 1.745173e-01, -3.206666e-01, 1.799411e-01, -3.949045e-01, -6.707294e-02, 4.395735e-01, 4.064999e-02, -2.389647e-03, -9.256696e-01, -2.314047e-01, 1.836521e-01, -4.417133e-01],
        [-2.464981e-01, -2.159708e-01, -7.117272e-02, -9.849703e-03, -2.352960e-01, 6.680672e-02, 4.756903e-02, 2.861011e-01, -3.774474e-01, 1.000158e-01, -2.165234e-01, 3.329281e-03, -1.037336e+00, -2.564007e-01, 2.059609e-01, -1.316222e-01],
        [2.751216e-01, 6.374308e-02, 2.830135e-01, 1.025112e-02, -9.651891e-02, 3.645265e-01, 4.946307e-01, -3.061444e-02, 3.301793e-01, 9.003884e-02, 2.449760e-01, -1.891628e-01, -2.855264e-01, -5.244859e-01, -1.646982e-01, 2.885509e-01],
        [3.056140e-01, 9.420159e-02, -9.922554e-02, 4.367512e-01, -1.434010e-01, -1.215377e-01, 3.175508e-01, -1.515377e-01, 1.487271e-01, 1.487375e-01, 2.469849e-01, 4.613002e-01, 2.162624e-02, 1.404180e-01, 5.432784e-01, 4.158781e-01],
        [-6.854301e-03, 4.166916e-02, 3.726164e-01, 2.033018e-01, -3.110142e-01, -2.353107e-01, 2.881975e-01, 4.450284e-02, 1.010038e-01, -6.382351e-02, 2.633734e-01, 4.034402e-01, 2.403129e-01, -1.706658e-01, -3.286087e-01, 1.764734e-01],
        [9.770580e-02, 2.121929e-01, 3.777070e-01, -7.612327e-02, 4.471968e-01, -2.161294e-01, 1.128191e-01, -1.677305e-01, 4.375816e-01, -1.680121e-02, 6.823616e-01, 2.376198e-01, 6.101940e-02, -3.198827e-02, 4.844446e-02, 1.867242e-01],
        [1.922849e+00, 1.707143e+00, 1.565630e+00, 1.689349e+00, -4.177911e-01, 2.043759e-01, 1.445034e+00, -2.965147e-01, 1.403917e+00, -1.533090e-01, 1.738230e+00, 1.795643e+00, -4.188361e-01, 1.952018e-01, -9.727664e-02, 1.964202e+00],
        [1.789220e+00, 1.795860e+00, 1.735746e+00, 1.339617e+00, 4.909172e-02, 6.435499e-02, 1.245365e+00, -3.438827e-01, 1.369664e+00, -5.257125e-01, 1.587193e+00, 1.217320e+00, 3.418182e-01, -3.856743e-01, -1.111880e-01, 1.714557e+00],
        [8.552380e-01, 1.040328e+00, 8.659550e-01, 7.226480e-01, -1.370146e-01, 1.095737e-01, 6.558105e-01, -4.740155e-01, 1.211118e+00, 3.542172e-03, 8.200199e-01, 1.235430e+00, 7.616166e-02, -2.866092e-02, -1.159669e-01, 1.420270e+00],
        [2.950549e-01, 3.689976e-01, 2.273588e-01, 3.465130e-01, -1.958243e-01, 2.974509e-02, 8.181398e-03, -2.787960e-02, 4.518759e-01, -5.898719e-01, 7.659534e-02, -1.028890e-01, -1.729818e+00, 9.393910e-03, 6.998916e-03, -1.500637e-01],
        [1.761108e-01, 6.517752e-02, 3.861091e-01, 1.325068e-01, 6.478336e-02, -9.870096e-02, 3.264372e-01, 6.071555e-02, 2.379069e-01, -7.138651e-02, -1.421133e-02, 2.214175e-02, -1.230990e+00, -3.256999e-02, -3.158396e-03, 3.558468e-02],
        [1.964312e-01, -3.246076e-01, 4.142994e-02, -1.469596e-01, -2.154179e-01, 1.409941e-01, 1.662040e-01, 2.397389e-01, 9.044942e-02, -2.185389e-01, 2.449506e-02, -7.607269e-02, -1.425134e+00, -2.706851e-01, -2.106317e-02, 8.148603e-02],
        [-3.249781e-01, 4.752504e-02, -5.084126e-01, -5.147414e-02, 7.738969e-02, 1.085790e-01, 2.191462e-01, 1.251348e-01, 1.131766e-01, -3.841833e-01, 5.943691e-02, -7.150988e-02, 1.138458e+00, 1.958537e-01, -2.390287e-01, 1.224502e-01],
        [-1.682667e-01, -3.469965e-02, -9.376697e-02, -4.225675e-02, -1.168100e-01, 1.484558e-02, -9.828841e-02, -1.191760e-01, -6.741335e-02, -2.789671e-01, 5.865360e-02, -3.940770e-02, 1.207103e+00, 4.214714e-02, -1.591435e-01, -3.335843e-01],
        [-1.627661e-02, -3.103636e-01, 2.413063e-01, 3.894870e-02, -1.881508e-01, -1.144351e-01, -6.504051e-02, -1.488577e-01, -6.017463e-01, -1.909619e-01, 9.323325e-02, 1.088992e-01, 9.437738e-01, 3.434607e-01, 1.683857e-01, -4.520989e-01],
        [-1.457531e-02, -2.540772e-01, 7.086363e-02, 3.007667e-02, 3.818315e-01, -6.454093e-02, 1.474158e-01, -4.026489e-02, -2.383136e-01, -4.761686e-01, -3.568995e-01, -4.109916e-02, 1.043362e+00, -3.300826e-01, 2.820649e-02, -5.808101e-01],
        [-3.412095e-01, -4.824291e-01, 1.558872e-01, 2.674616e-01, 2.493963e-02, -3.204742e-02, 2.155373e-01, -8.818140e-02, -2.589228e-01, 1.157780e-01, -7.873969e-02, 9.467266e-03, 1.978698e+00, -3.454261e-01, -8.165114e-02, -3.050128e-02],
    ];
    pub static RANK_B1: [f32; 16] = [-6.336346e-02, -2.783416e-01, 4.078907e-02, 8.259144e-02, 0.000000e+00, 0.000000e+00, 6.201970e-02, -6.526209e-02, -4.523359e-01, -6.556395e-02, 8.210511e-02, 5.996805e-02, 6.873611e-01, -5.977165e-02, 0.000000e+00, -4.383584e-01];
    pub static RANK_W2: [f32; 16] = [-9.685620e-01, -7.195233e-01, -1.010642e+00, -8.954938e-01, 1.615798e-01, 3.485184e-01, -1.199638e+00, 1.050719e-01, -7.050840e-01, 4.368637e-02, -7.899532e-01, -8.485752e-01, 5.039205e+00, 3.375761e-02, -1.621048e-01, -8.380573e-01];
}
use weights::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// The SIMD path must match the portable one bit for bit, so output does not depend on the
    /// target.
    #[test]
    fn mlp_matches_scalar_order() {
        let mut x = 12345u32;
        let mut rnd = || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 24) as u8
        };
        let mut blk = || {
            let mut b: Block = [[0u8; 4]; 16];
            for t in &mut b {
                t[0] = rnd();
                t[1] = rnd();
            }
            b
        };
        let src = blk();
        let rs = RankSrc::new(&src, [0, 1], [1000, 2000], 1.0);
        for d in 1..200 {
            let f = rs.features(&RankBlock::new(&blk(), [0, 1]), d);
            let mut acc = [0f32; 4];
            for j in 0..16 {
                let (mut e, mut o) = (rs.h0[j], 0f32);
                for i in (0..18).step_by(2) {
                    e += f[i] * RANK_W1[i][j];
                    o += f[i + 1] * RANK_W1[i + 1][j];
                }
                acc[j % 4] += (e + o).max(0.0) * RANK_W2[j];
            }
            let want = (acc[0] + acc[2]) + (acc[1] + acc[3]);
            assert_eq!(rs.mlp(&f).to_bits(), want.to_bits(), "d = {d}");
        }
    }
}
