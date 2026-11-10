//! Vectorized BC7 partition ranking: line-fit residuals of 8 partitions per SIMD step.
//!
//! A partition's residual is the sum over its subsets of the squared distances from the subset's
//! texels to their best-fit line, `trace(S) - λmax(S)` of each subset's scatter matrix S. Every BC7
//! palette lies on a line segment, so this is a lower bound on any encoding's error and both
//! ranks partitions and prunes the search in [`super::encode`].
//!
//! Per batch of 8 partitions (one per lane), each subset's count, sums and second moments are
//! accumulated with 0/1 lane masks over the 16 texels (subset 0 is the block total minus the
//! others), then λmax comes from a lane-parallel power iteration. Texels are centered on the block
//! mean first, which keeps the f32 scatter accurate.

use super::tables::{PART2, PART3};
#[cfg(target_arch = "x86_64")]
use crate::simd::Level;
use crate::simd::{LANES, SimdF32, SimdI32, f32x8};
use crate::surface::Block;

/// Second-moment slots: products (a, b) with a <= b over up to 4 channels.
const PAIRS: [(usize, usize); 10] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
];

#[inline(always)]
const fn pair(a: usize, b: usize) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    match (a, b) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (0, 3) => 3,
        (1, 1) => 4,
        (1, 2) => 5,
        (1, 3) => 6,
        (2, 2) => 7,
        (2, 3) => 8,
        _ => 9,
    }
}

/// Centered texels and their products, shared by every batch of one block.
struct Prep {
    x: [[f32; 16]; 4],
    xx: [[f32; 16]; 10],
    sum_x: [f32; 4],
    sum_xx: [f32; 10],
}

fn prep(px: &Block, dims: usize) -> Prep {
    let mut mean = [0f32; 4];
    for t in px {
        for c in 0..dims {
            mean[c] += t[c] as f32;
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }
    let mut p = Prep {
        x: [[0.0; 16]; 4],
        xx: [[0.0; 16]; 10],
        sum_x: [0.0; 4],
        sum_xx: [0.0; 10],
    };
    for (i, t) in px.iter().enumerate() {
        for c in 0..dims {
            p.x[c][i] = t[c] as f32 - mean[c];
            p.sum_x[c] += p.x[c][i];
        }
        for (k, &(a, b)) in PAIRS.iter().enumerate() {
            if a < dims && b < dims {
                p.xx[k][i] = p.x[a][i] * p.x[b][i];
                p.sum_xx[k] += p.xx[k][i];
            }
        }
    }
    p
}

/// `m · v` for a D×D matrix of lanes. (Kernels avoid closures and `array::from_fn`: closures do
/// not inherit `#[target_feature]`, so AVX2 intrinsics inside them become function calls.)
#[inline(always)]
fn matvec<V: SimdF32, const D: usize>(m: &[[V; D]; D], v: &[V; D]) -> [V; D] {
    let mut out = [V::splat(0.0); D];
    for a in 0..D {
        let mut acc = m[a][0] * v[0];
        for b in 1..D {
            acc += m[a][b] * v[b];
        }
        out[a] = acc;
    }
    out
}

/// (`trace(S) - λmax(S)`, `λmax(S)`) per lane for subsets with count `n`, sums `s` and second
/// moments `q`.
#[inline(always)]
fn residual<V: SimdF32, const D: usize>(n: V, s: &[V; 4], q: &[V; 10]) -> (V, V) {
    let zero = V::splat(0.0);
    let has = zero.simd_lt(n);
    let inv_n = V::select(has, V::splat(1.0) / n, zero);
    let mut m = [[zero; D]; D];
    for a in 0..D {
        for b in a..D {
            let v = q[pair(a, b)] - s[a] * s[b] * inv_n;
            m[a][b] = v;
            m[b][a] = v;
        }
    }
    let mut trace = zero;
    for a in 0..D {
        trace += m[a][a];
    }
    // Power iteration from the column with the largest diagonal entry (never orthogonal to the
    // dominant eigenvector unless that entry is zero).
    let mut v = m[0];
    let mut big = m[0][0];
    for a in 1..D {
        let more = big.simd_lt(m[a][a]);
        for b in 0..D {
            v[b] = V::select(more, m[a][b], v[b]);
        }
        big = big.simd_max(m[a][a]);
    }
    for _ in 0..8 {
        let w = matvec(&m, &v);
        let mut norm2 = zero;
        for a in 0..D {
            norm2 += w[a] * w[a];
        }
        let inv = V::select(zero.simd_lt(norm2), V::splat(1.0) / norm2.simd_sqrt(), zero);
        for a in 0..D {
            v[a] = w[a] * inv;
        }
    }
    // Rayleigh quotient on the (unit or zero) vector.
    let w = matvec(&m, &v);
    let mut lambda = zero;
    for a in 0..D {
        lambda += v[a] * w[a];
    }
    ((trace - lambda).simd_max(zero), lambda.simd_max(zero))
}

/// Residuals of 8 partitions with `ns` subsets; `codes` are their PART2/PART3 entries.
#[inline(always)]
fn residuals8<V: SimdF32, const D: usize>(p: &Prep, codes: [u32; LANES], ns: u8) -> ([f32; LANES], [f32; LANES]) {
    let zero = V::splat(0.0);
    let codes = V::Int::from_array(codes.map(|c| c as i32));
    let one = V::Int::splat(1);
    // Accumulators for subsets 1 and 2 (subset 0 is the total minus these).
    let mut n = [zero; 2];
    let mut s = [[zero; 4]; 2];
    let mut q = [[zero; 10]; 2];
    for i in 0..16 {
        let sel: [V; 2] = if ns == 2 {
            [((codes >> i as i32) & one).to_float(), zero]
        } else {
            let c = (codes >> (2 * i) as i32) & V::Int::splat(3);
            [(c & one).to_float(), (c >> 1).to_float()]
        };
        for k in 0..(ns as usize - 1) {
            n[k] += sel[k];
            for c in 0..D {
                s[k][c] += sel[k] * V::splat(p.x[c][i]);
            }
            for (j, &(a, b)) in PAIRS.iter().enumerate() {
                if a < D && b < D {
                    q[k][j] += sel[k] * V::splat(p.xx[j][i]);
                }
            }
        }
    }
    let mut n0 = V::splat(16.0);
    let mut s0 = [zero; 4];
    for c in 0..4 {
        s0[c] = V::splat(p.sum_x[c]);
    }
    let mut q0 = [zero; 10];
    for j in 0..10 {
        q0[j] = V::splat(p.sum_xx[j]);
    }
    let (mut total, mut lambdas) = (zero, zero);
    for k in 0..(ns as usize - 1) {
        n0 = n0 - n[k];
        for c in 0..D {
            s0[c] = s0[c] - s[k][c];
        }
        for j in 0..10 {
            q0[j] = q0[j] - q[k][j];
        }
        let (r, l) = residual::<V, D>(n[k], &s[k], &q[k]);
        total += r;
        lambdas += l;
    }
    let (r, l) = residual::<V, D>(n0, &s0, &q0);
    ((total + r).to_array(), (lambdas + l).to_array())
}

#[inline(always)]
fn residuals_all<V: SimdF32, const D: usize>(px: &Block, ns: u8, count: u8, out: &mut Fits) {
    let p = prep(px, D);
    let table = if ns == 2 { &PART2 } else { &PART3 };
    for base in (0..count as usize).step_by(LANES) {
        let codes: [u32; LANES] = std::array::from_fn(|l| table[base + l]);
        let (r, l) = residuals8::<V, D>(&p, codes, ns);
        out.residual[base..base + LANES].copy_from_slice(&r);
        out.lambda[base..base + LANES].copy_from_slice(&l);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn residuals_avx2(px: &Block, ns: u8, dims: usize, count: u8, out: &mut Fits) {
    use crate::simd::avx2::F;
    if dims == 4 {
        residuals_all::<F, 4>(px, ns, count, out);
    } else {
        residuals_all::<F, 3>(px, ns, count, out);
    }
}

fn residuals_generic(px: &Block, ns: u8, dims: usize, count: u8, out: &mut Fits) {
    if dims == 4 {
        residuals_all::<f32x8, 4>(px, ns, count, out);
    } else {
        residuals_all::<f32x8, 3>(px, ns, count, out);
    }
}

/// Line fits of every partition of one kind.
pub(super) struct Fits {
    /// Σ over subsets of the squared distances to the subset's best-fit line: a lower bound on
    /// any encoding's error.
    pub residual: [f32; 64],
    /// Σ over subsets of λmax, the scatter along the line. With `Δw` the index step of a mode
    /// (1/3 for 2-bit indices, 1/7 for 3-bit), `Δw² · λ` estimates its index quantization error,
    /// which the residual misses entirely (on grayscale blocks every residual is ~0).
    pub lambda: [f32; 64],
}

/// Line fits of the first `count` (a multiple of 8, at most 64) partitions with `ns` subsets
/// (2 or 3), over `dims` channels (3 = RGB, 4 = RGBA).
pub(super) fn residuals(px: &Block, ns: u8, dims: usize, count: u8) -> Fits {
    debug_assert!((count as usize).is_multiple_of(LANES) && count <= 64 && (ns == 2 || ns == 3));
    let mut out = Fits {
        residual: [f32::INFINITY; 64],
        lambda: [f32::INFINITY; 64],
    };
    match crate::simd::level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { residuals_avx2(px, ns, dims, count, &mut out) },
        _ => residuals_generic(px, ns, dims, count, &mut out),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{V4, line_fit_error};

    #[test]
    fn matches_scalar_line_fit() {
        for img in [
            crate::synth::foliage(32),
            crate::synth::roughness(64),
            crate::synth::gradient(32),
        ] {
            for (bi, px) in img.blocks().iter().enumerate() {
                for (ns, dims) in [(2u8, 3usize), (3, 3), (2, 4)] {
                    let fast = residuals(px, ns, dims, 64).residual;
                    for part in 0..64u8 {
                        let mut slow = 0.0;
                        for s in 0..ns as usize {
                            let pts: Vec<V4> = (0..16)
                                .filter(|&i| super::super::subset_of(ns, part, i) == s)
                                .map(|i| px[i].map(|v| v as f32))
                                .collect();
                            slow += line_fit_error(&pts, dims);
                        }
                        let tol = 1.0 + 2e-3 * slow;
                        assert!(
                            (fast[part as usize] - slow).abs() <= tol,
                            "block {bi} ns {ns} dims {dims} part {part}: simd {} scalar {slow}",
                            fast[part as usize]
                        );
                    }
                }
            }
        }
    }
}
