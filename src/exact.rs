//! Exact integer arithmetic in f32 SIMD lanes, shared by the encoders whose GPU kernels must
//! reproduce the CPU's blocks: every value is an integer below 2^24, so results do not depend on
//! rounding, summation order or fused multiply-adds.

use crate::simd::SimdF32;

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
        $crate::exact::reduce16!(|$i| $e, |a, b| a + b)
    };
}

/// Dot product Σ `$x`·`$y` over `$i` in 0..16 in four fused multiply-add chains (exact like
/// [`sum16`] when every product and partial sum is an integer below 2^24).
macro_rules! dot16 {
    (|$i:ident| $x:expr, $y:expr) => {{
        let mut acc = [
            {
                let $i = 0usize;
                $x * $y
            },
            {
                let $i = 1usize;
                $x * $y
            },
            {
                let $i = 2usize;
                $x * $y
            },
            {
                let $i = 3usize;
                $x * $y
            },
        ];
        let mut $i = 4usize;
        while $i < 16 {
            acc[$i & 3] = ($x).mul_add($y, acc[$i & 3]);
            $i += 1;
        }
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    }};
}

pub(crate) use {dot16, reduce16, sum16};

/// Exact floor(a / d) and remainder a - q d, for integer lanes with d > 0, |a| < 2^23 and
/// `inv` = 1 / d rounded: the estimate floor(a inv) is within one of the quotient, and the
/// correction is exact since q d stays within d of a (fused or not, every product is an
/// integer below 2^24).
#[inline(always)]
pub(crate) fn div_floor<V: SimdF32>(a: V, d: V, inv: V) -> (V, V) {
    let one = V::splat(1.0);
    let q = (a * inv).floor();
    let r = a - q * d;
    let lo = r.simd_lt(V::splat(0.0));
    let (q, r) = (V::select(lo, q - one, q), V::select(lo, r + d, r));
    let hi = (d - V::splat(0.5)).simd_lt(r);
    (V::select(hi, q + one, q), V::select(hi, r - d, r))
}

/// Exact floor(a / d) for an integer lane |a| < 2^22 and `inv` = 1 / d rounded (d >= 1): the
/// quotient (a + 1/2) / d is at least 1/(2d) from an integer, and the two roundings (of 1 / d
/// and of the product) move it by under |a / d| 2^-23 < 1/(2d). One multiply, no correction.
#[inline(always)]
pub(crate) fn div_floor_q<V: SimdF32>(a: V, inv: V) -> V {
    ((a + V::splat(0.5)) * inv).floor()
}

/// Power-of-two scale 2^-s per lane, with s the least shift that brings the non-negative
/// integer `m` (below 2^24) under 2^(bits + 1): the largest s with m >= 2^(bits + s), or 0
/// (i.e. max(0, floor(log2 m) - bits)), from the exponent of max(m, 2^bits). Exact.
#[inline(always)]
pub(crate) fn shrink<V: SimdF32>(m: V, bits: i32) -> V {
    let p = V::splat((1u32 << bits) as f32);
    m.simd_max(p).pow2_recip() * p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::f32x8;

    #[test]
    fn shrink_matches_log2() {
        for bits in [6, 9, 10, 12] {
            let mut m = 0u32;
            while m < 1 << 24 {
                let want = 1.0 / (1u32 << (m.max(1).ilog2() as i32 - bits).max(0)) as f32;
                let got = shrink(f32x8::splat(m as f32), bits).to_array()[0];
                assert_eq!(got, want, "m {m} bits {bits}");
                m = m * 9 / 8 + 1;
            }
        }
    }

    #[test]
    fn div_floor_q_exact() {
        for d in [1, 2, 3, 7, 255, 4093, 8191, 195_075] {
            let inv = f32x8::splat(1.0 / d as f32);
            // Every multiple of d (and its neighbours) in range, the worst cases.
            let lim = (1i32 << 22) - 1;
            for k in -lim / d..=lim / d {
                for a in [k * d - 1, k * d, k * d + 1] {
                    if a.abs() <= lim {
                        let got = div_floor_q(f32x8::splat(a as f32), inv).to_array()[0];
                        assert_eq!(got, a.div_euclid(d) as f32, "a {a} d {d}");
                    }
                }
            }
        }
    }
}
