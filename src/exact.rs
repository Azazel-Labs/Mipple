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

pub(crate) use {reduce16, sum16};

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

/// Power-of-two scale 2^-s per lane, with s the least shift that brings the non-negative
/// integer `m` (below 2^24) under 2^(bits + 1): the largest s with m >= 2^(bits + s), or 0
/// (i.e. max(0, floor(log2 m) - bits)), found by halving steps. Exact: every product is a
/// power-of-two scaling.
#[inline(always)]
pub(crate) fn shrink<V: SimdF32>(m: V, bits: i32) -> V {
    let mut ms = m;
    let mut scale = V::splat(1.0);
    for k in [16, 8, 4, 2, 1] {
        let small = ms.simd_lt(V::splat((1u64 << (bits + k)) as f32));
        let f = V::splat(1.0 / (1u32 << k) as f32);
        ms = V::select(small, ms, ms * f);
        scale = V::select(small, scale, scale * f);
    }
    scale
}
