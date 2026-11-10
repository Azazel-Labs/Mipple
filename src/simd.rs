//! 8-lane SIMD on stable Rust, with runtime selection of the instruction set.
//!
//! Kernels are written once, generic over [`SimdF32`] (and its integer and mask companions), and
//! instantiated per backend:
//!
//! * [`f32x8`] / [`i32x8`] / [`mask32x8`]: portable arrays. Used by the generic instantiation
//!   (SSE2 baseline on x86-64, NEON on aarch64) and as the reference in tests.
//! * [`avx2::F`] / [`avx2::I`] / [`avx2::M`]: `__m256` wrappers over explicit AVX2 intrinsics.
//!   Only valid inside a `#[target_feature(enable = "avx2,fma")]` function; every method is
//!   `#[inline(always)]`, so the intrinsics land in that function. Dispatch on [`level`].
//!
//! Method names follow `std::simd` (`splat`, `simd_lt`, `simd_min`, `simd_clamp`, ...), so moving
//! to `std::simd` once `portable_simd` is stable is mostly a rename. Differences: `select` is
//! `V::select(mask, t, f)` rather than `mask.select(t, f)`, float-to-int conversion is
//! `to_int_trunc` (`to_int_unchecked`: callers keep values in range), and lookups use
//! `lookup17` (in-register permutes instead of hardware gathers, which are slow on many CPUs).
//!
//! Both backends perform the same IEEE operations in the same order (min/max argument order
//! included), and Rust never contracts `a * b + c` into an FMA, so results are bit-identical
//! across backends. There is deliberately no `mul_add`: without FMA hardware it is a libm call.

use std::ops::{Add, AddAssign, BitAnd, BitOr, Div, Mul, Shl, Shr, Sub};
use std::sync::OnceLock;

/// Lane count of every vector type here.
pub const LANES: usize = 8;

/// Lane order of [`SimdI32::load_block_row`]: lane `l` holds block `ROW_ORDER[l]` of the eight.
pub const ROW_ORDER: [usize; LANES] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Instruction sets a kernel can be instantiated for, best last.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// The build's baseline (SSE2 on x86-64, NEON on aarch64).
    Generic,
    /// x86-64 with AVX2 and FMA.
    Avx2,
}

/// The best instruction set this CPU supports, detected once. Setting the environment variable
/// `MIPPLE_SIMD=generic` forces the baseline path (for A/B timing and tests).
pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        if std::env::var("MIPPLE_SIMD").is_ok_and(|v| v.eq_ignore_ascii_case("generic")) {
            return Level::Generic;
        }
        detect()
    })
}

fn detect() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Level::Avx2;
        }
    }
    Level::Generic
}

/// Eight `f32` lanes.
pub trait SimdF32:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + AddAssign
{
    /// Lane-wise IEEE square root (`std::simd::StdFloat::sqrt`).
    fn simd_sqrt(self) -> Self;
    fn from_array(a: [f32; LANES]) -> Self;
    /// Mask from per-lane flags (`Mask::from_array`).
    fn mask_from_bools(b: [bool; LANES]) -> Self::Mask;
    /// Per-lane flags of a mask (`Mask::to_array`).
    fn mask_to_bools(m: Self::Mask) -> [bool; LANES];
    /// Any lane set (`Mask::any`).
    #[inline(always)]
    fn mask_any(m: Self::Mask) -> bool {
        Self::mask_to_bools(m).iter().any(|&b| b)
    }
    type Int: SimdI32<Float = Self, Mask = Self::Mask>;
    type Mask: Copy + BitAnd<Output = Self::Mask> + BitOr<Output = Self::Mask>;

    fn splat(v: f32) -> Self;
    /// Loads the first 8 elements (panics if `s` is shorter).
    fn from_slice(s: &[f32]) -> Self;
    fn to_array(self) -> [f32; LANES];
    /// Per lane `if o < self { o } else { self }`.
    fn simd_min(self, o: Self) -> Self;
    /// Per lane `if o > self { o } else { self }`.
    fn simd_max(self, o: Self) -> Self;
    fn simd_lt(self, o: Self) -> Self::Mask;
    /// Per lane: `t` where `m` is set, else `f`.
    fn select(m: Self::Mask, t: Self, f: Self) -> Self;
    /// Truncating conversion; lanes must be in `i32` range.
    fn to_int_trunc(self) -> Self::Int;
    /// `[row[idx[0]], ..., row[idx[7]]]` for indices in `0..=16` (17-entry tables such as
    /// prefix sums). AVX2 does it in registers with two `vpermps` and two blends.
    fn lookup17(row: &[f32; 32], idx: Self::Int) -> Self;

    /// Fused `self * a + b` with a single rounding (`f32::mul_add`; `vfmadd` on AVX2), so both
    /// backends agree bit for bit.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Lane-wise `f32::floor` (`vroundps` on AVX2).
    fn floor(self) -> Self;

    #[inline(always)]
    fn simd_gt(self, o: Self) -> Self::Mask {
        o.simd_lt(self)
    }

    #[inline(always)]
    fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }
}

/// Eight `i32` lanes.
pub trait SimdI32:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Shl<i32, Output = Self>
    + Shr<i32, Output = Self>
{
    type Float: SimdF32<Int = Self, Mask = Self::Mask>;
    type Mask: Copy;

    fn splat(v: i32) -> Self;
    fn from_array(a: [i32; LANES]) -> Self;
    /// Loads the first 8 elements (panics if `s` is shorter).
    fn from_slice(s: &[i32]) -> Self;
    fn to_array(self) -> [i32; LANES];
    fn simd_lt(self, o: Self) -> Self::Mask;
    fn select(m: Self::Mask, t: Self, f: Self) -> Self;
    fn to_float(self) -> Self::Float;
    /// Lane-wise signed minimum.
    fn simd_min(self, o: Self) -> Self;
    /// Lane-wise signed maximum.
    fn simd_max(self, o: Self) -> Self;
    /// One texel row of eight side-by-side 4x4 blocks: `px` is 32 consecutive RGBA8 texels.
    /// Returns column j (texel x = 4b + j of block b) for j in 0..4, as little-endian `u32`
    /// texels, with blocks in lane order [`ROW_ORDER`] (a 4x4 transpose per 128-bit half).
    fn load_block_row(px: &[[u8; 4]; 32]) -> [Self; 4];
    /// `table[idx]` per lane for indices in `0..16` (one `vpshufb` on AVX2). `table[0]` must be 0:
    /// the upper bytes of each lane look up entry 0.
    fn lookup16(table: &[u8; 16], idx: Self) -> Self;
    /// `out[i] = self[idx[i]]` (`simd_swizzle!` with a runtime index; one `vpermd` on AVX2).
    fn permute(self, idx: [i32; LANES]) -> Self;
    /// `(a0 b0 a1 b1 a2 b2 a3 b3, a4 b4 ... a7 b7)` (`Simd::interleave`).
    fn interleave(self, o: Self) -> (Self, Self);
    /// The lanes as little-endian bytes (`Simd::to_le_bytes`).
    fn to_le_bytes(self) -> [u8; 4 * LANES];
}

// ---------------------------------------------------------------------------------------------
// Portable backend
// ---------------------------------------------------------------------------------------------

/// Eight `f32` lanes as an array (portable backend).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(32))]
pub struct f32x8(pub [f32; LANES]);

/// Eight `i32` lanes as an array (portable backend).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct i32x8(pub [i32; LANES]);

/// Eight lane flags (portable backend).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct mask32x8(pub [bool; LANES]);

macro_rules! array_binop {
    ($ty:ident, $trait:ident, $method:ident, $op:tt) => {
        impl $trait for $ty {
            type Output = $ty;
            #[inline(always)]
            fn $method(self, rhs: $ty) -> $ty {
                $ty(std::array::from_fn(|i| self.0[i] $op rhs.0[i]))
            }
        }
    };
}

array_binop!(f32x8, Add, add, +);
array_binop!(f32x8, Sub, sub, -);
array_binop!(f32x8, Mul, mul, *);
array_binop!(f32x8, Div, div, /);
array_binop!(i32x8, Add, add, +);
array_binop!(i32x8, Sub, sub, -);
array_binop!(i32x8, BitAnd, bitand, &);
array_binop!(i32x8, BitOr, bitor, |);
array_binop!(mask32x8, BitAnd, bitand, &);
array_binop!(mask32x8, BitOr, bitor, |);

impl AddAssign for f32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32x8) {
        *self = *self + rhs;
    }
}

impl Shl<i32> for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn shl(self, n: i32) -> i32x8 {
        i32x8(self.0.map(|v| v << n))
    }
}

impl Shr<i32> for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn shr(self, n: i32) -> i32x8 {
        i32x8(self.0.map(|v| v >> n))
    }
}

impl SimdF32 for f32x8 {
    type Int = i32x8;
    type Mask = mask32x8;

    #[inline(always)]
    fn splat(v: f32) -> Self {
        Self([v; LANES])
    }
    #[inline(always)]
    fn from_slice(s: &[f32]) -> Self {
        Self(s[..LANES].try_into().unwrap())
    }
    #[inline(always)]
    fn to_array(self) -> [f32; LANES] {
        self.0
    }
    #[inline(always)]
    fn simd_min(self, o: Self) -> Self {
        Self(std::array::from_fn(
            |i| {
                if o.0[i] < self.0[i] { o.0[i] } else { self.0[i] }
            },
        ))
    }
    #[inline(always)]
    fn simd_max(self, o: Self) -> Self {
        Self(std::array::from_fn(
            |i| {
                if o.0[i] > self.0[i] { o.0[i] } else { self.0[i] }
            },
        ))
    }
    #[inline(always)]
    fn simd_lt(self, o: Self) -> mask32x8 {
        mask32x8(std::array::from_fn(|i| self.0[i] < o.0[i]))
    }
    #[inline(always)]
    fn select(m: mask32x8, t: Self, f: Self) -> Self {
        Self(std::array::from_fn(|i| if m.0[i] { t.0[i] } else { f.0[i] }))
    }
    #[inline(always)]
    fn to_int_trunc(self) -> i32x8 {
        // SAFETY: the trait contract requires lanes in i32 range (and not NaN).
        i32x8(self.0.map(|v| unsafe { v.to_int_unchecked::<i32>() }))
    }
    #[inline(always)]
    fn lookup17(row: &[f32; 32], idx: i32x8) -> Self {
        Self(std::array::from_fn(|l| row[(idx.0[l] & 31) as usize]))
    }
    #[inline(always)]
    fn simd_sqrt(self) -> Self {
        Self(self.0.map(f32::sqrt))
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(fma(self.0, a.0, b.0))
    }
    #[inline(always)]
    fn floor(self) -> Self {
        Self(self.0.map(f32::floor))
    }
    #[inline(always)]
    fn from_array(a: [f32; LANES]) -> Self {
        Self(a)
    }
    #[inline(always)]
    fn mask_from_bools(b: [bool; LANES]) -> mask32x8 {
        mask32x8(b)
    }
    #[inline(always)]
    fn mask_to_bools(m: mask32x8) -> [bool; LANES] {
        m.0
    }
}

impl SimdI32 for i32x8 {
    type Float = f32x8;
    type Mask = mask32x8;

    #[inline(always)]
    fn splat(v: i32) -> Self {
        Self([v; LANES])
    }
    #[inline(always)]
    fn from_array(a: [i32; LANES]) -> Self {
        Self(a)
    }
    #[inline(always)]
    fn from_slice(s: &[i32]) -> Self {
        Self(s[..LANES].try_into().unwrap())
    }
    #[inline(always)]
    fn to_array(self) -> [i32; LANES] {
        self.0
    }
    #[inline(always)]
    fn simd_lt(self, o: Self) -> mask32x8 {
        mask32x8(std::array::from_fn(|i| self.0[i] < o.0[i]))
    }
    #[inline(always)]
    fn select(m: mask32x8, t: Self, f: Self) -> Self {
        Self(std::array::from_fn(|i| if m.0[i] { t.0[i] } else { f.0[i] }))
    }
    #[inline(always)]
    fn lookup16(table: &[u8; 16], idx: Self) -> Self {
        debug_assert_eq!(table[0], 0);
        i32x8(idx.0.map(|i| table[i as usize & 15] as i32))
    }
    #[inline(always)]
    fn permute(self, idx: [i32; LANES]) -> Self {
        i32x8(idx.map(|i| self.0[i as usize & 7]))
    }
    #[inline(always)]
    fn interleave(self, o: Self) -> (Self, Self) {
        let (a, b) = (self.0, o.0);
        (
            i32x8([a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]]),
            i32x8([a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7]]),
        )
    }
    #[inline(always)]
    fn to_le_bytes(self) -> [u8; 4 * LANES] {
        let mut out = [0u8; 4 * LANES];
        for (o, v) in out.chunks_exact_mut(4).zip(self.0) {
            o.copy_from_slice(&v.to_le_bytes());
        }
        out
    }
    #[inline(always)]
    fn load_block_row(px: &[[u8; 4]; 32]) -> [Self; 4] {
        let mut out = [i32x8([0; LANES]); 4];
        for (j, col) in out.iter_mut().enumerate() {
            for (l, v) in col.0.iter_mut().enumerate() {
                *v = i32::from_le_bytes(px[4 * ROW_ORDER[l] + j]);
            }
        }
        out
    }
    #[inline(always)]
    fn to_float(self) -> f32x8 {
        f32x8(self.0.map(|v| v as f32))
    }
    #[inline(always)]
    fn simd_min(self, o: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].min(o.0[i])))
    }
    #[inline(always)]
    fn simd_max(self, o: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].max(o.0[i])))
    }
}

/// Lane-wise `a * b + c` with a single rounding: `f32::mul_add`.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn fma(a: [f32; LANES], b: [f32; LANES], c: [f32; LANES]) -> [f32; LANES] {
    std::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
}

/// Lane-wise `a * b + c` with a single rounding. WebAssembly has no FMA instruction, so
/// `f32::mul_add` is an out-of-line libm `fmaf` call per lane (most of a BC7 RDO encode's time in
/// the browser demo). This is libm's algorithm (`fma_wide_round`, from musl) over the whole vector:
/// the f64 product is exact and the f64 sum is correctly rounded, so narrowing it only
/// double-rounds when the sum lands exactly halfway between two f32s, which [`fma_fixup`] corrects.
/// The common path is branch-free, so it vectorizes.
#[cfg(any(target_arch = "wasm32", test))]
#[inline(always)]
fn fma_wide(a: [f32; LANES], b: [f32; LANES], c: [f32; LANES]) -> [f32; LANES] {
    let mut out = [0.0; LANES];
    let mut halfway = false;
    for i in 0..LANES {
        let r = f64::from(a[i]) * f64::from(b[i]) + f64::from(c[i]);
        halfway |= r.to_bits() & 0x1fff_ffff == 0x1000_0000;
        out[i] = r as f32;
    }
    if halfway { fma_fixup(a, b, c) } else { out }
}

#[cfg(target_arch = "wasm32")]
use fma_wide as fma;

#[cfg(any(target_arch = "wasm32", test))]
#[cold]
#[inline(never)]
fn fma_fixup(a: [f32; LANES], b: [f32; LANES], c: [f32; LANES]) -> [f32; LANES] {
    std::array::from_fn(|i| {
        let xy = f64::from(a[i]) * f64::from(b[i]);
        let r = xy + f64::from(c[i]);
        if r.to_bits() & 0x1fff_ffff == 0x1000_0000 {
            fma_halfway(xy, f64::from(c[i]), r)
        } else {
            r as f32
        }
    })
}

#[cfg(any(target_arch = "wasm32", test))]
fn fma_halfway(xy: f64, z: f64, r: f64) -> f32 {
    if r.is_nan() || (r - xy == z && r - z == xy) {
        return r as f32; // exact: no double rounding
    }
    // Nudge the f64 off the halfway point toward the true value, then narrow.
    let bits = r.to_bits();
    let neg = r.is_sign_negative();
    let err = if neg == (z > xy) { xy - r + z } else { z - r + xy };
    f64::from_bits(if neg == (err < 0.0) { bits + 1 } else { bits - 1 }) as f32
}

// ---------------------------------------------------------------------------------------------
// AVX2 backend
// ---------------------------------------------------------------------------------------------

/// `__m256` wrappers over AVX2 intrinsics. Only use these inside functions compiled with
/// `#[target_feature(enable = "avx2,fma")]` and reached through [`level`] == `Avx2`.
#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::{LANES, SimdF32, SimdI32};
    use std::arch::x86_64::*;
    use std::ops::{Add, AddAssign, BitAnd, BitOr, Div, Mul, Shl, Shr, Sub};

    /// Eight `f32` lanes in a `__m256`.
    #[derive(Clone, Copy)]
    pub struct F(__m256);
    /// Eight `i32` lanes in a `__m256i`.
    #[derive(Clone, Copy)]
    pub struct I(__m256i);
    /// Eight lane masks (all-ones / all-zeros) in a `__m256`.
    #[derive(Clone, Copy)]
    pub struct M(__m256);

    // SAFETY (all `unsafe` blocks below): these types are only used inside AVX2+FMA functions
    // selected at runtime by `simd::level()`; slice loads check the length first.

    macro_rules! binop {
        ($ty:ident, $trait:ident, $method:ident, $intr:ident) => {
            impl $trait for $ty {
                type Output = $ty;
                #[inline(always)]
                fn $method(self, rhs: $ty) -> $ty {
                    $ty(unsafe { $intr(self.0, rhs.0) })
                }
            }
        };
    }

    binop!(F, Add, add, _mm256_add_ps);
    binop!(F, Sub, sub, _mm256_sub_ps);
    binop!(F, Mul, mul, _mm256_mul_ps);
    binop!(F, Div, div, _mm256_div_ps);
    binop!(I, Add, add, _mm256_add_epi32);
    binop!(I, Sub, sub, _mm256_sub_epi32);
    binop!(I, BitAnd, bitand, _mm256_and_si256);
    binop!(I, BitOr, bitor, _mm256_or_si256);
    binop!(M, BitAnd, bitand, _mm256_and_ps);
    binop!(M, BitOr, bitor, _mm256_or_ps);

    impl AddAssign for F {
        #[inline(always)]
        fn add_assign(&mut self, rhs: F) {
            *self = *self + rhs;
        }
    }

    impl Shl<i32> for I {
        type Output = I;
        #[inline(always)]
        fn shl(self, n: i32) -> I {
            I(unsafe { _mm256_sll_epi32(self.0, _mm_cvtsi32_si128(n)) })
        }
    }

    impl Shr<i32> for I {
        type Output = I;
        /// Arithmetic shift, like `i32 >>`.
        #[inline(always)]
        fn shr(self, n: i32) -> I {
            I(unsafe { _mm256_sra_epi32(self.0, _mm_cvtsi32_si128(n)) })
        }
    }

    impl SimdF32 for F {
        type Int = I;
        type Mask = M;

        #[inline(always)]
        fn splat(v: f32) -> Self {
            F(unsafe { _mm256_set1_ps(v) })
        }
        #[inline(always)]
        fn from_slice(s: &[f32]) -> Self {
            assert!(s.len() >= LANES);
            F(unsafe { _mm256_loadu_ps(s.as_ptr()) })
        }
        #[inline(always)]
        fn to_array(self) -> [f32; LANES] {
            let mut a = [0f32; LANES];
            unsafe { _mm256_storeu_ps(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline(always)]
        fn simd_min(self, o: Self) -> Self {
            // minps(a, b) = a < b ? a : b, matching the portable `if o < self { o } else { self }`.
            F(unsafe { _mm256_min_ps(o.0, self.0) })
        }
        #[inline(always)]
        fn simd_max(self, o: Self) -> Self {
            F(unsafe { _mm256_max_ps(o.0, self.0) })
        }
        #[inline(always)]
        fn simd_lt(self, o: Self) -> M {
            M(unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, o.0) })
        }
        #[inline(always)]
        fn select(m: M, t: Self, f: Self) -> Self {
            F(unsafe { _mm256_blendv_ps(f.0, t.0, m.0) })
        }
        #[inline(always)]
        fn to_int_trunc(self) -> I {
            I(unsafe { _mm256_cvttps_epi32(self.0) })
        }
        #[inline(always)]
        fn simd_sqrt(self) -> Self {
            F(unsafe { _mm256_sqrt_ps(self.0) })
        }
        #[inline(always)]
        fn mul_add(self, a: Self, b: Self) -> Self {
            F(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
        }
        #[inline(always)]
        fn floor(self) -> Self {
            F(unsafe { _mm256_floor_ps(self.0) })
        }
        #[inline(always)]
        fn from_array(a: [f32; LANES]) -> Self {
            F(unsafe { _mm256_loadu_ps(a.as_ptr()) })
        }
        #[inline(always)]
        fn mask_from_bools(b: [bool; LANES]) -> M {
            let v = b.map(|x| -(x as i32));
            M(unsafe { _mm256_castsi256_ps(_mm256_loadu_si256(v.as_ptr().cast::<__m256i>())) })
        }
        #[inline(always)]
        fn mask_to_bools(m: M) -> [bool; LANES] {
            let bits = unsafe { _mm256_movemask_ps(m.0) };
            std::array::from_fn(|l| bits >> l & 1 != 0)
        }
        #[inline(always)]
        fn mask_any(m: M) -> bool {
            unsafe { _mm256_movemask_ps(m.0) != 0 }
        }
        #[inline(always)]
        fn lookup17(row: &[f32; 32], idx: I) -> Self {
            F(unsafe {
                // The table loads are loop-invariant in callers and get hoisted.
                let lo = _mm256_loadu_ps(row.as_ptr());
                let hi = _mm256_loadu_ps(row.as_ptr().add(8));
                let top = _mm256_set1_ps(row[16]);
                let a = _mm256_permutevar8x32_ps(lo, idx.0); // uses the low 3 index bits
                let b = _mm256_permutevar8x32_ps(hi, idx.0);
                let ge8 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx.0, _mm256_set1_epi32(7)));
                let is16 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idx.0, _mm256_set1_epi32(16)));
                _mm256_blendv_ps(_mm256_blendv_ps(a, b, ge8), top, is16)
            })
        }
    }

    impl SimdI32 for I {
        type Float = F;
        type Mask = M;

        #[inline(always)]
        fn splat(v: i32) -> Self {
            I(unsafe { _mm256_set1_epi32(v) })
        }
        #[inline(always)]
        fn from_array(a: [i32; LANES]) -> Self {
            I(unsafe { _mm256_loadu_si256(a.as_ptr().cast::<__m256i>()) })
        }
        #[inline(always)]
        fn from_slice(s: &[i32]) -> Self {
            assert!(s.len() >= LANES);
            I(unsafe { _mm256_loadu_si256(s.as_ptr().cast::<__m256i>()) })
        }
        #[inline(always)]
        fn to_array(self) -> [i32; LANES] {
            let mut a = [0i32; LANES];
            unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), self.0) };
            a
        }
        #[inline(always)]
        fn simd_lt(self, o: Self) -> M {
            M(unsafe { _mm256_castsi256_ps(_mm256_cmpgt_epi32(o.0, self.0)) })
        }
        #[inline(always)]
        fn select(m: M, t: Self, f: Self) -> Self {
            I(unsafe {
                _mm256_castps_si256(_mm256_blendv_ps(
                    _mm256_castsi256_ps(f.0),
                    _mm256_castsi256_ps(t.0),
                    m.0,
                ))
            })
        }
        #[inline(always)]
        fn to_float(self) -> F {
            F(unsafe { _mm256_cvtepi32_ps(self.0) })
        }
        #[inline(always)]
        fn simd_min(self, o: Self) -> Self {
            I(unsafe { _mm256_min_epi32(self.0, o.0) })
        }
        #[inline(always)]
        fn simd_max(self, o: Self) -> Self {
            I(unsafe { _mm256_max_epi32(self.0, o.0) })
        }
        #[inline(always)]
        fn lookup16(table: &[u8; 16], idx: Self) -> Self {
            // Indices below 16 leave the upper three bytes of each lane zero; those look up
            // table[0], which callers keep at 0.
            unsafe {
                let t = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast::<__m128i>()));
                I(_mm256_shuffle_epi8(t, idx.0))
            }
        }
        #[inline(always)]
        fn permute(self, idx: [i32; LANES]) -> Self {
            unsafe {
                I(_mm256_permutevar8x32_epi32(
                    self.0,
                    _mm256_loadu_si256(idx.as_ptr().cast::<__m256i>()),
                ))
            }
        }
        #[inline(always)]
        fn interleave(self, o: Self) -> (Self, Self) {
            unsafe {
                let (lo, hi) = (_mm256_unpacklo_epi32(self.0, o.0), _mm256_unpackhi_epi32(self.0, o.0));
                (
                    I(_mm256_permute2x128_si256::<0x20>(lo, hi)),
                    I(_mm256_permute2x128_si256::<0x31>(lo, hi)),
                )
            }
        }
        #[inline(always)]
        fn to_le_bytes(self) -> [u8; 4 * LANES] {
            let mut out = [0u8; 4 * LANES];
            unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), self.0) };
            out
        }
        #[inline(always)]
        fn load_block_row(px: &[[u8; 4]; 32]) -> [Self; 4] {
            unsafe {
                let p = px.as_ptr().cast::<__m256i>();
                // r_k = blocks 2k | 2k+1, four texels each.
                let (r0, r1) = (_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1)));
                let (r2, r3) = (_mm256_loadu_si256(p.add(2)), _mm256_loadu_si256(p.add(3)));
                let (t0, t1) = (_mm256_unpacklo_epi32(r0, r1), _mm256_unpackhi_epi32(r0, r1));
                let (t2, t3) = (_mm256_unpacklo_epi32(r2, r3), _mm256_unpackhi_epi32(r2, r3));
                [
                    I(_mm256_unpacklo_epi64(t0, t2)),
                    I(_mm256_unpackhi_epi64(t0, t2)),
                    I(_mm256_unpacklo_epi64(t1, t3)),
                    I(_mm256_unpackhi_epi64(t1, t3)),
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small kernel run on both backends must give bit-identical lanes.
    #[inline(always)]
    fn kernel<V: SimdF32>(a: &[f32], b: &[f32], row: &[f32; 32]) -> ([f32; 8], [i32; 8]) {
        let (x, y) = (V::from_slice(a), V::from_slice(b));
        let m = x.simd_lt(y);
        let s = V::select(m, x * y - x, y + x).simd_clamp(V::splat(-50.0), V::splat(50.0));
        let q = (s + V::splat(60.0)).to_int_trunc();
        let r = (q << 3) | (q >> 2);
        let g = V::lookup17(row, q & V::Int::splat(15));
        (
            (s + g + r.to_float()).to_array(),
            V::Int::select(q.simd_lt(V::Int::splat(70)), q, r).to_array(),
        )
    }

    /// The wasm32 `mul_add` (inlined f64 fma) agrees with `f32::mul_add` bit for bit, including
    /// sums that land halfway between two f32s in f64 (`a * b` tiny against `c`).
    #[test]
    fn wide_fma_matches_mul_add() {
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut halfway = 0;
        let (mut va, mut vb, mut vc) = ([0.0; LANES], [0.0; LANES], [0.0; LANES]);
        for i in 0..2_000_000 {
            let r = next();
            let (a, b, c) = match i % 3 {
                // Values like the kernels': texel-scale products and sums.
                0 => (
                    (r & 0xffff) as f32 / 97.0 - 300.0,
                    ((r >> 16) & 0xffff) as f32 / 211.0 - 150.0,
                    ((r >> 32) & 0xffff) as f32 * 3.5 - 1e5,
                ),
                // Arbitrary finite bit patterns.
                1 => {
                    let f = |x: u64| f32::from_bits(x as u32 & 0xbfff_ffff);
                    (f(r), f(r >> 21), f(r.rotate_left(22)))
                }
                // c = 1 + k ulp and a * b just under half an f32 ulp: f64 sums at or near halfway.
                _ => {
                    let c = 1.0 + (r & 0xff) as f32 * f32::EPSILON;
                    let a = f32::EPSILON / 2.0 * (1.0 + ((r >> 8) & 0x3) as f32 * f32::EPSILON);
                    let b = if r >> 63 == 0 { 1.0 - f32::EPSILON / 2.0 } else { -1.0 };
                    (a, b, c)
                }
            };
            let xy = f64::from(a) * f64::from(b);
            halfway += i32::from((xy + f64::from(c)).to_bits() & 0x1fff_ffff == 0x1000_0000);
            // Lanes mix the three kinds, so halfway lanes share vectors with ordinary ones.
            let l = i % LANES;
            (va[l], vb[l], vc[l]) = (a, b, c);
            if l + 1 < LANES {
                continue;
            }
            let w = fma_wide(va, vb, vc);
            for l in 0..LANES {
                let (a, b, c) = (va[l], vb[l], vc[l]);
                let (w, m) = (w[l], a.mul_add(b, c));
                assert!(
                    w.to_bits() == m.to_bits() || (w.is_nan() && m.is_nan()),
                    "{a:e} {b:e} {c:e}: {w:e} vs {m:e}"
                );
            }
        }
        assert!(halfway > 1000, "only {halfway} halfway cases");
    }

    #[test]
    fn block_row_loads_agree() {
        let px: [[u8; 4]; 32] = std::array::from_fn(|i| [i as u8, 100 + i as u8, 200, 7]);
        let g = i32x8::load_block_row(&px).map(super::SimdI32::to_array);
        for (j, col) in g.iter().enumerate() {
            for (l, &v) in col.iter().enumerate() {
                assert_eq!(v.to_le_bytes()[0] as usize, 4 * ROW_ORDER[l] + j);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if level() == Level::Avx2 {
            #[target_feature(enable = "avx2,fma")]
            fn run(px: &[[u8; 4]; 32]) -> [[i32; 8]; 4] {
                avx2::I::load_block_row(px).map(super::SimdI32::to_array)
            }
            // SAFETY: guarded by the level check.
            assert_eq!(unsafe { run(&px) }, g);
            let table: [u8; 16] = std::array::from_fn(|i| if i == 0 { 0 } else { (200 - 7 * i) as u8 });
            let idx = [0, 3, 15, 7, 1, 9, 12, 4];
            #[target_feature(enable = "avx2,fma")]
            fn lut(t: &[u8; 16], i: [i32; 8]) -> [i32; 8] {
                avx2::I::lookup16(t, avx2::I::from_array(i)).to_array()
            }
            // SAFETY: guarded by the level check.
            let a = unsafe { lut(&table, idx) };
            assert_eq!(a, i32x8::lookup16(&table, i32x8(idx)).0);
            assert_eq!(a[2], table[15] as i32);
            let (x, y) = (
                i32x8(std::array::from_fn(|i| i as i32)),
                i32x8(std::array::from_fn(|i| 10 + i as i32)),
            );
            let p = [3, 0, 7, 1, 6, 2, 5, 4];
            #[target_feature(enable = "avx2,fma")]
            fn ops(x: [i32; 8], y: [i32; 8], p: [i32; 8]) -> ([i32; 8], [i32; 8], [i32; 8], [u8; 32]) {
                let (x, y) = (avx2::I::from_array(x), avx2::I::from_array(y));
                let (a, b) = x.interleave(y);
                (x.permute(p).to_array(), a.to_array(), b.to_array(), y.to_le_bytes())
            }
            // SAFETY: guarded by the level check.
            let (pa, ia, ib, by) = unsafe { ops(x.0, y.0, p) };
            let (ga, gb) = x.interleave(y);
            assert_eq!((pa, ia, ib, by), (x.permute(p).0, ga.0, gb.0, y.to_le_bytes()));
            assert_eq!(ia, [0, 10, 1, 11, 2, 12, 3, 13]);
        }
    }

    #[test]
    fn backends_agree() {
        let a = [0.5, -3.0, 7.25, 100.0, -0.0, 3.5, 9.0, -20.0];
        let b = [1.0, -4.0, 7.25, 2.0, 0.0, -3.5, 10.0, 30.0];
        let row: [f32; 32] = std::array::from_fn(|i| i as f32 * 1.5);
        let portable = kernel::<f32x8>(&a, &b, &row);
        assert_eq!(
            portable.0[0],
            (0.5f32 * 1.0 - 0.5) + row[60 & 15] + ((60 << 3) | (60 >> 2)) as f32
        );
        #[cfg(target_arch = "x86_64")]
        if level() == Level::Avx2 {
            #[target_feature(enable = "avx2,fma")]
            fn run(a: &[f32], b: &[f32], row: &[f32; 32]) -> ([f32; 8], [i32; 8]) {
                kernel::<avx2::F>(a, b, row)
            }
            // SAFETY: guarded by the runtime check above.
            let fast = unsafe { run(&a, &b, &row) };
            assert_eq!(portable.0.map(f32::to_bits), fast.0.map(f32::to_bits));
            assert_eq!(portable.1, fast.1);
        }
    }
}
