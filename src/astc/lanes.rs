//! Eight i16 lanes for the encoder's integer error kernels and four f32 lanes for its separable
//! transforms: SSE2 (x86-64's baseline) there, plain arrays with the same arithmetic elsewhere.
//! Each f32 lane rounds exactly as the scalar code would (no fused multiply-add).

// SAFETY (every `unsafe` block in `imp` for x86-64): SSE2 is part of the x86-64 baseline, and
// the loads and stores are unaligned 16-byte accesses of 16-byte arrays.
#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;

    /// Eight i16 lanes.
    #[derive(Clone, Copy)]
    pub struct I16x8(__m128i);

    /// Four i32 lanes.
    #[derive(Clone, Copy)]
    pub struct I32x4(__m128i);

    impl I16x8 {
        #[inline]
        pub fn splat(v: i16) -> Self {
            Self(unsafe { _mm_set1_epi16(v) })
        }

        #[inline]
        pub fn from_array(v: [i16; 8]) -> Self {
            // SAFETY: 16 bytes, unaligned load.
            Self(unsafe { _mm_loadu_si128(v.as_ptr().cast()) })
        }

        /// The first eight values of `v`.
        #[inline]
        pub fn load(v: &[i16]) -> Self {
            let v: &[i16; 8] = v[..8].try_into().unwrap();
            Self::from_array(*v)
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi16(self.0, o.0) })
        }

        #[inline]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm_sub_epi16(self.0, o.0) })
        }

        #[inline]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm_mullo_epi16(self.0, o.0) })
        }

        #[inline]
        pub fn and(self, o: Self) -> Self {
            Self(unsafe { _mm_and_si128(self.0, o.0) })
        }

        /// Arithmetic shift right by `N`.
        #[inline]
        pub fn shr<const N: i32>(self) -> Self {
            Self(unsafe { _mm_srai_epi16::<N>(self.0) })
        }

        /// The squares, summed in adjacent pairs.
        #[inline]
        pub fn sq_pairs(self) -> I32x4 {
            I32x4(unsafe { _mm_madd_epi16(self.0, self.0) })
        }
    }

    impl I32x4 {
        #[inline]
        pub fn zero() -> Self {
            Self(unsafe { _mm_setzero_si128() })
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_epi32(self.0, o.0) })
        }

        #[inline]
        pub fn to_array(self) -> [i32; 4] {
            let mut v = [0i32; 4];
            // SAFETY: 16 bytes, unaligned store.
            unsafe { _mm_storeu_si128(v.as_mut_ptr().cast(), self.0) };
            v
        }
    }

    /// Four f32 lanes.
    #[derive(Clone, Copy)]
    pub struct F32x4(__m128);

    impl F32x4 {
        #[inline]
        pub fn zero() -> Self {
            Self(unsafe { _mm_setzero_ps() })
        }

        #[inline]
        pub fn splat(v: f32) -> Self {
            Self(unsafe { _mm_set1_ps(v) })
        }

        /// The first four values of `v`.
        #[inline]
        pub fn load(v: &[f32]) -> Self {
            let v: &[f32; 4] = v[..4].try_into().unwrap();
            Self(unsafe { _mm_loadu_ps(v.as_ptr()) })
        }

        /// Stores the lanes to the first four values of `v`.
        #[inline]
        pub fn store(self, v: &mut [f32]) {
            let v: &mut [f32; 4] = (&mut v[..4]).try_into().unwrap();
            unsafe { _mm_storeu_ps(v.as_mut_ptr(), self.0) }
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(unsafe { _mm_add_ps(self.0, o.0) })
        }

        #[inline]
        pub fn sub(self, o: Self) -> Self {
            Self(unsafe { _mm_sub_ps(self.0, o.0) })
        }

        #[inline]
        pub fn mul(self, o: Self) -> Self {
            Self(unsafe { _mm_mul_ps(self.0, o.0) })
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    /// Eight i16 lanes.
    #[derive(Clone, Copy)]
    pub struct I16x8([i16; 8]);

    /// Four i32 lanes.
    #[derive(Clone, Copy)]
    pub struct I32x4([i32; 4]);

    impl I16x8 {
        #[inline]
        pub fn splat(v: i16) -> Self {
            Self([v; 8])
        }

        #[inline]
        pub fn from_array(v: [i16; 8]) -> Self {
            Self(v)
        }

        /// The first eight values of `v`.
        #[inline]
        pub fn load(v: &[i16]) -> Self {
            Self(v[..8].try_into().unwrap())
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l].wrapping_add(o.0[l])))
        }

        #[inline]
        pub fn sub(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l].wrapping_sub(o.0[l])))
        }

        #[inline]
        pub fn mul(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l].wrapping_mul(o.0[l])))
        }

        #[inline]
        pub fn and(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l] & o.0[l]))
        }

        /// Arithmetic shift right by `N`.
        #[inline]
        pub fn shr<const N: i32>(self) -> Self {
            Self(self.0.map(|x| x >> N))
        }

        /// The squares, summed in adjacent pairs.
        #[inline]
        pub fn sq_pairs(self) -> I32x4 {
            let s = |k: usize| self.0[k] as i32 * self.0[k] as i32;
            I32x4(std::array::from_fn(|l| s(2 * l).wrapping_add(s(2 * l + 1))))
        }
    }

    impl I32x4 {
        #[inline]
        pub fn zero() -> Self {
            Self([0; 4])
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l].wrapping_add(o.0[l])))
        }

        #[inline]
        pub fn to_array(self) -> [i32; 4] {
            self.0
        }
    }

    /// Four f32 lanes.
    #[derive(Clone, Copy)]
    pub struct F32x4([f32; 4]);

    impl F32x4 {
        #[inline]
        pub fn zero() -> Self {
            Self([0.0; 4])
        }

        #[inline]
        pub fn splat(v: f32) -> Self {
            Self([v; 4])
        }

        /// The first four values of `v`.
        #[inline]
        pub fn load(v: &[f32]) -> Self {
            Self(v[..4].try_into().unwrap())
        }

        /// Stores the lanes to the first four values of `v`.
        #[inline]
        pub fn store(self, v: &mut [f32]) {
            v[..4].copy_from_slice(&self.0);
        }

        #[inline]
        pub fn add(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l] + o.0[l]))
        }

        #[inline]
        pub fn sub(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l] - o.0[l]))
        }

        #[inline]
        pub fn mul(self, o: Self) -> Self {
            Self(std::array::from_fn(|l| self.0[l] * o.0[l]))
        }
    }
}

pub use imp::{F32x4, I16x8, I32x4};
