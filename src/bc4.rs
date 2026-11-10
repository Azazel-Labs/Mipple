//! BC4 (single channel) and BC5 (two channels) block codecs.
//!
//! BC4 layout: `e0: u8`, `e1: u8`, then 48 bits of 3-bit indices, texel 0 in the low bits.
//! `e0 > e1` selects 8-value mode; otherwise 6-value mode plus literal 0 and 255.
//! BC5 is two BC4 blocks: red then green.

pub(crate) mod batch;

pub use batch::{encode_batch, encode_channels8, encode_surface};

use crate::surface::Block;

/// Palette for an endpoint pair, using the truncating reference interpolation.
pub fn palette(e0: u8, e1: u8) -> [u8; 8] {
    let (a, b) = (e0 as u32, e1 as u32);
    let mut p = [e0, e1, 0, 0, 0, 0, 0, 0];
    if e0 > e1 {
        for k in 1..7u32 {
            p[k as usize + 1] = (((7 - k) * a + k * b) / 7) as u8;
        }
    } else {
        for k in 1..5u32 {
            p[k as usize + 1] = (((5 - k) * a + k * b) / 5) as u8;
        }
        p[6] = 0;
        p[7] = 255;
    }
    p
}

/// Weighted SSE of one encoded BC4 block against channel `c` of `px`, without materializing the
/// decode. Sums row by row and may return early once the total exceeds `bound` (the result is
/// then some value above `bound`); otherwise it is exact.
#[inline]
pub fn sse_bounded(raw: &[u8], px: &Block, c: usize, w: u32, bound: u64) -> u64 {
    let pal = palette(raw[0], raw[1]);
    let mut bits = 0u64;
    for (i, &b) in raw[2..8].iter().enumerate() {
        bits |= (b as u64) << (8 * i);
    }
    let mut e = 0u64;
    for row in 0..4 {
        let mut r = 0u32;
        for j in 0..4 {
            let i = 4 * row + j;
            let d = px[i][c] as i32 - pal[((bits >> (3 * i)) & 7) as usize] as i32;
            r += (d * d) as u32;
        }
        e += w as u64 * r as u64;
        if e > bound {
            return e;
        }
    }
    e
}

/// Decodes one 8-byte BC4 block to 16 channel values.
pub fn decode_channel(raw: &[u8]) -> [u8; 16] {
    let pal = palette(raw[0], raw[1]);
    let mut bits = 0u64;
    for (i, &b) in raw[2..8].iter().enumerate() {
        bits |= (b as u64) << (8 * i);
    }
    let mut out = [0u8; 16];
    for (i, v) in out.iter_mut().enumerate() {
        *v = pal[((bits >> (3 * i)) & 7) as usize];
    }
    out
}

fn assign(v: &[u8; 16], pal: &[u8; 8]) -> (u64, u64) {
    let mut bits = 0u64;
    let mut err = 0u64;
    for (i, &x) in v.iter().enumerate() {
        let mut best = (u32::MAX, 0u64);
        for (k, &p) in pal.iter().enumerate() {
            let d = x.abs_diff(p) as u32;
            if d * d < best.0 {
                best = (d * d, k as u64);
            }
        }
        bits |= best.1 << (3 * i);
        err += best.0 as u64;
    }
    (bits, err)
}

fn to_bytes(e0: u8, e1: u8, bits: u64) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0] = e0;
    b[1] = e1;
    for i in 0..6 {
        b[2 + i] = (bits >> (8 * i)) as u8;
    }
    b
}

fn finalize(e0: u8, e1: u8, v: &[u8; 16]) -> ([u8; 8], u64) {
    let (bits, err) = assign(v, &palette(e0, e1));
    (to_bytes(e0, e1, bits), err)
}

/// Least-squares endpoints (e0, e1) of 8-value-mode texels from the sums of their steps toward e1
/// (`j`: 0 = e0 .. 7 = e1, weight j / 7): `j` = Σj, `jj` = Σj², `jx` = Σj·x, `x` = Σx. Rounded half
/// up and clamped to 0..=255, in exact integer arithmetic, so every encoder (SIMD lanes, the GPU
/// kernel) gets the same endpoints. `None` when every step is equal.
///
/// The normal equations scaled by 49: [Σ(7-j)², Σj(7-j); Σj(7-j), Σj²] [e0; e1] =
/// 7 [Σ(7-j)x; Σjx]. Every intermediate fits i32 (the largest, `2 * num + d`, stays below 2^29).
pub(crate) fn ls_steps(j: i32, jj: i32, jx: i32, x: i32) -> Option<(i32, i32)> {
    let s00 = 784 - 14 * j + jj; // Σ (7 - j)²
    let s01 = 7 * j - jj; // Σ j (7 - j)
    let s0x = 7 * x - jx; // Σ (7 - j) x
    let d = s00 * jj - s01 * s01;
    if d <= 0 {
        return None;
    }
    let round = |n: i32| if n <= 0 { 0 } else { ((2 * n + d) / (2 * d)).min(255) };
    Some((round(7 * (jj * s0x - s01 * jx)), round(7 * (s00 * jx - s01 * s0x))))
}

/// [`ls_steps`] for 8-value-mode index bits (`bits`, 3 per texel) over values `v`.
fn ls_bits(bits: u64, v: &[u8; 16]) -> Option<(i32, i32)> {
    let (mut j, mut jj, mut jx, mut sx) = (0, 0, 0, 0);
    for (i, &x) in v.iter().enumerate() {
        // Step toward e1: index 0 is e0 (0), 1 is e1 (7), k >= 2 is step k - 1.
        let s = match (bits >> (3 * i)) & 7 {
            0 => 0,
            1 => 7,
            k => k as i32 - 1,
        };
        j += s;
        jj += s * s;
        jx += s * x as i32;
        sx += x as i32;
    }
    ls_steps(j, jj, jx, sx)
}

/// Encodes 16 values. Effort 0 projects onto the min/max line, 1 adds a least-squares refit,
/// 2..=9 run the exact search with ever wider endpoint windows. Runs the SIMD batch encoder on
/// one lane; batch callers use [`encode_batch`].
pub fn encode_channel(v: &[u8; 16], effort: u8) -> [u8; 8] {
    encode_channels8(&[*v; crate::simd::LANES], effort)[0]
}

/// Scalar reference for the exact encoder (efforts 1-3), with endpoint window radius `r`.
#[cfg(test)]
fn reference_channel(v: &[u8; 16], r: i32) -> [u8; 8] {
    let lo = *v.iter().min().unwrap();
    let hi = *v.iter().max().unwrap();
    if lo == hi {
        return to_bytes(hi, hi, 0);
    }
    let mut best = finalize(hi, lo, v);
    // 6-value mode can represent exact 0 / 255 for free; fit the rest.
    let inner: Vec<u8> = v.iter().copied().filter(|&x| x != 0 && x != 255).collect();
    if inner.len() < 16 {
        let (ilo, ihi) = match (inner.iter().min(), inner.iter().max()) {
            (Some(&a), Some(&b)) => (a, b),
            _ => (0, 0),
        };
        let cand = finalize(ilo, ihi, v);
        if cand.1 < best.1 {
            best = cand;
        }
    }
    if best.1 == 0 {
        return best.0;
    }

    // Least-squares refit in 8-value mode.
    for _ in 0..2 {
        if best.0[0] <= best.0[1] {
            break;
        }
        let bits = u64::from_le_bytes([best.0[2], best.0[3], best.0[4], best.0[5], best.0[6], best.0[7], 0, 0]);
        let Some((a, b)) = ls_bits(bits, v) else {
            break;
        };
        let (a, b) = (a as u8, b as u8);
        if a <= b {
            break;
        }
        let cand = finalize(a, b, v);
        if cand.1 < best.1 {
            best = cand;
        } else {
            break;
        }
    }

    // Exhaustive search in a small window around the current endpoints.
    if r > 0 && best.1 > 0 {
        let (b0, b1) = (best.0[0] as i32, best.0[1] as i32);
        for d0 in -r..=r {
            for d1 in -r..=r {
                let (e0, e1) = (b0 + d0, b1 + d1);
                if !(0..=255).contains(&e0) || !(0..=255).contains(&e1) {
                    continue;
                }
                // Stay in the same mode as the current best.
                if (e0 > e1) != (b0 > b1) {
                    continue;
                }
                let cand = finalize(e0 as u8, e1 as u8, v);
                if cand.1 < best.1 {
                    best = cand;
                }
            }
        }
    }
    best.0
}

/// Keeps the endpoints of `raw`, re-chooses every index. Used by RDO.
pub fn reindex_channel(raw: &[u8], v: &[u8; 16]) -> [u8; 8] {
    finalize(raw[0], raw[1], v).0
}

/// [`reindex_channel`] and the result's unweighted squared error.
pub fn reindex_channel_err(raw: &[u8], v: &[u8; 16]) -> ([u8; 8], u64) {
    finalize(raw[0], raw[1], v)
}

/// Keeps the indices of `raw`, least-squares refits the endpoints. Used by RDO.
pub fn refit_channel(raw: &[u8], v: &[u8; 16]) -> Option<[u8; 8]> {
    if raw[0] <= raw[1] {
        return None;
    }
    let bits = u64::from_le_bytes([raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], 0, 0]);
    let (a, b) = ls_bits(bits, v)?;
    let (a, b) = (a as u8, b as u8);
    if a <= b {
        return None;
    }
    let mut out = [0u8; 8];
    out.copy_from_slice(raw);
    out[0] = a;
    out[1] = b;
    Some(out)
}

/// Extracts one channel of a block.
pub fn channel(px: &Block, c: usize) -> [u8; 16] {
    let mut v = [0u8; 16];
    for (i, t) in px.iter().enumerate() {
        v[i] = t[c];
    }
    v
}

/// BC4 block decode: value replicated to RGB, opaque alpha.
pub fn decode_bc4(raw: &[u8], out: &mut Block) {
    let r = decode_channel(&raw[0..8]);
    for (i, t) in out.iter_mut().enumerate() {
        *t = [r[i], r[i], r[i], 255];
    }
}

/// BC5 block decode: R and G, blue 0, opaque alpha. Normal-map Z is reconstructed by the caller.
pub fn decode_bc5(raw: &[u8], out: &mut Block) {
    let r = decode_channel(&raw[0..8]);
    let g = decode_channel(&raw[8..16]);
    for (i, t) in out.iter_mut().enumerate() {
        *t = [r[i], g[i], 0, 255];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_beats_uniform_quantization() {
        // 16 values over 0..255 through 8 levels: uniform quantization gives ~108 per texel.
        let v: [u8; 16] = std::array::from_fn(|i| (i * 17) as u8);
        let raw = encode_channel(&v, 2);
        let d = decode_channel(&raw);
        let sse: u32 = v.iter().zip(&d).map(|(a, b)| (a.abs_diff(*b) as u32).pow(2)).sum();
        assert!(sse < 16 * 100, "sse {sse}");
    }

    #[test]
    fn extremes_use_six_value_mode() {
        let mut v = [100u8; 16];
        v[0] = 0;
        v[1] = 255;
        v[2] = 120;
        let raw = encode_channel(&v, 2);
        let d = decode_channel(&raw);
        assert_eq!(d[0], 0);
        assert_eq!(d[1], 255);
    }
}
