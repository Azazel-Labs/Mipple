//! In-memory RGBA8 images and 4x4 block access.

use anyhow::{Context, Result};
use std::path::Path;

/// One RGBA8 texel.
pub type Rgba = [u8; 4];

/// A 4x4 block of texels in row-major order (texel `i` is at x = i % 4, y = i / 4).
pub type Block = [Rgba; 16];

/// A simple RGBA8 image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgba>,
}

impl Surface {
    /// Creates an opaque black surface.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0, 0, 0, 255]; width as usize * height as usize],
        }
    }

    /// Loads PNG, JPEG or TGA into RGBA8.
    pub fn load(path: &Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("failed to open image {}", path.display()))?
            .to_rgba8();
        let (width, height) = img.dimensions();
        let pixels = img.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }

    /// Saves as a PNG.
    pub fn save_png(&self, path: &Path) -> Result<()> {
        let buf: Vec<u8> = self.pixels.iter().flatten().copied().collect();
        image::save_buffer(path, &buf, self.width, self.height, image::ColorType::Rgba8)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Rgba {
        self.pixels[(y * self.width + x) as usize]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: Rgba) {
        let w = self.width;
        self.pixels[(y * w + x) as usize] = v;
    }

    pub fn blocks_x(&self) -> u32 {
        self.width.div_ceil(4)
    }

    pub fn blocks_y(&self) -> u32 {
        self.height.div_ceil(4)
    }

    pub fn block_count(&self) -> usize {
        self.blocks_x() as usize * self.blocks_y() as usize
    }

    /// Reads block (bx, by), replicating edge texels when the image is not a multiple of 4.
    pub fn read_block(&self, bx: u32, by: u32) -> Block {
        let mut b = [[0u8; 4]; 16];
        let (x0, y0) = (bx * 4, by * 4);
        if x0 + 4 <= self.width && y0 + 4 <= self.height {
            let w = self.width as usize;
            for (r, row) in b.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let at = (y0 as usize + r) * w + x0 as usize;
                row.copy_from_slice(&self.pixels[at..at + 4]);
            }
            return b;
        }
        for (i, t) in b.iter_mut().enumerate() {
            let x = (bx * 4 + (i as u32 & 3)).min(self.width - 1);
            let y = (by * 4 + (i as u32 >> 2)).min(self.height - 1);
            *t = self.get(x, y);
        }
        b
    }

    /// Channel `c` of block (bx, by), replicating edge texels like [`Self::read_block`].
    #[inline]
    pub fn read_block_channel(&self, bx: u32, by: u32, c: usize) -> [u8; 16] {
        let mut v = [0u8; 16];
        let (x0, y0) = (bx * 4, by * 4);
        if x0 + 4 <= self.width && y0 + 4 <= self.height {
            let w = self.width as usize;
            for (r, out) in v.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let at = (y0 as usize + r) * w + x0 as usize;
                for (o, t) in out.iter_mut().zip(&self.pixels[at..at + 4]) {
                    *o = t[c];
                }
            }
        } else {
            for (o, t) in v.iter_mut().zip(self.read_block(bx, by)) {
                *o = t[c];
            }
        }
        v
    }

    /// All blocks in raster order.
    pub fn blocks(&self) -> Vec<Block> {
        let bw = self.blocks_x();
        (0..self.block_count() as u32)
            .map(|i| self.read_block(i % bw, i / bw))
            .collect()
    }

    /// Writes block (bx, by), dropping texels outside the image.
    pub fn write_block(&mut self, bx: u32, by: u32, b: &Block) {
        for (i, t) in b.iter().enumerate() {
            let x = bx * 4 + (i as u32 & 3);
            let y = by * 4 + (i as u32 >> 2);
            if x < self.width && y < self.height {
                self.set(x, y, *t);
            }
        }
    }

    /// True if any texel is not fully opaque.
    pub fn has_alpha(&self) -> bool {
        self.pixels.iter().any(|p| p[3] != 255)
    }
}

/// Weighted squared error between two blocks. `w` weights R, G, B, A.
#[inline]
pub fn block_sse(a: &Block, b: &Block, w: &[u32; 4]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if w.iter().all(|&x| x <= 128) {
        // SAFETY: SSE2 is part of the x86-64 baseline.
        return unsafe { block_sse_sse2(a, b, w) };
    }
    block_sse_scalar(a, b, w)
}

/// [`block_sse`] on SSE2 (x86-64's baseline) for weights up to 128: per texel row, the
/// differences as i16, times the weights (|d w| <= 32640), multiplied by the differences and
/// summed in pairs to i32 (`madd`; each lane totals at most 8 * 2 * 255 * 32640 < 2^31).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
fn block_sse_sse2(a: &Block, b: &Block, w: &[u32; 4]) -> u64 {
    use std::arch::x86_64::*;
    let [w0, w1, w2, w3] = w.map(|x| x as i16);
    let wv = _mm_setr_epi16(w0, w1, w2, w3, w0, w1, w2, w3);
    let z = _mm_setzero_si128();
    let mut acc = z;
    for r in 0..4 {
        // SAFETY: a Block is 64 bytes, so row r's 16 bytes are in bounds; unaligned loads.
        let (pa, pb) = unsafe {
            (
                _mm_loadu_si128(a.as_ptr().add(r * 4).cast()),
                _mm_loadu_si128(b.as_ptr().add(r * 4).cast()),
            )
        };
        let lo = _mm_sub_epi16(_mm_unpacklo_epi8(pa, z), _mm_unpacklo_epi8(pb, z));
        let hi = _mm_sub_epi16(_mm_unpackhi_epi8(pa, z), _mm_unpackhi_epi8(pb, z));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(lo, _mm_mullo_epi16(lo, wv)));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(hi, _mm_mullo_epi16(hi, wv)));
    }
    let s = _mm_add_epi32(acc, _mm_shuffle_epi32::<0b01_00_11_10>(acc));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    u64::from(_mm_cvtsi128_si32(s) as u32)
}

fn block_sse_scalar(a: &Block, b: &Block, w: &[u32; 4]) -> u64 {
    let mut e = 0u64;
    for i in 0..16 {
        for c in 0..4 {
            let d = a[i][c] as i32 - b[i][c] as i32;
            e += (w[c] as i32 * d * d) as u64;
        }
    }
    e
}

/// Weighted squared error between two texels.
#[inline]
pub fn texel_sse(a: &Rgba, b: &Rgba, w: &[u32; 4]) -> u32 {
    let mut e = 0u32;
    for c in 0..4 {
        let d = a[c] as i32 - b[c] as i32;
        e += w[c] * (d * d) as u32;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sse_matches_scalar() {
        let mut x = 0x2545_f491_u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let ws = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [3, 7, 128, 0], [128, 128, 128, 128], [200, 1, 1, 1]];
        for k in 0..2000 {
            let mut a: Block = [[0; 4]; 16];
            let mut b: Block = [[0; 4]; 16];
            for i in 0..16 {
                for c in 0..4 {
                    // Every fifth pair at the extremes (0 vs 255) to exercise the bound.
                    (a[i][c], b[i][c]) = if k % 5 == 0 { (0, 255) } else { (next() as u8, next() as u8) };
                }
            }
            for w in &ws {
                assert_eq!(block_sse(&a, &b, w), block_sse_scalar(&a, &b, w), "w {w:?}");
                assert_eq!(block_sse(&b, &a, w), block_sse_scalar(&b, &a, w), "w {w:?}");
            }
        }
    }
}
