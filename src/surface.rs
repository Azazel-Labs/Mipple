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
