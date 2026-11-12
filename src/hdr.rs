//! HDR images for BC6H: a linear float RGB surface ([`HdrSurface`]), loading (Radiance `.hdr`, or
//! 8/16-bit images decoded from sRGB), box-filtered mips, and the quality metrics HDR codecs are
//! compared with: multi-exposure PSNR ([`mpsnr`]) and log-space RMSE ([`log_rmse`]). OpenEXR is not
//! read: the `image` crate's EXR codec instantiates in every crate that saves an image and slows
//! builds by tens of minutes.

use std::path::Path;

use anyhow::{Context, Result};

use crate::bc6h::{HdrBlock, f32_to_uf16, half_to_f32};

/// A linear RGB float image, row-major. Negative and NaN values are stored as given; BC6H
/// (unsigned) encodes them as 0.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrSurface {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

impl HdrSurface {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 3]; width as usize * height as usize],
        }
    }

    /// Loads an image as linear light: float formats (Radiance `.hdr`) as stored, 8- and 16-bit
    /// images decoded from sRGB unless `linear` (values then read as 0-1 fractions).
    pub fn load(path: &Path, linear: bool) -> Result<Self> {
        let img = image::open(path).with_context(|| format!("loading {}", path.display()))?;
        let float = matches!(img.color(), image::ColorType::Rgb32F | image::ColorType::Rgba32F);
        let rgb = img.to_rgb32f();
        let decode = !float && !linear;
        let pixels = rgb
            .pixels()
            .map(|p| p.0.map(|v| if decode { srgb_to_linear_f(v) } else { v }))
            .collect();
        Ok(Self {
            width: rgb.width(),
            height: rgb.height(),
            pixels,
        })
    }

    /// An 8-bit surface's RGB as linear light, decoded from sRGB when `srgb`.
    pub fn from_ldr(s: &crate::surface::Surface, srgb: bool) -> Self {
        let f = |v: u8| {
            if srgb {
                crate::color::srgb_to_linear(v)
            } else {
                f32::from(v) / 255.0
            }
        };
        Self {
            width: s.width,
            height: s.height,
            pixels: s.pixels.iter().map(|p| [f(p[0]), f(p[1]), f(p[2])]).collect(),
        }
    }

    /// Writes a Radiance `.hdr` (or, clamped to 8 bits, any LDR format the extension names).
    pub fn save(&self, path: &Path) -> Result<()> {
        let flat: Vec<f32> = self.pixels.iter().flatten().copied().collect();
        let img = image::Rgb32FImage::from_raw(self.width, self.height, flat).context("image size")?;
        image::DynamicImage::ImageRgb32F(img)
            .save(path)
            .with_context(|| format!("writing {}", path.display()))
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 3] {
        self.pixels[(y * self.width + x) as usize]
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

    /// The 4x4 block at block coordinates (`bx`, `by`) as BC6H targets (unsigned half floats),
    /// replicating the last row and column past the edges.
    pub fn read_block(&self, bx: u32, by: u32) -> HdrBlock {
        let mut out = [[0u16; 3]; 16];
        for (i, o) in out.iter_mut().enumerate() {
            let x = (bx * 4 + (i % 4) as u32).min(self.width - 1);
            let y = (by * 4 + (i / 4) as u32).min(self.height - 1);
            *o = self.get(x, y).map(f32_to_uf16);
        }
        out
    }

    /// The image of decoded half-float blocks in raster order (each `block` covers 4x4 texels,
    /// cropped at the edges).
    pub fn from_blocks(width: u32, height: u32, blocks: impl IntoIterator<Item = HdrBlock>) -> Self {
        let mut s = Self::new(width, height);
        let bw = width.div_ceil(4);
        for (bi, b) in blocks.into_iter().enumerate() {
            let (bx, by) = (bi as u32 % bw, bi as u32 / bw);
            for (i, t) in b.iter().enumerate() {
                let (x, y) = (bx * 4 + (i % 4) as u32, by * 4 + (i / 4) as u32);
                if x < width && y < height {
                    s.pixels[(y * width + x) as usize] = t.map(half_to_f32);
                }
            }
        }
        s
    }

    /// The next mip level: a 2x2 box filter (edges clamped for odd sizes), as [`crate::mips`].
    pub fn downsample(&self) -> Self {
        let (w, h) = ((self.width / 2).max(1), (self.height / 2).max(1));
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let xs = [(2 * x).min(self.width - 1), (2 * x + 1).min(self.width - 1)];
                let ys = [(2 * y).min(self.height - 1), (2 * y + 1).min(self.height - 1)];
                let mut c = [0f32; 3];
                for yy in ys {
                    for xx in xs {
                        let p = self.get(xx, yy);
                        for k in 0..3 {
                            c[k] += p[k];
                        }
                    }
                }
                out.pixels[(y * w + x) as usize] = c.map(|v| v * 0.25);
            }
        }
        out
    }

    /// Full mip chain down to 1x1, starting with a copy of `self`.
    pub fn chain(&self) -> Vec<Self> {
        let mut levels = vec![self.clone()];
        while let Some(last) = levels.last() {
            if last.width == 1 && last.height == 1 {
                break;
            }
            let next = last.downsample();
            levels.push(next);
        }
        levels
    }

    /// The values BC6H can store: clamped to [0, 65504] (NaN as 0) and rounded to half precision.
    pub fn representable(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            pixels: self
                .pixels
                .iter()
                .map(|p| p.map(|v| half_to_f32(f32_to_uf16(v))))
                .collect(),
        }
    }
}

/// sRGB-encoded 0-1 to linear 0-1.
fn srgb_to_linear_f(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Clean value for the metrics: NaN and negatives as 0 (what BC6H unsigned stores).
#[inline]
fn clean(v: f32) -> f32 {
    if v > 0.0 { v.min(65504.0) } else { 0.0 }
}

/// The exposure range (whole stops) [`mpsnr`] uses for an image, from its brightest and darkest
/// texels: from the stop that brings the 99.9th percentile of max(R, G, B) to 1 to the stop that
/// brings the 1st percentile of the positive values to 1, at most 20 stops (dropping the darkest).
pub fn exposure_range(s: &HdrSurface) -> (i32, i32) {
    let mut v: Vec<f32> = s
        .pixels
        .iter()
        .map(|p| clean(p[0]).max(clean(p[1])).max(clean(p[2])))
        .filter(|&m| m > 0.0)
        .collect();
    if v.is_empty() {
        return (0, 0);
    }
    v.sort_unstable_by(f32::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let lo = (-at(0.999).log2()).round() as i32;
    let hi = (-at(0.01).log2()).round() as i32;
    (lo, hi.clamp(lo, lo + 20))
}

/// Multi-exposure PSNR (Munkberg et al. 2006, "High dynamic range texture compression for graphics
/// hardware"): each image is exposed at every whole stop `c` in `stops`, tone mapped as
/// round(255 * (2^c * v)^(1/2.2)) clamped to 0-255, and the squared error is averaged over all
/// exposures, texels and RGB channels. In dB against 255.
pub fn mpsnr(a: &HdrSurface, b: &HdrSurface, stops: (i32, i32)) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
    let tm = |v: f32, scale: f32| ((255.0 * (scale * clean(v)).powf(1.0 / 2.2)).round()).clamp(0.0, 255.0);
    let mut sum = 0f64;
    let mut n = 0u64;
    for c in stops.0..=stops.1 {
        let scale = 2f32.powi(c);
        for (p, q) in a.pixels.iter().zip(&b.pixels) {
            for k in 0..3 {
                let d = f64::from(tm(p[k], scale) - tm(q[k], scale));
                sum += d * d;
            }
        }
        n += a.pixels.len() as u64 * 3;
    }
    let mse = sum / n.max(1) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Root mean square of log2 differences over RGB, each value floored at 2^-14 (the smallest
/// normal half): a relative error, about the error in stops.
pub fn log_rmse(a: &HdrSurface, b: &HdrSurface) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");
    let floor = 2f32.powi(-14);
    let l = |v: f32| f64::from(clean(v).max(floor).log2());
    let mut sum = 0f64;
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        for k in 0..3 {
            let d = l(p[k]) - l(q[k]);
            sum += d * d;
        }
    }
    (sum / (a.pixels.len() as f64 * 3.0).max(1.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> HdrSurface {
        let mut s = HdrSurface::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = 2f32.powf(x as f32 / w as f32 * 16.0 - 8.0);
                s.pixels[(y * w + x) as usize] = [v, v * 0.5, v * (1.0 + y as f32 / h as f32)];
            }
        }
        s
    }

    #[test]
    fn metrics_are_exact_on_identity_and_drop_with_error() {
        let a = ramp(64, 32);
        let r = exposure_range(&a);
        assert!(r.0 < r.1 && r.1 - r.0 <= 20, "{r:?}");
        assert_eq!(mpsnr(&a, &a, r), f64::INFINITY);
        assert_eq!(log_rmse(&a, &a), 0.0);
        let mut b = a.clone();
        for p in &mut b.pixels {
            p[0] *= 1.05;
        }
        let mut c = a.clone();
        for p in &mut c.pixels {
            p[0] *= 1.2;
        }
        assert!(mpsnr(&a, &b, r) > mpsnr(&a, &c, r));
        // A uniform 5% error is log2(1.05) on one channel of three.
        let want = (1.05f64.log2().powi(2) / 3.0).sqrt();
        assert!((log_rmse(&a, &b) - want).abs() < 1e-3, "{} vs {want}", log_rmse(&a, &b));
    }

    #[test]
    fn blocks_round_trip_and_mips_average() {
        let a = ramp(10, 6).representable();
        let blocks: Vec<HdrBlock> = (0..a.block_count() as u32)
            .map(|i| a.read_block(i % a.blocks_x(), i / a.blocks_x()))
            .collect();
        assert_eq!(HdrSurface::from_blocks(10, 6, blocks), a);
        let chain = a.chain();
        assert_eq!(chain.len(), 4); // 10x6, 5x3, 2x1, 1x1
        let m = &chain[1];
        let want = (a.get(0, 0)[2] + a.get(1, 0)[2] + a.get(0, 1)[2] + a.get(1, 1)[2]) / 4.0;
        assert!((m.get(0, 0)[2] - want).abs() < 1e-6);
    }
}
