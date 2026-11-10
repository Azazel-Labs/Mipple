//! Image quality metrics: PSNR, luma SSIM, and normal-map angular error.

use crate::color::luma;
use crate::surface::Surface;

/// PSNR in dB over the listed channels (0 = R .. 3 = A). Returns `f64::INFINITY` for identical images.
pub fn psnr(a: &Surface, b: &Surface, channels: &[usize]) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "image size mismatch");
    if channels.is_empty() {
        return f64::INFINITY;
    }
    let mut sse = 0f64;
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        for &c in channels {
            let d = p[c] as f64 - q[c] as f64;
            sse += d * d;
        }
    }
    let mse = sse / (a.pixels.len() * channels.len()) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// PSNR on Rec. 709 luma.
pub fn luma_psnr(a: &Surface, b: &Surface) -> f64 {
    let mut sse = 0f64;
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        let d = (luma(p) - luma(q)) as f64;
        sse += d * d;
    }
    let mse = sse / a.pixels.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Mean SSIM on luma, using 8x8 windows with a stride of 4.
///
/// This is the plain (unweighted window) SSIM, good enough to rank encoders; use NVIDIA FLIP
/// or similar for perceptual sign-off.
pub fn ssim_luma(a: &Surface, b: &Surface) -> f64 {
    let la: Vec<f64> = a.pixels.iter().map(|p| luma(p) as f64).collect();
    let lb: Vec<f64> = b.pixels.iter().map(|p| luma(p) as f64).collect();
    ssim_plane(&la, &lb, a.width as usize, a.height as usize)
}

/// Mean SSIM on a single channel.
pub fn ssim_channel(a: &Surface, b: &Surface, c: usize) -> f64 {
    let la: Vec<f64> = a.pixels.iter().map(|p| p[c] as f64).collect();
    let lb: Vec<f64> = b.pixels.iter().map(|p| p[c] as f64).collect();
    ssim_plane(&la, &lb, a.width as usize, a.height as usize)
}

fn ssim_plane(la: &[f64], lb: &[f64], w: usize, h: usize) -> f64 {
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);
    let win = 8.min(w).min(h).max(1);
    let mut total = 0.0;
    let mut count = 0usize;
    let mut y = 0;
    while y + win <= h {
        let mut x = 0;
        while x + win <= w {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for yy in y..y + win {
                for xx in x..x + win {
                    let (p, q) = (la[yy * w + xx], lb[yy * w + xx]);
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let n = (win * win) as f64;
            let (ma, mb) = (sa / n, sb / n);
            let va = saa / n - ma * ma;
            let vb = sbb / n - mb * mb;
            let cov = sab / n - ma * mb;
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            count += 1;
            x += 4;
        }
        y += 4;
    }
    if count == 0 { 1.0 } else { total / count as f64 }
}

/// Decodes an RG-encoded tangent-space normal to a unit vector (Z reconstructed).
#[inline]
pub fn normal_from_rg(r: u8, g: u8) -> [f32; 3] {
    let x = r as f32 / 127.5 - 1.0;
    let y = g as f32 / 127.5 - 1.0;
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    let len = (x * x + y * y + z * z).sqrt().max(1e-6);
    [x / len, y / len, z / len]
}

/// Mean angular error in degrees between two normal maps, using only R and G
/// (Z reconstructed on both sides, as a BC5 shader would).
pub fn normal_angle_error_deg(a: &Surface, b: &Surface) -> f64 {
    let n = |r: u8, g: u8| {
        let v = normal_from_rg(r, g);
        [v[0] as f64, v[1] as f64, v[2] as f64]
    };
    let mut sum = 0f64;
    for (p, q) in a.pixels.iter().zip(&b.pixels) {
        if p[..2] == q[..2] {
            continue;
        }
        let (u, v) = (n(p[0], p[1]), n(q[0], q[1]));
        // atan2(|u×v|, u·v) is accurate for tiny angles, unlike acos.
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let s = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let c = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        sum += s.atan2(c).to_degrees();
    }
    sum / a.pixels.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_are_perfect() {
        let mut s = Surface::new(16, 16);
        for (i, p) in s.pixels.iter_mut().enumerate() {
            *p = [(i * 7) as u8, (i * 3) as u8, i as u8, 255];
        }
        assert!(psnr(&s, &s, &[0, 1, 2]).is_infinite());
        assert!((ssim_luma(&s, &s) - 1.0).abs() < 1e-9);
        assert!(normal_angle_error_deg(&s, &s) < 1e-6);
    }
}
