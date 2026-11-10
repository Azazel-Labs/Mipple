//! Mip chain generation: 2x2 box filter in linear light for sRGB data, with normal renormalization.

use crate::color::{linear_to_srgb, srgb_to_linear};
use crate::surface::Surface;

/// How texel values should be filtered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipFilter {
    /// Plain average of stored values (data maps: roughness, masks, BC4/BC5 data).
    Linear,
    /// RGB decoded from sRGB, averaged in linear light, re-encoded. Alpha averaged as-is.
    Srgb,
    /// RGB treated as a tangent-space normal, averaged and renormalized.
    Normal,
}

/// Full mip chain down to 1x1, starting with a copy of `top`.
pub fn chain(top: &Surface, filter: MipFilter) -> Vec<Surface> {
    let mut levels = vec![top.clone()];
    while let Some(last) = levels.last() {
        if last.width == 1 && last.height == 1 {
            break;
        }
        let next = downsample(last, filter);
        levels.push(next);
    }
    levels
}

/// One 2x2 box-filter step (odd edges clamp).
pub fn downsample(s: &Surface, filter: MipFilter) -> Surface {
    let w = (s.width / 2).max(1);
    let h = (s.height / 2).max(1);
    let mut out = Surface::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let xs = [(2 * x).min(s.width - 1), (2 * x + 1).min(s.width - 1)];
            let ys = [(2 * y).min(s.height - 1), (2 * y + 1).min(s.height - 1)];
            let taps = [
                s.get(xs[0], ys[0]),
                s.get(xs[1], ys[0]),
                s.get(xs[0], ys[1]),
                s.get(xs[1], ys[1]),
            ];
            let a = taps.iter().map(|t| t[3] as u32).sum::<u32>();
            let alpha = ((a + 2) / 4) as u8;
            let rgb = match filter {
                MipFilter::Linear => {
                    let mut c = [0u8; 3];
                    for (ch, v) in c.iter_mut().enumerate() {
                        let sum = taps.iter().map(|t| t[ch] as u32).sum::<u32>();
                        *v = ((sum + 2) / 4) as u8;
                    }
                    c
                }
                MipFilter::Srgb => {
                    let mut c = [0u8; 3];
                    for (ch, v) in c.iter_mut().enumerate() {
                        let sum: f32 = taps.iter().map(|t| srgb_to_linear(t[ch])).sum();
                        *v = linear_to_srgb(sum / 4.0);
                    }
                    c
                }
                MipFilter::Normal => {
                    let mut n = [0f32; 3];
                    for t in &taps {
                        for ch in 0..3 {
                            n[ch] += t[ch] as f32 / 127.5 - 1.0;
                        }
                    }
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len < 1e-6 {
                        [128, 128, 255]
                    } else {
                        let enc = |v: f32| ((v / len + 1.0) * 127.5 + 0.5).clamp(0.0, 255.0) as u8;
                        [enc(n[0]), enc(n[1]), enc(n[2])]
                    }
                }
            };
            out.set(x, y, [rgb[0], rgb[1], rgb[2], alpha]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_reaches_one_by_one() {
        let s = Surface::new(37, 10);
        let c = chain(&s, MipFilter::Srgb);
        assert_eq!(c.last().map(|l| (l.width, l.height)), Some((1, 1)));
        assert_eq!(c.len(), 6); // 37x10, 18x5, 9x2, 4x1, 2x1, 1x1
    }

    #[test]
    fn flat_normal_stays_flat() {
        let mut s = Surface::new(8, 8);
        s.pixels.iter_mut().for_each(|p| *p = [128, 128, 255, 255]);
        let d = downsample(&s, MipFilter::Normal);
        assert!(d.pixels.iter().all(|p| p[2] == 255 && p[0].abs_diff(128) <= 1));
    }
}
