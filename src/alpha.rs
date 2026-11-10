//! Alpha as transparency: the colour under transparent texels, and alpha-tested coverage.
//!
//! Texels with alpha 0 are invisible, but their colour still leaks into what is seen: a 2x2 mip
//! average mixes it into visible edge texels, and bilinear sampling mixes it into the texels next
//! to it. A cutout drawn with whatever was left under its mask (black, or the photo it was cut
//! from) gets dark or coloured fringes. And an encoder that counts that colour's error spends
//! bits on texels nobody sees. With [`AlphaMode::Transparency`]:
//!
//! - [`bleed`] replaces the colour of alpha-0 texels with a smooth extension of the visible
//!   colour around them (push-pull: alpha-weighted averages down a pyramid, then each hidden
//!   texel filled from the next coarser level), at every mip level;
//! - mips average RGB weighted by alpha ([`crate::mips`]), so hidden colour never reaches a
//!   visible texel;
//! - [`hide`] frees the colour of alpha-0 texels that are more than [`RING`] texels from any
//!   visible one before a level is encoded: in a block with visible texels they take the mean of
//!   the block's kept texels, which leaves the fit of those texels where it was, and fully freed
//!   blocks become one constant block, identical bytes that Zstd stores almost for free;
//! - with an alpha-test cutoff, each mip's alpha is scaled so the fraction of texels that pass
//!   the test matches the top level ([`preserve_coverage`]); a plain average thins foliage and
//!   fences out at a distance.

use crate::surface::Surface;

/// What alpha means ([`crate::pipeline::EncodeOptions::alpha`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum AlphaMode {
    /// Transparency for BC7 colour textures (sRGB, not normal maps), data otherwise.
    #[default]
    Auto,
    /// Transparency: the colour under alpha 0 is free (see the module docs).
    Transparency,
    /// An independent channel: every texel's colour is kept and mips average it unweighted.
    Data,
}

impl std::fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AlphaMode::Auto => "auto",
            AlphaMode::Transparency => "transparency",
            AlphaMode::Data => "data",
        })
    }
}

/// Texels within this many texels (in x and y) of a visible one keep their (bled) colour when a
/// level is encoded: bilinear sampling reads the nearest neighbours of visible texels, and a
/// second texel covers the half-texel offsets of block-compressed decoding and sampling.
pub const RING: u32 = 2;

/// Replaces the RGB of every alpha-0 texel with a smooth extension of the visible colour around
/// it (push-pull). Visible texels and all alpha are kept. A surface with no visible texel, or none
/// hidden, is left as it is.
pub fn bleed(s: &mut Surface) {
    let (w, h) = (s.width as usize, s.height as usize);
    if !s.pixels.iter().any(|p| p[3] == 0) || !s.pixels.iter().any(|p| p[3] != 0) {
        return;
    }
    // Pull: alpha-weighted colour sums (r a, g a, b a, a) down a pyramid of halving sizes.
    let mut levels: Vec<(usize, usize, Vec<[f32; 4]>)> = Vec::new();
    let top: Vec<[f32; 4]> = s
        .pixels
        .iter()
        .map(|p| {
            let a = f32::from(p[3]);
            [f32::from(p[0]) * a, f32::from(p[1]) * a, f32::from(p[2]) * a, a]
        })
        .collect();
    levels.push((w, h, top));
    while let Some((lw, lh, last)) = levels.last() {
        if *lw == 1 && *lh == 1 {
            break;
        }
        let (nw, nh) = (lw.div_ceil(2), lh.div_ceil(2));
        let mut next = vec![[0f32; 4]; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < *lw && sy < *lh {
                        let v = last[sy * lw + sx];
                        for c in 0..4 {
                            acc[c] += v[c];
                        }
                    }
                }
                next[y * nw + x] = acc;
            }
        }
        levels.push((nw, nh, next));
    }
    // Push: from the coarsest level up, each level's colour where it has weight, else the next
    // coarser level's filled colour sampled bilinearly at the texel's centre.
    let mut filled: Vec<[f32; 3]> = Vec::new();
    let mut fw = 0usize;
    let mut fh = 0usize;
    for (lw, lh, sums) in levels.iter().rev() {
        let mut out = vec![[0f32; 3]; lw * lh];
        for y in 0..*lh {
            for x in 0..*lw {
                let v = sums[y * lw + x];
                out[y * lw + x] = if v[3] > 0.0 {
                    [v[0] / v[3], v[1] / v[3], v[2] / v[3]]
                } else {
                    sample(
                        &filled,
                        fw,
                        fh,
                        (x as f32 + 0.5) / 2.0 - 0.5,
                        (y as f32 + 0.5) / 2.0 - 0.5,
                    )
                };
            }
        }
        filled = out;
        fw = *lw;
        fh = *lh;
    }
    for (p, f) in s.pixels.iter_mut().zip(&filled) {
        if p[3] == 0 {
            for c in 0..3 {
                p[c] = (f[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Bilinear sample of a `w` x `h` colour grid at (x, y) in texel coordinates, clamped to the edges.
fn sample(g: &[[f32; 3]], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let (x, y) = (x.clamp(0.0, (w - 1) as f32), y.clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let mut o = [0f32; 3];
    for c in 0..3 {
        let top = g[y0 * w + x0][c] * (1.0 - fx) + g[y0 * w + x1][c] * fx;
        let bot = g[y1 * w + x0][c] * (1.0 - fx) + g[y1 * w + x1][c] * fx;
        o[c] = top * (1.0 - fy) + bot * fy;
    }
    o
}

/// Which texels' colour is seen: visible ones (alpha above 0) and those within [`RING`] texels of
/// one. Row-major, one per texel.
pub fn kept(s: &Surface) -> Vec<bool> {
    let (w, h) = (s.width as usize, s.height as usize);
    let r = RING as usize;
    // Separable dilation: rows, then columns.
    let mut rows = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if s.pixels[y * w + x][3] != 0 {
                for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    rows[y * w + xx] = true;
                }
            }
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if rows[y * w + x] {
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    out[yy * w + x] = true;
                }
            }
        }
    }
    out
}

/// Frees the colour of the texels [`kept`] leaves out, for encoding: in a block with kept texels
/// they take the kept texels' mean colour (a point on those texels' principal axis, inside their
/// extent, so the endpoint fit stays theirs), and in a block without any every texel becomes
/// transparent black. Alpha is unchanged (freed texels are alpha 0). Returns `None` when nothing
/// changes (no freed texel).
pub fn hide(s: &Surface) -> Option<Surface> {
    let keep = kept(s);
    if keep.iter().all(|&k| k) {
        return None;
    }
    let w = s.width as usize;
    let mut out = s.clone();
    for by in 0..s.blocks_y() {
        for bx in 0..s.blocks_x() {
            let texels = (0..16u32)
                .map(|i| (bx * 4 + (i & 3), by * 4 + (i >> 2)))
                .filter(|&(x, y)| x < s.width && y < s.height);
            let mut sum = [0u32; 3];
            let mut n = 0u32;
            for (x, y) in texels.clone() {
                let i = y as usize * w + x as usize;
                if keep[i] {
                    for c in 0..3 {
                        sum[c] += u32::from(s.pixels[i][c]);
                    }
                    n += 1;
                }
            }
            let fill = if n == 0 {
                [0, 0, 0, 0]
            } else {
                let m = |c: usize| ((sum[c] + n / 2) / n) as u8;
                [m(0), m(1), m(2), 0]
            };
            for (x, y) in texels {
                let i = y as usize * w + x as usize;
                if !keep[i] {
                    out.pixels[i] = fill;
                }
            }
        }
    }
    Some(out)
}

/// Fraction of texels whose alpha, scaled by `scale`, exceeds `cutoff` (0-1).
fn coverage(s: &Surface, cutoff: f32, scale: f32) -> f32 {
    let t = cutoff * 255.0;
    let n = s.pixels.iter().filter(|p| f32::from(p[3]) * scale > t).count();
    n as f32 / s.pixels.len().max(1) as f32
}

/// Fraction of texels that pass an alpha test at `cutoff` (0-1).
pub fn coverage_at(s: &Surface, cutoff: f32) -> f32 {
    coverage(s, cutoff, 1.0)
}

/// Scales `s`'s alpha so the fraction of texels above `cutoff` (0-1) is as close to `target` as a
/// scale can make it (Castaño's coverage-preserving mips: a bisection on the scale).
pub fn preserve_coverage(s: &mut Surface, cutoff: f32, target: f32) {
    let (mut lo, mut hi) = (0f32, 4f32);
    let mut best = (f32::INFINITY, 1f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let c = coverage(s, cutoff, mid);
        if (c - target).abs() < best.0 {
            best = ((c - target).abs(), mid);
        }
        if c < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let scale = best.1;
    for p in &mut s.pixels {
        p[3] = (f32::from(p[3]) * scale + 0.5).min(255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disc of colour on a junk background (bright magenta) with alpha 0.
    fn disc(size: u32) -> Surface {
        let mut s = Surface::new(size, size);
        let c = size as f32 / 2.0;
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                let p = if d < c * 0.6 {
                    [40, 160, 60, 255]
                } else {
                    [255, 0, 255, 0]
                };
                s.set(x, y, p);
            }
        }
        s
    }

    #[test]
    fn bleed_extends_visible_colour() {
        let mut s = disc(32);
        bleed(&mut s);
        // No magenta left; visible texels and alpha untouched.
        for p in &s.pixels {
            assert!(
                p[0].abs_diff(40) <= 1 && p[1].abs_diff(160) <= 1 && p[2].abs_diff(60) <= 1,
                "{p:?}"
            );
        }
        assert_eq!(
            s.pixels.iter().filter(|p| p[3] == 255).count(),
            disc(32).pixels.iter().filter(|p| p[3] == 255).count()
        );
    }

    #[test]
    fn hide_frees_far_texels_only() {
        let s = disc(32);
        let keep = kept(&s);
        let h = hide(&s).unwrap();
        for (i, (a, b)) in s.pixels.iter().zip(&h.pixels).enumerate() {
            if keep[i] {
                assert_eq!(a, b);
            } else {
                assert_eq!(b[3], 0);
            }
        }
        // Corner blocks are far from the disc: transparent black.
        assert_eq!(h.get(0, 0), [0, 0, 0, 0]);
        // Opaque surfaces are left alone.
        let mut opaque = Surface::new(8, 8);
        opaque.pixels.iter_mut().for_each(|p| p[3] = 255);
        assert!(hide(&opaque).is_none());
    }

    #[test]
    fn coverage_is_preserved() {
        let s = disc(64);
        let target = coverage_at(&s, 0.5);
        let mut m = crate::mips::downsample(&s, crate::mips::MipFilter::Srgb, true);
        for _ in 0..3 {
            m = crate::mips::downsample(&m, crate::mips::MipFilter::Srgb, true);
        }
        preserve_coverage(&mut m, 0.5, target);
        assert!(
            (coverage_at(&m, 0.5) - target).abs() < 0.05,
            "{} vs {target}",
            coverage_at(&m, 0.5)
        );
    }
}
