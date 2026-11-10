//! Deterministic procedural test textures, so the benchmark runs without downloading anything.
//!
//! Each image targets a known weakness: smooth gradients (banding), high-frequency color
//! (detail loss), single-channel masks (BC4), normals (BC5 / angular error), and hard and
//! soft alpha edges (BC7 alpha modes).

use crate::surface::Surface;
use anyhow::Result;
use std::path::Path;

#[inline]
fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h =
        (x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263) ^ seed.wrapping_mul(2_246_822_519);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

#[inline]
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (smooth(x - xi as f32), smooth(y - yi as f32));
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    let top = a + (b - a) * fx;
    let bot = c + (d - c) * fx;
    top + (bot - top) * fy
}

/// Fractal noise in [0, 1].
fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(o * 101));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn generate(size: u32, f: impl Fn(f32, f32) -> [u8; 4]) -> Surface {
    let mut s = Surface::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let (u, v) = (x as f32 / size as f32, y as f32 / size as f32);
            s.set(x, y, f(u, v));
        }
    }
    s
}

/// Brick wall albedo: hard mortar edges plus per-brick tint and grain.
pub fn bricks(size: u32) -> Surface {
    generate(size, |u, v| {
        let rows = 10.0;
        let row = (v * rows).floor();
        let offset = if row as i32 % 2 == 0 { 0.0 } else { 0.5 };
        let cols = 5.0;
        let col = (u * cols + offset).floor();
        let fu = (u * cols + offset).fract();
        let fv = (v * rows).fract();
        let mortar = !(0.04..=0.96).contains(&fu) || !(0.08..=0.92).contains(&fv);
        let grain = fbm(u * 64.0, v * 64.0, 4, 7);
        if mortar {
            let g = 0.55 + 0.15 * grain;
            [to_u8(g), to_u8(g * 0.97), to_u8(g * 0.92), 255]
        } else {
            let tint = hash(col as i32, row as i32, 3);
            let r = 0.45 + 0.25 * tint + 0.2 * (grain - 0.5);
            let g = 0.22 + 0.10 * tint + 0.15 * (grain - 0.5);
            let b = 0.15 + 0.05 * tint + 0.10 * (grain - 0.5);
            [to_u8(r), to_u8(g), to_u8(b), 255]
        }
    })
}

/// Smooth four-corner gradient with ±1 LSB dither: the banding stress test.
pub fn gradient(size: u32) -> Surface {
    generate(size, |u, v| {
        let c00 = [0.05, 0.10, 0.35];
        let c10 = [0.95, 0.55, 0.20];
        let c01 = [0.20, 0.70, 0.60];
        let c11 = [0.90, 0.90, 0.95];
        let n = (hash((u * 4096.0) as i32, (v * 4096.0) as i32, 11) - 0.5) / 255.0;
        let mut out = [0u8; 4];
        for c in 0..3 {
            let top = c00[c] + (c10[c] - c00[c]) * u;
            let bot = c01[c] + (c11[c] - c01[c]) * u;
            out[c] = to_u8(top + (bot - top) * v + n);
        }
        out[3] = 255;
        out
    })
}

/// Busy, saturated, high-frequency color.
pub fn foliage(size: u32) -> Surface {
    generate(size, |u, v| {
        let a = fbm(u * 24.0, v * 24.0, 5, 21);
        let b = fbm(u * 48.0 + 3.0, v * 48.0, 4, 22);
        let veins = ((u * 90.0 + a * 12.0).sin() * 0.5 + 0.5).powf(6.0);
        let r = 0.15 + 0.35 * b + 0.3 * veins;
        let g = 0.35 + 0.45 * a;
        let bl = 0.10 + 0.25 * (1.0 - a) * b;
        [to_u8(r), to_u8(g), to_u8(bl), 255]
    })
}

/// Grayscale roughness-style mask with sharp spots.
pub fn roughness(size: u32) -> Surface {
    generate(size, |u, v| {
        let base = fbm(u * 16.0, v * 16.0, 6, 31);
        let spots = value_noise(u * 40.0, v * 40.0, 33);
        let val = if spots > 0.82 { 0.1 } else { 0.35 + 0.5 * base };
        let g = to_u8(val);
        [g, g, g, 255]
    })
}

/// Tangent-space normal map derived from a rocky height field.
pub fn normal_rocks(size: u32) -> Surface {
    let height = |u: f32, v: f32| {
        let h = fbm(u * 12.0, v * 12.0, 6, 41);
        let ridge = 1.0 - (2.0 * value_noise(u * 6.0, v * 6.0, 43) - 1.0).abs();
        0.7 * h + 0.3 * ridge * ridge
    };
    let e = 1.0 / size as f32;
    generate(size, |u, v| {
        let dx = (height(u + e, v) - height(u - e, v)) / (2.0 * e);
        let dy = (height(u, v + e) - height(u, v - e)) / (2.0 * e);
        let strength = 0.04;
        let (nx, ny, nz) = (-dx * strength, -dy * strength, 1.0);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let enc = |c: f32| to_u8((c / len) * 0.5 + 0.5);
        [enc(nx), enc(ny), enc(nz), 255]
    })
}

/// UI-like sprite sheet: anti-aliased shapes over transparency.
pub fn ui_alpha(size: u32) -> Surface {
    generate(size, |u, v| {
        let mut col = [0.0f32, 0.0, 0.0, 0.0];
        let mut over = |c: [f32; 3], a: f32| {
            for k in 0..3 {
                col[k] = col[k] * (1.0 - a) + c[k] * a;
            }
            col[3] = col[3] + a * (1.0 - col[3]);
        };
        let px = 1.0 / 512.0;
        // Rounded panel.
        let (cx, cy, hw, hh, r) = (0.5f32, 0.30f32, 0.40f32, 0.20f32, 0.05f32);
        let qx = ((u - cx).abs() - hw + r).max(0.0);
        let qy = ((v - cy).abs() - hh + r).max(0.0);
        let d = (qx * qx + qy * qy).sqrt() - r;
        over([0.12, 0.14, 0.20], (0.5 - d / px).clamp(0.0, 1.0) * 0.9);
        // Buttons.
        for (i, c) in [[0.20, 0.55, 0.95], [0.95, 0.40, 0.30], [0.30, 0.80, 0.45]]
            .iter()
            .enumerate()
        {
            let bx = 0.2 + 0.3 * i as f32;
            let d = ((u - bx).powi(2) + (v - 0.75).powi(2)).sqrt() - 0.09;
            over(*c, (0.5 - d / px).clamp(0.0, 1.0));
        }
        // Text-like bars.
        if (0.18..0.82).contains(&u) && (0.18..0.42).contains(&v) {
            let line = ((v - 0.18) * 40.0).fract();
            let word = hash((u * 30.0) as i32, ((v - 0.18) * 40.0) as i32, 51) > 0.25;
            if line < 0.45 && word {
                over([0.92, 0.93, 0.96], 1.0);
            }
        }
        if col[3] > 0.0 {
            let a = col[3].max(1e-6);
            for c in col.iter_mut().take(3) {
                *c /= a;
            }
        }
        [to_u8(col[0]), to_u8(col[1]), to_u8(col[2]), to_u8(col[3])]
    })
}

/// A procedural image generator taking the image size in pixels.
pub type Generator = fn(u32) -> Surface;

/// The synthetic set: (file name, generator). Deterministic, so it doubles as the
/// regression-check corpus (`mipple check`).
pub fn set() -> [(&'static str, Generator); 6] {
    [
        ("albedo_bricks.png", bricks),
        ("albedo_gradient.png", gradient),
        ("albedo_foliage.png", foliage),
        ("mask_roughness.png", roughness),
        ("normal_rocks.png", normal_rocks),
        ("ui_alpha.png", ui_alpha),
    ]
}

/// Writes the full synthetic set as PNGs into `dir`. Returns the written paths.
pub fn write_all(dir: &Path, size: u32) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut out = Vec::new();
    for (name, f) in set() {
        let p = dir.join(name);
        f(size).save_png(&p)?;
        out.push(p);
    }
    Ok(out)
}
