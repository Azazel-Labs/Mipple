//! sRGB <-> linear conversion helpers.

use std::sync::OnceLock;

fn srgb_table() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

/// sRGB-encoded byte to linear [0, 1].
#[inline]
pub fn srgb_to_linear(v: u8) -> f32 {
    srgb_table()[v as usize]
}

/// Linear [0, 1] to sRGB-encoded byte (rounded).
#[inline]
pub fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Rec. 709 luma of an RGB byte triple, in [0, 255].
#[inline]
pub fn luma(p: &[u8; 4]) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip_is_exact_for_bytes() {
        for i in 0..=255u8 {
            assert_eq!(linear_to_srgb(srgb_to_linear(i)), i);
        }
    }
}
