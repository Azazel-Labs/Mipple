#![allow(dead_code, reason = "each test binary uses a different subset of these helpers")]

use mipple::{Format, Surface, synth};

/// Small synthetic corpus for fast tests.
pub fn corpus(size: u32) -> Vec<(&'static str, Surface)> {
    vec![
        ("bricks", synth::bricks(size)),
        ("gradient", synth::gradient(size)),
        ("foliage", synth::foliage(size)),
        ("roughness", synth::roughness(size)),
        ("normal", synth::normal_rocks(size)),
        ("ui_alpha", synth::ui_alpha(size)),
    ]
}

/// Decodes a block with the independent `texture2ddecoder` crate, returned as RGBA. Not for BC6H:
/// that crate's BC6H unquantization departs from the D3D specification (BC6H is checked against
/// texconv in `bc6h_reference.rs` instead).
pub fn reference_decode(format: Format, raw: &[u8]) -> [[u8; 4]; 16] {
    let mut buf = [0u32; 16];
    match format {
        Format::Bc1 => texture2ddecoder::decode_bc1_block(raw, &mut buf),
        Format::Bc3 => texture2ddecoder::decode_bc3_block(raw, &mut buf),
        Format::Bc4 => {
            buf = [0xFF00_0000; 16];
            texture2ddecoder::decode_bc4_block(raw, &mut buf);
        }
        Format::Bc5 => {
            buf = [0xFF00_0000; 16];
            texture2ddecoder::decode_bc5_block(raw, &mut buf);
        }
        Format::Bc7 => texture2ddecoder::decode_bc7_block(raw, &mut buf),
        Format::Bc6h => panic!("no independent BC6H reference decoder here"),
        Format::Etc2Rgb => texture2ddecoder::decode_etc2_rgb_block(raw, &mut buf),
        Format::Etc2Rgba => texture2ddecoder::decode_etc2_rgba8_block(raw, &mut buf),
        // texture2ddecoder's EAC R11 decoder departs from the Khronos specification, so these go
        // through the spec transcription below.
        Format::EacR11 => {
            let r = eac_r11(&raw[..8]);
            return std::array::from_fn(|i| [r[i], r[i], r[i], 255]);
        }
        Format::EacRg11 => {
            let (r, g) = (eac_r11(&raw[..8]), eac_r11(&raw[8..16]));
            return std::array::from_fn(|i| [r[i], g[i], 0, 255]);
        }
    }
    // texture2ddecoder packs pixels as little-endian BGRA.
    buf.map(|p| {
        let [b, g, r, a] = p.to_le_bytes();
        [r, g, b, a]
    })
}

/// EAC R11 (unsigned) per the Khronos Data Format Specification, section 21.3, rounded to 8 bits
/// as `(v * 255 + 1023) / 2047`, in raster order.
pub fn eac_r11(raw: &[u8]) -> [u8; 16] {
    const T: [[i32; 8]; 16] = [
        [-3, -6, -9, -15, 2, 5, 8, 14],
        [-3, -7, -10, -13, 2, 6, 9, 12],
        [-2, -5, -8, -13, 1, 4, 7, 12],
        [-2, -4, -6, -13, 1, 3, 5, 12],
        [-3, -6, -8, -12, 2, 5, 7, 11],
        [-3, -7, -9, -11, 2, 6, 8, 10],
        [-4, -7, -8, -11, 3, 6, 7, 10],
        [-3, -5, -8, -11, 2, 4, 7, 10],
        [-2, -6, -8, -10, 1, 5, 7, 9],
        [-2, -5, -8, -10, 1, 4, 7, 9],
        [-2, -4, -8, -10, 1, 3, 7, 9],
        [-2, -5, -7, -10, 1, 4, 6, 9],
        [-3, -4, -7, -10, 2, 3, 6, 9],
        [-1, -2, -3, -10, 0, 1, 2, 9],
        [-4, -6, -8, -9, 3, 5, 7, 8],
        [-3, -5, -7, -9, 2, 4, 6, 8],
    ];
    let base = i32::from(raw[0]);
    let mult = i32::from(raw[1] >> 4);
    let table = &T[usize::from(raw[1] & 15)];
    let bits = raw[2..8].iter().fold(0u64, |a, &b| a << 8 | u64::from(b));
    let mut out = [0u8; 16];
    for x in 0..4 {
        for y in 0..4 {
            let m = table[((bits >> (45 - 3 * (x * 4 + y))) & 7) as usize];
            let v = if mult == 0 {
                base * 8 + 4 + m
            } else {
                base * 8 + 4 + m * mult * 8
            };
            out[y * 4 + x] = ((v.clamp(0, 2047) * 255 + 1023) / 2047) as u8;
        }
    }
    out
}
