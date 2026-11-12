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

/// Decodes a block with the independent `texture2ddecoder` crate, returned as RGBA.
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
    }
    // texture2ddecoder packs pixels as little-endian BGRA.
    buf.map(|p| {
        let [b, g, r, a] = p.to_le_bytes();
        [r, g, b, a]
    })
}
