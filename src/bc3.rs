//! BC3 (DXT5) RGBA block codec: a BC4 block for alpha (bytes 0..8) followed by a BC1 block for
//! color (bytes 8..16) that always decodes in 4-color mode.
//!
//! The color half is written so that it decodes the same as plain BC1 ([`bc1::is_four_safe`]),
//! so the BC1 kernels (error, re-index, refit) apply to it unchanged.

use crate::surface::{Block, Surface};
use crate::{bc1, bc4};

/// Encodes one block.
pub fn encode_block(px: &Block, effort: u8) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&bc4::encode_channel(&bc4::channel(px, 3), effort));
    let w = crate::format::Format::Bc1.error_weights();
    out[8..].copy_from_slice(&bc1::four_safe(bc1::encode_fast(px, effort), px, &w));
    out
}

/// Encodes a whole surface, reading texels straight from it. Returns raw 16-byte blocks in
/// raster order.
pub fn encode_surface(surface: &Surface, effort: u8) -> Vec<u8> {
    let (alpha, color) = rayon::join(
        || bc4::encode_surface_channel(surface, 3, effort),
        || bc1::encode_surface_four(surface, effort),
    );
    let mut out = Vec::with_capacity(alpha.len() * 2);
    for (a, c) in alpha.as_chunks::<8>().0.iter().zip(color.as_chunks::<8>().0) {
        out.extend_from_slice(a);
        out.extend_from_slice(c);
    }
    out
}

/// Decodes one 16-byte block.
pub fn decode(raw: &[u8], out: &mut Block) {
    bc1::decode4(&raw[8..16], out);
    let a = bc4::decode_channel(&raw[..8]);
    for (t, a) in out.iter_mut().zip(a) {
        t[3] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_matches_blocks_and_round_trips() {
        let mut s = Surface::new(36, 20);
        let mut seed = 0x9e37_79b9u32;
        for (i, p) in s.pixels.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let (x, y) = ((i % 36) as u32, (i / 36) as u32);
            *p = [
                (x * 7) as u8,
                (y * 11) as u8,
                (seed % 64) as u8,
                if (x / 4 + y / 4) % 3 == 0 {
                    255
                } else {
                    (x * 5 + y * 3) as u8
                },
            ];
        }
        for effort in [0, 5] {
            let raw = encode_surface(&s, effort);
            assert_eq!(raw.len(), s.block_count() * 16);
            let bw = s.blocks_x();
            for (k, b) in raw.as_chunks::<16>().0.iter().enumerate() {
                assert!(bc1::is_four_safe(&b[8..]));
                let px = s.read_block(k as u32 % bw, k as u32 / bw);
                let mut out = [[0u8; 4]; 16];
                decode(b, &mut out);
                for (t, o) in px.iter().zip(&out) {
                    for c in 0..4 {
                        assert!(t[c].abs_diff(o[c]) <= 40, "effort {effort} block {k}: {t:?} vs {o:?}");
                    }
                }
            }
        }
    }
}
