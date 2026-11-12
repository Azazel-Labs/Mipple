//! Every block we produce must decode the same way in an independent decoder, and our decoder
//! must agree with that decoder on arbitrary BC7 bitstreams (all 8 modes).

mod common;

use common::{corpus, reference_decode};
use mipple::pipeline::encode_base;
use mipple::{BlockOpts, Format, bc1, bc7};

fn channels(format: Format) -> &'static [usize] {
    match format {
        Format::Bc1 => &[0, 1, 2],
        Format::Bc4 => &[0],
        Format::Bc5 => &[0, 1],
        Format::Bc6h => &[],
        Format::Bc3 | Format::Bc7 => &[0, 1, 2, 3],
        Format::Etc2Rgb => &[0, 1, 2],
        Format::Etc2Rgba => &[0, 1, 2, 3],
        Format::EacR11 => &[0],
        Format::EacRg11 => &[0, 1],
    }
}

#[test]
fn encoded_blocks_match_reference_decoder() {
    for (name, img) in corpus(64) {
        for format in Format::ALL.into_iter().filter(|&f| f != Format::Bc6h) {
            for (effort, reduce_entropy) in [(0u8, false), (1, false), (5, false), (6, false), (5, true)] {
                let opts = BlockOpts { effort, reduce_entropy };
                let (_, blocks) = encode_base(&img, format, opts);
                let n = format.block_bytes();
                let mut ours = [[0u8; 4]; 16];
                for (bi, b) in blocks.iter().enumerate() {
                    format.decode_block(&b[..n], &mut ours);
                    let theirs = reference_decode(format, &b[..n]);
                    for i in 0..16 {
                        for &c in channels(format) {
                            assert_eq!(
                                ours[i][c],
                                theirs[i][c],
                                "{name} {format} {opts:?} block {bi} texel {i} ch {c}: ours {:?} ref {:?} raw {:02x?}",
                                ours[i],
                                theirs[i],
                                &b[..n]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Every ETC2 mode (individual, differential, T, H, planar) and EAC alpha on arbitrary bits, against
/// `texture2ddecoder`; EAC R11 against the spec transcription in `common`.
#[test]
fn etc2_and_eac_decoders_match_reference_on_random_blocks() {
    let mut seed = 0x51_7cc1_b727_220au64;
    let mut ours = [[0u8; 4]; 16];
    let mut modes = std::collections::HashSet::new();
    for i in 0..40000u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let mut raw = [0u8; 16];
        raw[..8].copy_from_slice(&seed.to_le_bytes());
        raw[8..].copy_from_slice(&seed.rotate_left(23).wrapping_mul(i | 1).to_le_bytes());
        if i % 5 == 0 {
            raw[1] &= 0x0f; // EAC multiplier 0
        }
        modes.insert(format!("{:?}", mipple::etc::mode(&raw[8..])));
        for format in [Format::Etc2Rgb, Format::Etc2Rgba, Format::EacR11, Format::EacRg11] {
            let n = format.block_bytes();
            let raw = if format == Format::Etc2Rgb {
                &raw[8..]
            } else {
                &raw[..n]
            };
            format.decode_block(raw, &mut ours);
            let theirs = reference_decode(format, raw);
            for t in 0..16 {
                for &c in channels(format) {
                    assert_eq!(ours[t][c], theirs[t][c], "{format} texel {t} ch {c} raw {raw:02x?}");
                }
            }
        }
    }
    assert_eq!(modes.len(), 5, "{modes:?}");
}

#[test]
fn bc7_decoder_matches_reference_on_random_blocks() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut ours = [[0u8; 4]; 16];
    for mode in 0..8u32 {
        for _ in 0..2000 {
            let mut v = next() as u128 | (next() as u128) << 64;
            v &= !((1u128 << (mode + 1)) - 1);
            v |= 1u128 << mode;
            let raw = v.to_le_bytes();
            bc7::decode(&raw, &mut ours);
            let theirs = reference_decode(Format::Bc7, &raw);
            assert_eq!(ours, theirs, "mode {mode} raw {raw:02x?}");
        }
    }
}

/// Alpha always matches; color matches on the four-safe color halves mipple writes. Other color
/// halves (c0 <= c1) are 4-color in the D3D spec, which mipple follows, while `texture2ddecoder`,
/// like some older hardware, reads them as BC1's 3-color mode.
#[test]
fn bc3_decoder_matches_reference_on_random_blocks() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut ours = [[0u8; 4]; 16];
    for i in 0..20000u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let mut raw = [0u8; 16];
        raw[..8].copy_from_slice(&seed.to_le_bytes());
        raw[8..].copy_from_slice(&seed.rotate_left(29).wrapping_mul(i | 1).to_le_bytes());
        if i % 7 == 0 {
            raw.copy_within(8..10, 10);
        }
        Format::Bc3.decode_block(&raw, &mut ours);
        let theirs = reference_decode(Format::Bc3, &raw);
        if bc1::is_four_safe(&raw[8..]) {
            assert_eq!(ours, theirs, "raw {raw:02x?}");
        } else {
            assert_eq!(ours.map(|p| p[3]), theirs.map(|p| p[3]), "raw {raw:02x?}");
        }
    }
}

#[test]
fn bc7_encoder_uses_multiple_modes() {
    // A sanity check that mode search is wired up: busy content should not be all mode 6.
    let img = mipple::synth::bricks(64);
    let (_, blocks) = encode_base(&img, Format::Bc7, 6);
    let mut hist = [0usize; 8];
    for b in &blocks {
        hist[bc7::unpack(b).unwrap().mode as usize] += 1;
    }
    let used = hist.iter().filter(|&&c| c > 0).count();
    assert!(used >= 3, "mode histogram {hist:?}");
}

#[test]
fn bc7_reduced_entropy_mode6_endpoints() {
    // Entropy reduction restricts mode-6 endpoints to 64 levels per channel with fixed p-bits,
    // keeps opaque blocks exactly opaque, and makes mode 6 the majority mode.
    let levels: Vec<u8> = (0..64).map(|j| ((j * 127 + 31) / 63) as u8).collect();
    for (name, img) in corpus(64) {
        let opts = BlockOpts {
            effort: 1,
            reduce_entropy: true,
        };
        let (src, blocks) = encode_base(&img, Format::Bc7, opts);
        let (_, plain) = encode_base(&img, Format::Bc7, 1);
        let count6 = |bs: &[[u8; 16]]| bs.iter().filter(|b| bc7::unpack(*b).unwrap().mode == 6).count();
        assert!(count6(&blocks) >= count6(&plain), "{name}: fewer mode-6 blocks");
        let mut dec = [[0u8; 4]; 16];
        for (b, s) in blocks.iter().zip(&src) {
            let blk = bc7::unpack(b).unwrap();
            if blk.mode != 6 {
                continue;
            }
            for e in 0..2 {
                for c in 0..4 {
                    assert!(levels.contains(&blk.ep[0][e][c]), "{name}: endpoint {:?}", blk.ep[0]);
                }
            }
            let opaque = s.iter().all(|t| t[3] == 255);
            let mut p = blk.pbits[0];
            p.sort_unstable(); // the anchor fix-up may swap endpoints
            if opaque {
                assert_eq!(p, [1, 1], "{name}: opaque mode-6 block needs p-bits (1, 1)");
            } else {
                assert!(p == [0, 1] || p == [1, 1], "{name}: p-bits {p:?}");
            }
            if opaque {
                bc7::decode(b, &mut dec);
                assert!(dec.iter().all(|t| t[3] == 255), "{name}: opaque block lost alpha");
            }
        }
    }
}
