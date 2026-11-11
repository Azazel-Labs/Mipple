//! KTX2 output must round-trip through our reader and parse correctly with the independent
//! `ktx2` crate, including a DFD identical to that crate's own generator.

mod common;

use mipple::pipeline::{self, EncodeOptions};
use mipple::{Format, ktx2, synth};

fn encode(format: Format, srgb: bool) -> mipple::EncodedTexture {
    let img = synth::bricks(72); // not a power of two, not a multiple of 16
    let mut opts = EncodeOptions::new(format);
    opts.effort = 0;
    opts.srgb = srgb;
    pipeline::encode(&img, &opts).0
}

#[test]
fn roundtrip_raw_and_zstd() {
    for format in Format::ALL {
        let tex = encode(format, format.default_srgb());
        assert_eq!(tex.levels.len(), 7); // 72, 36, 18, 9, 4, 2, 1
        for zstd in [None, Some(19)] {
            let bytes = ktx2::write(&tex, zstd, None, ktx2::PrepMode::Off).unwrap();
            let back = ktx2::read(&bytes).unwrap();
            assert_eq!(back, tex, "{format} zstd={zstd:?}");
        }
    }
}

#[test]
fn parses_with_ktx2_crate() {
    for format in Format::ALL {
        for srgb in [false, true] {
            if srgb && !format.supports_srgb() {
                continue;
            }
            let tex = encode(format, srgb);
            for zstd in [None, Some(3)] {
                let bytes = ktx2::write(&tex, zstd, Some("--test"), ktx2::PrepMode::Off).unwrap();
                let r = ::ktx2::Reader::new(&bytes[..]).expect("ktx2 crate rejected our file");
                let h = r.header();
                assert_eq!(h.format.map(|f| f.value()), Some(format.vk_format(srgb)));
                assert_eq!((h.pixel_width, h.pixel_height, h.pixel_depth), (72, 72, 0));
                assert_eq!((h.layer_count, h.face_count, h.level_count), (0, 1, 7));
                assert_eq!(h.type_size, 1);
                assert_eq!(h.supercompression_scheme.is_some(), zstd.is_some());
                // The spec stores KTXwriter as a NUL-terminated string; the crate returns it raw.
                let writer = r.writer().map(|s| s.trim_end_matches('\0'));
                assert_eq!(writer, Some(concat!("mipple v", env!("CARGO_PKG_VERSION"))));

                let dfd = r.basic_dfd().expect("basic DFD");
                let want_tf = if srgb {
                    ::ktx2::TransferFunction::SRGB
                } else {
                    ::ktx2::TransferFunction::Linear
                };
                assert_eq!(dfd.transfer_function, Some(want_tf));
                assert_eq!(
                    dfd.bytes_planes[0],
                    if zstd.is_some() { 0 } else { format.block_bytes() as u8 }
                );

                for (li, level) in r.levels().enumerate() {
                    let data = match zstd {
                        Some(_) => zstd::bulk::decompress(level.data, level.uncompressed_byte_length as usize).unwrap(),
                        None => level.data.to_vec(),
                    };
                    assert_eq!(data, tex.levels[li].data, "{format} level {li}");
                }
            }
        }
    }
}

#[test]
fn dfd_matches_ktx2_crate_generator() {
    for format in Format::ALL {
        for srgb in [false, true] {
            if srgb && !format.supports_srgb() {
                continue;
            }
            let vk = ::ktx2::Format::new(format.vk_format(srgb)).unwrap();
            let (basic, type_size) = ::ktx2::dfd::Basic::from_format(vk).unwrap();
            assert_eq!(type_size, 1);
            let ours = ktx2::build_dfd(format, srgb, false);
            // ours = dfdTotalSize (4) + block header (8) + basic block body.
            assert_eq!(&ours[12..], &basic.to_vec()[..], "{format} srgb={srgb}");
        }
    }
}

#[test]
fn mip_data_is_stored_smallest_first() {
    let tex = encode(Format::Bc7, true);
    let bytes = ktx2::write(&tex, None, None, ktx2::PrepMode::Off).unwrap();
    let h = ktx2::read_header(&bytes).unwrap();
    for w in h.levels.windows(2) {
        assert!(
            w[1].0 < w[0].0,
            "level offsets must decrease with level index: {:?}",
            h.levels
        );
    }
    // Uncompressed levels are aligned to the block size.
    for (off, _, _) in &h.levels {
        assert_eq!(off % 16, 0);
    }
}

#[test]
fn prep_roundtrips_through_ktx2() {
    let src = mipple::synth::bricks(64);
    for format in Format::ALL {
        let (tex, _) = pipeline::encode(&src, &EncodeOptions::new(format));
        for prep in [ktx2::PrepMode::On, ktx2::PrepMode::Auto] {
            let bytes = ktx2::write(&tex, Some(19), None, prep).unwrap();
            let back = ktx2::read(&bytes).unwrap();
            assert_eq!(back, tex, "{format:?} {prep:?}");
            let h = ktx2::read_header(&bytes).unwrap();
            let flagged = h.key_values.iter().any(|(k, _)| k == ktx2::PREP_KEY);
            assert!(
                flagged || prep == ktx2::PrepMode::Auto,
                "{format:?}: On must flag levels"
            );
        }
    }
}

/// Crafted headers must come back as errors: no panic, no allocation sized by the header.
#[test]
fn malformed_headers_are_errors() {
    let tex = encode(Format::Bc7, true);
    let put32 = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |b: &mut Vec<u8>, at: usize, v: u64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    for zstd in [None, Some(3)] {
        let good = ktx2::write(&tex, zstd, None, ktx2::PrepMode::Off).unwrap();
        let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
        // Level 0 offset near u64::MAX: offset + length wraps.
        let mut b = good.clone();
        put64(&mut b, 80, u64::MAX - 1);
        cases.push(("wrapping level offset", b));
        // A 2^32-1 square texture: its level size overflows and the zstd output would be huge.
        let mut b = good.clone();
        put32(&mut b, 20, u32::MAX);
        put32(&mut b, 24, u32::MAX);
        cases.push(("huge dimensions", b));
        // A huge uncompressed length must not size the output buffer.
        let mut b = good.clone();
        put64(&mut b, 96, u64::MAX);
        cases.push(("huge uncompressed length", b));
        // Key/value data running past the end of the file.
        let mut b = good.clone();
        put32(&mut b, 56, u32::MAX - 2);
        put32(&mut b, 60, 16);
        cases.push(("wrapping key/value range", b));
        for (what, b) in cases {
            if what == "huge uncompressed length" {
                // Ignored by the reader: the level decodes from its own size.
                assert_eq!(ktx2::read(&b).unwrap(), tex, "{what} zstd={zstd:?}");
            } else {
                assert!(ktx2::read(&b).is_err(), "{what} zstd={zstd:?}");
            }
        }
    }
    // A 1x1 texture claiming 40 levels (level indices past 31 must not shift a u32 by 32 or more),
    // each pointing at the same 16-byte block.
    let good = ktx2::write(&tex, None, None, ktx2::PrepMode::Off).unwrap();
    let n = 40;
    let data_at = 80 + 24 * n;
    let mut b = good[..80].to_vec();
    for (at, v) in [(20, 1), (24, 1), (40, n as u32), (44, 0), (56, 0), (60, 0)] {
        put32(&mut b, at, v);
    }
    for _ in 0..n {
        for v in [data_at as u64, 16, 16] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b.extend_from_slice(&[0u8; 16]);
    let back = ktx2::read(&b).unwrap();
    assert_eq!(back.levels.len(), n);
}
