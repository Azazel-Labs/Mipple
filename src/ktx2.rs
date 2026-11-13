//! Minimal KTX 2.0 writer and reader for block-compressed 2D textures.
//!
//! Supports every [`Format`] (BCn, ETC2, EAC, ASTC LDR), full mip chains, and supercompression scheme 0 (none) or
//! 2 (Zstandard). Layout follows the KTX 2.0 specification:
//!
//! ```text
//! identifier | header | index | level index | DFD | key/value data | mip data (smallest first)
//! ```

// Header fields are untrusted: sizes and offsets go through checked conversions and arithmetic.
#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use crate::format::Format;
use crate::pipeline::{EncodedTexture, Level};
use anyhow::{Context, Result, bail, ensure};
use std::io::Read;

pub const IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
pub const SCHEME_NONE: u32 = 0;
pub const SCHEME_ZSTD: u32 = 2;

const HEADER_LEN: usize = 80;
const LEVEL_ENTRY_LEN: usize = 24;

/// KHR_DF color model for each format.
fn color_model(format: Format) -> u8 {
    match format {
        Format::Bc1 => 128,                                                           // KHR_DF_MODEL_BC1A
        Format::Bc3 => 130,                                                           // KHR_DF_MODEL_BC3
        Format::Bc4 => 131,                                                           // KHR_DF_MODEL_BC4
        Format::Bc5 => 132,                                                           // KHR_DF_MODEL_BC5
        Format::Bc6h => 133,                                                          // KHR_DF_MODEL_BC6H
        Format::Bc7 => 134,                                                           // KHR_DF_MODEL_BC7
        Format::Etc2Rgb | Format::Etc2Rgba | Format::EacR11 | Format::EacRg11 => 161, // KHR_DF_MODEL_ETC2
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 162,                   // KHR_DF_MODEL_ASTC
    }
}

/// Builds the Data Format Descriptor (including the leading dfdTotalSize field).
pub fn build_dfd(format: Format, srgb: bool, supercompressed: bool) -> Vec<u8> {
    // (bitOffset, bitLength - 1, channelType)
    let samples: &[(u16, u8, u8)] = match format {
        Format::Bc1 => &[(0, 63, 0)],
        // Alpha (KHR_DF_CHANNEL_BC3_ALPHA), linear even in an sRGB texture, then color.
        Format::Bc3 if srgb => &[(0, 63, 15 | 0x10), (64, 63, 0)],
        Format::Bc3 => &[(0, 63, 15), (64, 63, 0)],
        Format::Bc4 => &[(0, 63, 0)],
        Format::Bc5 => &[(0, 63, 0), (64, 63, 1)],
        // KHR_DF_SAMPLE_DATATYPE_FLOAT; unsigned.
        Format::Bc6h => &[(0, 127, 0x80)],
        Format::Bc7 => &[(0, 127, 0)],
        // KHR_DF_CHANNEL_ETC2_COLOR.
        Format::Etc2Rgb => &[(0, 63, 2)],
        // KHR_DF_CHANNEL_ETC2_ALPHA (linear even in an sRGB texture), then color.
        Format::Etc2Rgba if srgb => &[(0, 63, 15 | 0x10), (64, 63, 2)],
        Format::Etc2Rgba => &[(0, 63, 15), (64, 63, 2)],
        // KHR_DF_CHANNEL_ETC2_RED and _GREEN.
        Format::EacR11 => &[(0, 63, 0)],
        Format::EacRg11 => &[(0, 63, 0), (64, 63, 1)],
        // KHR_DF_CHANNEL_ASTC_DATA; LDR, so not float.
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => &[(0, 127, 0)],
    };
    let block_size = 24 + 16 * samples.len();
    let mut v = Vec::with_capacity(4 + block_size);
    v.extend_from_slice(&u32_field(4 + block_size).to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // vendorId = Khronos, descriptorType = basic
    v.extend_from_slice(&2u16.to_le_bytes()); // versionNumber
    v.extend_from_slice(&u16::try_from(block_size).expect("DFD block is small").to_le_bytes());
    v.push(color_model(format));
    v.push(1); // colorPrimaries = BT709
    v.push(if srgb { 2 } else { 1 }); // transferFunction: sRGB or linear
    v.push(0); // flags: straight alpha
    // texelBlockDimension, stored minus one: 4x4, or the ASTC footprint.
    let (bw, bh) = format.block_dims();
    let dim = |d: u32| u8::try_from(d - 1).expect("block dimensions are at most 12");
    v.extend_from_slice(&[dim(bw), dim(bh), 0, 0]);
    // bytesPlane0 is 0 ("unsized") when the data is supercompressed.
    let bytes_plane0 = if supercompressed {
        0
    } else {
        u8::try_from(format.block_bytes()).expect("8 or 16")
    };
    v.extend_from_slice(&[bytes_plane0, 0, 0, 0, 0, 0, 0, 0]);
    for &(offset, len, channel) in samples {
        v.extend_from_slice(&offset.to_le_bytes());
        v.push(len);
        v.push(channel);
        v.extend_from_slice(&[0, 0, 0, 0]); // samplePosition
        // sampleLower and sampleUpper: the full integer range, or 0.0 to 1.0 for float data.
        let (lower, upper) = if channel & 0x80 != 0 {
            (0u32, 1f32.to_bits())
        } else {
            (0, u32::MAX)
        };
        v.extend_from_slice(&lower.to_le_bytes());
        v.extend_from_slice(&upper.to_le_bytes());
    }
    v
}

fn push_kv(out: &mut Vec<u8>, key: &str, value: &str) {
    let len = key.len() + 1 + value.len() + 1;
    out.extend_from_slice(&u32_field(len).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.push(0);
    out.extend_from_slice(value.as_bytes());
    out.push(0);
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

/// A 32-bit header field (level count, DFD and key/value offsets and lengths): these sit before the
/// level data and stay small.
fn u32_field(v: usize) -> u32 {
    u32::try_from(v).expect("KTX2 header field fits in 32 bits")
}

fn round_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

/// Whether [`write()`] applies the [`crate::prep`] transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum PrepMode {
    /// Standard KTX2: raw blocks under Zstd.
    #[default]
    Off,
    /// Every level field-split before Zstd (needs mipple's reader or an equivalent inverse).
    On,
    /// Per level, only when the split makes it smaller.
    Auto,
}

/// Key/value entry listing the levels stored with the [`crate::prep`] transform under Zstd: one
/// `0`/`1` per level, level 0 first. Not part of the KTX2 standard: other readers will not undo it.
pub const PREP_KEY: &str = "mipple.prep";

/// Serializes a texture. `zstd_level = None` stores raw blocks; `Some(level)` applies Zstd per mip.
/// With Zstd, `prep` applies the lossless [`crate::prep`] field split before compressing (`Auto`:
/// per level, only when it makes the level smaller).
pub fn write(
    tex: &EncodedTexture,
    zstd_level: Option<i32>,
    writer_params: Option<&str>,
    prep: PrepMode,
) -> Result<Vec<u8>> {
    ensure!(!tex.levels.is_empty(), "texture has no levels");
    let supercompressed = zstd_level.is_some();
    let mut prepped = vec![false; tex.levels.len()];
    let payloads: Vec<Vec<u8>> = tex
        .levels
        .iter()
        .enumerate()
        .map(|(li, l)| match zstd_level {
            Some(z) => {
                let plain = zstd::bulk::compress(&l.data, z).context("zstd compression failed")?;
                if prep == PrepMode::Off {
                    return Ok(plain);
                }
                let t = zstd::bulk::compress(&crate::prep::forward(tex.format, &l.data), z)
                    .context("zstd compression failed")?;
                if prep == PrepMode::On || t.len() < plain.len() {
                    prepped[li] = true;
                    Ok(t)
                } else {
                    Ok(plain)
                }
            }
            None => Ok(l.data.clone()),
        })
        .collect::<Result<_>>()?;

    let n = tex.levels.len();
    let dfd = build_dfd(tex.format, tex.srgb, supercompressed);
    let mut kvd = Vec::new();
    // Keys must be sorted by their byte values.
    push_kv(&mut kvd, "KTXwriter", &format!("mipple v{}", env!("CARGO_PKG_VERSION")));
    if let Some(p) = writer_params {
        push_kv(&mut kvd, "KTXwriterScParams", p);
    }
    if prepped.iter().any(|&p| p) {
        let flags: String = prepped.iter().map(|&p| if p { '1' } else { '0' }).collect();
        push_kv(&mut kvd, PREP_KEY, &flags);
    }

    let dfd_off = HEADER_LEN + LEVEL_ENTRY_LEN * n;
    let kvd_off = dfd_off + dfd.len();
    let data_start = kvd_off + kvd.len();
    let align = if supercompressed {
        1
    } else {
        // lcm(block size, 4); block sizes are 8 or 16, both multiples of 4.
        tex.format.block_bytes()
    };

    // Mip data is stored smallest level first.
    let mut offsets = vec![0usize; n];
    let mut cur = data_start;
    for li in (0..n).rev() {
        cur = round_up(cur, align);
        offsets[li] = cur;
        cur += payloads[li].len();
    }

    let mut out = vec![0u8; cur];
    let put32 = |o: &mut Vec<u8>, at: usize, v: u32| o[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |o: &mut Vec<u8>, at: usize, v: u64| o[at..at + 8].copy_from_slice(&v.to_le_bytes());

    out[..12].copy_from_slice(&IDENTIFIER);
    put32(&mut out, 12, tex.format.vk_format(tex.srgb));
    put32(&mut out, 16, 1); // typeSize: 1 for block-compressed formats
    put32(&mut out, 20, tex.width);
    put32(&mut out, 24, tex.height);
    put32(&mut out, 28, 0); // pixelDepth
    put32(&mut out, 32, 0); // layerCount
    put32(&mut out, 36, 1); // faceCount
    put32(&mut out, 40, u32_field(n));
    put32(&mut out, 44, if supercompressed { SCHEME_ZSTD } else { SCHEME_NONE });
    put32(&mut out, 48, u32_field(dfd_off));
    put32(&mut out, 52, u32_field(dfd.len()));
    put32(&mut out, 56, if kvd.is_empty() { 0 } else { u32_field(kvd_off) });
    put32(&mut out, 60, u32_field(kvd.len()));
    put64(&mut out, 64, 0); // sgdByteOffset
    put64(&mut out, 72, 0); // sgdByteLength

    for li in 0..n {
        let at = HEADER_LEN + LEVEL_ENTRY_LEN * li;
        put64(&mut out, at, offsets[li] as u64);
        put64(&mut out, at + 8, payloads[li].len() as u64);
        put64(&mut out, at + 16, tex.levels[li].data.len() as u64);
    }
    out[dfd_off..dfd_off + dfd.len()].copy_from_slice(&dfd);
    out[kvd_off..kvd_off + kvd.len()].copy_from_slice(&kvd);
    for li in 0..n {
        out[offsets[li]..offsets[li] + payloads[li].len()].copy_from_slice(&payloads[li]);
    }
    Ok(out)
}

/// Header fields, for `mipple info`.
#[derive(Clone, Debug)]
pub struct Header {
    pub vk_format: u32,
    pub width: u32,
    pub height: u32,
    pub level_count: u32,
    pub scheme: u32,
    /// (byte offset, byte length, uncompressed length) per level.
    pub levels: Vec<(u64, u64, u64)>,
    pub key_values: Vec<(String, String)>,
}

/// Parses the header, level index and key/value data.
pub fn read_header(bytes: &[u8]) -> Result<Header> {
    ensure!(
        bytes.len() >= HEADER_LEN && bytes[..12] == IDENTIFIER,
        "not a KTX2 file"
    );
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let level_count = u32_at(40);
    let n = level_count.max(1) as usize;
    ensure!(bytes.len() >= HEADER_LEN + LEVEL_ENTRY_LEN * n, "truncated level index");
    let levels = (0..n)
        .map(|li| {
            let at = HEADER_LEN + LEVEL_ENTRY_LEN * li;
            (u64_at(at), u64_at(at + 8), u64_at(at + 16))
        })
        .collect();
    let (kvd_off, kvd_len) = (u32_at(56) as usize, u32_at(60) as usize);
    let mut key_values = Vec::new();
    if kvd_len > 0 {
        let kvd = kvd_off
            .checked_add(kvd_len)
            .and_then(|end| bytes.get(kvd_off..end))
            .context("truncated key/value data")?;
        let mut p = 0;
        while p + 4 <= kvd.len() {
            let len = u32::from_le_bytes(kvd[p..p + 4].try_into().unwrap()) as usize;
            let end = (p + 4).saturating_add(len);
            let entry = &kvd[p + 4..end.min(kvd.len())];
            if let Some(z) = entry.iter().position(|&b| b == 0) {
                let key = String::from_utf8_lossy(&entry[..z]).into_owned();
                let val = entry[z + 1..].strip_suffix(&[0]).unwrap_or(&entry[z + 1..]);
                key_values.push((key, String::from_utf8_lossy(val).into_owned()));
            }
            if end >= kvd.len() {
                break;
            }
            p = round_up(end, 4);
        }
    }
    Ok(Header {
        vk_format: u32_at(12),
        width: u32_at(20),
        height: u32_at(24),
        level_count,
        scheme: u32_at(44),
        levels,
        key_values,
    })
}

/// Reads a KTX2 file written by mipple (or any 2D BC1/BC4/BC5/BC7 KTX2, scheme 0 or 2).
pub fn read(bytes: &[u8]) -> Result<EncodedTexture> {
    let h = read_header(bytes)?;
    let (format, srgb) =
        Format::from_vk(h.vk_format).with_context(|| format!("unsupported vkFormat {}", h.vk_format))?;
    if h.scheme != SCHEME_NONE && h.scheme != SCHEME_ZSTD {
        bail!("unsupported supercompression scheme {}", h.scheme);
    }
    let prep: Vec<bool> = h
        .key_values
        .iter()
        .find(|(k, _)| k == PREP_KEY)
        .map(|(_, v)| v.chars().map(|c| c == '1').collect())
        .unwrap_or_default();
    let mut levels = Vec::with_capacity(h.levels.len());
    for (li, &(off, len, _)) in h.levels.iter().enumerate() {
        let shift = u32::try_from(li).unwrap_or(u32::MAX);
        let width = h.width.checked_shr(shift).unwrap_or(0).max(1);
        let height = h.height.checked_shr(shift).unwrap_or(0).max(1);
        let expected = format
            .level_bytes(width, height)
            .with_context(|| format!("level {li}: {width}x{height} is too large"))?;
        let raw = usize::try_from(off)
            .ok()
            .zip(usize::try_from(len).ok())
            .and_then(|(off, len)| bytes.get(off..off.checked_add(len)?))
            .with_context(|| format!("level {li} out of bounds"))?;
        let prepped = prep.get(li).copied().unwrap_or(false);
        let mut data = if h.scheme == SCHEME_ZSTD {
            // At most the level's size (a prepped level inflates to its larger streams), read as a
            // stream so memory follows the actual output, not a size the header claims.
            let cap = if prepped {
                crate::prep::max_len(expected)
            } else {
                expected
            };
            let mut out = Vec::new();
            zstd::stream::read::Decoder::with_buffer(raw)
                .and_then(|d| d.take(cap as u64 + 1).read_to_end(&mut out))
                .with_context(|| format!("level {li}: zstd"))?;
            out
        } else {
            raw.to_vec()
        };
        if prepped {
            data =
                crate::prep::inverse(format, &data).with_context(|| format!("level {li}: malformed prep streams"))?;
        }
        ensure!(
            data.len() == expected,
            "level {li}: {} bytes, expected {expected}",
            data.len()
        );
        levels.push(Level { width, height, data });
    }
    Ok(EncodedTexture {
        format,
        srgb,
        width: h.width,
        height: h.height,
        levels,
    })
}
