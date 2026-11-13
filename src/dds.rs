//! Minimal DDS reader/writer for BC1/BC3/BC4/BC5/BC6H/BC7, so other encoders' output (bc7enc_rdo,
//! bc7f harness, texconv...) can be scored with the same metrics and back ends as mipple.
//!
//! Reads legacy FourCC (DXT1, DXT4/DXT5, ATI1/BC4U, ATI2/BC5U) and DX10-extended headers; writes DX10.
//! DXGI has no ETC2 or EAC formats and Direct3D does not use its ASTC values, so those are KTX2
//! only (ASTC can also go to a `.astc` file) and [`write`] rejects them ([`supports`]).

// Header fields are untrusted: sizes and offsets go through checked conversions and arithmetic.
#![warn(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]

use crate::format::Format;
use crate::pipeline::{EncodedTexture, Level};
use anyhow::{Context, Result, bail, ensure};

const MAGIC: &[u8; 4] = b"DDS ";
const HEADER_LEN: usize = 124;
const DX10_LEN: usize = 20;

fn dxgi(format: Format, srgb: bool) -> Option<u32> {
    Some(match (format, srgb) {
        (Format::Bc1, false) => 71,
        (Format::Bc1, true) => 72,
        (Format::Bc3, false) => 77,
        (Format::Bc3, true) => 78,
        (Format::Bc4, _) => 80,
        (Format::Bc5, _) => 83,
        (Format::Bc6h, _) => 95,
        (Format::Bc7, false) => 98,
        (Format::Bc7, true) => 99,
        (Format::Etc2Rgb | Format::Etc2Rgba | Format::EacR11 | Format::EacRg11, _) => return None,
        (Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8, _) => return None,
    })
}

/// Whether [`write`] can store `format` (the BC formats; not ETC2, EAC or ASTC).
pub fn supports(format: Format) -> bool {
    dxgi(format, false).is_some()
}

fn from_dxgi(v: u32) -> Option<(Format, bool)> {
    Some(match v {
        70..=71 => (Format::Bc1, false),
        72 => (Format::Bc1, true),
        76..=77 => (Format::Bc3, false),
        78 => (Format::Bc3, true),
        79..=80 => (Format::Bc4, false),
        82..=83 => (Format::Bc5, false),
        94..=95 => (Format::Bc6h, false),
        97..=98 => (Format::Bc7, false),
        99 => (Format::Bc7, true),
        _ => return None,
    })
}

/// A header size or count that is structurally small (a header length, a mip count).
fn u32_field(v: usize) -> u32 {
    u32::try_from(v).expect("DDS header field fits in 32 bits")
}

/// Writes a DX10-header DDS containing every level of `tex`. Fails for the formats DDS does not
/// carry ([`supports`]: ETC2, EAC and ASTC; use KTX2).
pub fn write(tex: &EncodedTexture) -> Result<Vec<u8>> {
    let Some(dxgi) = dxgi(tex.format, tex.srgb) else {
        bail!(
            "DDS has no {} format (it carries BC formats only, not ETC2, EAC or ASTC); write KTX2 instead",
            tex.format
        );
    };
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    let mut h = [0u32; 31];
    h[0] = u32_field(HEADER_LEN);
    // CAPS | HEIGHT | WIDTH | PIXELFORMAT | MIPMAPCOUNT | LINEARSIZE
    h[1] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000;
    h[2] = tex.height;
    h[3] = tex.width;
    // pitchOrLinearSize is advisory: saturate for levels of 4 GiB or more.
    h[4] = u32::try_from(tex.levels[0].data.len()).unwrap_or(u32::MAX);
    h[6] = u32_field(tex.levels.len());
    // DDS_PIXELFORMAT at dword 18: size, flags (FOURCC), fourCC "DX10"
    h[18] = 32;
    h[19] = 0x4;
    h[20] = u32::from_le_bytes(*b"DX10");
    // caps: TEXTURE | MIPMAP | COMPLEX (the latter two only with mips)
    h[26] = 0x1000 | if tex.levels.len() > 1 { 0x40_0008 } else { 0 };
    for v in h {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [dxgi, 3, 0, 1, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for l in &tex.levels {
        out.extend_from_slice(&l.data);
    }
    Ok(out)
}

/// Reads all levels present in a DDS file.
pub fn read(bytes: &[u8]) -> Result<EncodedTexture> {
    ensure!(bytes.len() >= 4 + HEADER_LEN && &bytes[..4] == MAGIC, "not a DDS file");
    let dw = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    let (height, width) = (dw(2), dw(3));
    // At most 32 levels: a u32 dimension halves to 1 by then.
    let mip_count = dw(6).clamp(1, 32);
    let fourcc = dw(20).to_le_bytes();
    let mut offset = 4 + HEADER_LEN;
    let (format, srgb) = match &fourcc {
        b"DX10" => {
            ensure!(bytes.len() >= offset + DX10_LEN, "truncated DX10 header");
            let fmt = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += DX10_LEN;
            match from_dxgi(fmt) {
                Some(f) => f,
                None => bail!("unsupported DXGI format {fmt}"),
            }
        }
        b"DXT1" => (Format::Bc1, false),
        b"DXT4" | b"DXT5" => (Format::Bc3, false),
        b"ATI1" | b"BC4U" => (Format::Bc4, false),
        b"ATI2" | b"BC5U" => (Format::Bc5, false),
        other => bail!("unsupported DDS FourCC {:?}", String::from_utf8_lossy(other)),
    };
    let mut levels = Vec::new();
    for li in 0..mip_count {
        let (w, h) = ((width >> li).max(1), (height >> li).max(1));
        let n = format
            .level_bytes(w, h)
            .with_context(|| format!("DDS level {li}: {w}x{h} is too large"))?;
        let Some(data) = offset.checked_add(n).and_then(|end| bytes.get(offset..end)) else {
            ensure!(li > 0, "DDS truncated: level 0 needs {n} bytes");
            break; // tolerate files that claim more mips than they hold
        };
        levels.push(Level {
            width: w,
            height: h,
            data: data.to_vec(),
        });
        offset += n;
    }
    Ok(EncodedTexture {
        format,
        srgb,
        width,
        height,
        levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{self, EncodeOptions};

    /// Crafted headers come back as errors or clamped, never as a panic.
    #[test]
    fn malformed_headers() {
        let img = crate::synth::foliage(8);
        let mut o = EncodeOptions::new(Format::Bc7);
        o.effort = 0;
        let good = write(&pipeline::encode(&img, &o).0).unwrap();
        let set =
            |b: &mut [u8], dword: usize, v: u32| b[4 + 4 * dword..8 + 4 * dword].copy_from_slice(&v.to_le_bytes());
        // A 2^32-1 square: the level size overflows usize arithmetic.
        let mut b = good.clone();
        set(&mut b, 2, u32::MAX);
        set(&mut b, 3, u32::MAX);
        assert!(read(&b).is_err());
        // 2^32-1 mips of a 1x1 texture: clamped, and the shifts stay below 32.
        let mut b = good.clone();
        set(&mut b, 2, 1);
        set(&mut b, 3, 1);
        set(&mut b, 6, u32::MAX);
        b.extend_from_slice(&[0u8; 16 * 40]);
        assert_eq!(read(&b).unwrap().levels.len(), 32);
    }

    #[test]
    fn roundtrip() {
        for format in Format::ALL {
            let img = crate::synth::foliage(40);
            let mut o = EncodeOptions::new(format);
            o.effort = 0;
            let (tex, _) = pipeline::encode(&img, &o);
            if !supports(format) {
                assert!(write(&tex).is_err(), "{format}");
                continue;
            }
            let back = read(&write(&tex).unwrap()).unwrap();
            assert_eq!(back, tex, "{format}");
        }
    }
}
