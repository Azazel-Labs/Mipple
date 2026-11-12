//! BC6H (BPTC float) block codec: 16-byte blocks of RGB half floats, for HDR color.
//!
//! The encoder writes BC6H_UF16 (unsigned); the decoder reads both the unsigned and the signed
//! variant. Errors are measured on the half-float bit patterns, which grow like the logarithm of
//! the value (1024 steps per octave), so a squared error there is a relative error: a dark texel
//! counts as much as a bright one with the same ratio of error to value. That is also the domain
//! the format interpolates in.
//!
//! The bit layouts of the 14 modes ([`LAYOUT`]) were generated from the D3D11 specification's mode
//! tables and are checked against Microsoft's reference decoder (texconv) in
//! `tests/bc6h_reference.rs`.

use crate::bc7::{anchors, subset_of};
use crate::fit::V4;

/// Half-float bits (R, G, B) of one texel.
pub type Half3 = [u16; 3];
/// The 16 texels of a 4x4 block in raster order.
pub type HdrBlock = [Half3; 16];

/// Largest finite half, 65504.
pub const HALF_MAX: u16 = 0x7bff;

/// `f` as the nearest half float (round to nearest, ties to even).
pub fn f32_to_half(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let a = x & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        // Infinity, or NaN (kept quiet).
        return sign | 0x7c00 | if a > 0x7f80_0000 { 0x200 } else { 0 };
    }
    if a >= 0x477f_f000 {
        // 65520 and up round to infinity.
        return sign | 0x7c00;
    }
    if a < 0x3880_0000 {
        // Below the smallest normal half, 2^-14: a multiple of 2^-24.
        return sign | (f32::from_bits(a) * 16_777_216.0).round_ties_even() as u16;
    }
    let mut h = (((a >> 23) - 112) << 10) | ((a >> 13) & 0x3ff);
    let rest = a & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The value of half float `h`.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);
    let bits = match e {
        0 => return f32::from_bits(sign | (m as f32 * (1.0 / 16_777_216.0)).to_bits()),
        31 => sign | 0x7f80_0000 | (m << 13),
        _ => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// The half an unsigned BC6H block can hold nearest `f`: negative values and NaN become 0, values
/// past [`HALF_MAX`] become it.
pub fn f32_to_uf16(f: f32) -> u16 {
    if f.is_nan() || f <= 0.0 {
        0
    } else {
        f32_to_half(f).min(HALF_MAX)
    }
}

#[derive(Clone, Copy, Debug)]
struct Mode {
    /// The 2- or 5-bit value at the start of the block.
    value: u8,
    mode_bits: u8,
    /// Endpoints after the first are stored as deltas from it.
    transformed: bool,
    two_regions: bool,
    /// Bits of the first endpoint.
    eb: u8,
    /// Bits of every other endpoint, per channel (equal to `eb` when not transformed).
    db: [u8; 3],
}

impl Mode {
    const fn new(value: u8, transformed: bool, two_regions: bool, eb: u8, db: [u8; 3]) -> Self {
        let mode_bits = if value < 2 { 2 } else { 5 };
        Self {
            value,
            mode_bits,
            transformed,
            two_regions,
            eb,
            db,
        }
    }

    fn regions(self) -> usize {
        if self.two_regions { 2 } else { 1 }
    }
}

/// The 14 modes in D3D order (mode 1 at index 0).
const MODES: [Mode; 14] = [
    Mode::new(0, true, true, 10, [5, 5, 5]),
    Mode::new(1, true, true, 7, [6, 6, 6]),
    Mode::new(2, true, true, 11, [5, 4, 4]),
    Mode::new(6, true, true, 11, [4, 5, 4]),
    Mode::new(10, true, true, 11, [4, 4, 5]),
    Mode::new(14, true, true, 9, [5, 5, 5]),
    Mode::new(18, true, true, 8, [6, 5, 5]),
    Mode::new(22, true, true, 8, [5, 6, 5]),
    Mode::new(26, true, true, 8, [5, 5, 6]),
    Mode::new(30, false, true, 6, [6, 6, 6]),
    Mode::new(3, false, false, 10, [10, 10, 10]),
    Mode::new(7, true, false, 11, [9, 9, 9]),
    Mode::new(11, true, false, 12, [8, 8, 8]),
    Mode::new(15, true, false, 16, [4, 4, 4]),
];

/// Index into [`MODES`] of the mode whose bits open `raw`, or `None` for the four reserved values.
fn mode_index(raw: u128) -> Option<usize> {
    let v = if raw & 2 == 0 { raw & 3 } else { raw & 31 } as u8;
    MODES.iter().position(|m| m.value == v)
}

/// Where each header bit after the mode bits goes, per mode in [`MODES`] order: `field << 4 | bit`
/// with `field = 3 * endpoint + channel`, endpoints w, x, y, z (subset 0's pair, then subset 1's)
/// and channels R, G, B.
#[rustfmt::skip]
const LAYOUT: [&[u8]; 14] = [
    // mode 1 (mode value 0)
    &[
        116, 132, 180, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 176, 160, 161, 162, 163, 80,
        81, 82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ],
    // mode 2 (mode value 1)
    &[
        117, 164, 165, 0, 1, 2, 3, 4, 5, 6, 176, 177, 132, 16, 17, 18, 19, 20, 21, 22, 133, 178, 116, 32, 33, 34, 35,
        36, 37, 38, 179, 181, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115, 64, 65, 66, 67, 68, 69, 160, 161, 162,
        163, 80, 81, 82, 83, 84, 85, 128, 129, 130, 131, 96, 97, 98, 99, 100, 101, 144, 145, 146, 147, 148, 149,
    ],
    // mode 3 (mode value 2)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 52, 10, 112, 113, 114, 115, 64, 65, 66, 67, 26, 176, 160, 161, 162, 163, 80, 81, 82, 83, 42,
        177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ],
    // mode 4 (mode value 6)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 10, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 26, 160, 161, 162, 163, 80, 81, 82, 83, 42,
        177, 128, 129, 130, 131, 96, 97, 98, 99, 176, 178, 144, 145, 146, 147, 116, 179,
    ],
    // mode 5 (mode value 10)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 10, 132, 112, 113, 114, 115, 64, 65, 66, 67, 26, 176, 160, 161, 162, 163, 80, 81, 82, 83, 84,
        42, 128, 129, 130, 131, 96, 97, 98, 99, 177, 178, 144, 145, 146, 147, 180, 179,
    ],
    // mode 6 (mode value 14)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 132, 16, 17, 18, 19, 20, 21, 22, 23, 24, 116, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 176, 160, 161, 162, 163, 80, 81, 82,
        83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ],
    // mode 7 (mode value 18)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 164, 132, 16, 17, 18, 19, 20, 21, 22, 23, 178, 116, 32, 33, 34, 35, 36, 37, 38, 39,
        179, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115, 64, 65, 66, 67, 68, 176, 160, 161, 162, 163, 80, 81,
        82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 101, 144, 145, 146, 147, 148, 149,
    ],
    // mode 8 (mode value 22)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 176, 132, 16, 17, 18, 19, 20, 21, 22, 23, 117, 116, 32, 33, 34, 35, 36, 37, 38, 39,
        165, 180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 69, 160, 161, 162, 163, 80, 81,
        82, 83, 84, 177, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ],
    // mode 9 (mode value 26)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 177, 132, 16, 17, 18, 19, 20, 21, 22, 23, 133, 116, 32, 33, 34, 35, 36, 37, 38, 39,
        181, 180, 48, 49, 50, 51, 52, 164, 112, 113, 114, 115, 64, 65, 66, 67, 68, 176, 160, 161, 162, 163, 80, 81,
        82, 83, 84, 85, 128, 129, 130, 131, 96, 97, 98, 99, 100, 178, 144, 145, 146, 147, 148, 179,
    ],
    // mode 10 (mode value 30)
    &[
        0, 1, 2, 3, 4, 5, 164, 176, 177, 132, 16, 17, 18, 19, 20, 21, 117, 133, 178, 116, 32, 33, 34, 35, 36, 37,
        165, 179, 181, 180, 48, 49, 50, 51, 52, 53, 112, 113, 114, 115, 64, 65, 66, 67, 68, 69, 160, 161, 162, 163,
        80, 81, 82, 83, 84, 85, 128, 129, 130, 131, 96, 97, 98, 99, 100, 101, 144, 145, 146, 147, 148, 149,
    ],
    // mode 11 (mode value 3)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 80, 81, 82, 83, 84, 85, 86,
        87, 88, 89,
    ],
    // mode 12 (mode value 7)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 10, 64, 65, 66, 67, 68, 69, 70, 71, 72, 26, 80, 81, 82, 83, 84, 85, 86,
        87, 88, 42,
    ],
    // mode 13 (mode value 11)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 52, 53, 54, 55, 11, 10, 64, 65, 66, 67, 68, 69, 70, 71, 27, 26, 80, 81, 82, 83, 84, 85, 86,
        87, 43, 42,
    ],
    // mode 14 (mode value 15)
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        48, 49, 50, 51, 15, 14, 13, 12, 11, 10, 64, 65, 66, 67, 31, 30, 29, 28, 27, 26, 80, 81, 82, 83, 47, 46, 45,
        44, 43, 42,
    ],
];

const W3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// A run of header bits: `len` bits of field `field` from bit `bit` up, stored consecutively.
#[derive(Clone, Copy)]
struct Run {
    field: u8,
    bit: u8,
    len: u8,
}

/// [`LAYOUT`] as runs, per mode: the runs and their count.
static RUNS: [([Run; 40], usize); 14] = {
    let mut out = [(
        [Run {
            field: 0,
            bit: 0,
            len: 0,
        }; 40],
        0,
    ); 14];
    let mut mi = 0;
    while mi < 14 {
        let l = LAYOUT[mi];
        let mut n = 0;
        let mut i = 0;
        while i < l.len() {
            let (f, b) = (l[i] >> 4, l[i] & 15);
            let mut len = 1;
            while i + len < l.len() && l[i + len] >> 4 == f && (l[i + len] & 15) as usize == b as usize + len {
                len += 1;
            }
            out[mi].0[n] = Run {
                field: f,
                bit: b,
                len: len as u8,
            };
            n += 1;
            i += len;
        }
        out[mi].1 = n;
        mi += 1;
    }
    out
};

/// A BC6H block with its fields unpacked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bc6hBlock {
    /// Index of the mode, 0..14 (the D3D mode number less one).
    pub mode: u8,
    /// Partition, 0..32 (two-region modes only).
    pub partition: u8,
    /// Endpoint fields as stored, `[w, x, y, z][channel]`: deltas from w (two's complement in the
    /// field's width) for transformed modes.
    pub fields: [[u16; 3]; 4],
    pub idx: [u8; 16],
}

/// Anchor texel of each region (texel 0, and the second region's from the BC7 table).
fn region_anchors(mode: Mode, partition: u8) -> [usize; 2] {
    if mode.two_regions {
        let a = anchors(2, partition);
        [a[0], a[1]]
    } else {
        [0, usize::MAX]
    }
}

fn region_of(mode: Mode, partition: u8, i: usize) -> usize {
    if mode.two_regions {
        subset_of(2, partition, i)
    } else {
        0
    }
}

/// Unpacks a block, or `None` for a reserved mode.
pub fn unpack(raw: &[u8]) -> Option<Bc6hBlock> {
    let v = u128::from_le_bytes(raw[..16].try_into().expect("16-byte block"));
    let mi = mode_index(v)?;
    let m = MODES[mi];
    let mut b = Bc6hBlock {
        mode: mi as u8,
        ..Default::default()
    };
    let mut pos = u32::from(m.mode_bits);
    let (runs, n) = &RUNS[mi];
    for r in &runs[..*n] {
        let f = usize::from(r.field);
        b.fields[f / 3][f % 3] |= (((v >> pos) as u16) & ((1 << r.len) - 1)) << r.bit;
        pos += u32::from(r.len);
    }
    let ib = if m.two_regions {
        b.partition = ((v >> pos) & 31) as u8;
        pos += 5;
        3
    } else {
        4
    };
    let anchor = region_anchors(m, b.partition);
    for (i, x) in b.idx.iter_mut().enumerate() {
        let n = if anchor.contains(&i) { ib - 1 } else { ib };
        *x = ((v >> pos) & ((1 << n) - 1)) as u8;
        pos += n;
    }
    debug_assert_eq!(pos, 128);
    Some(b)
}

/// Packs a block. Anchor indices must have their top bit clear, and fields must fit their widths.
pub fn pack(b: &Bc6hBlock) -> [u8; 16] {
    let mi = usize::from(b.mode);
    let m = MODES[mi];
    let mut v = u128::from(m.value);
    let mut pos = u32::from(m.mode_bits);
    let (runs, n) = &RUNS[mi];
    for r in &runs[..*n] {
        let f = usize::from(r.field);
        v |= u128::from((b.fields[f / 3][f % 3] >> r.bit) & ((1 << r.len) - 1)) << pos;
        pos += u32::from(r.len);
    }
    let ib = if m.two_regions {
        v |= u128::from(b.partition & 31) << pos;
        pos += 5;
        3
    } else {
        4
    };
    let anchor = region_anchors(m, b.partition);
    for (i, &x) in b.idx.iter().enumerate() {
        let n = if anchor.contains(&i) { ib - 1 } else { ib };
        debug_assert!(u32::from(x) < 1 << n, "index {x} of texel {i} needs more than {n} bits");
        v |= u128::from(x) << pos;
        pos += n;
    }
    debug_assert_eq!(pos, 128);
    v.to_le_bytes()
}

fn sign_extend(v: i32, bits: u8) -> i32 {
    let s = 32 - u32::from(bits);
    (v << s) >> s
}

/// An endpoint component back on the 16-bit scale the format interpolates on.
fn unquantize(v: i32, eb: u8, signed: bool) -> i32 {
    if signed {
        if eb >= 16 {
            return v;
        }
        let a = v.abs();
        let u = if a == 0 {
            0
        } else if a >= (1 << (eb - 1)) - 1 {
            0x7fff
        } else {
            ((a << 15) + 0x4000) >> (eb - 1)
        };
        if v < 0 { -u } else { u }
    } else if eb >= 15 {
        v
    } else if v == 0 {
        0
    } else if v == (1 << eb) - 1 {
        0xffff
    } else {
        ((v << 16) + 0x8000) >> eb
    }
}

/// An interpolated value as half-float bits.
fn finish(v: i32, signed: bool) -> u16 {
    if !signed {
        ((v * 31) >> 6) as u16
    } else if v < 0 {
        ((((-v) * 31) >> 5) as u16) | 0x8000
    } else {
        ((v * 31) >> 5) as u16
    }
}

fn interp(e0: i32, e1: i32, w: i32) -> i32 {
    (e0 * (64 - w) + e1 * w + 32) >> 6
}

/// Endpoint values on the 16-bit scale, `[region][end][channel]`.
fn endpoints(b: &Bc6hBlock, signed: bool) -> [[[i32; 3]; 2]; 2] {
    let m = MODES[usize::from(b.mode)];
    let mut e = [[0i32; 3]; 4];
    for c in 0..3 {
        let mut w = i32::from(b.fields[0][c]);
        if signed {
            w = sign_extend(w, m.eb);
        }
        e[0][c] = w;
        for k in 1..2 * m.regions() {
            let mut v = i32::from(b.fields[k][c]);
            if signed || m.transformed {
                v = sign_extend(v, m.db[c]);
            }
            if m.transformed {
                v = (v + w) & ((1 << m.eb) - 1);
                if signed {
                    v = sign_extend(v, m.eb);
                }
            }
            e[k][c] = v;
        }
    }
    let mut out = [[[0i32; 3]; 2]; 2];
    for (k, ek) in e.iter().enumerate() {
        for c in 0..3 {
            out[k / 2][k % 2][c] = unquantize(ek[c], m.eb, signed);
        }
    }
    out
}

/// Decodes an unpacked block to half-float bits.
pub fn decode_block(b: &Bc6hBlock, signed: bool) -> HdrBlock {
    let m = MODES[usize::from(b.mode)];
    let e = endpoints(b, signed);
    let wt: &[i32] = if m.two_regions { &W3 } else { &W4 };
    let mut out = [[0u16; 3]; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let r = region_of(m, b.partition, i);
        let w = wt[usize::from(b.idx[i])];
        for c in 0..3 {
            o[c] = finish(interp(e[r][0][c], e[r][1][c], w), signed);
        }
    }
    out
}

/// Decodes a 16-byte block to half-float bits. Reserved modes decode to zero, as the D3D
/// specification requires.
pub fn decode(raw: &[u8], signed: bool) -> HdrBlock {
    unpack(raw).map_or([[0; 3]; 16], |b| decode_block(&b, signed))
}

/// Decodes a 16-byte unsigned block to 8-bit RGB (values clamped to [0, 1], alpha 255): what an
/// LDR viewer or [`crate::format::Format::decode_block`] sees.
pub fn decode_ldr(raw: &[u8], out: &mut crate::surface::Block) {
    for (o, h) in out.iter_mut().zip(decode(raw, false)) {
        for c in 0..3 {
            o[c] = (half_to_f32(h[c]).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        o[3] = 255;
    }
}

/// The half-float targets for an 8-bit block read as values in [0, 1].
pub fn ldr_block(px: &crate::surface::Block) -> HdrBlock {
    px.map(|p| [0, 1, 2].map(|c| f32_to_uf16(f32::from(p[c]) / 255.0)))
}

/// Sum of squared differences between two blocks' half-float bits.
pub fn sse(a: &HdrBlock, b: &HdrBlock) -> u64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            (0..3)
                .map(|c| {
                    let d = i64::from(x[c]) - i64::from(y[c]);
                    (d * d) as u64
                })
                .sum::<u64>()
        })
        .sum()
}

/// Search settings for an effort level 0..=9.
#[derive(Clone, Copy, Debug)]
struct Search {
    /// Partitions (best estimated fit first) tried with the two-region modes.
    partitions: usize,
    /// One-region modes tried per block: the most precise ones whose deltas fit.
    modes1: usize,
    /// Two-region modes tried per partition, likewise.
    modes2: usize,
    /// Least-squares refits after the first indexing.
    refits: usize,
    /// Two-region modes are skipped when the one-region error is at most this multiple of the best
    /// partition's estimated error (0: never skipped).
    skip: f32,
}

fn search(effort: u8) -> Search {
    let e = usize::from(effort.min(crate::format::MAX_EFFORT));
    Search {
        partitions: [1, 1, 2, 2, 3, 4, 6, 8, 12, 32][e],
        modes1: [1, 1, 1, 2, 2, 2, 2, 3, 4, 4][e],
        modes2: [1, 1, 1, 1, 2, 2, 3, 3, 4, 10][e],
        refits: [1, 1, 1, 2, 2, 2, 2, 3, 3, 4][e],
        skip: [1.0, 0.7, 0.5, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0][e],
    }
}

/// One-region modes ([`MODES`] indices), most precise endpoints first.
const ONE_ORDER: [usize; 4] = [13, 12, 11, 10];
/// Two-region modes, most precise endpoints first.
const TWO_ORDER: [usize; 10] = [2, 3, 4, 0, 5, 6, 7, 8, 1, 9];

/// For an interpolation weight 0..=64, the index of the nearest weight in a table.
const fn nearest_weight<const N: usize>(w: [i32; N]) -> [u8; 65] {
    let mut out = [0u8; 65];
    let mut x = 0;
    while x <= 64 {
        let mut best = 0;
        let mut j = 1;
        while j < N {
            if (w[j] - x).abs() < (w[best] - x).abs() {
                best = j;
            }
            j += 1;
        }
        out[x as usize] = best as u8;
        x += 1;
    }
    out
}

const NEAREST3: [u8; 65] = nearest_weight(W3);
const NEAREST4: [u8; 65] = nearest_weight(W4);

/// An encoder's view of one block: targets on the 16-bit interpolation scale (`t`) and as
/// half bits (`h`).
struct Target {
    h: [[i32; 3]; 16],
    hf: [[f32; 3]; 16],
    t: [V4; 16],
}

impl Target {
    fn new(px: &HdrBlock) -> Self {
        let mut h = [[0i32; 3]; 16];
        let mut hf = [[0f32; 3]; 16];
        let mut t = [[0f32; 4]; 16];
        for i in 0..16 {
            for c in 0..3 {
                let v = px[i][c];
                let v = if v & 0x8000 != 0 { 0 } else { v.min(HALF_MAX) };
                h[i][c] = i32::from(v);
                hf[i][c] = f32::from(v);
                t[i][c] = f32::from(v) * (64.0 / 31.0);
            }
        }
        Self { h, hf, t }
    }
}

/// The `eb`-bit value whose unquantized value is nearest `x`.
fn quantize(x: f32, eb: u8) -> i32 {
    if eb >= 15 {
        return ((x.max(0.0) + 0.5) as i32).min(0xffff);
    }
    let max = (1 << eb) - 1;
    let q0 = ((x * (1 << eb) as f32 / 65536.0) as i32).clamp(0, max);
    // Between the ends, values unquantize to (q + 1/2) * 2^(16 - eb): evenly spaced, so the floor
    // is nearest. Only 0 (to 0) and `max` (to 65535) break the pattern.
    if q0 > 1 && q0 < max - 1 {
        return q0;
    }
    let mut best = (f32::MAX, q0);
    for q in (q0 - 1).max(0)..=(q0 + 1).min(max) {
        let d = (unquantize(q, eb, false) as f32 - x).abs();
        if d < best.0 {
            best = (d, q);
        }
    }
    best.1
}

/// Float endpoints, `[region][end]` on the 16-bit scale (channel 3 unused).
type Ends = [[V4; 2]; 2];

/// One encoding attempt: the block and its error.
#[derive(Clone, Copy)]
struct Trial {
    err: u64,
    block: Bc6hBlock,
}

/// Quantizes `ends` for mode `mi`, indexes every texel to its nearest palette entry and fixes the
/// anchors. Returns the trial and the indices' interpolation weights for a refit.
fn quantize_and_index(tg: &Target, mi: usize, partition: u8, ends: &Ends) -> Trial {
    let m = MODES[mi];
    let nr = m.regions();
    let wt: &[i32] = if m.two_regions { &W3 } else { &W4 };
    let half = wt.len() / 2;
    let anchor = region_anchors(m, partition);
    let mut q = [[[0i32; 3]; 2]; 2];
    for r in 0..nr {
        for k in 0..2 {
            for c in 0..3 {
                q[r][k][c] = quantize(ends[r][k][c], m.eb);
            }
        }
    }
    let lim = |c: usize| (-(1 << (m.db[c] - 1)), (1 << (m.db[c] - 1)) - 1);
    if m.transformed {
        for (r, k) in [(0, 1), (1, 0), (1, 1)].into_iter().take(2 * nr - 1) {
            for c in 0..3 {
                let (lo, hi) = lim(c);
                q[r][k][c] = q[0][0][c] + (q[r][k][c] - q[0][0][c]).clamp(lo, hi);
            }
        }
    }
    let (idx, mut err) = index(tg, m, partition, &q);
    let mut idx = idx;
    for r in 0..nr {
        let a = anchor[r];
        if usize::from(idx[a]) < half {
            continue;
        }
        // Swapping the region's ends and mirroring its indices decodes the same. In a transformed
        // mode, swapping region 0 moves the base, so the deltas must still fit.
        let mut sq = q;
        sq[r].swap(0, 1);
        let fits = !m.transformed
            || r == 1
            || [(0, 1), (1, 0), (1, 1)].into_iter().take(2 * nr - 1).all(|(rr, k)| {
                (0..3).all(|c| {
                    let (lo, hi) = lim(c);
                    (lo..=hi).contains(&(sq[rr][k][c] - sq[0][0][c]))
                })
            });
        if fits {
            q = sq;
            for (i, x) in idx.iter_mut().enumerate() {
                if region_of(m, partition, i) == r {
                    *x = (wt.len() - 1 - usize::from(*x)) as u8;
                }
            }
        } else {
            // Keep the ends; the anchor takes its best index with the top bit clear.
            let pal = palette(m, &q[r]);
            let old = texel_err(&tg.h[a], &pal[usize::from(idx[a])]);
            let (j, e) = (0..half)
                .map(|j| (j, texel_err(&tg.h[a], &pal[j])))
                .min_by_key(|&(_, e)| e)
                .expect("palette has entries");
            idx[a] = j as u8;
            err = err - old + e;
        }
    }
    let mut block = Bc6hBlock {
        mode: mi as u8,
        partition: if m.two_regions { partition } else { 0 },
        idx,
        ..Default::default()
    };
    for (k, f) in block.fields.iter_mut().enumerate().take(2 * nr) {
        let v = q[k / 2][k % 2];
        for c in 0..3 {
            *f.get_mut(c).expect("3 channels") = if k > 0 && m.transformed {
                ((v[c] - q[0][0][c]) & ((1 << m.db[c]) - 1)) as u16
            } else {
                v[c] as u16
            };
        }
    }
    Trial { err, block }
}

/// A region's palette as half bits.
fn palette(m: Mode, q: &[[i32; 3]; 2]) -> [[i32; 3]; 16] {
    let wt: &[i32] = if m.two_regions { &W3 } else { &W4 };
    let u = q.map(|e| e.map(|v| unquantize(v, m.eb, false)));
    let mut pal = [[0i32; 3]; 16];
    for (p, &w) in pal.iter_mut().zip(wt) {
        for c in 0..3 {
            p[c] = i32::from(finish(interp(u[0][c], u[1][c], w), false));
        }
    }
    pal
}

fn texel_err(h: &[i32; 3], p: &[i32; 3]) -> u64 {
    (0..3)
        .map(|c| {
            let d = i64::from(h[c] - p[c]);
            (d * d) as u64
        })
        .sum()
}

/// Nearest palette entry per texel, and the total error. The palette is a line in half-bit space
/// (up to rounding), so each texel is projected on it and the nearest weight and its neighbors
/// are checked exactly.
fn index(tg: &Target, m: Mode, partition: u8, q: &[[[i32; 3]; 2]; 2]) -> ([u8; 16], u64) {
    let (n, near, wt): (usize, _, &[i32]) = if m.two_regions {
        (8, &NEAREST3, &W3)
    } else {
        (16, &NEAREST4, &W4)
    };
    let mut pals = [[[0i32; 3]; 16]; 2];
    let mut line = [([0f32; 3], [0f32; 3], 0f32); 2];
    for r in 0..m.regions() {
        pals[r] = palette(m, &q[r]);
        let p0 = pals[r][0].map(|v| v as f32);
        let d: [f32; 3] = std::array::from_fn(|c| pals[r][n - 1][c] as f32 - p0[c]);
        let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        line[r] = (p0, d, if dd > 0.0 { 64.0 / dd } else { 0.0 });
    }
    let mut idx = [0u8; 16];
    let mut total = 0;
    for i in 0..16 {
        let r = region_of(m, partition, i);
        let (p0, d, s) = line[r];
        let x = &tg.hf[i];
        let t = ((x[0] - p0[0]) * d[0] + (x[1] - p0[1]) * d[1] + (x[2] - p0[2]) * d[2]) * s;
        let j0 = usize::from(near[((t.max(0.0) + 0.5) as usize).min(64)]);
        let pal = &pals[r];
        let mut best = (texel_err(&tg.h[i], &pal[j0]), j0);
        // Palette rounding moves entries off the line by under a step, so only the neighbor on
        // the projection's side can be nearer.
        let j1 = if t > wt[j0] as f32 { j0 + 1 } else { j0.wrapping_sub(1) };
        if j1 < n {
            best = best.min((texel_err(&tg.h[i], &pal[j1]), j1));
        }
        idx[i] = best.1 as u8;
        total += best.0;
    }
    (idx, total)
}

/// Least-squares ends for the block's indices, or `None` where a region's indices are all equal.
fn refit(tg: &Target, b: &Bc6hBlock) -> Ends {
    let m = MODES[usize::from(b.mode)];
    let wt: &[i32] = if m.two_regions { &W3 } else { &W4 };
    let mut ends = [[[0f32; 4]; 2]; 2];
    for (r, er) in ends.iter_mut().enumerate().take(m.regions()) {
        let (mut aa, mut ab, mut bb) = (0f32, 0f32, 0f32);
        let (mut ax, mut bx, mut sum) = ([0f32; 3], [0f32; 3], [0f32; 3]);
        let mut n = 0f32;
        for i in 0..16 {
            if region_of(m, b.partition, i) != r {
                continue;
            }
            let t = wt[usize::from(b.idx[i])] as f32 / 64.0;
            let s = 1.0 - t;
            aa += s * s;
            ab += s * t;
            bb += t * t;
            for c in 0..3 {
                ax[c] += s * tg.t[i][c];
                bx[c] += t * tg.t[i][c];
                sum[c] += tg.t[i][c];
            }
            n += 1.0;
        }
        let det = aa * bb - ab * ab;
        for c in 0..3 {
            let (e0, e1) = if det.abs() > 1e-6 {
                ((ax[c] * bb - bx[c] * ab) / det, (bx[c] * aa - ax[c] * ab) / det)
            } else {
                let mean = sum[c] / n.max(1.0);
                (mean, mean)
            };
            er[0][c] = e0.clamp(0.0, 65535.0);
            er[1][c] = e1.clamp(0.0, 65535.0);
        }
    }
    ends
}

/// The ends of a point set's projection on its principal axis, on the 16-bit scale.
fn line_ends(pts: &[V4]) -> [V4; 2] {
    let (mean, cov) = mean_cov3(pts);
    line_ends_from(pts, &mean, &axis3(&cov))
}

/// Mean (RGB) and covariance (rr, rg, rb, gg, gb, bb) of a point set.
fn mean_cov3(pts: &[V4]) -> (V4, [f32; 6]) {
    let n = pts.len().max(1) as f32;
    let mut m = [0f32; 4];
    for p in pts {
        m[0] += p[0];
        m[1] += p[1];
        m[2] += p[2];
    }
    m = m.map(|v| v / n);
    let mut c = [0f32; 6];
    for p in pts {
        let (r, g, b) = (p[0] - m[0], p[1] - m[1], p[2] - m[2]);
        c[0] += r * r;
        c[1] += r * g;
        c[2] += r * b;
        c[3] += g * g;
        c[4] += g * b;
        c[5] += b * b;
    }
    (m, c.map(|v| v / n))
}

/// Principal axis of a covariance from [`mean_cov3`] by power iteration, as a unit vector (the
/// gray diagonal for a flat set).
fn axis3(c: &[f32; 6]) -> V4 {
    let rows = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
    // Start from the row with the largest diagonal entry: never orthogonal to the dominant
    // eigenvector unless that entry is zero.
    let mut v = if c[0] >= c[3] && c[0] >= c[5] {
        rows[0]
    } else if c[3] >= c[5] {
        rows[1]
    } else {
        rows[2]
    };
    for _ in 0..6 {
        let w = [
            rows[0][0] * v[0] + rows[0][1] * v[1] + rows[0][2] * v[2],
            rows[1][0] * v[0] + rows[1][1] * v[1] + rows[1][2] * v[2],
            rows[2][0] * v[0] + rows[2][1] * v[1] + rows[2][2] * v[2],
        ];
        // Rescale (by any positive amount) so the iteration can't overflow.
        let mx = w[0].abs().max(w[1].abs()).max(w[2].abs());
        if mx < 1e-20 {
            break;
        }
        v = w.map(|x| x / mx);
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        let g = 1.0 / 3f32.sqrt();
        return [g, g, g, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len, 0.0]
}

/// The ends of `pts`' projection on the line through `mean` along `axis`.
fn line_ends_from(pts: &[V4], mean: &V4, axis: &V4) -> [V4; 2] {
    let (mut lo, mut hi) = (0f32, 0f32);
    for p in pts {
        let t: f32 = (0..3).map(|c| (p[c] - mean[c]) * axis[c]).sum();
        lo = lo.min(t);
        hi = hi.max(t);
    }
    let mut e = [[0f32; 4]; 2];
    for c in 0..3 {
        e[0][c] = (mean[c] + axis[c] * lo).clamp(0.0, 65535.0);
        e[1][c] = (mean[c] + axis[c] * hi).clamp(0.0, 65535.0);
    }
    e
}

/// Per-nibble moment tables of a block's texels (centered on the block mean): `tab[k][m]` sums
/// r, g, b, rr, rg, rb, gg, gb, bb over texels `4k + j` with bit `j` of `m` set; and the totals.
fn moment_tables(c: &[[f32; 3]; 16]) -> ([[[f32; 9]; 16]; 4], [f32; 9]) {
    let mut tab = [[[0f32; 9]; 16]; 4];
    let mut tot = [0f32; 9];
    for (k, t) in tab.iter_mut().enumerate() {
        for m in 1..16usize {
            let p = c[4 * k + m.trailing_zeros() as usize];
            let mo = [
                p[0],
                p[1],
                p[2],
                p[0] * p[0],
                p[0] * p[1],
                p[0] * p[2],
                p[1] * p[1],
                p[1] * p[2],
                p[2] * p[2],
            ];
            let prev = t[m & (m - 1)];
            t[m] = std::array::from_fn(|v| prev[v] + mo[v]);
        }
        for v in 0..9 {
            tot[v] += t[15][v];
        }
    }
    (tab, tot)
}

/// Adds to `out[p]` the estimated squared error of fitting partition `p`'s region (`n[p]` texels,
/// moments `s[v][p]`) by a line with 3-bit indices: the scatter off the line, through the Rayleigh
/// quotient after one power step from `axis`, plus the index rounding along it. Laid out across the
/// 32 partitions so it vectorizes.
fn region_estimates(s: &[[f32; 32]; 9], n: &[f32; 32], axis: &[f32; 3], out: &mut [f32; 32]) {
    for p in 0..32 {
        let n = n[p];
        let (s0, s1, s2) = (s[0][p], s[1][p], s[2][p]);
        // n times the scatter matrix.
        let a0 = n * s[3][p] - s0 * s0;
        let a1 = n * s[4][p] - s0 * s1;
        let a2 = n * s[5][p] - s0 * s2;
        let a3 = n * s[6][p] - s1 * s1;
        let a4 = n * s[7][p] - s1 * s2;
        let a5 = n * s[8][p] - s2 * s2;
        let tr = a0 + a3 + a5;
        let v0 = a0 * axis[0] + a1 * axis[1] + a2 * axis[2];
        let v1 = a1 * axis[0] + a3 * axis[1] + a4 * axis[2];
        let v2 = a2 * axis[0] + a4 * axis[1] + a5 * axis[2];
        let w0 = a0 * v0 + a1 * v1 + a2 * v2;
        let w1 = a1 * v0 + a3 * v1 + a4 * v2;
        let w2 = a2 * v0 + a4 * v1 + a5 * v2;
        let vv = v0 * v0 + v1 * v1 + v2 * v2;
        let vw = v0 * w0 + v1 * w1 + v2 * w2;
        let lambda = if vv > 0.0 { vw / vv } else { 0.0 };
        let e = (tr - lambda * (48.0 / 49.0)).max(0.0) / n.max(1.0);
        out[p] += if n >= 2.0 { e } else { 0.0 };
    }
}

/// The two-region partitions by estimated error, the `k` best first and sorted, with the
/// estimates (on the 16-bit scale).
fn rank_partitions(tg: &Target, mean: &V4, axis: &V4, k: usize) -> ([u8; 32], [f32; 32]) {
    let c: [[f32; 3]; 16] = std::array::from_fn(|i| std::array::from_fn(|ch| tg.t[i][ch] - mean[ch]));
    let (tab, tot) = moment_tables(&c);
    let ax = [axis[0], axis[1], axis[2]];
    let mut s1 = [[0f32; 32]; 9];
    let mut n1 = [0f32; 32];
    for p in 0..32 {
        let mask = crate::bc7::tables::PART2[p];
        for (kk, t) in tab.iter().enumerate() {
            let m = &t[((mask >> (4 * kk)) & 15) as usize];
            for v in 0..9 {
                s1[v][p] += m[v];
            }
        }
        n1[p] = (mask & 0xffff).count_ones() as f32;
    }
    let s0: [[f32; 32]; 9] = std::array::from_fn(|v| std::array::from_fn(|p| tot[v] - s1[v][p]));
    let n0: [f32; 32] = std::array::from_fn(|p| 16.0 - n1[p]);
    let mut e = [0f32; 32];
    region_estimates(&s0, &n0, &ax, &mut e);
    region_estimates(&s1, &n1, &ax, &mut e);
    let mut est: [(f32, u8); 32] = std::array::from_fn(|p| (e[p], p as u8));
    let k = k.clamp(1, 32);
    if k == 1 {
        let b = (1..32).fold(0, |b, p| if est[p].0 < est[b].0 { p } else { b });
        est.swap(0, b);
    } else {
        if k < 32 {
            est.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        }
        est[..k].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }
    (est.map(|e| e.1), est.map(|e| e.0))
}

/// Estimated squared error (16-bit scale) of storing `ends` in mode `mi`: each end quantized to the
/// mode's precision (`q`), deltas clamped to their bits, and the displacement `d0`, `d1` of a region's
/// ends spread over its `cnt` texels as interpolation does, `cnt * (d0^2 + d0.d1 + d1^2) / 3`.
fn mode_estimate(mi: usize, ends: &Ends, q: [[[i32; 3]; 2]; 2], cnt: [f32; 2]) -> f32 {
    let m = MODES[mi];
    let nr = m.regions();
    let mut q = q;
    if m.transformed {
        for (r, k) in [(0, 1), (1, 0), (1, 1)].into_iter().take(2 * nr - 1) {
            for c in 0..3 {
                let h = 1 << (m.db[c] - 1);
                q[r][k][c] = q[0][0][c] + (q[r][k][c] - q[0][0][c]).clamp(-h, h - 1);
            }
        }
    }
    let mut e = 0.0;
    for r in 0..nr {
        let d = |k: usize, c: usize| unquantize(q[r][k][c], m.eb, false) as f32 - ends[r][k][c];
        let mut s = 0.0;
        for c in 0..3 {
            let (a, b) = (d(0, c), d(1, c));
            s += a * a + a * b + b * b;
        }
        e += cnt[r] * s / 3.0;
    }
    e
}

/// Up to `n` modes from `order` for `ends`, least estimated error first ([`mode_estimate`]; ties
/// to the earlier, more precise, mode).
fn pick_modes<const N: usize>(order: &[usize; N], ends: &Ends, cnt: [f32; 2], n: usize) -> ([usize; N], usize) {
    let nr = if cnt[1] > 0.0 { 2 } else { 1 };
    // Quantized ends per endpoint precision, shared by the modes with that precision.
    let mut cache = [[[[0i32; 3]; 2]; 2]; 17];
    let mut have = 0u32;
    let mut est = [(0f32, 0usize); N];
    for (i, &mi) in order.iter().enumerate() {
        let eb = MODES[mi].eb;
        if have & (1 << eb) == 0 {
            have |= 1 << eb;
            for r in 0..nr {
                for k in 0..2 {
                    for c in 0..3 {
                        cache[usize::from(eb)][r][k][c] = quantize(ends[r][k][c], eb);
                    }
                }
            }
        }
        est[i] = (mode_estimate(mi, ends, cache[usize::from(eb)], cnt), i);
    }
    let k = n.clamp(1, N);
    if k == 1 {
        let b = (1..N).fold(0, |b, i| if est[i].0 < est[b].0 { i } else { b });
        est.swap(0, b);
    } else {
        est.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    (est.map(|e| order[e.1]), k)
}

/// Tries mode `mi` from `ends`, refitting `refits` times; keeps the best into `best`.
fn try_mode(tg: &Target, mi: usize, partition: u8, ends: &Ends, refits: usize, best: &mut Trial) {
    let mut t = quantize_and_index(tg, mi, partition, ends);
    for _ in 0..refits {
        if t.err < best.err {
            *best = t;
        }
        if t.err == 0 {
            return;
        }
        let e = refit(tg, &t.block);
        let n = quantize_and_index(tg, mi, partition, &e);
        if n.err >= t.err {
            break;
        }
        t = n;
    }
    if t.err < best.err {
        *best = t;
    }
}

/// Encodes one block of half floats (negative values and NaN as 0) at effort 0..=9.
pub fn encode_block(px: &HdrBlock, effort: u8) -> [u8; 16] {
    pack(&encode_unpacked(px, effort).block)
}

fn encode_unpacked(px: &HdrBlock, effort: u8) -> Trial {
    let s = search(effort);
    let tg = Target::new(px);
    let mut best = Trial {
        err: u64::MAX,
        block: Bc6hBlock::default(),
    };
    // One region: modes 11..14, from the line through all 16 texels.
    let (mean, cov) = mean_cov3(&tg.t);
    let axis = axis3(&cov);
    let one = line_ends_from(&tg.t, &mean, &axis);
    let ends = [one, one];
    let (modes, k) = pick_modes(&ONE_ORDER, &ends, [16.0, 0.0], s.modes1);
    for &mi in &modes[..k] {
        try_mode(&tg, mi, 0, &ends, s.refits, &mut best);
    }
    if best.err == 0 {
        return best;
    }
    // Two regions: the partitions estimated to fit best, each with the most precise modes that
    // hold its ends.
    let (parts, est) = rank_partitions(&tg, &mean, &axis, s.partitions);
    // The estimate is on the 16-bit scale; errors are in half bits.
    let scale = (31.0f32 / 64.0) * (31.0 / 64.0);
    if s.skip > 0.0 && best.err as f32 <= s.skip * est[0] * scale {
        return best;
    }
    for &p in &parts[..s.partitions.min(32)] {
        let mut pts = [[[0f32; 4]; 16]; 2];
        let mut n = [0usize; 2];
        for i in 0..16 {
            let r = subset_of(2, p, i);
            pts[r][n[r]] = tg.t[i];
            n[r] += 1;
        }
        let ends = [line_ends(&pts[0][..n[0]]), line_ends(&pts[1][..n[1]])];
        let (modes, k) = pick_modes(&TWO_ORDER, &ends, [n[0] as f32, n[1] as f32], s.modes2);
        for &mi in &modes[..k] {
            try_mode(&tg, mi, p, &ends, s.refits, &mut best);
        }
    }
    best
}

/// Encodes an 8-bit block read as values in [0, 1] (what [`crate::format::Format::encode_block`]
/// does for BC6H; HDR sources go through [`encode_block`]).
pub fn encode_ldr(px: &crate::surface::Block, effort: u8) -> [u8; 16] {
    encode_block(&ldr_block(px), effort)
}

/// Encodes every block of an HDR surface, in parallel, raster order.
pub fn encode_surface(surface: &crate::hdr::HdrSurface, effort: u8) -> Vec<u8> {
    use rayon::prelude::*;
    let bx = surface.blocks_x();
    (0..surface.block_count() as u32)
        .into_par_iter()
        .flat_map_iter(|i| encode_block(&surface.read_block(i % bx, i / bx), effort))
        .collect()
}

/// Decodes unsigned BC6H blocks (raster order) to an HDR surface.
pub fn decode_surface(data: &[u8], width: u32, height: u32) -> crate::hdr::HdrSurface {
    crate::hdr::HdrSurface::from_blocks(width, height, data.as_chunks::<16>().0.iter().map(|b| decode(b, false)))
}

/// Encodes every block of an 8-bit surface (as [`encode_ldr`]), in parallel, raster order.
pub fn encode_surface_ldr(surface: &crate::surface::Surface, effort: u8) -> Vec<u8> {
    use rayon::prelude::*;
    let bx = surface.blocks_x();
    (0..surface.block_count() as u32)
        .into_par_iter()
        .flat_map_iter(|i| encode_ldr(&surface.read_block(i % bx, i / bx), effort))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn half_conversion_round_trips_and_rounds() {
        for h in 0..=0xffffu16 {
            let f = half_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_half(f) & 0x7c00 == 0x7c00 && f32_to_half(f) & 0x3ff != 0);
            } else {
                assert_eq!(f32_to_half(f), h, "{h:#06x} {f}");
            }
        }
        assert_eq!(f32_to_half(1.0), 0x3c00);
        assert_eq!(f32_to_half(65504.0), 0x7bff);
        assert_eq!(f32_to_half(65519.0), 0x7bff);
        assert_eq!(f32_to_half(65520.0), 0x7c00);
        // Halfway between 1 and the next half rounds to even.
        assert_eq!(f32_to_half(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_half(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_half(2f32.powi(-25)), 0);
        assert_eq!(f32_to_half(2f32.powi(-25) * 1.01), 1);
        assert_eq!(f32_to_uf16(-3.0), 0);
        assert_eq!(f32_to_uf16(1e9), HALF_MAX);
    }

    #[test]
    fn layouts_cover_every_bit_once() {
        for (mi, m) in MODES.iter().enumerate() {
            let lay = LAYOUT[mi];
            let mut seen = std::collections::HashSet::new();
            for &e in lay {
                assert!(seen.insert(e), "mode {mi}: field bit {e} twice");
                let (f, bit) = (usize::from(e >> 4), e & 15);
                let width = if f < 3 { m.eb } else { m.db[f % 3] };
                assert!(bit < width, "mode {mi}: bit {bit} of field {f} past width {width}");
                assert!(f < 6 * m.regions());
            }
            let fields: usize =
                3 * usize::from(m.eb) + (2 * m.regions() - 1) * m.db.iter().map(|&d| d as usize).sum::<usize>();
            assert_eq!(lay.len(), fields, "mode {mi}");
            let idx_bits = if m.two_regions { 5 + 46 } else { 63 };
            assert_eq!(usize::from(m.mode_bits) + lay.len() + idx_bits, 128, "mode {mi}");
        }
    }

    #[test]
    fn pack_unpack_round_trips_random_blocks() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut n = 0;
        while n < 20_000 {
            let raw = (u128::from(rng(&mut seed)) << 64 | u128::from(rng(&mut seed))).to_le_bytes();
            let Some(b) = unpack(&raw) else { continue };
            assert_eq!(pack(&b), raw);
            n += 1;
        }
        let reserved = [19u8, 23, 27, 31].map(|v| {
            let mut raw = [0u8; 16];
            raw[0] = v;
            unpack(&raw)
        });
        assert!(reserved.iter().all(Option::is_none));
    }

    fn block_from(f: impl Fn(usize, usize) -> f32) -> HdrBlock {
        std::array::from_fn(|i| [0, 1, 2].map(|c| f32_to_uf16(f(i, c))))
    }

    /// Mean squared error per channel sample, in half-float steps.
    fn mse(px: &HdrBlock, effort: u8) -> f64 {
        let raw = encode_block(px, effort);
        sse(px, &decode(&raw, false)) as f64 / 48.0
    }

    #[test]
    fn flat_and_smooth_blocks_encode_nearly_exactly() {
        let flat = block_from(|_, c| [0.25, 3.0, 1200.0][c]);
        assert!(mse(&flat, 5) < 1.0, "{}", mse(&flat, 5));
        // Within one octave a ramp is a line in half bits: 16 evenly spaced values, which 4-bit
        // indices hold to a few bits (their weights are 64ths, not 15ths).
        let ramp = block_from(|i, c| (1.0 + (i % 4) as f32 * 0.25 + (i / 4) as f32 * 0.0625) * [0.5, 2.0, 8.0][c]);
        for e in [0, 5] {
            assert!(mse(&ramp, e) < 30.0, "{}", mse(&ramp, e));
        }
        // A wide range: four octaves across the block, a curve in half bits (about 1600).
        let wide = block_from(|i, c| 2f32.powf(i as f32 / 4.0) * [1.0, 0.8, 0.5][c]);
        assert!(mse(&wide, 5) < 2000.0, "{}", mse(&wide, 5));
    }

    #[test]
    fn effort_does_not_hurt_and_random_blocks_decode() {
        let mut seed = 77u64;
        let (mut lo, mut hi) = (0.0, 0.0);
        for _ in 0..200 {
            let base = [0, 1, 2].map(|_| (rng(&mut seed) % 2000) as f32 / 100.0);
            let px = block_from(|i, c| {
                let edge = if (i % 4) + (i / 4) > 3 { 4.0 } else { 1.0 };
                base[c] * edge * (1.0 + (rng(&mut seed.wrapping_add(i as u64)) % 100) as f32 / 400.0)
            });
            lo += mse(&px, 1);
            hi += mse(&px, 9);
        }
        assert!(hi <= lo, "effort 9 {hi} vs effort 1 {lo}");
    }

    #[test]
    fn signed_decode_handles_negative_endpoints() {
        // Mode 11 (index 10, 10-bit raw endpoints): w = -1 (all ones in 10 bits), x = +1.
        let b = Bc6hBlock {
            mode: 10,
            fields: [[0x3ff; 3], [1; 3], [0; 3], [0; 3]],
            idx: [0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            ..Default::default()
        };
        let d = decode(&pack(&b), true);
        // -1 at 10 bits unquantizes to -((1 << 15) + 0x4000) >> 9 = -96, finished to -(96*31>>5) = -93.
        assert_eq!(d[0], [0x8000 | 93; 3]);
        assert_eq!(d[1], [93; 3]);
    }
}
