//! Lossless field-split transform of BC blocks ("prep"): same-kind fields of every block go to their
//! own byte stream, so a downstream LZ compressor sees long runs of similar data instead of endpoints
//! and indices alternating every block (the idea of Oodle's BC7Prep). [`inverse`] restores the exact
//! blocks; it is the step a loader (or a GPU compute pass) runs after decompression.
//!
//! Layout: a header of little-endian `u32` stream lengths, then the streams in order.
//!
//! * BC1: color0 bytes, color1 bytes (each as low-byte plane then high-byte plane), selector bytes.
//! * BC4 / BC5: per channel, endpoint 0 bytes, endpoint 1 bytes, then the 6 index bytes.
//! * BC3: the BC4 streams of the alpha halves, then the BC1 streams of the color halves.
//! * ETC2 RGB, EAC R11 / RG11, ETC2 RGBA8: the BC1, BC4 / BC5 and BC3 splits, which fit their
//!   layouts (four header bytes then four selector bytes; base and multiplier/table bytes then
//!   six index bytes).
//! * BC6H, ASTC: no field split yet; one stream of whole blocks.
//! * BC7: mode byte per block (8 = invalid block, stored raw in the escape stream); shape byte
//!   (partition, or rotation + index selection); endpoint channels one per byte (with the p-bit
//!   folded in as the low bit for modes with a p-bit per endpoint); shared p-bits (mode 1); index
//!   bits, bit-packed per block in the block's own order and padded to whole bytes; escape stream.
//!   Fields are byte aligned so equal values line up across blocks; the padding bits are constant
//!   and cost the entropy coder next to nothing.

use crate::bc7;
use crate::format::Format;

/// Splits `raw` blocks of `format` into field streams.
pub fn forward(format: Format, raw: &[u8]) -> Vec<u8> {
    let streams = match format {
        Format::Bc1 | Format::Etc2Rgb => split_bc1(raw),
        Format::Bc3 | Format::Etc2Rgba => split_bc3(raw),
        Format::Bc4 | Format::EacR11 => split_bc4(raw, 1),
        Format::Bc5 | Format::EacRg11 => split_bc4(raw, 2),
        // No field split yet: one stream of whole blocks.
        Format::Bc6h | Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => vec![raw.to_vec()],
        Format::Bc7 => split_bc7(raw),
    };
    let mut out = Vec::with_capacity(raw.len() + 4 * streams.len());
    for s in &streams {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for s in &streams {
        out.extend_from_slice(s);
    }
    out
}

/// Upper bound on `forward(format, raw).len()` for `raw_len` bytes of blocks (no block expands
/// to more than twice its size: at most 29 stream bytes per 16-byte BC7 block).
pub fn max_len(raw_len: usize) -> usize {
    raw_len.saturating_mul(2).saturating_add(64)
}

/// Restores the blocks of `forward(format, raw)`; `None` if `data` is malformed.
pub fn inverse(format: Format, data: &[u8]) -> Option<Vec<u8>> {
    let n = match format {
        Format::Bc1 | Format::Etc2Rgb => 5,
        Format::Bc3 | Format::Etc2Rgba => 8,
        Format::Bc4 | Format::EacR11 => 3,
        Format::Bc5 | Format::EacRg11 => 6,
        Format::Bc6h | Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 1,
        Format::Bc7 => 6,
    };
    let mut lens = Vec::with_capacity(n);
    for k in 0..n {
        let b = data.get(4 * k..4 * k + 4)?;
        lens.push(u32::from_le_bytes(b.try_into().ok()?) as usize);
    }
    let mut streams = Vec::with_capacity(n);
    let mut at = 4 * n;
    for &l in &lens {
        streams.push(data.get(at..at + l)?);
        at += l;
    }
    if at != data.len() {
        return None;
    }
    match format {
        Format::Bc1 | Format::Etc2Rgb => join_bc1(&streams),
        Format::Bc3 | Format::Etc2Rgba => join_bc3(&streams),
        Format::Bc4 | Format::EacR11 => join_bc4(&streams, 1),
        Format::Bc5 | Format::EacRg11 => join_bc4(&streams, 2),
        Format::Bc6h | Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => {
            Some(streams[0].to_vec()).filter(|s| s.len() % 16 == 0)
        }
        Format::Bc7 => join_bc7(&streams),
    }
}

fn split_bc1(raw: &[u8]) -> Vec<Vec<u8>> {
    let n = raw.len() / 8;
    let mut s: Vec<Vec<u8>> = vec![
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(4 * n),
    ];
    // Planes: c0 low, c0 high, c1 low, c1 high, selectors.
    for b in raw.as_chunks::<8>().0 {
        s[0].push(b[0]);
        s[1].push(b[1]);
        s[2].push(b[2]);
        s[3].push(b[3]);
        s[4].extend_from_slice(&b[4..8]);
    }
    s
}

fn join_bc1(s: &[&[u8]]) -> Option<Vec<u8>> {
    let n = s[0].len();
    if s[1].len() != n || s[2].len() != n || s[3].len() != n || s[4].len() != 4 * n {
        return None;
    }
    let mut out = Vec::with_capacity(8 * n);
    for i in 0..n {
        out.extend_from_slice(&[s[0][i], s[1][i], s[2][i], s[3][i]]);
        out.extend_from_slice(&s[4][4 * i..4 * i + 4]);
    }
    Some(out)
}

fn split_bc3(raw: &[u8]) -> Vec<Vec<u8>> {
    let (mut alpha, mut color) = (Vec::with_capacity(raw.len() / 2), Vec::with_capacity(raw.len() / 2));
    for b in raw.as_chunks::<16>().0 {
        alpha.extend_from_slice(&b[..8]);
        color.extend_from_slice(&b[8..]);
    }
    let mut s = split_bc4(&alpha, 1);
    s.extend(split_bc1(&color));
    s
}

fn join_bc3(s: &[&[u8]]) -> Option<Vec<u8>> {
    let alpha = join_bc4(&s[..3], 1)?;
    let color = join_bc1(&s[3..])?;
    if alpha.len() != color.len() {
        return None;
    }
    let mut out = Vec::with_capacity(2 * alpha.len());
    for (a, c) in alpha.as_chunks::<8>().0.iter().zip(color.as_chunks::<8>().0) {
        out.extend_from_slice(a);
        out.extend_from_slice(c);
    }
    Some(out)
}

fn split_bc4(raw: &[u8], channels: usize) -> Vec<Vec<u8>> {
    let bb = 8 * channels;
    let n = raw.len() / bb;
    let mut s: Vec<Vec<u8>> = (0..3 * channels).map(|_| Vec::with_capacity(6 * n)).collect();
    for b in raw.chunks_exact(bb) {
        for c in 0..channels {
            let h = &b[8 * c..8 * c + 8];
            s[c].push(h[0]);
            s[channels + c].push(h[1]);
            s[2 * channels + c].extend_from_slice(&h[2..8]);
        }
    }
    s
}

fn join_bc4(s: &[&[u8]], channels: usize) -> Option<Vec<u8>> {
    let n = s[0].len();
    for c in 0..channels {
        if s[c].len() != n || s[channels + c].len() != n || s[2 * channels + c].len() != 6 * n {
            return None;
        }
    }
    let mut out = Vec::with_capacity(8 * channels * n);
    for i in 0..n {
        for c in 0..channels {
            out.push(s[c][i]);
            out.push(s[channels + c][i]);
            out.extend_from_slice(&s[2 * channels + c][6 * i..6 * i + 6]);
        }
    }
    Some(out)
}

/// Streams of the BC7 layout.
const MODE: usize = 0;
const SHAPE: usize = 1;
const ENDPOINTS: usize = 2;
const PBITS: usize = 3;
const INDICES: usize = 4;
const ESCAPE: usize = 5;

/// Index bits of a block of `mode` (anchors carry one bit less): primary and secondary sets.
fn index_bits(mode: u8, partition: u8) -> Vec<u8> {
    let mi = bc7::MODES[mode as usize];
    let anc = bc7::anchors(mi.ns, partition);
    let mut out: Vec<u8> = (0..16).map(|i| mi.ib - anc.contains(&i) as u8).collect();
    if mi.ib2 > 0 {
        out.extend((0..16).map(|i| mi.ib2 - (i == 0) as u8));
    }
    out
}

/// Little bit writer / reader for the index stream (bits low to high, flushed per block).
struct BitOut<'a> {
    out: &'a mut Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitOut<'_> {
    fn put(&mut self, v: u8, bits: u8) {
        self.acc |= (v as u32) << self.n;
        self.n += bits as u32;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    fn flush(&mut self) {
        if self.n > 0 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.n = 0;
        }
    }
}

struct BitIn<'a> {
    src: &'a [u8],
    pos: usize,
    acc: u32,
    n: u32,
}

impl BitIn<'_> {
    fn get(&mut self, bits: u8) -> Option<u8> {
        while self.n < bits as u32 {
            self.acc |= (*self.src.get(self.pos)? as u32) << self.n;
            self.pos += 1;
            self.n += 8;
        }
        let v = (self.acc & ((1 << bits) - 1)) as u8;
        self.acc >>= bits;
        self.n -= bits as u32;
        Some(v)
    }
    fn align(&mut self) {
        self.acc = 0;
        self.n = 0;
    }
}

fn split_bc7(raw: &[u8]) -> Vec<Vec<u8>> {
    let n = raw.len() / 16;
    let mut s: Vec<Vec<u8>> = vec![
        Vec::with_capacity(n),
        Vec::new(),
        Vec::with_capacity(8 * n),
        Vec::new(),
        Vec::with_capacity(8 * n),
        Vec::new(),
    ];
    for b in raw.as_chunks::<16>().0 {
        let Some(blk) = bc7::unpack(b) else {
            s[MODE].push(8);
            s[ESCAPE].extend_from_slice(b);
            continue;
        };
        let mi = bc7::MODES[blk.mode as usize];
        s[MODE].push(blk.mode);
        if mi.pb > 0 {
            s[SHAPE].push(blk.partition);
        } else if mi.rb > 0 {
            s[SHAPE].push(blk.rotation | (blk.idx_sel << 2));
        }
        let channels = if mi.ab > 0 { 4 } else { 3 };
        for sub in 0..mi.ns as usize {
            for e in 0..2 {
                for c in 0..channels {
                    let q = blk.ep[sub][e][c];
                    let v = if mi.epb == 1 { (q << 1) | blk.pbits[sub][e] } else { q };
                    s[ENDPOINTS].push(v);
                }
            }
        }
        if mi.spb == 1 {
            let mut p = 0u8;
            for sub in 0..mi.ns as usize {
                p |= blk.pbits[sub][0] << sub;
            }
            s[PBITS].push(p);
        }
        let bits = index_bits(blk.mode, blk.partition);
        let (mut idx_stream, mut bw) = (std::mem::take(&mut s[INDICES]), 0);
        {
            let mut w = BitOut {
                out: &mut idx_stream,
                acc: 0,
                n: 0,
            };
            for (i, &nb) in bits.iter().enumerate() {
                let v = if i < 16 { blk.idx[i] } else { blk.idx2[i - 16] };
                w.put(v, nb);
                bw += nb as u32;
            }
            w.flush();
        }
        let _ = bw;
        s[INDICES] = idx_stream;
    }
    s
}

fn join_bc7(s: &[&[u8]]) -> Option<Vec<u8>> {
    let n = s[MODE].len();
    let mut out = Vec::with_capacity(16 * n);
    let (mut shape, mut ep, mut pb, mut esc) = (0usize, 0usize, 0usize, 0usize);
    let mut idx = BitIn {
        src: s[INDICES],
        pos: 0,
        acc: 0,
        n: 0,
    };
    for i in 0..n {
        let mode = s[MODE][i];
        if mode == 8 {
            out.extend_from_slice(s[ESCAPE].get(esc..esc + 16)?);
            esc += 16;
            continue;
        }
        if mode > 7 {
            return None;
        }
        let mi = bc7::MODES[mode as usize];
        let mut blk = bc7::Bc7Block {
            mode,
            ..Default::default()
        };
        if mi.pb > 0 {
            blk.partition = *s[SHAPE].get(shape)?;
            shape += 1;
        } else if mi.rb > 0 {
            let v = *s[SHAPE].get(shape)?;
            shape += 1;
            blk.rotation = v & 3;
            blk.idx_sel = v >> 2;
        }
        let channels = if mi.ab > 0 { 4 } else { 3 };
        for sub in 0..mi.ns as usize {
            for e in 0..2 {
                for c in 0..channels {
                    let v = *s[ENDPOINTS].get(ep)?;
                    ep += 1;
                    if mi.epb == 1 {
                        blk.ep[sub][e][c] = v >> 1;
                        blk.pbits[sub][e] = v & 1;
                    } else {
                        blk.ep[sub][e][c] = v;
                    }
                }
            }
        }
        if mi.spb == 1 {
            let p = *s[PBITS].get(pb)?;
            pb += 1;
            for sub in 0..mi.ns as usize {
                blk.pbits[sub] = [(p >> sub) & 1; 2];
            }
        }
        let bits = index_bits(mode, blk.partition);
        for (k, &nb) in bits.iter().enumerate() {
            let v = idx.get(nb)?;
            if k < 16 {
                blk.idx[k] = v;
            } else {
                blk.idx2[k - 16] = v;
            }
        }
        idx.align();
        out.extend_from_slice(&bc7::pack(&blk));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(format: Format, raw: &[u8]) {
        let t = forward(format, raw);
        assert_eq!(inverse(format, &t).as_deref(), Some(raw), "{format:?}");
    }

    #[test]
    fn roundtrips_real_and_random_blocks() {
        let surf = crate::synth::foliage(64);
        for format in Format::ALL {
            for effort in [0u8, 5] {
                let opts = crate::pipeline::EncodeOptions {
                    effort,
                    mips: false,
                    ..crate::pipeline::EncodeOptions::new(format)
                };
                let (tex, _) = crate::pipeline::encode(&surf, &opts);
                roundtrip(format, &tex.levels[0].data);
            }
            // Arbitrary bytes: every BC1/BC4/BC5 pattern is valid; for BC7 they hit every mode,
            // including the invalid one.
            let mut x = 0x2545_F491_4F6C_DD1Du64;
            let raw: Vec<u8> = (0..16 * 512)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect();
            let n = raw.len() / format.block_bytes() * format.block_bytes();
            roundtrip(format, &raw[..n]);
        }
    }

    /// Diagnostic: compressed sizes with and without the transform on Kodak 1-6.
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn sizes() {
        for format in Format::ALL {
            for lambda in [0.0f32, 1.0, 4.0] {
                let (mut z0, mut z1, mut d0, mut d1) = (0usize, 0usize, 0usize, 0usize);
                for k in 1..=6 {
                    let s = crate::surface::Surface::load(std::path::Path::new(&format!(
                        "test-images/kodak/kodim{k:02}.png"
                    )))
                    .unwrap();
                    let mut opts = crate::pipeline::EncodeOptions::new(format);
                    opts.mips = false;
                    opts.rdo = crate::rdo::RdoParams {
                        lambda,
                        ..Default::default()
                    };
                    let (tex, _) = crate::pipeline::encode(&s, &opts);
                    let raw = &tex.levels[0].data;
                    let t = forward(format, raw);
                    z0 += zstd::bulk::compress(raw, 19).unwrap().len();
                    z1 += zstd::bulk::compress(&t, 19).unwrap().len();
                    d0 += crate::bench::deflate_len(raw, 9).unwrap();
                    d1 += crate::bench::deflate_len(&t, 9).unwrap();
                }
                println!(
                    "{format:?} λ={lambda}: zstd {z0} -> {z1} ({:+.2}%), deflate {d0} -> {d1} ({:+.2}%)",
                    (z1 as f64 / z0 as f64 - 1.0) * 100.0,
                    (d1 as f64 / d0 as f64 - 1.0) * 100.0
                );
            }
        }
    }
}
