//! BC7 (BPTC) block codec.
//!
//! * [`unpack`] / [`pack`] convert between the 128-bit block and [`Bc7Block`] for all 8 modes.
//! * [`decode`] decodes all 8 modes (rotation and index-select included).
//! * [`encode`] is the scalar encoder for every mode and effort: the reference, and the encoder
//!   for blocks with alpha and for entropy-reduced (RDO) bases. [`fast`] is the SIMD encoder for
//!   opaque blocks (and the GPU kernel's twin); [`encode_batch`] runs entropy-reduced search
//!   levels 0-1.
//! * [`reindex_bounded`], [`with_indices`] and [`with_layout`] are the field-level edits used
//!   by RDO.

mod batch;
pub mod fast;
mod partition;
pub use batch::encode_batch;
pub(crate) mod tables;

use crate::fit::{V4, ls_endpoints, mean_cov, pca_endpoints};
use crate::format::BlockOpts;
use crate::surface::{Block, texel_sse};
use tables::{ANCHOR2, ANCHOR3, PART2, PART3};

/// The fast search ([`fast`]) for every effort, when entropy reduction is off (it changes mode 6's
/// quantization, which only the full encoders implement). Each tier is the cheapest search
/// found to beat a competitor tier on Kodak (see docs/BENCHMARKS.md).
pub fn fast_cfg(opts: BlockOpts) -> Option<fast::Cfg> {
    if opts.reduce_entropy {
        return None;
    }
    let base = fast::Cfg {
        power: 3,
        refits: 1,
        two: 0,
        two_refits: 0,
        parts: 1,
        rank: false,
        refine: 0,
        busy_err: 0.0,
        refine6: true,
        three: 0,
        parts_three: 1,
        mode5: false,
        uber: 0,
        polish_modes: false,
        polish_parts: false,
        polish_top: 0,
    };
    // Mode 1 on the 4 best ranked partitions with a refit and ±1 refinement.
    let deep = fast::Cfg {
        two: 1,
        two_refits: 1,
        parts: 4,
        rank: true,
        refine: 1,
        ..base
    };
    let top = fast::Cfg {
        two: 3,
        three: 3,
        parts_three: 4,
        mode5: true,
        refine: 2,
        refits: 2,
        two_refits: 2,
        ..deep
    };
    Some(match opts.effort {
        // Mode 6 only.
        0 => fast::Cfg { refits: 0, ..base },
        // + mode 1 on bc7f's popcount partition estimate (one or two candidates), for blocks
        // that mode 6 leaves enough error.
        1 => fast::Cfg {
            refits: 0,
            two: 1,
            busy_err: 512.0,
            ..base
        },
        2 => fast::Cfg {
            two: 1,
            parts: 2,
            busy_err: 128.0,
            ..base
        },
        // + mode 1 on the best partition by line-fit ranking.
        3 => fast::Cfg {
            two: 1,
            rank: true,
            ..base
        },
        4 => fast::Cfg { refine6: false, ..deep },
        5 => deep,
        // + modes 3, 0, 2 and 5, with two refinement passes.
        6 => fast::Cfg {
            two: 3,
            three: 3,
            parts_three: 2,
            mode5: true,
            refine: 2,
            refits: 2,
            ..deep
        },
        // + more partitions and refits.
        7 => top,
        // bc7e-style: uber index-set refits, and every mode's best (plus its top two
        // candidates, or every candidate at effort 9) refined before the modes compete.
        8 => fast::Cfg {
            uber: 4,
            polish_modes: true,
            polish_parts: true,
            polish_top: 2,
            ..top
        },
        _ => fast::Cfg {
            uber: 4,
            polish_modes: true,
            polish_parts: true,
            ..top
        },
    })
}

/// The plain search RDO keeps as a second base beside an entropy-reduced one: [`fast_cfg`], except
/// that effort 8 polishes only each mode's best candidate (not its top two), refines ±1 once (not
/// twice) and stretches uber index sets by at most one step (not three). RDO replaces or re-indexes
/// most of these blocks, so the extra search mostly goes unused: BC7 RDO λ 1 on Kodak-24 takes 15%
/// less CPU time, and vs Oodle 2.9.16 the size at equal RMSE moves by at most 0.1% at any λ (see
/// docs/BENCHMARKS.md). Effort 7's search (no uber trials or polish at all) would lose 0.4%.
pub fn rdo_alt_cfg(opts: BlockOpts) -> Option<fast::Cfg> {
    let c = fast_cfg(opts)?;
    Some(if opts.effort == 8 {
        fast::Cfg {
            polish_top: 1,
            refine: 1,
            uber: 2,
            ..c
        }
    } else {
        c
    })
}

/// Mode 6 alone (effort 0's search) with two refits: an RDO base whose blocks all share one layout.
pub fn mode6_cfg() -> fast::Cfg {
    fast::Cfg {
        refits: 2,
        ..fast_cfg(BlockOpts {
            effort: 0,
            reduce_entropy: false,
        })
        .expect("plain BC7 has a fast configuration")
    }
}

/// Static description of one BC7 mode.
#[derive(Clone, Copy, Debug)]
pub struct ModeInfo {
    /// Number of subsets.
    pub ns: u8,
    /// Partition bits.
    pub pb: u8,
    /// Rotation bits.
    pub rb: u8,
    /// Index-selection bits.
    pub isb: u8,
    /// Color bits per channel (without p-bit).
    pub cb: u8,
    /// Alpha bits (0 = opaque mode).
    pub ab: u8,
    /// One p-bit per endpoint.
    pub epb: u8,
    /// One shared p-bit per subset.
    pub spb: u8,
    /// Primary index bits.
    pub ib: u8,
    /// Secondary index bits (modes 4 and 5).
    pub ib2: u8,
}

#[expect(clippy::too_many_arguments, reason = "one row of the mode table")]
const fn m(ns: u8, pb: u8, rb: u8, isb: u8, cb: u8, ab: u8, epb: u8, spb: u8, ib: u8, ib2: u8) -> ModeInfo {
    ModeInfo {
        ns,
        pb,
        rb,
        isb,
        cb,
        ab,
        epb,
        spb,
        ib,
        ib2,
    }
}

pub const MODES: [ModeInfo; 8] = [
    m(3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    m(2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    m(3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    m(2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    m(1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    m(1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    m(1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    m(2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
];

const W2: [u32; 4] = [0, 21, 43, 64];
const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

#[inline]
fn weights(bits: u8) -> &'static [u32] {
    match bits {
        2 => &W2,
        3 => &W3,
        _ => &W4,
    }
}

#[inline]
fn interp(e0: u32, e1: u32, w: u32) -> u8 {
    (((64 - w) * e0 + w * e1 + 32) >> 6) as u8
}

/// A BC7 block with its fields unpacked. Endpoint values are stored quantized, without p-bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bc7Block {
    pub mode: u8,
    pub partition: u8,
    pub rotation: u8,
    pub idx_sel: u8,
    /// `ep[subset][endpoint][channel]`.
    pub ep: [[[u8; 4]; 2]; 3],
    /// `pbits[subset][endpoint]` (equal pairs for shared-p-bit modes).
    pub pbits: [[u8; 2]; 3],
    pub idx: [u8; 16],
    pub idx2: [u8; 16],
}

/// Subset of texel `i` for a partition.
#[inline]
pub fn subset_of(ns: u8, part: u8, i: usize) -> usize {
    match ns {
        2 => ((PART2[part as usize] >> i) & 1) as usize,
        3 => ((PART3[part as usize] >> (2 * i)) & 3) as usize,
        _ => 0,
    }
}

/// Anchor texel of each subset (`usize::MAX` for unused subsets).
#[inline]
pub fn anchors(ns: u8, part: u8) -> [usize; 3] {
    let p = part as usize;
    match ns {
        2 => [0, ANCHOR2[p] as usize, usize::MAX],
        3 => [0, ANCHOR3[0][p] as usize, ANCHOR3[1][p] as usize],
        _ => [0, usize::MAX, usize::MAX],
    }
}

struct Bits {
    v: u128,
    pos: u32,
}

impl Bits {
    #[inline]
    fn read(&mut self, n: u8) -> u8 {
        if n == 0 {
            return 0;
        }
        let r = ((self.v >> self.pos) & ((1u128 << n) - 1)) as u8;
        self.pos += n as u32;
        r
    }

    #[inline]
    fn write(&mut self, val: u8, n: u8) {
        if n == 0 {
            return;
        }
        self.v |= ((val as u128) & ((1u128 << n) - 1)) << self.pos;
        self.pos += n as u32;
    }
}

/// Unpacks a 16-byte block. Returns `None` for the reserved all-zero mode byte.
pub fn unpack(raw: &[u8]) -> Option<Bc7Block> {
    let (mut blk, mut b) = unpack_head(raw)?;
    let mi = MODES[blk.mode as usize];
    let anc = anchors(mi.ns, blk.partition);
    for i in 0..16 {
        let n = mi.ib - anc.contains(&i) as u8;
        blk.idx[i] = b.read(n);
    }
    if mi.ib2 > 0 {
        for i in 0..16 {
            let n = mi.ib2 - (i == 0) as u8;
            blk.idx2[i] = b.read(n);
        }
    }
    Some(blk)
}

/// Every field before the indices (left zero), and the reader positioned at the indices.
fn unpack_head(raw: &[u8]) -> Option<(Bc7Block, Bits)> {
    let v = u128::from_le_bytes(raw[..16].try_into().ok()?);
    // One copy per mode: with the layout constant, the reads unroll to fixed shifts (RDO
    // decodes the heads of thousands of candidate blocks per block).
    Some(match (v as u8).trailing_zeros() {
        0 => unpack_head_mode::<0>(v),
        1 => unpack_head_mode::<1>(v),
        2 => unpack_head_mode::<2>(v),
        3 => unpack_head_mode::<3>(v),
        4 => unpack_head_mode::<4>(v),
        5 => unpack_head_mode::<5>(v),
        6 => unpack_head_mode::<6>(v),
        7 => unpack_head_mode::<7>(v),
        _ => return None,
    })
}

#[inline(always)]
fn unpack_head_mode<const MODE: u8>(v: u128) -> (Bc7Block, Bits) {
    let mode = MODE;
    let mi = const { MODES[MODE as usize] };
    let mut b = Bits {
        v,
        pos: mode as u32 + 1,
    };
    let mut blk = Bc7Block {
        mode,
        ..Default::default()
    };
    blk.partition = b.read(mi.pb);
    blk.rotation = b.read(mi.rb);
    blk.idx_sel = b.read(mi.isb);
    let ns = mi.ns as usize;
    for c in 0..3 {
        for s in 0..ns {
            for e in 0..2 {
                blk.ep[s][e][c] = b.read(mi.cb);
            }
        }
    }
    if mi.ab > 0 {
        for s in 0..ns {
            for e in 0..2 {
                blk.ep[s][e][3] = b.read(mi.ab);
            }
        }
    }
    if mi.epb == 1 {
        for s in 0..ns {
            for e in 0..2 {
                blk.pbits[s][e] = b.read(1);
            }
        }
    }
    if mi.spb == 1 {
        for s in 0..ns {
            let p = b.read(1);
            blk.pbits[s] = [p, p];
        }
    }
    (blk, b)
}

/// Packs a block. Anchor indices must already have their top bit clear.
pub fn pack(blk: &Bc7Block) -> [u8; 16] {
    let mi = MODES[blk.mode as usize];
    let mut b = Bits { v: 0, pos: 0 };
    b.write(1 << blk.mode, blk.mode + 1);
    b.write(blk.partition, mi.pb);
    b.write(blk.rotation, mi.rb);
    b.write(blk.idx_sel, mi.isb);
    let ns = mi.ns as usize;
    for c in 0..3 {
        for s in 0..ns {
            for e in 0..2 {
                b.write(blk.ep[s][e][c], mi.cb);
            }
        }
    }
    if mi.ab > 0 {
        for s in 0..ns {
            for e in 0..2 {
                b.write(blk.ep[s][e][3], mi.ab);
            }
        }
    }
    if mi.epb == 1 {
        for s in 0..ns {
            for e in 0..2 {
                b.write(blk.pbits[s][e], 1);
            }
        }
    }
    if mi.spb == 1 {
        for s in 0..ns {
            b.write(blk.pbits[s][0], 1);
        }
    }
    let anc = anchors(mi.ns, blk.partition);
    for i in 0..16 {
        let n = mi.ib - anc.contains(&i) as u8;
        b.write(blk.idx[i], n);
    }
    if mi.ib2 > 0 {
        for i in 0..16 {
            let n = mi.ib2 - (i == 0) as u8;
            b.write(blk.idx2[i], n);
        }
    }
    debug_assert_eq!(b.pos, 128);
    b.v.to_le_bytes()
}

/// Expands a quantized channel (plus optional p-bit) to 8 bits by bit replication.
#[inline(always)]
fn dequant(q: u8, bits: u8, p: Option<u8>) -> u32 {
    if bits == 0 {
        return 255;
    }
    let (mut v, mut n) = (q as u32, bits as u32);
    if let Some(p) = p {
        v = (v << 1) | p as u32;
        n += 1;
    }
    (v << (8 - n)) | (v >> (2 * n - 8))
}

#[inline(always)]
fn dequant_endpoint(q: &[u8; 4], mi: &ModeInfo, p: u8) -> [u32; 4] {
    let p = (mi.epb == 1 || mi.spb == 1).then_some(p);
    [
        dequant(q[0], mi.cb, p),
        dequant(q[1], mi.cb, p),
        dequant(q[2], mi.cb, p),
        dequant(q[3], mi.ab, p),
    ]
}

/// 8-bit endpoint colors per subset, before rotation.
pub fn endpoint_colors(b: &Bc7Block) -> [[[u32; 4]; 2]; 3] {
    let mi = MODES[b.mode as usize];
    let mut out = [[[0u32; 4]; 2]; 3];
    for s in 0..mi.ns as usize {
        for e in 0..2 {
            out[s][e] = dequant_endpoint(&b.ep[s][e], &mi, b.pbits[s][e]);
        }
    }
    out
}

/// Decodes an unpacked block.
pub fn decode_block(b: &Bc7Block, out: &mut Block) {
    let mi = MODES[b.mode as usize];
    let cols = endpoint_colors(b);
    let w1 = weights(mi.ib);
    let w2 = weights(mi.ib2.max(2));
    for (i, t) in out.iter_mut().enumerate() {
        let s = subset_of(mi.ns, b.partition, i);
        let (e0, e1) = (cols[s][0], cols[s][1]);
        let (cw, aw) = if mi.ib2 == 0 {
            let w = w1[b.idx[i] as usize];
            (w, w)
        } else if b.idx_sel == 0 {
            (w1[b.idx[i] as usize], w2[b.idx2[i] as usize])
        } else {
            (w2[b.idx2[i] as usize], w1[b.idx[i] as usize])
        };
        let mut p = [
            interp(e0[0], e1[0], cw),
            interp(e0[1], e1[1], cw),
            interp(e0[2], e1[2], cw),
            interp(e0[3], e1[3], aw),
        ];
        match b.rotation {
            1 => p.swap(0, 3),
            2 => p.swap(1, 3),
            3 => p.swap(2, 3),
            _ => {}
        }
        *t = p;
    }
}

/// Mode 6 straight from the bits: a 7-bit prefix, R0 R1 G0 G1 B0 B1 A0 A1 (7 bits each), the two
/// p-bits, then 16 indices of 4 bits (3 for the anchor, texel 0). Each 7-bit channel plus its
/// p-bit is already the 8-bit endpoint. The RDO pass decodes hundreds of candidates per block,
/// and entropy reduction makes most of them mode 6.
fn decode_mode6(v: u128, out: &mut Block) {
    let pal = mode6_palette(v as u64, (v >> 64) as u64 & 1);
    let idx = mode6_indices((v >> 64) as u64);
    for (t, &k) in out.iter_mut().zip(&idx) {
        *t = pal[k as usize];
    }
}

/// The 8-bit endpoints of mode 6 from the low 64 bits (7-bit channels, P0 at bit 63) and P1.
#[inline]
fn mode6_endpoints(lo: u64, p1: u64) -> [[u32; 4]; 2] {
    let ch = |pos: u32| ((lo >> pos) & 0x7f) as u32;
    let p0 = (lo >> 63) as u32;
    let p1 = p1 as u32 & 1;
    [
        [0, 1, 2, 3].map(|c| ch(7 + 14 * c) << 1 | p0),
        [0, 1, 2, 3].map(|c| ch(14 + 14 * c) << 1 | p1),
    ]
}

/// The 8-bit endpoints of a mode-6 block from its first 8 bytes, with P1 (bit 64, past them) clear.
pub fn mode6_head_endpoints(raw: &[u8]) -> [[u32; 4]; 2] {
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&raw[..8]);
    mode6_endpoints(u64::from_le_bytes(lo), 0)
}

/// What the first 8 bytes of a multi-subset block fix of its endpoints ([`multi_head_endpoints`]).
pub struct HeadEndpoints {
    /// Subsets (2 or 3).
    pub ns: u8,
    /// Partition.
    pub part: u8,
    /// `mid[s][e][c]`: endpoint `e` of subset `s`, channel `c`, 8-bit, halfway between its two
    /// p-bit choices (where `known[s][c]`).
    pub mid: [[[f32; 4]; 2]; 3],
    /// Whether both endpoints of channel `c` of subset `s` lie in those bytes.
    pub known: [[bool; 4]; 3],
    /// Distance from `mid` to either p-bit choice (0 without p-bits).
    pub half: f32,
}

/// What the first 8 bytes of a multi-subset block (modes 0-3 and 7) fix of its endpoints, for a
/// lower bound on re-indexing it with the rest taken from another block. The p-bits lie past
/// byte 8, so a known channel is pinned to within `half` of `mid`. The opaque modes' alpha (255)
/// is known with no p-bit uncertainty. Other modes give `None`.
pub fn multi_head_endpoints(raw: &[u8]) -> Option<HeadEndpoints> {
    let lo = u64::from_le_bytes(raw[..8].try_into().ok()?);
    let mode = (lo as u8).trailing_zeros() as usize;
    if !matches!(mode, 0..=3 | 7) {
        return None;
    }
    let mi = MODES[mode];
    let ns = mi.ns as usize;
    let part = ((lo >> (mode + 1)) & ((1 << mi.pb) - 1)) as u8;
    let start = mode + 1 + mi.pb as usize;
    let cb = mi.cb as usize;
    let pbit = mi.epb + mi.spb > 0;
    let bits = cb + usize::from(pbit);
    let half = if pbit { (1u32 << (8 - bits)) as f32 / 2.0 } else { 0.0 };
    let mut mid = [[[0f32; 4]; 2]; 3];
    let mut known = [[false; 4]; 3];
    for s in 0..ns {
        if mi.ab == 0 {
            known[s][3] = true;
            mid[s] = [[0.0, 0.0, 0.0, 255.0]; 2];
        }
        for c in 0..3 {
            let at = |e: usize| start + (c * 2 * ns + 2 * s + e) * cb;
            if at(1) + cb > 64 {
                continue;
            }
            known[s][c] = true;
            for e in 0..2 {
                let q = ((lo >> at(e)) & ((1 << cb) - 1)) as u32;
                // The p-bit-clear value; with the p-bit set it is 2 * half more (the low bits
                // replicated into the expansion are the top ones, which the p-bit leaves alone).
                let v = if pbit { q << 1 } else { q };
                let x = (v << (8 - bits)) | (v >> (2 * bits - 8));
                mid[s][e][c] = x as f32 + half;
            }
        }
    }
    Some(HeadEndpoints {
        ns: mi.ns,
        part,
        mid,
        known,
        half,
    })
}

/// The 16-entry mode-6 palette from the low 64 bits of a mode-6 block and P1.
fn mode6_palette(lo: u64, p1: u64) -> [[u8; 4]; 16] {
    let e = mode6_endpoints(lo, p1);
    let mut pal = [[0u8; 4]; 16];
    for c in 0..4 {
        for (k, &w) in W4.iter().enumerate() {
            pal[k][c] = interp(e[0][c], e[1][c], w);
        }
    }
    pal
}

/// The 16 mode-6 indices from the high 64 bits (P1, a 3-bit anchor index, then 4 bits each).
#[inline]
fn mode6_indices(hi: u64) -> [u8; 16] {
    let mut idx = [0u8; 16];
    idx[0] = (hi >> 1 & 7) as u8;
    for (i, k) in idx.iter_mut().enumerate().skip(1) {
        *k = (hi >> (4 * i) & 15) as u8;
    }
    idx
}

/// Whether a block is mode 6.
#[inline]
pub fn is_mode6(raw: &[u8]) -> bool {
    raw[0] & 0x7f == 0x40
}

/// Errors of a mode-6 block (its endpoints and P0) against every texel at every palette entry,
/// for both values of P1. A candidate that keeps the low 8 bytes and takes P1 and the indices
/// (the high 8 bytes) from another block then costs 16 lookups instead of a decode.
pub struct Mode6Lo {
    err: [[[u32; 16]; 16]; 2],
}

impl Mode6Lo {
    pub fn new(raw: &[u8; 16], src: &Block, w: &[u32; 4]) -> Self {
        let lo = u64::from_le_bytes(raw[..8].try_into().unwrap());
        let mut err = [[[0u32; 16]; 16]; 2];
        for (p1, tab) in err.iter_mut().enumerate() {
            let pal = mode6_palette(lo, p1 as u64);
            for (row, t) in tab.iter_mut().zip(src) {
                for (e, p) in row.iter_mut().zip(&pal) {
                    *e = texel_sse(t, p, w);
                }
            }
        }
        Self { err }
    }

    /// Error with `hi` (bytes 8-15 as a little-endian word) in place of the high half; at least
    /// `bound` + 1 once it passes `bound`. Equal to decoding the block and taking [`block_sse`].
    ///
    /// [`block_sse`]: crate::surface::block_sse
    pub fn err(&self, hi: u64, bound: u64) -> u64 {
        let tab = &self.err[(hi & 1) as usize];
        let idx = mode6_indices(hi);
        let mut e = 0u64;
        for q in 0..4 {
            for i in q * 4..q * 4 + 4 {
                e += tab[i][idx[i] as usize] as u64;
            }
            if e > bound {
                return e;
            }
        }
        e
    }
}

/// The interpolation weight of each texel of a mode-6 block. A candidate that keeps P1 and the
/// indices (the high 8 bytes) and takes the endpoints and P0 (the low 8 bytes, mode 6) from
/// another block is scored by [`Mode6Hi::err`] without a decode.
pub struct Mode6Hi {
    wt: [u32; 16],
    p1: u64,
}

impl Mode6Hi {
    pub fn new(raw: &[u8; 16]) -> Self {
        let hi = u64::from_le_bytes(raw[8..].try_into().unwrap());
        Self {
            wt: mode6_indices(hi).map(|k| W4[k as usize]),
            p1: hi & 1,
        }
    }

    /// Error with `lo` (bytes 0-7 of a mode-6 block) in place of the low half; at least `bound`
    /// + 1 once it passes `bound`. Equal to decoding the block and taking [`block_sse`].
    ///
    /// [`block_sse`]: crate::surface::block_sse
    pub fn err(&self, lo: u64, src: &Block, w: &[u32; 4], bound: u64) -> u64 {
        let ep = mode6_endpoints(lo, self.p1);
        let mut e = 0u64;
        for q in 0..4 {
            for i in q * 4..q * 4 + 4 {
                let t = [0, 1, 2, 3].map(|c| interp(ep[0][c], ep[1][c], self.wt[i]));
                e += texel_sse(&src[i], &t, w) as u64;
            }
            if e > bound {
                return e;
            }
        }
        e
    }
}

/// Decodes one 16-byte block. Reserved (mode byte 0) blocks decode to transparent black.
pub fn decode(raw: &[u8], out: &mut Block) {
    if raw[0] & 0x7f == 0x40 {
        return decode_mode6(u128::from_le_bytes(raw[..16].try_into().unwrap()), out);
    }
    match unpack(raw) {
        Some(b) => decode_block(&b, out),
        None => *out = [[0; 4]; 16],
    }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Quantized {
    q: [[u8; 4]; 2],
    p: [u8; 2],
}

/// Closest quantized value (with optional p-bit) to `v` after bit-replication expansion.
fn quant_channel(v: f32, bits: u8, p: Option<u8>) -> u8 {
    let maxq = (1i32 << bits) - 1;
    let n = bits as u32 + p.is_some() as u32;
    let mut ideal = v.clamp(0.0, 255.0) * ((1u32 << n) - 1) as f32 / 255.0;
    if let Some(p) = p {
        ideal = (ideal - p as f32) / 2.0;
    }
    // `+ 0.5` and truncate: `f32::round` is a libm call on baseline x86-64. `ideal` >= -0.5,
    // and the neighbours are searched anyway.
    let c = (ideal + 0.5) as i32;
    let mut best = (f32::MAX, 0u8);
    for q in (c - 1)..=(c + 1) {
        if q < 0 || q > maxq {
            continue;
        }
        let d = dequant(q as u8, bits, p) as f32 - v;
        if d * d < best.0 {
            best = (d * d, q as u8);
        }
    }
    best.1
}

/// Mode 6 under entropy reduction: the 7-bit endpoint field is limited to 64 evenly spread
/// levels (`(j * 127 + 31) / 63`), with the p-bit fixed by the caller (bc7enc's
/// `m_quant_mode6_endpoints`). Fewer distinct endpoint bytes means more LZ matches.
fn quant_channel_mode6_reduced(v: f32, p: u8) -> u8 {
    let jc = ((v.clamp(0.0, 255.0) - p as f32) / 2.0 * 63.0 / 127.0 + 0.5) as i32;
    let mut best = (f32::MAX, 0u8);
    for j in (jc - 1).max(0)..=(jc + 1).min(63) {
        let q = ((j * 127 + 31) / 63) as u8;
        let d = dequant(q, 7, Some(p)) as f32 - v;
        if d * d < best.0 {
            best = (d * d, q);
        }
    }
    best.1
}

/// Entropy-reduction tuning, from bc7enc's `-e` (`rdo_bc_encoder.cpp`). Its p-bit-1 weighting
/// (1.3) and low-frequency partition bias are left out: on mipple's check corpora the first made
/// no difference with mode 6 quantized and cost +0.5% BD-rate on its own.
#[derive(Clone, Copy, Debug)]
struct Entropy {
    /// Mode 6's error is scaled by this before comparing with other modes (opaque blocks).
    mode6_weight_opaque: f32,
    /// Same, for blocks with alpha.
    mode6_weight_alpha: f32,
    /// Restrict mode 6 endpoints to 64 levels with fixed p-bits.
    quant_mode6: bool,
}

const ENTROPY: Entropy = Entropy {
    mode6_weight_opaque: 0.4,
    mode6_weight_alpha: 0.6,
    quant_mode6: true,
};

fn quant_endpoint(v: &V4, mi: &ModeInfo, p: Option<u8>, w: &[u32; 4], dims: usize) -> ([u8; 4], f32) {
    let mut q = [0u8; 4];
    let mut err = 0.0;
    for c in 0..dims {
        let bits = if c < 3 { mi.cb } else { mi.ab };
        if bits == 0 {
            continue;
        }
        q[c] = quant_channel(v[c], bits, p);
        let d = dequant(q[c], bits, p) as f32 - v[c];
        err += w[c] as f32 * d * d;
    }
    (q, err)
}

/// Quantizes a float endpoint pair, choosing p-bits to minimize endpoint error.
///
/// With `reduce` (entropy reduction), mode 6 uses [`quant_channel_mode6_reduced`] with fixed
/// p-bits: (1, 1) for opaque blocks, so alpha stays exactly 255, else bc7enc's (0, 1).
fn quantize_endpoints(e: &[V4; 2], mi: &ModeInfo, w: &[u32; 4], dims: usize, reduce: bool) -> Quantized {
    let mut out = Quantized::default();
    let is_mode6 = mi.ns == 1 && mi.ib == 4;
    if reduce && ENTROPY.quant_mode6 && is_mode6 {
        let opaque = e[0][3] >= 254.5 && e[1][3] >= 254.5;
        out.p = if opaque { [1, 1] } else { [0, 1] };
        for k in 0..2 {
            for c in 0..4 {
                out.q[k][c] = quant_channel_mode6_reduced(e[k][c], out.p[k]);
            }
        }
    } else if mi.epb == 1 {
        for k in 0..2 {
            let (q0, err0) = quant_endpoint(&e[k], mi, Some(0), w, dims);
            let (q1, err1) = quant_endpoint(&e[k], mi, Some(1), w, dims);
            (out.q[k], out.p[k]) = if err0 <= err1 { (q0, 0) } else { (q1, 1) };
        }
    } else if mi.spb == 1 {
        let mut best = (f32::MAX, Quantized::default());
        for p in 0..2u8 {
            let (a, ea) = quant_endpoint(&e[0], mi, Some(p), w, dims);
            let (b, eb) = quant_endpoint(&e[1], mi, Some(p), w, dims);
            let err = ea + eb;
            if err < best.0 {
                best = (err, Quantized { q: [a, b], p: [p, p] });
            }
        }
        out = best.1;
    } else {
        out.q[0] = quant_endpoint(&e[0], mi, None, w, dims).0;
        out.q[1] = quant_endpoint(&e[1], mi, None, w, dims).0;
    }
    out
}

fn subset_palette(qz: &Quantized, mi: &ModeInfo) -> ([[u8; 4]; 16], usize) {
    let c0 = dequant_endpoint(&qz.q[0], mi, qz.p[0]);
    let c1 = dequant_endpoint(&qz.q[1], mi, qz.p[1]);
    let wt = weights(mi.ib);
    let n = 1usize << mi.ib;
    let mut pal = [[0u8; 4]; 16];
    for k in 0..n {
        for c in 0..4 {
            pal[k][c] = interp(c0[c], c1[c], wt[k]);
        }
    }
    (pal, n)
}

/// Nearest palette entry for each 6-bit interpolation position (0..=64), per index width.
const fn nearest_weight_lut(wt: &[u32]) -> [u8; 65] {
    let mut lut = [0u8; 65];
    let mut s = 0;
    while s <= 64 {
        let mut best = 0;
        let mut k = 1;
        while k < wt.len() {
            if (wt[k] as i32 - s as i32).abs() < (wt[best] as i32 - s as i32).abs() {
                best = k;
            }
            k += 1;
        }
        lut[s] = best as u8;
        s += 1;
    }
    lut
}
const LUT3: [u8; 65] = nearest_weight_lut(&W3);
const LUT4: [u8; 65] = nearest_weight_lut(&W4);

/// A subset palette with a fast nearest-entry search: the texel is projected onto the endpoint
/// line (in the weighted metric), rounded to the nearest interpolation weight, and only that
/// entry and its two neighbours are compared. Palettes of 4 entries are searched exhaustively.
struct LinePal {
    pal: [[u8; 4]; 16],
    n: usize,
    lut: &'static [u8; 65],
    origin: [f32; 4],
    /// Weighted direction scaled so that the projection lands in 0..=64.
    dir: [f32; 4],
}

impl LinePal {
    fn new(pal: [[u8; 4]; 16], n: usize, w: &[u32; 4]) -> Self {
        let lo = pal[0];
        let hi = pal[n - 1];
        let mut d = [0f32; 4];
        let mut dd = 0f32;
        for c in 0..4 {
            d[c] = hi[c] as f32 - lo[c] as f32;
            dd += w[c] as f32 * d[c] * d[c];
        }
        let scale = if dd > 0.0 { 64.0 / dd } else { 0.0 };
        let dir = std::array::from_fn(|c| w[c] as f32 * d[c] * scale);
        Self {
            pal,
            n,
            lut: if n == 16 { &LUT4 } else { &LUT3 },
            origin: std::array::from_fn(|c| lo[c] as f32),
            dir,
        }
    }

    fn from_quantized(qz: &Quantized, mi: &ModeInfo, w: &[u32; 4]) -> Self {
        let (pal, n) = subset_palette(qz, mi);
        Self::new(pal, n, w)
    }

    #[inline]
    fn nearest(&self, t: &[u8; 4], w: &[u32; 4]) -> (u8, u32) {
        if self.n <= 4 {
            return best_index(t, &self.pal, self.n, w);
        }
        let mut s = 0f32;
        for c in 0..4 {
            s += (t[c] as f32 - self.origin[c]) * self.dir[c];
        }
        let g = self.lut[(s + 0.5).clamp(0.0, 64.0) as usize] as usize;
        let lo = g.saturating_sub(1);
        let hi = (g + 1).min(self.n - 1);
        let mut best = (u32::MAX, 0usize);
        for k in lo..=hi {
            let e = texel_sse(t, &self.pal[k], w);
            if e < best.0 {
                best = (e, k);
            }
        }
        // Close endpoints round several neighbouring entries to the same color. Like the
        // exhaustive search, take the lowest index among equal errors: consistent indices in
        // flat regions are what LZ back ends match on.
        while best.1 > 0 && texel_sse(t, &self.pal[best.1 - 1], w) == best.0 {
            best.1 -= 1;
        }
        (best.1 as u8, best.0)
    }

    /// [`Self::nearest`] for all 16 texels of `px` at once, for a 16-entry palette and weights
    /// of at most 128: the same projection (the same operations in the same order), window and
    /// tie-breaking, so the same indices and errors exactly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn nearest16_avx2(&self, px: &Block, w: &[u32; 4]) -> ([u8; 16], [u32; 16]) {
        use std::arch::x86_64::*;
        debug_assert!(self.n == 16 && w.iter().all(|&x| x <= 128));
        let pal: [u32; 16] = self.pal.map(u32::from_le_bytes);
        let tex: [u32; 16] = px.map(u32::from_le_bytes);
        // SAFETY: the loads read 8 u32s inside 16-element arrays; the rest needs AVX2 only.
        unsafe {
            let pa = _mm256_loadu_si256(pal.as_ptr().cast());
            let pb = _mm256_loadu_si256(pal.as_ptr().add(8).cast());
            // Weights as 16-bit pairs per lane: channels 0 and 2 (even bytes), 1 and 3 (odd).
            let we = _mm256_set1_epi32((w[0] | w[2] << 16) as i32);
            let wo = _mm256_set1_epi32((w[1] | w[3] << 16) as i32);
            let (m8, lo8, seven) = (_mm256_set1_epi32(0xff), _mm256_set1_epi16(0xff), _mm256_set1_epi32(7));
            let (zero, fifteen) = (_mm256_setzero_si256(), _mm256_set1_epi32(15));
            let (mut idx, mut err) = ([0u8; 16], [0u32; 16]);
            for h in 0..2 {
                let t = _mm256_loadu_si256(tex.as_ptr().add(8 * h).cast());
                // Weighted squared error to palette entry `k` per lane: |t - p| per byte, split
                // into even and odd bytes as 16-bit values, each squared and weighted by madd.
                let sse = |k: __m256i| {
                    let p = _mm256_blendv_epi8(
                        _mm256_permutevar8x32_epi32(pa, k),
                        _mm256_permutevar8x32_epi32(pb, k),
                        _mm256_cmpgt_epi32(k, seven),
                    );
                    let ad = _mm256_sub_epi8(_mm256_max_epu8(t, p), _mm256_min_epu8(t, p));
                    let (e, o) = (_mm256_and_si256(ad, lo8), _mm256_srli_epi16::<8>(ad));
                    _mm256_add_epi32(
                        _mm256_madd_epi16(e, _mm256_mullo_epi16(e, we)),
                        _mm256_madd_epi16(o, _mm256_mullo_epi16(o, wo)),
                    )
                };
                let mut s = _mm256_setzero_ps();
                for c in 0..4 {
                    let tc = _mm256_and_si256(_mm256_srlv_epi32(t, _mm256_set1_epi32(8 * c as i32)), m8);
                    let d = _mm256_sub_ps(_mm256_cvtepi32_ps(tc), _mm256_set1_ps(self.origin[c]));
                    s = _mm256_add_ps(s, _mm256_mul_ps(d, _mm256_set1_ps(self.dir[c])));
                }
                let r = _mm256_min_ps(
                    _mm256_max_ps(_mm256_add_ps(s, _mm256_set1_ps(0.5)), _mm256_setzero_ps()),
                    _mm256_set1_ps(64.0),
                );
                let mut gi = [0i32; 8];
                _mm256_storeu_si256(gi.as_mut_ptr().cast(), _mm256_cvttps_epi32(r));
                let gi = gi.map(|x| i32::from(self.lut[x as usize]));
                let g = _mm256_loadu_si256(gi.as_ptr().cast());
                let one = _mm256_set1_epi32(1);
                let k0 = _mm256_max_epi32(_mm256_sub_epi32(g, one), zero);
                let k2 = _mm256_min_epi32(_mm256_add_epi32(g, one), fifteen);
                let (mut bk, mut be) = (k0, sse(k0));
                for k in [g, k2] {
                    let e = sse(k);
                    let lt = _mm256_cmpgt_epi32(be, e);
                    be = _mm256_blendv_epi8(be, e, lt);
                    bk = _mm256_blendv_epi8(bk, k, lt);
                }
                // Lanes whose next lower entry ties continue down one at a time below.
                let kp = _mm256_max_epi32(_mm256_sub_epi32(bk, one), zero);
                let tie = _mm256_and_si256(_mm256_cmpeq_epi32(sse(kp), be), _mm256_cmpgt_epi32(bk, zero));
                let ties = _mm256_movemask_ps(_mm256_castsi256_ps(tie));
                let (mut ks, mut es) = ([0i32; 8], [0u32; 8]);
                _mm256_storeu_si256(ks.as_mut_ptr().cast(), bk);
                _mm256_storeu_si256(es.as_mut_ptr().cast(), be);
                for l in 0..8 {
                    let i = 8 * h + l;
                    let mut k = ks[l] as usize;
                    if ties >> l & 1 != 0 {
                        while k > 0 && texel_sse(&px[i], &self.pal[k - 1], w) == es[l] {
                            k -= 1;
                        }
                    }
                    idx[i] = k as u8;
                    err[i] = es[l];
                }
            }
            (idx, err)
        }
    }
}

#[inline]
fn best_index(t: &[u8; 4], pal: &[[u8; 4]; 16], n: usize, w: &[u32; 4]) -> (u8, u32) {
    let mut best = (u32::MAX, 0u8);
    for (k, p) in pal.iter().enumerate().take(n) {
        let e = texel_sse(t, p, w);
        if e < best.0 {
            best = (e, k as u8);
        }
    }
    (best.1, best.0)
}

#[inline]
fn to_v4(t: &[u8; 4]) -> V4 {
    [t[0] as f32, t[1] as f32, t[2] as f32, t[3] as f32]
}

/// Encodes one (mode, partition) combination. Returns the block and its weighted SSE.
fn encode_mode_part(px: &Block, w: &[u32; 4], mode: u8, part: u8, iters: u32, reduce: bool) -> (Bc7Block, u64) {
    let mi = MODES[mode as usize];
    let dims = if mi.ab > 0 { 4 } else { 3 };
    let mut blk = Bc7Block {
        mode,
        partition: part,
        ..Default::default()
    };
    let anc = anchors(mi.ns, part);
    let wt = weights(mi.ib);
    let mut total = 0u64;
    for s in 0..mi.ns as usize {
        let (mem_buf, pts_buf, cnt) = subset_points(px, mi.ns, part, s);
        let members = &mem_buf[..cnt];
        let pts = &pts_buf[..cnt];
        let (mut e0, mut e1) = pca_endpoints(pts, dims);
        if dims == 3 {
            e0[3] = 255.0;
            e1[3] = 255.0;
        }
        let mut best: Option<(Quantized, [u8; 16], u64)> = None;
        for _ in 0..iters {
            let qz = quantize_endpoints(&[e0, e1], &mi, w, dims, reduce);
            let lp = LinePal::from_quantized(&qz, &mi, w);
            let mut idx = [0u8; 16];
            let mut err = 0u64;
            for &i in members {
                let (k, e) = lp.nearest(&px[i], w);
                idx[i] = k;
                err += e as u64;
            }
            match &best {
                Some(b) if err >= b.2 => break, // refit stopped helping
                _ => best = Some((qz, idx, err)),
            }
            if err == 0 {
                break;
            }
            let mut t = [0f32; 16];
            for (k, &i) in members.iter().enumerate() {
                t[k] = wt[idx[i] as usize] as f32 / 64.0;
            }
            match ls_endpoints(pts, &t[..cnt], dims) {
                Some((a, b)) => {
                    e0 = a;
                    e1 = b;
                    if dims == 3 {
                        e0[3] = 255.0;
                        e1[3] = 255.0;
                    }
                }
                None => break,
            }
        }
        let (mut qz, mut idx, err) = best.expect("at least one iteration");
        // The anchor texel's index must have its top bit clear: swap endpoints if not.
        let n = 1u8 << mi.ib;
        if idx[anc[s]] >= n / 2 {
            qz.q.swap(0, 1);
            qz.p.swap(0, 1);
            for &i in members {
                idx[i] = n - 1 - idx[i];
            }
        }
        blk.ep[s] = qz.q;
        blk.pbits[s] = qz.p;
        for &i in members {
            blk.idx[i] = idx[i];
        }
        total += err;
    }
    (blk, total)
}

/// Texel indices and values of one subset, on the stack: (indices, points, count).
#[inline]
fn subset_points(px: &Block, ns: u8, part: u8, s: usize) -> ([usize; 16], [V4; 16], usize) {
    let mut mem = [0usize; 16];
    let mut pts = [[0f32; 4]; 16];
    let mut n = 0;
    for i in 0..16 {
        if subset_of(ns, part, i) == s {
            mem[n] = i;
            pts[n] = to_v4(&px[i]);
            n += 1;
        }
    }
    (mem, pts, n)
}

/// Line fits of the first `count` partitions with `ns` subsets ([`partition::residuals`]).
fn fit_partitions(px: &Block, ns: u8, dims: usize, count: u8) -> partition::Fits {
    partition::residuals(px, ns, dims, count)
}

/// The `k` most promising partitions for a mode with `ib`-bit indices, as (error bound,
/// partition). Ranked by line-fit residual plus the estimated index quantization error
/// `Δw² · λ`; the bound is the residual alone (a true lower bound, see [`partition`]), which
/// [`encode`] uses to skip partitions that cannot win.
fn top_partitions(fits: &partition::Fits, count: u8, ib: u8, k: usize) -> Vec<(f32, u8)> {
    let dw = 1.0 / ((1u32 << ib) - 1) as f32;
    let mut scored: Vec<(f32, f32, u8)> = (0..count)
        .map(|p| {
            let (r, l) = (fits.residual[p as usize], fits.lambda[p as usize]);
            (r + dw * dw * l, r, p)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(k);
    scored.into_iter().map(|(_, bound, p)| (bound, p)).collect()
}

/// Largest per-channel sum of squared deviations (16 × variance) over the given channels.
fn max_channel_energy(px: &Block, dims: usize) -> f32 {
    let pts: [V4; 16] = std::array::from_fn(|i| to_v4(&px[i]));
    let (_, cov) = mean_cov(&pts, dims);
    (0..dims).map(|c| cov[c][c] * 16.0).fold(0.0, f32::max)
}

/// Encodes one block. `effort` 0..=3: 0 = mode 6 only, 3 = every supported mode and partition.
///
/// Effort 1 sends low-energy blocks straight to mode 6 (a bc7f shortcut).
///
/// With [`BlockOpts::reduce_entropy`] (bc7enc's `-e`), mode 6 wins unless another mode's error
/// is below 0.4× (opaque) or 0.6× (alpha) of mode 6's, and mode 6 endpoints use 64 levels with
/// fixed p-bits. The output is less exact but far more repetitive, which
/// is what LZ back ends and the RDO pass feed on.
pub fn encode(px: &Block, w: &[u32; 4], opts: impl Into<BlockOpts>) -> [u8; 16] {
    let opts = opts.into();
    let reduce = opts.reduce_entropy;
    let effort = opts.effort.min(3) as usize;
    let iters = [1, 2, 3, 4][effort];
    let topk = [0usize, 4, 12, 64][effort];
    let alpha = px.iter().any(|t| t[3] != 255);
    // Interpolated palette entries are rounded, so they can sit up to ~0.5 off the fitted
    // line; shrink the bound slightly so pruning never discards a real winner.
    let bound_scale = 0.9f32;

    let m6 = encode_mode_part(px, w, 6, 0, iters, reduce);
    let m6_weight = match (reduce, alpha) {
        (false, _) => 1.0,
        (true, false) => ENTROPY.mode6_weight_opaque,
        (true, true) => ENTROPY.mode6_weight_alpha,
    };
    // (block, error, score): candidates compete on score; only mode 6's is weighted.
    let m6_score = (m6.1 as f64 * m6_weight as f64).round() as u64;
    let mut best = (m6.0, m6.1, m6_score);

    // Effort 1: blocks with almost no variation are already near-exact in mode 6.
    // (Kodak: costs ~0.03 dB, and the mode-6 blocks compress slightly better.)
    const TRIVIAL_ENERGY: f32 = 16.0 * 4.0; // per-channel std dev below ~2
    if effort == 1 && max_channel_energy(px, if alpha { 4 } else { 3 }) < TRIVIAL_ENERGY {
        return pack(&best.0);
    }
    let consider = |mode: u8, parts: &[(f32, u8)], best: &mut (Bc7Block, u64, u64)| {
        for &(bound, p) in parts {
            if best.2 == 0 {
                return;
            }
            if (bound * bound_scale) as u64 >= best.2 {
                continue; // cannot win (the list is ranked by estimate, not by bound)
            }
            let cand = encode_mode_part(px, w, mode, p, iters, reduce);
            if cand.1 < best.2 {
                *best = (cand.0, cand.1, cand.1);
            }
        }
    };

    // Every effort ranks all 64 two-subset partitions (vectorized line fits, cheaper than the
    // old bc7f popcount pre-selection of 16), per mode by residual plus index quantization error.
    if effort >= 1 && best.2 > 0 {
        if alpha {
            let fits = fit_partitions(px, 2, 4, 64);
            consider(7, &top_partitions(&fits, 64, 2, topk), &mut best);
        } else {
            let fits = fit_partitions(px, 2, 3, 64);
            consider(1, &top_partitions(&fits, 64, 3, topk), &mut best);
            consider(3, &top_partitions(&fits, 64, 2, topk), &mut best);
            if effort >= 2 {
                let fits3 = fit_partitions(px, 3, 3, 64);
                consider(2, &top_partitions(&fits3, 64, 2, topk / 2), &mut best);
                consider(0, &top_partitions(&fits3, 16, 3, (topk / 2).min(16)), &mut best);
            }
        }
    }
    // Dual-plane modes 4/5. With alpha they separate it from color; on opaque blocks a rotation
    // gives one color channel its own indices (the constant alpha moves into that color slot).
    // Entropy reduction wants mode 6, so at effort 1 it skips them.
    if effort >= 1 && best.2 > 0 && (alpha && !reduce || effort >= 2) {
        let rotations: &[u8] = match (alpha, effort >= 2) {
            (true, false) => &[0],
            (true, true) => &[0, 1, 2, 3],
            (false, _) => &[1, 2, 3],
        };
        for &rot in rotations {
            for (mode, sel) in [(5u8, 0u8), (4, 0), (4, 1)] {
                if best.2 == 0 {
                    break;
                }
                let cand = encode_dual(px, w, mode, rot, sel, iters);
                if cand.1 < best.2 {
                    best = (cand.0, cand.1, cand.1);
                }
            }
        }
    }
    // Entropy-reduced mode 6 keeps its restricted endpoint levels; everything else is refined.
    let passes = REFINE_PASSES[effort];
    if passes > 0 && best.1 > 0 && !(reduce && best.0.mode == 6) {
        refine_endpoints(&mut best.0, px, w, passes);
    }
    pack(&best.0)
}

/// Fits one plane of a dual-plane block (modes 4/5): the channels in `chans` share one index
/// set of `ib` bits and endpoints of `bits` bits (no p-bits). Returns the quantized endpoints
/// (other channels zero), indices with texel 0's top bit clear, and the weighted error.
fn fit_plane(px: &Block, chans: &[usize], bits: u8, ib: u8, w: &[u32; 4], iters: u32) -> ([[u8; 4]; 2], [u8; 16], u64) {
    let pts: [V4; 16] = std::array::from_fn(|i| {
        let mut p = [0f32; 4];
        for &c in chans {
            p[c] = px[i][c] as f32;
        }
        p
    });
    let wt = weights(ib);
    let n = 1usize << ib;
    let (mut e0, mut e1) = pca_endpoints(&pts, 4);
    let mut best: Option<([[u8; 4]; 2], [u8; 16], u64)> = None;
    let eval = |q: &[[u8; 4]; 2]| -> ([u8; 16], u64) {
        let mut pal = [[0u32; 4]; 16];
        for k in 0..n {
            for &c in chans {
                pal[k][c] = interp(dequant(q[0][c], bits, None), dequant(q[1][c], bits, None), wt[k]) as u32;
            }
        }
        let mut idx = [0u8; 16];
        let mut err = 0u64;
        for i in 0..16 {
            let mut b = (u64::MAX, 0u8);
            for (k, p) in pal.iter().enumerate().take(n) {
                let mut e = 0u64;
                for &c in chans {
                    let d = p[c] as i64 - px[i][c] as i64;
                    e += w[c] as u64 * (d * d) as u64;
                }
                if e < b.0 {
                    b = (e, k as u8);
                }
            }
            idx[i] = b.1;
            err += b.0;
        }
        (idx, err)
    };
    for _ in 0..iters.max(1) {
        let mut q = [[0u8; 4]; 2];
        for &c in chans {
            q[0][c] = quant_channel(e0[c], bits, None);
            q[1][c] = quant_channel(e1[c], bits, None);
        }
        let (idx, err) = eval(&q);
        match &best {
            Some(b) if err >= b.2 => break,
            _ => best = Some((q, idx, err)),
        }
        if err == 0 {
            break;
        }
        let t: [f32; 16] = std::array::from_fn(|i| wt[idx[i] as usize] as f32 / 64.0);
        match ls_endpoints(&pts, &t, 4) {
            Some((a, b)) => (e0, e1) = (a, b),
            None => break,
        }
    }
    let (mut q, mut idx, mut err) = best.expect("at least one iteration");
    // One greedy ±1 pass per channel endpoint.
    let maxq = (1i32 << bits) - 1;
    for &c in chans {
        for k in 0..2 {
            for d in [-1i32, 1] {
                let v = q[k][c] as i32 + d;
                if v < 0 || v > maxq {
                    continue;
                }
                let mut cand = q;
                cand[k][c] = v as u8;
                let (ci, ce) = eval(&cand);
                if ce < err {
                    (q, idx, err) = (cand, ci, ce);
                }
            }
        }
    }
    let n = n as u8;
    if idx[0] >= n / 2 {
        q.swap(0, 1);
        for v in &mut idx {
            *v = n - 1 - *v;
        }
    }
    (q, idx, err)
}

/// Encodes a dual-plane block: mode 4 or 5 with channel `rotation` (0 = none, 1/2/3 swap R/G/B
/// with alpha) and, for mode 4, `idx_sel` (1 = the 3-bit indices drive color).
fn encode_dual(px: &Block, w: &[u32; 4], mode: u8, rotation: u8, idx_sel: u8, iters: u32) -> (Bc7Block, u64) {
    let mi = MODES[mode as usize];
    let mut p = *px;
    let mut wr = *w;
    if rotation > 0 {
        let c = rotation as usize - 1;
        for t in &mut p {
            t.swap(c, 3);
        }
        wr.swap(c, 3);
    }
    // Index widths: color uses the primary set unless mode 4's idx_sel picks the 3-bit one.
    let (cib, aib) = if idx_sel == 0 { (mi.ib, mi.ib2) } else { (mi.ib2, mi.ib) };
    let (cq, cidx, cerr) = fit_plane(&p, &[0, 1, 2], mi.cb, cib, &wr, iters);
    let (aq, aidx, aerr) = fit_plane(&p, &[3], mi.ab, aib, &wr, iters);
    let mut blk = Bc7Block {
        mode,
        rotation,
        idx_sel,
        ..Default::default()
    };
    for e in 0..2 {
        blk.ep[0][e] = [cq[e][0], cq[e][1], cq[e][2], aq[e][3]];
    }
    (blk.idx, blk.idx2) = if idx_sel == 0 { (cidx, aidx) } else { (aidx, cidx) };
    (blk, cerr + aerr)
}

/// Endpoint refinement passes per effort level (see [`refine_endpoints`]).
const REFINE_PASSES: [u32; 4] = [0, 1, 2, 4];

/// Greedy search over each subset's quantized endpoints and p-bits, scored by the real error
/// with indices re-chosen (bc7e-style perturbation). Rounding least-squares endpoints to the
/// format's precision is rarely optimal once the palette is re-matched, so this tries ±1 steps
/// on each channel (one endpoint, both together, or opposite directions) and p-bit flips,
/// keeping any move that lowers the error. Indices and the anchor constraint are redone at the end.
fn refine_endpoints(blk: &mut Bc7Block, px: &Block, w: &[u32; 4], passes: u32) {
    const MOVES: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)];
    let mi = MODES[blk.mode as usize];
    if mi.ib2 > 0 {
        return;
    }
    let dims = if mi.ab > 0 { 4 } else { 3 };
    let anc = anchors(mi.ns, blk.partition);
    for s in 0..mi.ns as usize {
        let (mem_buf, _, cnt) = subset_points(px, mi.ns, blk.partition, s);
        let members = &mem_buf[..cnt];
        let eval = |qz: &Quantized| -> u64 {
            let lp = LinePal::from_quantized(qz, &mi, w);
            members.iter().map(|&i| lp.nearest(&px[i], w).1 as u64).sum()
        };
        let mut cur = Quantized {
            q: blk.ep[s],
            p: blk.pbits[s],
        };
        let mut cur_err = eval(&cur);
        for _ in 0..passes {
            if cur_err == 0 {
                break;
            }
            let mut improved = false;
            for c in 0..dims {
                let maxq = (1i32 << if c < 3 { mi.cb } else { mi.ab }) - 1;
                for (d0, d1) in MOVES {
                    let a = cur.q[0][c] as i32 + d0;
                    let b = cur.q[1][c] as i32 + d1;
                    if a < 0 || b < 0 || a > maxq || b > maxq {
                        continue;
                    }
                    let mut cand = cur;
                    cand.q[0][c] = a as u8;
                    cand.q[1][c] = b as u8;
                    let e = eval(&cand);
                    if e < cur_err {
                        (cur, cur_err, improved) = (cand, e, true);
                    }
                }
            }
            let flips: &[[u8; 2]] = if mi.epb == 1 {
                &[[1, 0], [0, 1], [1, 1]]
            } else if mi.spb == 1 {
                &[[1, 1]]
            } else {
                &[]
            };
            for f in flips {
                let mut cand = cur;
                cand.p = [cur.p[0] ^ f[0], cur.p[1] ^ f[1]];
                let e = eval(&cand);
                if e < cur_err {
                    (cur, cur_err, improved) = (cand, e, true);
                }
            }
            if !improved {
                break;
            }
        }
        let lp = LinePal::from_quantized(&cur, &mi, w);
        let mut idx = [0u8; 16];
        for &i in members {
            idx[i] = lp.nearest(&px[i], w).0;
        }
        let n = lp.n as u8;
        if idx[anc[s]] >= n / 2 {
            cur.q.swap(0, 1);
            cur.p.swap(0, 1);
            for &i in members {
                idx[i] = n - 1 - idx[i];
            }
        }
        blk.ep[s] = cur.q;
        blk.pbits[s] = cur.p;
        for &i in members {
            blk.idx[i] = idx[i];
        }
    }
}

/// Keeps every field except the indices, which are re-chosen; returns the block with its
/// weighted squared error, or None as soon as that error exceeds `bound`. Used by RDO.
pub fn reindex_bounded(
    raw: &[u8],
    px: &Block,
    w: &[u32; 4],
    bound: u64,
    seg: &mut SegBound,
) -> Option<([u8; 16], u64)> {
    let v = u128::from_le_bytes(raw[..16].try_into().ok()?);
    // One copy per mode, as in `unpack_head`; two-index modes (4, 5) are not re-indexed.
    match (v as u8).trailing_zeros() {
        0 => reindex_bounded_mode::<0>(v, px, w, bound, seg),
        1 => reindex_bounded_mode::<1>(v, px, w, bound, seg),
        2 => reindex_bounded_mode::<2>(v, px, w, bound, seg),
        3 => reindex_bounded_mode::<3>(v, px, w, bound, seg),
        6 => reindex_bounded_mode::<6>(v, px, w, bound, seg),
        7 => reindex_bounded_mode::<7>(v, px, w, bound, seg),
        _ => None,
    }
}

#[inline(always)]
fn reindex_bounded_mode<const MODE: u8>(
    v: u128,
    px: &Block,
    w: &[u32; 4],
    bound: u64,
    seg: &mut SegBound,
) -> Option<([u8; 16], u64)> {
    // The indices are all re-chosen, so only the fields before them are read.
    let (mut b, _) = unpack_head_mode::<MODE>(v);
    let mi = const { MODES[MODE as usize] };
    let anc = anchors(mi.ns, b.partition);
    // Dequantized only for the subsets the rejection test reaches.
    let ends = |s: usize| [0, 1].map(|e| dequant_endpoint(&b.ep[s][e], &mi, b.pbits[s][e]));
    if seg.rejects(ends, mi.ns, mi.ib, b.partition, bound) {
        return None;
    }
    let pals: [Option<LinePal>; 3] = std::array::from_fn(|s| {
        (s < mi.ns as usize).then(|| {
            let qz = Quantized {
                q: b.ep[s],
                p: b.pbits[s],
            };
            LinePal::from_quantized(&qz, &mi, w)
        })
    });
    let mut err = 0u64;
    #[cfg(target_arch = "x86_64")]
    if MODE == 6 && crate::simd::level() == crate::simd::Level::Avx2 && w.iter().all(|&x| x <= 128) {
        let lp = pals[0].as_ref().unwrap();
        // SAFETY: AVX2 was detected; the palette has 16 entries and the weights are in range.
        let (idx, errs) = unsafe { lp.nearest16_avx2(px, w) };
        for i in 0..16 {
            let (k, e) = if i == 0 {
                best_index(&px[0], &lp.pal, lp.n / 2, w)
            } else {
                (idx[i], errs[i])
            };
            b.idx[i] = k;
            err += u64::from(e);
            if err > bound {
                return None;
            }
        }
        return Some((pack(&b), err));
    }
    for i in 0..16 {
        let lp = pals[subset_of(mi.ns, b.partition, i)].as_ref().unwrap();
        // Anchors must keep their top index bit clear (no endpoint swap here): search the lower
        // half exhaustively. Everything else uses the projected search.
        let (k, e) = if anc.contains(&i) {
            best_index(&px[i], &lp.pal, lp.n / 2, w)
        } else {
            lp.nearest(&px[i], w)
        };
        b.idx[i] = k;
        err += u64::from(e);
        if err > bound {
            return None;
        }
    }
    Some((pack(&b), err))
}

/// Channel pairs of [`SegBound`]'s scatter entries.
const SCATTER_PAIRS: [(usize, usize); 10] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
];

/// Per-source-block state for [`reindex_bounded`]'s lower bound: the texels scaled by the
/// square roots of the channel weights and, per partition seen, each subset's texel count and
/// mean texel. Reset for each source block.
pub struct SegBound {
    sw: [f32; 4],
    slack: f32,
    px: [[f32; 4]; 16],
    /// `px` by channel, for [`Self::texels_far`].
    pxs: [[f32; 16]; 4],
    /// `slack` plus a margin for `f32` rounding in [`Self::texels_far`].
    slack_f32: f32,
    /// Bit `p` of word `ns - 2`: the means of `ns`-subset partition `p` are cached.
    have: [u64; 2],
    means: [[[f32; 4]; 3]; 128],
    cnt: [[f32; 3]; 128],
    whole: [f32; 4],
    /// The scaled texels' scatter about `whole` (sum of outer products): xx, yy, zz, ww, xy, xz, xw, yz, yw, zw.
    scatter: [f64; 10],
    /// The `bound` that [`Self::thr`] was computed for (`None`: none yet).
    thr_bound: Option<u64>,
    /// One-subset rejection threshold on the squared distance; see [`Self::rejects`].
    thr: f64,
    /// One-subset rejection threshold on the texels' summed squared distance to the line.
    thr_line: f64,
}

impl SegBound {
    pub fn new(px: &Block, w: &[u32; 4]) -> Self {
        let mut s = Self {
            sw: [0.0; 4],
            slack: 0.0,
            px: [[0.0; 4]; 16],
            pxs: [[0.0; 16]; 4],
            slack_f32: 0.0,
            have: [0; 2],
            means: [[[0.0; 4]; 3]; 128],
            cnt: [[0.0; 3]; 128],
            whole: [0.0; 4],
            scatter: [0.0; 10],
            thr_bound: None,
            thr: 0.0,
            thr_line: 0.0,
        };
        s.reset(px, w);
        s
    }

    pub fn reset(&mut self, px: &Block, w: &[u32; 4]) {
        self.sw = w.map(|x| (x as f32).sqrt());
        // Half a step per channel: the interpolation's rounding.
        self.slack = 0.5 * w.iter().map(|&x| x as f32).sum::<f32>().sqrt();
        // Coordinates reach 255 * sw; each texel's distance in `texels_far` errs by well under
        // 1e-3 * sw (a few f32 ulps of them, plus the projection's error times the length).
        self.slack_f32 = self.slack + 4e-3 * self.sw.iter().fold(0f32, |m, &x| m.max(x));
        self.whole = [0.0; 4];
        for i in 0..16 {
            for c in 0..4 {
                self.px[i][c] = px[i][c] as f32 * self.sw[c];
                self.pxs[c][i] = self.px[i][c];
                self.whole[c] += self.px[i][c] / 16.0;
            }
        }
        self.scatter = [0.0; 10];
        for p in &self.px {
            let d: [f64; 4] = std::array::from_fn(|c| f64::from(p[c]) - f64::from(self.whole[c]));
            for (k, (a, b)) in SCATTER_PAIRS.iter().enumerate() {
                self.scatter[k] += d[*a] * d[*b];
            }
        }
        self.have = [0; 2];
        self.thr_bound = None;
    }

    /// Whether a lower bound proves every indexing on these endpoints costs more than `bound`.
    /// Multi-subset blocks use [`Self::lower_bound`]. One-subset blocks (mode 6, most of them)
    /// take an equivalent test without the division and square root: with `d` the distance from
    /// the mean texel to the segment and `s` the slack, `0.999 * 16 * (d - s)^2 > bound` iff
    /// `d^2 > (s + sqrt(bound / 15.984))^2`, a threshold computed once per bound; the projection
    /// is multiplied through by the segment's squared length instead of divided, and summed in
    /// f64. They also take a second bound from every texel rather than the mean: with `D` the
    /// texels' summed squared distance to the endpoints' line (the scatter orthogonal to it, plus
    /// 16 times the mean's), each texel's error is at least `(d_i - s)+^2 >= d_i^2 - 2 s d_i`, so
    /// the block's is at least `D - 8 s sqrt(D)` (Cauchy-Schwarz on the sum of `d_i`), above
    /// `bound / 0.999` iff `D > (4 s + sqrt(16 s^2 + bound / 0.999))^2`.
    #[inline]
    fn rejects(&mut self, ends: impl Fn(usize) -> [[u32; 4]; 2], ns: u8, ib: u8, partition: u8, bound: u64) -> bool {
        if ns != 1 {
            return self.lower_bound(ends, ns, partition, bound);
        }
        if self.thr_bound != Some(bound) {
            self.thr_bound = Some(bound);
            let s = f64::from(self.slack);
            let r = s + (bound as f64 / (0.999 * 16.0)).sqrt();
            self.thr = r * r;
            let r = 4.0 * s + (16.0 * s * s + bound as f64 / 0.999).sqrt();
            self.thr_line = r * r;
        }
        let [a, b] = ends(0);
        let (mut len2, mut dot, mut am2, mut bm2) = (0f64, 0f64, 0f64, 0f64);
        let mut ab = [0f64; 4];
        for c in 0..4 {
            let sw = f64::from(self.sw[c]);
            let ac = f64::from(a[c]) * sw;
            ab[c] = f64::from(b[c]) * sw - ac;
            let am = f64::from(self.whole[c]) - ac;
            len2 += ab[c] * ab[c];
            dot += am * ab[c];
            am2 += am * am;
            bm2 += (am - ab[c]) * (am - ab[c]);
        }
        let mean_far = if dot <= 0.0 || len2 == 0.0 {
            am2 > self.thr
        } else if dot >= len2 {
            bm2 > self.thr
        } else {
            am2 * len2 - dot * dot > self.thr * len2
        };
        if mean_far {
            return true;
        }
        // D * len2 = len2 * tr(S) - ab' S ab + 16 (am2 * len2 - dot^2); a point when len2 = 0.
        let sc = &self.scatter;
        let tr = sc[0] + sc[1] + sc[2] + sc[3];
        if len2 == 0.0 {
            return tr + 16.0 * am2 > self.thr_line;
        }
        let sab = sc[0] * ab[0] * ab[0]
            + sc[1] * ab[1] * ab[1]
            + sc[2] * ab[2] * ab[2]
            + sc[3] * ab[3] * ab[3]
            + 2.0
                * (sc[4] * ab[0] * ab[1]
                    + sc[5] * ab[0] * ab[2]
                    + sc[6] * ab[0] * ab[3]
                    + sc[7] * ab[1] * ab[2]
                    + sc[8] * ab[1] * ab[3]
                    + sc[9] * ab[2] * ab[3]);
        len2 * tr - sab + 16.0 * (am2 * len2 - dot * dot) > self.thr_line * len2 || self.texels_far(a, b, ib, bound)
    }

    /// The one-subset bound texel by texel: palette color `k` lies within `slack` of the point
    /// `W_k / 64` along the endpoints' segment, so a texel's error is at least `(d_i - slack)+^2`
    /// with `d_i` its distance to the nearest such point. That is at least the distance to the
    /// segment combined with the gap along it from the texel's projection `t` to the nearest
    /// `k / (n - 1)` less `1 / 128` (each `W_k / 64` is within that of `k / (n - 1)`). Tighter
    /// than the line bound when texels lie beyond the endpoints, fall between palette colors, or
    /// sit at uneven distances. In `f32` over the texels' lanes; `slack_f32` and the 0.999 cover
    /// its rounding.
    #[inline]
    fn texels_far(&self, a: [u32; 4], b: [u32; 4], ib: u8, bound: u64) -> bool {
        // Adding and subtracting 1.5 * 2^23 rounds to an integer (|x| < 2^22) and vectorizes.
        const ROUND: f32 = 12_582_912.0;
        let steps = ((1u32 << ib) - 1) as f32;
        let ac: [f32; 4] = std::array::from_fn(|c| a[c] as f32 * self.sw[c]);
        let ab: [f32; 4] = std::array::from_fn(|c| b[c] as f32 * self.sw[c] - ac[c]);
        let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2] + ab[3] * ab[3];
        let inv = if len2 > 0.0 { 1.0 / len2 } else { 0.0 };
        let len = len2.sqrt();
        let mut lb = [0f32; 16];
        for (i, l) in lb.iter_mut().enumerate() {
            let am: [f32; 4] = std::array::from_fn(|c| self.pxs[c][i] - ac[c]);
            let t = ((am[0] * ab[0] + am[1] * ab[1] + am[2] * ab[2] + am[3] * ab[3]) * inv).clamp(0.0, 1.0);
            let e: [f32; 4] = std::array::from_fn(|c| am[c] - t * ab[c]);
            let x = t * steps;
            let gap = (((x - ((x + ROUND) - ROUND)).abs() / steps - 1.0 / 128.0).max(0.0)) * len;
            let d = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3] + gap * gap).sqrt() - self.slack_f32;
            let d = d.max(0.0);
            *l = d * d;
        }
        f64::from(lb.iter().sum::<f32>() * 0.999) > bound as f64
    }

    /// Whether a lower bound on the error of any indices into palettes on these endpoints
    /// exceeds `bound`: per subset, the mean of the chosen palette colors lies within half a step
    /// per channel of the endpoints' segment, so the subset's error is at least its texel count
    /// times the squared distance from its mean texel to that segment, less that slack. Stops at
    /// the first subset whose running sum decides it (sums only grow, so the verdict is that of
    /// the full sum).
    fn lower_bound(&mut self, ends: impl Fn(usize) -> [[u32; 4]; 2], ns: u8, partition: u8, bound: u64) -> bool {
        let (means, cnt) = if ns == 1 {
            ([self.whole, [0.0; 4], [0.0; 4]], [16.0, 0.0, 0.0])
        } else {
            let slot = (ns as usize - 2) * 64 + partition as usize;
            let bit = 1u64 << partition;
            if self.have[ns as usize - 2] & bit == 0 {
                self.have[ns as usize - 2] |= bit;
                let mut sum = [[0f32; 4]; 3];
                let mut cnt = [0f32; 3];
                for i in 0..16 {
                    let s = subset_of(ns, partition, i);
                    cnt[s] += 1.0;
                    for c in 0..4 {
                        sum[s][c] += self.px[i][c];
                    }
                }
                for s in 0..3 {
                    if cnt[s] > 0.0 {
                        for c in 0..4 {
                            sum[s][c] /= cnt[s];
                        }
                    }
                }
                self.means[slot] = sum;
                self.cnt[slot] = cnt;
            }
            (self.means[slot], self.cnt[slot])
        };
        let mut lb = 0f32;
        for s in 0..ns as usize {
            if cnt[s] == 0.0 {
                continue;
            }
            let [a, b] = ends(s);
            let mut ab = [0f32; 4];
            let mut am = [0f32; 4];
            for c in 0..4 {
                let ac = a[c] as f32 * self.sw[c];
                ab[c] = b[c] as f32 * self.sw[c] - ac;
                am[c] = means[s][c] - ac;
            }
            let len2: f32 = ab.iter().map(|x| x * x).sum();
            let t = if len2 > 0.0 {
                (am.iter().zip(&ab).map(|(x, y)| x * y).sum::<f32>() / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d2: f32 = (0..4).map(|c| (am[c] - t * ab[c]).powi(2)).sum();
            let d = d2.sqrt() - self.slack;
            if d > 0.0 {
                lb += cnt[s] * d * d;
                // Margin for float rounding.
                if (lb * 0.999) as u64 > bound {
                    return true;
                }
            }
        }
        false
    }
}

/// Least-squares refit of every subset's endpoints, keeping the indices.
fn refit_endpoints(b: &mut Bc7Block, px: &Block, w: &[u32; 4], reduce: bool) {
    let mi = MODES[b.mode as usize];
    let dims = if mi.ab > 0 { 4 } else { 3 };
    let wt = weights(mi.ib);
    for s in 0..mi.ns as usize {
        let (mem, pts, n) = subset_points(px, mi.ns, b.partition, s);
        let mut t = [0f32; 16];
        for k in 0..n {
            t[k] = wt[b.idx[mem[k]] as usize] as f32 / 64.0;
        }
        if let Some((mut a, mut c)) = ls_endpoints(&pts[..n], &t[..n], dims) {
            if dims == 3 {
                a[3] = 255.0;
                c[3] = 255.0;
            }
            let qz = quantize_endpoints(&[a, c], &mi, w, dims, reduce);
            b.ep[s] = qz.q;
            b.pbits[s] = qz.p;
        }
    }
}

/// What [`with_indices`] needs from an unpacked previous block: its indices packed 4 bits each,
/// or None when they cannot be used for `base` (different mode or partition, or a two-index
/// mode). Equal keys give equal results for the same base and pixels.
pub fn index_key(base: &Bc7Block, prev: &Bc7Block) -> Option<u64> {
    if base.mode != prev.mode || base.partition != prev.partition || MODES[base.mode as usize].ib2 > 0 {
        return None;
    }
    Some(prev.idx.iter().fold(0, |k, &i| k << 4 | u64::from(i)))
}

/// Takes `idx` (indices of a previous block with an [`index_key`]) and refits `base`'s
/// endpoints to them, so the index bits become an exact LZ match. Used by RDO. `reduce` keeps
/// the entropy-reduced endpoint quantization (see [`encode`]).
pub fn with_indices(base: &Bc7Block, idx: &[u8; 16], px: &Block, w: &[u32; 4], reduce: bool) -> [u8; 16] {
    let mut b = *base;
    b.idx = *idx;
    refit_endpoints(&mut b, px, w, reduce);
    pack(&b)
}

/// Key for [`with_layout`]: `prev`'s mode, partition, rotation and indices, or None for
/// two-index modes. Equal keys give equal results for the same pixels.
pub fn layout_key(prev: &Bc7Block) -> Option<u64> {
    if MODES[prev.mode as usize].ib2 > 0 {
        return None;
    }
    let idx = prev.idx.iter().fold(0u64, |k, &i| k << 4 | u64::from(i));
    let tag = u64::from(prev.mode) << 16 | u64::from(prev.partition) << 8 | u64::from(prev.rotation);
    Some(idx.rotate_left(17) ^ (tag + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Keeps everything of `prev` but its endpoints (mode, partition, rotation, indices), refit to
/// `px`: header and index bits become an exact LZ match even when `prev`'s mode or partition
/// differs from the block's own encoding. Used by RDO.
pub fn with_layout(prev: &Bc7Block, px: &Block, w: &[u32; 4], reduce: bool) -> [u8; 16] {
    let mut b = *prev;
    refit_endpoints(&mut b, px, w, reduce);
    pack(&b)
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

    /// [`LinePal::nearest16_avx2`] gives [`LinePal::nearest`]'s indices and errors exactly:
    /// random, near and equal mode-6 endpoints (palettes with tied entries), smooth and noisy
    /// texels, under several weight sets up to 128.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn nearest16_avx2_matches_nearest() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mi = MODES[6];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for k in 0..20000 {
            let w = [[1, 1, 1, 1], [1, 1, 1, 0], [2, 4, 1, 3], [128, 1, 77, 128]][k % 4];
            let r = |seed: &mut u64, m: u64| (rng(seed) % m) as u8;
            let a: [u8; 4] = std::array::from_fn(|_| r(&mut seed, 128));
            let spread = [0u64, 1, 3, 128][k / 4 % 4];
            let b: [u8; 4] = std::array::from_fn(|c| {
                let d = if spread == 0 {
                    0
                } else {
                    rng(&mut seed) % (2 * spread + 1)
                } as i32
                    - spread as i32;
                (a[c] as i32 + d).clamp(0, 127) as u8
            });
            let qz = Quantized {
                q: [a, b],
                p: [r(&mut seed, 2), r(&mut seed, 2)],
            };
            let lp = LinePal::from_quantized(&qz, &mi, &w);
            let base: [u8; 4] = std::array::from_fn(|_| rng(&mut seed) as u8);
            let noise = [255u64, 9, 2][k / 16 % 3];
            let px: Block = std::array::from_fn(|_| {
                std::array::from_fn(|c| {
                    (base[c] as i32 + (rng(&mut seed) % (noise + 1)) as i32 - noise as i32 / 2).clamp(0, 255) as u8
                })
            });
            // SAFETY: AVX2 was detected; 16 entries; weights at most 128.
            let (idx, err) = unsafe { lp.nearest16_avx2(&px, &w) };
            for i in 0..16 {
                assert_eq!((idx[i], err[i]), lp.nearest(&px[i], &w), "case {k} texel {i}");
            }
        }
    }

    /// [`multi_head_endpoints`]' known channels are within `half` of the decoded endpoints
    /// (exactly, for alpha), whatever the p-bits past the first 8 bytes.
    #[test]
    fn multi_head_endpoints_bracket_the_decoded_ones() {
        let mut seed = 0x51_7cc1_b727_220au64;
        let mut known_channels = [0usize; 8];
        for k in 0..5000 {
            let mode = [0u8, 1, 2, 3, 7][k % 5];
            let mut raw: [u8; 16] = std::array::from_fn(|_| rng(&mut seed) as u8);
            raw[0] = (raw[0] & !(((2u16 << mode) - 1) as u8)) | 1 << mode;
            let HeadEndpoints {
                ns,
                part,
                mid,
                known,
                half,
            } = multi_head_endpoints(&raw).unwrap();
            let b = unpack(&raw).unwrap();
            let mi = MODES[mode as usize];
            assert_eq!((ns, part, b.mode), (mi.ns, b.partition, mode));
            for s in 0..ns as usize {
                for e in 0..2 {
                    let v = dequant_endpoint(&b.ep[s][e], &mi, b.pbits[s][e]);
                    for c in 0..4 {
                        if known[s][c] {
                            let h = if c == 3 { 0.0 } else { half };
                            assert_eq!((v[c] as f32 - mid[s][e][c]).abs(), h, "mode {mode} s {s} e {e} c {c}");
                            known_channels[mode as usize] += 1;
                        }
                    }
                }
            }
        }
        assert!(known_channels.iter().all(|&n| n == 0 || n > 1000), "{known_channels:?}");
    }

    /// [`reindex_bounded`]'s rejection bounds never reject a block whose re-indexed error is
    /// within the bound: smooth, noisy and flat source blocks against random heads of every
    /// re-indexable mode, under even, alpha-free and uneven channel weights, with bounds at and
    /// just below the exact error.
    #[test]
    fn reindex_rejection_is_a_lower_bound() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut seg = SegBound::new(&[[0; 4]; 16], &[1; 4]);
        let (mut kept, mut rejected) = (0, 0);
        for k in 0..30000 {
            let w = [[1u32, 1, 1, 1], [1, 1, 1, 0], [5, 9, 2, 1]][k % 3];
            let base: [u8; 4] = std::array::from_fn(|_| rng(&mut seed) as u8);
            let step: [i32; 4] = std::array::from_fn(|_| (rng(&mut seed) % 9) as i32 - 4);
            let noise = [0, 2, 8, 40][k % 4];
            let px: Block = std::array::from_fn(|i| {
                std::array::from_fn(|c| {
                    let n = if noise > 0 {
                        (rng(&mut seed) % (2 * noise + 1)) as i32 - noise as i32
                    } else {
                        0
                    };
                    (i32::from(base[c]) + step[c] * i as i32 + n).clamp(0, 255) as u8
                })
            });
            let mut raw: [u8; 16] = std::array::from_fn(|_| rng(&mut seed) as u8);
            let mode = [0u8, 1, 2, 3, 6, 6, 6, 7][k % 8];
            raw[0] = (raw[0] & !(u8::MAX >> (7 - mode))) | (1 << mode);
            seg.reset(&px, &w);
            let Some((_, exact)) = reindex_bounded(&raw, &px, &w, u64::MAX, &mut seg) else {
                continue;
            };
            for bound in [exact, exact + 1, exact * 2] {
                seg.reset(&px, &w);
                let r = reindex_bounded(&raw, &px, &w, bound, &mut seg);
                assert_eq!(r.map(|x| x.1), Some(exact), "mode {mode} bound {bound}");
            }
            let below = exact * 3 / 4;
            seg.reset(&px, &w);
            if mode == 6 {
                let r = reindex_bounded(&raw, &px, &w, below, &mut seg);
                if r.is_none() {
                    rejected += 1;
                } else {
                    kept += 1;
                }
            }
        }
        assert!(rejected > kept, "rejected {rejected} kept {kept}");
    }

    /// Throwaway stage profile over a Kodak image (MIPPLE_PROFILE_IMG). Not a regression test.
    #[test]
    #[ignore = "timing report; run by hand with --nocapture on a quiet machine"]
    fn profile_stages() {
        let path = std::env::var("MIPPLE_PROFILE_IMG").unwrap_or("test-images/kodak/kodim01.png".into());
        let img = image::open(path).unwrap().to_rgba8();
        let mut surf = crate::surface::Surface::new(img.width(), img.height());
        for (x, y, p) in img.enumerate_pixels() {
            surf.set(x, y, p.0);
        }
        let blocks = surf.blocks();
        let w = [1, 1, 1, 1];
        let time = |name: &str, f: &mut dyn FnMut(&Block) -> u64| {
            let t = std::time::Instant::now();
            let mut acc = 0u64;
            for b in &blocks {
                acc = acc.wrapping_add(f(b));
            }
            println!("{name:<30} {:8.1} ms  (chk {acc})", t.elapsed().as_secs_f64() * 1e3);
        };
        let busy = |px: &Block| max_channel_energy(px, 3) >= 64.0;
        time("encode e0", &mut |px| encode(px, &w, 0)[0] as u64);
        time("encode e1", &mut |px| encode(px, &w, 1)[0] as u64);
        time("encode e2", &mut |px| encode(px, &w, 2)[0] as u64);
        time("mode 6, 2 iters", &mut |px| encode_mode_part(px, &w, 6, 0, 2, false).1);
        time("max_channel_energy", &mut |px| max_channel_energy(px, 3) as u64);
        time("rank 2s x64 keep 4 (busy)", &mut |px| {
            if busy(px) {
                top_partitions(&fit_partitions(px, 2, 3, 64), 64, 3, 4).len() as u64
            } else {
                0
            }
        });
        time("modes 1+3, 4 parts (busy)", &mut |px| {
            if !busy(px) {
                return 0;
            }
            let mut e = 0;
            for p in 0..4u8 {
                e += encode_mode_part(px, &w, 1, p, 2, false).1 + encode_mode_part(px, &w, 3, p, 2, false).1;
            }
            e
        });
        time("fit_partitions 2s x64", &mut |px| {
            fit_partitions(px, 2, 3, 64).residual[0] as u64
        });
        time("fit_partitions 3s x64", &mut |px| {
            fit_partitions(px, 3, 3, 64).residual[0] as u64
        });
        time("refine 1 pass (mode 6)", &mut |px| {
            let mut b = encode_mode_part(px, &w, 6, 0, 1, false).0;
            refine_endpoints(&mut b, px, &w, 1);
            b.ep[0][0][0] as u64
        });
        time("dual planes, 3 rot x 3", &mut |px| {
            let mut e = 0;
            for rot in 1..4 {
                for (m, s) in [(5u8, 0u8), (4, 0), (4, 1)] {
                    e += encode_dual(px, &w, m, rot, s, 2).1;
                }
            }
            e
        });
        let n_busy = blocks.iter().filter(|b| busy(b)).count();
        println!("blocks {} (busy {n_busy})", blocks.len());
    }

    #[test]
    fn line_palette_matches_exhaustive_search() {
        let mut seed = 0x51ed_2701_aa55_0f0fu64;
        let (mut worse, mut total, mut diff_idx) = (0u64, 0u64, 0u64);
        for mode in [0u8, 1, 6] {
            let mi = MODES[mode as usize];
            for _ in 0..20000 {
                let mut qz = Quantized::default();
                for k in 0..2 {
                    for c in 0..4 {
                        let bits = if c < 3 { mi.cb } else { mi.ab.max(1) };
                        qz.q[k][c] = (rng(&mut seed) % (1 << bits)) as u8;
                    }
                    qz.p[k] = (rng(&mut seed) & 1) as u8;
                }
                let w = [1, 1, 1, 1];
                let (pal, n) = subset_palette(&qz, &mi);
                let lp = LinePal::new(pal, n, &w);
                let t = [rng(&mut seed) as u8, rng(&mut seed) as u8, rng(&mut seed) as u8, 255];
                // Texels near the line, like real blocks.
                let k = (rng(&mut seed) % n as u64) as usize;
                let near: [u8; 4] = std::array::from_fn(|c| pal[k][c].saturating_add(t[c] % 7));
                let (bi, be) = best_index(&near, &pal, n, &w);
                let (li, le) = lp.nearest(&near, &w);
                total += 1;
                worse += (le > be) as u64;
                diff_idx += (li != bi) as u64;
            }
        }
        // Rounded palette entries are not exactly collinear, so a texel far off the line can
        // very rarely have its nearest entry two steps from the projection.
        assert!(
            worse * 1000 <= total,
            "worse {worse} / {total}, different index {diff_idx}"
        );
    }

    #[test]
    fn pack_unpack_roundtrip_all_modes() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for mode in 0..8u8 {
            for _ in 0..200 {
                let lo = rng(&mut seed) as u128;
                let hi = rng(&mut seed) as u128;
                let mut v = lo | (hi << 64);
                // Force the mode bits.
                v &= !((1u128 << (mode + 1)) - 1);
                v |= 1u128 << mode;
                let raw = v.to_le_bytes();
                let b = unpack(&raw).unwrap();
                assert_eq!(b.mode, mode);
                assert_eq!(pack(&b), raw, "mode {mode}");
            }
        }
    }

    #[test]
    fn solid_block_is_exact_in_mode6() {
        let px = [[90, 140, 200, 255]; 16];
        let raw = encode(&px, &[1, 1, 1, 1], 1);
        let mut out = [[0u8; 4]; 16];
        decode(&raw, &mut out);
        for t in out {
            for c in 0..4 {
                assert!(t[c].abs_diff(px[0][c]) <= 1, "{t:?}");
            }
        }
    }
}
