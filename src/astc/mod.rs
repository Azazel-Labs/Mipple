//! ASTC LDR: block layout, integer sequence encoding, unquantization and the decoder.
//!
//! Written from the Khronos Data Format Specification (ASTC chapter). One 128-bit block covers a
//! footprint of `bw` × `bh` texels (4x4 up to 12x12; mipple writes 4x4, 6x6 and 8x8). A block holds
//! a weight grid of up to 64 weights (one or two planes), one to four partitions, each with a color
//! endpoint mode (CEM) and its endpoint values, all stored with integer sequence encoding (ISE):
//! values in a range of 2^n, 3·2^n or 5·2^n levels, with trits and quints packed five to eight bits
//! and three to seven bits. HDR endpoint modes and HDR void extents decode to the error color, as
//! the LDR profile requires.

pub mod decode;
pub mod encode;
pub mod file;
mod lanes;

pub use decode::{decode_block, decode_rgba};

/// The ISE ranges by index: levels, then the (trits, quints, bits) layout of each.
pub const RANGES: [u16; 21] = [
    2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256,
];

/// For each range index: 0 = plain bits, 1 = one trit, 2 = one quint per value; and the bit count.
pub const LAYOUT: [(u8, u8); 21] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 1),
    (0, 3),
    (2, 1),
    (1, 2),
    (0, 4),
    (2, 2),
    (1, 3),
    (0, 5),
    (2, 3),
    (1, 4),
    (0, 6),
    (2, 4),
    (1, 5),
    (0, 7),
    (2, 5),
    (1, 6),
    (0, 8),
];

/// The highest weight range (32 levels).
pub const WEIGHT_RANGES: usize = 12;
/// The lowest color endpoint range a valid block may use (6 levels).
pub const MIN_COLOR_RANGE: usize = 4;

/// Bits taken by `n` values in range `q`.
pub const fn ise_bits(n: usize, q: usize) -> usize {
    let (kind, bits) = LAYOUT[q];
    let b = n * bits as usize;
    match kind {
        1 => b + (8 * n).div_ceil(5),
        2 => b + (7 * n).div_ceil(3),
        _ => b,
    }
}

/// Decodes the five trits packed in `t`.
const fn bt(x: u32, i: u32) -> u32 {
    (x >> i) & 1
}

const fn trits_of(t: u32) -> [u8; 5] {
    let (c, t4, t3);
    if (t >> 2) & 7 == 7 {
        c = ((t >> 5) & 7) << 2 | (t & 3);
        t4 = 2;
        t3 = 2;
    } else {
        c = t & 31;
        if (t >> 5) & 3 == 3 {
            t4 = 2;
            t3 = bt(t, 7);
        } else {
            t4 = bt(t, 7);
            t3 = (t >> 5) & 3;
        }
    }
    let (t2, t1, t0);
    if c & 3 == 3 {
        t2 = 2;
        t1 = bt(c, 4);
        t0 = bt(c, 3) << 1 | (bt(c, 2) & !bt(c, 3) & 1);
    } else if (c >> 2) & 3 == 3 {
        t2 = 2;
        t1 = 2;
        t0 = c & 3;
    } else {
        t2 = bt(c, 4);
        t1 = (c >> 2) & 3;
        t0 = bt(c, 1) << 1 | (bt(c, 0) & !bt(c, 1) & 1);
    }
    [t0 as u8, t1 as u8, t2 as u8, t3 as u8, t4 as u8]
}

/// Decodes the three quints packed in `q`.
const fn quints_of(q: u32) -> [u8; 3] {
    let (q0, q1, q2);
    if (q >> 1) & 3 == 3 && (q >> 5) & 3 == 0 {
        q2 = bt(q, 0) << 2 | (bt(q, 4) & !bt(q, 0) & 1) << 1 | (bt(q, 3) & !bt(q, 0) & 1);
        q1 = 4;
        q0 = 4;
    } else {
        let c;
        if (q >> 1) & 3 == 3 {
            q2 = 4;
            c = ((q >> 3) & 3) << 3 | (!(q >> 5) & 3) << 1 | bt(q, 0);
        } else {
            q2 = (q >> 5) & 3;
            c = q & 31;
        }
        if c & 7 == 5 {
            q1 = 4;
            q0 = (c >> 3) & 3;
        } else {
            q1 = (c >> 3) & 3;
            q0 = c & 7;
        }
    }
    [q0 as u8, q1 as u8, q2 as u8]
}

const fn trit_tables() -> ([[u8; 5]; 256], [u8; 243]) {
    let mut dec = [[0u8; 5]; 256];
    let mut enc = [0u8; 243];
    let mut seen = [false; 243];
    let mut t = 0;
    while t < 256 {
        let v = trits_of(t as u32);
        dec[t] = v;
        let k = v[0] as usize + 3 * v[1] as usize + 9 * v[2] as usize + 27 * v[3] as usize + 81 * v[4] as usize;
        if !seen[k] {
            seen[k] = true;
            enc[k] = t as u8;
        }
        t += 1;
    }
    (dec, enc)
}

const fn quint_tables() -> ([[u8; 3]; 128], [u8; 125]) {
    let mut dec = [[0u8; 3]; 128];
    let mut enc = [0u8; 125];
    let mut seen = [false; 125];
    let mut q = 0;
    while q < 128 {
        let v = quints_of(q as u32);
        dec[q] = v;
        let k = v[0] as usize + 5 * v[1] as usize + 25 * v[2] as usize;
        if !seen[k] {
            seen[k] = true;
            enc[k] = q as u8;
        }
        q += 1;
    }
    (dec, enc)
}

const TRITS: ([[u8; 5]; 256], [u8; 243]) = trit_tables();
const QUINTS: ([[u8; 3]; 128], [u8; 125]) = quint_tables();

/// Reads bits from a 128-bit block, as zeros past `end`.
struct Reader {
    v: u128,
    pos: u32,
    end: u32,
}

impl Reader {
    fn take(&mut self, n: u32) -> u32 {
        let mut out = 0;
        for i in 0..n {
            let p = self.pos + i;
            if p < self.end {
                out |= (((self.v >> p) & 1) as u32) << i;
            }
        }
        self.pos += n;
        out
    }
}

/// Decodes `out.len()` values in range `q` from bits `start..` of `v`.
pub fn ise_decode(v: u128, start: u32, q: usize, out: &mut [u8]) {
    let n = out.len();
    let (kind, bits) = LAYOUT[q];
    let bits = bits as u32;
    let mut r = Reader {
        v,
        pos: start,
        end: start + ise_bits(n, q) as u32,
    };
    match kind {
        0 => {
            for o in out.iter_mut() {
                *o = r.take(bits) as u8;
            }
        }
        1 => {
            for g in 0..n.div_ceil(5) {
                let mut m = [0u32; 5];
                let mut t = 0;
                for (j, (shift, w)) in [(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)].into_iter().enumerate() {
                    m[j] = r.take(bits);
                    t |= r.take(w) << shift;
                }
                let tr = TRITS.0[t as usize];
                for j in 0..5 {
                    if let Some(o) = out.get_mut(5 * g + j) {
                        *o = ((tr[j] as u32) << bits | m[j]) as u8;
                    }
                }
            }
        }
        _ => {
            for g in 0..n.div_ceil(3) {
                let mut m = [0u32; 3];
                let mut t = 0;
                for (j, (shift, w)) in [(0, 3), (3, 2), (5, 2)].into_iter().enumerate() {
                    m[j] = r.take(bits);
                    t |= r.take(w) << shift;
                }
                let qu = QUINTS.0[t as usize];
                for j in 0..3 {
                    if let Some(o) = out.get_mut(3 * g + j) {
                        *o = ((qu[j] as u32) << bits | m[j]) as u8;
                    }
                }
            }
        }
    }
}

/// Encodes `vals` (each below `RANGES[q]`) in range `q` into bits `start..` of `v`.
pub fn ise_encode(v: &mut u128, start: u32, q: usize, vals: &[u8]) {
    let n = vals.len();
    let (kind, bits) = LAYOUT[q];
    let bits = bits as u32;
    let len = ise_bits(n, q) as u32;
    // Fields go into `acc` from bit 0, which is cut to the sequence's length at the end (the last
    // trit or quint block is truncated).
    let mut acc = 0u128;
    let mut pos = 0u32;
    let mut put = |x: u32, w: u32, _: &mut u128| {
        if pos < 128 {
            acc |= ((x & ((1u32 << w) - 1)) as u128) << pos;
        }
        pos += w;
    };
    let mask = (1u32 << bits) - 1;
    match kind {
        0 => {
            for &x in vals {
                put(x as u32, bits, v);
            }
        }
        1 => {
            for g in vals.chunks(5) {
                let mut k = 0;
                let mut mul = 1;
                for &x in g {
                    k += (x as usize >> bits) * mul;
                    mul *= 3;
                }
                let t = TRITS.1[k] as u32;
                for (j, (shift, w)) in [(0, 2), (2, 2), (4, 1), (5, 2), (7, 1)].into_iter().enumerate() {
                    put(g.get(j).map_or(0, |&x| x as u32 & mask), bits, v);
                    put(t >> shift, w, v);
                }
            }
        }
        _ => {
            for g in vals.chunks(3) {
                let mut k = 0;
                let mut mul = 1;
                for &x in g {
                    k += (x as usize >> bits) * mul;
                    mul *= 5;
                }
                let t = QUINTS.1[k] as u32;
                for (j, (shift, w)) in [(0, 3), (3, 2), (5, 2)].into_iter().enumerate() {
                    put(g.get(j).map_or(0, |&x| x as u32 & mask), bits, v);
                    put(t >> shift, w, v);
                }
            }
        }
    }
    if len > 0 {
        *v |= (if len >= 128 { acc } else { acc & ((1u128 << len) - 1) }) << start;
    }
}

/// The color endpoint value (0..=255) of each ISE value in range `q` (`q >= MIN_COLOR_RANGE`).
pub const fn color_unquant(q: usize) -> [u8; 256] {
    let mut out = [0u8; 256];
    let (kind, bits) = LAYOUT[q];
    let bits = bits as u32;
    let mut i = 0;
    while i < RANGES[q] as usize {
        let v = i as u32;
        out[i] = if kind == 0 {
            // Bit replication to eight bits.
            let mut r = 0u32;
            let mut have = 0;
            while have < 8 {
                r = r << bits | v;
                have += bits;
            }
            (r >> (have - 8)) as u8
        } else {
            let m = v & ((1 << bits) - 1);
            let d = v >> bits;
            let a = if m & 1 != 0 { 0x1ff } else { 0 };
            let (b, c) = if kind == 1 {
                match bits {
                    1 => (0, 204),
                    2 => {
                        let b = (m >> 1) & 1;
                        (b << 8 | b << 4 | b << 2 | b << 1, 93)
                    }
                    3 => {
                        let cb = (m >> 1) & 3;
                        (cb << 7 | cb << 2 | cb, 44)
                    }
                    4 => {
                        let dcb = (m >> 1) & 7;
                        (dcb << 6 | dcb, 22)
                    }
                    5 => {
                        let e = (m >> 1) & 15;
                        (e << 5 | e >> 2, 11)
                    }
                    _ => {
                        let f = (m >> 1) & 31;
                        (f << 4 | f >> 4, 5)
                    }
                }
            } else {
                match bits {
                    1 => (0, 113),
                    2 => {
                        let b = (m >> 1) & 1;
                        (b << 8 | b << 3 | b << 2, 54)
                    }
                    3 => {
                        let cb = (m >> 1) & 3;
                        (cb << 7 | cb << 1 | cb >> 1, 26)
                    }
                    4 => {
                        let dcb = (m >> 1) & 7;
                        (dcb << 6 | dcb >> 1, 13)
                    }
                    _ => {
                        let e = (m >> 1) & 15;
                        (e << 5 | e >> 3, 6)
                    }
                }
            };
            let t = (d * c + b) ^ a;
            ((a & 0x80) | (t >> 2)) as u8
        };
        i += 1;
    }
    out
}

/// The weight (0..=64) of each ISE value in range `q` (`q < WEIGHT_RANGES`).
pub const fn weight_unquant(q: usize) -> [u8; 32] {
    let mut out = [0u8; 32];
    let (kind, bits) = LAYOUT[q];
    let bits = bits as u32;
    let mut i = 0;
    while i < RANGES[q] as usize {
        let v = i as u32;
        let t = if kind == 0 {
            let mut r = 0u32;
            let mut have = 0;
            while have < 6 {
                r = r << bits | v;
                have += bits;
            }
            r >> (have - 6)
        } else if bits == 0 {
            if kind == 1 {
                [0, 32, 63][v as usize]
            } else {
                [0, 16, 32, 47, 63][v as usize]
            }
        } else {
            let m = v & ((1 << bits) - 1);
            let d = v >> bits;
            let a = if m & 1 != 0 { 0x7f } else { 0 };
            let (b, c) = if kind == 1 {
                match bits {
                    1 => (0, 50),
                    2 => {
                        let b = (m >> 1) & 1;
                        (b << 6 | b << 2 | b, 23)
                    }
                    _ => {
                        let cb = (m >> 1) & 3;
                        (cb << 5 | cb, 11)
                    }
                }
            } else {
                match bits {
                    1 => (0, 28),
                    _ => {
                        let b = (m >> 1) & 1;
                        (b << 6 | b << 1, 13)
                    }
                }
            };
            let t = (d * c + b) ^ a;
            (a & 0x20) | (t >> 2)
        };
        out[i] = if t > 32 { t + 1 } else { t } as u8;
        i += 1;
    }
    out
}

/// Color endpoint unquantization tables for every range (unused below [`MIN_COLOR_RANGE`]).
pub static COLOR_UNQUANT: [[u8; 256]; 21] = {
    let mut t = [[0u8; 256]; 21];
    let mut q = MIN_COLOR_RANGE;
    while q < 21 {
        t[q] = color_unquant(q);
        q += 1;
    }
    t
};

/// Weight unquantization tables for every weight range.
pub static WEIGHT_UNQUANT: [[u8; 32]; WEIGHT_RANGES] = {
    let mut t = [[0u8; 32]; WEIGHT_RANGES];
    let mut q = 0;
    while q < WEIGHT_RANGES {
        t[q] = weight_unquant(q);
        q += 1;
    }
    t
};

/// A decoded block mode: weight grid size, weight range, dual plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockMode {
    pub gx: u8,
    pub gy: u8,
    pub quant: u8,
    pub dual: bool,
}

impl BlockMode {
    /// The number of stored weights (both planes).
    pub fn weights(self) -> usize {
        self.gx as usize * self.gy as usize * (1 + self.dual as usize)
    }

    /// Bits the weights take.
    pub fn weight_bits(self) -> usize {
        ise_bits(self.weights(), self.quant as usize)
    }
}

/// Decodes the 11-bit 2D block mode field, or `None` for reserved encodings and grids no valid
/// block can hold (more than 64 weights, or weight bits outside 24..=96).
pub fn block_mode(m: u32) -> Option<BlockMode> {
    let mut base = (m >> 4) & 1;
    let mut h = (m >> 9) & 1;
    let mut d = (m >> 10) & 1;
    let a = (m >> 5) & 3;
    let (gx, gy);
    if m & 3 != 0 {
        base |= (m & 3) << 1;
        let b = (m >> 7) & 3;
        match (m >> 2) & 3 {
            0 => (gx, gy) = (b + 4, a + 2),
            1 => (gx, gy) = (b + 8, a + 2),
            2 => (gx, gy) = (a + 2, b + 8),
            _ => {
                let b = b & 1;
                if m & 0x100 != 0 {
                    (gx, gy) = (b + 2, a + 2);
                } else {
                    (gx, gy) = (a + 2, b + 6);
                }
            }
        }
    } else {
        base |= ((m >> 2) & 3) << 1;
        if (m >> 2) & 3 == 0 {
            return None;
        }
        let b = (m >> 9) & 3;
        match (m >> 7) & 3 {
            0 => (gx, gy) = (12, a + 2),
            1 => (gx, gy) = (a + 2, 12),
            2 => {
                (gx, gy) = (a + 6, b + 6);
                d = 0;
                h = 0;
            }
            _ => match (m >> 5) & 3 {
                0 => (gx, gy) = (6, 10),
                1 => (gx, gy) = (10, 6),
                _ => return None,
            },
        }
    }
    let bm = BlockMode {
        gx: gx as u8,
        gy: gy as u8,
        quant: (base - 2 + 6 * h) as u8,
        dual: d != 0,
    };
    let bits = bm.weight_bits();
    (bm.weights() <= 64 && (24..=96).contains(&bits)).then_some(bm)
}

/// The first block mode field value that decodes to `bm`, if any.
pub fn block_mode_bits(bm: BlockMode) -> Option<u32> {
    // Indexed by grid size, weight range and dual plane; u16::MAX where no field value decodes to
    // the mode.
    fn key(bm: BlockMode) -> Option<usize> {
        (bm.gx < 16 && bm.gy < 16 && bm.quant < 12)
            .then(|| ((bm.gx as usize * 16 + bm.gy as usize) * 12 + bm.quant as usize) * 2 + bm.dual as usize)
    }
    static TABLE: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = vec![u16::MAX; 16 * 16 * 12 * 2];
        for m in (0..2048u16).rev() {
            if let Some(k) = block_mode(m as u32).and_then(key) {
                t[k] = m;
            }
        }
        t
    });
    let m = t[key(bm)?];
    (m != u16::MAX).then_some(m as u32)
}

fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// The partition (0..count) of texel (`x`, `y`) for partition pattern `seed` (10 bits), in a block
/// of `texels` texels.
pub fn partition_of(seed: u32, x: u32, y: u32, count: u32, texels: u32) -> u8 {
    if count <= 1 {
        return 0;
    }
    let (x, y, z) = if texels < 31 { (x << 1, y << 1, 0) } else { (x, y, 0) };
    let seed = seed + (count - 1) * 1024;
    let r = hash52(seed);
    let mut s = [0u32; 12];
    for (i, v) in s.iter_mut().take(8).enumerate() {
        *v = (r >> (4 * i)) & 15;
    }
    s[8] = (r >> 18) & 15;
    s[9] = (r >> 22) & 15;
    s[10] = (r >> 26) & 15;
    s[11] = r.rotate_left(2) & 15;
    for v in &mut s {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (if seed & 2 != 0 { 4 } else { 5 }, if count == 3 { 6 } else { 5 })
    } else {
        (if count == 3 { 6 } else { 5 }, if seed & 2 != 0 { 4 } else { 5 })
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, v) in s.iter_mut().enumerate() {
        *v >>= if i >= 8 {
            sh3
        } else if i % 2 == 0 {
            sh1
        } else {
            sh2
        };
    }
    let a = (s[0] * x + s[1] * y + s[10] * z + (r >> 14)) & 0x3f;
    let b = (s[2] * x + s[3] * y + s[11] * z + (r >> 10)) & 0x3f;
    let mut c = (s[4] * x + s[5] * y + s[8] * z + (r >> 6)) & 0x3f;
    let mut d = (s[6] * x + s[7] * y + s[9] * z + (r >> 2)) & 0x3f;
    if count < 4 {
        d = 0;
    }
    if count < 3 {
        c = 0;
    }
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

/// The number of endpoint values color endpoint mode `cem` takes.
pub const fn cem_values(cem: u8) -> usize {
    2 * ((cem as usize >> 2) + 1)
}

/// Whether `cem` is an HDR mode (an error in the LDR profile).
pub const fn cem_is_hdr(cem: u8) -> bool {
    matches!(cem, 2 | 3 | 7 | 11 | 14 | 15)
}

/// A block in symbolic form: what the bits say, before decoding to texels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbolic {
    pub mode: BlockMode,
    /// Partition count, 1..=4.
    pub partitions: u8,
    /// Partition pattern (10 bits); 0 for one partition.
    pub seed: u16,
    /// Color endpoint mode of each partition.
    pub cems: [u8; 4],
    /// The component (0..4) that takes the second weight plane, for dual-plane modes.
    pub ccs: u8,
    /// Endpoint values as ISE values in range `color_quant`, partition after partition.
    pub colors: [u8; 18],
    /// The endpoint range, implied by the other fields (see [`color_range`]).
    pub color_quant: u8,
    /// Weights as ISE values in range `mode.quant`, planes interleaved per grid point.
    pub weights: [u8; 64],
}

/// Bits taken by the CEM field above the weights for these partitions' modes: 0 when all modes
/// are the same.
fn extra_cem_bits(partitions: u8, cems: &[u8]) -> usize {
    if partitions == 1 || cems.iter().all(|&c| c == cems[0]) {
        0
    } else {
        3 * partitions as usize - 4
    }
}

/// The endpoint range a block with this mode, partition count and CEMs uses: the largest that
/// fits the bits left, or `None` if the block is invalid.
pub fn color_range(mode: BlockMode, partitions: u8, cems: &[u8]) -> Option<u8> {
    color_range_with(
        mode,
        partitions,
        cems,
        extra_cem_bits(partitions, &cems[..partitions as usize]),
    )
}

/// [`color_range`] for a block whose mode field takes `extra` bits below the weights. A block may
/// spell equal CEMs the long way (a nonzero class selector), as astcenc does, and still pay for the
/// extra bits.
fn color_range_with(mode: BlockMode, partitions: u8, cems: &[u8], extra: usize) -> Option<u8> {
    if partitions == 4 && mode.dual {
        return None;
    }
    let values: usize = cems[..partitions as usize].iter().map(|&c| cem_values(c)).sum();
    if values > 18 {
        return None;
    }
    let config = if partitions == 1 { 17 } else { 29 };
    let used = config + mode.weight_bits() + extra + 2 * mode.dual as usize;
    let avail = 128usize.checked_sub(used)?;
    (MIN_COLOR_RANGE..21)
        .rev()
        .find(|&q| ise_bits(values, q) <= avail)
        .map(|q| q as u8)
}

impl Symbolic {
    /// Reads a block's fields, or `None` for void-extent, reserved and invalid blocks.
    pub fn read(v: u128, bw: u32, bh: u32) -> Option<Self> {
        let mode = block_mode((v & 0x7ff) as u32)?;
        if mode.gx as u32 > bw || mode.gy as u32 > bh {
            return None;
        }
        let partitions = ((v >> 11) & 3) as u8 + 1;
        let weight_bits = mode.weight_bits();
        let mut below = 128 - weight_bits;
        let mut cems = [0u8; 4];
        let mut extra = 0;
        let seed;
        if partitions == 1 {
            seed = 0;
            cems[0] = ((v >> 13) & 15) as u8;
        } else {
            seed = ((v >> 13) & 0x3ff) as u16;
            let mut ty = ((v >> 23) & 63) as u32;
            if ty & 3 == 0 {
                cems = [(ty >> 2) as u8 & 15; 4];
            } else {
                extra = 3 * partitions as usize - 4;
                below -= extra;
                ty |= (((v >> below) & ((1u128 << extra) - 1)) as u32) << 6;
                let base = (ty & 3) - 1;
                let p = partitions as u32;
                for i in 0..p {
                    let c = (ty >> (2 + i)) & 1;
                    let m = (ty >> (2 + p + 2 * i)) & 3;
                    cems[i as usize] = (((c + base) << 2) | m) as u8;
                }
            }
        }
        let color_quant = color_range_with(mode, partitions, &cems, extra)?;
        let ccs = if mode.dual { ((v >> (below - 2)) & 3) as u8 } else { 0 };
        let mut colors = [0u8; 18];
        let n: usize = cems[..partitions as usize].iter().map(|&c| cem_values(c)).sum();
        ise_decode(
            v,
            if partitions == 1 { 17 } else { 29 },
            color_quant as usize,
            &mut colors[..n],
        );
        let mut weights = [0u8; 64];
        ise_decode(v.reverse_bits(), 0, mode.quant as usize, &mut weights[..mode.weights()]);
        Some(Symbolic {
            mode,
            partitions,
            seed,
            cems,
            ccs,
            colors,
            color_quant,
            weights,
        })
    }

    /// Packs the fields into a block, or `None` if they don't make a valid one. `color_quant` must
    /// be [`color_range`] of the other fields.
    pub fn write(&self) -> Option<u128> {
        let mode_bits = block_mode_bits(self.mode)?;
        let p = self.partitions as usize;
        let cems = &self.cems[..p];
        if color_range(self.mode, self.partitions, cems)? != self.color_quant {
            return None;
        }
        let mut v = mode_bits as u128 | ((p as u128 - 1) << 11);
        let mut below = 128 - self.mode.weight_bits();
        if p == 1 {
            v |= (cems[0] as u128) << 13;
        } else {
            v |= (self.seed as u128 & 0x3ff) << 13;
            if extra_cem_bits(self.partitions, cems) == 0 {
                v |= (cems[0] as u128) << 25;
            } else {
                // The base class is at most 2; four class-3 modes are base 2, each one class up.
                let base = cems.iter().map(|&c| c >> 2).min()?.min(2);
                if cems.iter().any(|&c| c >> 2 > base + 1) {
                    return None;
                }
                let mut ty = base as u32 + 1;
                for (i, &c) in cems.iter().enumerate() {
                    ty |= (((c >> 2) - base) as u32) << (2 + i);
                    ty |= ((c & 3) as u32) << (2 + p + 2 * i);
                }
                v |= ((ty & 63) as u128) << 23;
                let extra = 3 * p - 4;
                below -= extra;
                v |= ((ty >> 6) as u128) << below;
            }
        }
        if self.mode.dual {
            v |= ((self.ccs & 3) as u128) << (below - 2);
        }
        let n: usize = cems.iter().map(|&c| cem_values(c)).sum();
        ise_encode(
            &mut v,
            if p == 1 { 17 } else { 29 },
            self.color_quant as usize,
            &self.colors[..n],
        );
        let mut w = 0u128;
        ise_encode(
            &mut w,
            0,
            self.mode.quant as usize,
            &self.weights[..self.mode.weights()],
        );
        Some(v | w.reverse_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ise_bit_counts_match_the_spec() {
        // A few hand-checked counts: 16 weights at 5 trit-levels... and so on.
        assert_eq!(ise_bits(16, 1), 26); // 3 levels: ceil(128/5)
        assert_eq!(ise_bits(16, 3), 38); // 5 levels: ceil(112/3)
        assert_eq!(ise_bits(16, 11), 80); // 32 levels: 5 bits each
        assert_eq!(ise_bits(6, 20), 48);
        assert_eq!(ise_bits(8, 19), 8 * 6 + 13);
    }

    #[test]
    fn ise_round_trips_every_range_and_length() {
        let mut s = 0x1234_5678u32;
        for q in 0..21 {
            for n in 1..=64usize {
                if ise_bits(n, q) > 128 {
                    continue;
                }
                let vals: Vec<u8> = (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((s >> 8) % RANGES[q] as u32) as u8
                    })
                    .collect();
                let mut v = 0u128;
                ise_encode(&mut v, 0, q, &vals);
                assert!(
                    ise_bits(n, q) == 128 || v >> ise_bits(n, q) == 0,
                    "q {q} n {n} overflow"
                );
                let mut out = vec![0u8; n];
                ise_decode(v, 0, q, &mut out);
                assert_eq!(out, vals, "q {q} n {n}");
            }
        }
    }

    #[test]
    fn trit_and_quint_codes_cover_every_tuple() {
        for k in 0..243usize {
            let t = TRITS.1[k] as usize;
            let v = TRITS.0[t];
            assert_eq!(
                v[0] as usize + 3 * v[1] as usize + 9 * v[2] as usize + 27 * v[3] as usize + 81 * v[4] as usize,
                k
            );
        }
        for k in 0..125usize {
            let q = QUINTS.1[k] as usize;
            let v = QUINTS.0[q];
            assert_eq!(v[0] as usize + 5 * v[1] as usize + 25 * v[2] as usize, k);
        }
    }

    #[test]
    fn unquantized_levels_are_symmetric_and_distinct() {
        for q in MIN_COLOR_RANGE..21 {
            let n = RANGES[q] as usize;
            let mut v: Vec<u8> = COLOR_UNQUANT[q][..n].to_vec();
            v.sort_unstable();
            v.dedup();
            assert_eq!(v.len(), n, "color range {q}");
            assert_eq!((v[0], v[n - 1]), (0, 255));
            for i in 0..n {
                assert_eq!(v[i] as u32 + v[n - 1 - i] as u32, 255, "color range {q}");
            }
        }
        for q in 0..WEIGHT_RANGES {
            let n = RANGES[q] as usize;
            let mut v: Vec<u8> = WEIGHT_UNQUANT[q][..n].to_vec();
            v.sort_unstable();
            v.dedup();
            assert_eq!(v.len(), n, "weight range {q}");
            for i in 0..n {
                assert_eq!(v[i] as u32 + v[n - 1 - i] as u32, 64, "weight range {q}");
            }
        }
        // Published weight levels for 6, 12 and 20 levels.
        let sorted = |q: usize| {
            let mut v = WEIGHT_UNQUANT[q][..RANGES[q] as usize].to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(4), [0, 12, 25, 39, 52, 64]);
        assert_eq!(sorted(7), [0, 5, 11, 17, 23, 28, 36, 41, 47, 53, 59, 64]);
        assert_eq!(
            sorted(9),
            [
                0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 35, 38, 41, 45, 48, 51, 55, 58, 61, 64
            ]
        );
    }

    #[test]
    fn block_modes_cover_the_usual_grids() {
        for (gx, gy) in [(4, 4), (6, 6), (8, 5), (5, 8), (6, 5), (2, 2)] {
            for quant in 0..WEIGHT_RANGES as u8 {
                for dual in [false, true] {
                    let bm = BlockMode { gx, gy, quant, dual };
                    let ok = bm.weights() <= 64 && (24..=96).contains(&bm.weight_bits());
                    if let Some(m) = block_mode_bits(bm) {
                        assert!(ok);
                        assert_eq!(block_mode(m), Some(bm));
                    }
                }
            }
        }
        // Every 4x4 grid that fits must be encodable.
        for quant in 0..WEIGHT_RANGES as u8 {
            let bm = BlockMode {
                gx: 4,
                gy: 4,
                quant,
                dual: false,
            };
            if (24..=96).contains(&bm.weight_bits()) {
                assert!(block_mode_bits(bm).is_some(), "4x4 q{quant}");
            }
        }
    }

    #[test]
    fn partitions_are_balanced_enough() {
        // Every two-partition pattern in an 8x8 block uses at most two labels, and most use both (863 of
        // the 1024).
        let mut both = 0;
        for seed in 0..1024 {
            let mut seen = [false; 4];
            for y in 0..8 {
                for x in 0..8 {
                    seen[partition_of(seed, x, y, 2, 64) as usize] = true;
                }
            }
            assert!(!seen[2] && !seen[3]);
            both += (seen[0] && seen[1]) as u32;
        }
        assert!(both > 800, "{both}");
    }

    #[test]
    fn symbolic_blocks_round_trip() {
        let mut s = 99u32;
        let mut rnd = |n: u32| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (s >> 8) % n
        };
        let mut done = 0;
        while done < 2000 {
            let m = rnd(2048);
            let Some(mode) = block_mode(m) else { continue };
            let (bw, bh) = (8, 8);
            if mode.gx as u32 > bw || mode.gy as u32 > bh {
                continue;
            }
            let partitions = rnd(4) as u8 + 1;
            let mut cems = [0u8; 4];
            let base = rnd(3) as u8;
            for c in cems.iter_mut().take(partitions as usize) {
                *c = ((base + rnd(2) as u8) << 2) | rnd(4) as u8;
            }
            let Some(color_quant) = color_range(mode, partitions, &cems) else {
                continue;
            };
            let mut colors = [0u8; 18];
            let n: usize = cems[..partitions as usize].iter().map(|&c| cem_values(c)).sum();
            for c in colors.iter_mut().take(n) {
                *c = rnd(RANGES[color_quant as usize] as u32) as u8;
            }
            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(mode.weights()) {
                *w = rnd(RANGES[mode.quant as usize] as u32) as u8;
            }
            let sym = Symbolic {
                mode,
                partitions,
                seed: if partitions == 1 { 0 } else { rnd(1024) as u16 },
                cems,
                ccs: if mode.dual { rnd(4) as u8 } else { 0 },
                colors,
                color_quant,
                weights,
            };
            let v = sym.write().expect("valid");
            let back = Symbolic::read(v, bw, bh).expect("readable");
            // CEMs past the partition count are not stored.
            let mut want = sym.clone();
            if partitions > 1 && want.cems[..partitions as usize].iter().all(|&c| c == want.cems[0]) {
                want.cems = [want.cems[0]; 4];
            }
            let mut got = back;
            for i in partitions as usize..4 {
                got.cems[i] = want.cems[i];
            }
            assert_eq!(got, want, "mode {m}");
            done += 1;
        }
    }
}
