//! EAC single-channel block codec (8 bytes per 4x4 block): the 8-bit alpha half of ETC2 RGBA8
//! and the 11-bit EAC R11 / RG11 formats (unsigned).
//!
//! A block is a base value, a multiplier and one of 16 modifier tables; each texel picks one of
//! the table's eight entries (3 bits, column-major). 8-bit: `clamp(base + entry × mult)`; 11-bit:
//! `clamp(base × 8 + 4 + entry × mult × 8)`, with multiplier 0 meaning `entry` unscaled.
//!
//! The encoder starts each table at the base and multiplier that span the block's range, scores
//! a window around them, then alternates least-squares refits of (base, multiplier) to the
//! chosen entries with rescoring their neighbors. 11-bit blocks are scored against the 8-bit
//! source scaled to 11 bits, with two fractional bits.

use crate::surface::{Block, Surface};

/// EAC modifier tables.
pub const TABLES: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

/// Raster index of column-major texel `p`.
#[inline(always)]
fn raster(p: usize) -> usize {
    (p & 3) * 4 + (p >> 2)
}

fn fields(raw: &[u8]) -> (i32, i32, usize, u64) {
    let bits = u64::from_be_bytes([0, 0, raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]);
    (
        i32::from(raw[0]),
        i32::from(raw[1] >> 4),
        usize::from(raw[1] & 15),
        bits,
    )
}

#[inline(always)]
fn index(bits: u64, p: usize) -> usize {
    (bits >> (45 - 3 * p) & 7) as usize
}

/// Decodes an 8-bit EAC block (ETC2 RGBA8 alpha) to 16 values in raster order.
pub fn decode_a8(raw: &[u8]) -> [u8; 16] {
    let (base, mult, table, bits) = fields(raw);
    let mut out = [0u8; 16];
    for p in 0..16 {
        out[raster(p)] = (base + TABLES[table][index(bits, p)] * mult).clamp(0, 255) as u8;
    }
    out
}

/// 11-bit value of an unsigned R11 block's entry.
#[inline(always)]
fn r11_value(base: i32, mult: i32, entry: i32) -> i32 {
    let m = if mult == 0 { entry } else { entry * mult * 8 };
    (base * 8 + 4 + m).clamp(0, 2047)
}

/// Decodes an unsigned 11-bit EAC block to 16 values (0..=2047) in raster order.
pub fn decode_r11(raw: &[u8]) -> [u16; 16] {
    let (base, mult, table, bits) = fields(raw);
    let mut out = [0u16; 16];
    for p in 0..16 {
        out[raster(p)] = r11_value(base, mult, TABLES[table][index(bits, p)]) as u16;
    }
    out
}

/// 11-bit value to the nearest 8-bit one.
#[inline]
pub fn r11_to_u8(v: u16) -> u8 {
    ((u32::from(v) * 255 + 1023) / 2047) as u8
}

fn decode_r11_u8(raw: &[u8]) -> [u8; 16] {
    decode_r11(raw).map(r11_to_u8)
}

/// EAC R11 block decode: value replicated to RGB, opaque alpha (like BC4).
pub fn decode_r11_block(raw: &[u8], out: &mut Block) {
    let r = decode_r11_u8(&raw[..8]);
    for (t, r) in out.iter_mut().zip(r) {
        *t = [r, r, r, 255];
    }
}

/// EAC RG11 block decode: R and G, blue 0, opaque alpha (like BC5).
pub fn decode_rg11_block(raw: &[u8], out: &mut Block) {
    let r = decode_r11_u8(&raw[..8]);
    let g = decode_r11_u8(&raw[8..16]);
    for (t, (r, g)) in out.iter_mut().zip(r.into_iter().zip(g)) {
        *t = [r, g, 0, 255];
    }
}

/// Search settings for one effort level.
#[derive(Clone, Copy, Debug)]
struct Cfg {
    /// Multiplier window radius around the range fit.
    dm: i32,
    /// Base window radius around the range fit.
    db: i32,
    /// Least-squares refit rounds.
    refits: u32,
}

fn cfg(effort: u8) -> Cfg {
    let (dm, db, refits) = match effort {
        0 => (0, 1, 0),
        1..=3 => (0, 1, 1),
        4..=6 => (1, 1, 2),
        7 | 8 => (1, 2, 3),
        _ => (2, 3, 4),
    };
    Cfg { dm, db, refits }
}

/// One channel of a block in column-major order, with its targets in palette units.
struct Target {
    eleven: bool,
    /// 8-bit: the values; 11-bit: the values scaled to 11 bits, times 4.
    t: [i32; 16],
}

impl Target {
    fn new(v: &[u8; 16], eleven: bool) -> Self {
        let t = std::array::from_fn(|p| {
            let x = i32::from(v[raster(p)]);
            if eleven { (x * 8188 + 127) / 255 } else { x }
        });
        Self { eleven, t }
    }

    /// The table's eight values in target units.
    #[inline(always)]
    fn palette(&self, base: i32, mult: i32, table: usize) -> [i32; 8] {
        let tab = &TABLES[table];
        if self.eleven {
            std::array::from_fn(|k| r11_value(base, mult, tab[k]) * 4)
        } else {
            std::array::from_fn(|k| (base + tab[k] * mult).clamp(0, 255))
        }
    }

    #[inline(always)]
    fn error(&self, pal: &[i32; 8]) -> i32 {
        let mut e = [i32::MAX; 16];
        for &v in pal {
            for p in 0..16 {
                let d = self.t[p] - v;
                e[p] = e[p].min(d * d);
            }
        }
        e.iter().sum()
    }

    fn indices(&self, pal: &[i32; 8]) -> [u8; 16] {
        std::array::from_fn(|p| {
            let mut best = (i32::MAX, 0u8);
            for (k, &v) in pal.iter().enumerate() {
                let d = (self.t[p] - v).pow(2);
                if d < best.0 {
                    best = (d, k as u8);
                }
            }
            best.1
        })
    }

    /// Smallest allowed multiplier: 11-bit blocks have the unscaled multiplier 0.
    fn min_mult(&self) -> i32 {
        i32::from(!self.eleven)
    }

    /// Base and multiplier that put the table's extremes at `lo` and `hi` (target units).
    fn range_fit(&self, lo: i32, hi: i32, table: usize) -> (i32, i32) {
        let tab = &TABLES[table];
        let (tmin, tmax) = (tab[3], tab[7]);
        let span = tmax - tmin;
        if self.eleven {
            // Value units are 11-bit × 4.
            let m = ((hi - lo + 16 * span) / (32 * span)).clamp(0, 15);
            let scale = if m == 0 { 1 } else { m * 8 };
            let mid = (lo + hi) / 8 - 4 - scale * (tmin + tmax) / 2;
            (((mid + 4) / 8).clamp(0, 255), m)
        } else {
            let m = ((hi - lo + span / 2) / span).clamp(1, 15);
            let mid = (lo + hi + 1) / 2 - (m * (tmin + tmax)) / 2;
            (mid.clamp(0, 255), m)
        }
    }

    /// Least-squares (base, multiplier) for the entries the texels picked.
    fn refit(&self, idx: &[u8; 16], table: usize, old: (i32, i32)) -> (i32, i32) {
        let tab = &TABLES[table];
        let (mut st, mut stt, mut sv, mut stv) = (0i64, 0i64, 0i64, 0i64);
        for p in 0..16 {
            let e = i64::from(tab[idx[p] as usize]);
            let v = i64::from(self.t[p]);
            st += e;
            stt += e * e;
            sv += v;
            stv += e * v;
        }
        let den = 16 * stt - st * st;
        if den == 0 {
            return old;
        }
        // v ≈ a + slope · entry.
        let slope = (16 * stv - st * sv) as f32 / den as f32;
        let a = (sv as f32 - slope * st as f32) / 16.0;
        if self.eleven {
            // v/4 ≈ base·8 + 4 + entry·mult·8.
            let m = (slope / 32.0).round().clamp(0.0, 15.0) as i32;
            let b = ((a / 4.0 - 4.0) / 8.0).round().clamp(0.0, 255.0) as i32;
            (b, m)
        } else {
            let m = slope.round().clamp(1.0, 15.0) as i32;
            (a.round().clamp(0.0, 255.0) as i32, m)
        }
    }
}

#[derive(Clone, Copy)]
struct Fit {
    err: i32,
    base: i32,
    mult: i32,
    table: usize,
}

fn search(tg: &Target, cfg: &Cfg) -> Fit {
    let lo = *tg.t.iter().min().unwrap();
    let hi = *tg.t.iter().max().unwrap();
    let mut best = Fit {
        err: i32::MAX,
        base: 0,
        mult: 0,
        table: 0,
    };
    if !tg.eleven && lo == hi {
        // Multiplier 0: every texel is the base.
        return Fit {
            err: 0,
            base: lo,
            mult: 0,
            table: 0,
        };
    }
    let mmin = tg.min_mult();
    for table in 0..16 {
        let mut top = Fit {
            err: i32::MAX,
            base: 0,
            mult: 0,
            table,
        };
        let try_window = |top: &mut Fit, (b0, m0): (i32, i32), dm: i32, db: i32| {
            for m in (m0 - dm).max(mmin)..=(m0 + dm).min(15) {
                for b in (b0 - db).max(0)..=(b0 + db).min(255) {
                    let e = tg.error(&tg.palette(b, m, table));
                    if e < top.err {
                        *top = Fit {
                            err: e,
                            base: b,
                            mult: m,
                            table,
                        };
                    }
                }
            }
        };
        try_window(&mut top, tg.range_fit(lo, hi, table), cfg.dm, cfg.db);
        for _ in 0..cfg.refits {
            let idx = tg.indices(&tg.palette(top.base, top.mult, table));
            let fit = tg.refit(&idx, table, (top.base, top.mult));
            let before = (top.base, top.mult);
            try_window(&mut top, fit, 1, 1);
            if (top.base, top.mult) == before {
                break;
            }
        }
        if top.err < best.err {
            best = top;
        }
    }
    best
}

fn pack(tg: &Target, f: Fit) -> [u8; 8] {
    let idx = tg.indices(&tg.palette(f.base, f.mult, f.table));
    let bits = idx
        .iter()
        .enumerate()
        .fold(0u64, |a, (p, &k)| a | u64::from(k) << (45 - 3 * p));
    let b = bits.to_be_bytes();
    [
        f.base as u8,
        (f.mult << 4 | f.table as i32) as u8,
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
    ]
}

#[inline(always)]
fn encode(v: &[u8; 16], eleven: bool, cfg: &Cfg) -> [u8; 8] {
    let tg = Target::new(v, eleven);
    pack(&tg, search(&tg, cfg))
}

/// Encodes 16 raster-order values as an 8-bit EAC block (ETC2 RGBA8 alpha).
pub fn encode_a8(v: &[u8; 16], effort: u8) -> [u8; 8] {
    encode(v, false, &cfg(effort))
}

/// Encodes 16 raster-order 8-bit values as an unsigned 11-bit EAC block.
pub fn encode_r11(v: &[u8; 16], effort: u8) -> [u8; 8] {
    encode(v, true, &cfg(effort))
}

fn encode_span(surface: &Surface, first: usize, out: &mut [u8], chans: (usize, usize), eleven: bool, effort: u8) {
    let cfg = cfg(effort);
    let bw = surface.blocks_x() as usize;
    let bb = 8 * chans.1;
    for (k, o) in out.chunks_exact_mut(bb).enumerate() {
        let i = first + k;
        let (bx, by) = ((i % bw) as u32, (i / bw) as u32);
        for (c, o) in o.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            *o = encode(&surface.read_block_channel(bx, by, chans.0 + c), eleven, &cfg);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn encode_span_avx2(surface: &Surface, first: usize, out: &mut [u8], chans: (usize, usize), eleven: bool, effort: u8) {
    encode_span(surface, first, out, chans, eleven, effort);
}

/// Encodes channels `chans.0 .. chans.0 + chans.1` of every block (one EAC block each, in
/// order), 11-bit when `eleven`, else 8-bit. Returns raw blocks in raster order.
pub fn encode_surface(surface: &Surface, chans: (usize, usize), eleven: bool, effort: u8) -> Vec<u8> {
    use rayon::prelude::*;
    const SPAN: usize = 512;
    let bb = 8 * chans.1;
    let mut out = vec![0u8; surface.block_count() * bb];
    let lvl = crate::simd::level();
    out.par_chunks_mut(SPAN * bb).enumerate().for_each(|(ci, o)| match lvl {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        crate::simd::Level::Avx2 => unsafe { encode_span_avx2(surface, ci * SPAN, o, chans, eleven, effort) },
        _ => encode_span(surface, ci * SPAN, o, chans, eleven, effort),
    });
    out
}

/// `raw`'s base, multiplier and table with every index re-chosen for the 16 raster-order values `v`
/// (for RDO: a copy of another block's first two bytes).
pub fn reindex(raw: &[u8], v: &[u8; 16], eleven: bool) -> [u8; 8] {
    let tg = Target::new(v, eleven);
    let (base, mult, table, _) = fields(raw);
    pack(
        &tg,
        Fit {
            err: 0,
            base,
            mult,
            table,
        },
    )
}

/// The indices in `idx` (the last six bytes of another block) with the base, multiplier and table
/// fit to the 16 raster-order values `v`: a least-squares fit per table and its ±1 neighbors (for
/// RDO: a copy of another block's indices).
pub fn refit(idx: &[u8], v: &[u8; 16], eleven: bool) -> [u8; 8] {
    let tg = Target::new(v, eleven);
    let bits = u64::from_be_bytes([0, 0, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]]);
    let ix: [u8; 16] = std::array::from_fn(|p| index(bits, p) as u8);
    let lo = *tg.t.iter().min().unwrap();
    let hi = *tg.t.iter().max().unwrap();
    let mmin = tg.min_mult();
    let mut best = (i32::MAX, 0, 0, 0);
    for table in 0..16 {
        let (b0, m0) = tg.refit(&ix, table, tg.range_fit(lo, hi, table));
        for m in (m0 - 1).max(mmin)..=(m0 + 1).min(15) {
            for b in (b0 - 1).max(0)..=(b0 + 1).min(255) {
                let pal = tg.palette(b, m, table);
                let e: i32 = (0..16).map(|p| (tg.t[p] - pal[ix[p] as usize]).pow(2)).sum();
                if e < best.0 {
                    best = (e, b, m, table);
                }
            }
        }
    }
    let (_, b, m, t) = best;
    [
        b as u8,
        (m << 4 | t as i32) as u8,
        idx[0],
        idx[1],
        idx[2],
        idx[3],
        idx[4],
        idx[5],
    ]
}

/// Squared error of an encoded channel against its 8-bit source, after decoding to 8 bits.
pub fn channel_error(raw: &[u8], v: &[u8; 16], eleven: bool) -> u32 {
    let d = if eleven { decode_r11_u8(raw) } else { decode_a8(raw) };
    d.iter().zip(v).map(|(a, b)| u32::from(a.abs_diff(*b)).pow(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(base: u8, mult: u8, table: u8, idx: [u8; 16]) -> [u8; 8] {
        let bits = idx
            .iter()
            .enumerate()
            .fold(0u64, |a, (p, &k)| a | u64::from(k) << (45 - 3 * p));
        let b = bits.to_be_bytes();
        [base, mult << 4 | table, b[2], b[3], b[4], b[5], b[6], b[7]]
    }

    /// Column-major index list that walks entries 0..8 twice.
    const WALK: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0];

    #[test]
    fn a8_tables_and_multipliers() {
        let raw = block(100, 3, 0, WALK);
        let v = decode_a8(&raw);
        // p=0 is raster 0, p=1 raster 4 (x0, y1), p=4 raster 1 (x1, y0).
        assert_eq!(v[0], 100 - 9);
        assert_eq!(v[4], 100 - 18);
        assert_eq!(v[12], 100 - 45);
        assert_eq!(v[1], 100 + 6);
        assert_eq!(v[13], 100 + 42);
        // Clamping, and table 13 with a large multiplier.
        let v = decode_a8(&block(250, 15, 13, WALK));
        assert_eq!(v[12], 250 - 150);
        assert_eq!(v[1], 250);
        assert_eq!(v[13], 255);
        // Multiplier 0: the base everywhere.
        assert_eq!(decode_a8(&block(77, 0, 5, WALK)), [77; 16]);
    }

    #[test]
    fn r11_extended_precision() {
        // base·8 + 4 + entry·mult·8.
        let v = decode_r11(&block(100, 2, 1, WALK));
        assert_eq!(v[0], 804 - 48);
        assert_eq!(v[13], 804 + 192);
        // Multiplier 0 adds the entry unscaled: 11-bit steps of one.
        let v = decode_r11(&block(100, 0, 13, WALK));
        assert_eq!(v[0], 803);
        assert_eq!(v[1], 804);
        assert_eq!(v[5], 805);
        assert_eq!(v[9], 806);
        assert_eq!(v[13], 813);
        assert_eq!(v[12], 794);
        // Clamps to 0..=2047.
        let v = decode_r11(&block(255, 15, 0, WALK));
        assert_eq!(v[13], 2047);
        let v = decode_r11(&block(0, 15, 0, WALK));
        assert_eq!(v[12], 0);
        assert_eq!(r11_to_u8(2047), 255);
        assert_eq!(r11_to_u8(0), 0);
        assert_eq!(r11_to_u8(1024), 128);
    }

    fn rng(seed: &mut u32) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed
    }

    #[test]
    fn encoders_are_close_and_improve_with_effort() {
        let mut seed = 0x51ed_270b;
        for eleven in [false, true] {
            let mut totals = [0u64; 10];
            for k in 0..500 {
                let lo = rng(&mut seed) % 256;
                let span = [4, 20, 80, 256][k % 4];
                let v: [u8; 16] = std::array::from_fn(|_| (lo + rng(&mut seed) % span).min(255) as u8);
                for e in 0..10u8 {
                    let raw = if eleven { encode_r11(&v, e) } else { encode_a8(&v, e) };
                    totals[e as usize] += u64::from(channel_error(&raw, &v, eleven));
                }
            }
            assert!(
                totals[9] <= totals[4] && totals[4] <= totals[0],
                "eleven {eleven}: {totals:?}"
            );
            // Under ~2.5 per texel RMS on this mix.
            assert!(totals[5] < 500 * 16 * 7, "eleven {eleven}: {totals:?}");
        }
    }

    /// RDO helpers: re-indexing never loses against the block's own indices, and refitting its
    /// own indices stays close.
    #[test]
    fn reindex_and_refit() {
        let mut seed = 0x1357_9bdfu32;
        for k in 0..400 {
            let v: [u8; 16] = std::array::from_fn(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                if k % 2 == 0 {
                    (seed % 256) as u8
                } else {
                    (i * 9 + (seed % 20) as usize) as u8
                }
            });
            for eleven in [false, true] {
                let raw = encode(&v, eleven, &cfg(2));
                let e = channel_error(&raw, &v, eleven);
                let r = reindex(&raw, &v, eleven);
                assert_eq!(r[..2], raw[..2]);
                assert!(channel_error(&r, &v, eleven) <= e + 16, "{k} {eleven}");
                let f = refit(&raw[2..], &v, eleven);
                assert_eq!(f[2..], raw[2..]);
                assert!(channel_error(&f, &v, eleven) <= e * 2 + 32, "{k} {eleven}");
            }
        }
    }

    #[test]
    fn flat_blocks_are_exact() {
        for x in [0u8, 1, 77, 128, 254, 255] {
            assert_eq!(channel_error(&encode_a8(&[x; 16], 0), &[x; 16], false), 0);
            assert_eq!(channel_error(&encode_r11(&[x; 16], 0), &[x; 16], true), 0);
        }
    }
}
