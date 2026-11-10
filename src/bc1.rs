//! BC1 (DXT1) RGB block codec.
//!
//! Layout: `c0: u16 LE (565)`, `c1: u16 LE (565)`, then 32 bits of 2-bit indices, texel 0 in the
//! low bits. `c0 > c1` selects 4-color mode; otherwise 3-color mode where index 3 is black.
//! The interpolation rule matches the common reference decoders (`(2*a + b) / 3`, truncating).

mod batch;

pub use batch::{encode_block as encode_fast, encode_surface};

use crate::fit::{V4, ls_endpoints, pca_endpoints};
use crate::simd::{LANES, Level, SimdF32, SimdI32, f32x8};
use crate::surface::{Block, texel_sse};
use std::sync::OnceLock;

#[inline]
const fn expand5(v: u32) -> u32 {
    (v << 3) | (v >> 2)
}

#[inline]
const fn expand6(v: u32) -> u32 {
    (v << 2) | (v >> 4)
}

/// Unpacks a 565 color to 8-bit RGB.
#[inline]
pub fn unpack565(c: u16) -> [u32; 3] {
    let c = c as u32;
    [expand5((c >> 11) & 31), expand6((c >> 5) & 63), expand5(c & 31)]
}

#[inline]
fn pack565(q: [i32; 3]) -> u16 {
    ((q[0] as u16) << 11) | ((q[1] as u16) << 5) | q[2] as u16
}

#[inline]
fn split565(c: u16) -> [i32; 3] {
    [(c >> 11) as i32 & 31, (c >> 5) as i32 & 63, c as i32 & 31]
}

/// Nearest 565 levels. `x + 0.5` truncated equals `round` for the clamped, non-negative inputs,
/// without the libm call `f32::round` costs on baseline x86-64.
#[inline]
fn quant565(v: &V4) -> [i32; 3] {
    [
        (v[0].clamp(0.0, 255.0) * (31.0 / 255.0) + 0.5) as i32,
        (v[1].clamp(0.0, 255.0) * (63.0 / 255.0) + 0.5) as i32,
        (v[2].clamp(0.0, 255.0) * (31.0 / 255.0) + 0.5) as i32,
    ]
}

/// The four palette entries a decoder produces for (c0, c1).
pub fn palette(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    let a = unpack565(c0);
    let b = unpack565(c1);
    let mut p = [[0u8, 0, 0, 255]; 4];
    for ch in 0..3 {
        p[0][ch] = a[ch] as u8;
        p[1][ch] = b[ch] as u8;
        if c0 > c1 {
            p[2][ch] = ((2 * a[ch] + b[ch]) / 3) as u8;
            p[3][ch] = ((a[ch] + 2 * b[ch]) / 3) as u8;
        } else {
            p[2][ch] = ((a[ch] + b[ch]) / 2) as u8;
            p[3][ch] = 0;
        }
    }
    p
}

/// Decodes one 8-byte block.
pub fn decode(raw: &[u8], out: &mut Block) {
    let c0 = u16::from_le_bytes([raw[0], raw[1]]);
    let c1 = u16::from_le_bytes([raw[2], raw[3]]);
    let pal = palette(c0, c1);
    let bits = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    for (i, t) in out.iter_mut().enumerate() {
        *t = pal[((bits >> (2 * i)) & 3) as usize];
    }
}

/// Weighted RGB SSE of an encoded block against `px`, without materializing the decode. Sums
/// row by row and may return early once the total exceeds `bound` (the result is then some value
/// above `bound`); otherwise it is exact.
#[inline]
pub fn sse_bounded(raw: &[u8], px: &Block, w: [u32; 3], bound: u64) -> u64 {
    let c0 = u16::from_le_bytes([raw[0], raw[1]]);
    let c1 = u16::from_le_bytes([raw[2], raw[3]]);
    let pal = palette(c0, c1);
    let mut e = 0u64;
    for row in 0..4 {
        let b = raw[4 + row];
        let mut r = 0u32;
        for j in 0..4 {
            let t = &px[4 * row + j];
            let p = &pal[((b >> (2 * j)) & 3) as usize];
            for c in 0..3 {
                let d = t[c] as i32 - p[c] as i32;
                r += w[c] * (d * d) as u32;
            }
        }
        e += r as u64;
        if e > bound {
            return e;
        }
    }
    e
}

/// Picks the best index per texel for a fixed palette. Returns (index bits, weighted SSE).
fn assign(px: &Block, pal: &[[u8; 4]; 4], w: &[u32; 4]) -> (u32, u64) {
    let wrgb = [w[0], w[1], w[2], 0];
    let mut bits = 0u32;
    let mut err = 0u64;
    for (i, t) in px.iter().enumerate() {
        let mut best = (u32::MAX, 0u32);
        for (k, p) in pal.iter().enumerate() {
            let e = texel_sse(t, p, &wrgb);
            if e < best.0 {
                best = (e, k as u32);
            }
        }
        bits |= best.1 << (2 * i);
        err += best.0 as u64;
    }
    (bits, err)
}

fn to_bytes(c0: u16, c1: u16, bits: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..2].copy_from_slice(&c0.to_le_bytes());
    b[2..4].copy_from_slice(&c1.to_le_bytes());
    b[4..8].copy_from_slice(&bits.to_le_bytes());
    b
}

/// Packs endpoints in 4-color order (c0 > c1) and assigns the best indices.
fn finalize(q0: [i32; 3], q1: [i32; 3], px: &Block, w: &[u32; 4]) -> ([u8; 8], u64) {
    let (mut c0, mut c1) = (pack565(q0), pack565(q1));
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    let (bits, err) = assign(px, &palette(c0, c1), w);
    (to_bytes(c0, c1, bits), err)
}

/// For each 8-bit value, the (hi, lo) quantized endpoint pair whose 2/3 interpolant hits it.
type MatchTable = [(u8, u8); 256];

/// Optimal single-color endpoint pair for every 8-bit value: the pair whose 1/3 interpolant is
/// nearest the value, ties broken by the closer endpoints, then the first (a, b). Built at
/// compile time: building it lazily cost ~2 ms, charged to the first image with a solid block.
/// For each `a` only the few `b` around `3v - 2a` can be optimal (the interpolant moves by
/// ~2.7 per 5-bit step, the 1000× weight makes interpolation error dominate), so the window
/// gives the full search's answer (checked by `single_color_tables_match_full_search`).
const fn build_match(levels: u32, six: bool) -> MatchTable {
    let mut t = [(0u8, 0u8); 256];
    let mut v = 0;
    while v < 256 {
        let mut best = (u32::MAX, 0u32, 0u32);
        let mut a = 0;
        while a < levels {
            let ea = if six { expand6(a) } else { expand5(a) };
            let mut target = 3 * v as i32 - 2 * ea as i32;
            if target < 0 {
                target = 0;
            } else if target > 255 {
                target = 255;
            }
            let bt = (target * (levels as i32 - 1) + 127) / 255;
            let lo = if bt > 3 { bt - 3 } else { 0 };
            let hi = if bt + 3 < levels as i32 - 1 {
                bt + 3
            } else {
                levels as i32 - 1
            };
            let mut b = lo;
            while b <= hi {
                let eb = if six { expand6(b as u32) } else { expand5(b as u32) };
                let interp = (2 * ea + eb) / 3;
                let err = interp.abs_diff(v as u32) * 1000 + ea.abs_diff(eb);
                // Strict: the first pair in (a, b) order wins ties, as in a full search.
                if err < best.0 {
                    best = (err, a, b as u32);
                }
                b += 1;
            }
            a += 1;
        }
        t[v] = (best.1 as u8, best.2 as u8);
        v += 1;
    }
    t
}

static MATCH5: MatchTable = build_match(32, false);
static MATCH6: MatchTable = build_match(64, true);

/// Optimal single-color tables for 5-bit and 6-bit channels.
fn single_color_tables() -> (&'static MatchTable, &'static MatchTable) {
    (&MATCH5, &MATCH6)
}

fn encode_solid(px: &Block, w: &[u32; 4]) -> ([u8; 8], u64) {
    let (t5, t6) = single_color_tables();
    let c = px[0];
    let a = [t5[c[0] as usize].0, t6[c[1] as usize].0, t5[c[2] as usize].0];
    let b = [t5[c[0] as usize].1, t6[c[1] as usize].1, t5[c[2] as usize].1];
    let qa = [a[0] as i32, a[1] as i32, a[2] as i32];
    let qb = [b[0] as i32, b[1] as i32, b[2] as i32];
    finalize(qa, qb, px, w)
}

/// Interpolation weight (0 = c0, 1 = c1) for each 4-color index.
const T4: [f32; 4] = [0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0];

fn refit_from_bits(bits: u32, pts: &[V4]) -> Option<(V4, V4)> {
    let t: Vec<f32> = (0..16).map(|i| T4[((bits >> (2 * i)) & 3) as usize]).collect();
    ls_endpoints(pts, &t, 3)
}

/// Packs endpoints in 3-color order (c0 <= c1: index 2 is the midpoint, index 3 is black) and
/// assigns the best indices. Dark texels can use the black entry; KTX2 output is
/// `VK_FORMAT_BC1_RGB_*`, whose alpha is ignored, so the transparent-black alpha never shows.
fn finalize3(q0: [i32; 3], q1: [i32; 3], px: &Block, w: &[u32; 4]) -> ([u8; 8], u64) {
    let (mut c0, mut c1) = (pack565(q0), pack565(q1));
    if c0 > c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    let (bits, err) = assign(px, &palette(c0, c1), w);
    (to_bytes(c0, c1, bits), err)
}

/// One way to cut `n` sorted texels into contiguous runs, one per palette weight, with the
/// split's least-squares coefficients (they depend only on the run lengths).
#[derive(Clone, Copy, Debug)]
struct Split {
    /// Cut points i <= j (<= k for 4-color): runs are [0, i), [i, j), [j, k), [k, n).
    cuts: [u8; 3],
    /// Σ(1-t)², Σt², Σ(1-t)t.
    a2: f32,
    b2: f32,
    ab: f32,
    inv_det: f32,
}

/// Every split of `n` texels in canonical order (i, then j, then k ascending), including
/// degenerate ones (inv_det = 0), so a split's position is a stable id.
fn all_splits(n: usize, three: bool) -> Vec<Split> {
    let mut out = Vec::new();
    let mut push = |cuts: [usize; 3], a2: f32, b2: f32, ab: f32| {
        let det = a2 * b2 - ab * ab;
        out.push(Split {
            cuts: [cuts[0] as u8, cuts[1] as u8, cuts[2] as u8],
            a2,
            b2,
            ab,
            inv_det: if det < 1e-4 { 0.0 } else { 1.0 / det },
        });
    };
    if three {
        for i in 0..=n {
            for j in i..=n {
                let (n0, n1, n2) = (i as f32, (j - i) as f32, (n - j) as f32);
                push([i, j, n], n0 + 0.25 * n1, 0.25 * n1 + n2, 0.25 * n1);
            }
        }
    } else {
        const T: f32 = 1.0 / 3.0;
        for i in 0..=n {
            for j in i..=n {
                for k in j..=n {
                    let (n0, n1, n2, n3) = (i as f32, (j - i) as f32, (k - j) as f32, (n - k) as f32);
                    push(
                        [i, j, k],
                        n0 + T * T * (4.0 * n1 + n2),
                        T * T * (n1 + 4.0 * n2) + n3,
                        2.0 * T * T * (n1 + n2),
                    );
                }
            }
        }
    }
    out
}

/// A split table in structure-of-arrays form for the SIMD scorer, padded to a multiple of
/// [`LANES`] with degenerate entries (`inv_det` = 0, never chosen).
struct SplitTable {
    i: Vec<i32>,
    j: Vec<i32>,
    k: Vec<i32>,
    a2: Vec<f32>,
    b2: Vec<f32>,
    ab: Vec<f32>,
    inv_det: Vec<f32>,
    /// Real (unpadded) length.
    len: usize,
}

impl SplitTable {
    fn new(splits: &[Split]) -> Self {
        let padded = splits.len().div_ceil(LANES).max(1) * LANES;
        let get = |f: &dyn Fn(&Split) -> f32, pad: f32| -> Vec<f32> {
            let mut v: Vec<f32> = splits.iter().map(f).collect();
            v.resize(padded, pad);
            v
        };
        let cut = |c: usize| -> Vec<i32> {
            let mut v: Vec<i32> = splits.iter().map(|s| s.cuts[c] as i32).collect();
            v.resize(padded, 0);
            v
        };
        Self {
            i: cut(0),
            j: cut(1),
            k: cut(2),
            a2: get(&|s| s.a2, 1.0),
            b2: get(&|s| s.b2, 1.0),
            ab: get(&|s| s.ab, 0.0),
            inv_det: get(&|s| s.inv_det, 0.0),
            len: splits.len(),
        }
    }
}

/// Cached split tables: `ordered[three]` for 16 texels in frequency order (learned on the
/// synthetic corpus, see `bc1_order.rs`, degenerate splits dropped), and `canonical[n][three]`
/// with every split of `n` texels in canonical order (index = canonical id).
struct SplitTables {
    ordered: [SplitTable; 2],
    canonical: Vec<[SplitTable; 2]>,
}

fn split_tables() -> &'static SplitTables {
    static T: OnceLock<SplitTables> = OnceLock::new();
    T.get_or_init(|| {
        let ordered = |three: bool, rank: &[u16]| {
            let all = all_splits(16, three);
            let v: Vec<Split> = rank
                .iter()
                .map(|&id| all[id as usize])
                .filter(|s| s.inv_det > 0.0)
                .collect();
            SplitTable::new(&v)
        };
        SplitTables {
            ordered: [
                ordered(false, &crate::bc1_order::ORDER4),
                ordered(true, &crate::bc1_order::ORDER3),
            ],
            canonical: (0..=16)
                .map(|n| {
                    [
                        SplitTable::new(&all_splits(n, false)),
                        SplitTable::new(&all_splits(n, true)),
                    ]
                })
                .collect(),
        }
    })
}

/// Prefix sums of the sorted texels, one row per channel, padded so `& 31` indexing needs no
/// bounds check.
type Prefix = [[f32; 32]; 3];

/// 565 snap of a vector of endpoint channels: quantize, then expand back to 8 bits.
#[inline(always)]
fn snap_lanes<V: SimdF32>(v: V, six_bits: bool) -> V {
    let (maxq, hi, lo) = if six_bits { (63.0, 2, 4) } else { (31.0, 3, 2) };
    let q = (v * V::splat(maxq / 255.0) + V::splat(0.5)).to_int_trunc();
    ((q << hi) | (q >> lo)).to_float()
}

/// Per-split least-squares coefficients for 8 lanes.
#[derive(Clone, Copy)]
struct Coef<V> {
    a2: V,
    b2: V,
    ab: V,
    inv: V,
}

/// Least-squares endpoint channel pair and error term of one channel for 8 splits, shared by
/// the vector scorer and the endpoint recovery so both compute bit-identical values.
/// `snap` = Some(six_bits) scores with 565-quantized endpoints.
#[inline(always)]
fn split_channel<V: SimdF32>(cut_sum: V, scale: f32, total: f32, k: Coef<V>, snap: Option<bool>) -> (V, V, V) {
    let ax = V::splat(scale) * cut_sum;
    let bx = V::splat(total) - ax;
    let (lo, hi) = (V::splat(0.0), V::splat(255.0));
    let mut a = ((ax * k.b2 - bx * k.ab) * k.inv).simd_clamp(lo, hi);
    let mut b = ((bx * k.a2 - ax * k.ab) * k.inv).simd_clamp(lo, hi);
    if let Some(six) = snap {
        a = snap_lanes(a, six);
        b = snap_lanes(b, six);
    }
    // Σ|(1-t)a + t·b - p|² minus the constant Σ|p|².
    let two = V::splat(2.0);
    let err = a * a * k.a2 + b * b * k.b2 + two * (a * b * k.ab - a * ax - b * bx);
    (a, b, err)
}

/// Inputs of one cluster-fit search.
struct SplitSearch<'a> {
    pre: &'a Prefix,
    total: [f32; 3],
    wc: [f32; 3],
    table: &'a SplitTable,
    len: usize,
    three: bool,
    snap: bool,
}

/// Scores the first `len` splits of the table, 8 per step; returns the index of the lowest
/// error (lowest index on ties, like a sequential scan).
#[inline(always)]
fn best_split_kernel<V: SimdF32>(q: &SplitSearch) -> Option<usize> {
    let t = q.table;
    let scale = if q.three { 0.5 } else { 1.0 / 3.0 };
    let inf = V::splat(f32::INFINITY);
    let zero = V::splat(0.0);
    let mut best_err = inf;
    let mut best_idx = V::Int::splat(-1);
    let lane = V::Int::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let len = V::Int::splat(q.len as i32);
    let mut base = 0;
    while base < q.len {
        let (ii, jj, kk) = (
            V::Int::from_slice(&t.i[base..]),
            V::Int::from_slice(&t.j[base..]),
            V::Int::from_slice(&t.k[base..]),
        );
        let k = Coef {
            a2: V::from_slice(&t.a2[base..]),
            b2: V::from_slice(&t.b2[base..]),
            ab: V::from_slice(&t.ab[base..]),
            inv: V::from_slice(&t.inv_det[base..]),
        };
        let mut err = zero;
        for c in 0..3 {
            let row = &q.pre[c];
            let mut cut_sum = V::lookup17(row, ii) + V::lookup17(row, jj);
            if !q.three {
                cut_sum += V::lookup17(row, kk);
            }
            let (_, _, e) = split_channel(cut_sum, scale, q.total[c], k, q.snap.then_some(c == 1));
            err += V::splat(q.wc[c]) * e;
        }
        // Degenerate splits and padding past `len` never win.
        let idx = lane + V::Int::splat(base as i32);
        let valid = k.inv.simd_gt(zero) & idx.simd_lt(len);
        let err = V::select(valid, err, inf);
        let better = err.simd_lt(best_err);
        best_err = V::select(better, err, best_err);
        best_idx = V::Int::select(better, idx, best_idx);
        base += LANES;
    }
    let (e, ix) = (best_err.to_array(), best_idx.to_array());
    let mut best = (f32::INFINITY, -1i32);
    for l in 0..LANES {
        if ix[l] >= 0 && (e[l] < best.0 || (e[l] == best.0 && ix[l] < best.1)) {
            best = (e[l], ix[l]);
        }
    }
    (best.1 >= 0).then_some(best.1 as usize)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn best_split_avx2(q: &SplitSearch) -> Option<usize> {
    best_split_kernel::<crate::simd::avx2::F>(q)
}

/// Runtime dispatch to the best instantiation of [`best_split_kernel`].
fn best_split(q: &SplitSearch) -> Option<usize> {
    match crate::simd::level() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level()` returns Avx2 only when the CPU reports AVX2 and FMA.
        Level::Avx2 => unsafe { best_split_avx2(q) },
        _ => best_split_kernel::<f32x8>(q),
    }
}

/// Texels sorted along an axis, as per-channel prefix sums: the input of [`cluster_fit`].
struct Sorted {
    pre: Prefix,
    n: usize,
}

/// Sorts `pts` (2..=16 texels) by their projection on `axis` and builds the prefix sums.
fn sort_along(pts: &[V4], axis: &V4) -> Option<Sorted> {
    let n = pts.len();
    if !(2..=16).contains(&n) {
        return None;
    }
    let mut order: [(f32, u8); 16] = [(0.0, 0); 16];
    for (i, p) in pts.iter().enumerate() {
        order[i] = (p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2], i as u8);
    }
    order[..n].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut pre: Prefix = [[0f32; 32]; 3];
    for k in 0..n {
        let p = &pts[order[k].1 as usize];
        for c in 0..3 {
            pre[c][k + 1] = pre[c][k] + p[c];
        }
    }
    Some(Sorted { pre, n })
}

/// Cluster fit (as in libsquish and icbc): texels sorted along an axis ([`sort_along`]) are
/// split into contiguous runs, one per palette entry (4-color: weights 0, 1/3, 2/3, 1; 3-color:
/// 0, 1/2, 1), and each split's least-squares endpoints come from prefix sums. With prefix sums
/// P over the sorted texels, a split's Σ(1-t)p is `(P[i] + P[j] + P[k]) / 3` (4-color) or
/// `(P[i] + P[j]) / 2` (3-color), and Σt·p is the total minus that, so each split costs a
/// handful of flops and 8 splits are scored per SIMD step ([`best_split`]).
///
/// For 16 texels only the first `limit` splits in frequency order are tried (icbc does the same);
/// smaller sets try every split. With `snap`, each split is scored with its endpoints quantized
/// to 565; otherwise with the exact endpoints. Returns the winning quantized endpoint pair
/// (weight-0 end first) and its estimated error minus Σ|p|² over all 16 texels: texels left out
/// of `sorted` are assumed to take the black entry (error |p|²), so estimates of 4-color,
/// 3-color and dark-excluded fits of one block are directly comparable. Records the winner's
/// canonical id in `stats` if given.
fn cluster_fit(
    sorted: &Sorted,
    w: &[u32; 4],
    three: bool,
    snap: bool,
    limit: usize,
    stats: Option<&mut [u32]>,
) -> Option<([i32; 3], [i32; 3], f32)> {
    let (pre, n) = (&sorted.pre, sorted.n);
    let tables = split_tables();
    let (table, len) = if n == 16 && stats.is_none() {
        let t = &tables.ordered[three as usize];
        (t, limit.min(t.len))
    } else {
        let t = &tables.canonical[n][three as usize];
        (t, t.len)
    };
    let q = SplitSearch {
        pre,
        total: [pre[0][n], pre[1][n], pre[2][n]],
        wc: [w[0] as f32, w[1] as f32, w[2] as f32],
        table,
        len,
        three,
        snap,
    };
    let best = best_split(&q)?;
    if let Some(stats) = stats {
        stats[best] += 1;
    }
    // Recover the winner's endpoints with the same arithmetic (lane 0 of a splat).
    let scale = if three { 0.5 } else { 1.0 / 3.0 };
    let k = Coef {
        a2: f32x8::splat(table.a2[best]),
        b2: f32x8::splat(table.b2[best]),
        ab: f32x8::splat(table.ab[best]),
        inv: f32x8::splat(table.inv_det[best]),
    };
    let (mut ea, mut eb, mut err) = ([0f32; 3], [0f32; 3], 0f32);
    for c in 0..3 {
        let row = &pre[c];
        let mut cut_sum = row[table.i[best] as usize] + row[table.j[best] as usize];
        if !three {
            cut_sum += row[table.k[best] as usize];
        }
        let (a, b, e) = split_channel(f32x8::splat(cut_sum), scale, q.total[c], k, snap.then_some(c == 1));
        ea[c] = a.to_array()[0];
        eb[c] = b.to_array()[0];
        err += q.wc[c] * e.to_array()[0];
    }
    Some((
        quant565(&[ea[0], ea[1], ea[2], 0.0]),
        quant565(&[eb[0], eb[1], eb[2], 0.0]),
        err,
    ))
}

/// Split counts tried per effort level (4-color, 3-color); `usize::MAX` = all.
const CLUSTER_LIMITS: [(usize, usize); 4] = [(0, 0), (256, 32), (256, 153), (usize::MAX, usize::MAX)];

/// Texels this dark are left out of the 3-color fit, so they can use the black palette entry.
const DARK_MAX: u8 = 12;

/// A cluster-fit candidate: (estimated error, endpoint pair, 3-color mode).
type Candidate = (f32, [i32; 3], [i32; 3], bool);

/// Encodes one block. `effort` 0..=3 trades speed for quality.
///
/// Effort 0 is a PCA fit with a least-squares refit. Effort 1 adds snapped 4-color and 3-color
/// cluster fits (the 3-color one also with dark texels left to the black entry) and fully
/// evaluates only the one with the lowest estimated error. Effort 2 tries more splits, evaluates
/// every candidate and adds a ±1 endpoint search; effort 3 tries every split, repeats the cluster
/// fit along the refined axis and searches longer.
pub fn encode(px: &Block, w: &[u32; 4], effort: u8) -> [u8; 8] {
    if px.iter().all(|t| t[..3] == px[0][..3]) {
        return encode_solid(px, w).0;
    }
    let pts: [V4; 16] = std::array::from_fn(|i| [px[i][0] as f32, px[i][1] as f32, px[i][2] as f32, 0.0]);
    // From effort 1 the cluster fit does the real work; the PCA fit is only the fallback.
    let mut best = encode_pca(px, w, &pts, if effort == 1 { 0 } else { effort });
    if effort >= 1 && best.1 > 0 {
        let (_, cov) = crate::fit::mean_cov(&pts, 3);
        let mut axis = crate::fit::principal_axis(&cov, 3);
        let rounds = if effort >= 3 { 2 } else { 1 };
        let (lim4, lim3) = CLUSTER_LIMITS[effort.min(3) as usize];
        for round in 0..rounds {
            let Some(sorted) = sort_along(&pts, &axis) else {
                break;
            };
            // Candidates: (estimated error, endpoints, 3-color).
            let mut cands: [Option<Candidate>; 3] = [None; 3];
            cands[0] = cluster_fit(&sorted, w, false, true, lim4, None).map(|(a, b, e)| (e, a, b, false));
            if round == 0 {
                cands[1] = cluster_fit(&sorted, w, true, true, lim3, None).map(|(a, b, e)| (e, a, b, true));
                // Again without the dark texels, which can take the black entry instead.
                let mut lit = [[0f32; 4]; 16];
                let mut n_lit = 0;
                for (t, p) in px.iter().zip(&pts) {
                    if t[..3].iter().any(|&v| v > DARK_MAX) {
                        lit[n_lit] = *p;
                        n_lit += 1;
                    }
                }
                if n_lit < 16 {
                    if let Some(s) = sort_along(&lit[..n_lit], &axis) {
                        cands[2] = cluster_fit(&s, w, true, true, usize::MAX, None).map(|(a, b, e)| (e, a, b, true));
                    }
                }
            }
            if effort == 1 {
                // Keep only the best estimate; the others are not worth a full index search.
                let keep = cands.iter().flatten().min_by(|x, y| x.0.total_cmp(&y.0)).copied();
                cands = [keep, None, None];
            }
            for &(_, a, b, three) in cands.iter().flatten() {
                let cand = if three {
                    finalize3(a, b, px, w)
                } else {
                    finalize(a, b, px, w)
                };
                if cand.1 < best.1 {
                    best = cand;
                }
            }
            // Next round: order texels along the current best endpoints' direction.
            let c0 = unpack565(u16::from_le_bytes([best.0[0], best.0[1]]));
            let c1 = unpack565(u16::from_le_bytes([best.0[2], best.0[3]]));
            let d: [f32; 3] = std::array::from_fn(|c| c1[c] as f32 - c0[c] as f32);
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len < 1e-3 {
                break;
            }
            axis = [d[0] / len, d[1] / len, d[2] / len, 0.0];
        }
    }
    if effort >= 2 && best.1 > 0 {
        best = refine_neighbors(best, px, w, if effort >= 3 { 4 } else { 1 });
    }
    best.0
}

/// PCA endpoints (inset), then least-squares refits while they help. Always 4-color.
fn encode_pca(px: &Block, w: &[u32; 4], pts: &[V4], effort: u8) -> ([u8; 8], u64) {
    let (mut lo, mut hi) = pca_endpoints(pts, 3);
    // Inset the bounding line slightly: extremes are usually outliers.
    for c in 0..3 {
        let d = (hi[c] - lo[c]) / 16.0;
        lo[c] += d;
        hi[c] -= d;
    }
    let mut best = finalize(quant565(&hi), quant565(&lo), px, w);

    let iters = [1, 2, 3, 4][effort.min(3) as usize];
    for _ in 0..iters {
        if best.1 == 0 {
            break;
        }
        let c0 = u16::from_le_bytes([best.0[0], best.0[1]]);
        let c1 = u16::from_le_bytes([best.0[2], best.0[3]]);
        if c0 <= c1 {
            break;
        }
        let bits = u32::from_le_bytes([best.0[4], best.0[5], best.0[6], best.0[7]]);
        let Some((a, b)) = refit_from_bits(bits, pts) else {
            break;
        };
        let cand = finalize(quant565(&a), quant565(&b), px, w);
        if cand.1 < best.1 {
            best = cand;
        } else {
            break;
        }
    }
    best
}

/// Greedy ±1 search on the six quantized endpoint components, keeping the block's
/// 4-color or 3-color mode.
fn refine_neighbors(mut best: ([u8; 8], u64), px: &Block, w: &[u32; 4], passes: u32) -> ([u8; 8], u64) {
    const MAX: [i32; 3] = [31, 63, 31];
    for _ in 0..passes {
        let mut improved = false;
        let c0 = u16::from_le_bytes([best.0[0], best.0[1]]);
        let c1 = u16::from_le_bytes([best.0[2], best.0[3]]);
        let base = [split565(c0), split565(c1)];
        let three = c0 <= c1;
        for e in 0..2 {
            for ch in 0..3 {
                for delta in [-1, 1] {
                    let mut q = base;
                    q[e][ch] = (q[e][ch] + delta).clamp(0, MAX[ch]);
                    if q == base {
                        continue;
                    }
                    let cand = if three {
                        finalize3(q[0], q[1], px, w)
                    } else {
                        finalize(q[0], q[1], px, w)
                    };
                    if cand.1 < best.1 {
                        best = cand;
                        improved = true;
                    }
                }
            }
        }
        if !improved || best.1 == 0 {
            break;
        }
    }
    best
}

/// Keeps the endpoint bytes of `raw` and re-chooses every index. Used by RDO.
pub fn reindex(raw: &[u8], px: &Block, w: &[u32; 4]) -> [u8; 8] {
    let c0 = u16::from_le_bytes([raw[0], raw[1]]);
    let c1 = u16::from_le_bytes([raw[2], raw[3]]);
    let (bits, _) = assign(px, &palette(c0, c1), w);
    to_bytes(c0, c1, bits)
}

/// Keeps the index bytes of `raw` and least-squares refits the endpoints. Used by RDO.
/// Returns `None` when the refit would flip the block out of 4-color mode.
pub fn refit(raw: &[u8], px: &Block) -> Option<[u8; 8]> {
    let c0 = u16::from_le_bytes([raw[0], raw[1]]);
    let c1 = u16::from_le_bytes([raw[2], raw[3]]);
    if c0 <= c1 {
        return None;
    }
    let bits = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let pts: Vec<V4> = px
        .iter()
        .map(|t| [t[0] as f32, t[1] as f32, t[2] as f32, 0.0])
        .collect();
    let (a, b) = refit_from_bits(bits, &pts)?;
    let (n0, n1) = (pack565(quant565(&a)), pack565(quant565(&b)));
    if n0 <= n1 {
        return None;
    }
    Some(to_bytes(n0, n1, bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_color_tables_match_full_search() {
        for (levels, six, table) in [(32u32, false, &MATCH5), (64, true, &MATCH6)] {
            let expand = |x: u32| if six { expand6(x) } else { expand5(x) };
            for v in 0..256u32 {
                let mut best = (u32::MAX, 0u32, 0u32);
                for a in 0..levels {
                    for b in 0..levels {
                        let (ea, eb) = (expand(a), expand(b));
                        let err = ((2 * ea + eb) / 3).abs_diff(v) * 1000 + ea.abs_diff(eb);
                        if err < best.0 {
                            best = (err, a, b);
                        }
                    }
                }
                assert_eq!(table[v as usize], (best.1 as u8, best.2 as u8), "levels {levels} v {v}");
            }
        }
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
        let w = [1, 1, 1, 0];
        let time = |name: &str, f: &mut dyn FnMut(&Block) -> u64| {
            let t = std::time::Instant::now();
            let mut acc = 0u64;
            for b in &blocks {
                acc = acc.wrapping_add(f(b));
            }
            println!("{name:<28} {:7.1} ms  (chk {acc})", t.elapsed().as_secs_f64() * 1e3);
        };
        let pts_of = |px: &Block| -> Vec<V4> {
            px.iter()
                .map(|t| [t[0] as f32, t[1] as f32, t[2] as f32, 0.0])
                .collect()
        };
        time("encode e0", &mut |px| encode(px, &w, 0)[0] as u64);
        time("encode e1", &mut |px| encode(px, &w, 1)[0] as u64);
        time("pts + pca/LS (e1)", &mut |px| encode_pca(px, &w, &pts_of(px), 1).1);
        time("pts + cov + axis", &mut |px| {
            let pts = pts_of(px);
            let (_, cov) = crate::fit::mean_cov(&pts, 3);
            crate::fit::principal_axis(&cov, 3)[0] as u64
        });
        for (name, three, lim) in [
            ("cluster 4c x64", false, 64),
            ("cluster 3c x32", true, 32),
            ("cluster 4c all", false, usize::MAX),
        ] {
            time(name, &mut |px| {
                let pts = pts_of(px);
                let (_, cov) = crate::fit::mean_cov(&pts, 3);
                let axis = crate::fit::principal_axis(&cov, 3);
                let sorted = sort_along(&pts, &axis).unwrap();
                cluster_fit(&sorted, &w, three, true, lim, None).map_or(0, |r| r.0[0] as u64)
            });
        }
        time("finalize x1", &mut |px| finalize([10, 20, 10], [20, 40, 20], px, &w).1);
    }

    /// Learns the cluster-fit split order: exhaustive, snapped cluster fits over the synthetic
    /// corpus (not Kodak, which is the benchmark), counting how often each split wins. Prints
    /// `src/bc1_order.rs`.
    #[test]
    #[ignore = "diagnostic report; run by hand with --nocapture"]
    fn print_split_order() {
        let mut stats = [vec![0u32; 969], vec![0u32; 153]];
        let w = [1, 1, 1, 0];
        for (_, generator) in crate::synth::set() {
            for px in generator(512).blocks() {
                {
                    if px.iter().all(|t| t[..3] == px[0][..3]) {
                        continue;
                    }
                    let pts: Vec<V4> = px
                        .iter()
                        .map(|t| [t[0] as f32, t[1] as f32, t[2] as f32, 0.0])
                        .collect();
                    let (_, cov) = crate::fit::mean_cov(&pts, 3);
                    let axis = crate::fit::principal_axis(&cov, 3);
                    let sorted = sort_along(&pts, &axis).unwrap();
                    for three in [false, true] {
                        cluster_fit(&sorted, &w, three, true, usize::MAX, Some(&mut stats[three as usize]));
                    }
                }
            }
        }
        let rank = |s: &[u32]| {
            let mut ids: Vec<usize> = (0..s.len()).collect();
            ids.sort_by_key(|&i| (std::cmp::Reverse(s[i]), i));
            ids
        };
        let arr = |name: &str, ids: &[usize], doc: &str| {
            let rows: Vec<String> = ids
                .chunks(16)
                .map(|c| {
                    format!(
                        "    {}",
                        c.iter()
                            .map(std::string::ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect();
            format!(
                "{doc}pub const {name}: [u16; {}] = [
{},
];
",
                ids.len(),
                rows.join(
                    ",
"
                )
            )
        };
        let (r4, r3) = (rank(&stats[0]), rank(&stats[1]));
        let cum = |s: &[u32], r: &[usize], k: usize| {
            let tot: u64 = s.iter().map(|&x| x as u64).sum();
            r.iter().take(k).map(|&i| s[i] as u64).sum::<u64>() as f64 / tot.max(1) as f64
        };
        println!("//! Cluster-fit split order for BC1 (see `bc1::cluster_fit`), learned on the synthetic corpus");
        println!("//! (`synth::set()` at 512²): exhaustive snapped cluster fits, splits ranked by how often they win.");
        println!(
            "//! Coverage of the winners: 4-color top 64 {:.1}%, top 256 {:.1}%; 3-color top 32 {:.1}%.",
            100.0 * cum(&stats[0], &r4, 64),
            100.0 * cum(&stats[0], &r4, 256),
            100.0 * cum(&stats[1], &r3, 32)
        );
        println!(
            "//! Regenerate with `cargo test --release --lib bc1::tests::print_split_order -- --ignored --nocapture`.
"
        );
        print!(
            "{}",
            arr(
                "ORDER4",
                &r4,
                "/// 4-color splits of 16 texels by canonical id, most frequent winner first.
"
            )
        );
        println!();
        print!(
            "{}",
            arr(
                "ORDER3",
                &r3,
                "/// 3-color splits of 16 texels by canonical id, most frequent winner first.
"
            )
        );
    }

    #[test]
    fn solid_colors_are_near_exact() {
        let w = [1, 1, 1, 0];
        for v in [0u8, 1, 37, 128, 200, 254, 255] {
            let px = [[v, 255 - v, v / 2, 255]; 16];
            let raw = encode(&px, &w, 2);
            let mut out = [[0u8; 4]; 16];
            decode(&raw, &mut out);
            for t in out {
                for c in 0..3 {
                    assert!(t[c].abs_diff(px[0][c]) <= 2, "v={v} got {t:?}");
                }
            }
        }
    }

    #[test]
    fn gradient_beats_uniform_quantization() {
        // 16 evenly spaced colors on a line through 4 palette entries: uniform 4-level
        // quantization of R (0..240) and B gives ~533 per channel per texel, G ~133.
        let mut px = [[0u8; 4]; 16];
        for (i, t) in px.iter_mut().enumerate() {
            let v = (i * 16) as u8;
            *t = [v, v / 2, 255 - v, 255];
        }
        let raw = encode(&px, &[1, 1, 1, 0], 2);
        let mut out = [[0u8; 4]; 16];
        decode(&raw, &mut out);
        let sse = crate::surface::block_sse(&px, &out, &[1, 1, 1, 0]);
        assert!(sse < 16 * 1200, "sse {sse}");
    }
}
