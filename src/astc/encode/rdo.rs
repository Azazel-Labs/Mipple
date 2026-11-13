//! Rate-distortion optimization for ASTC: the LZ-aware pass of [`crate::rdo`], for blocks whose
//! footprints are not the 4x4 source blocks that pass scores against.
//!
//! Each block, in raster order, picks among its base encoding and candidates built from earlier
//! blocks of its chunk, minimizing `J = smooth · SSE / 16 + λ · bits` (the same scale as the other
//! formats, whatever the footprint), with the bits from [`LzModel`]. An ASTC block keeps its mode,
//! partitioning and endpoints in its low bytes and its weights in its high bytes, so the candidates
//! are:
//!
//! - a whole earlier block, copied;
//! - an earlier block's mode, partitioning and endpoints, with weights chosen for this block's
//!   texels (its low bytes match);
//! - an earlier block's mode, partitioning and weights, with endpoints refitted to this block's
//!   texels (its high bytes match).
//!
//! The re-chosen half is polished with the other half held. Like the other formats, a chunk that
//! real Zstd does not shrink goes back to its base encoding.

use super::{
    Footprint, Layout, MAX_TEXELS, NO_CCS, Texels, infill, polish_parts, quantize, read_block, refit, round_pos,
    sym_endpoints, tables,
};
use crate::astc::{COLOR_UNQUANT, Symbolic, WEIGHT_UNQUANT, decode_block, partition_of};
use crate::format::Format;
use crate::rdo::{LzModel, LzState, RdoParams, RdoStats, VERIFY_QUICK, bit_budget};
use crate::surface::{Rgba, Surface};
use rayon::prelude::*;
use std::collections::HashSet;

/// How far back the reweight and refit candidates look (whole copies look `max_back` far). On Kodak,
/// 256 is 4.8-6.6% smaller at equal PSNR than 32 for 2.4x the pass's time, and 512 another 0.3-0.7%
/// for 1.8x more (docs/DEADENDS.md).
const REFIT_BACK: usize = 256;
/// Polish passes over the re-chosen half of a candidate.
const PASSES: usize = 1;

/// A block already chosen: its bytes and, when the candidates can build on it, its fields, labels
/// and grid.
struct Prev {
    bytes: [u8; 16],
    sym: Option<Box<(Symbolic, Layout, usize)>>,
}

/// One block's texels for the error kernels, all four channels counted (a copied block may carry
/// alpha the source lacks, or lack alpha the source has).
fn texels(px: &[Rgba]) -> Texels<'_> {
    let mut t = Texels {
        px,
        f: [[0.0; 4]; MAX_TEXELS],
        xl: [[0; MAX_TEXELS]; 4],
        n: px.len(),
        ch: 4,
    };
    for (i, p) in px.iter().enumerate() {
        t.f[i] = p.map(f32::from);
        for c in 0..4 {
            t.xl[c][i] = p[c] as i16;
        }
    }
    t
}

/// The labels of `sym`'s partitioning (the only parts of a layout polish and refit read).
fn layout_of(fp: &Footprint, sym: &Symbolic) -> Layout {
    let n = fp.texels();
    let mut lay = Layout {
        count: sym.partitions,
        seed: sym.seed,
        labels: [0; MAX_TEXELS],
        lab16: [-1; MAX_TEXELS],
        ccs: if sym.mode.dual { sym.ccs as usize } else { NO_CCS },
        u: [0.0; MAX_TEXELS],
        scale: [0.0; MAX_TEXELS],
        u2: [0.0; MAX_TEXELS],
        scale2: 0.0,
        orth: 0.0,
    };
    for i in 0..n {
        let l = partition_of(
            sym.seed as u32,
            i as u32 % fp.bw,
            i as u32 / fp.bw,
            sym.partitions as u32,
            n as u32,
        );
        lay.labels[i] = l;
        lay.lab16[i] = l as i16;
    }
    lay
}

/// `bytes`' fields, labels and grid, if the candidates can build on them: up to three partitions,
/// all RGB (CEM 8) or all RGBA (CEM 12), as the encoder writes.
fn prev_sym(fp: &Footprint, bytes: [u8; 16]) -> Option<Box<(Symbolic, Layout, usize)>> {
    let sym = Symbolic::read(u128::from_le_bytes(bytes), fp.bw, fp.bh)?;
    let p = sym.partitions as usize;
    if p > 3 || !matches!(sym.cems[0], 8 | 12) || sym.cems[..p].iter().any(|&c| c != sym.cems[0]) {
        return None;
    }
    let gi = fp
        .grids
        .iter()
        .position(|g| g.gx == sym.mode.gx && g.gy == sym.mode.gy)?;
    let lay = layout_of(fp, &sym);
    Some(Box::new((sym, lay, gi)))
}

/// `sym` with weights chosen for `t`: each texel projected onto its partition's endpoint segment,
/// decimated to the grid, quantized and polished with the endpoints held. Returns it and its error,
/// or `None` if no weights for these endpoints can bring the error down to `max_err`.
fn reweight(
    fp: &Footprint,
    t: &Texels,
    lay: &Layout,
    gi: usize,
    sym: &Symbolic,
    max_err: f32,
) -> Option<(Symbolic, u32)> {
    let grid = &fp.grids[gi];
    let n = t.n;
    let planes = 1 + sym.mode.dual as usize;
    let ccs = lay.ccs;
    let mut eps = [([0i32; 4], [0i32; 4]); 3];
    let mut inv = [0f32; 3];
    for p in 0..lay.count as usize {
        eps[p] = sym_endpoints(sym, p);
        let (e0, e1) = eps[p];
        let len2: i32 = (0..4).filter(|&c| c != ccs).map(|c| (e1[c] - e0[c]).pow(2)).sum();
        inv[p] = if len2 > 0 { 64.0 / len2 as f32 } else { 0.0 };
    }
    // A texel decodes to its weight's point on the segment rounded per channel, within 0.5 of it
    // in each channel: so within `slack` of the segment over the first plane's channels, and its
    // error is at least its distance from the segment less that.
    let slack = if ccs < NO_CCS { 0.75f32.sqrt() } else { 1.0 };
    let mut lb = 0f32;
    let mut u = [0f32; MAX_TEXELS];
    let mut u2 = [0f32; MAX_TEXELS];
    for i in 0..n {
        let p = lay.labels[i] as usize;
        let (e0, e1) = eps[p];
        let x = t.px[i].map(i32::from);
        let dot: i32 = (0..4)
            .filter(|&c| c != ccs)
            .map(|c| (x[c] - e0[c]) * (e1[c] - e0[c]))
            .sum();
        u[i] = (dot as f32 * inv[p]).clamp(0.0, 64.0);
        let mut d2 = 0f32;
        for c in (0..4).filter(|&c| c != ccs) {
            d2 += (x[c] as f32 - e0[c] as f32 - u[i] * (1.0 / 64.0) * (e1[c] - e0[c]) as f32).powi(2);
        }
        lb += (d2.sqrt() - slack).max(0.0).powi(2);
        if ccs < NO_CCS {
            let d = e1[ccs] - e0[ccs];
            u2[i] = if d != 0 {
                (64.0 * (x[ccs] - e0[ccs]) as f32 / d as f32).clamp(0.0, 64.0)
            } else {
                0.0
            };
            let r = x[ccs] as f32 - e0[ccs] as f32 - u2[i] * (1.0 / 64.0) * d as f32;
            lb += (r.abs() - 0.5).max(0.0).powi(2);
        }
        if lb > max_err {
            return None;
        }
    }
    let mut out = sym.clone();
    quantize(
        grid,
        sym.mode.quant as usize,
        planes,
        &u[..n],
        &u2[..n],
        &mut out.weights,
    );
    let err = polish_parts::<true, false>(t, lay, grid, &mut out, PASSES);
    Some((out, err))
}

/// `sym` with endpoints refitted to `t` for its weights, quantized and polished with the weights
/// held. Returns it and its error, or `None` if no endpoints for these weights can bring the error
/// down to `max_err`.
fn refit_to(
    fp: &Footprint,
    t: &Texels,
    lay: &Layout,
    gi: usize,
    sym: &Symbolic,
    max_err: f32,
) -> Option<(Symbolic, u32)> {
    let grid = &fp.grids[gi];
    let tb = tables();
    let wu = &WEIGHT_UNQUANT[sym.mode.quant as usize];
    let planes = 1 + sym.mode.dual as usize;
    let kk = grid.rev.len() * planes;
    let mut g = [0u8; 64];
    for j in 0..kk {
        g[j] = wu[sym.weights[j] as usize];
    }
    let mut w = [[0u8; MAX_TEXELS]; 2];
    for (pl, wp) in w.iter_mut().enumerate().take(planes) {
        infill(grid, &g, planes, pl, wp);
    }
    let cq = sym.color_quant as usize;
    let (cqt, cu) = (&tb.cq[cq], &COLOR_UNQUANT[cq]);
    let cem = sym.cems[0];
    let nv = if cem == 12 { 4 } else { 3 };
    // With the weights held, no endpoints fit better than each channel's least-squares line (not
    // clamped to the color range), and decoding moves each texel at most 0.5 per channel from its
    // line point, so at most sqrt(n) over the block.
    let mut sums = [[[0f64; 6]; 4]; 3];
    for i in 0..t.n {
        let s = &mut sums[lay.labels[i] as usize];
        for c in 0..4 {
            let (x, wi) = (t.px[i][c] as f64, if c == lay.ccs { w[1][i] } else { w[0][i] } as f64);
            let a = &mut s[c];
            a[0] += 1.0;
            a[1] += wi;
            a[2] += wi * wi;
            a[3] += x;
            a[4] += x * wi;
            a[5] += x * x;
        }
    }
    let mut res = 0f64;
    for s in &sums[..lay.count as usize] {
        for (c, &[n, sw, sww, sx, sxw, sxx]) in s.iter().enumerate() {
            res += if cem == 8 && c == 3 {
                // RGB endpoints decode alpha to 255.
                sxx - 510.0 * sx + 65025.0 * n
            } else {
                let det = n * sww - sw * sw;
                if det > 1e-6 {
                    sxx - (sww * sx * sx - 2.0 * sw * sx * sxw + n * sxw * sxw) / det
                } else if n > 0.0 {
                    sxx - sx * sx / n
                } else {
                    0.0
                }
            }
            .max(0.0);
        }
    }
    if (res.sqrt() - (t.n as f64).sqrt()).max(0.0).powi(2) > max_err as f64 {
        return None;
    }
    let mut out = sym.clone();
    let fits = refit(t, &w[0], &w[1], lay.ccs, &lay.lab16, lay.count);
    for (p, &(f0, f1)) in fits.iter().enumerate().take(lay.count as usize) {
        let mut c0: [u8; 4] = std::array::from_fn(|ch| cqt[round_pos(f0[ch])]);
        let mut c1: [u8; 4] = std::array::from_fn(|ch| cqt[round_pos(f1[ch])]);
        let sum = |c: &[u8; 4]| (0..3).map(|ch| cu[c[ch] as usize] as i32).sum::<i32>();
        // A lower second sum would decode with blue contraction; swapping would invert the
        // weights this candidate keeps, so take the partition's mean color instead.
        if sum(&c1) < sum(&c0) {
            c0 = std::array::from_fn(|ch| cqt[round_pos((f0[ch] + f1[ch]) * 0.5)]);
            c1 = c0;
        }
        for ch in 0..nv {
            out.colors[2 * nv * p + 2 * ch] = c0[ch];
            out.colors[2 * nv * p + 2 * ch + 1] = c1[ch];
        }
    }
    let err = polish_parts::<false, true>(t, lay, grid, &mut out, PASSES);
    Some((out, err))
}

/// Squared error over all four channels.
fn sse(a: &[Rgba], b: &[Rgba]) -> u64 {
    let mut s = 0u32;
    for (x, y) in a.iter().zip(b) {
        for c in 0..4 {
            s += (x[c] as i32 - y[c] as i32).pow(2) as u32;
        }
    }
    s as u64
}

/// [`crate::rdo`]'s distortion weight for smooth blocks, over this footprint's texels.
fn smooth_factor(px: &[Rgba], p: &RdoParams) -> f64 {
    let n = px.len() as f32;
    let mut max_std = 0f32;
    for c in 0..4 {
        let mean = px.iter().map(|t| t[c] as f32).sum::<f32>() / n;
        let var = px.iter().map(|t| (t[c] as f32 - mean).powi(2)).sum::<f32>() / n;
        max_std = max_std.max(var.sqrt());
    }
    let k2 = p.smooth_std * p.smooth_std;
    (1.0 + (p.smooth_scale - 1.0) * k2 / (k2 + max_std * max_std)) as f64
}

#[expect(clippy::too_many_arguments, reason = "one chunk's surface, position and settings")]
fn optimize_chunk(
    format: Format,
    fp: &Footprint,
    s: &Surface,
    nx: usize,
    first: usize,
    data: &mut [u8],
    srgb: bool,
    p: &RdoParams,
) -> RdoStats {
    let count = data.len() / 16;
    let n = fp.texels();
    let (bw, bh) = (fp.bw, fp.bh);
    let window = p.window_for(format);
    let max_back = p.max_back_for(format).min(window / 16);
    let mut model = LzModel::new(window, p.chain_depth);
    model.set_literals(data);
    let lambda = p.lambda as f64;
    let ratio2 = (p.max_rms_ratio as f64).powi(2);
    let floor = (n * 4) as f64 * (p.min_rms as f64).powi(2);
    let original = p.verify_zstd_level.map(|_| data.to_vec());
    let mut stats = RdoStats {
        blocks: count,
        ..Default::default()
    };
    let mut prevs: Vec<Prev> = Vec::with_capacity(count);
    let mut decs: Vec<Rgba> = vec![[0; 4]; count * n];
    let mut px = vec![[0u8; 4]; n];
    let mut dec = [[0u8; 4]; MAX_TEXELS];
    let mut seen: HashSet<u128> = HashSet::new();
    let mut heads: HashSet<u128> = HashSet::new();
    let mut tails: HashSet<u128> = HashSet::new();
    for i in 0..count {
        let bi = first + i;
        read_block(s, bw, bh, (bi % nx) as u32, (bi / nx) as u32, &mut px);
        let t = texels(&px);
        let smooth = smooth_factor(&px, p);
        let score = |err: u64, bits: f32| smooth * err as f64 / 16.0 + lambda * bits as f64;
        let base: [u8; 16] = data[i * 16..i * 16 + 16].try_into().unwrap();
        decode_block(&base, bw, bh, srgb, &mut dec[..n]);
        let base_err = sse(&px, &dec[..n]);
        let (base_bits, base_rep) = model.cost(&base);
        let min_bits = lambda * model.min_cost(16) as f64;
        // Error budget: a multiple of the base error, plus a floor so near-lossless blocks still
        // have room to move.
        let limit = (base_err as f64 * ratio2 + floor) as u64;
        let mut best: (f64, [u8; 16], f32, LzState) = (score(base_err, base_bits), base, base_bits, base_rep);
        let consider = |c: [u8; 16], err: u64, best: &mut (f64, [u8; 16], f32, LzState)| {
            if err > limit {
                return;
            }
            let dist = smooth * err as f64 / 16.0;
            if dist >= best.0 || (dist + min_bits >= best.0 && !model.may_continue(&c)) {
                return;
            }
            let Some((bits, rep)) = model.cost_until(&c, bit_budget(best.0, dist, lambda)) else {
                return;
            };
            let j = dist + lambda * bits as f64;
            if j < best.0 {
                *best = (j, c, bits, rep);
            }
        };
        seen.clear();
        seen.insert(u128::from_le_bytes(base));
        for d in 1..=max_back.min(i) {
            let j = i - d;
            if seen.insert(u128::from_le_bytes(prevs[j].bytes)) {
                consider(prevs[j].bytes, sse(&px, &decs[j * n..j * n + n]), &mut best);
            }
        }
        heads.clear();
        tails.clear();
        for d in 1..=REFIT_BACK.min(i) {
            let j = i - d;
            let Some(ps) = &prevs[j].sym else { continue };
            let (sym, lay, gi) = &**ps;
            let v = u128::from_le_bytes(prevs[j].bytes);
            let below = 128 - sym.mode.weight_bits();
            let config = if sym.partitions == 1 { 17 } else { 29 };
            // Candidates from the same fields come out the same: each head and each set of weights
            // (with its mode, partitioning and second-plane channel) once.
            // A candidate must beat the best on distortion alone (with a little room for rounding
            // in the bounds).
            let cap = (limit as f64).min(best.0 * 16.0 / smooth) as f32 * 1.001 + 1.0;
            let mut cands = [None, None];
            if heads.insert(v & ((1u128 << below) - 1)) {
                cands[0] = reweight(fp, &t, lay, *gi, sym, cap);
            }
            let tail = (v & ((1u128 << config) - 1)) | (sym.ccs as u128) << config | (v >> below) << below;
            if tails.insert(tail) {
                cands[1] = refit_to(fp, &t, lay, *gi, sym, cap);
            }
            for (c, err) in cands.into_iter().flatten() {
                let Some(b) = c.write() else { continue };
                if seen.insert(b) {
                    let b = b.to_le_bytes();
                    debug_assert_eq!(err as u64, {
                        let mut d = [[0u8; 4]; MAX_TEXELS];
                        decode_block(&b, bw, bh, srgb, &mut d[..n]);
                        sse(&px, &d[..n])
                    });
                    consider(b, err as u64, &mut best);
                }
            }
        }
        stats.est_bits_before += base_bits as f64;
        stats.est_bits_after += best.2 as f64;
        if best.1 != base {
            stats.changed += 1;
        }
        data[i * 16..i * 16 + 16].copy_from_slice(&best.1);
        decode_block(&best.1, bw, bh, srgb, &mut decs[i * n..i * n + n]);
        prevs.push(Prev {
            bytes: best.1,
            sym: prev_sym(fp, best.1),
        });
        model.push(&best.1, best.3);
    }
    // Safety net: the rate model is an estimate. If real Zstd says RDO did not help this chunk,
    // keep the base encoding.
    if let (Some(level), Some(original)) = (p.verify_zstd_level, original) {
        let size = |b: &[u8], level: i32| zstd::bulk::compress(b, level).map_or(usize::MAX, |v| v.len());
        let quick = level > 1 && size(data, 1) as f64 <= VERIFY_QUICK * size(&original, 1) as f64;
        if !quick && size(data, level) >= size(&original, level) {
            data.copy_from_slice(&original);
            stats.changed = 0;
            stats.est_bits_after = stats.est_bits_before;
            stats.reverted_chunks = 1;
        }
    }
    stats
}

/// Runs RDO in place over one level's ASTC blocks (`data`, raster order, as
/// [`super::encode_surface`] writes them) of surface `s`.
pub fn optimize(format: Format, s: &Surface, srgb: bool, data: &mut [u8], p: &RdoParams) -> RdoStats {
    assert!(format.is_astc(), "ASTC RDO for {format}");
    let blocks = data.len() / 16;
    if !p.enabled() || blocks == 0 {
        return RdoStats {
            blocks,
            ..Default::default()
        };
    }
    let (bw, bh) = format.block_dims();
    let fp = Footprint::new(bw, bh);
    let nx = s.width.div_ceil(bw) as usize;
    let chunk = (p.chunk_bytes_for(format) / 16).max(1);
    data.par_chunks_mut(chunk * 16)
        .enumerate()
        .map(|(k, d)| optimize_chunk(format, &fp, s, nx, k * chunk, d, srgb, p))
        .reduce(RdoStats::default, RdoStats::merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every candidate's error matches its real decode (the debug assertion in [`optimize_chunk`]),
    /// and RDO moves blocks and lowers the estimated rate, at every footprint, with and without alpha.
    #[test]
    fn candidates_score_their_decode() {
        for s in [crate::synth::bricks(64), crate::synth::cutout(64)] {
            for format in [Format::Astc4x4, Format::Astc6x6, Format::Astc8x8] {
                let (bw, bh) = format.block_dims();
                let mut data = super::super::encode_surface(&s, bw, bh, true, 0);
                let p = RdoParams {
                    lambda: 4.0,
                    verify_zstd_level: None,
                    ..Default::default()
                };
                let st = optimize(format, &s, true, &mut data, &p);
                assert!(st.changed > 0, "{format}");
                assert!(st.est_bits_after < st.est_bits_before, "{format}");
            }
        }
    }
}
