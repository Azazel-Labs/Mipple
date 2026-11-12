//! Encoding ASTC LDR blocks from RGBA8.
//!
//! Each block is fitted with a line in color space (the principal axis), its texels' positions on
//! that line become ideal weights, and those are decimated to every weight grid the footprint
//! allows. Each (grid, weight range) pair is scored by an error estimate, and the best few are
//! encoded exactly: weights quantized, endpoints refitted to them and quantized, the result decoded
//! and measured.

use super::decode::{ERROR, lerp1};
use super::{BlockMode, COLOR_UNQUANT, RANGES, Symbolic, WEIGHT_RANGES, WEIGHT_UNQUANT, block_mode, color_range};
use crate::surface::{Rgba, Surface};

/// The largest footprint (12x12) in texels.
pub const MAX_TEXELS: usize = 144;

/// One texel's bilinear infill: four grid points and their weights in sixteenths.
#[derive(Clone, Copy, Default)]
struct Tap {
    idx: [u8; 4],
    w: [u8; 4],
}

/// A weight grid of a footprint, with the infill both ways and the single-plane modes using it.
struct Grid {
    gx: u8,
    gy: u8,
    identity: bool,
    taps: Vec<Tap>,
    /// For each grid point, the texels it reaches and the weight (sixteenths).
    rev: Vec<Vec<(u8, u8)>>,
    /// The taps' weights as fractions.
    tapf: Vec<[f32; 4]>,
    /// `rev` flattened, grid point `j` at `revf[rev_at[j]..rev_at[j + 1]]`, each weight divided
    /// by the grid point's total.
    revf: Vec<(u8, f32)>,
    rev_at: Vec<u16>,
    /// Single-plane block modes on this grid, with the endpoint range for one to three
    /// partitions of RGB (CEM 8) and RGBA (CEM 12); `u8::MAX` where the mode leaves too few bits.
    modes: Vec<(BlockMode, [[u8; 2]; 3])>,
    /// Dual-plane block modes on this grid, with the endpoint range for one partition of RGB and
    /// RGBA.
    dual: Vec<(BlockMode, [u8; 2])>,
    /// Per mode (then per dual mode): the expected squared weight error per unit of color span,
    /// from quantizing the weights (the part of that noise the infill passes on: the mean over
    /// texels of the sum of squared tap weights) and from rounding the infill.
    wnoise: Vec<f32>,
    wnoise_dual: Vec<f32>,
    /// The least of `wnoise` and of `wnoise_dual`.
    wnoise_min: f32,
    wnoise_dual_min: f32,
}

/// Expected squared endpoint error per channel and texel for each color range: a uniform error
/// over one quantization step, reduced by interpolation (about 1/18 of the step squared).
static CNOISE: std::sync::LazyLock<[f32; 21]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|q| {
        let step = 255.0 / (RANGES[q] as f32 - 1.0);
        step * step / 18.0
    })
});

/// A partition pattern: its seed and each texel's partition.
struct Pattern {
    seed: u16,
    labels: Vec<u8>,
}

/// The weight grids and partition patterns of one footprint, built once per surface.
pub struct Footprint {
    pub bw: u32,
    pub bh: u32,
    grids: Vec<Grid>,
    /// Distinct two- and three-partition patterns with no empty partition.
    patterns: [Vec<Pattern>; 2],
    /// The patterns' texel masks, packed: per pattern, `nw` words for partition 0, then (three
    /// partitions) `nw` for partition 1.
    pmasks: [Vec<u64>; 2],
    /// Mask words per texel mask.
    nw: usize,
    /// Orthonormal DCT-II bases across and down: `dct_x[x * bw + f]` (transposed),
    /// `dct_y[f * bh + y]`.
    dct_x: Vec<f32>,
    dct_y: Vec<f32>,
}

/// The orthonormal DCT-II basis of size `n`, frequency-major.
fn dct_basis(n: u32) -> Vec<f32> {
    let n = n as usize;
    let mut b = vec![0f32; n * n];
    for f in 0..n {
        let c = if f == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        for x in 0..n {
            b[f * n + x] = (c * (std::f64::consts::PI * (2 * x + 1) as f64 * f as f64 / (2 * n) as f64).cos()) as f32;
        }
    }
    b
}

/// Prefix sums of the DCT energy of a texel signal: `[fy][fx]` holds the energy below frequency
/// (fx, fy), so the energy a gx × gy grid cannot hold is `[bh][bw] - [gy][gx]`.
type Spectrum = [[f32; 13]; 13];

fn spectrum(fp: &Footprint, v: &[f32]) -> Spectrum {
    let (bw, bh) = (fp.bw as usize, fp.bh as usize);
    let mut r = [0f32; MAX_TEXELS];
    for y in 0..bh {
        let ry = &mut r[y * bw..(y + 1) * bw];
        for x in 0..bw {
            let vx = v[y * bw + x];
            for (o, &b) in ry.iter_mut().zip(&fp.dct_x[x * bw..(x + 1) * bw]) {
                *o += vx * b;
            }
        }
    }
    let mut p = [[0f32; 13]; 13];
    for fy in 0..bh {
        let mut c = [0f32; 12];
        for y in 0..bh {
            let d = fp.dct_y[fy * bh + y];
            for (o, &x) in c.iter_mut().zip(&r[y * bw..(y + 1) * bw]) {
                *o += d * x;
            }
        }
        let mut row = 0.0;
        for fx in 0..bw {
            row += c[fx] * c[fx];
            p[fy + 1][fx + 1] = p[fy][fx + 1] + row;
        }
    }
    p
}

impl Footprint {
    pub fn new(bw: u32, bh: u32) -> Self {
        assert!((4..=12).contains(&bw) && (4..=12).contains(&bh), "footprint {bw}x{bh}");
        let n = (bw * bh) as usize;
        let mut grids = Vec::new();
        for gy in 2..=bh {
            for gx in 2..=bw {
                let mut modes: Vec<(BlockMode, [[u8; 2]; 3])> = Vec::new();
                let mut dual: Vec<(BlockMode, [u8; 2])> = Vec::new();
                for m in 0..2048 {
                    let Some(bm) = block_mode(m) else { continue };
                    if bm.gx as u32 != gx || bm.gy as u32 != gy {
                        continue;
                    }
                    let cq = |p: u8, cem: u8| color_range(bm, p, &[cem; 4]).unwrap_or(u8::MAX);
                    if bm.dual {
                        if !dual.iter().any(|e| e.0 == bm) && cq(1, 8) != u8::MAX {
                            dual.push((bm, [cq(1, 8), cq(1, 12)]));
                        }
                    } else if !modes.iter().any(|e| e.0 == bm) {
                        modes.push((bm, std::array::from_fn(|p| [cq(p as u8 + 1, 8), cq(p as u8 + 1, 12)])));
                    }
                }
                if modes.is_empty() {
                    continue;
                }
                modes.sort_by_key(|e| e.0.quant);
                dual.sort_by_key(|e| e.0.quant);
                let identity = gx == bw && gy == bh;
                let mut taps = vec![Tap::default(); n];
                let mut rev = vec![Vec::new(); (gx * gy) as usize];
                let ds = (1024 + bw / 2) / (bw - 1);
                let dt = (1024 + bh / 2) / (bh - 1);
                for t in 0..bh {
                    for s in 0..bw {
                        let i = (t * bw + s) as usize;
                        let tap = if identity {
                            Tap {
                                idx: [i as u8; 4],
                                w: [16, 0, 0, 0],
                            }
                        } else {
                            let gs = (ds * s * (gx - 1) + 32) >> 6;
                            let gt = (dt * t * (gy - 1) + 32) >> 6;
                            let (js, fs, jt, ft) = (gs >> 4, gs & 15, gt >> 4, gt & 15);
                            let w11 = (fs * ft + 8) >> 4;
                            let ws = [16 - fs - ft + w11, fs - w11, ft - w11, w11];
                            let pts = [(js, jt), (js + 1, jt), (js, jt + 1), (js + 1, jt + 1)];
                            let mut tap = Tap::default();
                            for k in 0..4 {
                                let (x, y) = pts[k];
                                debug_assert!(ws[k] == 0 || (x < gx && y < gy));
                                tap.idx[k] = (y.min(gy - 1) * gx + x.min(gx - 1)) as u8;
                                tap.w[k] = ws[k] as u8;
                            }
                            tap
                        };
                        for k in 0..4 {
                            if tap.w[k] > 0 {
                                rev[tap.idx[k] as usize].push((i as u8, tap.w[k]));
                            }
                        }
                        taps[i] = tap;
                    }
                }
                let tapf = taps.iter().map(|t| t.w.map(|w| w as f32 / 16.0)).collect();
                let mut revf = Vec::new();
                let mut rev_at = vec![0u16];
                for r in &rev {
                    let inv = 1.0 / r.iter().map(|e| e.1 as f32).sum::<f32>().max(1.0);
                    revf.extend(r.iter().map(|&(i, w)| (i, w as f32 * inv)));
                    rev_at.push(revf.len() as u16);
                }
                let noise = taps
                    .iter()
                    .map(|t| t.w.iter().map(|&w| (w as f32 / 16.0).powi(2)).sum::<f32>())
                    .sum::<f32>()
                    / n as f32;
                let round = if identity { 0.0 } else { 1.0 / 12.0 };
                let wn = |bm: &BlockMode| {
                    let wstep = 64.0 / (RANGES[bm.quant as usize] as f32 - 1.0);
                    noise * wstep * wstep / 12.0 + round
                };
                let wnoise: Vec<f32> = modes.iter().map(|m| wn(&m.0)).collect();
                let wnoise_dual: Vec<f32> = dual.iter().map(|m| wn(&m.0)).collect();
                let wnoise_min = wnoise.iter().fold(f32::MAX, |a, &b| a.min(b));
                let wnoise_dual_min = wnoise_dual.iter().fold(f32::MAX, |a, &b| a.min(b));
                grids.push(Grid {
                    gx: gx as u8,
                    gy: gy as u8,
                    identity,
                    taps,
                    rev,
                    tapf,
                    revf,
                    rev_at,
                    modes,
                    dual,
                    wnoise,
                    wnoise_dual,
                    wnoise_min,
                    wnoise_dual_min,
                });
            }
        }
        let patterns = std::array::from_fn(|pi| {
            let count = pi as u32 + 2;
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for seed in 0..1024u32 {
                let labels: Vec<u8> = (0..n as u32)
                    .map(|i| super::partition_of(seed, i % bw, i / bw, count, n as u32))
                    .collect();
                // Relabel in order of first appearance, so patterns that only permute labels match.
                let mut map = [u8::MAX; 4];
                let mut next = 0;
                let canon: Vec<u8> = labels
                    .iter()
                    .map(|&l| {
                        if map[l as usize] == u8::MAX {
                            map[l as usize] = next;
                            next += 1;
                        }
                        map[l as usize]
                    })
                    .collect();
                if next as u32 != count || !seen.insert(canon) {
                    continue;
                }
                out.push(Pattern {
                    seed: seed as u16,
                    labels,
                });
            }
            out
        });
        let nw = n.div_ceil(64);
        let pmasks = std::array::from_fn(|pi| {
            let pats: &Vec<Pattern> = &patterns[pi];
            let mut m = vec![0u64; pats.len() * nw * (pi + 1)];
            for (j, pat) in pats.iter().enumerate() {
                for (i, &l) in pat.labels.iter().enumerate() {
                    if (l as usize) <= pi {
                        m[(j * (pi + 1) + l as usize) * nw + i / 64] |= 1 << (i % 64);
                    }
                }
            }
            m
        });
        Footprint {
            bw,
            bh,
            grids,
            patterns,
            pmasks,
            nw,
            dct_x: {
                let b = dct_basis(bw);
                let w = bw as usize;
                (0..w * w).map(|i| b[(i % w) * w + i / w]).collect()
            },
            dct_y: dct_basis(bh),
        }
    }

    fn texels(&self) -> usize {
        (self.bw * self.bh) as usize
    }
}

/// Nearest-level lookups for quantizing weights and endpoint values.
struct Tables {
    /// Weight range `q`: the code nearest each value 0..=64.
    wq: [[u8; 65]; WEIGHT_RANGES],
    /// Weight range `q`: the code whose value is 64 minus this code's.
    winv: [[u8; 32]; WEIGHT_RANGES],
    /// Endpoint range `q`: the code nearest each value 0..=255.
    cq: [[u8; 256]; 21],
    /// Weight range `q`: each code's neighbors by value, below and above (itself at the ends).
    wnb: [[[u8; 32]; 2]; WEIGHT_RANGES],
    /// Endpoint range `q`: each code's neighbors by value, below and above.
    cnb: [[[u8; 256]; 2]; 21],
}

/// Each code's neighbors by value among `u`'s levels: below and above, itself at the ends.
fn neighbors<const N: usize>(u: &[u8]) -> [[u8; N]; 2] {
    let mut order: Vec<usize> = (0..u.len()).collect();
    order.sort_by_key(|&c| u[c]);
    let mut nb = [[0u8; N]; 2];
    for (r, &c) in order.iter().enumerate() {
        nb[0][c] = order[r.saturating_sub(1)] as u8;
        nb[1][c] = order[(r + 1).min(order.len() - 1)] as u8;
    }
    nb
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Box<Tables>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new(Tables {
            wq: [[0; 65]; WEIGHT_RANGES],
            winv: [[0; 32]; WEIGHT_RANGES],
            cq: [[0; 256]; 21],
            wnb: [[[0; 32]; 2]; WEIGHT_RANGES],
            cnb: [[[0; 256]; 2]; 21],
        });
        for q in 0..WEIGHT_RANGES {
            let n = RANGES[q] as usize;
            let u = &WEIGHT_UNQUANT[q][..n];
            for v in 0..=64i32 {
                t.wq[q][v as usize] = (0..n).min_by_key(|&c| ((u[c] as i32 - v).abs(), c)).unwrap() as u8;
            }
            for c in 0..n {
                t.winv[q][c] = (0..n)
                    .find(|&d| u[d] as i32 == 64 - u[c] as i32)
                    .expect("symmetric levels") as u8;
            }
            t.wnb[q] = neighbors(u);
        }
        for q in 0..21 {
            let n = RANGES[q] as usize;
            let u = &COLOR_UNQUANT[q][..n];
            for v in 0..=255i32 {
                t.cq[q][v as usize] = (0..n).min_by_key(|&c| ((u[c] as i32 - v).abs(), c)).unwrap() as u8;
            }
            t.cnb[q] = neighbors(u);
        }
        t
    })
}

/// One block's texels as floats, with the channel count that matters (3, or 4 with alpha).
struct Texels<'a> {
    px: &'a [Rgba],
    f: [[f32; 4]; MAX_TEXELS],
    n: usize,
    ch: usize,
    srgb: bool,
}

/// No channel on a second weight plane.
const NO_CCS: usize = 4;

impl Texels<'_> {
    /// The decoded block's squared error for each partition's endpoints (0..=255), texel labels
    /// and texel weights (0..=64: `w`, and `w2` in channel `ccs`), stopping once past `limit`.
    fn error(
        &self,
        eps: &[([i32; 4], [i32; 4]); 3],
        labels: &[u8],
        w: &[u8],
        w2: &[u8],
        ccs: usize,
        limit: u32,
    ) -> u32 {
        let mut err = 0u32;
        for i in 0..self.n {
            let (e0, e1) = eps[labels[i] as usize];
            for c in 0..self.ch {
                let wc = if c == ccs { w2[i] } else { w[i] };
                err += self.channel_error(i, c, e0[c], e1[c], wc);
            }
            if err > limit {
                return err;
            }
        }
        err
    }

    /// Texel `i`'s squared error in channel `c` for endpoint values `e0`, `e1` and weight `w`.
    #[inline]
    fn channel_error(&self, i: usize, c: usize, e0: i32, e1: i32, w: u8) -> u32 {
        let x = lerp1(e0, e1, w as u32, self.srgb && c < 3) - self.px[i][c] as i32;
        (x * x) as u32
    }
}

/// One texel's weight from plane `pl` of grid weights `g` (0..=64, `planes` interleaved), as the
/// decoder infills it.
#[inline]
fn tap_weight(tap: &Tap, g: &[u8], planes: usize, pl: usize) -> u8 {
    let mut s = 8u32;
    for k in 0..4 {
        s += tap.w[k] as u32 * g[planes * tap.idx[k] as usize + pl] as u32;
    }
    (s >> 4) as u8
}

/// Bilinear infill of plane `pl` of grid weights `g` to texel weights, exactly as the decoder does.
fn infill(grid: &Grid, g: &[u8], planes: usize, pl: usize, out: &mut [u8]) {
    for (o, tap) in out.iter_mut().zip(&grid.taps) {
        *o = tap_weight(tap, g, planes, pl);
    }
}

/// Least-squares grid weights for ideal texel weights `u`: the weighted mean of the texels each
/// grid point reaches, then a few residual corrections.
fn decimate(grid: &Grid, u: &[f32], g: &mut [f32]) {
    if grid.identity {
        g[..u.len()].copy_from_slice(u);
        return;
    }
    let k = grid.rev.len();
    let back = |j: usize, x: &[f32]| -> f32 {
        let (a, b) = (grid.rev_at[j] as usize, grid.rev_at[j + 1] as usize);
        grid.revf[a..b].iter().map(|&(i, w)| w * x[i as usize]).sum()
    };
    for j in 0..k {
        g[j] = back(j, u);
    }
    let mut r = [0f32; MAX_TEXELS];
    for _ in 0..2 {
        for i in 0..u.len() {
            r[i] = u[i] - infill_f(grid, i, g);
        }
        for j in 0..k {
            g[j] = (g[j] + back(j, &r)).clamp(0.0, 64.0);
        }
    }
}

/// Texel `i`'s infilled value of (unquantized) grid weights `g`.
#[inline]
fn infill_f(grid: &Grid, i: usize, g: &[f32]) -> f32 {
    let (tap, w) = (&grid.taps[i], &grid.tapf[i]);
    w[0] * g[tap.idx[0] as usize]
        + w[1] * g[tap.idx[1] as usize]
        + w[2] * g[tap.idx[2] as usize]
        + w[3] * g[tap.idx[3] as usize]
}

/// Like `decimate`, and returns the squared error the (unquantized) infilled weights leave, each
/// texel's weighted by `scale`.
fn decimate_err(grid: &Grid, u: &[f32], g: &mut [f32], scale: impl Fn(usize) -> f32) -> f32 {
    decimate(grid, u, g);
    if grid.identity {
        return 0.0;
    }
    let mut err = 0.0;
    for i in 0..u.len() {
        let d = u[i] - infill_f(grid, i, g);
        err += d * d * scale(i);
    }
    err
}

/// Quantizes ideal texel weights `u` (and `u2` for a second plane) to codes of weight range `q`
/// on `grid`, interleaved by plane.
fn quantize(grid: &Grid, q: usize, planes: usize, u: &[f32], u2: &[f32], codes: &mut [u8]) {
    let mut g = [[0f32; 64]; 2];
    for (pl, uu) in [u, u2].into_iter().take(planes).enumerate() {
        decimate(grid, uu, &mut g[pl]);
    }
    quantize_decimated(grid, q, planes, &g, codes);
}

/// Quantizes grid weights `g` (per plane, 0..=64) to codes of weight range `q`, interleaved by
/// plane.
fn quantize_decimated(grid: &Grid, q: usize, planes: usize, g: &[[f32; 64]; 2], codes: &mut [u8]) {
    let wq = &tables().wq[q];
    for (pl, gp) in g.iter().enumerate().take(planes) {
        for j in 0..grid.rev.len() {
            codes[planes * j + pl] = wq[gp[j].round() as usize];
        }
    }
}

/// Least-squares endpoints for texel weights (0..=64: `w`, and `w2` in channel `ccs`), over the
/// texels labelled `p`.
fn refit(t: &Texels, w: &[u8], w2: &[u8], ccs: usize, labels: &[u8], p: u8) -> ([f32; 4], [f32; 4]) {
    // The normal equations' matrix for each plane: sums of (1-f)^2, f(1-f), f^2.
    let mut abc = [[0f32; 3]; 2];
    let mut r0 = [0f32; 4];
    let mut r1 = [0f32; 4];
    let mut mean = [0f32; 4];
    let mut count = 0f32;
    let planes = 1 + (ccs < NO_CCS) as usize;
    for i in 0..t.n {
        if labels[i] != p {
            continue;
        }
        let f = [w[i] as f32 * (1.0 / 64.0), w2[i] as f32 * (1.0 / 64.0)];
        for pl in 0..planes {
            let (x, y) = (f[pl], 1.0 - f[pl]);
            abc[pl][0] += y * y;
            abc[pl][1] += x * y;
            abc[pl][2] += x * x;
        }
        count += 1.0;
        for ch in 0..4 {
            let x = f[(ch == ccs) as usize];
            r0[ch] += (1.0 - x) * t.f[i][ch];
            r1[ch] += x * t.f[i][ch];
            mean[ch] += t.f[i][ch];
        }
    }
    let mut e0 = [0f32; 4];
    let mut e1 = [0f32; 4];
    for ch in 0..4 {
        let [a, b, c] = abc[(ch == ccs) as usize];
        let det = a * c - b * b;
        if det.abs() < 1e-3 * count.max(1.0) {
            let m = mean[ch] / count.max(1.0);
            (e0[ch], e1[ch]) = (m, m);
        } else {
            let inv = 1.0 / det;
            e0[ch] = ((c * r0[ch] - b * r1[ch]) * inv).clamp(0.0, 255.0);
            e1[ch] = ((a * r1[ch] - b * r0[ch]) * inv).clamp(0.0, 255.0);
        }
    }
    (e0, e1)
}

/// The best encoding found so far.
struct Best {
    err: u32,
    sym: Option<Symbolic>,
    /// Its layout and grid.
    at: (usize, usize),
}

/// The endpoints (0..=255) block `sym` gives partition `p`.
fn sym_endpoints(sym: &Symbolic, p: usize) -> ([i32; 4], [i32; 4]) {
    let cem = sym.cems[p];
    let nv = if cem == 12 { 4 } else { 3 };
    let cu = &COLOR_UNQUANT[sym.color_quant as usize];
    let mut e = ([0i32, 0, 0, 255], [0i32, 0, 0, 255]);
    for ch in 0..nv {
        e.0[ch] = cu[sym.colors[2 * nv * p + 2 * ch] as usize] as i32;
        e.1[ch] = cu[sym.colors[2 * nv * p + 2 * ch + 1] as usize] as i32;
    }
    e
}

/// Coordinate descent on a finished block: steps each weight and endpoint code to its neighbor
/// levels while the decoded error falls. Returns the new error.
fn polish(t: &Texels, lay: &Layout, grid: &Grid, sym: &mut Symbolic, passes: usize) -> u32 {
    let tb = tables();
    let q = sym.mode.quant as usize;
    let cq = sym.color_quant as usize;
    let wu = &WEIGHT_UNQUANT[q];
    let planes = 1 + sym.mode.dual as usize;
    let ccs = if sym.mode.dual { sym.ccs as usize } else { NO_CCS };
    let kk = grid.rev.len() * planes;
    let n = t.n;
    let ch = t.ch;
    let mut g = [0u8; 64];
    for j in 0..kk {
        g[j] = wu[sym.weights[j] as usize];
    }
    let mut w = [[0u8; MAX_TEXELS]; 2];
    for (pl, wp) in w.iter_mut().enumerate().take(planes) {
        infill(grid, &g, planes, pl, wp);
    }
    let mut eps = [([0i32; 4], [0i32; 4]); 3];
    for (p, e) in eps.iter_mut().enumerate().take(lay.count as usize) {
        *e = sym_endpoints(sym, p);
    }
    // The plane that drives channel `c`.
    let plane_of = |c: usize| (c == ccs) as usize;
    // Per channel and texel: every weight and endpoint step changes only some channels.
    let mut errs = [[0u32; MAX_TEXELS]; 4];
    for c in 0..ch {
        for i in 0..n {
            let e = eps[lay.labels[i] as usize];
            errs[c][i] = t.channel_error(i, c, e.0[c], e.1[c], w[plane_of(c)][i]);
        }
    }
    let nv = if sym.cems[0] == 12 { 4 } else { 3 };
    for _ in 0..passes {
        let mut improved = false;
        for j in 0..kk {
            let (gp, pl) = (j / planes, j % planes);
            let old = sym.weights[j];
            for dir in 0..2 {
                let nc = tb.wnb[q][dir][old as usize];
                if nc == old {
                    continue;
                }
                g[j] = wu[nc as usize];
                let mut delta = 0i64;
                for &(i, _) in &grid.rev[gp] {
                    let i = i as usize;
                    let nw = tap_weight(&grid.taps[i], &g, planes, pl);
                    let e = eps[lay.labels[i] as usize];
                    for c in (0..ch).filter(|&c| plane_of(c) == pl) {
                        delta += t.channel_error(i, c, e.0[c], e.1[c], nw) as i64 - errs[c][i] as i64;
                    }
                }
                if delta < 0 {
                    sym.weights[j] = nc;
                    for &(i, _) in &grid.rev[gp] {
                        let i = i as usize;
                        w[pl][i] = tap_weight(&grid.taps[i], &g, planes, pl);
                        let e = eps[lay.labels[i] as usize];
                        for c in (0..ch).filter(|&c| plane_of(c) == pl) {
                            errs[c][i] = t.channel_error(i, c, e.0[c], e.1[c], w[pl][i]);
                        }
                    }
                    improved = true;
                    break;
                }
                g[j] = wu[old as usize];
            }
        }
        for p in 0..lay.count as usize {
            for v in 0..2 * nv {
                let c = v / 2;
                let at = 2 * nv * p + v;
                let old = sym.colors[at];
                let wc = &w[plane_of(c)];
                let cur: u32 = (0..n)
                    .filter(|&i| lay.labels[i] as usize == p)
                    .map(|i| errs[c][i])
                    .sum();
                for dir in 0..2 {
                    let nc = tb.cnb[cq][dir][old as usize];
                    if nc == old {
                        continue;
                    }
                    sym.colors[at] = nc;
                    let e = sym_endpoints(sym, p);
                    if e.1[0] + e.1[1] + e.1[2] < e.0[0] + e.0[1] + e.0[2] {
                        sym.colors[at] = old;
                        continue;
                    }
                    let err: u32 = (0..n)
                        .filter(|&i| lay.labels[i] as usize == p)
                        .map(|i| t.channel_error(i, c, e.0[c], e.1[c], wc[i]))
                        .sum();
                    if err < cur {
                        eps[p] = e;
                        for i in (0..n).filter(|&i| lay.labels[i] as usize == p) {
                            errs[c][i] = t.channel_error(i, c, e.0[c], e.1[c], wc[i]);
                        }
                        improved = true;
                        break;
                    }
                    sym.colors[at] = old;
                }
            }
        }
        if !improved {
            break;
        }
    }
    errs[..ch].iter().map(|e| e[..n].iter().sum::<u32>()).sum()
}

/// A partitioning of the block's texels, the channel on a second weight plane if any, and the
/// ideal weights they imply.
struct Layout {
    count: u8,
    seed: u16,
    labels: [u8; MAX_TEXELS],
    /// The channel the second weight plane drives, or `NO_CCS`.
    ccs: usize,
    /// Ideal weights (0..=64): each texel's position along its partition's principal axis
    /// (leaving out channel `ccs`).
    u: [f32; MAX_TEXELS],
    /// Squared color distance per unit of weight, per texel (its partition's axis length / 64,
    /// squared).
    scale: [f32; MAX_TEXELS],
    /// Second-plane ideal weights: channel `ccs` between its extremes.
    u2: [f32; MAX_TEXELS],
    /// Squared distance per unit of second-plane weight.
    scale2: f32,
    /// The texels' squared distance from their partitions' lines, which no weight grid or range
    /// can remove.
    orth: f32,
}

impl Layout {
    fn new(t: &Texels, count: u8, seed: u16, labels: &[u8], ccs: usize) -> Self {
        let n = t.n;
        let mut l = Layout {
            count,
            seed,
            labels: [0; MAX_TEXELS],
            ccs,
            u: [0.0; MAX_TEXELS],
            scale: [0.0; MAX_TEXELS],
            u2: [0.0; MAX_TEXELS],
            scale2: 0.0,
            orth: 0.0,
        };
        l.labels[..n].copy_from_slice(&labels[..n]);
        let ms = Moments::of(t, labels, count);
        let mut axes = [[0f32; 4]; 3];
        let mut means = [[0f32; 4]; 3];
        for p in 0..count as usize {
            let (mean, mut cov) = ms[p].mean_cov();
            if ccs < NO_CCS {
                for a in 0..4 {
                    cov[a][ccs] = 0.0;
                    cov[ccs][a] = 0.0;
                }
            }
            let mut axis = dominant(&cov);
            // Point the axis so the second endpoint has the larger RGB sum, as the decoder wants.
            if axis[0] + axis[1] + axis[2] < 0.0 {
                axis = axis.map(|x| -x);
            }
            // The scatter off the line: all of it, less the part along the axis.
            let along: f32 = (0..4)
                .map(|a| axis[a] * (0..4).map(|b| cov[a][b] * axis[b]).sum::<f32>())
                .sum();
            let total: f32 = (0..t.ch).map(|c| cov[c][c]).sum();
            l.orth += (total - along).max(0.0);
            axes[p] = axis;
            means[p] = mean;
        }
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for i in 0..n {
            let p = labels[i] as usize;
            let (m, a, q) = (&means[p], &axes[p], &t.f[i]);
            let d = (q[0] - m[0]) * a[0] + (q[1] - m[1]) * a[1] + (q[2] - m[2]) * a[2] + (q[3] - m[3]) * a[3];
            l.u[i] = d;
            lo[p] = lo[p].min(d);
            hi[p] = hi[p].max(d);
        }
        let mut sc = [0f32; 3];
        let mut inv = [0f32; 3];
        for p in 0..count as usize {
            let len = (hi[p] - lo[p]).max(1e-3);
            sc[p] = (len / 64.0) * (len / 64.0);
            inv[p] = 64.0 / len;
        }
        for i in 0..n {
            let p = labels[i] as usize;
            l.u[i] = (l.u[i] - lo[p]) * inv[p];
            l.scale[i] = sc[p];
        }
        if ccs < NO_CCS {
            let (lo, hi) = (0..n).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
                (lo.min(t.f[i][ccs]), hi.max(t.f[i][ccs]))
            });
            let len = (hi - lo).max(1e-3);
            l.scale2 = (len / 64.0) * (len / 64.0);
            for i in 0..n {
                l.u2[i] = (t.f[i][ccs] - lo) * (64.0 / len);
            }
        }
        l
    }
}

/// A partition's color moments, exact in integers: its texel count, channel sums and sums of
/// channel products.
#[derive(Clone, Copy, Default)]
struct Moments {
    n: u32,
    s: [u32; 4],
    ss: [[u32; 4]; 4],
}

impl Moments {
    /// The moments of partitions 0..`count` of the texels by `labels`, in one pass.
    fn of(t: &Texels, labels: &[u8], count: u8) -> [Moments; 3] {
        let mut m = [Moments::default(); 3];
        for i in 0..t.n {
            let p = &t.px[i];
            let mm = &mut m[labels[i] as usize];
            mm.n += 1;
            for a in 0..4 {
                let x = p[a] as u32;
                mm.s[a] += x;
                for b in a..4 {
                    mm.ss[a][b] += x * p[b] as u32;
                }
            }
        }
        debug_assert!(m[count as usize..].iter().all(|x| x.n == 0));
        m
    }

    /// The mean and scatter matrix (the covariance times the count).
    fn mean_cov(&self) -> ([f32; 4], [[f32; 4]; 4]) {
        let n = self.n.max(1) as f64;
        let mean = self.s.map(|x| (x as f64 / n) as f32);
        let mut cov = [[0f32; 4]; 4];
        for a in 0..4 {
            for b in a..4 {
                let v = (self.ss[a][b] as f64 - self.s[a] as f64 * self.s[b] as f64 / n) as f32;
                cov[a][b] = v;
                cov[b][a] = v;
            }
        }
        (mean, cov)
    }
}

/// The block mode and endpoint range of mode `mi` of `grid` for layout `lay`, with RGBA endpoints
/// if `ce` is 1.
fn mode_of(lay: &Layout, grid: &Grid, mi: usize, ce: usize) -> (BlockMode, u8) {
    if lay.ccs < NO_CCS {
        let (bm, c) = grid.dual[mi];
        (bm, c[ce])
    } else {
        let (bm, c) = grid.modes[mi];
        (bm, c[lay.count as usize - 1][ce])
    }
}

/// Encodes one candidate exactly: quantized grid weights `codes` (planes interleaved) on `grid`
/// with mode `bm`, CEM `cem` in every partition of `lay` at endpoint range `cq`; refits endpoints,
/// quantizes them, and keeps it if it beats `best`. Returns each partition's endpoints
/// (unquantized values).
#[expect(clippy::too_many_arguments, reason = "one candidate's fields")]
fn try_candidate(
    t: &Texels,
    lay: &Layout,
    grid: &Grid,
    bm: BlockMode,
    cem: u8,
    cq: u8,
    codes: &mut [u8],
    best: &mut Best,
) -> [([i32; 4], [i32; 4]); 3] {
    let tb = tables();
    let q = bm.quant as usize;
    let wu = &WEIGHT_UNQUANT[q];
    let planes = 1 + bm.dual as usize;
    let ccs = if bm.dual { lay.ccs } else { NO_CCS };
    let kk = grid.rev.len() * planes;
    let mut g = [0u8; 64];
    for j in 0..kk {
        g[j] = wu[codes[j] as usize];
    }
    let mut w = [[0u8; MAX_TEXELS]; 2];
    for (pl, wp) in w.iter_mut().enumerate().take(planes) {
        infill(grid, &g, planes, pl, wp);
    }
    let cqt = &tb.cq[cq as usize];
    let cu = &COLOR_UNQUANT[cq as usize];
    let mut cc = [([0u8; 4], [0u8; 4]); 3];
    let mut eps = [([0i32; 4], [0i32; 4]); 3];
    for p in 0..lay.count as usize {
        let (f0, f1) = refit(t, &w[0][..t.n], &w[1][..t.n], ccs, &lay.labels, p as u8);
        let mut c0: [u8; 4] = std::array::from_fn(|ch| cqt[f0[ch].round() as usize]);
        let mut c1: [u8; 4] = std::array::from_fn(|ch| cqt[f1[ch].round() as usize]);
        let un = |c: [u8; 4]| -> [i32; 4] {
            let mut e = c.map(|x| cu[x as usize] as i32);
            if cem == 8 {
                e[3] = 255;
            }
            e
        };
        let (mut e0, mut e1) = (un(c0), un(c1));
        // Endpoints whose second sum is lower decode with blue contraction. One partition can
        // swap them and invert the weights; with several, the weights are shared, so fall back to
        // the partition's mean color.
        if e1[0] + e1[1] + e1[2] < e0[0] + e0[1] + e0[2] {
            if lay.count == 1 {
                std::mem::swap(&mut c0, &mut c1);
                std::mem::swap(&mut e0, &mut e1);
                for c in &mut codes[..kk] {
                    *c = tb.winv[q][*c as usize];
                }
                for j in 0..kk {
                    g[j] = wu[codes[j] as usize];
                }
                for (pl, wp) in w.iter_mut().enumerate().take(planes) {
                    infill(grid, &g, planes, pl, wp);
                }
            } else {
                c0 = std::array::from_fn(|ch| cqt[((f0[ch] + f1[ch]) * 0.5).round() as usize]);
                c1 = c0;
                (e0, e1) = (un(c0), un(c1));
            }
        }
        cc[p] = (c0, c1);
        eps[p] = (e0, e1);
    }
    let err = t.error(&eps, &lay.labels, &w[0][..t.n], &w[1][..t.n], ccs, best.err);
    if err < best.err {
        let mut colors = [0u8; 18];
        let nv = if cem == 12 { 4 } else { 3 };
        for p in 0..lay.count as usize {
            for ch in 0..nv {
                colors[2 * nv * p + 2 * ch] = cc[p].0[ch];
                colors[2 * nv * p + 2 * ch + 1] = cc[p].1[ch];
            }
        }
        let mut weights = [0u8; 64];
        weights[..kk].copy_from_slice(&codes[..kk]);
        best.err = err;
        best.sym = Some(Symbolic {
            mode: bm,
            partitions: lay.count,
            seed: lay.seed,
            cems: [cem; 4],
            ccs: if bm.dual { lay.ccs as u8 } else { 0 },
            colors,
            color_quant: cq,
            weights,
        });
    }
    eps
}

/// A void-extent block of one color.
fn void_extent(c: Rgba) -> [u8; 16] {
    let mut v: u128 = 0x1fc | (3 << 10) | (((1u128 << 52) - 1) << 12);
    for (i, &x) in c.iter().enumerate() {
        v |= (x as u128 * 257) << (64 + 16 * i);
    }
    v.to_le_bytes()
}

/// The principal axis (unit length, or zero if they are all one color) and mean of the texels
/// labelled `p`, leaving out channel `skip` (or none, if it is `NO_CCS`).
fn principal_axis(t: &Texels, labels: &[u8], p: u8, skip: usize) -> ([f32; 4], [f32; 4]) {
    let (mean, mut cov) = moments(t, labels, p);
    if skip < NO_CCS {
        for a in 0..4 {
            cov[a][skip] = 0.0;
            cov[skip][a] = 0.0;
        }
    }
    (dominant(&cov), mean)
}

/// The mean and covariance (scatter) of the colors of the texels in partition `p`.
fn moments(t: &Texels, labels: &[u8], p: u8) -> ([f32; 4], [[f32; 4]; 4]) {
    let mut m = Moments::default();
    for i in (0..t.n).filter(|&i| labels[i] == p) {
        let q = &t.px[i];
        m.n += 1;
        for a in 0..4 {
            let x = q[a] as u32;
            m.s[a] += x;
            for b in a..4 {
                m.ss[a][b] += x * q[b] as u32;
            }
        }
    }
    m.mean_cov()
}

/// The unit dominant eigenvector of a covariance matrix by power iteration (zero if it is
/// degenerate).
fn dominant(cov: &[[f32; 4]; 4]) -> [f32; 4] {
    // Start from the channel with the most variance.
    let tr = cov[0][0] + cov[1][1] + cov[2][2] + cov[3][3];
    if tr < 1e-3 {
        return [0.0; 4];
    }
    // Scaled by the trace, no eigenvalue is above 1, so eight steps neither overflow nor need
    // normalizing on the way.
    let k = 1.0 / tr;
    let a = cov.map(|r| r.map(|x| x * k));
    let mut v = [0f32; 4];
    let top = (0..4).max_by(|&i, &j| cov[i][i].total_cmp(&cov[j][j])).unwrap();
    v[top] = 1.0;
    for _ in 0..8 {
        v = std::array::from_fn(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2] + a[i][3] * v[3]);
    }
    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len < 1e-20 {
        return [0.0; 4];
    }
    v.map(|x| x / len)
}

/// A k-means clustering of the texels' colors into `k` groups (2 or 3), seeded from the extremes
/// of the principal axis, as one bit mask per group.
fn kmeans(t: &Texels, k: usize) -> [[u64; 3]; 3] {
    let zero = [0u8; MAX_TEXELS];
    let (axis, mean) = principal_axis(t, &zero, 0, NO_CCS);
    let proj = |i: usize| -> f32 { (0..4).map(|c| (t.f[i][c] - mean[c]) * axis[c]).sum() };
    let lo = (0..t.n).min_by(|&a, &b| proj(a).total_cmp(&proj(b))).unwrap();
    let hi = (0..t.n).max_by(|&a, &b| proj(a).total_cmp(&proj(b))).unwrap();
    let dist = |a: &[f32; 4], b: &[f32; 4]| -> f32 { (0..t.ch).map(|c| (a[c] - b[c]) * (a[c] - b[c])).sum() };
    let mut centers = [t.f[lo], t.f[hi], t.f[lo]];
    if k == 3 {
        let far = (0..t.n)
            .max_by(|&a, &b| {
                let da = dist(&t.f[a], &centers[0]).min(dist(&t.f[a], &centers[1]));
                let db = dist(&t.f[b], &centers[0]).min(dist(&t.f[b], &centers[1]));
                da.total_cmp(&db)
            })
            .unwrap();
        centers[2] = t.f[far];
    }
    let mut labels = [0u8; MAX_TEXELS];
    for _ in 0..4 {
        let mut sum = [[0f32; 4]; 3];
        let mut cnt = [0f32; 3];
        for i in 0..t.n {
            let j = (0..k)
                .min_by(|&a, &b| dist(&t.f[i], &centers[a]).total_cmp(&dist(&t.f[i], &centers[b])))
                .unwrap();
            labels[i] = j as u8;
            cnt[j] += 1.0;
            for c in 0..4 {
                sum[j][c] += t.f[i][c];
            }
        }
        for j in 0..k {
            if cnt[j] > 0.0 {
                centers[j] = sum[j].map(|x| x / cnt[j]);
            }
        }
    }
    let mut masks = [[0u64; 3]; 3];
    for i in 0..t.n {
        masks[labels[i] as usize][i / 64] |= 1 << (i % 64);
    }
    masks
}

/// The `want` patterns that best agree with clustering `km` (most texels in the same partition,
/// under the best relabelling), best first, ties to the lower index.
fn rank_patterns(fp: &Footprint, k: usize, km: &[[u64; 3]; 3], n: u32, want: usize) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("popcnt") {
        // SAFETY: the CPU has POPCNT.
        return unsafe { rank_patterns_popcnt(fp, k, km, n, want) };
    }
    rank_patterns_generic(fp, k, km, n, want)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
fn rank_patterns_popcnt(fp: &Footprint, k: usize, km: &[[u64; 3]; 3], n: u32, want: usize) -> Vec<usize> {
    rank_patterns_generic(fp, k, km, n, want)
}

#[inline(always)]
fn rank_patterns_generic(fp: &Footprint, k: usize, km: &[[u64; 3]; 3], n: u32, want: usize) -> Vec<usize> {
    match fp.nw {
        1 => rank_scan::<1>(&fp.pmasks[k - 2], k, km, n, want),
        2 => rank_scan::<2>(&fp.pmasks[k - 2], k, km, n, want),
        _ => rank_scan::<3>(&fp.pmasks[k - 2], k, km, n, want),
    }
}

#[inline(always)]
fn rank_scan<const NW: usize>(pm: &[u64], k: usize, km: &[[u64; 3]; 3], n: u32, want: usize) -> Vec<usize> {
    // Sorted (misses, index), at most `want` long.
    let mut top: Vec<(u32, usize)> = Vec::with_capacity(want + 1);
    let mut offer = |miss: u32, i: usize| {
        if top.len() == want && miss >= top[want - 1].0 {
            return;
        }
        let at = top.partition_point(|e| e.0 <= miss);
        top.insert(at, (miss, i));
        top.truncate(want);
    };
    let pop = |a: &[u64], b: &[u64; 3]| -> u32 { (0..NW).map(|w| (a[w] & b[w]).count_ones()).sum() };
    if k == 2 {
        for (i, p) in pm.as_chunks::<NW>().0.iter().enumerate() {
            let x: u32 = (0..NW).map(|w| (p[w] ^ km[0][w]).count_ones()).sum();
            offer(n - x.max(n - x), i);
        }
    } else {
        let size: [u32; 3] = std::array::from_fn(|b| km[b][..NW].iter().map(|w| w.count_ones()).sum());
        for (i, p) in pm.chunks_exact(2 * NW).enumerate() {
            let (p0, p1) = p.split_at(NW);
            let a = [pop(p0, &km[0]), pop(p0, &km[1]), pop(p0, &km[2])];
            let b = [pop(p1, &km[0]), pop(p1, &km[1]), pop(p1, &km[2])];
            let c: [u32; 3] = std::array::from_fn(|j| size[j] - a[j] - b[j]);
            let best = (a[0] + b[1] + c[2])
                .max(a[0] + b[2] + c[1])
                .max(a[1] + b[0] + c[2])
                .max(a[1] + b[2] + c[0])
                .max(a[2] + b[0] + c[1])
                .max(a[2] + b[1] + c[0]);
            offer(n - best, i);
        }
    }
    top.into_iter().map(|e| e.1).collect()
}

/// Candidates whose decimation error is measured, per candidate encoded.
const SHORTLIST: usize = 4;

/// Settings for one effort level.
#[derive(Clone, Copy)]
struct Effort {
    /// Two- and three-partition patterns tried (closest to the k-means clustering first).
    patterns: [usize; 2],
    /// Candidates (partitioning, grid, weight range) encoded exactly, best estimates first.
    candidates: usize,
    /// Rounds of re-deriving weights from the quantized endpoints.
    refine: usize,
    /// Coordinate-descent passes over the final block.
    polish: usize,
    /// Dual-plane layouts tried (one channel on the second plane, those leaving the other
    /// channels closest to a line first).
    dual: usize,
}

fn effort_settings(effort: u8) -> Effort {
    #[cfg(test)]
    if effort == u8::MAX {
        // Tuning runs: `MIPPLE_ASTC_EFFORT=p2,p3,candidates,refine,polish,dual`.
        static E: std::sync::OnceLock<Effort> = std::sync::OnceLock::new();
        return *E.get_or_init(|| {
            let v: Vec<usize> = std::env::var("MIPPLE_ASTC_EFFORT")
                .unwrap()
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect();
            Effort {
                patterns: [v[0], v[1]],
                candidates: v[2],
                refine: v[3],
                polish: v[4],
                dual: v[5],
            }
        });
    }
    match effort {
        0 => Effort {
            patterns: [1, 0],
            candidates: 2,
            refine: 0,
            polish: 1,
            dual: 1,
        },
        1 => Effort {
            patterns: [2, 0],
            candidates: 3,
            refine: 1,
            polish: 1,
            dual: 1,
        },
        2 => Effort {
            patterns: [2, 1],
            candidates: 4,
            refine: 1,
            polish: 2,
            dual: 2,
        },
        3 => Effort {
            patterns: [3, 1],
            candidates: 6,
            refine: 1,
            polish: 2,
            dual: 2,
        },
        4 => Effort {
            patterns: [4, 2],
            candidates: 8,
            refine: 2,
            polish: 3,
            dual: 4,
        },
        5 => Effort {
            patterns: [6, 3],
            candidates: 12,
            refine: 2,
            polish: 4,
            dual: 4,
        },
        6 => Effort {
            patterns: [8, 4],
            candidates: 16,
            refine: 2,
            polish: 4,
            dual: 4,
        },
        7 => Effort {
            patterns: [12, 6],
            candidates: 24,
            refine: 3,
            polish: 6,
            dual: 4,
        },
        8 => Effort {
            patterns: [16, 8],
            candidates: 32,
            refine: 3,
            polish: 6,
            dual: 4,
        },
        _ => Effort {
            patterns: [32, 16],
            candidates: 64,
            refine: 4,
            polish: 8,
            dual: 4,
        },
    }
}

/// Encodes one block of `fp`'s footprint (row-major texels).
pub fn encode_block(fp: &Footprint, px: &[Rgba], srgb: bool, effort: u8) -> [u8; 16] {
    let n = fp.texels();
    debug_assert_eq!(px.len(), n);
    if px.iter().all(|&p| p == px[0]) {
        return void_extent(px[0]);
    }
    let alpha = px.iter().any(|p| p[3] != 255);
    let mut t = Texels {
        px,
        f: [[0.0; 4]; MAX_TEXELS],
        n,
        ch: if alpha { 4 } else { 3 },
        srgb,
    };
    for (f, p) in t.f.iter_mut().zip(px) {
        *f = p.map(|x| x as f32);
    }
    let cem = if alpha { 12u8 } else { 8 };
    let ce = alpha as usize;
    let es = effort_settings(effort);
    let mut layouts = vec![Layout::new(&t, 1, 0, &[0; MAX_TEXELS], NO_CCS)];
    if es.dual > 0 {
        // Channels for the second plane, those leaving the rest closest to a line first: the
        // variance of the other channels off their dominant axis.
        let (_, cov) = moments(&t, &[0; MAX_TEXELS], 0);
        let mut off: Vec<(f32, usize)> = (0..t.ch)
            .map(|c| {
                let mut sub = cov;
                for a in 0..4 {
                    sub[a][c] = 0.0;
                    sub[c][a] = 0.0;
                }
                let v = dominant(&sub);
                let along: f32 = (0..4)
                    .map(|a| v[a] * (0..4).map(|b| sub[a][b] * v[b]).sum::<f32>())
                    .sum();
                ((0..t.ch).map(|a| sub[a][a]).sum::<f32>() - along, c)
            })
            .collect();
        off.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, c) in off.iter().take(es.dual) {
            layouts.push(Layout::new(&t, 1, 0, &[0; MAX_TEXELS], c));
        }
    }
    for (pi, &want) in es.patterns.iter().enumerate() {
        let k = pi + 2;
        let pats = &fp.patterns[pi];
        if want == 0 || pats.is_empty() {
            continue;
        }
        let km = kmeans(&t, k);
        for i in rank_patterns(fp, k, &km, n as u32, want) {
            layouts.push(Layout::new(&t, k as u8, pats[i].seed, &pats[i].labels, NO_CCS));
        }
    }
    // Score every (partitioning, grid, weight range): the error of the ideal weights decimated to
    // the grid, the expected noise of quantizing them, the expected endpoint quantization error,
    // and the texels' distance from their lines. The decimation error is first estimated from the
    // ideal weights' spectrum (what the grid's frequencies cannot hold), then measured for the
    // shortlist.
    // The `short` best so far, by estimate: (score, score less the decimation error, layout,
    // grid, mode).
    let short = es.candidates * SHORTLIST;
    let mut cands: Vec<(f32, f32, u16, u16, u16)> = Vec::with_capacity(short + 1);
    let cnoise = &*CNOISE;
    let per_texel = n as f32 * t.ch as f32;
    let mut v = [0f32; MAX_TEXELS];
    for (li, lay) in layouts.iter().enumerate() {
        let dual = lay.ccs < NO_CCS;
        for i in 0..n {
            v[i] = lay.u[i] * lay.scale[i].sqrt();
        }
        let sp = spectrum(fp, &v[..n]);
        let sp2 = dual.then(|| {
            for i in 0..n {
                v[i] = lay.u2[i] * lay.scale2.sqrt();
            }
            spectrum(fp, &v[..n])
        });
        let (bw, bh) = (fp.bw as usize, fp.bh as usize);
        // Squared color per unit of weight squared, over the block.
        let span = lay.scale[..n].iter().sum::<f32>() + if dual { n as f32 * lay.scale2 } else { 0.0 };
        for (gi, grid) in fp.grids.iter().enumerate() {
            let nm = if dual { grid.dual.len() } else { grid.modes.len() };
            if nm == 0 {
                continue;
            }
            let (gx, gy) = (grid.gx as usize, grid.gy as usize);
            let mut derr = sp[bh][bw] - sp[gy][gx];
            if let Some(sp2) = &sp2 {
                derr += sp2[bh][bw] - sp2[gy][gx];
            }
            let wnoise = if dual { &grid.wnoise_dual } else { &grid.wnoise };
            // No mode on this grid can do better than its decimation, the texels' distance from
            // their lines and its finest weight range's noise.
            let floor = derr + lay.orth + span * if dual { grid.wnoise_dual_min } else { grid.wnoise_min };
            if cands.len() == short && floor >= cands[short - 1].0 {
                continue;
            }
            for mi in 0..nm {
                let (_, cq) = mode_of(lay, grid, mi, ce);
                if cq == u8::MAX {
                    continue;
                }
                let rest = span * wnoise[mi] + per_texel * cnoise[cq as usize] + lay.orth;
                let score = derr + rest;
                if cands.len() == short && score >= cands[short - 1].0 {
                    continue;
                }
                let at = cands.partition_point(|c| c.0 <= score);
                cands.insert(at, (score, rest, li as u16, gi as u16, mi as u16));
                cands.truncate(short);
            }
        }
    }
    let by_score = |a: &(f32, f32, u16, u16, u16), b: &(f32, f32, u16, u16, u16)| a.0.total_cmp(&b.0);
    // Measured decimation errors by (layout, grid), and the decimated weights per plane.
    let mut measured: Vec<(u16, u16, f32)> = Vec::with_capacity(short);
    let mut decimated: Vec<[[f32; 64]; 2]> = Vec::with_capacity(short);
    for c in &mut cands {
        let (li, gi) = (c.2, c.3);
        let derr = match measured.iter().find(|m| m.0 == li && m.1 == gi) {
            Some(m) => m.2,
            None => {
                let (lay, grid) = (&layouts[li as usize], &fp.grids[gi as usize]);
                let mut g = [[0f32; 64]; 2];
                let mut derr = decimate_err(grid, &lay.u[..n], &mut g[0], |i| lay.scale[i]);
                if lay.ccs < NO_CCS {
                    derr += decimate_err(grid, &lay.u2[..n], &mut g[1], |_| lay.scale2);
                }
                measured.push((li, gi, derr));
                decimated.push(g);
                derr
            }
        };
        c.0 = c.1 + derr;
    }
    let keep = es.candidates.min(cands.len());
    if keep < cands.len() {
        cands.select_nth_unstable_by(keep, by_score);
    }
    cands.truncate(keep);
    cands.sort_unstable_by(by_score);
    let mut best = Best {
        err: u32::MAX,
        sym: None,
        at: (0, 0),
    };
    let mut codes = [0u8; 64];
    for &(_, _, li, gi, mi) in &cands {
        let lay = &layouts[li as usize];
        let grid = &fp.grids[gi as usize];
        let (bm, cq) = mode_of(lay, grid, mi as usize, ce);
        let q = bm.quant as usize;
        let planes = 1 + bm.dual as usize;
        let at = measured.iter().position(|m| m.0 == li && m.1 == gi).expect("measured");
        quantize_decimated(grid, q, planes, &decimated[at], &mut codes);
        let before = best.err;
        let mut eps = try_candidate(&t, lay, grid, bm, cem, cq, &mut codes, &mut best);
        for _ in 0..es.refine {
            // Re-derive ideal weights from the quantized endpoints, then decimate and re-encode.
            let mut uu = [0f32; MAX_TEXELS];
            let mut uu2 = [0f32; MAX_TEXELS];
            for i in 0..n {
                let (e0, e1) = eps[lay.labels[i] as usize];
                let d: [f32; 4] = std::array::from_fn(|c| if c == lay.ccs { 0.0 } else { (e1[c] - e0[c]) as f32 });
                let dd: f32 = d[..t.ch].iter().map(|x| x * x).sum();
                uu[i] = if dd < 1.0 {
                    lay.u[i]
                } else {
                    let p: f32 = (0..t.ch).map(|c| (t.f[i][c] - e0[c] as f32) * d[c]).sum();
                    (p / dd * 64.0).clamp(0.0, 64.0)
                };
                if planes == 2 {
                    let c = lay.ccs;
                    let de = (e1[c] - e0[c]) as f32;
                    uu2[i] = if de.abs() < 1.0 {
                        lay.u2[i]
                    } else {
                        ((t.f[i][c] - e0[c] as f32) / de * 64.0).clamp(0.0, 64.0)
                    };
                }
            }
            quantize(grid, q, planes, &uu[..n], &uu2[..n], &mut codes);
            eps = try_candidate(&t, lay, grid, bm, cem, cq, &mut codes, &mut best);
        }
        if best.err < before {
            best.at = (li as usize, gi as usize);
        }
    }
    if es.polish > 0
        && let Some(sym) = best.sym.as_mut()
    {
        let (li, gi) = best.at;
        best.err = polish(&t, &layouts[li], &fp.grids[gi], sym, es.polish);
    }
    match best.sym.and_then(|s| s.write()) {
        Some(v) => v.to_le_bytes(),
        None => void_extent(ERROR),
    }
}

/// Reads the `bw` × `bh` block at block (`bx`, `by`), repeating edge texels past the image.
fn read_block(s: &Surface, bw: u32, bh: u32, bx: u32, by: u32, out: &mut [Rgba]) {
    for t in 0..bh {
        let y = (by * bh + t).min(s.height - 1);
        for u in 0..bw {
            let x = (bx * bw + u).min(s.width - 1);
            out[(t * bw + u) as usize] = s.pixels[(y * s.width + x) as usize];
        }
    }
}

/// Encodes a surface as ASTC `bw` × `bh` blocks, in parallel, raster order.
pub fn encode_surface(s: &Surface, bw: u32, bh: u32, srgb: bool, effort: u8) -> Vec<u8> {
    use rayon::prelude::*;
    let fp = Footprint::new(bw, bh);
    let (nx, ny) = (s.width.div_ceil(bw), s.height.div_ceil(bh));
    (0..ny)
        .into_par_iter()
        .flat_map_iter(|by| {
            let mut px = vec![[0u8; 4]; (bw * bh) as usize];
            let fp = &fp;
            (0..nx)
                .flat_map(|bx| {
                    read_block(s, bw, bh, bx, by, &mut px);
                    encode_block(fp, &px, srgb, effort)
                })
                .collect::<Vec<u8>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::decode::decode_block;
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn sse(a: &[Rgba], b: &[Rgba]) -> u64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (0..4).map(|c| (p[c] as i64 - q[c] as i64).pow(2) as u64).sum::<u64>())
            .sum()
    }

    /// Encodes images at a footprint and effort and prints the RGB PSNR and single-thread time;
    /// with an ISPC `.astc` per image (`<stem>_<W>x<H>.astc` in `ISPC`), scores that too.
    /// `MIPPLE_ASTC_BENCH=<image dir>;<W>x<H>;<effort>[;<ispc dir>]`, several separated by `|`.
    #[test]
    #[ignore = "a benchmark over external images"]
    fn bench_external_images() {
        let Ok(specs) = std::env::var("MIPPLE_ASTC_BENCH") else {
            return;
        };
        for spec in specs.split('|') {
            bench_one(spec);
        }
    }

    fn bench_one(spec: &str) {
        let parts: Vec<&str> = spec.split(';').collect();
        let (bw, bh) = parts[1]
            .split_once('x')
            .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
            .unwrap();
        let effort: u8 = parts[2].parse().unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let mut files: Vec<_> = std::fs::read_dir(parts[0])
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        let (mut secs, mut sum, mut sum_ispc, mut count) = (0f64, 0f64, 0f64, 0);
        for f in files.iter().filter(|f| f.extension().is_some_and(|e| e == "png")) {
            let src = Surface::load(f).unwrap();
            let t0 = std::time::Instant::now();
            let data = pool.install(|| encode_surface(&src, bw, bh, false, effort));
            secs += t0.elapsed().as_secs_f64();
            if let Ok(dir) = std::env::var("MIPPLE_ASTC_OUT") {
                let f2 = super::super::file::AstcFile {
                    bw,
                    bh,
                    width: src.width,
                    height: src.height,
                    blocks: data.clone(),
                };
                let name = format!("{}_{bw}x{bh}.astc", f.file_stem().unwrap().to_string_lossy());
                std::fs::write(std::path::Path::new(&dir).join(name), super::super::file::write(&f2)).unwrap();
            }
            let dec = super::super::decode::decode_rgba(&data, src.width, src.height, bw, bh, false);
            let p = crate::metrics::psnr(&src, &dec, &[0, 1, 2]);
            sum += p;
            count += 1;
            let mut line = format!("{}: {p:.3}", f.file_stem().unwrap().to_string_lossy());
            if let Some(dir) = parts.get(3) {
                let path = std::path::Path::new(dir)
                    .join(format!("{}_{bw}x{bh}.astc", f.file_stem().unwrap().to_string_lossy()));
                if let Ok(bytes) = std::fs::read(&path) {
                    let a = super::super::file::read(&bytes).unwrap();
                    let d = super::super::decode::decode_rgba(&a.blocks, a.width, a.height, bw, bh, false);
                    let q = crate::metrics::psnr(&src, &d, &[0, 1, 2]);
                    sum_ispc += q;
                    line += &format!("  ispc {q:.3}");
                }
            }
            println!("{line}");
        }
        println!(
            "{bw}x{bh} effort {effort}: mean RGB PSNR {:.3} dB (ispc {:.3}), {secs:.3} s on one thread",
            sum / count as f64,
            sum_ispc / count as f64
        );
    }

    #[test]
    fn constant_blocks_are_exact() {
        let fp = Footprint::new(6, 6);
        let px = vec![[12, 200, 77, 255]; 36];
        let mut out = vec![[0u8; 4]; 36];
        decode_block(&encode_block(&fp, &px, false, 0), 6, 6, false, &mut out);
        assert_eq!(out, px);
    }

    #[test]
    fn gradients_encode_closely_at_every_footprint() {
        for (bw, bh) in [(4, 4), (5, 4), (6, 6), (8, 8), (10, 10), (12, 12)] {
            let fp = Footprint::new(bw, bh);
            let n = (bw * bh) as usize;
            let px: Vec<Rgba> = (0..n)
                .map(|i| {
                    let (x, y) = (
                        (i as u32 % bw) as f32 / (bw - 1) as f32,
                        (i as u32 / bw) as f32 / (bh - 1) as f32,
                    );
                    let v = (x + y) * 0.5;
                    [
                        (40.0 + 180.0 * v) as u8,
                        (200.0 - 150.0 * v) as u8,
                        (90.0 + 60.0 * v) as u8,
                        255,
                    ]
                })
                .collect();
            let mut out = vec![[0u8; 4]; n];
            decode_block(&encode_block(&fp, &px, false, 5), bw, bh, false, &mut out);
            let e = sse(&px, &out) as f64 / (n * 3) as f64;
            assert!(e < 4.0, "{bw}x{bh}: mse {e}");
        }
    }

    #[test]
    fn random_blocks_decode_validly_and_effort_helps() {
        let mut seed = 0x1234_5678u64;
        let fp = Footprint::new(6, 6);
        let (mut e0, mut e9) = (0u64, 0u64);
        for b in 0..200 {
            let base: [u8; 4] = std::array::from_fn(|_| rng(&mut seed) as u8);
            let px: Vec<Rgba> = (0..36)
                .map(|_| {
                    std::array::from_fn(|c| {
                        if c == 3 && b % 2 == 0 {
                            255
                        } else {
                            base[c].saturating_add((rng(&mut seed) % 40) as u8)
                        }
                    })
                })
                .collect();
            let mut out = vec![[0u8; 4]; 36];
            decode_block(&encode_block(&fp, &px, false, 0), 6, 6, false, &mut out);
            assert!(!out.contains(&ERROR));
            e0 += sse(&px, &out);
            decode_block(&encode_block(&fp, &px, false, 9), 6, 6, false, &mut out);
            e9 += sse(&px, &out);
        }
        assert!(e9 <= e0, "{e9} > {e0}");
    }
}
