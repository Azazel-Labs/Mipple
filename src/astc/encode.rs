//! Encoding ASTC LDR blocks from RGBA8.
//!
//! Each block is fitted with a line in color space (the principal axis), its texels' positions on
//! that line become ideal weights, and those are decimated to every weight grid the footprint
//! allows. Each (grid, weight range) pair is scored by an error estimate, and the best few are
//! encoded exactly: weights quantized, endpoints refitted to them and quantized, the result decoded
//! and measured.

use super::decode::ERROR;
use super::lanes::{F32x4, I16x8, I32x4};
use super::{
    BlockMode, COLOR_UNQUANT, RANGES, Symbolic, WEIGHT_RANGES, WEIGHT_UNQUANT, block_mode, cem_values, color_range,
};
use crate::surface::{Rgba, Surface};

pub mod rdo;

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
    /// The infill taken as separable (bilinear across, then down), and its least-squares inverse
    /// each way: `sep.pxt[x * pad(gx) + gx_i]` maps a row of texels to grid columns, `sep.py[gy_i *
    /// bh + y]` texel rows to grid rows, `sep.fxt[gx_i * pad(bw) + x]` grid columns back to texels,
    /// and `sep.fy[y]` is each texel row's two grid rows and the second's share. Rows are padded
    /// with zeros to whole `F32x4`s.
    sep: Sep,
    /// Single-plane block modes on this grid, with the endpoint range for one to three
    /// partitions of RGB (CEM 8), RGBA (CEM 12), RGB base+scale (CEM 6), and RGB with one or two
    /// partitions base+scale and the rest direct; `u8::MAX` where the mode leaves too few bits.
    modes: Vec<(BlockMode, [[u8; 5]; 3])>,
    /// Dual-plane block modes on this grid, with the endpoint range for one partition of RGB and
    /// RGBA.
    dual: Vec<(BlockMode, [u8; 2])>,
    /// The modes for each kind of layout (`est_kind`, then `SCALE_KIND` for base+scale RGB), by
    /// increasing weight noise.
    est: [Vec<Est>; 14],
}

/// Separable infill and decimation matrices of a grid (see `Grid::sep`).
#[derive(Default)]
struct Sep {
    bw: usize,
    pxt: Vec<f32>,
    py: Vec<f32>,
    fxt: Vec<f32>,
    fy: Vec<(usize, f32)>,
}

impl Sep {
    fn new(bw: u32, bh: u32, gx: u32, gy: u32) -> Self {
        // Texel `s` of `b` sits between grid points `j` and `j + 1` of `g`, a share `f` (of 16)
        // toward the second, as the decoder places it.
        let place = |b: u32, g: u32, s: u32| {
            let d = (1024 + b / 2) / (b - 1);
            let at = (d * s * (g - 1) + 32) >> 6;
            ((at >> 4) as usize, (at & 15) as f64 / 16.0)
        };
        // The `b` x `g` interpolation matrix one way, and its pseudo-inverse (`g` x `b`).
        let axis = |b: u32, g: u32| {
            let (b, gu) = (b as usize, g as usize);
            let mut f = vec![0f64; b * gu];
            for s in 0..b {
                let (j, w) = place(b as u32, g, s as u32);
                f[s * gu + j.min(gu - 1)] += 1.0 - w;
                f[s * gu + (j + 1).min(gu - 1)] += w;
            }
            // (FᵀF)⁻¹Fᵀ by Gauss-Jordan on [FᵀF | Fᵀ].
            let cols = gu + b;
            let mut m = vec![0f64; gu * cols];
            for r in 0..gu {
                for c in 0..gu {
                    m[r * cols + c] = (0..b).map(|s| f[s * gu + r] * f[s * gu + c]).sum::<f64>();
                }
                m[r * cols + r] += 1e-9;
                for s in 0..b {
                    m[r * cols + gu + s] = f[s * gu + r];
                }
            }
            for c in 0..gu {
                let piv = (c..gu)
                    .max_by(|&a, &b| m[a * cols + c].abs().total_cmp(&m[b * cols + c].abs()))
                    .unwrap();
                for k in 0..cols {
                    m.swap(c * cols + k, piv * cols + k);
                }
                let d = m[c * cols + c];
                for k in 0..cols {
                    m[c * cols + k] /= d;
                }
                for r in 0..gu {
                    if r != c {
                        let x = m[r * cols + c];
                        for k in 0..cols {
                            m[r * cols + k] -= x * m[c * cols + k];
                        }
                    }
                }
            }
            let pinv: Vec<f64> = (0..gu)
                .flat_map(|r| (0..b).map(move |s| (r, s)))
                .map(|(r, s)| m[r * cols + gu + s])
                .collect();
            (f, pinv)
        };
        let (fx, pinv_x) = axis(bw, gx);
        let (_, pinv_y) = axis(bh, gy);
        let (bwu, gxu) = (bw as usize, gx as usize);
        let (sw, sg) = (pad(bwu), pad(gxu));
        Sep {
            bw: bwu,
            pxt: (0..bwu * sg)
                .map(|i| {
                    if i % sg < gxu {
                        pinv_x[(i % sg) * bwu + i / sg] as f32
                    } else {
                        0.0
                    }
                })
                .collect(),
            py: pinv_y.iter().map(|&v| v as f32).collect(),
            fxt: (0..gxu * sw)
                .map(|i| {
                    if i % sw < bwu {
                        fx[(i % sw) * gxu + i / sw] as f32
                    } else {
                        0.0
                    }
                })
                .collect(),
            fy: (0..bh)
                .map(|t| {
                    let (j, w) = place(bh, gy, t);
                    if j + 1 >= gy as usize {
                        (gy as usize - 2, 1.0)
                    } else {
                        (j, w as f32)
                    }
                })
                .collect(),
        }
    }
}

/// `n` rounded up to whole `F32x4`s.
#[inline]
fn pad(n: usize) -> usize {
    (n + 3) & !3
}

/// `rows[r]` (padded to `F32x4`s) as `F32x4`s: `out[r * pad(w) ..][..w] = Σ_k c(r, k) · m[k]`,
/// where `m[k]` is row `k` of `m` (stride `pad(w)`), summed in order of `k` from zero, exactly as
/// the scalar loop would. `out` rows have stride `pad(w)` too.
#[inline(always)]
fn mix_rows(out: &mut [f32], nr: usize, nk: usize, w: usize, m: &[f32], c: impl Fn(usize, usize) -> f32) {
    match pad(w) / 4 {
        1 => mix_rows_n::<1>(out, nr, nk, m, c),
        2 => mix_rows_n::<2>(out, nr, nk, m, c),
        _ => mix_rows_n::<3>(out, nr, nk, m, c),
    }
}

/// [`mix_rows`] for rows of `NV` `F32x4`s, the whole row accumulated in registers.
#[inline(always)]
fn mix_rows_n<const NV: usize>(out: &mut [f32], nr: usize, nk: usize, m: &[f32], c: impl Fn(usize, usize) -> f32) {
    let sw = 4 * NV;
    for r in 0..nr {
        let mut a = [F32x4::zero(); NV];
        for k in 0..nk {
            let ck = F32x4::splat(c(r, k));
            let mk = &m[k * sw..];
            for (v, av) in a.iter_mut().enumerate() {
                *av = av.add(ck.mul(F32x4::load(&mk[4 * v..])));
            }
        }
        for (v, av) in a.iter().enumerate() {
            av.store(&mut out[r * sw + 4 * v..]);
        }
    }
}

/// A block mode's noise terms for the estimate.
#[derive(Clone, Copy)]
struct Est {
    /// The expected squared weight error per unit of color span, from quantizing the weights
    /// (the part of that noise the infill passes on: the mean over texels of the sum of squared
    /// tap weights) and from rounding the infill.
    wnoise: f32,
    /// The expected squared endpoint error per channel and texel (`CNOISE` of its range).
    cnoise: f32,
    /// The least `cnoise` of this mode and those after it.
    cnoise_after: f32,
    mi: u16,
}

/// The index into `Grid::est` for a layout of `count` partitions, dual plane or not, with RGBA
/// endpoints if `ce` is 1.
fn est_kind(count: u8, dual: bool, ce: usize) -> usize {
    if dual { 6 + ce } else { (count as usize - 1) * 2 + ce }
}

/// The index into `Grid::est` for one partition of base+scale RGB (CEM 6); `count` partitions
/// add `count - 1`.
const SCALE_KIND: usize = 8;

/// The index into `Grid::est` and the endpoint kind (the column of `Grid::modes`) for variant
/// `v` of a layout of `count` partitions: base+scale RGB in every partition (1), or in one (2) or
/// two (3) of them.
fn variant_kind(count: u8, v: usize) -> (usize, usize) {
    match v {
        1 => (SCALE_KIND + count as usize - 1, 2),
        2 => (SCALE_KIND + 3 + count as usize - 2, 3),
        _ => (SCALE_KIND + 5, 4),
    }
}

/// Expected squared endpoint error per channel and texel for each color range: a uniform error
/// over one quantization step, reduced by interpolation (about 1/18 of the step squared).
static CNOISE: std::sync::LazyLock<[f32; 21]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|q| {
        let step = 255.0 / (RANGES[q] as f32 - 1.0);
        step * step / 18.0
    })
});

/// A partition pattern: its seed and each texel's partition (as lanes: -1 past the texels).
struct Pattern {
    seed: u16,
    lab: Box<[i16; MAX_TEXELS]>,
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
    /// Orthonormal DCT-II bases across and down: `dct_x[x * pad(bw) + f]` (transposed, padded),
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
    let sw = pad(bw);
    // Across (each texel row to its frequencies), then down.
    let mut r = [0f32; 12 * 12];
    mix_rows(&mut r, bh, bw, bw, &fp.dct_x, |y, x| v[y * bw + x]);
    let mut c = [0f32; 12 * 12];
    mix_rows(&mut c, bh, bh, bw, &r, |fy, y| fp.dct_y[fy * bh + y]);
    let mut p = [[0f32; 13]; 13];
    for fy in 0..bh {
        let mut row = 0.0;
        for fx in 0..bw {
            let e = c[fy * sw + fx];
            row += e * e;
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
                let mut modes: Vec<(BlockMode, [[u8; 5]; 3])> = Vec::new();
                let mut dual: Vec<(BlockMode, [u8; 2])> = Vec::new();
                for m in 0..2048 {
                    let Some(bm) = block_mode(m) else { continue };
                    if bm.gx as u32 != gx || bm.gy as u32 != gy {
                        continue;
                    }
                    let cq = |p: u8, cem: u8| color_range(bm, p, &[cem; 4]).unwrap_or(u8::MAX);
                    // `k` partitions base+scale, the rest direct RGB.
                    let mixed = |p: u8, k: u8| {
                        if k >= p {
                            return u8::MAX;
                        }
                        let cems: [u8; 4] = std::array::from_fn(|i| if i < k as usize { 6 } else { 8 });
                        color_range(bm, p, &cems).unwrap_or(u8::MAX)
                    };
                    if bm.dual {
                        if !dual.iter().any(|e| e.0 == bm) && cq(1, 8) != u8::MAX {
                            dual.push((bm, [cq(1, 8), cq(1, 12)]));
                        }
                    } else if !modes.iter().any(|e| e.0 == bm) {
                        modes.push((
                            bm,
                            std::array::from_fn(|p| {
                                let p = p as u8 + 1;
                                [cq(p, 8), cq(p, 12), cq(p, 6), mixed(p, 1), mixed(p, 2)]
                            }),
                        ));
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
                            let ws = [16 + w11 - fs - ft, fs - w11, ft - w11, w11];
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
                let sep = if identity {
                    Sep::default()
                } else {
                    Sep::new(bw, bh, gx, gy)
                };
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
                let est = std::array::from_fn(|kind| {
                    let mut v: Vec<Est> = if (6..SCALE_KIND).contains(&kind) {
                        let ce = kind - 6;
                        (dual.iter().enumerate())
                            .filter(|(_, m)| m.1[ce] != u8::MAX)
                            .map(|(mi, m)| Est {
                                wnoise: wn(&m.0),
                                cnoise: CNOISE[m.1[ce] as usize],
                                cnoise_after: 0.0,
                                mi: mi as u16,
                            })
                            .collect()
                    } else {
                        let (p, ce) = match kind {
                            0..SCALE_KIND => (kind / 2, kind % 2),
                            8..11 => (kind - SCALE_KIND, 2),
                            11 | 12 => (kind - 10, 3),
                            _ => (2, 4),
                        };
                        (modes.iter().enumerate())
                            .filter(|(_, m)| m.1[p][ce] != u8::MAX)
                            .map(|(mi, m)| Est {
                                wnoise: wn(&m.0),
                                cnoise: CNOISE[m.1[p][ce] as usize],
                                cnoise_after: 0.0,
                                mi: mi as u16,
                            })
                            .collect()
                    };
                    v.sort_by(|a, b| a.wnoise.total_cmp(&b.wnoise));
                    let mut least = f32::MAX;
                    for e in v.iter_mut().rev() {
                        least = least.min(e.cnoise);
                        e.cnoise_after = least;
                    }
                    v
                });
                grids.push(Grid {
                    gx: gx as u8,
                    gy: gy as u8,
                    identity,
                    taps,
                    rev,
                    sep,
                    modes,
                    dual,
                    est,
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
                let mut lab = Box::new([-1i16; MAX_TEXELS]);
                for (o, &l) in lab.iter_mut().zip(&labels) {
                    *o = l as i16;
                }
                out.push(Pattern { seed: seed as u16, lab });
            }
            out
        });
        let nw = n.div_ceil(64);
        let pmasks = std::array::from_fn(|pi| {
            let pats: &Vec<Pattern> = &patterns[pi];
            let mut m = vec![0u64; pats.len() * nw * (pi + 1)];
            for (j, pat) in pats.iter().enumerate() {
                for (i, &l) in pat.lab[..n].iter().enumerate() {
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
                let (w, sw) = (bw as usize, pad(bw as usize));
                (0..w * sw)
                    .map(|i| if i % sw < w { b[(i % sw) * w + i / sw] } else { 0.0 })
                    .collect()
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
    /// Each channel of the texels (zero past `n`), for eight-lane kernels.
    xl: [[i16; MAX_TEXELS]; 4],
    n: usize,
    ch: usize,
}

/// No channel on a second weight plane.
const NO_CCS: usize = 4;

impl Texels<'_> {
    /// The decoded block's squared error for the endpoints (0..=255) of partitions 0..`count`,
    /// texel labels `lab` (as lanes: -1 past `n`) and texel weights (0..=64: `w`, and `w2` in
    /// channel `ccs`), eight texels at a time. A texel decodes to `(64 e0 + 32 + w (e1 - e0)) >> 6`
    /// in either color space (the sRGB path's `((e << 8 | 0x80) interpolated + 32) >> 14` is the
    /// same), every term of which fits in i16. Once past `limit`, it stops at the end of the
    /// channel, so it is only exact below `limit`.
    #[allow(clippy::too_many_arguments, reason = "one call site, all of it per candidate")]
    fn error(
        &self,
        eps: &[([i32; 4], [i32; 4]); 3],
        lab: &[i16; MAX_TEXELS],
        count: usize,
        w: &[u8; MAX_TEXELS],
        w2: &[u8; MAX_TEXELS],
        ccs: usize,
        limit: u32,
    ) -> u32 {
        let nl = self.n.next_multiple_of(8);
        let mut err = 0u32;
        for c in 0..self.ch {
            let w = if c == ccs { w2 } else { w };
            let bp: [I16x8; 3] = std::array::from_fn(|p| I16x8::splat((64 * eps[p].0[c] + 32) as i16));
            let sp: [I16x8; 3] = std::array::from_fn(|p| I16x8::splat((eps[p].1[c] - eps[p].0[c]) as i16));
            let mut acc = I32x4::zero();
            for k in (0..nl).step_by(8) {
                // Lanes past `n` match no partition, so decode to zero like their texels.
                let l = I16x8::load(&lab[k..]);
                let (mut b, mut s) = (I16x8::splat(0), I16x8::splat(0));
                for p in 0..count {
                    let m = l.eq(I16x8::splat(p as i16));
                    b = b.add(bp[p].and(m));
                    s = s.add(sp[p].and(m));
                }
                let y = b.add(I16x8::load_u8(&w[k..]).mul(s)).shr::<6>();
                acc = acc.add(y.sub(I16x8::load(&self.xl[c][k..])).sq_pairs());
            }
            err += acc.sum() as u32;
            if err > limit {
                return err;
            }
        }
        err
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

/// Least-squares grid weights (clamped to 0..=64) for ideal texel weights `u`, with the infill
/// taken as separable: the pseudo-inverse down, then across.
fn decimate(grid: &Grid, u: &[f32], g: &mut [f32]) {
    if grid.identity {
        g[..u.len()].copy_from_slice(u);
        return;
    }
    let sep = &grid.sep;
    let (bw, gx, gy) = (sep.bw, grid.gx as usize, grid.gy as usize);
    let bh = u.len() / bw;
    let mut rows = [0f32; 12 * 12];
    mix_rows(&mut rows, gy, bh, bw, &padded(u, bw), |yy, y| sep.py[yy * bh + y]);
    let sw = pad(bw);
    let mut out = [0f32; 12 * 12];
    mix_rows(&mut out, gy, bw, gx, &sep.pxt, |yy, x| rows[yy * sw + x]);
    let sg = pad(gx);
    for yy in 0..gy {
        for xx in 0..gx {
            g[yy * gx + xx] = out[yy * sg + xx].clamp(0.0, 64.0);
        }
    }
}

/// The rows of `u` (`w` wide) padded with zeros to whole `F32x4`s.
#[inline]
fn padded(u: &[f32], w: usize) -> [f32; 12 * 12] {
    let sw = pad(w);
    let mut p = [0f32; 12 * 12];
    for (y, row) in u.chunks_exact(w).enumerate() {
        p[y * sw..y * sw + w].copy_from_slice(row);
    }
    p
}

/// Like `decimate`, and returns the squared error the (unquantized, separably infilled) weights
/// leave, each texel's weighted by `scale`.
fn decimate_err(grid: &Grid, u: &[f32], g: &mut [f32], scale: impl Fn(usize) -> f32) -> f32 {
    decimate(grid, u, g);
    if grid.identity {
        return 0.0;
    }
    let sep = &grid.sep;
    let (bw, gx, gy) = (sep.bw, grid.gx as usize, grid.gy as usize);
    let sw = pad(bw);
    // Across: each grid row to a row of texel columns.
    let mut rows = [0f32; 12 * 12];
    mix_rows(&mut rows, gy, gx, bw, &sep.fxt, |yy, xx| g[yy * gx + xx]);
    let up = padded(u, bw);
    let mut err = 0.0;
    let mut dd = [0f32; 12];
    for (y, &(j, w)) in sep.fy.iter().enumerate() {
        let w4 = F32x4::splat(w);
        for v in (0..sw).step_by(4) {
            let r0 = F32x4::load(&rows[j * sw + v..]);
            let r1 = F32x4::load(&rows[(j + 1) * sw + v..]);
            let d = F32x4::load(&up[y * sw + v..]).sub(r0.add(w4.mul(r1.sub(r0))));
            d.mul(d).store(&mut dd[v..]);
        }
        for x in 0..bw {
            err += dd[x] * scale(y * bw + x);
        }
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
            codes[planes * j + pl] = wq[round_pos(gp[j])];
        }
    }
}

/// Least-squares endpoints for texel weights (0..=64: `w`, and `w2` in channel `ccs`) for each
/// of partitions 0..`count` by labels `lab` (as lanes: -1 past the texels).
fn refit(
    t: &Texels,
    w: &[u8; MAX_TEXELS],
    w2: &[u8; MAX_TEXELS],
    ccs: usize,
    lab: &[i16; MAX_TEXELS],
    count: u8,
) -> [([f32; 4], [f32; 4]); 3] {
    // Per partition, in sixty-fourths: the normal equations' matrix for each plane (sums of
    // (64-w)^2, w(64-w), w^2), the right-hand sides, the channel sums and the texel count. As
    // floats these sums would be exact too (under 2^24 units), so integers give the same values.
    // They all follow from the sums of w, w^2, v and wv, taken eight texels at a time.
    #[derive(Clone, Copy, Default)]
    struct Sums {
        abc: [[u32; 3]; 2],
        r0: [u32; 4],
        r1: [u32; 4],
        sum: [u32; 4],
        n: u32,
    }
    let mut sums = [Sums::default(); 3];
    let planes = 1 + (ccs < NO_CCS) as usize;
    let nl = t.n.next_multiple_of(8);
    let ones = I16x8::splat(1);
    for (p, s) in sums.iter_mut().enumerate().take(count as usize) {
        let pm = I16x8::splat(p as i16);
        let mut cnt = I32x4::zero();
        let mut sw = [I32x4::zero(); 2];
        let mut sww = [I32x4::zero(); 2];
        let mut sv = [I32x4::zero(); 4];
        let mut swv = [I32x4::zero(); 4];
        for k in (0..nl).step_by(8) {
            let m = I16x8::load(&lab[k..]).eq(pm);
            cnt = cnt.add(m.sq_pairs());
            let wp = [I16x8::load_u8(&w[k..]).and(m), I16x8::load_u8(&w2[k..]).and(m)];
            for pl in 0..planes {
                sw[pl] = sw[pl].add(wp[pl].madd(ones));
                sww[pl] = sww[pl].add(wp[pl].sq_pairs());
            }
            for ch in 0..4 {
                let x = I16x8::load(&t.xl[ch][k..]);
                sv[ch] = sv[ch].add(x.and(m).madd(ones));
                swv[ch] = swv[ch].add(wp[(ch == ccs) as usize].madd(x));
            }
        }
        s.n = cnt.sum() as u32;
        for pl in 0..planes {
            let (x, xx) = (sw[pl].sum() as u32, sww[pl].sum() as u32);
            s.abc[pl] = [4096 * s.n + xx - 128 * x, 64 * x - xx, xx];
        }
        for ch in 0..4 {
            let (v, wv) = (sv[ch].sum() as u32, swv[ch].sum() as u32);
            s.r0[ch] = 64 * v - wv;
            s.r1[ch] = wv;
            s.sum[ch] = v;
        }
    }
    let mut out = [([0f32; 4], [0f32; 4]); 3];
    for (s, (e0, e1)) in sums.iter().zip(&mut out).take(count as usize) {
        let n = s.n as f32;
        for ch in 0..4 {
            let [a, b, c] = s.abc[(ch == ccs) as usize].map(|v| v as f32 * (1.0 / 4096.0));
            let (r0, r1) = (s.r0[ch] as f32 * (1.0 / 64.0), s.r1[ch] as f32 * (1.0 / 64.0));
            let det = a * c - b * b;
            if det.abs() < 1e-3 * n.max(1.0) {
                let m = s.sum[ch] as f32 / n.max(1.0);
                (e0[ch], e1[ch]) = (m, m);
            } else {
                let inv = 1.0 / det;
                e0[ch] = ((c * r0 - b * r1) * inv).clamp(0.0, 255.0);
                e1[ch] = ((a * r1 - b * r0) * inv).clamp(0.0, 255.0);
            }
        }
    }
    out
}

/// `x.round()` for `x` in 0..2^23 (or NaN, as zero), without a library call where SSE4.1 is
/// missing: the fraction after truncating is exact.
#[inline]
fn round_pos(x: f32) -> usize {
    let i = x as i32;
    (i + (x - i as f32 >= 0.5) as i32) as usize
}

/// The best encoding found so far.
struct Best {
    err: u32,
    sym: Option<Symbolic>,
    /// Its layout and grid.
    at: (usize, usize),
}

/// Where partition `p`'s endpoint values start in `sym.colors`, and how many it has.
fn value_span(sym: &Symbolic, p: usize) -> (usize, usize) {
    let at = sym.cems[..p].iter().map(|&c| cem_values(c)).sum();
    (at, cem_values(sym.cems[p]))
}

/// The endpoints (0..=255) block `sym` gives partition `p`, as decoded.
fn sym_endpoints(sym: &Symbolic, p: usize) -> ([i32; 4], [i32; 4]) {
    let cu = &COLOR_UNQUANT[sym.color_quant as usize];
    let (at, k) = value_span(sym, p);
    let mut v = [0i32; 8];
    for (o, &c) in v.iter_mut().zip(&sym.colors[at..at + k]) {
        *o = cu[c as usize] as i32;
    }
    super::decode::endpoints(sym.cems[p], &v[..k]).expect("LDR mode")
}

/// Codes at endpoint range `cq` for base+offset endpoints (CEM 9, or 13 with alpha, in channels
/// 0..`nv`) nearest the endpoints `f0`, `f1`. Per channel the decoder reads the first value's top
/// seven bits as the base's low seven, and the second's top bit as the base's top bit and its bits
/// 1..=6 as a signed offset; a few codes near each target are tried.
fn quant_offset(f0: &[f32; 4], f1: &[f32; 4], nv: usize, cq: usize, out: &mut [u8]) {
    let tb = tables();
    let (cqt, cnb, cu) = (&tb.cq[cq], &tb.cnb[cq], &COLOR_UNQUANT[cq]);
    for c in 0..nv {
        let mut best = (f32::MAX, 0u8, 0u8);
        let lo = f0[c].floor().clamp(0.0, 255.0) as i32;
        for b in [lo, (lo + 1).min(255)] {
            let d = (f1[c] - b as f32).round().clamp(-32.0, 31.0) as i32;
            let n0 = cqt[((b & 0x7f) << 1) as usize];
            let n1 = cqt[((b & 0x80) | ((d & 0x3f) << 1)) as usize];
            for x0 in [n0, cnb[0][n0 as usize], cnb[1][n0 as usize]] {
                for x1 in [n1, cnb[0][n1 as usize], cnb[1][n1 as usize]] {
                    let (u0, u1) = (cu[x0 as usize] as i32, cu[x1 as usize] as i32);
                    let base = (u0 >> 1) | (u1 & 0x80);
                    let mut off = (u1 >> 1) & 0x3f;
                    if off & 0x20 != 0 {
                        off -= 0x40;
                    }
                    let e1 = (base + off).clamp(0, 255);
                    let cost = (base as f32 - f0[c]).powi(2) + (e1 as f32 - f1[c]).powi(2);
                    if cost < best.0 {
                        best = (cost, x0, x1);
                    }
                }
            }
        }
        out[2 * c] = best.1;
        out[2 * c + 1] = best.2;
    }
}

/// Codes at endpoint range `cq` for base+scale endpoints (CEM 6, or 10 with alpha) near the
/// endpoints `f0`, `f1`: the second endpoint's RGB, then the scale (in 256ths) that takes it
/// nearest the first, then (CEM 10) both alphas.
fn quant_scale(f0: &[f32; 4], f1: &[f32; 4], alpha: bool, cq: usize, out: &mut [u8]) {
    let tb = tables();
    let (cqt, cnb, cu) = (&tb.cq[cq], &tb.cnb[cq], &COLOR_UNQUANT[cq]);
    let mut u = [0i32; 3];
    for c in 0..3 {
        out[c] = cqt[round_pos(f1[c].clamp(0.0, 255.0))];
        u[c] = cu[out[c] as usize] as i32;
    }
    let uu: i32 = u.iter().map(|x| x * x).sum();
    let fu: f32 = (0..3).map(|c| f0[c] * u[c] as f32).sum();
    let st = if uu > 0 {
        (256.0 * fu / uu as f32).clamp(0.0, 255.0)
    } else {
        0.0
    };
    let n = cqt[round_pos(st)];
    let cost = |x: u8| -> f32 {
        let s = cu[x as usize] as i32;
        (0..3).map(|c| (((u[c] * s) >> 8) as f32 - f0[c]).powi(2)).sum()
    };
    out[3] = [n, cnb[0][n as usize], cnb[1][n as usize]]
        .into_iter()
        .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
        .unwrap();
    if alpha {
        out[4] = cqt[round_pos(f0[3].clamp(0.0, 255.0))];
        out[5] = cqt[round_pos(f1[3].clamp(0.0, 255.0))];
    }
}

/// Coordinate descent on a finished block: steps each weight and endpoint code to its neighbor
/// levels while the decoded error falls (the lower neighbor first). Returns the new error.
///
/// A texel's decoded channel at weight `w` is `(64 e0 + 32 + w (e1 - e0)) >> 6`, every term of
/// which fits in i16, so the errors are taken eight lanes at a time: for a weight step, one texel's
/// four channels at the step down and the step up; for an endpoint step, eight texels of the
/// partition in one channel.
fn polish(t: &Texels, lay: &Layout, grid: &Grid, sym: &mut Symbolic, passes: usize) -> u32 {
    polish_parts::<true, true>(t, lay, grid, sym, passes)
}

/// [`polish`] stepping only the weights (`W`), only the endpoints (`E`), or both. Labels and the
/// partition count are all it reads of `lay`.
fn polish_parts<const W: bool, const E: bool>(
    t: &Texels,
    lay: &Layout,
    grid: &Grid,
    sym: &mut Symbolic,
    passes: usize,
) -> u32 {
    let tb = tables();
    let q = sym.mode.quant as usize;
    let cq = sym.color_quant as usize;
    let wu = &WEIGHT_UNQUANT[q];
    let cu = &COLOR_UNQUANT[cq];
    let planes = 1 + sym.mode.dual as usize;
    let ccs = if sym.mode.dual { sym.ccs as usize } else { NO_CCS };
    let kk = grid.rev.len() * planes;
    let n = t.n;
    let ch = t.ch;
    let parts = lay.count as usize;
    let mut g = [0u8; 64];
    for j in 0..kk {
        g[j] = wu[sym.weights[j] as usize];
    }
    // The plane that drives channel `c`; per plane, a mask of the channel lanes it drives (lanes
    // 0..4 and 4..8 are the same four channels).
    let plane_of = |c: usize| (c == ccs) as usize;
    let mask: [I16x8; 2] = std::array::from_fn(|pl| {
        I16x8::from_array(std::array::from_fn(|l| -((l % 4 < ch && plane_of(l % 4) == pl) as i16)))
    });
    // Each texel's infill sum (16 times its weight, plus the rounding 8), so a grid point's step
    // moves its texels' sums by its tap weight times the step.
    let mut sums = [[0u16; MAX_TEXELS]; 2];
    for (pl, sp) in sums.iter_mut().enumerate().take(planes) {
        for (o, tap) in sp.iter_mut().zip(&grid.taps[..n]) {
            let mut v = 8u32;
            for k in 0..4 {
                v += tap.w[k] as u32 * g[planes * tap.idx[k] as usize + pl] as u32;
            }
            *o = v as u16;
        }
    }
    // Per partition, `64 e0 + 32` and `e1 - e0` in each channel lane.
    let lanes_of = |e: &([i32; 4], [i32; 4])| {
        (
            I16x8::from_array(std::array::from_fn(|l| (64 * e.0[l % 4] + 32) as i16)),
            I16x8::from_array(std::array::from_fn(|l| (e.1[l % 4] - e.0[l % 4]) as i16)),
        )
    };
    let mut eps = [([0i32; 4], [0i32; 4]); 3];
    let mut base = [I16x8::splat(0); 3];
    let mut slope = [I16x8::splat(0); 3];
    for p in 0..parts {
        eps[p] = sym_endpoints(sym, p);
        (base[p], slope[p]) = lanes_of(&eps[p]);
    }
    let mut xs = [I16x8::splat(0); MAX_TEXELS];
    for (x, p) in xs.iter_mut().zip(&t.px[..n]) {
        *x = I16x8::from_array(std::array::from_fn(|l| p[l % 4] as i16));
    }
    // Texel `i`'s error at weight `w` over the channels of plane `pl`.
    let texel_err = |base: &[I16x8; 3], slope: &[I16x8; 3], i: usize, pl: usize, w: u16| -> i32 {
        let p = lay.labels[i] as usize;
        let v = base[p].add(I16x8::splat(w as i16).mul(slope[p])).shr::<6>();
        let s = v.sub(xs[i]).and(mask[pl]).sq_pairs().to_array();
        s[0] + s[1]
    };
    // The texels by partition, eight lanes at a time: partition `p`'s are lanes `seg[p]..seg[p +
    // 1]` (each start a multiple of eight; `live` masks off the padding), texel `i` is lane
    // `pos[i]`, with its weight per plane in `wl` and its channels in `xl`.
    const LANES: usize = MAX_TEXELS + 24;
    let mut seg = [0usize; 4];
    let mut cnt = [0usize; 3];
    for &l in &lay.labels[..n] {
        cnt[l as usize] += 1;
    }
    for p in 0..parts {
        seg[p + 1] = seg[p] + cnt[p].next_multiple_of(8);
    }
    let mut pos = [0u8; MAX_TEXELS];
    let mut live = [0i16; LANES];
    let mut wl = [[0i16; LANES]; 2];
    let mut xl = [[0i16; LANES]; 4];
    let mut fill = seg;
    for i in 0..n {
        let p = lay.labels[i] as usize;
        let k = fill[p];
        fill[p] += 1;
        pos[i] = k as u8;
        live[k] = -1;
        for pl in 0..planes {
            wl[pl][k] = (sums[pl][i] >> 4) as i16;
        }
        for c in 0..4 {
            xl[c][k] = t.px[i][c] as i16;
        }
    }
    // Each texel's error over the channels of each plane.
    let mut errp = [[0i32; MAX_TEXELS]; 2];
    for pl in 0..planes {
        for i in 0..n {
            errp[pl][i] = texel_err(&base, &slope, i, pl, sums[pl][i] >> 4);
        }
    }
    let nv = if sym.cems[0] == 12 { 4 } else { 3 };
    // Direct endpoints (CEM 8 / 12) step one channel of one endpoint per code; other modes step
    // whatever their codes decode to.
    let direct = sym.cems[..parts].iter().all(|&c| c == 8 || c == 12);
    let spans: [(usize, usize); 3] = std::array::from_fn(|p| if p < parts { value_span(sym, p) } else { (0, 0) });
    let nvals: usize = spans.iter().map(|s| s.1).sum();
    // Steps tried since the last one taken. Each try depends only on the block's current state, so
    // once every weight and endpoint code has been tried in a row without a step, the rest of the
    // passes would only repeat those tries: stop there, mid-pass, with the same result.
    let steps = if W { kk } else { 0 } + if E { nvals } else { 0 };
    let mut idle = 0;
    'passes: for _ in 0..passes {
        for j in 0..if W { kk } else { 0 } {
            if idle >= steps {
                break 'passes;
            }
            idle += 1;
            let (gp, pl) = (j / planes, j % planes);
            let old = sym.weights[j];
            let nb = [tb.wnb[q][0][old as usize], tb.wnb[q][1][old as usize]];
            let d = nb.map(|c| wu[c as usize] as i32 - g[j] as i32);
            let dl = I16x8::from_array(std::array::from_fn(|l| d[l / 4] as i16));
            // The error of the texels the grid point reaches, now and after each step.
            let mut acc = I32x4::zero();
            let mut cur = 0i32;
            for &(i, tw) in &grid.rev[gp] {
                let i = i as usize;
                let p = lay.labels[i] as usize;
                let w = I16x8::splat(sums[pl][i] as i16)
                    .add(I16x8::splat(tw as i16).mul(dl))
                    .shr::<4>();
                let v = base[p].add(w.mul(slope[p])).shr::<6>();
                acc = acc.add(v.sub(xs[i]).and(mask[pl]).sq_pairs());
                cur += errp[pl][i];
            }
            let a = acc.to_array();
            let after = [a[0] + a[1], a[2] + a[3]];
            let Some(dir) = (0..2).find(|&dir| nb[dir] != old && after[dir] < cur) else {
                continue;
            };
            sym.weights[j] = nb[dir];
            g[j] = wu[nb[dir] as usize];
            for &(i, tw) in &grid.rev[gp] {
                let i = i as usize;
                sums[pl][i] = (sums[pl][i] as i32 + tw as i32 * d[dir]) as u16;
                let w = sums[pl][i] >> 4;
                wl[pl][pos[i] as usize] = w as i16;
                errp[pl][i] = texel_err(&base, &slope, i, pl, w);
            }
            idle = 0;
        }
        for p in 0..if E { parts } else { 0 } {
            for v in 0..spans[p].1 {
                if idle >= steps {
                    break 'passes;
                }
                idle += 1;
                if !direct {
                    // Partition `p`'s error over every channel with endpoints `e`.
                    let part_err = |e: &([i32; 4], [i32; 4])| -> i32 {
                        let mut acc = I32x4::zero();
                        for c in 0..ch {
                            let pl = plane_of(c);
                            let (b, s) = (
                                I16x8::splat((64 * e.0[c] + 32) as i16),
                                I16x8::splat((e.1[c] - e.0[c]) as i16),
                            );
                            for k in (seg[p]..seg[p + 1]).step_by(8) {
                                let w = I16x8::load(&wl[pl][k..]);
                                let y = b.add(w.mul(s)).shr::<6>();
                                let d = y.sub(I16x8::load(&xl[c][k..])).and(I16x8::load(&live[k..]));
                                acc = acc.add(d.sq_pairs());
                            }
                        }
                        acc.sum()
                    };
                    let (at, k) = spans[p];
                    let slot = at + v;
                    let old = sym.colors[slot];
                    let mut vals = [0i32; 8];
                    for i in 0..k {
                        vals[i] = cu[sym.colors[at + i] as usize] as i32;
                    }
                    let cur = part_err(&eps[p]);
                    let mut take = None;
                    for dir in 0..2 {
                        let nc = tb.cnb[cq][dir][old as usize];
                        if nc == old {
                            continue;
                        }
                        vals[v] = cu[nc as usize] as i32;
                        let e = super::decode::endpoints(sym.cems[p], &vals[..k]).expect("LDR mode");
                        if part_err(&e) < cur {
                            take = Some((nc, e));
                            break;
                        }
                    }
                    let Some((nc, e)) = take else { continue };
                    sym.colors[slot] = nc;
                    eps[p] = e;
                    (base[p], slope[p]) = lanes_of(&e);
                    for i in 0..n {
                        if lay.labels[i] as usize == p {
                            for pl in 0..planes {
                                errp[pl][i] = texel_err(&base, &slope, i, pl, sums[pl][i] >> 4);
                            }
                        }
                    }
                    idle = 0;
                    continue;
                }
                let c = v / 2;
                let pl = plane_of(c);
                let slot = 2 * nv * p + v;
                let old = sym.colors[slot];
                // The code and channel `c`'s endpoints after each step, unless it leaves the second
                // endpoint's sum below the first's (which would swap them). Only the code is kept
                // per step (whole endpoint pairs in an `Option` stalled on store forwarding).
                let mut cand = [None::<u8>; 2];
                let mut pairs = [(eps[p].0[c], eps[p].1[c]); 3];
                let s0 = eps[p].0[0] + eps[p].0[1] + eps[p].0[2];
                let s1 = eps[p].1[0] + eps[p].1[1] + eps[p].1[2];
                for dir in 0..2 {
                    let nc = tb.cnb[cq][dir][old as usize];
                    if nc == old {
                        continue;
                    }
                    let x = cu[nc as usize] as i32;
                    let (e0, e1) = pairs[0];
                    let (n0, n1) = if v % 2 == 0 { (x, e1) } else { (e0, x) };
                    let (t0, t1) = if c < 3 { (s0 - e0 + n0, s1 - e1 + n1) } else { (s0, s1) };
                    if t1 >= t0 {
                        cand[dir] = Some(nc);
                        pairs[dir + 1] = (n0, n1);
                    }
                }
                if cand == [None; 2] {
                    continue;
                }
                // The partition's error in channel `c` now and after each step.
                let bs = pairs.map(|(e0, _)| I16x8::splat((64 * e0 + 32) as i16));
                let ss = pairs.map(|(e0, e1)| I16x8::splat((e1 - e0) as i16));
                let mut acc = [I32x4::zero(); 3];
                for k in (seg[p]..seg[p + 1]).step_by(8) {
                    let w = I16x8::load(&wl[pl][k..]);
                    let x = I16x8::load(&xl[c][k..]);
                    let m = I16x8::load(&live[k..]);
                    for s in 0..3 {
                        let y = bs[s].add(w.mul(ss[s])).shr::<6>();
                        acc[s] = acc[s].add(y.sub(x).and(m).sq_pairs());
                    }
                }
                let err = acc.map(|a| a.to_array().iter().sum::<i32>());
                let Some(dir) = (0..2).find(|&dir| cand[dir].is_some() && err[dir + 1] < err[0]) else {
                    continue;
                };
                let nc = cand[dir].unwrap();
                sym.colors[slot] = nc;
                (eps[p].0[c], eps[p].1[c]) = pairs[dir + 1];
                let e = eps[p];
                (base[p], slope[p]) = lanes_of(&e);
                for i in 0..n {
                    if lay.labels[i] as usize == p {
                        errp[pl][i] = texel_err(&base, &slope, i, pl, sums[pl][i] >> 4);
                    }
                }
                idle = 0;
            }
        }
    }
    errp[..planes].iter().map(|e| e[..n].iter().sum::<i32>()).sum::<i32>() as u32
}

/// A partitioning of the block's texels, the channel on a second weight plane if any, and the
/// ideal weights they imply.
struct Layout {
    count: u8,
    seed: u16,
    labels: [u8; MAX_TEXELS],
    /// `labels` as lanes, -1 past the texels.
    lab16: [i16; MAX_TEXELS],
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
    /// As `orth` with base+scale endpoints (lines through black) in some partitions, and the mask
    /// of those partitions: all of them, then (two or three partitions) the one or two whose lines
    /// pass nearest black. `f32::MAX` where they do not apply (alpha, a second plane, too few
    /// partitions).
    scaled: [(f32, u8); 3],
}

impl Layout {
    /// The layout of partitions 0..`count` by labels `lab` (as lanes: -1 past the texels; whose
    /// moments are `ms`), with channel `ccs` on a second plane; `axis0`, when given, is partition
    /// 0's dominant axis (of its scatter without channel `ccs`), already computed.
    fn new(
        t: &Texels,
        count: u8,
        seed: u16,
        lab: &[i16; MAX_TEXELS],
        ccs: usize,
        ms: &[Moments; 3],
        axis0: Option<[f32; 4]>,
    ) -> Self {
        let n = t.n;
        let mut l = Layout {
            count,
            seed,
            labels: [0; MAX_TEXELS],
            lab16: *lab,
            ccs,
            u: [0.0; MAX_TEXELS],
            scale: [0.0; MAX_TEXELS],
            u2: [0.0; MAX_TEXELS],
            scale2: 0.0,
            orth: 0.0,
            scaled: [(f32::MAX, 0); 3],
        };
        let rgb = t.ch == 3 && ccs == NO_CCS;
        // Per partition, the squared distance from its line and from its line through black.
        let mut off = [(0f32, 0f32); 3];
        for (o, &x) in l.labels[..n].iter_mut().zip(lab) {
            *o = x as u8;
        }
        let labels = l.labels;
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
            let mut axis = match axis0 {
                Some(a) if p == 0 => a,
                _ => dominant(&cov),
            };
            // Point the axis so the second endpoint has the larger RGB sum, as the decoder wants.
            if axis[0] + axis[1] + axis[2] < 0.0 {
                axis = axis.map(|x| -x);
            }
            // The scatter off the line: all of it, less the part along the axis.
            let along: f32 = (0..4)
                .map(|a| axis[a] * (0..4).map(|b| cov[a][b] * axis[b]).sum::<f32>())
                .sum();
            let total: f32 = (0..t.ch).map(|c| cov[c][c]).sum();
            off[p].0 = (total - along).max(0.0);
            l.orth += off[p].0;
            if rgb {
                // The same about the line through black: the uncentered scatter.
                let mut raw = [[0f32; 4]; 4];
                for a in 0..3 {
                    for b in 0..3 {
                        raw[a][b] = ms[p].ss[a.min(b)][a.max(b)] as f32;
                    }
                }
                let ax = dominant(&raw);
                let along: f32 = (0..3)
                    .map(|a| ax[a] * (0..3).map(|b| raw[a][b] * ax[b]).sum::<f32>())
                    .sum();
                off[p].1 = (raw[0][0] + raw[1][1] + raw[2][2] - along).max(0.0);
            }
            axes[p] = axis;
            means[p] = mean;
        }
        if rgb {
            let parts = count as usize;
            let all = (1u8 << count) - 1;
            l.scaled[0] = (off[..parts].iter().map(|o| o.1).sum(), all);
            // The partitions by what base+scale endpoints add to their distance, least first.
            let mut by: [usize; 3] = [0, 1, 2];
            by[..parts].sort_by(|&a, &b| (off[a].1 - off[a].0).total_cmp(&(off[b].1 - off[b].0)));
            let mut mask = 0u8;
            for k in 1..parts {
                mask |= 1 << by[k - 1];
                let orth = (0..parts)
                    .map(|p| if mask >> p & 1 != 0 { off[p].1 } else { off[p].0 })
                    .sum();
                l.scaled[k] = (orth, mask);
            }
        }
        let mut u = [0f32; MAX_TEXELS];
        for i in 0..n {
            let p = labels[i] as usize;
            let (m, a, q) = (&means[p], &axes[p], &t.f[i]);
            u[i] = (q[0] - m[0]) * a[0] + (q[1] - m[1]) * a[1] + (q[2] - m[2]) * a[2] + (q[3] - m[3]) * a[3];
        }
        // Each partition's extent along its axis, a pass per partition (so in registers).
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        if count == 1 {
            (lo[0], hi[0]) = u[..n]
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &d| (lo.min(d), hi.max(d)));
        } else {
            for p in 0..count as usize {
                (lo[p], hi[p]) = u[..n]
                    .iter()
                    .zip(&labels)
                    .fold((f32::MAX, f32::MIN), |(lo, hi), (&d, &lb)| {
                        if lb as usize == p {
                            (lo.min(d), hi.max(d))
                        } else {
                            (lo, hi)
                        }
                    });
            }
        }
        let mut sc = [0f32; 3];
        let mut inv = [0f32; 3];
        for p in 0..count as usize {
            let len = (hi[p] - lo[p]).max(1e-3);
            sc[p] = (len / 64.0) * (len / 64.0);
            inv[p] = 64.0 / len;
        }
        for i in 0..n {
            let p = l.labels[i] as usize;
            l.u[i] = (u[i] - lo[p]) * inv[p];
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
    /// The moments of partitions 0..`count` of the texels by labels `lab` (as lanes: -1 past the
    /// texels), eight texels at a time.
    fn of(t: &Texels, lab: &[i16; MAX_TEXELS], count: u8) -> [Moments; 3] {
        let mut m = [Moments::default(); 3];
        let nl = t.n.next_multiple_of(8);
        let ones = I16x8::splat(1);
        for (p, mm) in m.iter_mut().enumerate().take(count as usize) {
            let pm = I16x8::splat(p as i16);
            let mut n = I32x4::zero();
            let mut s = [I32x4::zero(); 4];
            let mut ss = [[I32x4::zero(); 4]; 4];
            for k in (0..nl).step_by(8) {
                let msk = I16x8::load(&lab[k..]).eq(pm);
                n = n.add(msk.sq_pairs());
                let x: [I16x8; 4] = std::array::from_fn(|c| I16x8::load(&t.xl[c][k..]).and(msk));
                for a in 0..4 {
                    s[a] = s[a].add(x[a].madd(ones));
                    for b in a..4 {
                        ss[a][b] = ss[a][b].add(x[a].madd(x[b]));
                    }
                }
            }
            mm.n = n.sum() as u32;
            for a in 0..4 {
                mm.s[a] = s[a].sum() as u32;
                for b in a..4 {
                    mm.ss[a][b] = ss[a][b].sum() as u32;
                }
            }
        }
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

/// Power steps toward a partition's line in [`off_lines_rgb`].
const LINE_STEPS: usize = 3;

/// The texels' squared distance from the lines through partitions 0..`count` (by moments `ms`), in
/// channels 0..`ch`.
fn off_lines(ms: &[Moments; 3], count: usize, ch: usize) -> f32 {
    let mut off = 0.0;
    for m in &ms[..count] {
        let (_, cov) = m.mean_cov();
        let a = dominant(&cov);
        let along: f32 = (0..4)
            .map(|i| a[i] * (0..4).map(|j| cov[i][j] * a[j]).sum::<f32>())
            .sum();
        off += ((0..ch).map(|c| cov[c][c]).sum::<f32>() - along).max(0.0);
    }
    off
}

/// [`off_lines`] for RGB texels by labels `lab` (as lanes) into `count` partitions, whose moments
/// all together are `total`: the last partition's are what the others leave. Each partition's
/// line is found by `LINE_STEPS` power steps from the block's dominant `axis`: on 8 Kodak images,
/// 3 rank as well as 8 (and 1 nearly so).
fn off_lines_rgb(t: &Texels, lab: &[i16; MAX_TEXELS], count: usize, total: &Moments, axis: &[f32; 4]) -> f32 {
    let line = |n: u32, s: [u32; 3], ss: [u32; 6]| -> f32 {
        if n == 0 {
            return 0.0;
        }
        let k = 1.0 / n as f32;
        let s = s.map(|x| x as f32);
        let ss = ss.map(|x| x as f32);
        let c = [
            ss[0] - s[0] * s[0] * k,
            ss[1] - s[0] * s[1] * k,
            ss[2] - s[0] * s[2] * k,
            ss[3] - s[1] * s[1] * k,
            ss[4] - s[1] * s[2] * k,
            ss[5] - s[2] * s[2] * k,
        ];
        let tr = c[0] + c[3] + c[5];
        if tr < 1e-3 {
            return 0.0;
        }
        // Power steps scaled by the trace, from the block's dominant axis.
        let a = c.map(|x| x / tr);
        let mut v = [axis[0], axis[1], axis[2]];
        let mul = |a: &[f32; 6], v: [f32; 3]| {
            [
                a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
                a[1] * v[0] + a[3] * v[1] + a[4] * v[2],
                a[2] * v[0] + a[4] * v[1] + a[5] * v[2],
            ]
        };
        for _ in 0..LINE_STEPS {
            v = mul(&a, v);
        }
        let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if vv < 1e-30 {
            // No direction found (the start was orthogonal): rank it as off every line.
            return tr;
        }
        let cv = mul(&c, v);
        (tr - (v[0] * cv[0] + v[1] * cv[1] + v[2] * cv[2]) / vv).max(0.0)
    };
    let nl = t.n.next_multiple_of(8);
    let ones = I16x8::splat(1);
    let mut rn = total.n;
    let mut rs = [total.s[0], total.s[1], total.s[2]];
    let tss = &total.ss;
    let mut rss = [tss[0][0], tss[0][1], tss[0][2], tss[1][1], tss[1][2], tss[2][2]];
    let mut off = 0.0;
    for p in 0..count - 1 {
        let pm = I16x8::splat(p as i16);
        let mut nn = I32x4::zero();
        let mut sv = [I32x4::zero(); 3];
        let mut ssv = [I32x4::zero(); 6];
        for k in (0..nl).step_by(8) {
            let msk = I16x8::load(&lab[k..]).eq(pm);
            nn = nn.add(msk.sq_pairs());
            let x: [I16x8; 3] = std::array::from_fn(|c| I16x8::load(&t.xl[c][k..]).and(msk));
            for c in 0..3 {
                sv[c] = sv[c].add(x[c].madd(ones));
            }
            ssv[0] = ssv[0].add(x[0].madd(x[0]));
            ssv[1] = ssv[1].add(x[0].madd(x[1]));
            ssv[2] = ssv[2].add(x[0].madd(x[2]));
            ssv[3] = ssv[3].add(x[1].madd(x[1]));
            ssv[4] = ssv[4].add(x[1].madd(x[2]));
            ssv[5] = ssv[5].add(x[2].madd(x[2]));
        }
        let n = nn.sum() as u32;
        let s = sv.map(|v| v.sum() as u32);
        let ss = ssv.map(|v| v.sum() as u32);
        off += line(n, s, ss);
        rn -= n;
        for c in 0..3 {
            rs[c] -= s[c];
        }
        for j in 0..6 {
            rss[j] -= ss[j];
        }
    }
    off + line(rn, rs, rss)
}

/// The block mode and endpoint range of mode `mi` of `grid` for layout `lay`, with RGB endpoints
/// if `ce` is 0, RGBA if 1, RGB base+scale if 2, and RGB with one or two partitions base+scale if
/// 3 or 4.
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
/// (unquantized values). With partitions in `scale` (a mask), `cq` is the range for those
/// base+scale and the rest direct, and only that is tried.
#[expect(clippy::too_many_arguments, reason = "one candidate's fields")]
fn try_candidate(
    t: &Texels,
    lay: &Layout,
    grid: &Grid,
    bm: BlockMode,
    cem: u8,
    scale: u8,
    cq: u8,
    codes: &mut [u8],
    best: &mut Best,
    alts: bool,
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
    let mut fits = refit(t, &w[0], &w[1], ccs, &lay.lab16, lay.count);
    for (p, &(f0, f1)) in fits.clone().iter().enumerate().take(lay.count as usize) {
        let mut c0: [u8; 4] = std::array::from_fn(|ch| cqt[round_pos(f0[ch])]);
        let mut c1: [u8; 4] = std::array::from_fn(|ch| cqt[round_pos(f1[ch])]);
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
                fits[p] = (f1, f0);
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
                c0 = std::array::from_fn(|ch| cqt[round_pos((f0[ch] + f1[ch]) * 0.5)]);
                c1 = c0;
                (e0, e1) = (un(c0), un(c1));
            }
        }
        cc[p] = (c0, c1);
        eps[p] = (e0, e1);
    }
    let parts = lay.count as usize;
    let mut colors = [0u8; 18];
    let nv = if cem == 12 { 4 } else { 3 };
    for p in 0..parts {
        for ch in 0..nv {
            colors[2 * nv * p + 2 * ch] = cc[p].0[ch];
            colors[2 * nv * p + 2 * ch + 1] = cc[p].1[ch];
        }
    }
    let keep = |eps: &[([i32; 4], [i32; 4]); 3], cems: [u8; 4], cq: u8, colors: &[u8; 18], best: &mut Best| -> u32 {
        let err = t.error(eps, &lay.lab16, parts, &w[0], &w[1], ccs, best.err);
        if err < best.err {
            let mut weights = [0u8; 64];
            weights[..kk].copy_from_slice(&codes[..kk]);
            best.err = err;
            best.sym = Some(Symbolic {
                mode: bm,
                partitions: lay.count,
                seed: lay.seed,
                cems,
                ccs: if bm.dual { lay.ccs as u8 } else { 0 },
                colors: *colors,
                color_quant: cq,
                weights,
            });
        }
        err
    };
    let all = (1u8 << parts) - 1;
    if scale != 0 && scale != all {
        // Mixed: the direct partitions as above (at this range), the rest base+scale.
        let mut ac = [0u8; 18];
        let mut aeps = eps;
        let mut cems = [8u8; 4];
        let mut at = 0;
        for p in 0..parts {
            if scale >> p & 1 != 0 {
                cems[p] = 6;
                let (f0, f1) = &fits[p];
                quant_scale(f0, f1, false, cq as usize, &mut ac[at..at + 4]);
                let v: [i32; 4] = std::array::from_fn(|i| cu[ac[at + i] as usize] as i32);
                aeps[p] = super::decode::endpoints(6, &v).expect("LDR mode");
                at += 4;
            } else {
                for ch in 0..3 {
                    ac[at + 2 * ch] = cc[p].0[ch];
                    ac[at + 2 * ch + 1] = cc[p].1[ch];
                }
                at += 6;
            }
        }
        keep(&aeps, cems, cq, &ac, best);
        return aeps;
    }
    let mut out = if scale != 0 {
        (u32::MAX, eps)
    } else {
        (keep(&eps, [cem; 4], cq, &colors, best), eps)
    };
    // The same weights with base+offset endpoints (same range: as many values) and base+scale
    // endpoints (a finer range: fewer values), if `alts`; all base+scale is the candidate itself.
    let scale_cem = if cem == 12 { 10 } else { 6 };
    let tries = [cem + 1, scale_cem];
    let tries = match (scale != 0, alts) {
        (true, _) => &tries[1..],
        (false, true) => &tries[..],
        (false, false) => &tries[..0],
    };
    for &alt in tries {
        let Some(acq) = color_range(bm, lay.count, &[alt; 4]) else {
            continue;
        };
        let k = cem_values(alt);
        let mut ac = [0u8; 18];
        let mut aeps = [([0i32; 4], [0i32; 4]); 3];
        for p in 0..parts {
            let (f0, f1) = &fits[p];
            let codes = &mut ac[k * p..k * p + k];
            if alt == scale_cem {
                quant_scale(f0, f1, alt == 10, acq as usize, codes);
            } else {
                quant_offset(f0, f1, nv, acq as usize, codes);
            }
            let cu = &COLOR_UNQUANT[acq as usize];
            let v: [i32; 8] = std::array::from_fn(|i| if i < k { cu[codes[i] as usize] as i32 } else { 0 });
            aeps[p] = super::decode::endpoints(alt, &v[..k]).expect("LDR mode");
        }
        let err = keep(&aeps, [alt; 4], acq, &ac, best);
        if err < out.0 {
            out = (err, aeps);
        }
    }
    out.1
}

/// A void-extent block of one color.
fn void_extent(c: Rgba) -> [u8; 16] {
    let mut v: u128 = 0x1fc | (3 << 10) | (((1u128 << 52) - 1) << 12);
    for (i, &x) in c.iter().enumerate() {
        v |= (x as u128 * 257) << (64 + 16 * i);
    }
    v.to_le_bytes()
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
fn kmeans(t: &Texels, k: usize, axis: &[f32; 4], mean: &[f32; 4]) -> [[u64; 3]; 3] {
    let mut proj = [0f32; MAX_TEXELS];
    for (o, f) in proj.iter_mut().zip(&t.f[..t.n]) {
        *o = (0..4).map(|c| (f[c] - mean[c]) * axis[c]).sum();
    }
    let lo = (0..t.n).min_by(|&a, &b| proj[a].total_cmp(&proj[b])).unwrap();
    let hi = (0..t.n).max_by(|&a, &b| proj[a].total_cmp(&proj[b])).unwrap();
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
    // The channel sums are integers (exact as floats too), so group 0's is the rest of the total.
    let mut all = [0u32; 4];
    for p in &t.px[..t.n] {
        for c in 0..4 {
            all[c] += p[c] as u32;
        }
    }
    let mut labels = [0u8; MAX_TEXELS];
    for _ in 0..4 {
        let mut sum = [[0u32; 4]; 3];
        let mut cnt = [0u32; 3];
        for i in 0..t.n {
            // The nearest center, the first on a tie.
            let f = &t.f[i];
            let (d0, d1) = (dist(f, &centers[0]), dist(f, &centers[1]));
            let mut j = (d1 < d0) as usize;
            if k == 3 && dist(f, &centers[2]) < d0.min(d1) {
                j = 2;
            }
            labels[i] = j as u8;
            let (m1, m2) = ((j == 1) as u32, (j == 2) as u32);
            cnt[1] += m1;
            cnt[2] += m2;
            for c in 0..4 {
                let x = t.px[i][c] as u32;
                sum[1][c] += m1 * x;
                sum[2][c] += m2 * x;
            }
        }
        cnt[0] = t.n as u32 - cnt[1] - cnt[2];
        sum[0] = std::array::from_fn(|c| all[c] - sum[1][c] - sum[2][c]);
        for j in 0..k {
            if cnt[j] > 0 {
                centers[j] = sum[j].map(|x| x as f32 / cnt[j] as f32);
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
    // Each pattern's misses (at most `n`, so under 256), then a counting sort by (misses, index).
    let mut miss = [0u8; 1024];
    let mut hist = [0u16; 256];
    let pop = |a: &[u64], b: &[u64; 3]| -> u32 { (0..NW).map(|w| (a[w] & b[w]).count_ones()).sum() };
    let count = if k == 2 {
        let pats = pm.as_chunks::<NW>().0;
        for (o, p) in miss.iter_mut().zip(pats) {
            let x: u32 = (0..NW).map(|w| (p[w] ^ km[0][w]).count_ones()).sum();
            *o = (n - x.max(n - x)) as u8;
        }
        pats.len()
    } else {
        let size: [u32; 3] = std::array::from_fn(|b| km[b][..NW].iter().map(|w| w.count_ones()).sum());
        let pats = pm.chunks_exact(2 * NW);
        let count = pats.len();
        for (o, p) in miss.iter_mut().zip(pats) {
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
            *o = (n - best) as u8;
        }
        count
    };
    for &m in &miss[..count] {
        hist[m as usize] += 1;
    }
    // The fewest misses that take in `want` patterns, and where each count's patterns start.
    let mut at = [0u16; 256];
    let (mut total, mut thr) = (0usize, 255);
    for m in 0..256 {
        at[m] = total as u16;
        total += hist[m] as usize;
        if total >= want {
            thr = m;
            break;
        }
    }
    let mut out = vec![0usize; total];
    for (i, &m) in miss[..count].iter().enumerate() {
        if m as usize <= thr {
            out[at[m as usize] as usize] = i;
            at[m as usize] += 1;
        }
    }
    out.truncate(want);
    out
}

/// Candidates whose decimation error is measured, per candidate encoded, unless an effort level
/// sets its own shortlist.
const SHORTLIST: usize = 4;

/// The weight of decimation error against the quantization terms when ranking candidates. Finer
/// grids than the plain sum picks win often enough to matter: on Kodak at efforts 0 and 2, 1.5
/// gains 0.03-0.08 dB and 0.0005-0.0015 SSIM at 6x6 and 8x8 (at 4x4, SSIM up and PSNR within
/// 0.03 dB). The pick among the candidates encoded is still by their exact error.
const DECIMATION_BIAS: f32 = 1.5;

/// How much a candidate's exact error exceeds its estimate for direct endpoints, relative to one
/// plane and partition, in blocks of up to 16 texels: two and three partitions and a second
/// plane come out better than estimated. Measured as mean log(exact / estimate) over the
/// candidates tried on 8 Kodak images at 4x4 (e^-0.20, e^-0.24, e^-0.18); scaling scores by them
/// gains 0.07 dB there for about 3% more time. Base+scale estimates run the other way, but
/// scaling those up loses (base+scale suits flat, dark texels SSIM weighs), and at 6x6 and 8x8
/// the factors trade SSIM for PSNR.
fn calibration(n: usize, count: u8, v: usize, dual: bool) -> f32 {
    match (n <= 16, dual, v, count) {
        (false, ..) => 1.0,
        (_, true, ..) => 0.84,
        (_, _, 0, 2) => 0.82,
        (_, _, 0, 3) => 0.79,
        _ => 1.0,
    }
}

/// Settings for one effort level.
#[derive(Clone, Copy)]
struct Effort {
    /// Two- and three-partition patterns tried (closest to the k-means clustering first).
    patterns: [usize; 2],
    /// Candidates (partitioning, grid, weight range) encoded exactly, best estimates first.
    candidates: usize,
    /// Candidates whose decimation error is measured, to pick those encoded (0: `SHORTLIST` per
    /// candidate encoded).
    shortlist: usize,
    /// Rounds of re-deriving weights from the quantized endpoints.
    refine: usize,
    /// Coordinate-descent passes over the final block.
    polish: usize,
    /// Dual-plane layouts tried (one channel on the second plane, those leaving the other
    /// channels closest to a line first).
    dual: usize,
    /// Two- and three-partition patterns (`line[0]`, `line[1]`) and dual-plane layouts
    /// (`line[2]`) are tried only when the texels' mean squared distance from one line, per
    /// channel, is at least this.
    line: [f32; 3],
    /// The patterns ranked best by agreement with k-means, if more than are tried, re-ranked by how
    /// far their texels lie from their partitions' lines before the best are tried.
    rerank: usize,
}

/// The settings for `effort` on a footprint of `texels` texels.
fn effort_settings(effort: u8, texels: usize) -> Effort {
    let mut e = effort_table(effort, texels);
    // Larger footprints gain less from line-fit re-ranking: on 8 Kodak images at effort 2, 32
    // patterns score as 64 at 8x8 (at 4x4, 64 gains 0.05 dB over 32).
    if texels > 16 && effort != u8::MAX {
        e.rerank /= 2;
    }
    if e.shortlist == 0 {
        e.shortlist = e.candidates * SHORTLIST;
    }
    e
}

fn effort_table(effort: u8, texels: usize) -> Effort {
    #[cfg(test)]
    if effort == u8::MAX {
        // Tuning runs:
        // `MIPPLE_ASTC_EFFORT=p2,p3,candidates,refine,polish,dual,line2,line3,line_dual,shortlist`.
        static E: std::sync::OnceLock<Effort> = std::sync::OnceLock::new();
        return *E.get_or_init(|| {
            let mut v: Vec<f64> = std::env::var("MIPPLE_ASTC_EFFORT")
                .unwrap()
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect();
            v.resize(11, 0.0);
            let u = |i: usize| v[i] as usize;
            Effort {
                patterns: [u(0), u(1)],
                candidates: u(2),
                shortlist: u(9),
                refine: u(3),
                polish: u(4),
                dual: u(5),
                line: [v[6] as f32, v[7] as f32, v[8] as f32],
                rerank: u(10),
            }
        });
    }
    match effort {
        // From 25 to 63 texels (6x6 among them), a second candidate and a longer shortlist: on
        // Kodak at 6x6, +0.07 dB for about 10% more time (one thread).
        0 if (25..64).contains(&texels) => Effort {
            patterns: [1, 0],
            candidates: 2,
            shortlist: 5,
            refine: 0,
            polish: 2,
            dual: 1,
            line: [8.0, 16.0, 2.0],
            rerank: 0,
        },
        0 => Effort {
            patterns: [1, 0],
            candidates: 1,
            shortlist: 0,
            refine: 0,
            polish: 3,
            dual: 1,
            line: [8.0, 16.0, 2.0],
            rerank: 0,
        },
        1 => Effort {
            patterns: [2, 0],
            candidates: 3,
            shortlist: 0,
            refine: 1,
            polish: 1,
            dual: 1,
            line: [2.0, 16.0, 0.0],
            rerank: 64,
        },
        2 => Effort {
            patterns: [2, 1],
            candidates: 4,
            shortlist: 0,
            refine: 1,
            polish: 2,
            dual: 2,
            line: [2.0, 16.0, 0.0],
            rerank: 64,
        },
        3 => Effort {
            patterns: [3, 1],
            candidates: 6,
            shortlist: 0,
            refine: 1,
            polish: 2,
            dual: 2,
            line: [1.0, 8.0, 0.0],
            rerank: 64,
        },
        4 => Effort {
            patterns: [4, 2],
            candidates: 8,
            shortlist: 0,
            refine: 2,
            polish: 3,
            dual: 4,
            line: [1.0, 8.0, 0.0],
            rerank: 64,
        },
        5 => Effort {
            patterns: [6, 3],
            candidates: 12,
            shortlist: 0,
            refine: 2,
            polish: 4,
            dual: 4,
            line: [0.5, 4.0, 0.0],
            rerank: 64,
        },
        6 => Effort {
            patterns: [8, 4],
            candidates: 16,
            shortlist: 0,
            refine: 2,
            polish: 4,
            dual: 4,
            line: [0.5, 4.0, 0.0],
            rerank: 64,
        },
        7 => Effort {
            patterns: [12, 6],
            candidates: 24,
            shortlist: 0,
            refine: 3,
            polish: 6,
            dual: 4,
            line: [0.25, 2.0, 0.0],
            rerank: 128,
        },
        8 => Effort {
            patterns: [16, 8],
            candidates: 32,
            shortlist: 0,
            refine: 3,
            polish: 6,
            dual: 4,
            line: [0.25, 2.0, 0.0],
            rerank: 128,
        },
        _ => Effort {
            patterns: [32, 16],
            candidates: 64,
            shortlist: 0,
            refine: 4,
            polish: 8,
            dual: 4,
            line: [0.0, 0.0, 0.0],
            rerank: 256,
        },
    }
}

/// Encodes one block of `fp`'s footprint (row-major texels). `srgb` changes nothing: eight-bit
/// endpoints decode to the same values either way (see `Texels::error`).
pub fn encode_block(fp: &Footprint, px: &[Rgba], srgb: bool, effort: u8) -> [u8; 16] {
    let _ = srgb;
    let n = fp.texels();
    debug_assert_eq!(px.len(), n);
    if px.iter().all(|&p| p == px[0]) {
        return void_extent(px[0]);
    }
    let alpha = px.iter().any(|p| p[3] != 255);
    let mut t = Texels {
        px,
        f: [[0.0; 4]; MAX_TEXELS],
        xl: [[0; MAX_TEXELS]; 4],
        n,
        ch: if alpha { 4 } else { 3 },
    };
    for (i, p) in px.iter().enumerate() {
        t.f[i] = p.map(|x| x as f32);
        for c in 0..4 {
            t.xl[c][i] = p[c] as i16;
        }
    }
    let cem = if alpha { 12u8 } else { 8 };
    let ce = alpha as usize;
    let es = effort_settings(effort, n);
    // One partition's moments and principal axis, shared by the one-partition layouts and k-means.
    let lab1: [i16; MAX_TEXELS] = std::array::from_fn(|i| if i < n { 0 } else { -1 });
    let ms1 = Moments::of(&t, &lab1, 1);
    let (mean1, cov) = ms1[0].mean_cov();
    let axis1 = dominant(&cov);
    let mut layouts = Vec::with_capacity(1 + es.dual + es.patterns.iter().sum::<usize>());
    layouts.push(Layout::new(&t, 1, 0, &lab1, NO_CCS, &ms1, Some(axis1)));
    // The texels' mean squared distance from their line, per channel.
    let off_line = layouts[0].orth / (n * t.ch) as f32;
    if es.dual > 0 && off_line >= es.line[2] {
        // Channels for the second plane, those leaving the rest closest to a line first: the
        // variance of the other channels off their dominant axis.
        let mut off: Vec<(f32, usize, [f32; 4])> = (0..t.ch)
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
                ((0..t.ch).map(|a| sub[a][a]).sum::<f32>() - along, c, v)
            })
            .collect();
        off.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, c, v) in off.iter().take(es.dual) {
            layouts.push(Layout::new(&t, 1, 0, &lab1, c, &ms1, Some(v)));
        }
    }
    for (pi, &want) in es.patterns.iter().enumerate() {
        let k = pi + 2;
        let pats = &fp.patterns[pi];
        if want == 0 || pats.is_empty() || off_line < es.line[pi] {
            continue;
        }
        let km = kmeans(&t, k, &axis1, &mean1);
        let mut ranked = rank_patterns(fp, k, &km, n as u32, want.max(es.rerank));
        if es.rerank > want {
            let off = |i: usize| match t.ch {
                3 => off_lines_rgb(&t, &pats[i].lab, k, &ms1[0], &axis1),
                _ => off_lines(&Moments::of(&t, &pats[i].lab, k as u8), k, t.ch),
            };
            let mut by: Vec<(f32, usize)> = ranked.iter().map(|&i| (off(i), i)).collect();
            by.sort_by(|a, b| a.0.total_cmp(&b.0));
            ranked = by.into_iter().map(|e| e.1).collect();
        }
        for &i in ranked.iter().take(want) {
            let ms = Moments::of(&t, &pats[i].lab, k as u8);
            layouts.push(Layout::new(&t, k as u8, pats[i].seed, &pats[i].lab, NO_CCS, &ms, None));
        }
    }
    // Score every (partitioning, grid, weight range): the error of the ideal weights decimated to
    // the grid, the expected noise of quantizing them, the expected endpoint quantization error,
    // and the texels' distance from their lines. The decimation error is first estimated from the
    // ideal weights' spectrum (what the grid's frequencies cannot hold), then measured for the
    // shortlist.
    // The `short` best so far, by estimate, ascending: each the score's ordered bits above its
    // layout, grid and index into the grid's estimates (so equal scores go to the first in that
    // order, whatever order they are scored in), unused ones `empty`.
    let short = es.shortlist;
    // Floats' bits, flipped to order as unsigned integers as the floats do.
    let ord = |x: f32| {
        let b = x.to_bits();
        b ^ (((b as i32 >> 31) as u32) | 0x8000_0000)
    };
    let unord = |b: u32| f32::from_bits(b ^ (!((b as i32 >> 31) as u32) | 0x8000_0000));
    let empty = (ord(f32::MAX) as u64) << 32 | u32::MAX as u64;
    let mut top = vec![empty; short];
    assert!(layouts.len() < 256 && fp.grids.len() < 256);
    // Each layout's span (below) and estimate kind.
    let mut spans = Vec::with_capacity(layouts.len());
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
        let kind = est_kind(lay.count, dual, ce);
        spans.push((span, kind));
        let calib: [f32; 4] = std::array::from_fn(|v| calibration(n, lay.count, v, dual));
        // The last score's bits: no estimate above it goes in.
        let mut thr = (top[short - 1] >> 32) as u32;
        // The finest grids first: they usually score best, so the bound tightens soonest.
        for (gi, grid) in fp.grids.iter().enumerate().rev() {
            let (gx, gy) = (grid.gx as usize, grid.gy as usize);
            let mut derr = sp[bh][bw] - sp[gy][gx];
            if let Some(sp2) = &sp2 {
                derr += sp2[bh][bw] - sp2[gy][gx];
            }
            derr *= DECIMATION_BIAS;
            // Direct endpoints, then the base+scale variants (numbered in the estimate index's top
            // two bits).
            for v in 0..4 {
                let (kind, orth) = match v {
                    0 => (kind, lay.orth),
                    _ if lay.scaled[v - 1].0 == f32::MAX => continue,
                    _ => (variant_kind(lay.count, v).0, lay.scaled[v - 1].0),
                };
                let flag = (v as u64) << 14;
                for (ei, e) in grid.est[kind].iter().enumerate() {
                    // No later mode has less weight noise or less endpoint noise than this bound.
                    if ord(calib[v] * (derr + (span * e.wnoise + per_texel * e.cnoise_after + orth))) > thr {
                        break;
                    }
                    let rest = span * e.wnoise + per_texel * e.cnoise + orth;
                    let score = calib[v] * (derr + rest);
                    let mut x = (ord(score) as u64) << 32 | (li as u64) << 24 | (gi as u64) << 16 | ei as u64 | flag;
                    if x >= top[short - 1] {
                        continue;
                    }
                    // The last drops out.
                    for k in &mut top {
                        let lo = x.min(*k);
                        x = x.max(*k);
                        *k = lo;
                    }
                    thr = (top[short - 1] >> 32) as u32;
                }
            }
        }
    }
    let mut cands: Vec<(f32, f32, u16, u16, u16, u8)> = top
        .iter()
        .take_while(|&&k| k != empty)
        .map(|&k| {
            let (li, gi, ei) = ((k >> 24) as u8 as usize, (k >> 16) as u8 as usize, k as u16 as usize);
            let v = ei >> 14;
            let ((span, kind), lay) = (spans[li], &layouts[li]);
            let (kind, orth) = if v > 0 {
                (variant_kind(lay.count, v).0, lay.scaled[v - 1].0)
            } else {
                (kind, lay.orth)
            };
            let e = &fp.grids[gi].est[kind][ei & 0x3fff];
            let rest = span * e.wnoise + per_texel * e.cnoise + orth;
            (unord((k >> 32) as u32), rest, li as u16, gi as u16, e.mi, v as u8)
        })
        .collect();
    type Cand = (f32, f32, u16, u16, u16, u8);
    let by_score = |a: &Cand, b: &Cand| a.0.total_cmp(&b.0);
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
        let lay = &layouts[li as usize];
        c.0 = calibration(n, lay.count, c.5 as usize, lay.ccs < NO_CCS) * (c.1 + DECIMATION_BIAS * derr);
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
    // Base+offset and base+scale endpoints for the same weights are tried on each candidate's last
    // refinement only: on Kodak at effort 2, against every refinement, 12% / 3% / 3% less time at
    // 4x4 / 6x6 / 8x8 for 0.014 dB at 4x4 and none above. On the winner alone, 4x4 loses 0.039
    // dB, and 8x8 at effort 5 1.6% in BD-rate (`mipple check`).
    for &(_, _, li, gi, mi, v) in &cands {
        let lay = &layouts[li as usize];
        let grid = &fp.grids[gi as usize];
        let (ce, scale) = match v {
            0 => (ce, 0),
            _ => (variant_kind(lay.count, v as usize).1, lay.scaled[v as usize - 1].1),
        };
        let (bm, cq) = mode_of(lay, grid, mi as usize, ce);
        let q = bm.quant as usize;
        let planes = 1 + bm.dual as usize;
        let at = measured.iter().position(|m| m.0 == li && m.1 == gi).expect("measured");
        quantize_decimated(grid, q, planes, &decimated[at], &mut codes);
        let before = best.err;
        let mut eps = try_candidate(&t, lay, grid, bm, cem, scale, cq, &mut codes, &mut best, es.refine == 0);
        for r in 0..es.refine {
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
            eps = try_candidate(
                &t,
                lay,
                grid,
                bm,
                cem,
                scale,
                cq,
                &mut codes,
                &mut best,
                r + 1 == es.refine,
            );
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
        let (mut ssim, mut ssim_ispc) = (0f64, 0f64);
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
            ssim += crate::metrics::ssim_luma(&src, &dec);
            count += 1;
            let mut line = format!("{}: {p:.3}", f.file_stem().unwrap().to_string_lossy());
            if let Some(dir) = parts.get(3) {
                // ISPC's encodes, as this bench names them or as scripts/compete_astc.py does.
                let stem = f.file_stem().unwrap().to_string_lossy();
                let read = |name: String| std::fs::read(std::path::Path::new(dir).join(name));
                if let Ok(bytes) =
                    read(format!("{stem}_{bw}x{bh}.astc")).or_else(|_| read(format!("{stem}_{bw}x{bh}_ISPC_fast.astc")))
                {
                    let a = super::super::file::read(&bytes).unwrap();
                    let d = super::super::decode::decode_rgba(&a.blocks, a.width, a.height, bw, bh, false);
                    let q = crate::metrics::psnr(&src, &d, &[0, 1, 2]);
                    sum_ispc += q;
                    ssim_ispc += crate::metrics::ssim_luma(&src, &d);
                    line += &format!("  ispc {q:.3}");
                }
            }
            println!("{line}");
        }
        println!(
            "{bw}x{bh} effort {effort}: mean RGB PSNR {:.3} dB (ispc {:.3}), {secs:.3} s on one thread, luma SSIM {:.5} (ispc {:.5})",
            sum / count as f64,
            sum_ispc / count as f64,
            ssim / count as f64,
            ssim_ispc / count as f64
        );
    }

    #[test]
    fn round_pos_is_round() {
        // Every float in 0.25..4, around the first halves, and a stride of the rest up to 2^23.
        let mut b = 0u32;
        while f32::from_bits(b) < 8_388_608.0 {
            let x = f32::from_bits(b);
            assert_eq!(round_pos(x), x.round() as usize, "{x}");
            b += if (0.25..4.0).contains(&x) { 1 } else { 61 };
        }
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
