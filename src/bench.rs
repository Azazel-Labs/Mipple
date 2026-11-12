//! Benchmark harness: encode a corpus at several lambdas and report size vs quality.
//!
//! Sizes are measured *after* real back-end compressors (Zstd and Deflate), never as raw
//! block bytes, because that is what RDO optimizes and what ships.

use crate::format::{BlockOpts, EntropyMode, Format};
use crate::metrics;
use crate::pipeline::{EncodedTexture, decode_blocks, flatten, rdo_bases};
use crate::rdo::{self, RdoParams};
use crate::surface::Surface;
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// What to run.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub inputs: Vec<PathBuf>,
    pub formats: Vec<Format>,
    pub lambdas: Vec<f32>,
    pub effort: u8,
    /// Entropy-reduced base encoding (BC7), resolved per λ.
    pub entropy: EntropyMode,
    /// Template for RDO settings; `lambda` is overridden per run.
    pub rdo: RdoParams,
    pub zstd_level: i32,
    pub deflate_level: u32,
}

/// One (image, format, lambda) result.
#[derive(Clone, Debug)]
pub struct BenchRow {
    pub image: String,
    pub format: Format,
    pub lambda: f32,
    pub width: u32,
    pub height: u32,
    /// PSNR over the channels the format stores.
    pub psnr: f64,
    /// SSIM: luma for color formats, mean of stored channels for BC4/BC5.
    pub ssim: f64,
    /// Mean angular error in degrees, for inputs recognized as normal maps.
    pub normal_deg: Option<f64>,
    pub raw_bytes: usize,
    pub zstd_bytes: usize,
    pub deflate_bytes: usize,
    pub base_secs: f64,
    pub rdo_secs: f64,
    pub changed_pct: f64,
    /// FNV-1a of the encoded blocks.
    pub hash: u64,
    /// This row's base encode was made for it (false when shared with an earlier row).
    pub base_new: bool,
}

impl BenchRow {
    /// Bits per texel for a byte count.
    pub fn bpt(&self, bytes: usize) -> f64 {
        bytes as f64 * 8.0 / (self.width as f64 * self.height as f64)
    }
}

/// Expands directories (recursively) into image files, sorted.
pub fn collect_inputs(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if p.is_dir() {
            for e in std::fs::read_dir(p).with_context(|| format!("reading {}", p.display()))? {
                walk(&e?.path(), out)?;
            }
        } else if is_image(p) {
            out.push(p.to_path_buf());
        }
        Ok(())
    }
    let mut out = Vec::new();
    for p in paths {
        walk(p, &mut out)?;
    }
    out.sort();
    Ok(out)
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "tga")
    )
}

/// Heuristic: file names containing "normal", or ending in `_n` / `_nrm`, are normal maps.
pub fn is_normal_map(p: &Path) -> bool {
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    stem.contains("normal") || stem.ends_with("_n") || stem.ends_with("_nrm")
}

/// Size after Deflate at `level`.
pub fn deflate_len(data: &[u8], level: u32) -> Result<usize> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(data)?;
    Ok(e.finish()?.len())
}

/// (PSNR over stored channels, SSIM, normal angular error) for a decoded image.
pub fn quality(format: Format, source: &Surface, dec: &Surface, normal: bool) -> (f64, f64, Option<f64>) {
    let channels = format.metric_channels(source.has_alpha());
    let psnr = metrics::psnr(source, dec, &channels);
    let ssim = match format {
        Format::Bc1 | Format::Bc3 | Format::Bc7 => metrics::ssim_luma(source, dec),
        Format::Bc4 => metrics::ssim_channel(source, dec, 0),
        Format::Bc5 => (metrics::ssim_channel(source, dec, 0) + metrics::ssim_channel(source, dec, 1)) / 2.0,
    };
    let normal_deg = normal.then(|| metrics::normal_angle_error_deg(source, dec));
    (psnr, ssim, normal_deg)
}

/// Quality and post-compression size of an already-encoded top level.
#[derive(Clone, Debug)]
pub struct Score {
    pub psnr: f64,
    pub ssim: f64,
    pub normal_deg: Option<f64>,
    pub raw_bytes: usize,
    pub zstd_bytes: usize,
    pub deflate_bytes: usize,
    pub zstd_bpt: f64,
    pub deflate_bpt: f64,
}

/// Scores level 0 of any encoded texture (ours or another encoder's) against its source.
pub fn score(
    source: &Surface,
    tex: &EncodedTexture,
    normal: bool,
    zstd_level: i32,
    deflate_level: u32,
) -> Result<Score> {
    anyhow::ensure!(
        (tex.width, tex.height) == (source.width, source.height),
        "size mismatch: encoded {}x{}, source {}x{}",
        tex.width,
        tex.height,
        source.width,
        source.height
    );
    let data = &tex.levels[0].data;
    let dec = decode_blocks(tex.format, data, tex.width, tex.height, normal);
    let (psnr, ssim, normal_deg) = quality(tex.format, source, &dec, normal);
    let zstd_bytes = zstd::bulk::compress(data, zstd_level)?.len();
    let deflate_bytes = deflate_len(data, deflate_level)?;
    let texels = tex.width as f64 * tex.height as f64;
    Ok(Score {
        psnr,
        ssim,
        normal_deg,
        raw_bytes: data.len(),
        zstd_bytes,
        deflate_bytes,
        zstd_bpt: zstd_bytes as f64 * 8.0 / texels,
        deflate_bpt: deflate_bytes as f64 * 8.0 / texels,
    })
}

/// Base-encoded blocks, alternate bases (BC7 RDO), and the seconds they took.
type BaseEncode = (Vec<[u8; 16]>, Vec<Vec<[u8; 16]>>, f64);

/// An in-memory benchmark input.
#[derive(Clone, Debug)]
pub struct BenchImage {
    pub name: String,
    pub surface: Surface,
    /// Score as a normal map (angular error, BC5 Z reconstruction).
    pub normal: bool,
}

impl BenchImage {
    /// Loads an image file; normal maps are recognized by name ([`is_normal_map`]).
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            name: path.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string(),
            surface: Surface::load(path)?,
            normal: is_normal_map(path),
        })
    }
}

/// 64-bit FNV-1a, used to tell whether an encoder change touched the output at all.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Runs the benchmark over `cfg.inputs`, calling `progress` after each row.
pub fn run(cfg: &BenchConfig, mut progress: impl FnMut(&BenchRow)) -> Result<Vec<BenchRow>> {
    let mut rows = Vec::new();
    for path in &cfg.inputs {
        let img = BenchImage::load(path)?;
        rows.extend(run_images(std::slice::from_ref(&img), cfg, &mut progress)?);
    }
    Ok(rows)
}

/// Runs the benchmark over in-memory images (`cfg.inputs` is ignored).
pub fn run_images(
    images: &[BenchImage],
    cfg: &BenchConfig,
    mut progress: impl FnMut(&BenchRow),
) -> Result<Vec<BenchRow>> {
    let mut rows = Vec::new();
    for img in images {
        let (surface, normal, name) = (&img.surface, img.normal, &img.name);
        let src = surface.blocks();
        for &format in &cfg.formats {
            let w = format.error_weights();
            // Base encodes, as `mipple encode` makes them, per (entropy-reduced, RDO on): each made
            // on first use and shared by the λ that need it.
            let mut bases: [Option<BaseEncode>; 4] = Default::default();
            let mut base_used = [false; 4];
            for &lambda in &cfg.lambdas {
                let reduce = cfg.entropy.resolve(lambda);
                let rdo_on = lambda > 0.0;
                let key = reduce as usize * 2 + rdo_on as usize;
                let slot = &mut bases[key];
                if slot.is_none() {
                    let t0 = Instant::now();
                    let opts = BlockOpts {
                        effort: cfg.effort,
                        reduce_entropy: reduce,
                    };
                    let (base, alts) = rdo_bases(surface, &src, format, opts, rdo_on);
                    *slot = Some((base, alts, t0.elapsed().as_secs_f64()));
                }
                let (base, alts, base_secs) = slot.as_ref().unwrap();
                let base_secs = *base_secs;
                let base_new = !std::mem::replace(&mut base_used[key], true);
                let mut blocks = base.clone();
                let params = RdoParams {
                    lambda,
                    reduce_entropy: reduce,
                    row: (surface.width as usize).div_ceil(4),
                    ..cfg.rdo
                };
                let t1 = Instant::now();
                let stats = rdo::optimize_with_alt(format, &src, &mut blocks, alts, &w, &params);
                let rdo_secs = t1.elapsed().as_secs_f64();
                let data = flatten(format, &blocks);
                let zstd_bytes = zstd::bulk::compress(&data, cfg.zstd_level)?.len();
                let deflate_bytes = deflate_len(&data, cfg.deflate_level)?;
                let dec = decode_blocks(format, &data, surface.width, surface.height, normal);
                let (psnr, ssim, normal_deg) = quality(format, surface, &dec, normal);
                let row = BenchRow {
                    hash: fnv1a(&data),
                    base_new,
                    image: name.clone(),
                    format,
                    lambda,
                    width: surface.width,
                    height: surface.height,
                    psnr,
                    ssim,
                    normal_deg,
                    raw_bytes: data.len(),
                    zstd_bytes,
                    deflate_bytes,
                    base_secs,
                    rdo_secs,
                    changed_pct: if stats.blocks > 0 {
                        100.0 * stats.changed as f64 / stats.blocks as f64
                    } else {
                        0.0
                    },
                };
                progress(&row);
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

fn fmt_db(v: f64) -> String {
    if v.is_infinite() {
        "inf".into()
    } else {
        format!("{v:.2}")
    }
}

/// One line per row, aligned for terminals.
pub fn row_line(r: &BenchRow) -> String {
    format!(
        "{:<28} {:<4} λ={:<5} PSNR {:>6} dB  SSIM {:.4}{}  zstd {:>5.2} bpt  deflate {:>5.2} bpt  changed {:>5.1}%  base {:.2}s rdo {:.2}s",
        truncate(&r.image, 28),
        r.format.name(),
        r.lambda,
        fmt_db(r.psnr),
        r.ssim,
        r.normal_deg.map(|d| format!("  normal {d:.2}°")).unwrap_or_default(),
        r.bpt(r.zstd_bytes),
        r.bpt(r.deflate_bytes),
        r.changed_pct,
        r.base_secs,
        r.rdo_secs,
    )
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

/// Writes every row as CSV.
pub fn write_csv(rows: &[BenchRow], path: &Path) -> Result<()> {
    let mut s = String::from(
        "image,format,lambda,width,height,psnr_db,ssim,normal_deg,raw_bytes,zstd_bytes,deflate_bytes,zstd_bpt,deflate_bpt,changed_pct,base_secs,rdo_secs\n",
    );
    for r in rows {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{:.5},{},{},{},{},{:.4},{:.4},{:.2},{:.4},{:.4}",
            r.image,
            r.format.name(),
            r.lambda,
            r.width,
            r.height,
            fmt_db(r.psnr),
            r.ssim,
            r.normal_deg.map(|d| format!("{d:.4}")).unwrap_or_default(),
            r.raw_bytes,
            r.zstd_bytes,
            r.deflate_bytes,
            r.bpt(r.zstd_bytes),
            r.bpt(r.deflate_bytes),
            r.changed_pct,
            r.base_secs,
            r.rdo_secs,
        );
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

/// Markdown summary: per format and lambda, totals across the corpus.
pub fn summary_markdown(rows: &[BenchRow]) -> String {
    #[derive(Default)]
    struct Agg {
        n: usize,
        psnr: f64,
        ssim: f64,
        normal: f64,
        normal_n: usize,
        raw: usize,
        zstd: usize,
        deflate: usize,
        texels: f64,
        rdo_secs: f64,
    }
    let mut groups: BTreeMap<(String, u32), Agg> = BTreeMap::new();
    for r in rows {
        let key = (r.format.name().to_string(), (r.lambda * 1000.0).round() as u32);
        let a = groups.entry(key).or_default();
        a.n += 1;
        a.psnr += if r.psnr.is_finite() { r.psnr } else { 99.0 };
        a.ssim += r.ssim;
        if let Some(d) = r.normal_deg {
            a.normal += d;
            a.normal_n += 1;
        }
        a.raw += r.raw_bytes;
        a.zstd += r.zstd_bytes;
        a.deflate += r.deflate_bytes;
        a.texels += r.width as f64 * r.height as f64;
        a.rdo_secs += r.rdo_secs;
    }
    let mut baseline: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for ((fmt, l), a) in &groups {
        if *l == 0 {
            baseline.insert(fmt.clone(), (a.zstd, a.deflate));
        }
    }
    let mut s = String::new();
    let images = rows
        .iter()
        .map(|r| r.image.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let _ = writeln!(
        s,
        "Corpus: {images} image(s). Sizes are totals across the corpus; quality is the mean per image.\n"
    );
    let _ = writeln!(
        s,
        "| Format | λ | Mean PSNR (dB) | Mean SSIM | Normal err (°) | Zstd bpt | Δ Zstd vs λ0 | Deflate bpt | Δ Deflate vs λ0 | RDO time (s) |"
    );
    let _ = writeln!(s, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for ((fmt, l), a) in &groups {
        let n = a.n as f64;
        let (bz, bd) = baseline.get(fmt).copied().unwrap_or((a.zstd, a.deflate));
        let pct = |v: usize, b: usize| {
            if b == 0 {
                "—".to_string()
            } else {
                format!("{:+.1}%", 100.0 * (v as f64 / b as f64 - 1.0))
            }
        };
        let _ = writeln!(
            s,
            "| {} | {} | {:.2} | {:.4} | {} | {:.3} | {} | {:.3} | {} | {:.2} |",
            fmt,
            *l as f64 / 1000.0,
            a.psnr / n,
            a.ssim / n,
            if a.normal_n > 0 {
                format!("{:.2}", a.normal / a.normal_n as f64)
            } else {
                "—".into()
            },
            a.zstd as f64 * 8.0 / a.texels,
            pct(a.zstd, bz),
            a.deflate as f64 * 8.0 / a.texels,
            pct(a.deflate, bd),
            a.rdo_secs,
        );
    }
    s
}
