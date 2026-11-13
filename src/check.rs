//! Fast regression check (`mipple check`).
//!
//! Encodes a small fixed corpus (the procedural synthetic set at 256², or the first few images of a
//! directory) across every format and a short λ sweep, then compares the result with a stored
//! baseline. It takes seconds, not the minutes a full benchmark or head-to-head needs, and is meant
//! to run after every encoder change and in CI.
//!
//! Gates (each per format, all configurable):
//! - **BD-rate** of the size/PSNR curve (Zstd) against the baseline curve must not exceed `max_bd_rate`.
//!   This is the headline: it judges trade-offs, so a change that loses PSNR but saves more bytes passes.
//!   ASTC has no RDO, so it has only λ = 0 rows and only the λ = 0 gates.
//! - **λ = 0** (no RDO) pooled PSNR must not drop more than `max_psnr_drop`, and its Zstd size must not
//!   grow more than `max_size_growth`. This guards the base encoder on its own.
//! - **Per image at λ = 0**, PSNR must not drop more than `max_image_psnr_drop`, so one broken image
//!   type can't hide inside a pooled average.
//!
//! Timing is reported, and only gated when `max_slowdown` is set (timings vary between machines).
//! Whether each encode is byte-identical to the baseline is also reported: a pure refactor should
//! show "identical".

use crate::bench::{self, BenchConfig, BenchImage, BenchRow};
use crate::format::Format;
use crate::synth;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Default λ sweep: five points, enough for a cubic BD-rate fit.
pub const DEFAULT_LAMBDAS: [f32; 5] = [0.0, 0.5, 1.0, 2.0, 4.0];
/// Default synthetic image size.
pub const DEFAULT_SIZE: u32 = 256;

/// Pass/fail thresholds.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Max BD-rate (%, Zstd) vs the baseline curve.
    pub max_bd_rate: f64,
    /// Max pooled PSNR drop (dB) at λ = 0.
    pub max_psnr_drop: f64,
    /// Max Zstd size growth (%) at λ = 0.
    pub max_size_growth: f64,
    /// Max PSNR drop (dB) of any single image at λ = 0.
    pub max_image_psnr_drop: f64,
    /// Max total-time ratio vs the baseline; `None` reports time without gating it.
    pub max_slowdown: Option<f64>,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            max_bd_rate: 1.0,
            max_psnr_drop: 0.10,
            max_size_growth: 1.0,
            max_image_psnr_drop: 0.5,
            max_slowdown: None,
        }
    }
}

/// The corpus: the synthetic set at `size`, or the first `limit` images under `dir`.
pub fn corpus(dir: Option<&Path>, limit: usize, size: u32) -> Result<Vec<BenchImage>> {
    match dir {
        None => Ok(synth::set()
            .into_iter()
            .map(|(name, f)| BenchImage {
                name: name.to_string(),
                surface: f(size),
                normal: bench::is_normal_map(Path::new(name)),
            })
            .collect()),
        Some(d) => {
            let mut paths = bench::collect_inputs(&[d.to_path_buf()])?;
            if limit > 0 {
                paths.truncate(limit);
            }
            if paths.is_empty() {
                bail!("no images under {}", d.display());
            }
            paths.iter().map(|p| BenchImage::load(p)).collect()
        }
    }
}

/// Default baseline file for a corpus: `bench-results/check-baseline[-<dir>].csv`.
pub fn default_baseline_path(dir: Option<&Path>) -> PathBuf {
    let suffix = dir
        .and_then(|d| d.file_name())
        .and_then(|s| s.to_str())
        .map(|s| format!("-{s}"))
        .unwrap_or_default();
    PathBuf::from(format!("bench-results/check-baseline{suffix}.csv"))
}

/// One (image, format, λ) result, as stored in a baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub image: String,
    pub format: Format,
    pub lambda: f32,
    pub texels: u64,
    pub psnr: f64,
    pub normal_deg: Option<f64>,
    pub zstd_bytes: usize,
    pub deflate_bytes: usize,
    pub hash: u64,
    pub secs: f64,
}

impl Entry {
    fn key(&self) -> (String, &'static str, u32) {
        (
            self.image.clone(),
            self.format.name(),
            (self.lambda * 1000.0).round() as u32,
        )
    }
}

impl From<&BenchRow> for Entry {
    fn from(r: &BenchRow) -> Self {
        Self {
            image: r.image.clone(),
            format: r.format,
            lambda: r.lambda,
            texels: r.width as u64 * r.height as u64,
            psnr: r.psnr,
            normal_deg: r.normal_deg,
            zstd_bytes: r.zstd_bytes,
            deflate_bytes: r.deflate_bytes,
            hash: r.hash,
            // Base encodes are shared across λ; charge each to the first row that used it.
            secs: r.rdo_secs + if r.base_new { r.base_secs } else { 0.0 },
        }
    }
}

/// Runs the corpus and returns one entry per (image, format, λ).
pub fn run(images: &[BenchImage], cfg: &BenchConfig, progress: impl FnMut(&BenchRow)) -> Result<Vec<Entry>> {
    Ok(bench::run_images(images, cfg, progress)?
        .iter()
        .map(Entry::from)
        .collect())
}

const HEADER: &str = "image,format,lambda,texels,psnr_db,normal_deg,zstd_bytes,deflate_bytes,hash,secs";

/// Writes a baseline CSV. `meta` lines are stored as `#` comments.
pub fn write_baseline(path: &Path, meta: &[String], entries: &[Entry]) -> Result<()> {
    let mut s = String::new();
    for m in meta {
        let _ = writeln!(s, "# {m}");
    }
    let _ = writeln!(s, "{HEADER}");
    for e in entries {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{:016x},{:.4}",
            e.image,
            e.format.name(),
            e.lambda,
            e.texels,
            if e.psnr.is_finite() {
                format!("{:.6}", e.psnr)
            } else {
                "inf".into()
            },
            e.normal_deg.map(|d| format!("{d:.6}")).unwrap_or_default(),
            e.zstd_bytes,
            e.deflate_bytes,
            e.hash,
            e.secs,
        );
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, s).with_context(|| format!("writing {}", path.display()))
}

/// Reads a baseline CSV written by [`write_baseline`]. Returns (meta lines, entries).
pub fn read_baseline(path: &Path) -> Result<(Vec<String>, Vec<Entry>)> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "reading baseline {} (create it with `mipple check --update-baseline`)",
            path.display()
        )
    })?;
    let mut meta = Vec::new();
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if let Some(m) = line.strip_prefix('#') {
            meta.push(m.trim().to_string());
            continue;
        }
        if line.trim().is_empty() || line == HEADER {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let bad = || format!("{}:{}: malformed baseline row", path.display(), n + 1);
        if f.len() != 10 {
            bail!(bad());
        }
        out.push(Entry {
            image: f[0].to_string(),
            format: f[1].parse().map_err(anyhow::Error::msg).with_context(bad)?,
            lambda: f[2].parse().with_context(bad)?,
            texels: f[3].parse().with_context(bad)?,
            psnr: if f[4] == "inf" {
                f64::INFINITY
            } else {
                f[4].parse().with_context(bad)?
            },
            normal_deg: if f[5].is_empty() {
                None
            } else {
                Some(f[5].parse().with_context(bad)?)
            },
            zstd_bytes: f[6].parse().with_context(bad)?,
            deflate_bytes: f[7].parse().with_context(bad)?,
            hash: u64::from_str_radix(f[8], 16).with_context(bad)?,
            secs: f[9].parse().with_context(bad)?,
        });
    }
    Ok((meta, out))
}

/// Pooled point for one (format, λ): PSNR of the summed squared error, plus summed sizes.
#[derive(Clone, Copy, Debug, Default)]
struct Pooled {
    psnr: f64,
    zstd: usize,
    deflate: usize,
    normal_deg: Option<f64>,
    secs: f64,
    identical: usize,
    n: usize,
}

/// Cap for lossless results, so pooled curves stay finite.
const PSNR_CAP: f64 = 99.0;

fn mse_of(psnr: f64) -> f64 {
    if psnr.is_finite() && psnr < PSNR_CAP {
        255.0 * 255.0 / 10f64.powf(psnr / 10.0)
    } else {
        0.0
    }
}

fn psnr_of(mse: f64) -> f64 {
    if mse <= 0.0 {
        PSNR_CAP
    } else {
        (10.0 * (255.0 * 255.0 / mse).log10()).min(PSNR_CAP)
    }
}

fn pool(entries: &[&Entry]) -> Pooled {
    let mut p = Pooled::default();
    let (mut sse, mut texels) = (0.0, 0.0);
    let (mut nsum, mut nn) = (0.0, 0usize);
    for e in entries {
        sse += mse_of(e.psnr) * e.texels as f64;
        texels += e.texels as f64;
        p.zstd += e.zstd_bytes;
        p.deflate += e.deflate_bytes;
        p.secs += e.secs;
        p.n += 1;
        if let Some(d) = e.normal_deg {
            nsum += d;
            nn += 1;
        }
    }
    p.psnr = psnr_of(sse / texels.max(1.0));
    p.normal_deg = (nn > 0).then(|| nsum / nn as f64);
    p
}

/// Bjøntegaard delta rate (%) of `test` vs `reference`; points are (bytes, PSNR dB).
///
/// Each curve is reduced to its Pareto front (points no other point beats on both size and
/// quality: a dominated operating point is never worth shipping), then ln(rate) is interpolated
/// piecewise-linearly in PSNR and averaged over the overlapping PSNR range. Negative = fewer bytes
/// at equal quality. `None` if a front has fewer than 2 points or the fronts don't overlap.
///
/// The classic least-squares cubic fit is avoided on purpose: RDO sweeps are often non-monotonic
/// (a light λ can *raise* PSNR) and can span well under 1 dB, where a cubic swings wildly; on the
/// BC1 synthetic check it reported +15% for a curve that was smaller at every matched quality.
pub fn bd_rate(reference: &[(f64, f64)], test: &[(f64, f64)]) -> Option<f64> {
    /// Pareto front sorted by ascending PSNR (and therefore ascending rate).
    fn front(p: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = p.iter().copied().filter(|&(r, q)| r > 0.0 && q.is_finite()).collect();
        // Highest PSNR first; keep a point only if it is smaller than everything above it.
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.total_cmp(&b.0)));
        let mut out: Vec<(f64, f64)> = Vec::new();
        for pt in v {
            if out
                .last()
                .is_none_or(|last: &(f64, f64)| pt.0 < last.0 && pt.1 < last.1)
            {
                out.push(pt);
            }
        }
        out.reverse();
        out
    }
    /// ln(rate) at PSNR `q` by linear interpolation on a front (q within its range).
    fn ln_rate_at(f: &[(f64, f64)], q: f64) -> f64 {
        let k = f.partition_point(|p| p.1 < q).clamp(1, f.len() - 1);
        let ((r0, q0), (r1, q1)) = (f[k - 1], f[k]);
        let t = ((q - q0) / (q1 - q0)).clamp(0.0, 1.0);
        r0.ln() + t * (r1.ln() - r0.ln())
    }
    let (r, t) = (front(reference), front(test));
    if r.len() < 2 || t.len() < 2 {
        return None;
    }
    let lo = r[0].1.max(t[0].1);
    let hi = r[r.len() - 1].1.min(t[t.len() - 1].1);
    if hi - lo < 1e-6 {
        return None;
    }
    // Both curves are piecewise linear in ln(rate), so integrating exactly between every breakpoint
    // of either front (trapezoids) gives the exact mean difference.
    let mut xs: Vec<f64> = r.iter().chain(&t).map(|p| p.1).filter(|&q| q > lo && q < hi).collect();
    xs.extend([lo, hi]);
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let diff_at = |q: f64| ln_rate_at(&t, q) - ln_rate_at(&r, q);
    let area: f64 = xs
        .windows(2)
        .map(|w| 0.5 * (diff_at(w[0]) + diff_at(w[1])) * (w[1] - w[0]))
        .sum();
    Some(((area / (hi - lo)).exp() - 1.0) * 100.0)
}

/// Result of comparing a run with a baseline.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Human-readable table.
    pub text: String,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

fn pct(now: f64, base: f64) -> f64 {
    if base == 0.0 { 0.0 } else { 100.0 * (now / base - 1.0) }
}

/// Compares `current` with `baseline` under `th`.
pub fn compare(baseline: &[Entry], current: &[Entry], th: &Thresholds) -> Result<Report> {
    let base: BTreeMap<_, &Entry> = baseline.iter().map(|e| (e.key(), e)).collect();
    let mut missing = Vec::new();
    for e in current {
        if !base.contains_key(&e.key()) {
            missing.push(format!("{} {} λ={}", e.image, e.format.name(), e.lambda));
        }
    }
    if !missing.is_empty() {
        bail!(
            "the baseline doesn't cover {} of this run's configurations (e.g. {}); the corpus, formats or \
             λ list changed. Re-create it with `mipple check --update-baseline`.",
            missing.len(),
            missing[0]
        );
    }

    let mut rep = Report::default();
    let t = &mut rep.text;
    let mut formats: Vec<Format> = Vec::new();
    for e in current {
        if !formats.contains(&e.format) {
            formats.push(e.format);
        }
    }
    let (mut secs_base, mut secs_now) = (0.0, 0.0);
    let (mut same_all, mut n_all) = (0, 0);
    for fmt in formats {
        // Group by λ.
        let mut lambdas: Vec<f32> = current.iter().filter(|e| e.format == fmt).map(|e| e.lambda).collect();
        lambdas.sort_by(f32::total_cmp);
        lambdas.dedup();
        let mut rows = Vec::new();
        for &l in &lambdas {
            let now: Vec<&Entry> = current.iter().filter(|e| e.format == fmt && e.lambda == l).collect();
            let then: Vec<&Entry> = now.iter().map(|e| base[&e.key()]).collect();
            let mut pn = pool(&now);
            let pb = pool(&then);
            pn.identical = now.iter().zip(&then).filter(|(a, b)| a.hash == b.hash).count();
            rows.push((l, pb, pn, now, then));
        }
        let curve = |sel: fn(&Pooled) -> usize, which: usize| -> Vec<(f64, f64)> {
            rows.iter()
                .map(|r| {
                    let p = if which == 0 { &r.1 } else { &r.2 };
                    (sel(p) as f64, p.psnr)
                })
                .collect()
        };
        let bd_z = bd_rate(&curve(|p| p.zstd, 0), &curve(|p| p.zstd, 1));
        let bd_d = bd_rate(&curve(|p| p.deflate, 0), &curve(|p| p.deflate, 1));
        let same: usize = rows.iter().map(|r| r.2.identical).sum();
        let n: usize = rows.iter().map(|r| r.2.n).sum();
        same_all += same;
        n_all += n;
        let fmt_bd = |v: Option<f64>| v.map_or("n/a".into(), |v| format!("{v:+.2}%"));
        let _ = writeln!(
            t,
            "{}  BD-rate zstd {}  deflate {}  ({same}/{n} encodes byte-identical)",
            fmt.name().to_uppercase(),
            fmt_bd(bd_z),
            fmt_bd(bd_d),
        );
        let _ = writeln!(
            t,
            "  {:>5}  {:>22}  {:>24}  {:>24}  {:>16}",
            "λ", "PSNR dB (base → now)", "zstd bytes (base → now)", "deflate (base → now)", "time s"
        );
        for (l, pb, pn, _, _) in &rows {
            let normal = match (pb.normal_deg, pn.normal_deg) {
                (Some(a), Some(b)) if fmt != Format::Bc4 => format!("  normal {a:.3}° → {b:.3}°"),
                _ => String::new(),
            };
            let _ = writeln!(
                t,
                "  {:>5}  {:>6.2} → {:>6.2} {:>+6.2}  {:>8} → {:>8} {:>+6.2}%  {:>8} → {:>8} {:>+6.2}%  {:>6.2} → {:>6.2}{}",
                l,
                pb.psnr,
                pn.psnr,
                pn.psnr - pb.psnr,
                pb.zstd,
                pn.zstd,
                pct(pn.zstd as f64, pb.zstd as f64),
                pb.deflate,
                pn.deflate,
                pct(pn.deflate as f64, pb.deflate as f64),
                pb.secs,
                pn.secs,
                normal,
            );
            secs_base += pb.secs;
            secs_now += pn.secs;
        }
        let name = fmt.name().to_uppercase();
        match bd_z {
            Some(v) if v > th.max_bd_rate => rep.failures.push(format!(
                "{name}: BD-rate {v:+.2}% (Zstd) exceeds +{:.2}%",
                th.max_bd_rate
            )),
            // A single λ (ASTC, which has no RDO) is no curve: the λ=0 gates below cover it.
            None if same < n && rows.len() > 1 => rep.failures.push(format!(
                "{name}: BD-rate undefined (size/PSNR curves no longer overlap)"
            )),
            _ => {}
        }
        if let Some((_, pb, pn, now, then)) = rows.iter().find(|r| r.0 == 0.0) {
            let drop = pb.psnr - pn.psnr;
            if drop > th.max_psnr_drop {
                rep.failures.push(format!(
                    "{name} λ=0: pooled PSNR dropped {drop:.3} dB (limit {:.2})",
                    th.max_psnr_drop
                ));
            }
            let growth = pct(pn.zstd as f64, pb.zstd as f64);
            if growth > th.max_size_growth {
                rep.failures.push(format!(
                    "{name} λ=0: Zstd size grew {growth:+.2}% (limit +{:.2}%)",
                    th.max_size_growth
                ));
            }
            for (a, b) in now.iter().zip(then) {
                let d = b.psnr.min(PSNR_CAP) - a.psnr.min(PSNR_CAP);
                if d > th.max_image_psnr_drop {
                    rep.failures.push(format!(
                        "{name} λ=0 {}: PSNR dropped {d:.2} dB (limit {:.2})",
                        a.image, th.max_image_psnr_drop
                    ));
                }
            }
        }
        let _ = writeln!(t);
    }
    let ratio = if secs_base > 0.0 { secs_now / secs_base } else { 1.0 };
    let _ = writeln!(
        t,
        "Total encode time {secs_base:.2} s → {secs_now:.2} s ({ratio:.2}×); {same_all}/{n_all} encodes byte-identical."
    );
    match th.max_slowdown {
        Some(m) if ratio > m => rep
            .failures
            .push(format!("encode time {ratio:.2}× the baseline (limit {m:.2}×)")),
        None if ratio > 1.5 => rep.warnings.push(format!(
            "encode time {ratio:.2}× the baseline (not gated; timings vary between machines)"
        )),
        _ => {}
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bd_rate_identity_and_scaling() {
        let r = [(1000.0, 30.0), (1500.0, 33.0), (2200.0, 36.0), (3000.0, 39.0)];
        assert!(bd_rate(&r, &r).unwrap().abs() < 1e-9);
        let t: Vec<_> = r.iter().map(|&(b, p)| (b * 0.9, p)).collect();
        assert!((bd_rate(&r, &t).unwrap() + 10.0).abs() < 1e-6);
        let up: Vec<_> = r.iter().map(|&(b, p)| (b * 1.05, p)).collect();
        assert!((bd_rate(&r, &up).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn bd_rate_ignores_dominated_points() {
        // A narrow, non-monotonic sweep (light RDO raising PSNR), like BC1 on the synthetic set.
        let base = [
            (111019.0, 38.50),
            (106644.0, 38.57),
            (104362.0, 38.53),
            (100684.0, 38.41),
            (93148.0, 37.96),
        ];
        // Smaller than the base at every matched PSNR, so the BD-rate must be negative.
        let new = [
            (113650.0, 38.79),
            (108464.0, 38.70),
            (106373.0, 38.72),
            (102265.0, 38.58),
            (94702.0, 38.12),
        ];
        let bd = bd_rate(&base, &new).unwrap();
        assert!(bd < 0.0, "bd {bd}");
    }

    #[test]
    fn pooled_psnr() {
        assert_eq!(psnr_of(0.0), PSNR_CAP);
        assert!((psnr_of(mse_of(40.0)) - 40.0).abs() < 1e-9);
    }

    fn entry(format: Format, lambda: f32, psnr: f64, zstd: usize) -> Entry {
        Entry {
            image: "a".into(),
            format,
            lambda,
            texels: 1024,
            psnr,
            normal_deg: None,
            zstd_bytes: zstd,
            deflate_bytes: zstd,
            hash: zstd as u64,
            secs: 0.1,
        }
    }

    fn sweep(scale: f64, psnr_shift: f64) -> Vec<Entry> {
        [(0.0, 45.0, 1000), (1.0, 42.0, 800), (2.0, 40.0, 700), (4.0, 38.0, 640)]
            .iter()
            .map(|&(l, p, z)| entry(Format::Bc7, l, p + psnr_shift, (z as f64 * scale).round() as usize))
            .collect()
    }

    #[test]
    fn compare_gates() {
        let th = Thresholds::default();
        let base = sweep(1.0, 0.0);
        assert!(compare(&base, &base, &th).unwrap().passed());
        // 0.5% smaller everywhere: fine.
        assert!(compare(&base, &sweep(0.995, 0.0), &th).unwrap().passed());
        // 5% bigger: BD-rate and λ0 size gates trip.
        let r = compare(&base, &sweep(1.05, 0.0), &th).unwrap();
        assert_eq!(r.failures.len(), 2, "{:?}", r.failures);
        // 1 dB worse everywhere: BD-rate, λ0 pooled and per-image PSNR gates trip.
        let r = compare(&base, &sweep(1.0, -1.0), &th).unwrap();
        assert_eq!(r.failures.len(), 3, "{:?}", r.failures);
        // Different λ list: refuse to compare.
        assert!(compare(&base[..2], &base, &th).is_err());
        // A format with only a λ = 0 row (ASTC) has no curve: a changed encode is judged by the
        // λ = 0 gates alone.
        let one = [entry(Format::Astc6x6, 0.0, 40.0, 1000)];
        assert!(
            compare(&one, &[entry(Format::Astc6x6, 0.0, 40.05, 999)], &th)
                .unwrap()
                .passed()
        );
        let r = compare(&one, &[entry(Format::Astc6x6, 0.0, 39.0, 1000)], &th).unwrap();
        assert_eq!(r.failures.len(), 2, "{:?}", r.failures);
    }
}
