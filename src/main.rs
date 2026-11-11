//! `mipple` command-line tool.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use mipple::bench::{self, BenchConfig};
use mipple::gpu::Gpu;
use mipple::pipeline::{self, EncodeOptions};
use mipple::{EntropyMode, Format, RdoParams, Surface, dds, ktx2, metrics, synth};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "mipple",
    version,
    about = "Rate-distortion optimized GPU texture encoder (BC1/BC4/BC5/BC7 → KTX2)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode an image to a KTX2 file.
    Encode(EncodeArgs),
    /// Decode one mip level of a KTX2 file to PNG.
    Decode {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Mip level to decode.
        #[arg(long, default_value_t = 0)]
        level: usize,
        /// Reconstruct Z (blue) for BC5 normal maps.
        #[arg(long)]
        normal: bool,
    },
    /// Print a KTX2 file's header, levels and key/value data.
    Info { input: PathBuf },
    /// Compare two images (PSNR, SSIM, optional normal-map angular error).
    Compare {
        a: PathBuf,
        b: PathBuf,
        /// Also report normal-map angular error.
        #[arg(long)]
        normal: bool,
    },
    /// Benchmark size vs quality across a corpus and a lambda sweep.
    Bench(BenchArgs),
    /// Fast regression check: encode a small fixed corpus and compare with a stored baseline.
    /// Exits non-zero on a regression.
    Check(CheckArgs),
    /// Score any encoder's output (KTX2 or DDS, top level) against its source image,
    /// with the same metrics and back-end compressors as `bench`.
    Score {
        source: PathBuf,
        encoded: PathBuf,
        /// Also report normal-map angular error.
        #[arg(long)]
        normal: bool,
        #[arg(long, default_value_t = 19)]
        zstd_level: i32,
        #[arg(long, default_value_t = 9)]
        deflate_level: u32,
        /// Label for the CSV row (e.g. "bc7enc_rdo z=1").
        #[arg(long)]
        label: Option<String>,
        /// Encode time to record in the CSV row.
        #[arg(long)]
        secs: Option<f64>,
        /// Print one CSV row: label,image,format,width,height,psnr,ssim,normal_deg,raw,zstd,deflate,zstd_bpt,deflate_bpt,secs
        #[arg(long)]
        csv: bool,
    },
    /// Write the procedural test-image set.
    Synth {
        #[arg(short, long, default_value = "test-images/synthetic")]
        output: PathBuf,
        #[arg(long, default_value_t = 512)]
        size: u32,
    },
    /// Report the GPU mipple would use and which formats it can encode there.
    Gpu,
}

#[derive(Args, Clone)]
struct RdoArgs {
    /// LZ window the rate model searches, in bytes [default: 16384 for BC7, 8192 for BC1, 3072
    /// for BC5, 2048 for BC4].
    #[arg(long)]
    window: Option<usize>,
    /// Previous blocks to try copying from (also capped by --window) [default: 128 for BC4/BC5
    /// and 16 for BC7 with the localized search, 256 otherwise].
    #[arg(long)]
    max_back: Option<usize>,
    /// Nearest previous blocks to also try partial byte-span copies from [default: 8 for BC1/BC4,
    /// 0 for BC5/BC7].
    #[arg(long)]
    span_back: Option<usize>,
    /// Previous blocks whose single fields are tried in the chosen block after the search
    /// (0 disables) [default: 256 for BC1/BC4/BC7, 0 for BC5].
    #[arg(long)]
    refine_back: Option<usize>,
    /// Beyond --max-back, earlier blocks whose endpoints are tried re-indexed, each distinct pair
    /// once (BC7 also tries whole-block copies; 0 disables; capped by --window) [default: 1024
    /// for BC1/BC7, 0 otherwise].
    #[arg(long)]
    far_back: Option<usize>,
    /// Localized search: beyond --max-back, earlier blocks of the chunk picked by the learned
    /// ranker for the full search (0 disables) [default: 16 for BC5, 8 for BC7, 32 for BC4, 0 otherwise;
    /// BC4/BC5/BC7 only].
    #[arg(long)]
    local: Option<usize>,
    /// Bytes per independently optimized chunk (the unit of parallelism; matches are not
    /// searched across chunks) [default: 65536 for BC7, 16384 otherwise].
    #[arg(long)]
    chunk_bytes: Option<usize>,
    /// Distortion multiplier for flat blocks (protects gradients).
    #[arg(long, default_value_t = 4.0)]
    smooth_scale: f32,
    /// Channel standard deviation at which a block gets half the extra weight; flatter blocks
    /// approach --smooth-scale, busier ones 1.
    #[arg(long, alias = "smooth-max-std", default_value_t = 12.0)]
    smooth_std: f32,
    /// Reject candidates whose RMS error exceeds this multiple of the base encoding's.
    #[arg(long, default_value_t = 10.0)]
    max_rms_ratio: f32,
    /// ... but always allow this RMS error per channel.
    #[arg(long, default_value_t = 3.0)]
    min_rms: f32,
}

impl RdoArgs {
    fn params(&self, lambda: f32) -> RdoParams {
        RdoParams {
            lambda,
            window: self.window,
            max_back: self.max_back,
            span_back: self.span_back,
            refine_back: self.refine_back,
            far_back: self.far_back,
            local_k: self.local,
            chunk_bytes: self.chunk_bytes,
            smooth_scale: self.smooth_scale,
            smooth_std: self.smooth_std,
            max_rms_ratio: self.max_rms_ratio,
            min_rms: self.min_rms,
            ..RdoParams::default()
        }
    }
}

#[derive(Args)]
struct EncodeArgs {
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long, value_enum, default_value_t = Format::Bc7)]
    format: Format,
    /// Encoder effort, 0 (fastest) to 9 (best); 5 is balanced.
    #[arg(short, long, default_value_t = mipple::format::DEFAULT_EFFORT, value_parser = clap::value_parser!(u8).range(0..=9))]
    effort: u8,
    /// RDO lambda; 0 disables RDO. Try 0.5 – 4.
    #[arg(short, long, default_value_t = 0.0)]
    lambda: f32,
    /// Treat the input as linear data (default for BC4/BC5).
    #[arg(long, conflicts_with = "srgb")]
    linear: bool,
    /// Treat the input as sRGB color (default for BC1/BC7).
    #[arg(long)]
    srgb: bool,
    /// Input is a tangent-space normal map (renormalized mips; implies linear).
    #[arg(long)]
    normal: bool,
    /// Only encode the top level.
    #[arg(long)]
    no_mips: bool,
    /// What alpha means. transparency: the colour under alpha 0 is bled from the visible colour
    /// around it (no dark mip fringes) and costs no bits; data: an independent channel; auto:
    /// transparency for BC7 colour textures.
    #[arg(long, value_enum, default_value_t = mipple::alpha::AlphaMode::Auto)]
    alpha: mipple::alpha::AlphaMode,
    /// Alpha-test cutoff (0-1) whose coverage every mip keeps, for cutouts (foliage, fences)
    /// that would otherwise thin out at a distance.
    #[arg(long, value_parser = parse_unit)]
    alpha_cutoff: Option<f32>,
    /// Zstd level for KTX2 supercompression; 0 stores raw blocks.
    #[arg(long, default_value_t = 19)]
    zstd: i32,
    /// Entropy-reduced BC7 base encoding (bc7enc `-e` style): off, on, or auto (on when λ ≥ 1).
    #[arg(long, value_enum, default_value_t = EntropyMode::default())]
    entropy: EntropyMode,
    /// Lossless field split of the blocks before Zstd (BC7Prep-style; 4-8% smaller without RDO).
    /// Not standard KTX2: readers must undo it (mipple's does, flagged by the `mipple.prep` key).
    #[arg(long, value_enum, default_value_t = mipple::ktx2::PrepMode::Off)]
    prep: mipple::ktx2::PrepMode,
    /// Base-encode on the CPU only. By default the GPU encodes the formats where its encoder is
    /// faster (see `mipple gpu`); both write the same blocks.
    #[arg(long, conflicts_with = "gpu")]
    cpu: bool,
    /// Base-encode on the GPU for every format it has an encoder for, even where the CPU is faster.
    #[arg(long)]
    gpu: bool,
    #[command(flatten)]
    rdo: RdoArgs,
}

#[derive(Args)]
struct BenchArgs {
    /// Image files or directories (searched recursively for PNG/JPEG/TGA).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(short, long, value_enum, value_delimiter = ',', default_values_t = [Format::Bc7])]
    formats: Vec<Format>,
    #[arg(short, long, value_delimiter = ',', default_values_t = [0.0f32, 0.5, 1.0, 2.0, 4.0])]
    lambdas: Vec<f32>,
    #[arg(short, long, default_value_t = mipple::format::DEFAULT_EFFORT, value_parser = clap::value_parser!(u8).range(0..=9))]
    effort: u8,
    #[arg(long, default_value_t = 19)]
    zstd_level: i32,
    #[arg(long, default_value_t = 9)]
    deflate_level: u32,
    /// Write every row as CSV.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write the per-format summary as Markdown.
    #[arg(long)]
    markdown: Option<PathBuf>,
    /// Entropy-reduced BC7 base encoding (bc7enc `-e` style): off, on, or auto (on when λ ≥ 1).
    #[arg(long, value_enum, default_value_t = EntropyMode::default())]
    entropy: EntropyMode,
    #[command(flatten)]
    rdo: RdoArgs,
}

#[derive(Args)]
struct CheckArgs {
    /// Use the first --limit images of this directory instead of the built-in synthetic set.
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Images to take from --corpus.
    #[arg(long, default_value_t = 4)]
    limit: usize,
    /// Synthetic image size in pixels.
    #[arg(long, default_value_t = mipple::check::DEFAULT_SIZE)]
    size: u32,
    /// Baseline CSV [default: bench-results/check-baseline.csv, or check-baseline-<dir>.csv with --corpus].
    #[arg(long)]
    baseline: Option<PathBuf>,
    /// Write this run as the new baseline instead of comparing.
    #[arg(long)]
    update_baseline: bool,
    #[arg(short, long, value_enum, value_delimiter = ',', default_values_t = Format::ALL)]
    formats: Vec<Format>,
    #[arg(short, long, value_delimiter = ',', default_values_t = mipple::check::DEFAULT_LAMBDAS)]
    lambdas: Vec<f32>,
    #[arg(short, long, default_value_t = mipple::format::DEFAULT_EFFORT, value_parser = clap::value_parser!(u8).range(0..=9))]
    effort: u8,
    /// Fail if a format's BD-rate (Zstd) vs the baseline exceeds this many percent.
    #[arg(long, default_value_t = 1.0)]
    max_bd_rate: f64,
    /// Fail if pooled PSNR at λ=0 drops more than this (dB).
    #[arg(long, default_value_t = 0.10)]
    max_psnr_drop: f64,
    /// Fail if Zstd size at λ=0 grows more than this many percent.
    #[arg(long, default_value_t = 1.0)]
    max_size_growth: f64,
    /// Fail if any single image's PSNR at λ=0 drops more than this (dB).
    #[arg(long, default_value_t = 0.5)]
    max_image_psnr_drop: f64,
    /// Fail if total encode time exceeds this multiple of the baseline (off by default).
    #[arg(long)]
    max_slowdown: Option<f64>,
    /// Entropy-reduced BC7 base encoding (bc7enc `-e` style): off, on, or auto (on when λ ≥ 1).
    #[arg(long, value_enum, default_value_t = EntropyMode::default())]
    entropy: EntropyMode,
    #[command(flatten)]
    rdo: RdoArgs,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Encode(a) => encode(a),
        Cmd::Decode {
            input,
            output,
            level,
            normal,
        } => {
            let tex = load_encoded(&input)?;
            if level >= tex.levels.len() {
                bail!("level {level} out of range (file has {})", tex.levels.len());
            }
            pipeline::decode_level(&tex, level, normal).save_png(&output)?;
            println!(
                "wrote {} ({}x{}, {})",
                output.display(),
                tex.levels[level].width,
                tex.levels[level].height,
                tex.format
            );
            Ok(())
        }
        Cmd::Info { input } => {
            let bytes = std::fs::read(&input)?;
            let h = ktx2::read_header(&bytes)?;
            let fmt = Format::from_vk(h.vk_format).map_or_else(
                || "unsupported".into(),
                |(f, s)| format!("{f}{}", if s { " sRGB" } else { "" }),
            );
            println!(
                "{}: {}x{}, vkFormat {} ({fmt}), {} level(s), supercompression {}",
                input.display(),
                h.width,
                h.height,
                h.vk_format,
                h.level_count,
                match h.scheme {
                    0 => "none",
                    2 => "zstd",
                    _ => "other",
                }
            );
            for (i, (off, len, ulen)) in h.levels.iter().enumerate() {
                println!("  level {i}: offset {off}, {len} bytes ({ulen} uncompressed)");
            }
            for (k, v) in &h.key_values {
                println!("  {k} = {v}");
            }
            println!("  file size {} bytes", bytes.len());
            Ok(())
        }
        Cmd::Compare { a, b, normal } => {
            let (sa, sb) = (Surface::load(&a)?, Surface::load(&b)?);
            if (sa.width, sa.height) != (sb.width, sb.height) {
                bail!(
                    "size mismatch: {}x{} vs {}x{}",
                    sa.width,
                    sa.height,
                    sb.width,
                    sb.height
                );
            }
            let pc = |c: usize| metrics::psnr(&sa, &sb, &[c]);
            println!("PSNR RGB  {:.3} dB", metrics::psnr(&sa, &sb, &[0, 1, 2]));
            println!(
                "PSNR R/G/B/A  {:.3} / {:.3} / {:.3} / {:.3} dB",
                pc(0),
                pc(1),
                pc(2),
                pc(3)
            );
            println!("PSNR luma {:.3} dB", metrics::luma_psnr(&sa, &sb));
            println!("SSIM luma {:.5}", metrics::ssim_luma(&sa, &sb));
            if normal {
                println!("Normal angular error {:.3}°", metrics::normal_angle_error_deg(&sa, &sb));
            }
            Ok(())
        }
        Cmd::Bench(a) => run_bench(a),
        Cmd::Check(a) => run_check(a),
        Cmd::Score {
            source,
            encoded,
            normal,
            zstd_level,
            deflate_level,
            label,
            secs,
            csv,
        } => {
            let src = Surface::load(&source)?;
            let tex = load_encoded(&encoded)?;
            let s = bench::score(&src, &tex, normal, zstd_level, deflate_level)?;
            if csv {
                println!(
                    "{},{},{},{},{},{:.4},{:.5},{},{},{},{},{:.4},{:.4},{}",
                    label.unwrap_or_default(),
                    source.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                    tex.format,
                    tex.width,
                    tex.height,
                    s.psnr,
                    s.ssim,
                    s.normal_deg.map(|d| format!("{d:.4}")).unwrap_or_default(),
                    s.raw_bytes,
                    s.zstd_bytes,
                    s.deflate_bytes,
                    s.zstd_bpt,
                    s.deflate_bpt,
                    secs.map(|t| format!("{t:.6}")).unwrap_or_default(),
                );
            } else {
                println!("{} vs {}: {}", encoded.display(), source.display(), tex.format);
                println!("  PSNR {:.3} dB, SSIM {:.5}", s.psnr, s.ssim);
                if let Some(d) = s.normal_deg {
                    println!("  normal angular error {d:.3}°");
                }
                println!(
                    "  raw {} bytes, zstd-{} {} bytes ({:.3} bpt), deflate-{} {} bytes ({:.3} bpt)",
                    s.raw_bytes, zstd_level, s.zstd_bytes, s.zstd_bpt, deflate_level, s.deflate_bytes, s.deflate_bpt
                );
            }
            Ok(())
        }
        Cmd::Synth { output, size } => {
            for p in synth::write_all(&output, size)? {
                println!("wrote {}", p.display());
            }
            Ok(())
        }
        Cmd::Gpu => {
            if !Gpu::built() {
                println!("GPU backend not built (cargo build --release --features gpu)");
                return Ok(());
            }
            let Some(gpu) = Gpu::shared() else {
                println!("no usable GPU adapter: every format encodes on the CPU");
                return Ok(());
            };
            let i = gpu.info();
            println!(
                "{} ({}, {}{})",
                i.name,
                i.backend,
                i.device_type,
                if i.driver.is_empty() {
                    String::new()
                } else {
                    format!(", {}", i.driver)
                }
            );
            println!("  compute shaders: {}", if i.compute { "yes" } else { "no" });
            println!("  BC texture sampling: {}", if i.bc_textures { "yes" } else { "no" });
            for f in Format::ALL {
                println!(
                    "  {f}: {}{}",
                    if !gpu.encodes(f) {
                        "CPU (no GPU encoder yet)"
                    } else if Gpu::preferred(f, mipple::BlockOpts::default()) {
                        "GPU"
                    } else {
                        "CPU by default (faster); GPU encoder with --gpu"
                    },
                    if gpu.encodes(f) && !Gpu::has_kernel(f, mipple::BlockOpts::from(9u8)) {
                        format!(" ({})", Gpu::kernel_scope(f))
                    } else {
                        String::new()
                    }
                );
            }
            Ok(())
        }
    }
}

/// Loads a KTX2 or DDS file, detected by its magic bytes.
fn load_encoded(path: &std::path::Path) -> Result<mipple::EncodedTexture> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.starts_with(b"DDS ") {
        dds::read(&bytes)
    } else {
        ktx2::read(&bytes)
    }
}

/// A number in 0-1 (`--alpha-cutoff`).
fn parse_unit(s: &str) -> std::result::Result<f32, String> {
    match s.parse::<f32>() {
        Ok(v) if (0.0..=1.0).contains(&v) => Ok(v),
        _ => Err(format!("expected a number from 0 to 1, got {s}")),
    }
}

fn encode(a: EncodeArgs) -> Result<()> {
    let surface = Surface::load(&a.input)?;
    let mut opts = EncodeOptions::new(a.format);
    opts.effort = a.effort;
    opts.entropy = a.entropy;
    opts.normal_map = a.normal;
    opts.srgb = if a.normal || a.linear {
        false
    } else if a.srgb {
        true
    } else {
        a.format.default_srgb()
    };
    opts.mips = !a.no_mips;
    opts.alpha = a.alpha;
    opts.alpha_cutoff = a.alpha_cutoff;
    opts.gpu = if a.cpu {
        mipple::GpuMode::Off
    } else if a.gpu {
        mipple::GpuMode::On
    } else {
        mipple::GpuMode::Auto
    };
    opts.rdo = a.rdo.params(a.lambda);
    opts.rdo.verify_zstd_level = (a.zstd > 0).then_some(a.zstd);

    // Start rayon's worker threads before timing, as a long-running host application would.
    rayon::broadcast(|_| ());
    let t = Instant::now();
    let (tex, report) = pipeline::encode(&surface, &opts);
    let encode_secs = t.elapsed().as_secs_f64();
    let zstd = (a.zstd > 0).then_some(a.zstd);
    let sc_params = format!(
        "--format {} --effort {} --lambda {} --entropy {} --window {} --zstd {}",
        a.format,
        a.effort,
        a.lambda,
        format!("{:?}", a.entropy).to_lowercase(),
        opts.rdo.window_for(a.format),
        a.zstd
    );
    let is_dds = a.output.extension().is_some_and(|e| e.eq_ignore_ascii_case("dds"));
    let bytes = if is_dds {
        dds::write(&tex)
    } else {
        ktx2::write(&tex, zstd, Some(&sc_params), a.prep)?
    };
    std::fs::write(&a.output, &bytes).with_context(|| format!("writing {}", a.output.display()))?;

    let raw: usize = tex.levels.iter().map(|l| l.data.len()).sum();
    let dec = pipeline::decode_level(&tex, 0, a.normal);
    let channels = a.format.metric_channels(surface.has_alpha());
    let texels: f64 = tex.levels.iter().map(|l| l.width as f64 * l.height as f64).sum();
    println!(
        "{} -> {}: {}x{} {}{}, {} level(s)",
        a.input.display(),
        a.output.display(),
        tex.width,
        tex.height,
        tex.format,
        if tex.srgb { " sRGB" } else { "" },
        tex.levels.len()
    );
    println!(
        "  blocks {} bytes, file {} bytes ({:.3} bits/texel incl. mips)",
        raw,
        bytes.len(),
        bytes.len() as f64 * 8.0 / texels
    );
    println!(
        "  level 0 PSNR {:.2} dB{}",
        metrics::psnr(&surface, &dec, &channels),
        if a.normal {
            format!(", normal error {:.2}°", metrics::normal_angle_error_deg(&surface, &dec))
        } else {
            String::new()
        }
    );
    if opts.rdo.enabled() {
        let r = &report.rdo;
        println!(
            "  RDO λ={}: {:.1}% of blocks changed, model estimate {:.0} -> {:.0} bits",
            a.lambda,
            100.0 * r.changed as f64 / r.blocks.max(1) as f64,
            r.est_bits_before,
            r.est_bits_after
        );
    }
    match &report.gpu {
        Some(name) => println!("  base encode: GPU ({name})"),
        None if a.cpu => println!("  base encode: CPU (--cpu)"),
        None if !Gpu::built() => println!("  base encode: CPU (built without the gpu feature)"),
        None if !Gpu::has_encoder(a.format) => println!("  base encode: CPU (no GPU {} encoder yet)", a.format),
        None if !Gpu::has_kernel(a.format, opts.block_opts()) => {
            println!(
                "  base encode: CPU (the GPU {} encoder covers {})",
                a.format,
                Gpu::kernel_scope(a.format)
            );
        }
        None if a.format == Format::Bc7 && report.rdo_secs > 0.0 => {
            println!("  base encode: CPU (BC7 RDO starts from CPU-only bases)");
        }
        None if !a.gpu && !Gpu::preferred(a.format, opts.block_opts()) => {
            println!(
                "  base encode: CPU (faster than the GPU {} encoder; --gpu forces it)",
                a.format
            );
        }
        None => println!("  base encode: CPU (no usable GPU)"),
    }
    println!(
        "  time: base {:.6}s, rdo {:.6}s, encode {:.6}s, total {:.6}s",
        report.base_secs,
        report.rdo_secs,
        encode_secs,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_bench(a: BenchArgs) -> Result<()> {
    let inputs = bench::collect_inputs(&a.inputs)?;
    if inputs.is_empty() {
        bail!("no PNG/JPEG/TGA images found");
    }
    let mut lambdas = a.lambdas.clone();
    if !lambdas.contains(&0.0) {
        lambdas.insert(0, 0.0);
    }
    let cfg = BenchConfig {
        inputs,
        formats: a.formats.clone(),
        lambdas,
        effort: a.effort,
        entropy: a.entropy,
        rdo: RdoParams {
            verify_zstd_level: Some(a.zstd_level),
            ..a.rdo.params(0.0)
        },
        zstd_level: a.zstd_level,
        deflate_level: a.deflate_level,
    };
    println!(
        "Benchmarking {} image(s) × {} format(s) × {} lambda(s), effort {}, zstd -{}, deflate -{}",
        cfg.inputs.len(),
        cfg.formats.len(),
        cfg.lambdas.len(),
        cfg.effort,
        cfg.zstd_level,
        cfg.deflate_level
    );
    let rows = bench::run(&cfg, |r| println!("{}", bench::row_line(r)))?;
    let md = bench::summary_markdown(&rows);
    println!("\n{md}");
    if let Some(p) = &a.csv {
        bench::write_csv(&rows, p)?;
        println!("wrote {}", p.display());
    }
    if let Some(p) = &a.markdown {
        std::fs::write(p, &md)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

fn run_check(a: CheckArgs) -> Result<()> {
    use mipple::check;
    let images = check::corpus(a.corpus.as_deref(), a.limit, a.size)?;
    let baseline = a
        .baseline
        .clone()
        .unwrap_or_else(|| check::default_baseline_path(a.corpus.as_deref()));
    let mut lambdas = a.lambdas.clone();
    if !lambdas.contains(&0.0) {
        lambdas.insert(0, 0.0);
    }
    let zstd_level = 19;
    let cfg = BenchConfig {
        inputs: Vec::new(),
        formats: a.formats.clone(),
        lambdas,
        effort: a.effort,
        entropy: a.entropy,
        rdo: RdoParams {
            verify_zstd_level: Some(zstd_level),
            ..a.rdo.params(0.0)
        },
        zstd_level,
        deflate_level: 9,
    };
    let corpus_desc = match &a.corpus {
        Some(d) => format!(
            "{} (first {})",
            d.file_name().and_then(|n| n.to_str()).unwrap_or("corpus"),
            a.limit
        ),
        None => format!("synthetic {}x{}", a.size, a.size),
    };
    println!(
        "Checking {} image(s) [{corpus_desc}] × {} format(s) × λ {:?}, effort {}{}",
        images.len(),
        cfg.formats.len(),
        cfg.lambdas,
        cfg.effort,
        match cfg.entropy {
            EntropyMode::Off => ", entropy off",
            EntropyMode::On => ", entropy on",
            EntropyMode::Auto => ", entropy auto",
        }
    );
    let t = Instant::now();
    let entries = check::run(&images, &cfg, |_| {})?;
    println!("encoded in {:.1} s\n", t.elapsed().as_secs_f64());
    if a.update_baseline {
        let meta = vec![
            format!("mipple check baseline, mipple {}", env!("CARGO_PKG_VERSION")),
            format!(
                "corpus: {corpus_desc}; effort {}; entropy {:?}; zstd -{zstd_level}, deflate -9",
                a.effort, a.entropy
            ),
            "regenerate with: mipple check --update-baseline (same corpus flags)".to_string(),
        ];
        check::write_baseline(&baseline, &meta, &entries)?;
        println!("wrote baseline {} ({} entries)", baseline.display(), entries.len());
        return Ok(());
    }
    let (meta, base) = check::read_baseline(&baseline)?;
    println!(
        "baseline {}: {}\n",
        baseline.display(),
        meta.get(1).map_or("", String::as_str)
    );
    let th = check::Thresholds {
        max_bd_rate: a.max_bd_rate,
        max_psnr_drop: a.max_psnr_drop,
        max_size_growth: a.max_size_growth,
        max_image_psnr_drop: a.max_image_psnr_drop,
        max_slowdown: a.max_slowdown,
    };
    let rep = check::compare(&base, &entries, &th)?;
    print!("{}", rep.text);
    for w in &rep.warnings {
        println!("warning: {w}");
    }
    if rep.passed() {
        println!("\nPASS: no regressions vs baseline");
        Ok(())
    } else {
        println!("\nFAIL: {} regression(s)", rep.failures.len());
        for f in &rep.failures {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
