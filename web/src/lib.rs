//! mipple in the browser: the WebAssembly side of the demo page in `web/www`.
//!
//! The page decodes an image ([`decode_image`]), then calls [`encode`] from a Web Worker whenever
//! a control changes. [`encode`] runs [`mipple::pipeline::encode`]'s stages and
//! [`mipple::ktx2::write`] with the options `mipple encode` derives from its flags, so the .ktx2
//! the page offers is byte-identical to the CLI's for the command [`Settings::command`] prints.
//!
//! wasm32-unknown-unknown has no threads: rayon runs every parallel loop on the calling thread.
//! Output does not depend on the thread count (RDO chunks are fixed in bytes), only the time does.
//!
//! When the browser has WebGPU, base blocks come from mipple's GPU kernels where one covers the
//! settings ([`mipple::pipeline::gpu_eligible`]): the same blocks as the CPU encoder, much sooner
//! than one wasm thread makes them. Each encode first checks a sample of blocks against the CPU
//! and stays on the CPU for good if a browser's shader compiler ever disagrees.

use anyhow::{Context, Result, ensure};
use mipple::format::{DEFAULT_EFFORT, MAX_EFFORT};
use mipple::gpu::Gpu;
use mipple::ktx2::{self, PrepMode};
use mipple::metrics::normal_from_rg;
use mipple::pipeline::{self, EncodeOptions};
use mipple::{EntropyMode, Format, RdoParams, Surface, bench};
use std::borrow::Cow;
use std::cell::Cell;
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// Zstd level `mipple encode` uses by default.
const DEFAULT_ZSTD: i32 = 19;

fn js_error(e: &anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// An RGBA8 image, possibly downscaled from a larger source.
#[wasm_bindgen]
pub struct Image {
    surface: Surface,
    /// Width of the decoded file before fitting it within the size limit.
    #[wasm_bindgen(readonly)]
    pub source_width: u32,
    /// Height of the decoded file before fitting it within the size limit.
    #[wasm_bindgen(readonly)]
    pub source_height: u32,
}

#[wasm_bindgen]
impl Image {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.surface.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.surface.height
    }

    /// The pixels, row-major RGBA8 (a copy, as a `Uint8Array`).
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        self.surface.pixels.iter().flatten().copied().collect()
    }
}

/// Decodes a PNG, JPEG or TGA file with the decoders `mipple encode` uses, then fits it within
/// `max_size` x `max_size` (Lanczos, aspect kept; 0 keeps the full size).
#[wasm_bindgen(js_name = decodeImage)]
pub fn decode_image(bytes: &[u8], max_size: u32) -> Result<Image, JsError> {
    decode(bytes, max_size).map_err(|e| js_error(&e))
}

/// Fits RGBA8 pixels within `max_size` x `max_size`, for images the browser decoded (formats the
/// `image` crate is not built with here, such as WebP or AVIF).
#[wasm_bindgen(js_name = fitImage)]
pub fn fit_image(rgba: Vec<u8>, width: u32, height: u32, max_size: u32) -> Result<Image, JsError> {
    image::RgbaImage::from_raw(width, height, rgba)
        .context("pixel buffer does not match the image size")
        .map(|img| fit(img, max_size))
        .map_err(|e| js_error(&e))
}

fn decode(bytes: &[u8], max_size: u32) -> Result<Image> {
    let img = image::load_from_memory(bytes).context("not a PNG, JPEG or TGA file")?;
    Ok(fit(img.to_rgba8(), max_size))
}

fn fit(img: image::RgbaImage, max_size: u32) -> Image {
    let (sw, sh) = img.dimensions();
    let img = if max_size > 0 && sw.max(sh) > max_size {
        let scale = max_size as f64 / sw.max(sh) as f64;
        let w = ((sw as f64 * scale).round() as u32).max(1);
        let h = ((sh as f64 * scale).round() as u32).max(1);
        image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let (width, height) = img.dimensions();
    Image {
        surface: Surface {
            width,
            height,
            pixels: img.pixels().map(|p| p.0).collect(),
        },
        source_width: sw,
        source_height: sh,
    }
}

/// Encoder settings: the knobs of `mipple encode`, with its defaults.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Settings {
    format: Format,
    entropy: EntropyMode,
    /// `-e`, 0 (fastest) to 9 (best).
    pub effort: u8,
    /// `-l`, RDO lambda; 0 disables RDO.
    pub lambda: f32,
    /// `--smooth-scale`: distortion multiplier for flat blocks.
    pub smooth_scale: f32,
    /// `--smooth-std`: channel deviation at which a block gets half the extra weight.
    pub smooth_std: f32,
    /// `--max-rms-ratio`: RDO candidates may have at most this multiple of the base RMS error...
    pub max_rms_ratio: f32,
    /// `--min-rms`: ... or this RMS error per channel.
    pub min_rms: f32,
    /// `--zstd`: KTX2 supercompression level; 0 stores raw blocks.
    pub zstd: i32,
    /// Full mip chain (`--no-mips` when false).
    pub mips: bool,
    /// `--normal`: tangent-space normal map (linear, renormalized mips, Z rebuilt for display).
    pub normal: bool,
    /// Base-encode on the GPU when the browser has WebGPU and a kernel covers the settings. The
    /// output is the same either way.
    pub gpu: bool,
}

#[wasm_bindgen]
impl Settings {
    /// `mipple encode` defaults for `format` (bc1, bc4, bc5, bc7) and BC7 `entropy` (off, on, auto).
    #[wasm_bindgen(constructor)]
    pub fn new(format: &str, entropy: &str) -> Result<Settings, JsError> {
        Self::parse(format, entropy).map_err(|e| js_error(&e))
    }

    /// The equivalent command line, for an input file named `input`.
    pub fn command(&self, input: &str) -> String {
        let d = RdoParams::default();
        let mut c = format!(
            "mipple encode {input} -o out.ktx2 -f {} -e {} -l {}",
            self.format, self.effort, self.lambda
        );
        if self.entropy != EntropyMode::default() {
            c += &format!(" --entropy {}", entropy_name(self.entropy));
        }
        if self.normal {
            c += " --normal";
        }
        if !self.mips {
            c += " --no-mips";
        }
        if self.zstd != DEFAULT_ZSTD {
            c += &format!(" --zstd {}", self.zstd);
        }
        for (flag, v, default) in [
            ("--smooth-scale", self.smooth_scale, d.smooth_scale),
            ("--smooth-std", self.smooth_std, d.smooth_std),
            ("--max-rms-ratio", self.max_rms_ratio, d.max_rms_ratio),
            ("--min-rms", self.min_rms, d.min_rms),
        ] {
            if v != default {
                c += &format!(" {flag} {v}");
            }
        }
        c
    }
}

fn entropy_name(e: EntropyMode) -> &'static str {
    match e {
        EntropyMode::Off => "off",
        EntropyMode::On => "on",
        EntropyMode::Auto => "auto",
    }
}

impl Settings {
    fn parse(format: &str, entropy: &str) -> Result<Self> {
        let format: Format = format.parse().map_err(anyhow::Error::msg)?;
        let entropy = [EntropyMode::Off, EntropyMode::On, EntropyMode::Auto]
            .into_iter()
            .find(|&e| entropy_name(e).eq_ignore_ascii_case(entropy))
            .with_context(|| format!("unknown entropy mode '{entropy}' (expected off, on or auto)"))?;
        let d = RdoParams::default();
        Ok(Self {
            format,
            entropy,
            effort: DEFAULT_EFFORT,
            lambda: 0.0,
            smooth_scale: d.smooth_scale,
            smooth_std: d.smooth_std,
            max_rms_ratio: d.max_rms_ratio,
            min_rms: d.min_rms,
            zstd: DEFAULT_ZSTD,
            mips: true,
            normal: false,
            gpu: true,
        })
    }

    /// The options `mipple encode` builds from the equivalent flags.
    fn options(&self) -> EncodeOptions {
        let mut o = EncodeOptions::new(self.format);
        o.effort = self.effort.min(MAX_EFFORT);
        o.entropy = self.entropy;
        o.normal_map = self.normal;
        o.srgb = !self.normal && self.format.default_srgb();
        o.mips = self.mips;
        o.rdo = RdoParams {
            lambda: self.lambda.max(0.0),
            smooth_scale: self.smooth_scale,
            smooth_std: self.smooth_std,
            max_rms_ratio: self.max_rms_ratio,
            min_rms: self.min_rms,
            verify_zstd_level: (self.zstd > 0).then_some(self.zstd),
            ..RdoParams::default()
        };
        o
    }

    /// The `KTXwriterScParams` value `mipple encode` writes.
    fn sc_params(&self, opts: &EncodeOptions) -> String {
        format!(
            "--format {} --effort {} --lambda {} --entropy {} --window {} --zstd {}",
            self.format,
            opts.effort,
            opts.rdo.lambda,
            entropy_name(self.entropy),
            opts.rdo.window_for(self.format),
            self.zstd
        )
    }
}

/// One encode: the .ktx2 file, level 0 decoded, and its size and quality.
#[wasm_bindgen]
pub struct Encoded {
    ktx2: Vec<u8>,
    original_view: Vec<u8>,
    decoded_view: Vec<u8>,
    #[wasm_bindgen(readonly)]
    pub width: u32,
    #[wasm_bindgen(readonly)]
    pub height: u32,
    /// Mip levels in the file.
    #[wasm_bindgen(readonly)]
    pub levels: usize,
    /// Level 0 blocks before Zstd.
    #[wasm_bindgen(readonly)]
    pub raw_bytes: usize,
    /// Level 0 blocks after Zstd (as stored in the file; the raw size with Zstd off).
    #[wasm_bindgen(readonly)]
    pub stored_bytes: usize,
    /// The whole .ktx2 file: every level plus header, DFD and key/value data.
    #[wasm_bindgen(readonly)]
    pub file_bytes: usize,
    /// Level 0 PSNR over the channels the format stores (as `mipple score`).
    #[wasm_bindgen(readonly)]
    pub psnr: f64,
    /// Level 0 RMS error over the same channels, in 8-bit steps.
    #[wasm_bindgen(readonly)]
    pub rmse: f64,
    /// Level 0 SSIM: luma for BC1/BC7, R for BC4, mean of R and G for BC5 (as `mipple score`).
    #[wasm_bindgen(readonly)]
    pub ssim: f64,
    /// Mean normal angular error in degrees (normal maps only, otherwise NaN).
    #[wasm_bindgen(readonly)]
    pub normal_deg: f64,
    /// Base block encoding time, all levels, in milliseconds.
    #[wasm_bindgen(readonly)]
    pub base_ms: f64,
    /// RDO time, all levels, in milliseconds.
    #[wasm_bindgen(readonly)]
    pub rdo_ms: f64,
    /// Blocks RDO changed, all levels.
    #[wasm_bindgen(readonly)]
    pub rdo_changed: usize,
    /// Blocks, all levels.
    #[wasm_bindgen(readonly)]
    pub blocks: usize,
    gpu: Option<String>,
}

#[wasm_bindgen]
impl Encoded {
    /// The .ktx2 file (moved out; empty on a second call).
    pub fn take_ktx2(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.ktx2)
    }

    /// The source as the format sees it, RGBA8: BC4 as gray, BC5 as RG (Z rebuilt for normal maps).
    pub fn take_original(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.original_view)
    }

    /// Level 0 decoded, shown the same way as [`Encoded::take_original`].
    pub fn take_decoded(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.decoded_view)
    }

    /// The WebGPU adapter that made the base blocks, or `undefined` when the CPU did.
    #[wasm_bindgen(getter)]
    pub fn gpu(&self) -> Option<String> {
        self.gpu.clone()
    }
}

/// The worker's GPU: not looked for yet, absent, open, or switched off after disagreeing with
/// the CPU. Opened once per worker and kept for its lifetime.
#[derive(Clone, Copy)]
enum GpuState {
    Unopened,
    Missing,
    Ready(&'static Gpu),
    Disabled,
}

thread_local! {
    static GPU: Cell<GpuState> = const { Cell::new(GpuState::Unopened) };
}

/// The GPU, opened on first use. `None` without WebGPU, without an adapter, or once disabled.
async fn gpu() -> Option<&'static Gpu> {
    if let GpuState::Unopened = GPU.get() {
        let state = match Gpu::open_async().await {
            Some(g) => GpuState::Ready(Box::leak(Box::new(g))),
            None => GpuState::Missing,
        };
        GPU.set(state);
    }
    match GPU.get() {
        GpuState::Ready(g) => Some(g),
        _ => None,
    }
}

/// Opens the browser's WebGPU adapter ahead of the first encode and returns its name, or
/// `undefined` when there is none (the encoder then runs on the CPU).
#[wasm_bindgen(js_name = initGpu)]
pub async fn init_gpu() -> Option<String> {
    gpu().await.map(gpu_name)
}

/// Encodes row-major RGBA8 pixels with `settings` (consumed). Resolves when the encode is done.
#[wasm_bindgen]
pub async fn encode(rgba: Vec<u8>, width: u32, height: u32, settings: Settings) -> Result<Encoded, JsError> {
    let surface = surface_from_rgba(&rgba, width, height).map_err(|e| js_error(&e))?;
    drop(rgba);
    let gpu = if settings.gpu { gpu().await } else { None };
    run(&surface, &settings, gpu).await.map_err(|e| js_error(&e))
}

fn surface_from_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Surface> {
    ensure!(width > 0 && height > 0, "empty image");
    ensure!(
        rgba.len() == width as usize * height as usize * 4,
        "pixel buffer does not match {width}x{height}"
    );
    Ok(Surface {
        width,
        height,
        pixels: rgba.as_chunks::<4>().0.to_vec(),
    })
}

/// Up to `n` blocks of `level` spread over it, as one surface: what [`gpu_agrees`] encodes twice.
fn sample_blocks(level: &Surface, n: u32) -> Surface {
    let (bw, bh) = (level.blocks_x(), level.height.div_ceil(4));
    let total = bw * bh;
    let side = n.min(total).isqrt().max(1);
    let mut out = Surface::new(side * 4, side * 4);
    for i in 0..side * side {
        let j = (u64::from(i) * u64::from(total) / u64::from(side * side)) as u32;
        let b = level.read_block(j % bw, j / bw);
        out.write_block(i % side, i / side, &b);
    }
    out
}

/// Whether the GPU encodes a sample of `level`'s blocks exactly as the CPU does. The kernels are
/// exact by construction and tested on native drivers; this guards against a browser's shader
/// compiler that is not.
fn gpu_agrees(sample: &Surface, got: &[u8], opts: &EncodeOptions) -> bool {
    let (_, cpu) = pipeline::encode_base(sample, opts.format, opts.base_opts());
    got == pipeline::flatten(opts.format, &cpu)
}

/// The GPU's base blocks for each of `levels`, or `None` to encode on the CPU (no kernel for these
/// settings, a level too large for the device, or a failed check, which disables the GPU).
async fn gpu_bases(
    gpu: &'static Gpu,
    levels: &[Cow<'_, Surface>],
    opts: &EncodeOptions,
) -> Option<Vec<Option<Vec<u8>>>> {
    if !pipeline::gpu_eligible(opts) || !gpu.encodes(opts.format) {
        return None;
    }
    // The check sample and every level go in one submission with one readback: each wait on the
    // device costs a round trip through the browser's event loop.
    let sample = sample_blocks(levels.first()?, 256);
    let mut batch: Vec<&Surface> = vec![&sample];
    batch.extend(levels.iter().map(|l| &**l));
    let mut out = gpu.encode_surfaces_async(&batch, opts.format, opts.base_opts()).await?;
    let bases = out.split_off(1);
    if !gpu_agrees(&sample, &out[0], opts) {
        GPU.set(GpuState::Disabled);
        return None;
    }
    Some(bases.into_iter().map(Some).collect())
}

/// The adapter's name for the report (WebGPU does not expose one, so it may be empty).
fn gpu_name(gpu: &Gpu) -> String {
    let name = &gpu.info().name;
    if name.is_empty() { "WebGPU".into() } else { name.clone() }
}

async fn run(surface: &Surface, s: &Settings, gpu: Option<&'static Gpu>) -> Result<Encoded> {
    let opts = s.options();
    let levels = pipeline::prepare(surface, &opts);
    let t0 = Instant::now();
    let bases = match gpu {
        Some(g) => gpu_bases(g, &levels, &opts).await,
        None => None,
    };
    let gpu_secs = t0.elapsed().as_secs_f64();
    let name = bases.as_ref().and(gpu).map(gpu_name);
    let (tex, mut report) = pipeline::encode_prepared(&levels, &opts, name.as_deref(), bases.unwrap_or_default());
    report.base_secs += gpu_secs;
    drop(levels);
    let zstd = (s.zstd > 0).then_some(s.zstd);
    let ktx2 = ktx2::write(&tex, zstd, Some(&s.sc_params(&opts)), PrepMode::Off)?;
    let header = ktx2::read_header(&ktx2)?;
    let (_, stored, raw) = header.levels[0];

    let dec = pipeline::decode_level(&tex, 0, false);
    let (psnr, ssim, normal_deg) = bench::quality(s.format, surface, &dec, s.normal);
    let mse = 255.0 * 255.0 / 10f64.powf(psnr / 10.0);
    Ok(Encoded {
        original_view: view(s.format, surface, s.normal),
        decoded_view: view(s.format, &dec, s.normal),
        width: tex.width,
        height: tex.height,
        levels: tex.levels.len(),
        raw_bytes: raw as usize,
        stored_bytes: stored as usize,
        file_bytes: ktx2.len(),
        psnr,
        rmse: mse.sqrt(),
        ssim,
        normal_deg: normal_deg.unwrap_or(f64::NAN),
        base_ms: report.base_secs * 1e3,
        rdo_ms: report.rdo_secs * 1e3,
        rdo_changed: report.rdo.changed,
        blocks: report.rdo.blocks,
        gpu: report.gpu,
        ktx2,
    })
}

/// What a format stores, as RGBA8 for display: BC1 opaque RGB, BC4 its R channel as gray, BC5 R
/// and G (with Z rebuilt into blue for normal maps, as `mipple decode --normal` does), BC7 RGBA.
fn view(format: Format, s: &Surface, normal: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.pixels.len() * 4);
    for &[r, g, b, a] in &s.pixels {
        out.extend_from_slice(&match format {
            Format::Bc1 => [r, g, b, 255],
            Format::Bc4 => [r, r, r, 255],
            Format::Bc5 if normal => [r, g, ((normal_from_rg(r, g)[2] + 1.0) * 127.5 + 0.5) as u8, 255],
            Format::Bc5 => [r, g, 0, 255],
            Format::Bc7 => [r, g, b, a],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs a future that completes without waiting, as every encode does on the host (native GPU
    /// readback blocks inside `device.poll`).
    fn now<F: std::future::Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(f).poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("encode waited on the host"),
        }
    }

    /// Level 0 PSNR straight from the library, to check [`run`]'s.
    fn psnr_of(src: &Surface, s: &Settings) -> f64 {
        let (tex, _) = pipeline::encode(src, &s.options());
        let dec = pipeline::decode_level(&tex, 0, false);
        mipple::metrics::psnr(src, &dec, &s.format.metric_channels(src.has_alpha()))
    }

    fn gradient(w: u32, h: u32) -> Surface {
        let mut s = Surface::new(w, h);
        for y in 0..h {
            for x in 0..w {
                s.set(
                    x,
                    y,
                    [(x * 255 / w) as u8, (y * 255 / h) as u8, ((x ^ y) & 0xff) as u8, 255],
                );
            }
        }
        s
    }

    #[test]
    fn matches_cli_options_and_reports_level_zero() {
        let src = gradient(36, 20);
        for f in ["bc1", "bc4", "bc5", "bc7"] {
            let mut s = Settings::parse(f, "auto").unwrap();
            s.lambda = 1.0;
            s.mips = false;
            let e = now(run(&src, &s, None)).unwrap();
            assert_eq!((e.width, e.height, e.levels), (36, 20, 1));
            assert_eq!(e.raw_bytes, 9 * 5 * s.format.block_bytes());
            assert!(e.stored_bytes < e.file_bytes);
            assert_eq!(e.decoded_view.len(), 36 * 20 * 4);
            assert!((e.psnr - psnr_of(&src, &s)).abs() < 1e-9);
            assert_eq!(ktx2::read(&e.ktx2).unwrap().levels[0].data.len(), e.raw_bytes);
        }
    }

    /// The GPU path (sample check, GPU bases, CPU finish) writes the CPU's file, on the host's GPU.
    /// Skips without one.
    #[test]
    fn gpu_bases_match_the_cpu() {
        let Some(gpu) = Gpu::shared() else {
            return;
        };
        let src = gradient(72, 40);
        for (f, effort, lambda) in [("bc1", 1, 0.0), ("bc4", 5, 0.0), ("bc5", 2, 1.0), ("bc7", 3, 0.0)] {
            let mut s = Settings::parse(f, "off").unwrap();
            s.effort = effort;
            s.lambda = lambda;
            let on = now(run(&src, &s, Some(gpu))).unwrap();
            let off = now(run(&src, &s, None)).unwrap();
            assert!(on.gpu.is_some(), "{f}: the GPU made no blocks");
            assert_eq!(off.gpu, None);
            assert_eq!(on.ktx2, off.ktx2, "{f} e{effort} λ{lambda}");
        }
    }

    #[test]
    fn command_lists_only_changed_flags() {
        let mut s = Settings::parse("bc7", "auto").unwrap();
        assert_eq!(s.command("in.png"), "mipple encode in.png -o out.ktx2 -f bc7 -e 5 -l 0");
        s.lambda = 0.5;
        s.mips = false;
        s.smooth_scale = 2.5;
        assert_eq!(
            s.command("in.png"),
            "mipple encode in.png -o out.ktx2 -f bc7 -e 5 -l 0.5 --no-mips --smooth-scale 2.5"
        );
        assert!(Settings::parse("bc3", "auto").is_err());
        assert!(Settings::parse("bc7", "maybe").is_err());
    }

    #[test]
    fn fit_keeps_aspect() {
        let img = fit(image::RgbaImage::new(1000, 500), 256);
        assert_eq!((img.width(), img.height(), img.source_width), (256, 128, 1000));
        let img = fit(image::RgbaImage::new(100, 50), 256);
        assert_eq!((img.width(), img.height()), (100, 50));
    }
}
