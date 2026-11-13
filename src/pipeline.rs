//! End-to-end encode/decode: image -> mips -> base blocks -> RDO -> [`EncodedTexture`].

use crate::alpha::{self, AlphaMode};
use crate::format::{BlockOpts, EntropyMode, Format};
use crate::metrics::normal_from_rg;
use crate::mips::{self, MipAlpha, MipFilter};
use crate::rdo::{self, RdoParams, RdoStats};
use crate::surface::{Block, Surface};
use rayon::prelude::*;
use std::borrow::Cow;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;
// `std::time::Instant::now` panics on wasm32-unknown-unknown (the browser demo in web/).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::Instant;

/// Encoder settings.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    pub format: Format,
    /// 0..=[`crate::format::MAX_EFFORT`], higher is slower and better.
    pub effort: u8,
    /// Data is sRGB-encoded color (affects mip filtering, the KTX2 vkFormat/DFD and, for ASTC,
    /// which decode mode the blocks are encoded for).
    pub srgb: bool,
    /// Input is a tangent-space normal map (renormalized mips).
    pub normal_map: bool,
    /// Entropy-reduced base encoding (BC7); see [`BlockOpts::reduce_entropy`].
    pub entropy: EntropyMode,
    /// Generate a full mip chain.
    pub mips: bool,
    /// What alpha means: with transparency, the colour under transparent texels is bled from the
    /// visible colour around it and freed for the encoder, and mips weight RGB by alpha
    /// ([`crate::alpha`]). [`AlphaMode::Auto`] (default) picks it for BC3, BC7, ETC2 RGBA8 and ASTC
    /// colour textures.
    pub alpha: AlphaMode,
    /// An alpha-test cutoff (0-1) whose coverage every mip keeps (cutouts drawn with alpha test or
    /// alpha to coverage); `None` averages alpha plainly.
    pub alpha_cutoff: Option<f32>,
    /// RDO settings (`lambda == 0` disables RDO). BC6H RDO runs in [`encode_hdr`].
    pub rdo: RdoParams,
    /// Where base encoding runs: [`GpuMode::Auto`] (default) uses the GPU where its encoder is the
    /// faster one ([`crate::gpu::Gpu::preferred`]). Formats without a GPU encoder, builds without
    /// the `gpu` feature and machines without a usable GPU use the CPU in every mode.
    pub gpu: GpuMode,
}

/// Where base encoding runs ([`EncodeOptions::gpu`]). Every GPU encoder writes the same blocks as
/// the CPU encoder, so this only changes speed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuMode {
    /// The GPU for formats whose GPU encoder is faster than the CPU's, else the CPU.
    #[default]
    Auto,
    /// The GPU for every format it has an encoder for.
    On,
    /// The CPU only.
    Off,
}

impl EncodeOptions {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            effort: crate::format::DEFAULT_EFFORT,
            srgb: format.default_srgb(),
            normal_map: false,
            entropy: EntropyMode::default(),
            mips: true,
            alpha: AlphaMode::Auto,
            alpha_cutoff: None,
            rdo: RdoParams::default(),
            gpu: GpuMode::Auto,
        }
    }

    /// The per-block settings these options imply.
    pub fn block_opts(&self) -> BlockOpts {
        BlockOpts {
            effort: self.effort,
            reduce_entropy: self.entropy.resolve(self.rdo.lambda),
            srgb: self.srgb && self.format.supports_srgb(),
        }
    }

    /// The per-block settings of the base encoding, on the CPU or the GPU: [`Self::block_opts`]
    /// with BC4/BC5's effort floor under RDO ([`base_block_opts`]).
    pub fn base_opts(&self) -> BlockOpts {
        base_block_opts(self.format, self.block_opts(), rdo_params(self).enabled())
    }

    /// Whether alpha is treated as transparency ([`Self::alpha`]): only for formats that keep
    /// alpha (BC3, BC7, ETC2 RGBA8, ASTC) and never for normal maps.
    pub fn alpha_transparency(&self) -> bool {
        (matches!(self.format, Format::Bc3 | Format::Bc7 | Format::Etc2Rgba) || self.format.is_astc())
            && !self.normal_map
            && match self.alpha {
                AlphaMode::Auto => self.srgb,
                AlphaMode::Transparency => true,
                AlphaMode::Data => false,
            }
    }

    /// How mips treat alpha.
    pub fn mip_alpha(&self) -> MipAlpha {
        MipAlpha {
            transparency: self.alpha_transparency(),
            coverage_cutoff: self.alpha_cutoff,
        }
    }

    pub fn mip_filter(&self) -> MipFilter {
        if self.normal_map {
            MipFilter::Normal
        } else if self.srgb {
            MipFilter::Srgb
        } else {
            MipFilter::Linear
        }
    }
}

/// One encoded mip level: raw GPU blocks in raster order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// A fully encoded texture, ready to be written to KTX2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedTexture {
    pub format: Format,
    pub srgb: bool,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Level>,
}

/// Timing and RDO statistics for one encode.
#[derive(Clone, Debug, Default)]
pub struct EncodeReport {
    /// The GPU adapter that base-encoded the levels, or `None` for the CPU.
    pub gpu: Option<String>,
    pub base_secs: f64,
    pub rdo_secs: f64,
    pub rdo: RdoStats,
}

/// Base-encodes every block of a surface in parallel. Returns (source blocks, encoded blocks).
///
/// # Panics
///
/// For ASTC, whose blocks do not pair with 4x4 source blocks: use [`encode`] (or
/// [`crate::astc::encode::encode_surface`]) instead.
pub fn encode_base(surface: &Surface, format: Format, opts: impl Into<BlockOpts>) -> (Vec<Block>, Vec<[u8; 16]>) {
    assert!(
        !format.is_astc(),
        "encode_base pairs blocks with 4x4 source blocks; {format} blocks are {:?} texels",
        format.block_dims()
    );
    let opts = opts.into();
    let src = surface.blocks();
    let enc = if direct(format, opts) {
        encode_surface_blocks(surface, format, opts)
    } else {
        encode_src_blocks(&src, format, opts)
    };
    (src, enc)
}

/// Effort floor for RDO's base encodings: BC7's plain search ([`rdo_bases`]) and BC4/BC5
/// ([`base_block_opts`]).
pub const RDO_BASE_EFFORT: u8 = 8;

/// `bo` for a BC4/BC5 base that RDO starts from: effort [`RDO_BASE_EFFORT`] or more (endpoint
/// window radius 4 instead of 1 at the default effort). RDO keeps most of the accuracy it adds,
/// and the base costs little next to RDO.
pub fn base_block_opts(format: Format, bo: BlockOpts, rdo: bool) -> BlockOpts {
    match format {
        Format::Bc4 | Format::Bc5 if rdo => BlockOpts {
            effort: bo.effort.max(RDO_BASE_EFFORT),
            ..bo
        },
        _ => bo,
    }
}

/// One surface's starting point for RDO, made as [`encode`] makes it: the base encoding and the
/// alternate bases each block may start from (BC7 only). `src` is `surface.blocks()`, or empty when
/// the format reads the surface directly and `rdo` is off.
///
/// # Panics
///
/// When `rdo` is set for ASTC, whose RDO pass starts from its plain encode.
pub fn rdo_bases(
    surface: &Surface,
    src: &[Block],
    format: Format,
    bo: BlockOpts,
    rdo: bool,
) -> (Vec<[u8; 16]>, Vec<Vec<[u8; 16]>>) {
    assert!(
        !(rdo && format.is_astc()),
        "{format}'s RDO starts from its plain encode"
    );
    let bo = base_block_opts(format, bo, rdo);
    // BC7 RDO: the plain search at RDO_BASE_EFFORT or more, as the base or (entropy-reduced
    // base) the second base. Its cost is small next to RDO's, and RDO keeps much of the
    // quality it adds (Kodak subset vs Oodle 2.9.16 on Zstd-19, λ 10-40 at equal RMSE:
    // -e 5 +8.3% -> +4.0%).
    let bc7_rdo = format == Format::Bc7 && rdo;
    let plain = BlockOpts {
        reduce_entropy: false,
        effort: if bc7_rdo {
            bo.effort.max(RDO_BASE_EFFORT)
        } else {
            bo.effort
        },
        ..bo
    };
    let base = if format == Format::Bc7 && !bo.reduce_entropy {
        plain
    } else {
        bo
    };
    let encode = |o: BlockOpts| {
        if direct(format, o) {
            encode_surface_blocks(surface, format, o)
        } else {
            encode_src_blocks(src, format, o)
        }
    };
    let blocks = encode(base);
    // BC7: with an entropy-reduced base, the plain (error-only) encoding; always, a mode-6-only
    // encoding, whose blocks share one layout and so give later blocks more to copy (Oodle's RDO
    // output is mostly mode 6).
    let mut alts = Vec::new();
    if bc7_rdo {
        if bo.reduce_entropy {
            alts.push(match crate::bc7::rdo_alt_cfg(plain) {
                Some(cfg) => unflatten(format, &crate::bc7::fast::encode_surface(surface, cfg, plain)),
                None => encode(plain),
            });
        }
        alts.push(unflatten(
            format,
            &crate::bc7::fast::encode_surface(surface, crate::bc7::mode6_cfg(), plain),
        ));
    }
    (blocks, alts)
}

/// Formats whose encoder reads the surface directly instead of materialized blocks. ASTC must:
/// its blocks are not 4x4.
fn direct(format: Format, opts: BlockOpts) -> bool {
    match format {
        Format::Bc7 => crate::bc7::fast_cfg(opts).is_some(),
        _ => true,
    }
}

/// Raw blocks (`format.block_bytes()` each, raster order over [`Format::block_dims`] blocks) of a
/// direct-path format.
fn encode_surface_raw(surface: &Surface, format: Format, opts: BlockOpts) -> Vec<u8> {
    match format {
        Format::Bc1 => crate::bc1::encode_surface(surface, opts.effort),
        Format::Bc3 => crate::bc3::encode_surface(surface, opts.effort),
        Format::Bc6h => crate::bc6h::encode_surface_ldr(surface, opts.effort),
        Format::Bc4 => crate::bc4::encode_surface(surface, 1, opts.effort),
        Format::Bc5 => crate::bc4::encode_surface(surface, 2, opts.effort),
        Format::Etc2Rgb => crate::etc::encode_surface(surface, opts.effort),
        Format::Etc2Rgba => crate::etc::encode_surface_rgba(surface, opts.effort),
        Format::EacR11 => crate::eac::encode_surface(surface, (0, 1), true, opts.effort),
        Format::EacRg11 => crate::eac::encode_surface(surface, (0, 2), true, opts.effort),
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => {
            let (bw, bh) = format.block_dims();
            crate::astc::encode::encode_surface(surface, bw, bh, opts.srgb, opts.effort)
        }
        Format::Bc7 => crate::bc7::fast::encode_surface(
            surface,
            crate::bc7::fast_cfg(opts).expect("BC7 direct path without a fast configuration"),
            opts,
        ),
    }
}

/// Expands raw blocks to the 16-byte slots RDO works on.
fn unflatten(format: Format, raw: &[u8]) -> Vec<[u8; 16]> {
    let n = format.block_bytes();
    raw.chunks_exact(n)
        .map(|c| {
            let mut b = [0u8; 16];
            b[..n].copy_from_slice(c);
            b
        })
        .collect()
}

fn encode_surface_blocks(surface: &Surface, format: Format, opts: BlockOpts) -> Vec<[u8; 16]> {
    unflatten(format, &encode_surface_raw(surface, format, opts))
}

fn encode_src_blocks(src: &[Block], format: Format, opts: BlockOpts) -> Vec<[u8; 16]> {
    let w = format.error_weights();
    if format == Format::Bc7 && opts.searched().effort <= 1 {
        // BC7 search levels 0-1 encode 8 blocks per SIMD batch.
        let opts = opts.searched();
        src.par_chunks(crate::simd::LANES)
            .flat_map_iter(|c| crate::bc7::encode_batch(c, &w, opts))
            .collect()
    } else {
        src.par_iter().map(|b| format.encode_block(b, &w, opts)).collect()
    }
}

/// Concatenates the meaningful bytes of each block.
pub fn flatten(format: Format, blocks: &[[u8; 16]]) -> Vec<u8> {
    let n = format.block_bytes();
    let mut out = Vec::with_capacity(blocks.len() * n);
    for b in blocks {
        out.extend_from_slice(&b[..n]);
    }
    out
}

/// The surfaces [`encode`] base-encodes, top level first: the mip chain (or the surface alone).
/// With alpha as transparency, the top level's hidden colour is bled (the mips bleed their own)
/// and each level's colour beyond the visible texels' ring is freed. [`encode_prepared`] encodes
/// them.
pub fn prepare<'a>(surface: &'a Surface, opts: &EncodeOptions) -> Vec<Cow<'a, Surface>> {
    let transparency = opts.alpha_transparency() && surface.has_alpha();
    let top = if transparency {
        let mut s = surface.clone();
        alpha::bleed(&mut s);
        Cow::Owned(s)
    } else {
        Cow::Borrowed(surface)
    };
    let levels = if opts.mips {
        mips::chain_with(&top, opts.mip_filter(), opts.mip_alpha())
            .into_iter()
            .map(Cow::Owned)
            .collect()
    } else {
        vec![top]
    };
    if !transparency {
        return levels;
    }
    levels
        .into_iter()
        .map(|l| alpha::hide(&l).map_or(l, Cow::Owned))
        .collect()
}

/// RDO settings for `opts`: entropy reduction is on when the base encoding asks for it.
fn rdo_params(opts: &EncodeOptions) -> RdoParams {
    RdoParams {
        reduce_entropy: opts.rdo.reduce_entropy || opts.block_opts().reduce_entropy,
        // The 8-bit RDO does not apply to BC6H, whose own pass runs in `encode_hdr`.
        lambda: if opts.format == Format::Bc6h {
            0.0
        } else {
            opts.rdo.lambda
        },
        ..opts.rdo
    }
}

/// Whether a GPU kernel can make the base blocks for `opts`: the format has one at this effort,
/// and the encode is not BC7 RDO, which starts from a higher-effort base and alternate bases that
/// only the CPU makes. Other formats' RDO starts from the GPU's base.
pub fn gpu_eligible(opts: &EncodeOptions) -> bool {
    let bc7_rdo = opts.format == Format::Bc7 && rdo_params(opts).enabled();
    !bc7_rdo && crate::gpu::Gpu::has_kernel(opts.format, opts.base_opts())
}

/// Encodes a surface (and optionally its mip chain).
pub fn encode(surface: &Surface, opts: &EncodeOptions) -> (EncodedTexture, EncodeReport) {
    let levels = prepare(surface, opts);
    let use_gpu = gpu_eligible(opts)
        && match opts.gpu {
            GpuMode::Auto => crate::gpu::Gpu::preferred(opts.format, opts.base_opts()),
            GpuMode::On => true,
            GpuMode::Off => false,
        };
    let gpu = use_gpu
        .then(crate::gpu::Gpu::shared)
        .flatten()
        .filter(|g| g.encodes(opts.format));
    // Every level in one submission, so the GPU is waited on once per texture.
    let bases = gpu.and_then(|g| {
        let refs: Vec<&Surface> = levels.iter().map(|l| &**l).collect();
        g.encode_surfaces(&refs, opts.format, opts.base_opts())
    });
    let name = bases.as_ref().and(gpu).map(|g| g.info().name.as_str());
    let mut bases = bases.unwrap_or_default().into_iter();
    encode_with(&levels, opts, name, |_, _| bases.next())
}

/// [`encode`] of [`prepare`]'s levels with base blocks the caller already made, one entry per level
/// (raw blocks as [`crate::gpu::Gpu::encode_surfaces_async`] returns them; `None` encodes that level
/// on the CPU). This is the browser's way in: it cannot wait for the GPU inside [`encode`]. `gpu`
/// names the adapter for the report. The bases are ignored unless [`gpu_eligible`] holds, and the
/// result is the same as [`encode`]'s, since every GPU kernel writes the CPU's blocks.
pub fn encode_prepared(
    levels: &[Cow<Surface>],
    opts: &EncodeOptions,
    gpu: Option<&str>,
    mut bases: Vec<Option<Vec<u8>>>,
) -> (EncodedTexture, EncodeReport) {
    let eligible = gpu_eligible(opts);
    encode_with(levels, opts, gpu, |i, _| {
        bases.get_mut(i).and_then(Option::take).filter(|_| eligible)
    })
}

/// Encodes `levels`, taking each level's base blocks from `base(index, level)` when it returns
/// them (a GPU encode) and from the CPU otherwise, then running RDO.
fn encode_with(
    levels_in: &[Cow<Surface>],
    opts: &EncodeOptions,
    gpu: Option<&str>,
    mut base: impl FnMut(usize, &Surface) -> Option<Vec<u8>>,
) -> (EncodedTexture, EncodeReport) {
    let w = opts.format.error_weights();
    let mut report = EncodeReport::default();
    let mut levels = Vec::with_capacity(levels_in.len());
    let rdo_params = rdo_params(opts);
    for (i, level) in levels_in.iter().enumerate() {
        let level: &Surface = level;
        let t0 = Instant::now();
        let on_gpu = base(i, level);
        if on_gpu.is_some() {
            report.gpu = gpu.map(str::to_owned);
        }
        if !rdo_params.enabled() && (on_gpu.is_some() || direct(opts.format, opts.block_opts())) {
            // Straight to raw bytes: no source blocks, no 16-byte slots.
            let data = on_gpu.unwrap_or_else(|| encode_surface_raw(level, opts.format, opts.block_opts()));
            report.base_secs += t0.elapsed().as_secs_f64();
            report.rdo = RdoStats::merge(
                report.rdo,
                RdoStats {
                    blocks: data.len() / opts.format.block_bytes(),
                    ..Default::default()
                },
            );
            levels.push(Level {
                width: level.width,
                height: level.height,
                data,
            });
            continue;
        }
        if opts.format.is_astc() {
            // ASTC's footprints are not 4x4 source blocks: its RDO pass reads the surface itself.
            let bo = opts.block_opts();
            let mut data = encode_surface_raw(level, opts.format, bo);
            let t1 = Instant::now();
            let stats = crate::astc::encode::rdo::optimize(opts.format, level, bo.srgb, &mut data, &rdo_params);
            report.base_secs += (t1 - t0).as_secs_f64();
            report.rdo_secs += t1.elapsed().as_secs_f64();
            report.rdo = RdoStats::merge(report.rdo, stats);
            levels.push(Level {
                width: level.width,
                height: level.height,
                data,
            });
            continue;
        }
        // Source blocks are only materialized when an encoder or RDO needs them.
        let src = if rdo_params.enabled() || !direct(opts.format, opts.block_opts()) {
            level.blocks()
        } else {
            Vec::new()
        };
        // A GPU base is RDO's only starting point (the CPU adds alternate bases for BC7).
        let (mut blocks, alts) = match on_gpu {
            Some(raw) => (unflatten(opts.format, &raw), Vec::new()),
            None => rdo_bases(level, &src, opts.format, opts.block_opts(), rdo_params.enabled()),
        };
        let t1 = Instant::now();
        let stats = if rdo_params.enabled() {
            let row = (level.width as usize).div_ceil(4);
            rdo::optimize_with_alt(
                opts.format,
                &src,
                &mut blocks,
                &alts,
                &w,
                &RdoParams { row, ..rdo_params },
            )
        } else {
            RdoStats {
                blocks: blocks.len(),
                ..Default::default()
            }
        };
        let t2 = Instant::now();
        report.base_secs += (t1 - t0).as_secs_f64();
        report.rdo_secs += (t2 - t1).as_secs_f64();
        report.rdo = RdoStats::merge(report.rdo, stats);
        levels.push(Level {
            width: level.width,
            height: level.height,
            data: flatten(opts.format, &blocks),
        });
    }
    let (width, height) = levels_in.first().map_or((0, 0), |l| (l.width, l.height));
    let tex = EncodedTexture {
        format: opts.format,
        srgb: opts.srgb && opts.format.supports_srgb(),
        width,
        height,
        levels,
    };
    (tex, report)
}

/// Encodes an HDR image to BC6H (unsigned half floats), with a box-filtered mip chain when
/// `opts.mips`, then RDO when `opts.rdo` enables it ([`crate::bc6h::rdo`]). Uses `opts.effort`,
/// `opts.mips` and `opts.rdo`; `opts.format` must be [`Format::Bc6h`].
pub fn encode_hdr(surface: &crate::hdr::HdrSurface, opts: &EncodeOptions) -> (EncodedTexture, EncodeReport) {
    assert_eq!(opts.format, Format::Bc6h, "encode_hdr encodes BC6H");
    let chain = if opts.mips {
        surface.chain()
    } else {
        vec![surface.clone()]
    };
    let t0 = Instant::now();
    let levels: Vec<Level> = chain
        .iter()
        .map(|l| Level {
            width: l.width,
            height: l.height,
            data: crate::bc6h::encode_surface(l, opts.effort),
        })
        .collect();
    let base_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut levels = levels;
    let rdo = chain
        .iter()
        .zip(levels.iter_mut())
        .map(|(l, lv)| crate::bc6h::rdo::optimize(l, &mut lv.data, &opts.rdo))
        .fold(RdoStats::default(), RdoStats::merge);
    let report = EncodeReport {
        base_secs,
        rdo_secs: if opts.rdo.enabled() { t1.elapsed().as_secs_f64() } else { 0.0 },
        rdo,
        ..Default::default()
    };
    let tex = EncodedTexture {
        format: Format::Bc6h,
        srgb: false,
        width: surface.width,
        height: surface.height,
        levels,
    };
    (tex, report)
}

/// Decodes one mip level of a BC6H texture to linear float RGB.
pub fn decode_level_hdr(tex: &EncodedTexture, level: usize) -> crate::hdr::HdrSurface {
    assert_eq!(tex.format, Format::Bc6h, "decode_level_hdr decodes BC6H");
    let l = &tex.levels[level];
    crate::bc6h::decode_surface(&l.data, l.width, l.height)
}

/// Decodes raw blocks back to an image. For BC5 with `normal_map`, Z is reconstructed into blue.
/// `srgb` is the decode mode, which only ASTC's decode depends on (other formats ignore it).
pub fn decode_blocks(format: Format, data: &[u8], width: u32, height: u32, srgb: bool, normal_map: bool) -> Surface {
    if format.is_astc() {
        let (bw, bh) = format.block_dims();
        return crate::astc::decode_rgba(data, width, height, bw, bh, srgb);
    }
    let mut s = Surface::new(width, height);
    let n = format.block_bytes();
    let bw = width.div_ceil(4);
    let mut out = [[0u8; 4]; 16];
    for (i, raw) in data.chunks_exact(n).enumerate() {
        format.decode_block(raw, &mut out);
        if format == Format::Bc5 && normal_map {
            for t in &mut out {
                let v = normal_from_rg(t[0], t[1]);
                t[2] = ((v[2] + 1.0) * 127.5 + 0.5) as u8;
            }
        }
        let i = i as u32;
        s.write_block(i % bw, i / bw, &out);
    }
    s
}

/// Decodes one mip level of an encoded texture.
pub fn decode_level(tex: &EncodedTexture, level: usize, normal_map: bool) -> Surface {
    let l = &tex.levels[level];
    decode_blocks(tex.format, &l.data, l.width, l.height, tex.srgb, normal_map)
}
