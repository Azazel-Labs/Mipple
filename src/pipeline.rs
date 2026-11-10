//! End-to-end encode/decode: image -> mips -> base blocks -> RDO -> [`EncodedTexture`].

use crate::format::{BlockOpts, EntropyMode, Format};
use crate::metrics::normal_from_rg;
use crate::mips::{self, MipFilter};
use crate::rdo::{self, RdoParams, RdoStats};
use crate::surface::{Block, Surface};
use rayon::prelude::*;
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
    /// Data is sRGB-encoded color (affects mip filtering and the KTX2 vkFormat/DFD).
    pub srgb: bool,
    /// Input is a tangent-space normal map (renormalized mips).
    pub normal_map: bool,
    /// Entropy-reduced base encoding (BC7); see [`BlockOpts::reduce_entropy`].
    pub entropy: EntropyMode,
    /// Generate a full mip chain.
    pub mips: bool,
    /// RDO settings (`lambda == 0` disables RDO).
    pub rdo: RdoParams,
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
            rdo: RdoParams::default(),
        }
    }

    /// The per-block settings these options imply.
    pub fn block_opts(&self) -> BlockOpts {
        BlockOpts {
            effort: self.effort,
            reduce_entropy: self.entropy.resolve(self.rdo.lambda),
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
    pub base_secs: f64,
    pub rdo_secs: f64,
    pub rdo: RdoStats,
}

/// Base-encodes every block of a surface in parallel. Returns (source blocks, encoded blocks).
pub fn encode_base(surface: &Surface, format: Format, opts: impl Into<BlockOpts>) -> (Vec<Block>, Vec<[u8; 16]>) {
    let opts = opts.into();
    let src = surface.blocks();
    let enc = if direct(format, opts) {
        encode_surface_blocks(surface, format, opts)
    } else {
        encode_src_blocks(&src, format, opts)
    };
    (src, enc)
}

/// BC7 plain-search effort floor for RDO's base encodings (see [`encode`]).
pub const RDO_BASE_EFFORT: u8 = 8;

/// Formats whose encoder reads the surface directly instead of materialized blocks.
fn direct(format: Format, opts: BlockOpts) -> bool {
    match format {
        Format::Bc7 => crate::bc7::fast_cfg(opts).is_some(),
        _ => true,
    }
}

/// Raw blocks (`format.block_bytes()` each) of a direct-path format.
fn encode_surface_raw(surface: &Surface, format: Format, opts: BlockOpts) -> Vec<u8> {
    match format {
        Format::Bc1 => crate::bc1::encode_surface(surface, opts.effort),
        Format::Bc4 => crate::bc4::encode_surface(surface, 1, opts.effort),
        Format::Bc5 => crate::bc4::encode_surface(surface, 2, opts.effort),
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

/// Encodes a surface (and optionally its mip chain).
pub fn encode(surface: &Surface, opts: &EncodeOptions) -> (EncodedTexture, EncodeReport) {
    let chain = if opts.mips {
        mips::chain(surface, opts.mip_filter())
    } else {
        Vec::new()
    };
    let levels_in: Vec<&Surface> = if opts.mips {
        chain.iter().collect()
    } else {
        vec![surface]
    };
    let w = opts.format.error_weights();
    let mut report = EncodeReport::default();
    let mut levels = Vec::with_capacity(levels_in.len());
    let rdo_params = RdoParams {
        reduce_entropy: opts.rdo.reduce_entropy || opts.block_opts().reduce_entropy,
        ..opts.rdo
    };
    for level in levels_in {
        let t0 = Instant::now();
        if direct(opts.format, opts.block_opts()) && !rdo_params.enabled() {
            // Straight to raw bytes: no source blocks, no 16-byte slots.
            let data = encode_surface_raw(level, opts.format, opts.block_opts());
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
        // Source blocks are only materialized when an encoder or RDO needs them.
        let src = if rdo_params.enabled() || !direct(opts.format, opts.block_opts()) {
            level.blocks()
        } else {
            Vec::new()
        };
        let bo = opts.block_opts();
        // BC7 RDO: the plain search at RDO_BASE_EFFORT or more, as the base or (entropy-reduced
        // base) the second base. Its cost is small next to RDO's, and RDO keeps much of the
        // quality it adds (Kodak subset vs Oodle 2.9.16 on Zstd-19, λ 10-40 at equal RMSE:
        // -e 5 +8.3% -> +4.0%).
        let plain = BlockOpts {
            reduce_entropy: false,
            effort: if opts.format == Format::Bc7 && rdo_params.enabled() {
                bo.effort.max(RDO_BASE_EFFORT)
            } else {
                bo.effort
            },
        };
        let base = if opts.format == Format::Bc7 && !bo.reduce_entropy {
            plain
        } else {
            bo
        };
        let mut blocks = if direct(opts.format, base) {
            encode_surface_blocks(level, opts.format, base)
        } else {
            encode_src_blocks(&src, opts.format, base)
        };
        let t1 = Instant::now();
        let stats = if rdo_params.enabled() {
            let row = (level.width as usize).div_ceil(4);
            // BC7: other bases each block may start from. With an entropy-reduced base, the plain
            // (error-only) encoding; always, a mode-6-only encoding, whose blocks share one layout
            // and so give later blocks more to copy (Oodle's RDO output is mostly mode 6).
            let mut alts = Vec::new();
            if opts.format == Format::Bc7 {
                if bo.reduce_entropy {
                    alts.push(encode_surface_blocks(level, opts.format, plain));
                }
                alts.push(unflatten(
                    opts.format,
                    &crate::bc7::fast::encode_surface(level, crate::bc7::mode6_cfg(), plain),
                ));
            }
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
    let tex = EncodedTexture {
        format: opts.format,
        srgb: opts.srgb && opts.format.supports_srgb(),
        width: surface.width,
        height: surface.height,
        levels,
    };
    (tex, report)
}

/// Decodes raw blocks back to an image. For BC5 with `normal_map`, Z is reconstructed into blue.
pub fn decode_blocks(format: Format, data: &[u8], width: u32, height: u32, normal_map: bool) -> Surface {
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
    decode_blocks(tex.format, &l.data, l.width, l.height, normal_map)
}
