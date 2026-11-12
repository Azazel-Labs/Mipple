//! Output format dispatch.

use crate::surface::Block;
use crate::{bc1, bc3, bc4, bc6h, bc7};

/// When to use entropy-reduced base encoding ([`BlockOpts::reduce_entropy`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum EntropyMode {
    /// Never: the base encoder minimizes error only.
    Off,
    /// Always, even without RDO.
    On,
    /// Only at λ ≥ [`EntropyMode::AUTO_MIN_LAMBDA`], where it wins (see `docs/BENCHMARKS.md`).
    #[default]
    Auto,
}

impl EntropyMode {
    /// `Auto` turns entropy reduction on from this λ up. Below it (high quality targets) plain
    /// base encoding + RDO reaches a given PSNR in fewer bytes; above it entropy reduction does.
    /// The curves cross around 43–44 dB on Kodak and 42.5 dB on the synthetic set.
    pub const AUTO_MIN_LAMBDA: f32 = 1.0;

    /// Whether to reduce entropy for an encode with this RDO λ.
    pub fn resolve(self, lambda: f32) -> bool {
        match self {
            EntropyMode::Off => false,
            EntropyMode::On => true,
            EntropyMode::Auto => lambda >= Self::AUTO_MIN_LAMBDA,
        }
    }
}

/// Highest effort level.
pub const MAX_EFFORT: u8 = 9;
/// Effort used when none is given: the balanced setting.
pub const DEFAULT_EFFORT: u8 = 5;

/// The four-level search of the full BC7 encoders (0 fastest, 3 best) used for an effort
/// level, until those formats get finer tiers.
pub fn search_level(effort: u8) -> u8 {
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3][effort.min(MAX_EFFORT) as usize]
}

/// Per-block encoder settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockOpts {
    /// 0..=[`MAX_EFFORT`], higher is slower and better. Each format maps it to its own search
    /// (see [`search_level`], [`crate::bc4`]).
    pub effort: u8,
    /// Bias the base encoder toward low-entropy blocks that LZ back ends compress well
    /// (bc7enc's `-e`). BC7 only; trades some PSNR for size. See [`crate::bc7::encode`].
    pub reduce_entropy: bool,
}

impl BlockOpts {
    /// These options with `effort` replaced by its [`search_level`] (what the scalar and batch BC7
    /// encoders take).
    pub fn searched(self) -> Self {
        Self {
            effort: search_level(self.effort),
            ..self
        }
    }
}

impl From<u8> for BlockOpts {
    fn from(effort: u8) -> Self {
        Self {
            effort,
            reduce_entropy: false,
        }
    }
}

/// GPU block formats mipple can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum Format {
    /// 4 bpp RGB (opaque albedo, bulk color).
    Bc1,
    /// 8 bpp RGBA with smooth alpha (BC1 color plus a BC4 alpha channel).
    Bc3,
    /// 4 bpp single channel (roughness, AO, masks, height).
    Bc4,
    /// 8 bpp two channels (tangent-space normals).
    Bc5,
    /// 8 bpp HDR RGB, unsigned half floats (BC6H_UF16: skies, lightmaps, emissive).
    Bc6h,
    /// 8 bpp RGB/RGBA, highest quality.
    Bc7,
}

impl Format {
    pub const ALL: [Format; 6] = [
        Format::Bc1,
        Format::Bc3,
        Format::Bc4,
        Format::Bc5,
        Format::Bc6h,
        Format::Bc7,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Format::Bc1 => "bc1",
            Format::Bc3 => "bc3",
            Format::Bc4 => "bc4",
            Format::Bc5 => "bc5",
            Format::Bc6h => "bc6h",
            Format::Bc7 => "bc7",
        }
    }

    /// Bytes per 4x4 block.
    pub fn block_bytes(self) -> usize {
        match self {
            Format::Bc1 | Format::Bc4 => 8,
            Format::Bc3 | Format::Bc5 | Format::Bc6h | Format::Bc7 => 16,
        }
    }

    /// Bytes of a `width`×`height` level, or `None` when that overflows `usize` (the KTX2 and DDS
    /// readers take sizes from untrusted headers).
    pub fn level_bytes(self, width: u32, height: u32) -> Option<usize> {
        let bw = usize::try_from(width.div_ceil(4)).ok()?;
        let bh = usize::try_from(height.div_ceil(4)).ok()?;
        bw.checked_mul(bh)?.checked_mul(self.block_bytes())
    }

    /// Whether color data in this format is normally stored sRGB-encoded.
    pub fn default_srgb(self) -> bool {
        matches!(self, Format::Bc1 | Format::Bc3 | Format::Bc7)
    }

    /// Whether the format has an sRGB variant at all.
    pub fn supports_srgb(self) -> bool {
        matches!(self, Format::Bc1 | Format::Bc3 | Format::Bc7)
    }

    /// Vulkan `VkFormat` value, as stored in a KTX2 header.
    pub fn vk_format(self, srgb: bool) -> u32 {
        match (self, srgb) {
            (Format::Bc1, false) => 131, // VK_FORMAT_BC1_RGB_UNORM_BLOCK
            (Format::Bc1, true) => 132,  // VK_FORMAT_BC1_RGB_SRGB_BLOCK
            (Format::Bc3, false) => 137, // VK_FORMAT_BC3_UNORM_BLOCK
            (Format::Bc3, true) => 138,  // VK_FORMAT_BC3_SRGB_BLOCK
            (Format::Bc4, _) => 139,     // VK_FORMAT_BC4_UNORM_BLOCK
            (Format::Bc5, _) => 141,     // VK_FORMAT_BC5_UNORM_BLOCK
            (Format::Bc6h, _) => 143,    // VK_FORMAT_BC6H_UFLOAT_BLOCK
            (Format::Bc7, false) => 145, // VK_FORMAT_BC7_UNORM_BLOCK
            (Format::Bc7, true) => 146,  // VK_FORMAT_BC7_SRGB_BLOCK
        }
    }

    /// Inverse of [`Format::vk_format`].
    pub fn from_vk(vk: u32) -> Option<(Format, bool)> {
        Some(match vk {
            131 => (Format::Bc1, false),
            132 => (Format::Bc1, true),
            137 => (Format::Bc3, false),
            138 => (Format::Bc3, true),
            139 => (Format::Bc4, false),
            141 => (Format::Bc5, false),
            143 => (Format::Bc6h, false),
            145 => (Format::Bc7, false),
            146 => (Format::Bc7, true),
            _ => return None,
        })
    }

    /// Per-channel error weights (R, G, B, A) for the channels this format stores.
    ///
    /// BC7 always weights alpha, even for opaque images, so mode 6/7 p-bit choices keep
    /// alpha at exactly 255.
    pub fn error_weights(self) -> [u32; 4] {
        match self {
            Format::Bc1 => [1, 1, 1, 0],
            Format::Bc3 => [1, 1, 1, 1],
            Format::Bc4 => [1, 0, 0, 0],
            Format::Bc5 => [1, 1, 0, 0],
            Format::Bc6h => [1, 1, 1, 0],
            Format::Bc7 => [1, 1, 1, 1],
        }
    }

    /// Channels to score in metrics.
    pub fn metric_channels(self, has_alpha: bool) -> Vec<usize> {
        match self {
            Format::Bc1 | Format::Bc6h => vec![0, 1, 2],
            Format::Bc4 => vec![0],
            Format::Bc5 => vec![0, 1],
            Format::Bc3 | Format::Bc7 if has_alpha => vec![0, 1, 2, 3],
            Format::Bc3 | Format::Bc7 => vec![0, 1, 2],
        }
    }

    /// Encodes one block. Only the first [`Format::block_bytes`] bytes are meaningful.
    pub fn encode_block(self, px: &Block, w: &[u32; 4], opts: impl Into<BlockOpts>) -> [u8; 16] {
        let opts = opts.into();
        let effort = opts.effort;
        let mut out = [0u8; 16];
        match self {
            Format::Bc1 => out[..8].copy_from_slice(&bc1::encode_fast(px, effort)),
            Format::Bc3 => out = bc3::encode_block(px, effort),
            Format::Bc4 => out[..8].copy_from_slice(&bc4::encode_channel(&bc4::channel(px, 0), effort)),
            Format::Bc5 => {
                out[..8].copy_from_slice(&bc4::encode_channel(&bc4::channel(px, 0), effort));
                out[8..].copy_from_slice(&bc4::encode_channel(&bc4::channel(px, 1), effort));
            }
            Format::Bc6h => out = bc6h::encode_ldr(px, effort),
            Format::Bc7 => out = bc7::encode(px, w, opts.searched()),
        }
        out
    }

    /// Decodes one block.
    pub fn decode_block(self, raw: &[u8], out: &mut Block) {
        match self {
            Format::Bc1 => bc1::decode(raw, out),
            Format::Bc3 => bc3::decode(raw, out),
            Format::Bc4 => bc4::decode_bc4(raw, out),
            Format::Bc5 => bc4::decode_bc5(raw, out),
            Format::Bc6h => bc6h::decode_ldr(raw, out),
            Format::Bc7 => bc7::decode(raw, out),
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Format {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Format::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| format!("unknown format '{s}' (expected bc1, bc3, bc4, bc5, bc6h or bc7)"))
    }
}
