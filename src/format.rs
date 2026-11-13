//! Output format dispatch.

use crate::surface::Block;
use crate::{bc1, bc3, bc4, bc6h, bc7, eac, etc};

/// What [`Format::encode_block`] and [`Format::decode_block`] panic with for ASTC.
const NOT_4X4: &str = "ASTC blocks are not 4x4 texels: encode whole surfaces (pipeline::encode, \
                        astc::encode::encode_surface) and decode with pipeline::decode_blocks";

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
    /// The decode mode to encode for: ASTC decodes differently in sRGB and linear (UNORM) mode,
    /// so its encoder needs to know which one the texture will be read with. Other formats ignore
    /// it.
    pub srgb: bool,
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
            srgb: false,
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
    /// 4 bpp RGB for mobile (OpenGL ES 3 / Vulkan baseline): ETC1 modes plus T, H and planar.
    Etc2Rgb,
    /// 8 bpp RGBA for mobile: an EAC alpha block plus ETC2 RGB (the BC3 analogue).
    Etc2Rgba,
    /// 4 bpp single channel for mobile, 11-bit (the BC4 analogue).
    EacR11,
    /// 8 bpp two channels for mobile, 11-bit (the BC5 analogue: normals).
    EacRg11,
    /// ASTC LDR with a 4x4 footprint: 8 bpp RGB/RGBA (mobile and Apple GPUs; the BC7 analogue).
    #[cfg_attr(feature = "cli", value(name = "astc-4x4"))]
    Astc4x4,
    /// ASTC LDR with a 6x6 footprint: 3.56 bpp RGB/RGBA.
    #[cfg_attr(feature = "cli", value(name = "astc-6x6"))]
    Astc6x6,
    /// ASTC LDR with an 8x8 footprint: 2 bpp RGB/RGBA.
    #[cfg_attr(feature = "cli", value(name = "astc-8x8"))]
    Astc8x8,
}

impl Format {
    pub const ALL: [Format; 13] = [
        Format::Bc1,
        Format::Bc3,
        Format::Bc4,
        Format::Bc5,
        Format::Bc6h,
        Format::Bc7,
        Format::Etc2Rgb,
        Format::Etc2Rgba,
        Format::EacR11,
        Format::EacRg11,
        Format::Astc4x4,
        Format::Astc6x6,
        Format::Astc8x8,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Format::Bc1 => "bc1",
            Format::Bc3 => "bc3",
            Format::Bc4 => "bc4",
            Format::Bc5 => "bc5",
            Format::Bc6h => "bc6h",
            Format::Bc7 => "bc7",
            Format::Etc2Rgb => "etc2-rgb",
            Format::Etc2Rgba => "etc2-rgba",
            Format::EacR11 => "eac-r11",
            Format::EacRg11 => "eac-rg11",
            Format::Astc4x4 => "astc-4x4",
            Format::Astc6x6 => "astc-6x6",
            Format::Astc8x8 => "astc-8x8",
        }
    }

    /// Whether this is an ETC2 or EAC format (mobile; no DDS container, no GPU encoder).
    pub fn is_etc(self) -> bool {
        matches!(
            self,
            Format::Etc2Rgb | Format::Etc2Rgba | Format::EacR11 | Format::EacRg11
        )
    }

    /// Whether this is an ASTC format. Its blocks cover [`Format::block_dims`] texels, which is
    /// not 4x4 in general, so it is encoded and decoded a surface at a time: the 4x4 [`Block`]
    /// paths ([`Format::encode_block`], [`Format::decode_block`], the RDO pass) panic for it.
    pub fn is_astc(self) -> bool {
        matches!(self, Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8)
    }

    /// Texels across and down one block: 4x4, or the ASTC footprint.
    pub fn block_dims(self) -> (u32, u32) {
        match self {
            Format::Astc6x6 => (6, 6),
            Format::Astc8x8 => (8, 8),
            _ => (4, 4),
        }
    }

    /// Bytes per block (of [`Format::block_dims`] texels).
    pub fn block_bytes(self) -> usize {
        match self {
            Format::Bc1 | Format::Bc4 | Format::Etc2Rgb | Format::EacR11 => 8,
            Format::Bc3 | Format::Bc5 | Format::Bc6h | Format::Bc7 | Format::Etc2Rgba | Format::EacRg11 => 16,
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
        }
    }

    /// Blocks across and down a `width`×`height` level.
    pub fn blocks_xy(self, width: u32, height: u32) -> (u32, u32) {
        let (bw, bh) = self.block_dims();
        (width.div_ceil(bw), height.div_ceil(bh))
    }

    /// Bytes of a `width`×`height` level, or `None` when that overflows `usize` (the KTX2 and DDS
    /// readers take sizes from untrusted headers).
    pub fn level_bytes(self, width: u32, height: u32) -> Option<usize> {
        let (nx, ny) = self.blocks_xy(width, height);
        let nx = usize::try_from(nx).ok()?;
        let ny = usize::try_from(ny).ok()?;
        nx.checked_mul(ny)?.checked_mul(self.block_bytes())
    }

    /// Whether color data in this format is normally stored sRGB-encoded.
    pub fn default_srgb(self) -> bool {
        self.supports_srgb()
    }

    /// Whether the format has an sRGB variant at all.
    pub fn supports_srgb(self) -> bool {
        matches!(
            self,
            Format::Bc1 | Format::Bc3 | Format::Bc7 | Format::Etc2Rgb | Format::Etc2Rgba
        ) || self.is_astc()
    }

    /// Vulkan `VkFormat` value, as stored in a KTX2 header.
    pub fn vk_format(self, srgb: bool) -> u32 {
        match (self, srgb) {
            (Format::Bc1, false) => 131,      // VK_FORMAT_BC1_RGB_UNORM_BLOCK
            (Format::Bc1, true) => 132,       // VK_FORMAT_BC1_RGB_SRGB_BLOCK
            (Format::Bc3, false) => 137,      // VK_FORMAT_BC3_UNORM_BLOCK
            (Format::Bc3, true) => 138,       // VK_FORMAT_BC3_SRGB_BLOCK
            (Format::Bc4, _) => 139,          // VK_FORMAT_BC4_UNORM_BLOCK
            (Format::Bc5, _) => 141,          // VK_FORMAT_BC5_UNORM_BLOCK
            (Format::Bc6h, _) => 143,         // VK_FORMAT_BC6H_UFLOAT_BLOCK
            (Format::Bc7, false) => 145,      // VK_FORMAT_BC7_UNORM_BLOCK
            (Format::Bc7, true) => 146,       // VK_FORMAT_BC7_SRGB_BLOCK
            (Format::Etc2Rgb, false) => 147,  // VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
            (Format::Etc2Rgb, true) => 148,   // VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK
            (Format::Etc2Rgba, false) => 151, // VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK
            (Format::Etc2Rgba, true) => 152,  // VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK
            (Format::EacR11, _) => 153,       // VK_FORMAT_EAC_R11_UNORM_BLOCK
            (Format::EacRg11, _) => 155,      // VK_FORMAT_EAC_R11G11_UNORM_BLOCK
            (Format::Astc4x4, false) => 157,  // VK_FORMAT_ASTC_4x4_UNORM_BLOCK
            (Format::Astc4x4, true) => 158,   // VK_FORMAT_ASTC_4x4_SRGB_BLOCK
            (Format::Astc6x6, false) => 165,  // VK_FORMAT_ASTC_6x6_UNORM_BLOCK
            (Format::Astc6x6, true) => 166,   // VK_FORMAT_ASTC_6x6_SRGB_BLOCK
            (Format::Astc8x8, false) => 171,  // VK_FORMAT_ASTC_8x8_UNORM_BLOCK
            (Format::Astc8x8, true) => 172,   // VK_FORMAT_ASTC_8x8_SRGB_BLOCK
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
            147 => (Format::Etc2Rgb, false),
            148 => (Format::Etc2Rgb, true),
            151 => (Format::Etc2Rgba, false),
            152 => (Format::Etc2Rgba, true),
            153 => (Format::EacR11, false),
            155 => (Format::EacRg11, false),
            157 => (Format::Astc4x4, false),
            158 => (Format::Astc4x4, true),
            165 => (Format::Astc6x6, false),
            166 => (Format::Astc6x6, true),
            171 => (Format::Astc8x8, false),
            172 => (Format::Astc8x8, true),
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
            Format::Etc2Rgb => [1, 1, 1, 0],
            Format::Etc2Rgba => [1, 1, 1, 1],
            Format::EacR11 => [1, 0, 0, 0],
            Format::EacRg11 => [1, 1, 0, 0],
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => [1, 1, 1, 1],
        }
    }

    /// Channels to score in metrics.
    pub fn metric_channels(self, has_alpha: bool) -> Vec<usize> {
        match self {
            Format::Bc1 | Format::Bc6h | Format::Etc2Rgb => vec![0, 1, 2],
            Format::Bc4 | Format::EacR11 => vec![0],
            Format::Bc5 | Format::EacRg11 => vec![0, 1],
            Format::Bc3 | Format::Bc7 | Format::Etc2Rgba if has_alpha => vec![0, 1, 2, 3],
            Format::Bc3 | Format::Bc7 | Format::Etc2Rgba => vec![0, 1, 2],
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 if has_alpha => vec![0, 1, 2, 3],
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => vec![0, 1, 2],
        }
    }

    /// Encodes one 4x4 block. Only the first [`Format::block_bytes`] bytes are meaningful.
    ///
    /// # Panics
    ///
    /// For ASTC ([`Format::is_astc`]), which is encoded a surface at a time.
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
            Format::Etc2Rgb => out[..8].copy_from_slice(&etc::encode_block(px, effort)),
            Format::Etc2Rgba => out = etc::encode_block_rgba(px, effort),
            Format::EacR11 => out[..8].copy_from_slice(&eac::encode_r11(&bc4::channel(px, 0), effort)),
            Format::EacRg11 => {
                out[..8].copy_from_slice(&eac::encode_r11(&bc4::channel(px, 0), effort));
                out[8..].copy_from_slice(&eac::encode_r11(&bc4::channel(px, 1), effort));
            }
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => panic!("{self}: {NOT_4X4}"),
        }
        out
    }

    /// Decodes one 4x4 block.
    ///
    /// # Panics
    ///
    /// For ASTC ([`Format::is_astc`]), which is decoded a surface at a time.
    pub fn decode_block(self, raw: &[u8], out: &mut Block) {
        match self {
            Format::Bc1 => bc1::decode(raw, out),
            Format::Bc3 => bc3::decode(raw, out),
            Format::Bc4 => bc4::decode_bc4(raw, out),
            Format::Bc5 => bc4::decode_bc5(raw, out),
            Format::Bc6h => bc6h::decode_ldr(raw, out),
            Format::Bc7 => bc7::decode(raw, out),
            Format::Etc2Rgb => etc::decode(raw, out),
            Format::Etc2Rgba => etc::decode_rgba(raw, out),
            Format::EacR11 => eac::decode_r11_block(raw, out),
            Format::EacRg11 => eac::decode_rg11_block(raw, out),
            Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => panic!("{self}: {NOT_4X4}"),
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
            .ok_or_else(|| format!("unknown format '{s}' (expected bc1, bc3, bc4, bc5, bc6h, bc7, etc2-rgb, etc2-rgba, eac-r11, eac-rg11, astc-4x4, astc-6x6 or astc-8x8)"))
    }
}
