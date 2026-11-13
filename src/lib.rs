//! mipple — rate-distortion optimized GPU texture encoding.
//!
//! Pipeline (see `docs/ARCHITECTURE.md`):
//!
//! ```text
//! source image -> mip chain -> base block encode (BCn, ETC2, ASTC) -> RDO pass (LZ-aware) -> KTX2 (+Zstd)
//!                                                                                       \-> benchmark harness
//! ```
//!
//! Covers BC1, BC3, BC4, BC5, BC6H and BC7 (every mode encoded and decoded), ETC2 RGB/RGBA8,
//! EAC R11/RG11 and ASTC LDR 4x4/6x6/8x8 (no RDO yet) for mobile,
//! a byte-level LZ rate model shared by every format, and a KTX2 writer/reader.

pub mod alpha;
pub mod astc;
pub mod bc1;
pub mod bc3;
pub mod bc4;
pub mod bc6h;
pub mod bc7;
pub mod bench;
pub mod check;
pub mod color;
pub mod dds;
pub mod eac;
pub mod etc;
mod exact;
pub mod fit;
pub mod format;
pub mod gpu;
pub mod hdr;
pub mod ktx2;
pub mod metrics;
pub mod mips;
pub mod pipeline;
pub mod prep;
pub mod ranker;
pub mod rdo;
pub mod simd;
pub mod surface;
pub mod synth;

pub use format::{BlockOpts, EntropyMode, Format};
pub use pipeline::{EncodeOptions, EncodedTexture, GpuMode, Level};
pub use rdo::RdoParams;
pub use surface::{Block, Rgba, Surface};
