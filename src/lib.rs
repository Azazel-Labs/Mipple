//! mipple — rate-distortion optimized GPU texture encoding.
//!
//! Pipeline (see `docs/ARCHITECTURE.md`):
//!
//! ```text
//! source image -> mip chain -> base block encode (BCn) -> RDO pass (LZ-aware) -> KTX2 (+Zstd)
//!                                                                           \-> benchmark harness
//! ```
//!
//! Covers BC1, BC4, BC5 and BC7 (every mode encoded and decoded),
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
