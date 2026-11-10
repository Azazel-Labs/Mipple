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

pub mod bc1;
mod bc1_order;
pub mod bc4;
pub mod bc7;
pub mod bench;
pub mod check;
pub mod color;
pub mod dds;
pub mod fit;
pub mod format;
pub mod ktx2;
pub mod metrics;
pub mod mips;
pub mod pipeline;
pub mod prep;
pub mod ranker;
pub mod ranker_bc4;
pub mod ranker_bc7;
pub mod rdo;
pub mod simd;
pub mod surface;
pub mod synth;

pub use format::{BlockOpts, EntropyMode, Format};
pub use pipeline::{EncodeOptions, EncodedTexture, Level};
pub use rdo::RdoParams;
pub use surface::{Block, Rgba, Surface};
