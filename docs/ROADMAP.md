# Roadmap

Each item should land with a benchmark delta (see `bench-results/`). Major and minor in `Cargo.toml` name the
milestone in progress; every green push to main is released as the next patch (`vX.Y.Z`, with binaries;
`.github/workflows/release.yml`).

## v0.1 (pre-release): encoder and container

- [x] BC1, BC4, BC5 and BC7 encoders and decoders, mip generation, KTX2 writer/reader with Zstd
  supercompression, DDS I/O.
- [x] A byte-level LZ rate model shared by every format and an RDO pass over it.

## v0.2 (pre-release): close the gap to the best open BCn RDO

- [x] **Head-to-head harness** (`scripts/head2head.py` + `mipple score` + DDS I/O): mipple vs bc7enc_rdo on Kodak, BD-rate. Result: mipple −13.4% (Zstd) vs bc7enc_rdo's bc7enc.cpp base. See `docs/BENCHMARKS.md`.
- [x] **bc7f partition pre-selection** at effort 1 (principal-axis split + popcount): 2.1× faster base encode, −0.18 dB, and a better RDO curve. (Superseded: effort 1 now ranks all 64 partitions with vectorized line fits.)
- [x] **Re-run head-to-head with bc7e** (bc7enc_rdo built with ISPC) and **bc7f**: done in `scripts/compete.py`.
- [x] **Fast regression check** (`mipple check`): synthetic 256² × 4 formats × 5 λ in ~30 s against a committed baseline; BD-rate, λ=0 PSNR/size and per-image gates; byte-identical count; CI step.
- [x] **BC7 modes 4 and 5** in the scalar encoder (alpha from effort 1, all opaque rotations from effort 2).

## v0.9: competitive parity and Oodle Texture (current)

Goal: meet or beat the quality and speed of the best open encoders per format, then Oodle Texture. Status and
numbers: [BENCHMARKS.md](BENCHMARKS.md#competitive-benchmark).

Shipped:

- [x] **Entropy-reduced base encoding** (bc7enc `-e`): mode-6 preference and 64-level mode-6 endpoints, kept through RDO refits. `--entropy auto` (on for λ ≥ 1) is the default: −3.2% BD-rate on Kodak-4 and synthetic, λ < 1 unchanged. Plain `on` gives −5.1% over a full λ sweep.
- [x] **Multi-format head-to-head** (`scripts/compete.py`, `tools/competitors/`): bc7enc_rdo (bc7e, rgbcx, RDO),
  bc7f, Intel ISPC Texture Compressor, icbc, stb_dxt, Compressonator, texconv; core pinning for single-thread
  timings on hybrid CPUs.
- [x] **Robust BD-rate** (Pareto front, piecewise-linear) in `mipple check`, `head2head.py` and `compete.py`.
- [x] **SIMD layer** (`src/simd.rs`): `std::simd`-named types, portable and AVX2 backends, runtime dispatch.
- [x] **BC1:** 3-color mode with black, SIMD cluster fit with a learned split order. Top-end quality matches
  rgbcx L18 / icbc 9. Batch encoder for every effort; effort 0 is faster than Intel ISPC and +0.34 dB.
- [x] **BC4/BC5 fast tier:** effort 0 beats Intel ISPC and effort 1 beats stb_dxt and rgbcx on PSNR and speed.
- [x] **BC7 fast encoder for every effort** (`bc7/fast.rs`): modes 6, 1, 3, 0, 2 and 5, ±1 endpoint/p-bit
  refinement, partitions ranked with an index quantization term. Beats bc7e u0/u2/u4, every Intel ISPC profile and
  every bc7f profile on both PSNR and speed; effort 8 is 46.10 dB at 1.2× bc7e u6's speed (46.08 dB).
- [x] **RDO candidate loop:** field dedup and per-field memoization, fused bounded error kernels, an exact LZ
  rate lower bound and table-driven rate costs make the 256-back-reference search the default and faster than
  bc7enc_rdo's RDO in every format (Kodak 24, 20 threads).
- [x] **BC4/BC5 RDO:** aligned byte-span candidates (ERT-style), wider error-cap floor (`--min-rms` 3), frozen
  literal costs, a 192-deep BC5 search, and a localized BC5 search ranked by a small MLP trained only on CC0 Poly
  Haven textures.
- [x] **BC1 RDO:** field-swap refinement (`--refine-back`) and far endpoints (`--far-back`).
- [x] **BC7 RDO:** 8 KiB window and 64 KiB chunks, a far search (whole-block copies and re-indexed endpoints from
  1024 blocks back, behind exact mean and segment bounds), the plain search at effort 8+ as the base (a second
  base with entropy reduction), and a cross-mode candidate. vs bc7enc_rdo: BD-rate −33.7% (bc7e) / −32.3% (`-e`).
- [x] **Smooth-block weight** (5× ramping to 1 at channel std 14): Kodak-24 BD-rate −2.3% to −3.6% in every
  format. vs bc7enc_rdo: BC1 −26.8%, BC4 −20.0%, BC5 −26.1%.
- [x] **Lossless field split** (`src/prep.rs`, BC7Prep-style, `mipple encode --prep on|auto`): −4% to −8%
  Zstd-19 without RDO. With strong RDO it can lose (RDO optimizes whole-block matches in the interleaved
  layout); `auto` keeps the smaller layout per level.
- [x] **vs Oodle Texture 2.9.16 (Unreal Engine 5.8), true source:** `scripts/ue_oodle/` gets Oodle's real blocks
  out of a stock UE cook (bit-exact, no engine source) and scores them on Zstd-19 next to a mipple sweep
  (`bench-results/oodle-ue/`). Kodak-24 at Oodle λ 10-40, mipple's size at equal RMSE / equal SSIM: BC1 −7.6% to
  −6.3% / −6.9% to −5.7%, BC4 −9.9% to −2.2% / −3.2% to −0.9%, BC5 −18.3% to −6.8% / −4.6% to −2.0%.
- [x] **Learned BC7 ranker** (`src/ranker_bc7.rs`, trained only on CC0 Poly Haven): a 32-deep exhaustive search,
  the 1024-block far search, then 16 blocks from anywhere in the 64 KiB chunk, picked by a two-key prefilter
  (decoded value SAD and least-squares refit residual) and a small MLP. Kodak-24 BD-rate −0.22% at 0.86× the
  time; Poly Haven held-out −0.73%.
- [x] **Learned BC4 ranker** (`src/ranker_bc4.rs`, CC0 Poly Haven only): 128 deep plus 32 blocks from the 16 KiB
  chunk (64 by shape SAD, 64 by value SAD, then a 13-16-1 MLP). Kodak-24 BD-rate −0.77% at 0.95× the time; Poly
  Haven held-out −0.71%.
- [x] **Mode-6 base for BC7 RDO:** a mode-6-only encode as a third starting point per block. Kodak-24 BD-rate
  −0.22% at 1.06× the time; vs Oodle 2.9.16 at equal RMSE +3.4-5.8% → +2.3-4.0%, at equal SSIM −0.4-2.8% →
  −1.3-4.8%.
- [x] **Learned BC1 ranker** (tried, not merged): the whole chunk holds little for BC1 (an exhaustive 2048-block
  search gains −0.09% BD-rate at 4.65× the time).
- [x] **Tooling:** `cargo xtask fmt|clippy|ci`, rustfmt at 120 columns, clippy lints in `Cargo.toml`, CI on
  Linux/Windows/macOS, a release for every green push.

Open for v0.9.x:

- [ ] **BC7 vs Oodle 2.9.16 at equal RMSE:** Oodle keeps 1.8-3.1% fewer bits at equal RMSE on Kodak-24 (mipple is
  2.3-5.1% ahead at equal SSIM; field-swap refinement for BC7 took the gap down from 2.3-4.0%). mipple effort 8-9 without RDO already matches Oodle's λ 0 accuracy at fewer
  bits, so the gap is in the RDO search: Oodle's RDO output has about 3× more index-half repeats. 128 KiB chunks
  are another −1.0% to −1.3% BD-rate at the same CPU time, waiting on chunk rebalancing (1.4-1.5× wall time).
- [ ] BC7 mode 4 in the fast encoder, p-bit search in quantization, and a faster uber step.
- [ ] **Smarter entropy reduction:** pick mode/partition/p-bits by an estimated rate *given the previous blocks*
  (not a fixed mode-6 weight), so the base encoder and RDO share one objective.
- [ ] **Wider searches elsewhere:** far endpoints for BC4 (−0.38%, 1.4×) need a tighter bound, and Oodle's far
  index-pattern reuse (about half mipple's distinct BC1 index patterns on kodim18) needs a cheap way to find
  index patterns that fit (refit candidates over 2048 blocks: −0.17%, several times slower).
- [ ] Rebalance RDO chunks for many threads without making output depend on the thread count (BC5 on Kodak:
  24 chunks on 20 threads run as two uneven rounds). Tried: splitting each block's search over back distances
  across threads; no faster on a 155H, since per-block joins wait for the slowest core.
- [ ] Add **Basis UASTC + RDO + Zstd** and **XUBC7** to the head-to-head (`basisu -unpack` to DDS, then score).
  Add **bc7g** when released (`scripts/head2head.py --bc7g`). NVTT 3 needs a manual download.

## v1.0: more formats

- [ ] BC3, BC6H (HDR, relative error metric; needs float input: EXR/HDR loading).
- [ ] ASTC LDR (4x4, 6x6, 8x8) output, with the same RDO pass.
- [ ] ETC2 / EAC for mobile fallbacks.

## v1.1: perception and speed

- [ ] **Perceptual distortion:** per-block masking (activity-weighted SSE) and λ normalized across texture
  types, as Oodle does; an NVIDIA FLIP metric in `bench`.
- [ ] **Normal maps:** score on lit output (Aras' method) inside RDO, not just in metrics.
- [ ] **Analytical mode prediction** (bc7f): closed-form SSE estimates (line-fit residual + endpoint/weight
  quantization noise) to choose mode pairs (1 vs 3, 0 vs 2, 4 vs 5) without trial encodes. See
  `docs/research/bc7f-bc7g.md`.
- [ ] **AVX-512 and NEON backends** for `src/simd.rs` (needs hardware to tune).
- [ ] **Cheaper chunk boundaries:** worth less than thought: a single chunk per Kodak image is only −0.86% BD-rate
  for BC5 and −0.54% for BC1 vs 16 KiB chunks. Seeding each chunk with the base encoder's preceding blocks is
  worse (+2.9%: they change under it).
- [ ] GPU compute base encoders (wgpu, `gpu` cargo feature, on by default; `src/gpu/`). Each kernel
  writes exactly the CPU encoder's blocks (tests compare every effort, block for block); `GpuMode::Auto`
  uses one only where it beats the CPU, `--gpu` forces it. Done: BC4/BC5 (`bc4.wgsl`; the CPU stays
  the default, since 20 AVX2 threads take 1-3 ms for a 1.6 MP BC5 vs 14 ms on an Intel Arc iGPU);
  BC1 efforts 0-1 (`bc1.wgsl`, after making the CPU line fit and refits exact integer arithmetic;
  CPU default again: 9 vs 71 ms for 16 MP at effort 0); BC7 effort 0 (`bc7.wgsl`, mode 6, after
  making the BC7 fast path's line fit, endpoints and index projection exact; 14 vs 59 ms for 16 MP).
  The BC7 two-subset fits (principal axis, endpoints, LS refits, quantizer rounding and index
  projection) are exact integers on the CPU too, at unchanged speed and quality, ready for the kernel.
  Next: BC7 refits and multi-subset modes up to effort 8 (where the GPU should win) and the RDO
  alternates, BC1 efforts 2+ (needs per-lane cluster-fit pruning and exact split scoring), then
  WebGPU in the web demo when the browser has it.

## v1.2: "output B": latent-space supercompression + transcoder

- [ ] Intermediate format: transform-coded BC7/ASTC block parameters (weight-grid DCT, endpoint DPCM, context-modeled entropy coder), following Basis' XUASTC/XUBC7 ideas.
- [ ] Runtime transcoder (CPU + compute shader) to BC7, BC1, ASTC 4x4, ETC2; measure transcode Mpix/s.
- [ ] KTX2 vendor supercompression scheme + DFD for the intermediate.

## Pipeline quality (unversioned)

- [ ] RDO rate model for the split layout (`--prep`), so RDO and the transform stop fighting; a KTX2
  key/value flag for the layout and a GPU (compute shader) inverse, so loaders can undo it after hardware
  decompression.
- [ ] Better mip filters (Kaiser/Lanczos), alpha-coverage-preserving mips for cutouts.
- [ ] Per-role presets (`--role albedo|normal|mask|ui`) choosing format, λ and smooth protection.
- [ ] Rate models for Kraken (if licensed) and GDeflate; `--backend` switch for verification.
- [ ] `ktx validate` (KTX-Software) in CI.
