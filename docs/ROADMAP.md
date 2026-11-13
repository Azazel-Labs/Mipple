# Roadmap

Each item should land with a benchmark delta (see `bench-results/`). Major and minor in `Cargo.toml` name the
milestone in progress; every green push to main is released as the next patch (`vX.Y.Z`, with binaries;
`.github/workflows/release.yml`). Ideas tried and dropped are catalogued in [DEADENDS.md](DEADENDS.md).

## Next up

1. ~~ASTC LDR (4x4, 6x6, 8x8) output, with the same RDO pass.~~ Done (v1.0 below); next for it: a GPU kernel,
   a clear lead over ISPC at 6x6 (0.01 dB ahead at about its time), more footprints, a faster RDO pass.
2. BC6H: RDO with a log-space distortion, signed BC6H, a GPU kernel.
3. ~~ETC2 / EAC for mobile fallbacks.~~ Done (v1.0 below); next for it: punch-through alpha (ETC2 RGB A1),
   signed EAC, a SIMD batch encoder.

## v0.1 (pre-release): encoder and container

- [x] BC1, BC4, BC5 and BC7 encoders and decoders, mip generation, KTX2 writer/reader with Zstd
  supercompression, DDS I/O.
- [x] A byte-level LZ rate model shared by every format and an RDO pass over it.

## v0.2 (pre-release): close the gap to the best open BCn RDO

- [x] **Head-to-head harness** (`mipple score` + DDS I/O; now `scripts/compete.py`): mipple vs bc7enc_rdo on Kodak, BD-rate. Result: mipple −13.4% (Zstd) vs bc7enc_rdo's bc7enc.cpp base. See `docs/BENCHMARKS.md`.
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
- [x] **Robust BD-rate** (Pareto front, piecewise-linear) in `mipple check` and `compete.py`.
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
  format. Since 0.9.18 4× falling as k²/(k²+σ²) with k = 12, which put BC7 ahead of Oodle Texture at equal RMSE
  (was 2-3% behind at λ 10-40) while staying ahead at equal SSIM. vs bc7enc_rdo: BC1 −26.8%, BC4 −20.0%, BC5 −26.1%.
- [x] **Lossless field split** (`src/prep.rs`, BC7Prep-style, `mipple encode --prep on|auto`): −4% to −8%
  Zstd-19 without RDO. With strong RDO it can lose (RDO optimizes whole-block matches in the interleaved
  layout); `auto` keeps the smaller layout per level.
- [x] **vs Oodle Texture 2.9.16 (Unreal Engine 5.8), true source:** `scripts/ue_oodle/` gets Oodle's real blocks
  out of a stock UE cook (bit-exact, no engine source) and scores them on Zstd-19 next to a mipple sweep
  (`bench-results/oodle-ue/`). Kodak-24 at Oodle λ 10-40, mipple's size at equal RMSE / equal SSIM: BC1 −9.6% to
  −8.8% / −5.7% to −4.0%, BC4 −13.9% to −5.7% / −3.3% to −0.9%, BC7 −1.5% to −0.6% / −3.1% to −0.3%, BC5
  −20.1% to −9.8% / −4.3% to −2.4%; ahead at every λ in every format, λ 0-1 included.
- [x] **Learned BC7 ranker** (`src/ranker/bc7.rs`, trained only on CC0 Poly Haven): a 32-deep exhaustive search,
  the 1024-block far search, then 16 blocks from anywhere in the 64 KiB chunk, picked by a two-key prefilter
  (decoded value SAD and least-squares refit residual) and a small MLP. Kodak-24 BD-rate −0.22% at 0.86× the
  time; Poly Haven held-out −0.73%.
- [x] **Learned BC4 ranker** (`src/ranker/bc4.rs`, CC0 Poly Haven only): 128 deep plus 32 blocks from the 16 KiB
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

- [x] **BC7 vs Oodle 2.9.16 at equal RMSE:** Oodle kept 1.8-3.1% fewer bits at equal RMSE on Kodak-24 until the
  smooth weight's new shape (0.9.18); mipple is now 0.6-1.5% smaller at equal RMSE and 0.3-3.1% at equal SSIM.
- [ ] **Widen the BC7 lead:** the margin at λ 30-40 on SSIM is thin (−0.7%, −0.3%), and per image smooth-sky
  shots (kodim07, kodim20) are still up to 5.5% behind there. Oodle's RDO output has about 3× more index-half
  repeats. 128 KiB chunks are another −1.0% to −1.3% BD-rate at the same CPU time, waiting on chunk rebalancing
  (1.4-1.5× wall time).
- [ ] **BC7 RDO clearly faster than Oodle** (paused 2026-11-11): quality is ahead in every cell, but CPU is
  only about level (0.92-1.01× Oodle's at the same size and quality; mipple λ 1-1.5 against Oodle λ 30; history in
  [BENCHMARKS.md](BENCHMARKS.md)). Target: about 25% less CPU at λ 1 with every table cell still ahead.
  Where λ 1 CPU goes (busy time, kodim05): main RDO loop 76%, of which far search 18% (half of it mode-6
  re-indexing), ranker prefilter scan 15%, near candidates 11%, refinement about 10%; the two extra bases
  (`pipeline::rdo_bases`) 20%; Zstd check and KTX2 write about 7%.
  - Kept: in refinement, a mode-6 block skips other modes' first halves (about 3% less at λ 1, byte-identical on
    Kodak-24 and the Poly Haven test set).
  - Rejected (quality, or slower with no gain): see [DEADENDS.md](DEADENDS.md#bc7-rdo).
  - Measured, not pursued: whole blocks and index halves are only 0-10% repeated within the ranker's 2048-block
    reach at λ 1 on most images (30-60% at λ 4+), so a deduplicated prefilter scan pays mostly at higher λ.
  - Next: a ranker reach of 1024 (the rate model's window) instead of 2048 (halves the prefilter scan; needs a
    quality run); structural cuts to the far re-indexing and the second base, using the cross-domain research
    sweep's ideas (hash-based block matching from screen-content video coding, lower-bound cascades).
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
  Add **bc7g** to `scripts/compete.py` when released. NVTT 3 needs a manual download.

## v1.0: more formats

- [x] BC3: a BC4 alpha half and a 4-color BC1 color half, both halves through the same RDO pass (each
  half's copy, re-index and refit, like BC5), alpha as transparency like BC7; −15.7% BD-rate against
  bc7enc_rdo at a fifth of its time (BENCHMARKS.md). No GPU kernel yet.
- [x] BC6H, unsigned (all 14 modes, Radiance `.hdr` input, scored with mPSNR and log2 RMSE): on 20 Poly
  Haven HDRIs effort 2 beats ISPC veryslow at a fifteenth of its time (BENCHMARKS.md). EXR input was dropped
  for build time (DEADENDS.md).
  - [x] Eight-block batch encoder (one block per lane, per-lane mode and partition): 2.0× faster at
    effort 0 and 2.6-2.8× at efforts 5 and 9, with the same quality.
  - [x] Partition ranking across lanes (byte-identical output): effort 0 is 0.13 dB ahead of ISPC fast
    and 13% faster. The mode estimates still run one lane at a time.
  - [ ] RDO for BC6H (a log-space distortion so λ means the same across exposures), signed BC6H, GPU kernel.
- [x] ASTC LDR 4x4, 6x6 and 8x8 (1-3 partitions, dual plane, every weight grid of the footprint), to KTX2
  (sRGB or UNORM) or `.astc`, through the CLI, library, `mipple check` and the web demo. On Kodak, one
  thread on one P-core, effort 0 leads ISPC fast in PSNR, SSIM, Zstd size and time at 4x4 / 6x6 / 8x8
  (0.63 / 0.01 / 0.18 dB ahead at 0.86× / 0.98× / 0.97× its time); effort 2 is 0.96 / 0.31 / 0.44 dB ahead
  at about 2.5× its time (BENCHMARKS.md). astcenc was not available to grade.
  - [x] RDO for ASTC: its own pass over the surface (blocks are not 4x4) with whole-block copies, earlier
    heads re-weighted and earlier weights refitted. On Kodak at λ 1, Zstd size falls 39% / 20% / 13% at
    4x4 / 6x6 / 8x8, costing 2.9 and 0.21 dB at 4x4 and 6x6 and nothing at 8x8 (PSNR 0.03 dB higher;
    BENCHMARKS.md).
  - [ ] A GPU kernel, a SIMD batch encoder, other footprints (the decoder already reads every 2D one), HDR.
- [x] ETC2 / EAC for mobile fallbacks: ETC2 RGB (all five modes), ETC2 RGBA8 and EAC R11 / RG11 (unsigned),
  KTX2 only, with the RDO pass. On Kodak, one thread, effort 0 is 0.37 dB ahead of Compressonator in an
  eighth of its time and effort 1 0.94 dB ahead in half; with RDO, files are about 25% smaller at
  Compressonator's PSNR (BENCHMARKS.md). etcpak and etc2comp were not available to grade.
  - [ ] Punch-through alpha (ETC2 RGB A1, vkFormat 149/150), signed EAC, a SIMD batch encoder (eight blocks
    per call, as BC1/BC6H), a GPU kernel, and a cheaper RDO candidate error than a full decode.

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
  BC7 efforts 1-2 on the same kernel (mode-6 least-squares refits, then mode 1 on bc7f's partition
  estimate), after making the two-subset fits (principal axis, endpoints, LS refits, quantizer
  rounding and index projection) exact on the CPU at unchanged speed and quality. BC7 effort 3
  (mode 1 on the best line-fit-ranked partition), after making the partition ranking exact
  (scatter matrices cut to 10 bits, a 5-bit power step, floored estimates; -0.004 dB on Kodak,
  ~15% slower ranking on the CPU). BC7 efforts 4-5: subset refits on the ranked candidates and ±1
  endpoint/p-bit refinement of the mode-1 or mode-6 winner. WebGPU in the web demo: the same
  kernels behind an async API (`Gpu::open_async`, `Gpu::encode_surfaces_async`), every mip level
  and a 256-block spot check against the CPU in one submission with one readback (one wait on the
  device per encode: 6-12 ms base for a 512² BC1/BC4/BC5/BC7 e0-2 mip chain in Chrome, 42 vs 902 ms
  for BC7 e5 on one wasm thread).
  Next: BC7 modes 3/0/2/5 up to effort 8 (where the GPU should win; a workgroup per block, a
  thread per candidate) and the RDO alternates, BC1 efforts 2+ (needs per-lane cluster-fit pruning
  and exact split scoring).

## v1.2: "output B": latent-space supercompression + transcoder

- [ ] Intermediate format: transform-coded BC7/ASTC block parameters (weight-grid DCT, endpoint DPCM, context-modeled entropy coder), following Basis' XUASTC/XUBC7 ideas.
- [ ] Runtime transcoder (CPU + compute shader) to BC7, BC1, ASTC 4x4, ETC2; measure transcode Mpix/s.
- [ ] KTX2 vendor supercompression scheme + DFD for the intermediate.

## Pipeline quality (unversioned)

- [ ] RDO rate model for the split layout (`--prep`), so RDO and the transform stop fighting (design notes and
  other next steps: [research/field-split.md](research/field-split.md)); a KTX2
  key/value flag for the layout and a GPU (compute shader) inverse, so loaders can undo it after hardware
  decompression.
- [x] Alpha as transparency (BC7 colour textures by default): alpha-weighted mips, push-pull bleed of hidden colour,
  hidden texels beyond a 2-texel ring freed for the encoder; alpha-coverage-preserving mips (`--alpha-cutoff`).
- [ ] Alpha-weighted RGB error for partly transparent texels (per-texel weights in the BC7 encoders and RDO), and
  RDO matching of hidden texels to anything (now they are pinned to the visible mean).
- [ ] Better mip filters (Kaiser/Lanczos).
- [ ] Per-role presets (`--role albedo|normal|mask|ui`) choosing format, λ and smooth protection.
- [ ] Rate model for GDeflate (GPU decompression); `--backend` switch for verification.
- [ ] `ktx validate` (KTX-Software) in CI.
