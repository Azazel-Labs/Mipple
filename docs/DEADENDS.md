# Dead ends

Ideas that were tried, measured and dropped, with enough detail to tell whether a new idea is
the same one. Each entry gives what was tried, what it measured, and why it doesn't pay. Small
tweaks that changed nothing don't belong here. Neither does work that is only paused: that goes
in [ROADMAP.md](ROADMAP.md).

Measurements are on Kodak-24 unless noted. BD-rate is at equal quality, so negative is better.
CPU times come from paired runs on the same core.

## RDO (all formats)

- **Whole-chunk or very large searches for BC1.** Searching the whole chunk (2048 blocks) for
  BC1 gains only −0.09% BD-rate at 4.65× the time, so a learned BC1 ranker has nothing to find.
  BC1 blocks carry too little structure for distant matches to beat near ones.
- **Seeding a chunk with the base encoder's preceding blocks.** This came out at +2.9% BD-rate
  (16 KiB chunks). RDO copies those blocks, but RDO itself then changes them, so the matches it
  counted on aren't in the final output.
- **Larger chunks as a speed or size lever.** One chunk per image is only −0.86% (BC5) and
  −0.54% (BC1) against 16 KiB chunks. On 20 threads, 32 KiB chunks were slower (13.3 s against
  10-12.5 s for λ 0.5 over the corpus). 128 KiB chunks for BC7 are still open (see the roadmap),
  but only together with chunk rebalancing.
- **Splitting one block's search across threads (by back distance).** This was no faster on a
  155H: the join for each block waits for the slowest core.
- **Frozen (non-adaptive) LZ costs in the rate model.** They score even in `mipple check`
  (λ ≤ 4) but worse at λ 2-8 (−19.8% → −19.2% against the competitor).
- **Far index-pattern reuse via refit candidates.** Refitting endpoints to index patterns from
  2048 blocks gains −0.17%, at several times the time. It needs a cheap way to find patterns that
  fit before this is worth retrying.
- **Far endpoints for BC4 without a tighter bound.** −0.38% at 1.4× the time.

## BC7 RDO

- **Cutting refinement or the ranker's picks to save CPU.** `--refine-back 0` saves 16% CPU,
  `64` saves 12% and `--local 0` saves 26%, but each puts λ 40 SSIM behind Oodle (+0.2% to
  +1.0%). Refinement and the ranker carry real quality.
- **Far re-indexing of multi-subset heads (modes 0-3, 7).** These almost never won, and removing
  them moved every Kodak-24 delta against Oodle by 0.02% or less. Only mode-6 heads are worth
  re-indexing.
- **Hybrids in refinement: a mode-6 block trying other modes' first halves over its indices.**
  These never won on Kodak-24 or the Poly Haven test set.
- **Cheaper second base:**
  - Effort 6 or 7, or no uber step, for the second base loses 0.3-0.6%.
  - Fewer polished candidates, or a single uber step, loses 0.1-0.25%.
  - Building the second base only where the entropy-reduced base errs by 64λ or more costs 1.9×
    the CPU, because that base also bounds the search.
- **Ranker prefilter with 4 blocks per key** (instead of 8) lost about 0.07%.
- **Byte-identical micro-variants that didn't help:**
  - u16 hash-chain deltas;
  - skipping `texels_far` when cached;
  - an equality-mask intra scan;
  - skipping runs of eight dropped blocks in the far loop (±0);
  - deduplicating refinement candidates after their error test (+4-5%, since refinement
    candidates often repeat earlier ones).
- **A deduplicated prefilter scan at low λ.** Whole blocks and index halves are only 0-10%
  repeated within the ranker's 2048-block reach at λ 1 (30-60% at λ 4+), so deduplication pays
  only at higher λ.

## Field split (`--prep`)

- **The transform alone at higher λ.** The field split loses to RDO at higher λ, because RDO
  optimizes whole-block matches in the interleaved stream. Details are in
  [research/field-split.md](research/field-split.md). It needs an RDO rate model for the split
  layout before it can help there.

## BC6H

Measured on 4 Poly Haven HDRIs, single thread, mPSNR. ISPC fast scores 45.10 dB in 0.186 s;
mipple effort 0 scores 45.18 dB in 0.385 s. (The 20-image results are in BENCHMARKS.md.)

- **Picking the most precise mode whose deltas fit.** This loses quality. Precise modes with
  clamped deltas often win, so rank modes by estimated endpoint error after quantizing and
  clamping instead.
- **Predicting one region against two from line-fit estimates, then encoding only one.** This
  saves 20-35% of the time but loses 0.5-0.9 dB at effort 0. The estimates ignore endpoint
  quantization, which is where two-region modes lose.
- **An error pre-skip before ranking partitions.** At effort 0, 78% of blocks reach the
  two-region search and two regions win 76% of those, so there is little to skip.
- **Dropping the least-squares refit at effort 0.** It loses 0.17 dB for 22% less time, which is
  below the ISPC quality-time curve.
- **Scalar micro-optimization to reach ISPC fast's speed.** At effort 0 the encoder's time is
  spread over a dozen small functions (indexing, partition ranking, mode estimates, PCA, refit,
  packing), and no single one is worth 2×. ISPC runs 8 blocks per instruction, so matching it
  needs a batch encoder across blocks, as BC7 has.
- **EXR input through the `image` crate's `exr` feature.** It raised iteration builds from 2-4
  minutes to about 22. Radiance `.hdr` covers the test corpus. EXR could return through a
  separate helper crate or binary.

## ASTC LDR

Measured on Kodak images 1-3, single thread, effort 0, unless noted. ISPC fast takes 0.235 /
0.151 / 0.160 s at 4x4 / 6x6 / 8x8.

- **Compiling for AVX2 (`target-cpu=x86-64-v3`) to vectorize the per-block code.** About 1%
  faster at first, 6% after decimation went dense (0.424 / 0.379 / 0.324 s -> 0.401 / 0.357 /
  0.302 s). The loops run over 16-64 texels with gathers through grid taps and per-partition
  filters, which LLVM doesn't vectorize; rewriting polish's loops as structure-of-arrays for the
  autovectorizer did nothing either. Explicit i16 lanes (`src/astc/lanes.rs`) are what worked.
- **Stopping the candidate loop once an estimate exceeds the best error by a margin.** At 1.5×
  and 2× it saves under 3% at efforts 0 and 2. The shortlist is short and its estimates are close
  to each other, so few candidates get cut.
- **A smaller shortlist, or half of polish, at effort 0.** Measuring two decimations per
  candidate instead of four, or polishing only weights or only endpoints, each loses 0.1-0.3 dB
  at 6x6 and 8x8.
- **Scalar micro-optimization to reach ISPC fast's speed.** After the cheap early-outs, the time
  is spread over polish (about 21%), layouts, candidate encoding, decimation and spectra, and no
  single stage is worth 2×. As with BC6H, matching ISPC needs work across blocks or across
  texels in SIMD.

## ETC2 / EAC

Measured on the first 8 Kodak images (ETC2 RGBA: the twelve RGBA images of BENCHMARKS.md), effort 5,
λ 0.5, 1 and 2.

- **Field-swap refinement for ETC2 RGB.** `--refine-back 256` against 0 is even: sizes within 0.2% and
  PSNR within 0.02 dB at each λ, so it is off by default. It stays on for ETC2 RGBA, where the alpha half
  gains (1.2-1.7% smaller for 0.02-0.05 dB), and for EAC (RG11 2-4% smaller for 0.04-0.16 dB).

## Engineering

- **Closures in `#[target_feature]` kernels.** A closure, including one inside `array::from_fn`,
  doesn't inherit the caller's target features, so AVX2 intrinsics inside it become real
  function calls. This made the first batch encoder 3× slower than scalar. Batch kernels use
  plain loops.
