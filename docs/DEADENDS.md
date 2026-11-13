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
- **A 12x12 (3x3-block) window for the neighbourhood smoothness weight.** Against the four
  overlapping 8x8 windows at the same k it loses 0.5-0.9 points of BC7 size at Oodle Texture's
  RMSE (λ 10-20 fall behind Oodle) for no SSIM gain at λ ≥ 20. Large windows mask texels next to
  edges that the error metric still sees.
- **A smaller near-tie threshold for the deferred decision, to save time.** `--defer` 0.1 against 0.3
  (λ 1, one thread on one P-core) keeps most of the time cost: 1.49× / 1.13× / 1.27× the RDO time of no
  deferral for BC5 / BC7 / BC3, against 1.51× / 1.21× / 1.31×, for about two thirds of the BD-rate gain
  (BC5 −0.16% vs −0.27%). The cost is in deferring at all, so 0.3 stays for every format.
- **Averaging the per-channel smoothness terms** (k²/(k²+σ²) per channel instead of with the
  largest channel's variance). BC7 moves strongly toward RMSE (−1.9% to −2.8% size at Oodle
  Texture's RMSE) but falls behind Oodle on SSIM at λ 30-40 (+0.6% / +1.1%).

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
- **A shorter ranker reach (1024 or 512 blocks instead of 2048).** No measurable time saved (8 Kodak
  images, λ 1, one pinned core, interleaved best of three: 11.20 / 11.29 / 11.12 s) for 0.3% / 0.4% larger
  Zstd files. Profiles put the prefilter scan near 15% of λ 1 time, but halving it doesn't show up.
- **Per-field refinement depth.** Refining only 64 blocks back for the endpoint half saves about 2% of
  λ 1 time, and for the index half about 4%, but each shrinks the lead over Oodle by 0.1-0.3 points
  (λ 40 SSIM −1.5% to −1.3% / −1.2%).
- **Starting from the plain base instead of the entropy-reduced one (`--entropy off`).** Slower (36.9 s
  vs 32.9 s for Kodak-24 at λ 1 on one pinned core) and 2% larger at λ 1, and the SSIM lead over Oodle at
  λ 30 / 40 falls from −1.8% / −1.5% to −1.0% / −0.6%.
- **A different hash-chain depth for the rate model (16 now).** Depth 4 is slower (44 s vs 37 s for
  Kodak-24 at λ 1): the weaker model's estimates leave looser bounds, so the search does more work. Depth 8
  saves nothing, and depths 32 / 64 add 6% / 22% to RDO time. Quality moves by 0.1 point or less.
- **Byte-identical micro-variants that didn't help:**
  - u16 hash-chain deltas;
  - skipping `texels_far` when cached;
  - an equality-mask intra scan;
  - skipping runs of eight dropped blocks in the far loop (±0);
  - the candidate dedup table with each tag beside its key (one cache line per probe instead of two;
    ±0 over five paired runs);
  - building the mode-6 alternate base inside the plain effort-8 pass, which fits the same line (390/390
    check encodes byte-identical, but that pass is only about 2% of base time: 2.51 vs 2.58 s best of five,
    medians reversed);
  - skipping the Zstd safety check for every BC7 chunk (8 Kodak images at λ 1: 8.71 vs 8.63 s, same bytes;
    the rate-model skip already covers the chunks where it would cost);
  - deduplicating refinement candidates after their error test (+4-5%, since refinement
    candidates often repeat earlier ones).
- **Exact (greedy) pruning with the deferred decision kept.** Deferring needs the search's pruning relaxed by
  the same margin, so the runner-up is found. With exact pruning, the runner-up is whatever survives by
  chance: on 8 Kodak images at λ 1 that keeps 31% of deferral's size gain (1,599,944 bytes against 1,590,486
  with deferral and 1,604,292 without) for 41% of its time saved (RDO 7.79 s against 8.42 / 6.90 s).
- **Gating refinement per block.** Refinement's wins are spread thin: no single feature predicts them (the
  best candidate's bits, best J over the starting J, runner-up over best, J itself). The most selective cut,
  runner-up over best above 1.3, skips only 17% of blocks while dropping 3% of refinement's gain.
- **Larger near-flat mode-6 gates on the BC7 RDO bases.** The plain effort-8 base keeps mode 6 when a
  block's mode-6 error is at most 384 and the entropy-reduced base when its score is at most 512 (both
  gains, see `bc7::rdo_alt_cfg`). Higher gates trade RMSE for SSIM vs Oodle on Kodak-24 rather than gain:
  the plain gate at 512 (with the reduced one at 512) moves λ 20 / 30 at RMSE from −1.7% / −2.3% to −1.6% /
  −2.2% while SSIM gains 0.1-0.2, at 768 to −1.3% / −1.5%, and at 4096 alone λ 30 at RMSE goes to +0.2%. The
  reduced gate at 1024 or 2048 moves λ 40 at RMSE −3.2% to −3.1% for about 2% less time.
- **The batch (effort-5 search) encoder as BC7 RDO's plain second base.** It is the encoder that builds the
  entropy-reduced base, so it costs a fraction of the effort-8 alt, but RDO keeps much of the alt's extra
  quality: Kodak-24 vs Oodle at RMSE λ 20 / 30 / 40 went from −1.7% / −2.3% / −3.2% to −1.1% / −0.9% / −2.1%
  (SSIM 0.2-0.3 behind too), for no measurable time saved on 8 Kodak images (the gated effort-8 alt is already
  only about 10% of λ 1).
- **A deduplicated prefilter scan at low λ.** Whole blocks and index halves are only 0-10%
  repeated within the ranker's 2048-block reach at λ 1 (30-60% at λ 4+), so deduplication pays
  only at higher λ.
- **Video-codec style fast-decision gates (skip the far search, build the plain alt lazily) as a
  25% speedup.** The ceilings are too low (Kodak 1-6, λ 1-8). Once the far back-reference search runs
  after the ranker's picks (same quality, −0.02% BD-rate), it improves only about 11% of blocks for
  1.2-1.5% of the summed J, and skipping it outright saves only about 20% of RDO time at +1.6% / +2.2%
  BD-rate (PSNR / SSIM). Single-feature gates (slack/λ, bits, ranker score) lose 22-35% of the far
  search's gain for 30% of its time; a linear gate over eight features, fitted on Poly Haven logs, was
  no better end to end than skipping whenever J is already below 0.8× its starting value (about −13% RDO
  time for +0.24% / +0.11%). Dropping the plain effort-8 alternate base saves its 13% of encode time
  but makes RDO about 10% slower, since its start bounds the search, at +1.3% BD-rate; its wins can't
  be predicted from the entropy base's error (computing it for 39% of blocks keeps half its wins at λ 1).
- **Rate-constrained elimination (tighter error bounds for candidates that can't continue a match).**
  A candidate that can't take the 1-bit match continuation costs at least λ·min_bits, so its error
  bound can drop to (best − λ·min_bits)·16/smooth in the far prefilter, the re-index and the near
  search. Exact (byte-identical on 576 bench rows and every check encode), but paired pinned timing
  on Kodak showed no gain (0.97-1.08× across λ 1-16, the sign flipping between runs). A continuation
  is possible in 15-95% of blocks, and there the candidate generator's memos need the loose bound
  (they assume the bound only shrinks); elsewhere λ·min_bits is a small part of best. The matching
  whole-block exit (best ≤ λ·min_bits) can never fire: without a continuation every candidate,
  the base included, costs at least the minimum.
- **A pattern-establishment prepass (crunch / GST style).** Each chunk encoded twice; in pass 2
  block i also tries the endpoint and index fields pass 1 chose for the next 256 blocks, with a credit
  of C bits for fields pass 1 introduced. Best BC1 result (6 Kodak images): C = 4, −0.66% BD-rate for
  2.4× the RDO time; C ≥ 16 loses (+1.6% to +12%), the credited fields costing more distortion than
  their reuse saves. Offering the base encoding's future fields instead of pass 1's did better for less
  (−0.7% to −1.0% at 1.8×), but the control with the previous 256 blocks' base fields gave −0.22%, so
  the real establishment effect is only about 0.45 points. BC7 (mode-6 endpoint fields of future base
  blocks): −0.35% to −0.50% BD-rate PSNR, −0.14% to −0.19% SSIM for 1.6-2× the RDO time.
- **Carrying re-index lower bounds from the neighbouring block (Elkan / LAESA).** The carried bound
  Σ max(0, √e_t − δ_t)² from the left or upper block's exact per-texel palette errors is valid, but
  neighbouring blocks are usually more than 2√T apart in weighted distance, so the triangle
  inequality rarely bites. Even as a free, ideal cache it would remove only 0.2-1.5% of BC7 RDO
  cycles at λ ≥ 4 and 0.4-6% at λ 1, before paying for the bound and the caches.
- **Steered re-indexing of mode-6 index bytes toward window runs ("lossy LZ" indices).** For the
  chosen block's endpoints and the 4 lowest-error re-indexed endpoint pairs of the search, index
  bytes were forced to equal the window's bytes at a repeat offset or 16·d back (d ≤ 16-64), the
  other indices left nearest; a per-texel, per-index error table priced each span in O(1) and
  survivors were costed exactly. Kodak-24: −0.11% / −0.23% BD-rate at equal PSNR (−0.88% / −1.11%
  at equal SSIM) for 1.9× / 3.2× the RDO time; repeat offsets alone gave +0.15% PSNR, −0.54% SSIM
  for 1.46×. Steered candidates win in 15-28% of blocks, but each saves few real Zstd bits: the
  rate model over-values 3-4-byte matches (a minimum span of 4-5 beats 3), and an estimated-J
  shortlist ranked too poorly to cut the pricing. Most of the gain is the endpoints plus a partial
  index run from the same source, so a cheap retry should start there, and only if SSIM is the
  target.

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
- **Insetting the line-fit endpoints at the one-region effort 0.** Pulling both ends in by a
  fraction of the fitted extent (all 20 HDRIs, mPSNR): with no refit, 1/64 gains 0.007 dB and
  1/32 / 1/16 lose 0.20 / 1.12 dB; with the refit, 1/32 loses 0.08 dB. The refit already moves the
  ends where they should be.
- **Skipping the effort 0 refit when the first trial's error is small.** Lanes are eight blocks,
  and time falls only when all eight skip: a threshold that saves 21% loses 0.10 dB, the same as
  no refit at all. It would need lanes compacted across groups to pay. Keeping the indices
  through the refit (re-quantizing the refitted ends only) costs 0.02 dB and skips the whole second
  indexing instead.
- **Indexing by projection only (no check of the neighbouring index).** At effort 0 it loses
  0.016 dB on the first indexing, 0.03 dB on the refit and 0.06 dB on both.
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
- **Reworking the estimate shortlist's data structure.** Its insert line shows as about 12% of
  samples at 6x6, but the cost is branch misses across the whole estimate loop (about 150
  estimates and 70 inserts per block, close to a coin flip each). Collecting every estimate in a
  buffer and trimming it once at the end was slower. A branchless insert of packed (score bits,
  index) keys by min/max came out within 3% (kept, since it is no slower). Visiting only the grids
  whose first estimate's bound (computed for every grid at once, four lanes at a time) is under
  the shortlist's last score cut the grid visits from 55 to about 20 per block at 6x6, output
  byte-identical, and was no faster either. The lever is fewer estimates, not a faster insert.
- **Summing the measured decimation error in lanes.** Taking `decimate_err`'s error sum eight
  texels at a time, with the grid padded once per block instead of per candidate, timed identical
  to the scalar loop in interleaved runs on one P-core. Decimation's cost is the per-grid filter
  taps, not the sum.
- **Scalar micro-optimization to reach ISPC fast's speed.** After the cheap early-outs, the time
  is spread over polish (about 21%), layouts, candidate encoding, decimation and spectra, and no
  single stage is worth 2×. What finally matched ISPC fast was less work per block, not faster
  code: one candidate encoded instead of two, which a better candidate ranking (decimation error
  weighted 1.5×) made affordable, and the dual-plane layouts gated by how far the texels sit from
  one line. Going well past ISPC's speed likely needs work across blocks in SIMD, as for BC6H.
- **Other ways to spend effort 0's time at 6x6.** Kodak 24, one P-core, against ISPC fast in the
  same run (1.13-1.18 s, 37.563 dB). The old effort 0 (one candidate, three polish passes,
  shortlist 4) gave 37.577 dB at 1.21 s. With two candidates: one polish pass and the default
  shortlist of 8 is 37.639 dB at 1.23 s; two passes 37.692 dB at 1.36 s; three passes 37.702 dB
  at 1.39 s; a refine round on top 37.766 dB at 1.62 s. Three candidates with one pass is
  37.686 dB at 1.47 s. Each shortlist entry costs about 7% time, and a shortlist of 2 loses
  0.16 dB. Dropping the dual-plane layouts loses 0.09-0.15 dB for little time. Raising the
  dual-plane and 2-partition line thresholds, which gate those layouts by how far texels sit from
  one line, barely saves any time. Gating the second candidate by its estimate against the
  first's measured error saves little time and loses quality, like the estimate-margin stop above. Kept: two candidates,
  two passes and a shortlist of 5 (37.649 dB at ISPC's time). A second candidate makes Zstd
  output about 0.9% larger than the old effort 0, though it is still 1.7% smaller than ISPC's.
- **A second polish pass on ASTC RDO candidates.** Polishing a re-weighted or refitted candidate's
  re-chosen half twice instead of once is even on Kodak 24 (BD-rate within 0.25% at every footprint)
  for up to 7% more RDO time; not polishing at all costs 0.5-1.4%. One pass stays.
- **Looking further back than 256 blocks for ASTC re-weights and refits.** 512 instead of 256 is
  another 0.3% / 0.5% / 0.7% smaller at equal PSNR at 4x4 / 6x6 / 8x8 for 1.8× the RDO time (one thread,
  one P-core, with the error lower bounds on). 256 against 32 was worth 4.8-6.6% for 2.4×, so the curve
  has flattened; 256 also matches the whole-copy window.

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
