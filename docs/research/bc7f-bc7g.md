# bc7f and bc7g — notes for mipple

Sources: Geldreich's post ["bc7f: new real-time analytical BC7 encoder" (Jan 2026)](https://richg42.blogspot.com/2026/01/bc7f-new-real-time-analytical-bc7.html),
the Basis Universal wiki page ["Transcoder Internals: Analytical Real-Time Encoders and Transcoders"](https://github.com/BinomialLLC/basis_universal/wiki/Transcoder-Internals-Analytical-Real-Time-Encoders-and-Transcoders),
and the source in `basis_universal/transcoder/basisu_transcoder.cpp` (namespace `bc7f`, ~9,300 lines, commit 073eb6a, 2026-10-30).
License: Apache 2.0 (Basis Universal). bc7enc_rdo is MIT / public domain, except `bc7e.ispc` (Apache 2.0).

## bc7f in one paragraph

bc7f encodes a BC7 block **without trial encodes**. It computes block statistics once (mean, integer
covariance, per-channel min/max, channel-pair correlations), predicts the squared error of each mode family
with closed-form estimates, and encodes only the predicted winner ("analytical"). Optional flags make it
"partially analytical" (compute real SSE on hard blocks to break ties) or "non-analytical" (brute force at
the top level). It covers the whole format: all modes, all partitions, dual-plane channels, and p-bit
optimization. Per the wiki, it is "several times faster" than `bc7e.ispc` level 1 without SIMD, at roughly
bc7e level ~0.8 quality. It also serves as the real-time BC7 fallback when Basis transcodes XUASTC.

## The tricks, with where they live

| Idea | How bc7f does it | Source |
| --- | --- | --- |
| Solid blocks | Exact compare of all 16 texels → mode 5 solid packer | `fast_pack_bc7_rgb_analytical` |
| Trivial blocks | If max channel variance < threshold: mode 6 from the luma-extreme texels, p-bits optimized, done | same |
| Line-fit error | `ortho_var = trace(cov) − aᵀ·cov·a`, with `a` one power-iteration step from the max-variance diagonal | `estimate_slam_to_line_sse_3D/4D` |
| Quantization error | `16 · [ Δe² · (1/12) · ((2N−1)/(3(N−1))) · 255² · chans · pbit_discount + Δw² · (1/12) · Σ span_c² ]`, with Δe = 1/(endpoint levels − 1), Δw = 1/(weight levels − 1), N = weight levels | `analytical_quant_est_sse` |
| Mode-pair choice | Same partition, compare estimates: 1 vs 3 (6-bit/3-bit vs 7-bit/2-bit), 0 vs 2, 4 vs 5 — "large span → more index bits" | `pack_mode1_or_3_rgb`, `pack_mode0_or_2_rgb`, `pack_mode4_or_5` |
| When to try 2/3 subsets | Only if `ortho_var / total_var` exceeds a ratio threshold and absolute orthogonal energy is high | `fast_pack_bc7_rgb_analytical` |
| 2-subset partition in one shot | Split texels by sign of `(p − mean)·axis` → 16-bit "desired" mask → pick the BC7 pattern with min `popcount(mask ^ pattern)` (or its complement) | `pack_mode1_or_3_rgb` |
| 3-subset partition | 2-way split as above, then split the higher-SSE half between its two luma-extreme texels; match against 3-subset patterns with popcount row sums | `determine_3subsets`, `pick_3subset_pat_index` |
| Dual plane (modes 4/5) | If one channel is weakly correlated with the others (min Pearson \|r\| < threshold), put it on the second plane; choose rotation from the correlations | `fast_pack_bc7_rgb_analytical` |
| Endpoints | Integer principal axis (scaled ×2048), project each texel, take extremes (index packed in low 4 bits so min/max returns the texel), then least-squares refit | `bc7_proj_minmax_indices_*`, `compute_least_squares_endpoints_*` |
| P-bits | Closed-form best unique / shared p-bits for the endpoint pair | `determine_unique_pbits`, `determine_shared_pbits` |
| ASTC compatibility | Optional flag restricts to the partitions BC7 and ASTC 4x4 share, so blocks transcode losslessly | `cPackBC7FlagASTCCompatible` |

## bc7g

Announced, not released. Geldreich describes it as bc7f's analytical techniques combined with SIMD, aiming
for a few CPU threads to compete with GPU encoders. The wiki says it "is in the works" and will be open
source. No code or numbers yet. `scripts/head2head.py` has a `--bc7g` hook for when it ships.

## What this means for mipple

1. **Base-encoder speed.** mipple ranks up to 64 partitions by line-fit residual, then trial-encodes the
   survivors. bc7f's sign-split + popcount match picks a near-best partition in about 64 popcounts and no fits.
   Adopting it, plus the solid and trivial fast paths, should make effort 0–1 several times faster.
2. **Mode choice without trial encodes.** The quantization-error estimate is cheap and separates the mode
   pairs well. Use it to choose which modes to trial-encode at effort 1, and trial-encode everything only at
   effort 2–3.
3. **Modes 4/5.** bc7f's correlation test for when dual-plane pays is the missing piece for our encoder.
4. **RDO interaction.** RDO quality is capped by the base encoder, and RDO wants alternatives cheaply. The
   analytical estimates double as a fast distortion prior when choosing among RDO candidates.
5. **Licensing.** Porting ideas (math, heuristics) is fine. Copying code means Apache 2.0 notices, which is
   compatible with any license we choose, but must be recorded in NOTICE.
