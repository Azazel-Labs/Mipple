# Benchmarks (v0.9, 2026-11-07)

Sizes are measured **after** a real compressor (Zstd level 19, Deflate level 9) on the raw blocks of the top
mip level. Raw BC1/BC4 = 4 bits/texel (bpt), BC5/BC7 = 8 bpt. PSNR is over the channels the format stores, as
a per-image mean. mipple RDO defaults: window 2048 bytes and 16 KiB chunks (BC7: 16 KiB and 64 KiB; BC1: 8 KiB window; BC5: 3072
bytes), 256 back-references (capped by the window: 192 for BC5), partial-span copies from the nearest 8, smooth scale 5 (ramping to 1 at channel std 14), max RMS ratio 10, min RMS 3. Machine: 2-core cloud VM; timings are only comparable within a table. Raw rows are in `bench-results/`.

## Base encoders vs competitors, one P-core (Kodak first 6)

`scripts/compete.py --threads 1 --base-only --affinity 0x1000 --repeat 3` on an Intel Core Ultra 7 155H:
every encoder pinned to one performance core (logical CPU 12) and timed with its own timer, keeping the
fastest of three runs per image. Note: on this CPU logical CPUs 2-9 and 20-21 are efficiency cores. A
competitor config counts as beaten when some mipple effort is at least as good (PSNR) and at least as fast. The
tables below are from development runs on the first 6 images; the committed rows are the all-24 run
(`bench-results/compete-kodak24-1t-p12/`, next section).

Timing notes: at these speeds an image takes about a millisecond, so mipple prints times with microsecond
precision (it printed 0.1 ms before, which quantized fast configs by ~15%). Every encoder starts from the
RGBA image inside its timed region: the harness used to build Intel ISPC's R8/RG8 input plane before
starting the clock, which the other encoders (stb_dxt, rgbcx, mipple) do not get; it is now filled inside
the timed loop with a plain strided copy. The harness still allocates and zeroes the output before timing,
while mipple's time includes allocating its output.

### BC7 top efforts vs bc7e u6

Interleaved, fastest of 2. Effort 8 adds bc7e-style "uber" refits
(least-squares fits from index sets with the extremes nudged inward and stretched, for every multi-subset
mode and mode 6) and refines every mode's best encoding and its two best partition candidates before the modes
compete; effort 9 refines every candidate.

| Config | PSNR (dB) | Time, 6 images (s) |
| --- | --- | --- |
| mipple e7 | 45.73 | 1.44 |
| **mipple e8** | **46.10** | **9.36** |
| bc7e u6 (`-u6`, bc7e.ispc) | 46.08 | 11.30 |

With that, every base-encoder configuration of every competitor in these runs is beaten on both PSNR and
speed by some mipple effort, for all four formats.

### RDO vs bc7enc_rdo

`bench-results/compete-kodak24-rdo/`: all 24 Kodak images, 20 threads per encoder. mipple effort 5 λ 0.25-8
against bc7enc_rdo's RDO at its top base level, λ 0.1-4 (BC7: bc7e u6 λ 0.05-1 and bc7enc.cpp `-e`). BD-rate of
mipple's curve (negative = fewer bits at equal PSNR; it does not depend on the thread count) and the range of
total encode times over each RDO sweep (base encode plus RDO, 24 images):

| Format | vs | Zstd-19 | Deflate-9 | mipple time (s) | competitor time (s) |
| --- | --- | --- | --- | --- | --- |
| BC1 | bc7enc_rdo rgbcx L18 | −26.8% | −23.9% | 7.8-11.7 | 11.9-13.3 |
| BC4 | bc7enc_rdo rgbcx hr5 | −20.0% | −18.4% | 4.1-5.9 | 8.3-9.4 |
| BC5 | bc7enc_rdo rgbcx hr5 | −26.1% | −23.8% | 6.6-9.8 | 10.8-12.3 |
| BC7 | bc7enc_rdo bc7e u6 | −31.5% | −29.2% | 17.1-28.5 | 51.8-57.8 |
| BC7 | bc7enc_rdo bc7enc.cpp -e | −30.5% | −28.6% | 17.1-28.5 | 38.1-44.4 |

Each format's RDO rows for both encoders were run together, interleaved per image, so each table row compares
like with like; the BC7 rows come from a later run with the far search (below), the BC1 rows from a later one with
far endpoints (below), the BC4 rows from a later one with frozen literal costs (below), the BC5 rows from a
later one with the localized search (below); mipple's rows were then rerun alone with the smooth weighting
below (bc7enc_rdo's rows are unchanged). BC5 is the closest: mipple's slowest setting (λ 0.25, 9.8 s) is 9% faster
than bc7enc_rdo's fastest (λ 2, 10.8 s). With the error floor at RMS 1.5 per channel instead of
3 (before `--min-rms`), BC4 was −11.0% and BC5 −16.1%; BC1 and BC7 are unchanged.

Every mipple RDO setting is faster than every bc7enc_rdo RDO setting of the same format, at better rate-distortion.
The wide search (256 back-references) became affordable through: field-level dedup (a previous block whose
endpoints or indices an earlier one already had adds no new re-index/refit candidate, and BC4/BC5 half results
are memoized by field value), fused bounded error kernels for BC1/BC4/BC5 (stop once the error cannot win; BC5
reuses the base block's half errors), partial spans only from the nearest 8 blocks, an exact lower bound on the
LZ cost to skip candidates that cannot win, and table-driven literal/length/offset costs. All of these except the
search width and the span distance leave the output byte-identical. BC7 spends the time it has left over on
context: an 8 KiB window and 64 KiB chunks (6 per Kodak image) instead of 2 KiB and 16 KiB took BC7 from
−23.7% / −20.8% to −28.4% / −26.7% at about 1.5× the time (12.2-18.9 s before).

Field-swap refinement (`--refine-back`, default 256 for BC1/BC4): after the search picks a block, try replacing
one of its fields (endpoints or indices) with that field of each of the previous 256 blocks, so a block can
take its endpoints from one earlier block and its indices from another. Kodak-24 BD-rate (`mipple check`): BC1
−0.98%, BC4 −0.39%; vs bc7enc_rdo BC1 −19.2% → −19.8%, BC4 −16.8% → −17.0%. It is off for BC5, where it gains
−0.17% at depth 32 and −0.31% at 256 for 4% and 20% more RDO time, which BC5 cannot spare against bc7enc_rdo.

Literal costs: the rate model used to learn its order-0 literal costs from the bytes emitted so far in the
chunk, so the first blocks of every chunk saw near-uniform 8-bit literals, and RDO's own copies skewed the
counts. BC4/BC5/BC7 now fix them per chunk from the chunk's base encoding: Kodak-24 BD-rate (`mipple check`)
BC4 −1.24%, BC5 −0.44%, BC7 −0.06% (Kodak-8); vs bc7enc_rdo BC4 −17.0% → −17.9%, BC5 −20.7% → −21.1%, at the
same or lower time. Variants that did no better: counting only bytes the parse codes as literals (BC4 −0.45%),
statistics over a sliding 2-8 KiB of base blocks or the whole image, and scaling the costs by 0.85 or 1.15. BC1
keeps the learned costs: frozen ones are even in `mipple check` (λ ≤ 4) but worse at λ 2-8, −19.8% → −19.2%
vs bc7enc_rdo.

Far endpoints (`--far-back`, default 1024 for BC1): beyond the 256-block search, try the endpoints of every
earlier block in the window (BC1's window is now 8 KiB, 1024 blocks), re-indexed for the current block. Oodle
Texture reuses endpoints from hundreds to thousands of blocks back, which a 256-block search cannot see. It is
made cheap by trying each distinct pair once, at its nearest position (a per-position "latest occurrence"
flag, no hashing in the scan), and only when a lower bound on its error leaves it a chance: the mean of any
indices' palette colors lies on the endpoints' segment, so the block error is at least 16 times the squared
distance from the block's mean color to that segment (less a slack for the truncated interpolation). About 1 in
5 distinct pairs passes; 3-color pairs are skipped. Kodak-24 BD-rate (`mipple check`) −0.55% for 1.09× the
time (2048 blocks with a 16 KiB window: −0.59%, 1.16×; trying every far block's endpoints and indices without
the bound: −0.69%, 4×); vs bc7enc_rdo −19.8% → −20.4%. Far index patterns (refit endpoints) add only
−0.17% at several times the cost. Not on by default for BC4, where the bound (each texel's distance from the
endpoint interval) passes too many pairs: −0.38% for 1.4× the time. BC5 is limited by depth instead.

Deeper BC5 search: BC5's window is now 3072 bytes, so its search reaches 192 blocks instead of 128 (Kodak-24
BD-rate −2.8%; 256 blocks: −3.6%). It became affordable by making each back-reference cheaper, all
byte-identical: BC4/BC5 candidates carry their exact error from generating them (whole-half copies, re-indexed
endpoints and refit indices are memoized per field value together with their error, and combined candidates
sum half errors), so losers are dropped before any decode; a half an earlier back-reference already had is
skipped outright; a re-index whose endpoint interval alone already loses is deferred until a combined
candidate needs it; and the LZ cost stops as soon as its bits exceed what could still win. Kodak-24 BC5 RDO
time at the old depth fell to 0.69× (BC4 0.9×), which pays for the 192-deep search; vs bc7enc_rdo BC5 went
from −21.1% to −23.5% at 7.4-9.2 s against its 9.7-11.2 s. Restricting the far half of the search to some
candidate types does not make it cheaper. A tighter LZ-cost lower bound (exact literal costs, matches only
where the 3-byte prefix occurs earlier) skipped almost nothing: RDO's candidates are copies, so they nearly
always contain a match.

Localized BC5 search (`--local`, default 16 for BC5): instead of the nearest 192 blocks, BC5 searches the nearest
128 plus 16 blocks picked from anywhere earlier in the chunk, plus the blocks around the one directly above
(±2). The picks are made in two steps: a shape distance (sum of absolute differences between the two blocks'
mean-removed 4×4 texels, both channels) keeps the 128 most similar, then a small neural network (23 features of
the pair: shape and level distances, base-block cost, distance, λ; one hidden layer of 16, ReLU) ranks those and
keeps 16. Every pick then goes through the normal exact candidate search, so the network only decides where to
look, never what is written. Inference is plain Rust (SSE2 with the scalar operation order, no FMA), so output
is identical on every machine; a unit test pins it to the scalar sum. The network was fitted
(`scripts/train_ranker.py`) only on CC0 textures from Poly Haven (`scripts/fetch_polyhaven.py`: every 27th
texture, 1k diffuse and normal maps; 323 encodes), never on Kodak or RAD's images. Ranked by the network, the
best back-reference is in the top 16 for 63% of blocks where one helps, against 50% for shape distance alone.
BD-rate: Kodak-8 −0.90%, Kodak-24 −0.76%, a disjoint Poly Haven held-out set −0.44%, at about the same time
(Kodak-24 1.02×); vs bc7enc_rdo BC5 went from −23.5% to −24.1% (Zstd-19), still faster at every setting.
`--local 0` restores the 192-deep search byte for byte.

Localized BC4 search (`--local`, default 32 for BC4, with `--max-back 128`): the same idea with a single-channel
ranker (`src/ranker_bc4.rs`). Beyond the nearest 128 blocks, 64 earlier blocks of the chunk by shape distance and
64 by value distance (exact integer SAD) go to a 13-16-1 network, which keeps 32; those and the blocks around the
one above get the exact search. Trained (`scripts/train_ranker_bc4.py`) only on single-channel images derived
from CC0 Poly Haven textures (324 encodes, 9.35M candidate rows). On a validation split the best block is in its
top 32 for 42% of groups against 41% for the best hand-made heuristic (min of refit residual and copy SSE) and 34%
for shape distance; end to end that heuristic gives −0.47% against the network's −0.67%. BD-rate against the old
256-deep search: held-out Poly Haven (48 images never used for tuning) −0.71% Zstd-19 / −0.85% Deflate, Kodak-24
−0.77% / −1.00%, at 0.95× the time (20 threads). `--local 0` restores the old search byte for byte.

BC7 far search (`--far-back`, now default 1024 for BC7, with a 16 KiB window): beyond the 256-block search, try
each earlier block in the window as a whole-block copy, and each distinct endpoint field re-indexed. Two exact
bounds keep it cheap: a copy is skipped when the difference of the two blocks' mean colors already costs more
than the best candidate (Cauchy-Schwarz), and a re-index when, in some subset, the mean texel is too far from
the endpoints' segment (as for BC1, now computed from the dequantized endpoints before any palette is built,
with the per-partition subset means cached per block). Re-indexing also stopped unpacking the index bits it
replaces. At the old settings BC7 RDO is byte-identical and 0.94× the time (Kodak-8). Kodak-8 BD-rate
(`mipple check`): −0.85% at 512 blocks for 1.07×, −1.46% at 1024 (16 KiB window) for 1.25×, −1.67% at 2048
(32 KiB); a wider window alone is neutral. Copies alone give −0.52% at 1024 almost for free. Kodak-24 vs
bc7enc_rdo: bc7e u6 −28.4% → −29.4%, `-e` −26.7% → −27.9%, at 22.8-34.0 s against 38.1-57.8 s.

Chunk boundaries cost little: on Kodak-24 BC5 (before frozen literals), 64 KiB chunks are −0.78% BD-rate
against the default 16 KiB and a single chunk per image −0.86% (BC1: −0.54%), but with 24 chunks per image on
20 threads larger chunks are slower (32 KiB: 13.3 s vs 10-12.5 s for λ 0.5 over the corpus). An earlier note
here put 64 KiB at −2.0%; that does not reproduce at the current defaults. Seeding each chunk's rate model with
the preceding blocks as the base encoder left them made it much worse (+2.9% at 16 KiB): RDO copies those
blocks, and the thread optimizing them changes many.


### BC4 and BC5

Fastest of 10 runs per image.

BC4:

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |
| --- | --- | --- | --- | --- | --- |
| mipple e0 | 42.91 | 3.834 | 3.840 | 0.01 | 453.71 |
| mipple e1 | 43.52 | 3.835 | 3.842 | 0.01 | 198.26 |
| mipple e2 | 44.28 | 3.830 | 3.836 | 0.03 | 82.78 |
| mipple e5 | 44.64 | 3.831 | 3.837 | 0.06 | 36.58 |
| mipple e6 | 44.81 | 3.831 | 3.838 | 0.11 | 22.24 |
| mipple e8 | 44.98 | 3.831 | 3.838 | 0.22 | 10.71 |
| rgbcx hl | 43.08 | 3.833 | 3.840 | 0.01 | 196.61 |
| rgbcx hr5 | 44.79 | 3.817 | 3.827 | 5.65 | 0.42 |
| stb_bc4 | 43.08 | 3.831 | 3.837 | 0.02 | 100.82 |
| ispc_bc4 | 42.79 | 3.835 | 3.841 | 0.01 | 337.04 |

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hl | 43.08 | 196.61 | mipple e1 (43.52 dB, 198.26 Mpix/s) | ✓ |
| rgbcx hr5 | 44.79 | 0.42 | mipple e6 (44.81 dB, 22.24 Mpix/s) | ✓ |
| stb_bc4 | 43.08 | 100.82 | mipple e1 (43.52 dB, 198.26 Mpix/s) | ✓ |
| ispc_bc4 | 42.79 | 337.04 | mipple e0 (42.91 dB, 453.71 Mpix/s) | ✓ |

BC5:

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |
| --- | --- | --- | --- | --- | --- |
| mipple e0 | 42.99 | 7.332 | 7.358 | 0.01 | 280.87 |
| mipple e1 | 43.59 | 7.337 | 7.364 | 0.02 | 118.56 |
| mipple e2 | 44.35 | 7.341 | 7.366 | 0.05 | 44.10 |
| mipple e5 | 44.70 | 7.344 | 7.368 | 0.11 | 21.33 |
| mipple e6 | 44.87 | 7.346 | 7.370 | 0.17 | 13.66 |
| mipple e8 | 45.03 | 7.348 | 7.373 | 0.48 | 4.88 |
| rgbcx hl | 43.17 | 7.337 | 7.364 | 0.03 | 81.36 |
| rgbcx hr5 | 44.87 | 7.327 | 7.358 | 11.01 | 0.21 |
| stb_bc5 | 43.17 | 7.335 | 7.361 | 0.05 | 50.74 |
| ispc_bc5 | 42.86 | 7.331 | 7.359 | 0.01 | 205.16 |

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hl | 43.17 | 81.36 | mipple e1 (43.59 dB, 118.56 Mpix/s) | ✓ |
| rgbcx hr5 | 44.87 | 0.21 | mipple e6 (44.87 dB, 13.66 Mpix/s) | ✓ |
| stb_bc5 | 43.17 | 50.74 | mipple e1 (43.59 dB, 118.56 Mpix/s) | ✓ |
| ispc_bc5 | 42.86 | 205.16 | mipple e0 (42.99 dB, 280.87 Mpix/s) | ✓ |

### BC7

`--repeat 2 --interleave` (per image, every config runs in rotation and keeps its fastest time, so load that
drifts over minutes hits all configs alike). Efforts 0-7 run
`src/bc7/fast.rs`, read straight from the surface: mode 6 (exact two-candidate indices, least-squares refits),
then mode 1 on 1-4 partitions per lane ranked by bc7f's popcount estimate or by line-fit error; from effort 4
±1 refinement of each lane's winner; at efforts 6-7 also modes 3, 0, 2 (3-subset partitions ranked the same
way) and dual-plane mode 5 (each color channel rotated into the scalar plane). Like the scalar encoder and
bc7e, opaque blocks pick p-bits by endpoint error, so alpha can decode as 254 (opaque images are scored on
RGB). Efforts 8-9 are the full encoders. Compressonator and texconv are not in this run (both were far behind
in the earlier runs).

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |
| --- | --- | --- | --- | --- | --- |
| mipple e0 | 41.62 | 6.685 | 6.864 | 0.02 | 155.22 |
| mipple e1 | 43.34 | 6.777 | 6.941 | 0.03 | 72.37 |
| mipple e2 | 44.05 | 6.956 | 7.095 | 0.08 | 28.12 |
| mipple e3 | 44.86 | 7.109 | 7.238 | 0.17 | 13.52 |
| mipple e4 | 45.39 | 7.193 | 7.312 | 0.50 | 4.68 |
| mipple e5 | 45.46 | 7.212 | 7.331 | 0.59 | 4.02 |
| mipple e6 | 45.71 | 7.342 | 7.439 | 1.74 | 1.36 |
| mipple e7 | 45.73 | 7.345 | 7.442 | 2.38 | 0.99 |
| mipple e8 | 45.89 | 7.466 | 7.558 | 40.91 | 0.06 |
| mipple e9 | 45.91 | 7.486 | 7.574 | 134.40 | 0.02 |
| bc7e u0 | 41.79 | 6.732 | 6.900 | 0.06 | 38.05 |
| bc7e u2 | 45.40 | 7.384 | 7.495 | 1.00 | 2.37 |
| bc7e u4 | 45.70 | 7.364 | 7.469 | 1.93 | 1.22 |
| bc7e u6 | 46.08 | 7.368 | 7.477 | 16.96 | 0.14 |
| bc7f:fastest | 41.35 | 6.574 | 6.766 | 0.05 | 43.53 |
| bc7f:faster | 41.35 | 6.574 | 6.766 | 0.06 | 41.10 |
| bc7f:fast | 42.99 | 6.729 | 6.892 | 0.08 | 30.60 |
| bc7f:default | 43.88 | 6.947 | 7.088 | 0.13 | 18.07 |
| bc7f:partial | 44.22 | 7.014 | 7.147 | 0.22 | 10.88 |
| ispc_bc7:ultrafast | 41.78 | 6.730 | 6.899 | 0.06 | 42.13 |
| ispc_bc7:veryfast | 45.22 | 7.312 | 7.434 | 0.81 | 2.93 |
| ispc_bc7:fast | 45.38 | 7.379 | 7.496 | 1.37 | 1.72 |
| ispc_bc7:basic | 45.58 | 7.410 | 7.515 | 3.36 | 0.70 |
| ispc_bc7:slow | 45.69 | 7.462 | 7.560 | 12.49 | 0.19 |

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| bc7e u0 | 41.79 | 38.05 | mipple e1 (43.34 dB, 72.37 Mpix/s) | ✓ |
| bc7e u2 | 45.40 | 2.37 | mipple e5 (45.46 dB, 4.02 Mpix/s) | ✓ |
| bc7e u4 | 45.70 | 1.22 | mipple e6 (45.71 dB, 1.36 Mpix/s) | ✓ |
| bc7e u6 | 46.08 | 0.14 | ✗ | best is mipple e9, 0.17 dB short |
| bc7f:fastest | 41.35 | 43.53 | mipple e0 (41.62 dB, 155.22 Mpix/s) | ✓ |
| bc7f:faster | 41.35 | 41.10 | mipple e0 (41.62 dB, 155.22 Mpix/s) | ✓ |
| bc7f:fast | 42.99 | 30.60 | mipple e1 (43.34 dB, 72.37 Mpix/s) | ✓ |
| bc7f:default | 43.88 | 18.07 | mipple e2 (44.05 dB, 28.12 Mpix/s) | ✓ |
| bc7f:partial | 44.22 | 10.88 | mipple e3 (44.86 dB, 13.52 Mpix/s) | ✓ |
| ispc_bc7:ultrafast | 41.78 | 42.13 | mipple e1 (43.34 dB, 72.37 Mpix/s) | ✓ |
| ispc_bc7:veryfast | 45.22 | 2.93 | mipple e4 (45.39 dB, 4.68 Mpix/s) | ✓ |
| ispc_bc7:fast | 45.38 | 1.72 | mipple e4 (45.39 dB, 4.68 Mpix/s) | ✓ |
| ispc_bc7:basic | 45.58 | 0.70 | mipple e6 (45.71 dB, 1.36 Mpix/s) | ✓ |
| ispc_bc7:slow | 45.69 | 0.19 | mipple e6 (45.71 dB, 1.36 Mpix/s) | ✓ |

What the pieces are worth (Kodak 1-6, mean PSNR, one thread, in-process):

| Search | PSNR (dB) | Time (ms) |
| --- | --- | --- |
| mode 6, exact indices (effort 0) | 41.62 | 12 |
| + mode 1 on the popcount partition, blocks with error > 512 (effort 1) | 43.34 | 28 |
| + mode 1 on the best line-fit-ranked partition (effort 3) | 44.86 | 160 |
| mode 1 on 4 ranked partitions, refit, refinement (effort 5) | 45.46 | 563 |
| same without refinement | 45.30 | 331 |
| + mode 3 | 45.52 | 842 |
| + modes 0 and 2 on 1 ranked 3-subset partition each | 45.66 | 1087 |
| + mode 5 (best of three rotations), 2 partitions per 3-subset mode, 2 refinement passes (effort 6) | 45.71 | 1380 |
| line-fit ranking with the batch encoder's `fits8` instead of the one-step λ estimate | same | 2× ranking time |
| 2-subset search only for blocks whose mode-6 error > 64 | −0.08 to −0.14 | |

### BC1

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |
| --- | --- | --- | --- | --- | --- |
| mipple e0 | 35.80 | 2.978 | 3.102 | 0.02 | 128.22 |
| mipple e1 | 35.88 | 2.974 | 3.099 | 0.04 | 66.84 |
| mipple e2 | 36.44 | 3.005 | 3.121 | 0.24 | 9.98 |
| mipple e3 | 36.54 | 2.996 | 3.114 | 0.47 | 5.07 |
| mipple e4 | 36.60 | 3.004 | 3.117 | 0.63 | 3.76 |
| mipple e5 | 36.62 | 3.001 | 3.113 | 1.00 | 2.35 |
| mipple e6 | 36.67 | 2.991 | 3.105 | 2.69 | 0.88 |
| mipple e7 | 36.67 | 2.987 | 3.101 | 3.07 | 0.77 |
| mipple e8 | 36.67 | 2.986 | 3.100 | 3.48 | 0.68 |
| rgbcx L0 | 35.73 | 2.980 | 3.105 | 0.08 | 28.09 |
| rgbcx L6 | 36.36 | 3.027 | 3.133 | 0.46 | 5.12 |
| rgbcx L12 | 36.60 | 2.996 | 3.108 | 3.91 | 0.60 |
| rgbcx L18 | 36.67 | 2.983 | 3.096 | 12.09 | 0.20 |
| icbc:1 | 35.78 | 2.986 | 3.112 | 0.20 | 12.04 |
| icbc:5 | 36.50 | 2.921 | 3.014 | 0.62 | 3.82 |
| icbc:8 | 36.60 | 2.987 | 3.103 | 1.53 | 1.54 |
| icbc:9 | 36.64 | 2.997 | 3.111 | 3.17 | 0.74 |
| stb_bc1 | 35.55 | 2.999 | 3.124 | 0.17 | 13.60 |
| stb_bc1hq | 35.76 | 2.985 | 3.112 | 0.21 | 11.06 |
| ispc_bc1 | 35.46 | 2.989 | 3.113 | 0.02 | 125.49 |

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx L0 | 35.73 | 28.09 | mipple e0 (35.80 dB, 128.22 Mpix/s) | ✓ |
| rgbcx L6 | 36.36 | 5.12 | mipple e2 (36.44 dB, 9.98 Mpix/s) | ✓ |
| rgbcx L12 | 36.60 | 0.60 | mipple e4 (36.60 dB, 3.76 Mpix/s) | ✓ |
| rgbcx L18 | 36.67 | 0.20 | mipple e6 (36.67 dB, 0.88 Mpix/s) | ✓ |
| icbc:1 | 35.78 | 12.04 | mipple e0 (35.80 dB, 128.22 Mpix/s) | ✓ |
| icbc:5 | 36.50 | 3.82 | mipple e3 (36.54 dB, 5.07 Mpix/s) | ✓ |
| icbc:8 | 36.60 | 1.54 | mipple e5 (36.62 dB, 2.35 Mpix/s) | ✓ |
| icbc:9 | 36.64 | 0.74 | mipple e6 (36.67 dB, 0.88 Mpix/s) | ✓ |
| stb_bc1 | 35.55 | 13.60 | mipple e0 (35.80 dB, 128.22 Mpix/s) | ✓ |
| stb_bc1hq | 35.76 | 11.06 | mipple e0 (35.80 dB, 128.22 Mpix/s) | ✓ |
| ispc_bc1 | 35.46 | 125.49 | mipple e0 (35.80 dB, 128.22 Mpix/s) | ✓ |

Every effort now runs in the SIMD batch encoder (`src/bc1/batch.rs`). What each step is worth on
kodim01-03 (pooled PSNR):

| Search | PSNR (dB) |
| --- | --- |
| line fit + 2 refits (effort 1) | 36.12 |
| 4-color cluster fit, every split, snapped (old effort 4) | 36.54 |
| + 3-color fit with dark texels on black | 36.86 |
| same, splits scored by closed-form error only | 36.74 |
| line fit + 3-color fit only | 36.51 |

The 3-color fit is worth +0.3 dB on photos. Scoring splits with the closed-form least-squares error is 5×
cheaper but loses 0.12 dB, so the shipped scorer uses it as a bound and snaps only splits that can still win
(`prune` adds a share of the expected snapping penalty to the bound: 0.8 costs 0.02 dB and halves the time).
On kodim01 mipple's top effort beats rgbcx L18 on high-variance blocks and loses slightly on smooth ones.

## Base encoders vs competitors, all 24 Kodak images, one P-core

`bench-results/compete-kodak24-1t-p12/`, with `--threads 1 --base-only --affinity 0x1000 --repeat 3 --interleave`.
Every one of the 33 competitor configs is beaten on both PSNR and speed by some mipple effort. The last one to fall
was ISPC BC1 (`ispc_bc1`): mipple effort 0 is 0.34 dB better (36.05 vs 35.71) and now slightly faster (467 vs
450 Mpix/s; it was 3% slower before the change below). BC7: effort 8 beats bc7e u6 (46.02 dB in 33.4 s vs
46.01 dB in 38.7 s), and effort 6 beats ISPC `slow` (45.64 vs 45.63 dB, 6.6× faster).

The BC1 effort-0 speedup came from dependency chains, not arithmetic: the line fit summed 16 texels per
accumulator in one serial chain (mean, covariance, projection min/max, refit sums), about 64 cycles of latency
each, which out-of-order execution could not hide across ~1000-instruction batches. Four interleaved chains
per sum (`reduce16!` in `src/bc1/batch.rs`) give +11% in-process (444 → 490 Mpix/s), and a dedicated effort-0 instantiation with the configuration as a constant another ~4%. The covariance and
refit sums are now taken over raw integer products, which are exact in f32 in any order, so the output is
unchanged apart from rounding (BC1 BD-rate ±0.00% in `mipple check`). The helpers are macros, not closures:
a closure does not inherit the caller's AVX2 target feature, and one that was not inlined made effort 1 15×
slower.

Two fixes went into this run:

- `--interleave` keyed the fastest time per config by label alone, so configs with the same label in different
  formats (e.g. `mipple e0` in BC1 and BC7) shared one minimum. The key is now (format, label).
- BC1 built its single-color match tables on first use, which cost about 2 ms per process inside the timed
  encode (~10% of an image at effort 0). They are now `const` tables built at compile time.

## vs Oodle Texture 2.9.16 (Unreal Engine 5.8)

Oodle Texture is closed source, so `scripts/ue_oodle/run_ue_oodle.py`
drives a stock UE 5.8.4 install as a user would (no engine source, no direct DLL calls): it imports each PNG
once per format and Lossy Compression Amount (None, Lowest, Low, Medium, High, Highest = Oodle λ 0, 1, 10, 20,
30, 40), cooks with Final encode speed and RDO on, and takes Oodle's blocks from the cook's
`-OodleDebugDumpFilter` dumps, checked bit-exact against the cooked package and against the PNG Oodle was
given. The DDC key confirms SDK 2.9.16. `scripts/ue_oodle/compare.py` then scores those blocks and a mipple
-e 5 λ sweep on the same pixels, sizes Zstd-19 over the block data for both, and writes
`bench-results/oodle-ue/` (per-image rows in `results.json`). BC1 and BC7 inputs are the PNGs unaltered; BC4
is the R channel; BC5 goes through UE's normal-map import (renormalized XY as R16G16), so that R16G16 input is
the reference and mipple encodes it rounded to 8 bits.

mipple's size at Oodle's quality, relative to Oodle's size (Kodak-24 totals; negative = mipple smaller):

| Oodle λ | BC1 at RMSE | BC1 at SSIM | BC7 at RMSE | BC7 at SSIM | BC4 at RMSE | BC4 at SSIM | BC5 at RMSE | BC5 at SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | −16.2% | −8.8% | −6.0% (15/24 images) | −2.7% (4/24) | −9.6% | −4.6% | −26.7% | −10.0% |
| 1 | −12.4% | −6.1% | −2.6% (12/24) | −1.0% (4/24) | +0.2% (20/24) | −0.7% | +4.0% (22/24) | +0.1% |
| 10 | −7.6% | −6.9% | +2.3% | −4.8% | −2.2% | −0.9% | −6.8% | −2.0% |
| 20 | −6.5% | −6.6% | +4.0% | −2.9% | −4.7% | −1.6% | −12.4% | −3.7% |
| 30 | −6.3% | −6.1% | +3.3% | −1.9% | −7.6% | −2.6% | −15.8% | −4.3% |
| 40 | −6.4% | −5.7% | +2.7% | −1.3% | −9.9% | −3.2% | −18.3% | −4.6% |

BC1 is ahead on every image at every λ on SSIM except kodim05 at λ 40 (+0.5%). BC7 is the open gap: Oodle
2.9.16 holds 2-4% fewer bits at equal RMSE (mipple is 1-5% ahead at equal SSIM). Its non-RDO BC7 is more
accurate than mipple's -e 5 base, so at λ 0-1 mipple's curve only reaches Oodle's RMSE on some images (counts
in brackets); at -e 8 or 9 without RDO mipple is as accurate as Oodle's λ 0 on 5 of those 6 images and
smaller on all 6 (kodim03: RMSE 1.650 at 6.85 bpt vs Oodle's 1.623 at 7.30), so the gap is in the RDO search,
not the base encoder. It was 7-9% (+8.2%,
+8.8%, +8.3%, +6.9%) before four RDO changes, which the first 6 images (kodim03, 05, 08, 13, 18, 24) sized:

- the plain search runs at effort 8 or more whenever RDO is on (`RDO_BASE_EFFORT` in `src/pipeline.rs`; base
  time 11 ms → 241 ms per Kodak image, next to 1.5-3 s of RDO). At λ below 1 (no entropy reduction) that is
  the base itself, and Oodle's λ 10 quality sits there: +11.0% → +2.0% at λ 10. Most of the gain.
- with an entropy-reduced base (λ ≥ 1), that plain encoding is a second base: each block starts from
  whichever of the two scores better (+9.4% → +8.5% at -e 5 before the effort floor).
- a cross-mode candidate: an earlier block (within 8, or the one above) whose mode or partition differs is
  tried with this block's endpoints refit to its layout, so mode, partition and index bits match exactly
  (+8.5% → +8.3%; every earlier block gives +8.1% at a quarter more RDO time at λ 1).
- a mode-6-only encode (`bc7::mode6_cfg`, 2 refits) as a further base, since Oodle's RDO output is 70-80% mode
  6: on the 6 images +3.9% → +2.8% at RMSE and −2.4% → −3.5% at SSIM (with the learned ranker); Kodak-24
  BD-rate −0.22% vs the ranker alone at 1.06× the time. 128 KiB chunks would add another −1.0% to −1.3% BD-rate
  at the same CPU time (6-image Oodle subset: +2.1% / −4.5%), but fewer, longer chunks leave threads idle: 1.4-1.5×
  the wall time at 20 threads, so BC7 chunks stay 64 KiB.

On those 6 images, mean of λ 10-40 at RMSE / SSIM: -e 5 +9.4% / −2.7% before, +4.0% / −2.2% after; -e 9 is
the same as -e 5 now. The smooth-block weight moves error from RMSE to SSIM but does not close the gap:
`--smooth-scale 1` gives +2.1% / +7.5%, `--smooth-scale 2.5` +2.0% / +1.6%, so the default (5) stays.

BC4 and BC5 on Kodak-24 are ahead of Oodle at both metrics at λ 0 and λ 10-40, and close at λ 1, where Oodle
comes out both more accurate and smaller than its own λ 0 (BC5: RMSE 2.687 at 5.69 bpt vs 2.756 at 7.43).
The crop of kodim18 alone is behind at equal SSIM for BC4 and BC5:
kodim18 (512×512 crop), at RMSE / at SSIM: BC4 −5.2% / −1.5% at λ 0, then +2.8% / +4.8%, +1.2% / +6.2%,
+0.2% / +6.7%, −0.6% / +6.6% at λ 10-40; BC5 −28.9% / −8.7% at λ 0, +5.6% / +3.8% at λ 1, then −7.3% / +2.3%,
−12.2% / +1.0%, −14.8% / +0.4%, −17.2% / +0.7% at λ 10-40. BC1 kodim18: −7.3% at λ 0, −0.8% to −1.8% at λ 10-40 on
RMSE, −2.6% to −4.8% on SSIM.

## Lossless field split (`--prep`)

`src/prep.rs` applies a BC7Prep-style lossless transform before Zstd: blocks are split into byte streams of
like fields (BC1: endpoint low/high bytes and selectors; BC4/BC5: endpoints and indices; BC7: mode, shape,
endpoints with p-bits folded in, shared p-bits, bit-packed indices and an escape stream for invalid blocks).
`mipple encode --zstd N --prep on|auto` stores it in KTX2 with a `mipple.prep` key/value entry (one flag per
level) and `ktx2::read` undoes it; `auto` keeps whichever layout compresses smaller per level. Loaders that
do not know the key must not read prepped files. Zstd-19 size change on Kodak 1-6 (`cargo test --release
prep::tests::sizes -- --ignored --nocapture`):

| Format | λ 0 | λ 1 | λ 4 |
| --- | --- | --- | --- |
| BC1 | −4.95% | −3.18% | +2.12% |
| BC4 | −3.88% | −1.08% | +3.23% |
| BC5 | −5.03% | −1.75% | +2.95% |
| BC7 | −7.66% | −5.21% | −1.09% |

At high λ, RDO has optimized whole-block repeats in the interleaved layout and splitting them loses; an RDO
rate model for the split layout is on the roadmap.

## v0.9: entropy-reduced BC7 base encoding

v0.9 ports bc7enc's `-e` ("reduced entropy") into the BC7 base encoder, behind `--entropy off|on|auto`
(default **auto** = on for λ ≥ 1). It was measured with `mipple check` only; the full Kodak head-to-head
below is still the v0.2 run and was **not** repeated.

What it does, per block:

- **Mode-6 preference:** another mode must have less than 0.4× mode 6's error (0.6× with alpha) to win.
  Mode 6 is one subset, 4-bit indices and a fixed layout, so neighbouring blocks share more bytes.
- **Mode-6 endpoint quantization:** each 7-bit endpoint is limited to 64 evenly spaced levels with fixed
  p-bits ((1, 1) for opaque blocks so alpha stays exactly 255; (0, 1) with alpha, as bc7enc does).
- **RDO keeps it:** when RDO refits endpoints onto a neighbour's indices, it uses the same restricted
  quantization (worth a further −0.4% BD-rate on Kodak).

Which parts help, as BD-rate vs plain encoding (Zstd-19, λ 0–4, `--entropy on`; negative = smaller at equal PSNR):

| Variant | Kodak first 4 | Synthetic 256² |
| --- | --- | --- |
| bc7enc `-e` as published (weights 0.4/0.6, mode-6 quant, p-bit-1 weight 1.3) | −3.35% | −3.09% |
| p-bit-1 weight 1.3 only | +0.53% | −0.19% |
| mode-6 quantization only | +5.00% | +1.30% |
| mode weights 0.4/0.6 only | −3.25% | −1.67% |
| weights 0.6/0.8 + quant + p-bit | −1.08% | −0.24% |
| weights 0.8/0.9 + quant + p-bit | +2.19% | +0.76% |
| weights 0.25/0.4 + quant + p-bit | −3.34% | −4.44% |
| **weights 0.4/0.6 + quant, RDO keeps the quantization (shipped)** | **−3.70%** | **−4.88%** |

The mode weights do most of the work. The quantization only pays when the mode weights put enough blocks in
mode 6, and when RDO keeps it. The p-bit weighting made no difference once mode 6 was quantized, so it is not
included.

**When it pays.** Dense λ sweep (0, 0.25, 0.5, 1, 2, 4, 8), pooled PSNR vs Zstd-19 bytes:

| λ | Kodak-4 off | Kodak-4 on | Synthetic off | Synthetic on |
| --- | --- | --- | --- | --- |
| 0 | 45.78 dB, 1,418,246 B | 44.12 dB, 1,237,481 B | 45.13 dB, 266,899 B | 43.23 dB, 243,328 B |
| 0.5 | 44.74 dB, 1,235,712 B | 43.50 dB, 1,099,729 B | 44.65 dB, 228,517 B | 42.95 dB, 214,088 B |
| 1 | 43.36 dB, 1,086,535 B | 42.39 dB, 966,324 B | 44.16 dB, 221,299 B | 42.51 dB, 206,230 B |
| 2 | 41.61 dB, 943,867 B | 40.75 dB, 821,604 B | 42.87 dB, 209,834 B | 41.37 dB, 190,525 B |
| 4 | 40.21 dB, 859,850 B | 39.09 dB, 714,449 B | 40.63 dB, 193,037 B | 39.21 dB, 164,876 B |
| 8 | 39.33 dB, 830,667 B | 37.95 dB, 666,483 B | 38.15 dB, 167,913 B | 37.17 dB, 139,298 B |

Over the whole sweep, entropy reduction is −5.1% (Kodak-4) and −5.3% (synthetic) BD-rate. The curves
cross, though. Above about 43.5 dB (Kodak) or 42.5 dB (synthetic), plain encoding plus light RDO reaches the
same PSNR in fewer bytes. For example, plain λ = 0.5 gives 44.74 dB in 1.236 MB, while reduced λ = 0 gives
44.12 dB in 1.237 MB. Below the crossover, entropy reduction wins by 4–15%. Hence `auto` switches it on
from λ = 1:

- λ < 1 output is byte-identical to v0.2.
- On the check corpora, `auto` is −3.2% (Kodak-4) and −3.3% (synthetic) BD-rate vs v0.2, with λ 0–4.
  BD-rate understates it a little, because the cubic fit has to bend through the switch point.

**λ means something different from v0.2 at λ ≥ 1.** On Kodak-4, λ = 1 is now 42.4 dB (it was 43.4 dB) at
11% fewer bytes. Use `--entropy off` for v0.2 behaviour.

## Regression check (`mipple check`)

`mipple check` is the fast guard to run instead of re-running this page. It encodes the six synthetic images at
256² in all four formats at λ 0, 0.5, 1, 2 and 4 (about 27 s on the 2-core dev VM; Kodak first 4 for BC7: about
36 s). It compares the result with `bench-results/check-baseline.csv` (or `check-baseline-kodak.csv`) and
fails on:

- BD-rate > +1%;
- λ = 0 pooled PSNR −0.1 dB, or λ = 0 size +1%;
- any single image −0.5 dB at λ = 0.

How the check judged this release's changes:

| Change | Check result |
| --- | --- |
| `BlockOpts` refactor (effort + flag threaded through every encoder) | PASS, 120/120 encodes byte-identical |
| `--entropy on` | FAIL: λ = 0 PSNR −1.9 dB (gate working as intended: base quality traded for size), BD-rate −3.1% |
| `--entropy auto` (shipped default) | PASS: BC1/BC4/BC5 identical, BC7 BD-rate −3.28% (synthetic) / −3.23% (Kodak-4), λ < 1 identical |

Both baselines were then regenerated with `--update-baseline`.

Later changes:

| Change | Check result |
| --- | --- |
| BC1 batch search for every effort (default effort 5 moves to it) | PASS: BC1 λ = 0 +0.10 dB, BD-rate −0.55%; baseline updated |
| BC7 fast encoder for efforts 0-6 (default effort 5 moves to it) | FAIL on one gate: BC7 λ = 0 Zstd size +1.86% at equal PSNR (45.93 dB), while BD-rate is −0.50%, λ = 0.5 is 2.8% smaller and the check runs 2.3× faster. Accepted (the new default picks mode 1 more often); baseline updated |
| RDO wide search by default (256 back-references, spans from the nearest 8) with the faster candidate loop | PASS: BD-rate BC1 −3.21%, BC4 −5.25%, BC5 −2.83%, BC7 −3.70% (synthetic; Kodak-4: −9.25%, −5.50%, −4.21%, −4.74%), λ = 0 identical, total time 0.82× (Kodak-4: 0.89×); both baselines updated |
| Field-swap refinement (`--refine-back`, 256 for BC1/BC4, off for BC5/BC7) | PASS: BD-rate BC1 −0.56%, BC4 −0.25% (synthetic; Kodak-24: −0.98%, −0.39%), BC5/BC7 identical; both baselines updated (they predated `--min-rms` and the BC7 window change) |
| Literal costs frozen per chunk from the base encoding (BC4/BC5/BC7) | PASS: BD-rate BC4 −0.87%, BC5 +0.15%, BC7 +0.13%, BC1 identical (synthetic; Kodak-24: BC4 −1.24%, BC5 −0.44%; Kodak-4 BC7 −0.06%); both baselines updated |
| Far endpoints for BC1 (`--far-back` 1024, BC1 window 8 KiB) | PASS: BD-rate BC1 −0.72%, BC4/BC5/BC7 identical, total time 0.94× (synthetic; Kodak-24: BC1 −0.55%, 1.02× total for BC1/BC4/BC5); synthetic baseline updated |
| Exact candidate errors, half memo, deferred re-index, bounded LZ cost | PASS: all identical (Kodak-24 BC1/BC4/BC5 360/360), Kodak-24 time BC5 0.80×, BC4 0.94× |
| BC5 window 3072 (192-deep search) | PASS: BD-rate BC5 −0.63%, others identical, total time 1.15× (synthetic; Kodak-24: BC5 −2.81%); synthetic baseline updated |
| BC7 re-index bound before palettes, no index unpack, cached subset means | PASS: all identical, Kodak-8 BC7 time 0.94× |
| BC7 far search (`--far-back` 1024, BC7 window 16 KiB) | PASS: BD-rate BC7 −1.62%, others identical, total time 1.33× (synthetic; Kodak-8: BC7 −1.46%, 1.18× vs before the bound changes); synthetic baseline updated |
| Localized BC5 search (`--local` 16: nearest 128 + 16 network-ranked + row above) | PASS: BD-rate BC5 −1.05%, others identical, total time 1.08× (synthetic; Kodak-24: BC5 −0.76%, 1.02×); synthetic baseline updated |

## Head-to-head: mipple vs bc7enc_rdo (Kodak 24, BC7, v0.2)

`scripts/head2head.py` runs each encoder to DDS and grades everything with `mipple score`, so metrics and back
ends are identical. bc7enc_rdo (commit b943862, 2026-07-30) was built from source **without ISPC**, so its base
encoder is `bc7enc.cpp` at `-u4` (modes 1/5/6/7), not the stronger `bc7e.ispc`; its RDO window was set to 1024
bytes (`-zc1024`). bc7f and bc7g are not in this run (see "Not yet measured").

**BD-rate vs bc7enc_rdo** (average size difference at equal PSNR; negative = smaller):

| Family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| **mipple** (effort 1, λ 0–8) | **−13.4%** | **−12.6%** |
| bc7enc_rdo `-C -e` (reduced-entropy base + RDO) | −7.9% | −6.7% |

| Encoder config | Mean PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Encode time, 24 images (s) |
| --- | --- | --- | --- | --- |
| mipple λ=0 | 45.03 | 7.266 | 7.358 | 9.4 |
| mipple λ=0.5 | 44.25 | 6.454 | 6.533 | 54.6 |
| mipple λ=1 | 43.00 | 5.703 | 5.792 | 56.8 |
| mipple λ=2 | 41.26 | 4.919 | 5.032 | 58.0 |
| mipple λ=4 | 39.91 | 4.506 | 4.638 | 59.2 |
| bc7enc_rdo λ=0 | 45.33 | 7.194 | 7.296 | 7.8 |
| bc7enc_rdo λ=0.1 | 44.75 | 6.902 | 6.989 | 185.1 |
| bc7enc_rdo λ=0.25 | 43.30 | 6.429 | 6.503 | 179.8 |
| bc7enc_rdo λ=0.5 | 40.94 | 5.890 | 5.953 | 177.1 |
| bc7enc_rdo λ=1 | 37.59 | 5.253 | 5.307 | 174.4 |
| bc7enc_rdo `-C -e` λ=0 | 43.83 | 6.408 | 6.579 | 8.2 |
| bc7enc_rdo `-C -e` λ=0.25 | 41.98 | 5.709 | 5.840 | 168.2 |

Full table (19 configs): `bench-results/h2h-kodak/summary.md`; per-image rows: `bench-results/h2h-kodak/rows.csv`.

What it says:

- **RDO:** at matched quality mipple ships fewer bytes: 43.0 dB at 5.70 bpt vs bc7enc_rdo's 43.3 dB at 6.43 bpt.
  mipple's RDO pass is also about 3× faster here.
- **Base encoder:** bc7enc's base is slightly better and faster at λ=0 (45.33 dB at 1.2 Mpix/s vs mipple's
  45.03 dB at 1.0 Mpix/s). With bc7e it would be stronger still.
- **λ scales differ:** bc7enc_rdo's λ = 1 is far more aggressive than mipple's; compare curves, not λ values.
- **Reduced-entropy encoding is a cheap win:** bc7enc_rdo's `-e` (no RDO time cost) cuts size 11% at λ=0 and
  improves its curve by 7.9%. mipple v0.9 adds it (see above). The gain is about 5% on mipple's curve, because
  mipple's RDO already captured part of it. Re-running this head-to-head with `--entropy auto` is still to do.

### bc7f-style effort 1 (this release)

Effort 1 now pre-selects 2-subset partitions bc7f's way (principal-axis split, popcount match against the 64
patterns) before line-fit ranking, and sends very low-variance blocks straight to mode 6. Measured on 8 Kodak
images at λ=0:

| Variant | PSNR (dB) | Zstd bpt | Base time (s) |
| --- | --- | --- | --- |
| Previous effort 1 (line-fit rank all 64, keep 4) | 45.04 | 7.389 | 6.59 |
| One-shot guess, keep 4 (pure bc7f style) | 44.38 | 7.297 | 2.62 |
| Popcount top 8 → line-fit → keep 4 | 44.62 | 7.301 | 2.69 |
| **Popcount top 16 → line-fit → keep 4 (shipped)** | **44.86** | **7.298** | **3.26** |
| Popcount top 24 → line-fit → keep 4 | 44.91 | 7.295 | 3.79 |

The shipped variant is 2.1× faster for −0.18 dB, and its blocks compress slightly better, which moved the
head-to-head BD-rate from −11.5% (previous effort 1) to −13.4%.

### Not yet measured

- **bc7e.ispc** (bc7enc_rdo's best base): needs ISPC; downloading its prebuilt compiler was blocked in the dev sandbox.
- **bc7f**: `tools/bc7f/` has the harness and build script; compiling Basis Universal was blocked in the dev sandbox.
- **bc7g**: announced, unreleased. `scripts/head2head.py --bc7g` is ready.

## Kodak 24 (photos, 768×512), λ sweep (v0.2, entropy reduction off)

| Format | λ | Mean PSNR (dB) | Mean SSIM | Zstd bpt | Δ Zstd vs λ0 | Deflate bpt | Δ Deflate vs λ0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BC1 | 0 | 36.16 | 0.9812 | 2.992 | — | 3.115 | — |
| BC1 | 1 | 36.20 | 0.9800 | 2.813 | −6.0% | 2.940 | −5.6% |
| BC1 | 2 | 36.07 | 0.9779 | 2.650 | −11.4% | 2.765 | −11.3% |
| BC1 | 4 | 35.64 | 0.9712 | 2.340 | −21.8% | 2.434 | −21.9% |
| BC7 | 0 | 45.03 | 0.9973 | 7.266 | — | 7.358 | — |
| BC7 | 0.5 | 44.25 | 0.9960 | 6.454 | −11.2% | 6.533 | −11.2% |
| BC7 | 1 | 43.00 | 0.9927 | 5.703 | −21.5% | 5.792 | −21.3% |
| BC7 | 2 | 41.26 | 0.9862 | 4.919 | −32.3% | 5.032 | −31.6% |
| BC7 | 4 | 39.91 | 0.9807 | 4.506 | −38.0% | 4.638 | −37.0% |

A zoomed visual check of kodim23 shows no block artifacts at λ = 4, only slight softening of fine texture.

## Synthetic corpus (512×512, `mipple synth`, v0.2)

| Format | λ | Mean PSNR (dB) | Normal err (°) | Zstd bpt | Δ Zstd vs λ0 |
| --- | --- | --- | --- | --- | --- |
| BC1 | 0 | 42.69 | 2.86 | 1.962 | — |
| BC1 | 1 | 42.98 | 2.86 | 1.822 | −7.2% |
| BC1 | 4 | 41.59 | 2.91 | 1.526 | −22.2% |
| BC4 | 0 | 66.03 | — | 2.872 | — |
| BC4 | 1 | 54.92 | — | 1.934 | −32.7% |
| BC5 | 0 | 66.03 | 0.60 | 5.117 | — |
| BC5 | 1 | 54.57 | 0.69 | 3.258 | −36.3% |
| BC7 | 0 | 51.53 | 1.52 | 4.973 | — |
| BC7 | 1 | 47.15 | 1.52 | 3.887 | −21.8% |
| BC7 | 4 | 43.96 | 1.98 | 3.129 | −37.1% |

Mean PSNR counts losslessly encoded images as 99 dB; see `bench-results/synthetic.csv` for per-image values.

## How to reproduce

```sh
cargo build --release
./target/release/mipple check                                         # regression check, ~30 s
./target/release/mipple check --corpus test-images/kodak --limit 4 -f bc7
./target/release/mipple check -f bc7 --entropy off                    # v0.9 vs v0.2 behaviour, same baseline
scripts/fetch-kodak.sh
./target/release/mipple bench test-images/kodak -f bc1,bc7 -l 0,0.5,1,2,4
python3 scripts/head2head.py --corpus test-images/kodak --out bench-results/h2h-kodak \
    --mipple target/release/mipple --bc7enc /path/to/bc7enc_rdo/build/bc7enc --bc7f tools/bc7f/bc7f_encode
```

Building bc7enc_rdo with recent GCC needs `-DCMAKE_CXX_FLAGS="-include cstdint"`; add `-DSUPPORT_BC7E=TRUE`
with ISPC on your PATH for its full-strength base encoder.
