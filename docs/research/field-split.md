# Field split ("prep"): where it stands and what to try next

mipple's lossless field-split transform plays the same role as Oodle's BC7Prep. This note covers what
the transform does today, why it stops paying at higher λ, and what we might build next.

## The idea

Every BCn block mixes two kinds of data:

- **Endpoints:** a few colours per block. They are smooth, correlated with neighbouring blocks, and often
  repeat.
- **Indices:** 2-4 bits per texel placing each texel between the endpoints. They look close to noise.

Stored block after block, the two alternate every 8 or 16 bytes, so an LZ compressor keeps switching
between two very different kinds of statistics. Pulling each kind of field into its own stream gives the
compressor long runs of similar data. A loader then re-interleaves the streams into ordinary blocks after
decompression and before upload. The GPU still receives standard BCn blocks; the transform only changes
what sits on disk and goes through the decompressor.

Oodle Texture ships this for BC7 as **BC7Prep**, a separate, optional step from its RDO encoder. RAD's
public material describes it as a lossless rearrangement of the bits in BC7 blocks:

- RAD says files come out 5-15% smaller after the general-purpose compressor.
- It needs a runtime untransform. Charles Bloom calls it "extremely fast" and suggests running it in
  software right after decompression, or on the GPU.
- Bloom's published table (see [oodle-rdo-ktx2.md](../reference/oodle-rdo-ktx2.md)) shows BC7Prep still
  adding about 15% on top of RDO at λ 5.

RAD has not published BC7Prep's layout, and we don't look at Oodle code, headers or binaries. Everything
below is our own design.

Sources: [Oodle Texture](https://radgametools.com/oodletexture.htm),
[oodle texture bc7prep data flow (cbloom rants)](https://cbloomrants.blogspot.com/2020/06/oodle-texture-bc7prep-data-flow.html).

## What mipple has today

[`src/prep.rs`](../../src/prep.rs) implements forward and inverse transforms for every format:

| Format | Streams |
| --- | --- |
| BC1 | colour-0 low/high byte planes, colour-1 low/high byte planes, selectors |
| BC4 / BC5 | per channel: endpoint 0, endpoint 1, 6 index bytes |
| BC7 | mode byte; shape byte (partition, or rotation + index selector); endpoint channels one per byte (p-bit folded in as the low bit); shared p-bits (mode 1); index bits packed per block and padded to whole bytes; escape stream for invalid blocks |

- **Using it:** pass `mipple encode --zstd N --prep on|auto`. `auto` keeps whichever layout compresses
  smaller, per mip level.
- **Container:** the KTX2 file gets a `mipple.prep` key/value entry with one flag per level, and
  `ktx2::read` undoes the transform.
- **Default:** off. A prepped file is not standard KTX2, and a loader that ignores the key reads garbage.

Zstd-19 size change on Kodak 1-6, from [BENCHMARKS.md](../BENCHMARKS.md#lossless-field-split---prep):

| Format | λ 0 | λ 1 | λ 4 |
| --- | --- | --- | --- |
| BC1 | −4.95% | −3.18% | +2.12% |
| BC4 | −3.88% | −1.08% | +3.23% |
| BC5 | −5.03% | −1.75% | +2.95% |
| BC7 | −7.66% | −5.21% | −1.09% |

**Why it fades and then loses at higher λ.** RDO picks blocks by asking what Zstd would pay for them in the
*interleaved* layout. Its cheapest move is copying a whole recent block, which is a single 16-byte match.
Splitting turns that one match into one short match per stream, each at a different byte offset. The
transform undoes the very thing RDO optimized for. The two are tuned for different layouts and fight.

## What we might do

Ordered by expected payoff.

### 1. A rate model for the split layout (the main item)

Teach RDO to cost candidates in the layout that will actually be stored.

- **Model:** keep one `LzModel` window per stream. A candidate's rate becomes the sum of its incremental
  cost in each stream.
- **Search:** the candidate generators can stay as they are, since they already think in fields (borrowed
  endpoints, borrowed indices, whole-block copies). Only the rate side changes.
- **What it buys:** RDO learns that borrowing a neighbour's endpoints is cheap even when the indices
  differ, because the endpoint stream sees a run of repeats. In the interleaved layout that same borrow is
  broken up by 8 bytes of index noise.
- **Cost:** about one parse per stream instead of one per block. The rate model is about 5% of RDO time
  today.
- **Measure it:** BD-rate on Kodak-24 against the current interleaved RDO and against Oodle's standard
  blocks. Report Zstd-19 and Deflate-9 sizes.

### 2. Fixed-stride records for mode 6

With `--entropy` on, most BC7 blocks are mode 6.

- **Layout:** if every mode-6 block takes the same number of bytes in each stream (8 endpoint-channel
  bytes, 8 index bytes), a whole-block copy at distance *d* becomes matches at the same record distance in
  every stream.
- **Why that's cheap:** Zstd's repeat-offset codes make a match that reuses a recent offset nearly free.
  Splitting would then stop penalizing whole-block copies, which narrows the λ 4 loss even before item 1.
- **Cost:** cheap to try. It only changes `split_bc7` / `join_bc7`.

### 3. Endpoint prediction

Store each endpoint as a residual from its left or upper neighbour, the way PNG filters rows.

- **For:** it should help λ 0 and other non-RDO output.
- **Against:** it fights LZ, because residuals of copied blocks are no longer copies. It also makes the
  inverse serial along rows, so a GPU inverse needs a scan.
- **Priority:** low. Try it only if items 1-2 leave λ 0-1 gains on the table.

### 4. Index bit planes

Indices look like noise, but their high bits follow edges in the image.

- **Idea:** splitting the high-bit plane from the rest may give the compressor something to find.
- **Cost:** an afternoon to measure. Drop it if the gain is under 1%.

### 5. Loader and container

- **GPU inverse:** a compute shader to rebuild blocks after a hardware or GPU decompressor (DirectStorage,
  console decoders).
  - BC1/BC4/BC5 have fixed strides, so the inverse is one thread per block in one pass.
  - BC7 has variable bytes per block per stream, so it takes two passes:
    1. Read the mode stream and prefix-sum each block's offsets into each stream.
    2. Scatter the fields into blocks.
  - Item 2 would make mode-6-only textures a single pass.
- **Failure mode:**
  - Today, a loader that ignores the `mipple.prep` key silently misreads the file.
  - Moving the flag into a vendor supercompression scheme in the KTX2 header would make unaware loaders
    reject the file instead. The roadmap already wants such a scheme for the latent-space intermediate.

## Policy

- **Default stays standard blocks.** The README's promise is that nothing new runs at load time. Prep
  stays an opt-in for engines that control their loader.
- **Competitor comparisons stay on standard blocks.** The Oodle numbers in BENCHMARKS.md compare plain RDO
  output on both sides. If we publish prepped numbers, they go in a separate table, labelled as needing a
  runtime step.
