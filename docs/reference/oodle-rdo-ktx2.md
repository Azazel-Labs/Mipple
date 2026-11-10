# Oodle, RDO, Texture Optimization & KTX2 — Reference

Exported from the living doc "Oodle, RDO, Texture Optimization & KTX2 — Reference" (as of Nov 6, 2026).
Live version: https://claude.ai/artifact/8Hxqjpm7U5tdv3nS8tsv8J — edits made there are not reflected here.
The two diagrams from the live doc are described in text.

## Overview

Oodle and KTX2/Basis solve the same problem — shipping GPU-compressed textures small — from opposite ends: Oodle makes standard BCn files compress better for PC and consoles, while KTX2 + Basis ships one portable file that transcodes to whatever GPU format a device supports. RDO is the shared trick: choose encodings a lossless compressor shrinks further.

**Diagram — texture pipeline, two paths:**

```text
                 ┌─ Oodle path (PC, consoles) ───────────────────────────────────────────────┐
                 │ Oodle Texture        →  Package          →  Load              →  GPU (VRAM) │
Source art  ─────┤ BCn + RDO (λ),          Kraken / zlib       LZ decompress        BC1–BC7    │
PNG/EXR/TGA/PSD  │ BC7Prep optional        in pak/IOStore      (HW on PS5)          as encoded │
                 │                                                                            │
                 └─ KTX2 + Basis path (web, mobile, cross-platform) ──────────────────────────┤
                   Basis encode         →  KTX2 file        →  Load              →  GPU (VRAM) │
                   ETC1S / UASTC /         BasisLZ or Zstd,    decode, then         BCn, ASTC, │
                   XUASTC, ASTC            small mips first    transcode            ETC2       │
                 ── everything left of GPU sets disk/download size ── format + size set VRAM ──
```

VRAM cost is fixed by the GPU format and resolution; everything before the GPU stage only changes disk and download size.

## Oodle

Oodle is RAD Game Tools' suite of game-focused compression libraries; Epic Games acquired RAD in 2021 and made Oodle free to use inside Unreal Engine from 4.27 and UE5 onward ([Epic](https://www.unrealengine.com/en-US/blog/oodle-now-free-to-use-in-unreal-engine-via-github)). Outside Unreal it is a commercial license from Epic/RAD.

**The four products**

| Product | What it does |
| --- | --- |
| Oodle Data | Lossless general-purpose compressors (Kraken, Mermaid, Selkie, Leviathan) for packages, saves, streaming data |
| Oodle Texture | Encodes source art to BC1–BC7 with RDO so the result compresses far better under any lossless back end |
| Oodle Network | Packet compression for UDP/TCP game traffic, trained on sample traffic |
| Oodle Lossless Image | Lossless image codec aimed at beating PNG on size and decode speed |

**Oodle Data codecs** ([UE docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/oodle-data))

| Codec | Ratio vs speed | Typical use |
| --- | --- | --- |
| Leviathan | Highest ratio, slowest decode of the four | Download size is king; CPU to spare |
| Kraken | High ratio, fast decode — the usual default | Pak/IOStore packages; PS5 hardware decoder |
| Mermaid | Less ratio, faster decode | CPU-limited platforms |
| Selkie | Least ratio, fastest decode | Near-memcpy decode budgets |

Encoder effort levels run from -3 (HyperFast3) through 0 (None, raw copy) up to 9 (Optimal5). Level 6 (Optimal2) is Epic's recommended baseline optimal level; level 3 (Fast) suits daily iteration. Effort changes encode time and ratio only — decode speed stays in the same class.

**Unreal setup.** Packaging uses `PackageCompressionMethod=Kraken` with separate levels per build type (e.g. Distribution 7, Test/Shipping 5, Debug/Development 3). The packaging log confirms it with a line like `Oodle v2.9.0 initializing with method=Kraken`.

**PS5 and consoles.** The PS5 I/O complex has a hardware decompressor for zlib and Kraken, and Sony licensed Oodle Texture for all PS4/PS5 developers ([Bloom](https://cbloomrants.blogspot.com/2020/09/how-oodle-kraken-and-oodle-texture.html)). On one game's texture set RAD measured Zip 1.64:1, Kraken 1.82:1, Zip + Oodle Texture 2.69:1, Kraken + Oodle Texture 3.16:1. Xbox Series uses zlib + BCPack in hardware; Oodle Texture RDO output is plain BCn, so it still helps there.

**Versions.** The 2.9 line is current (UE docs log 2.9.0; 2026 game builds ship 2.9.15–2.9.16). Decoders read their own generation and older, never newer, so ship a decoder at least as new as your encoder. Runtime DLLs are named by generation, e.g. `oo2core_9_win64.dll`.

## Oodle Texture

Oodle Texture is an offline BCn encoder whose RDO mode picks block encodings that a lossless compressor shrinks much further: near-lossless settings save about 10%, and small visible changes save 20–50% ([RAD](https://www.radgametools.com/oodletexture.htm)). Its output is ordinary BCn, so nothing new runs at load time.

**What it covers**

- RDO encoding for BC1, BC3, BC4, BC5, BC6H and BC7 (BC2 excluded — rarely used).
- No ASTC, ETC or other mobile formats ([UE docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/oodle-texture-in-unreal-engine)). Mobile builds need a different encoder.
- Optimizes for size after *any* general-purpose LZ back end (Kraken, zlib, zstd), so it pairs with console hardware decompressors.

**Lambda.** One knob, λ, sets the size–quality trade. λ = 0 turns RDO off (best quality); higher λ is smaller and blurrier. RAD says one λ usually works across a whole data set without per-texture tuning. Unreal defaults to λ 30 (`DefaultRDOLambda=30`) and exposes it as *Lossy Compression Amount* on texture LOD groups and per texture.

**RDO changes disk size, not VRAM.** A BC7 texture is still 8 bits per texel on the GPU; RDO only shrinks the pak/IOStore package and the download.

**BC7Prep.** A lossless bit-rearranging transform for BC7 only, worth 5–15% more after compression ([Bloom](https://cbloomrants.blogspot.com/2020/06/oodle-texture-bc7prep-data-flow.html)). Unlike RDO it needs a runtime untransform, which is very fast and can run as a GPU compute shader or on the CPU right after decompression. It works on BC7 from any encoder and stacks with near-lossless RDO.

**Measured ratios on BC7** (one sample texture, then a varied set; ratio = raw BC7 size ÷ compressed size; [sample run](https://cbloomrants.blogspot.com/2020/06/oodle-texture-sample-run.html))

| Encoding | Back end | Single texture | Varied set |
| --- | --- | --- | --- |
| Baseline BC7 | Kraken | 1.06:1 | 1.39:1 |
| Baseline BC7 + BC7Prep | Kraken | 1.26:1 | 1.53:1 |
| RDO λ 5 (near-lossless) | Kraken | 1.23:1 | — |
| RDO λ 5 + BC7Prep | Kraken | 1.41:1 | 1.65:1 |
| RDO λ 40 (medium) | Kraken | 2.06:1 | — |

RMSE per texel on that texture rose from 2.72 (baseline) to 4.23 at λ 40.

**Unreal specifics.** The plugin is on by default; log lines read `Oodle Texture 2.9.x init RDO On with DefaultRDOLambda=30`, and Oodle-built textures show an `OODLE_` format prefix. A global switch disables RDO entirely, and a debug mode paints each texture a solid color by encoded format. Pin the Oodle Texture SDK version per project so engine upgrades don't silently re-encode every texture and bloat patches.

## RDO explained

Rate-distortion optimization picks, among many valid encodings, the one that minimizes distortion plus λ times bits; for GPU textures the "bits" are measured *after* a later lossless compressor, because the GPU format itself is fixed-rate.

```text
J = D + λ · R
```

D is the error against the source (ideally perceptual, not just RMSE), R is the estimated compressed size, and λ sets the exchange rate. λ = 0 means pure quality; raising λ accepts more error for each byte saved.

**Why it works on BCn.** Every BC7 block is exactly 16 bytes in memory, so you cannot make blocks smaller. You *can* choose endpoints, modes and indices that repeat bytes seen in nearby blocks, or that have lower entropy. An LZ compressor (Kraken, zlib, zstd) then finds long matches. The GPU reads normal blocks; only the package shrinks.

**Where RDO shows up**

| Encoder | Format | What RDO targets | Knob |
| --- | --- | --- | --- |
| Oodle Texture | BC1–BC7 | Size after any LZ back end | λ (UE "Lossy Compression Amount") |
| Basis Universal ETC1S | ETC1S → BasisLZ | Built-in vector-quantized codebooks; inherently rate-controlled | `-q` 1–255, or v2.0 `-quality` 1–100 |
| Basis Universal UASTC LDR | UASTC 4x4 → Zstd | Size after Zstd | `-uastc_rdo_l` |
| Basis UASTC HDR 6x6 | UASTC HDR → ASTC/BC6H | Lambda > 0 enables coarser weight grids | lambda |
| Basis XUASTC LDR (v2.0+) | Supercompressed ASTC | DCT on the ASTC weight grid, entropy-coded | `-quality`, `-effort` |
| bc7enc_rdo (open source) | BC1–BC7 | Size after LZ | λ |

**Practical rules**

- Use lower λ (or near-lossless) for normal maps, UI, text and anything with sharp edges; RDO artifacts show as banding and blockiness there first.
- Diffuse, terrain detail and distant props tolerate high λ well.
- RDO gains depend on the back end and its chunk size: a stronger compressor (Kraken, Leviathan, high-level zstd) extracts more.
- RDO output is not always deterministic: UASTC RDO in KTX-Software varies run to run unless threading is disabled ([KTX-Software v4.3 notes](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.3.2)). Nondeterminism hurts build caching and patch sizes.

## GPU texture formats primer

Desktop and consoles read BCn; mobile reads ASTC (modern) or ETC2 (GLES 3 baseline); the web gets whatever extension the device exposes, which is why transcodable formats like Basis exist. Bits per texel set VRAM and are fixed per format.

| Format | Bits/texel | Channels | Best for | Platforms |
| --- | --- | --- | --- | --- |
| BC1 | 4 | RGB (+1-bit alpha) | Opaque albedo where some banding is OK | Desktop, consoles |
| BC3 | 8 | RGBA | Legacy alpha; BC7 usually better | Desktop, consoles |
| BC4 | 4 | R | Roughness, AO, masks, height | Desktop, consoles |
| BC5 | 8 | RG | Tangent-space normals (Z rebuilt in shader) | Desktop, consoles |
| BC6H | 8 | RGB half-float | HDR skies, lightmaps, env maps | Desktop, consoles |
| BC7 | 8 | RGB / RGBA | High-quality albedo, packed masks, UI | Desktop, consoles |
| ETC2 RGB / RGBA | 4 / 8 | RGB / RGBA | Mobile fallback | GLES 3, Vulkan mobile |
| EAC R11 / RG11 | 4 / 8 | R / RG | Mobile masks / normals | GLES 3, Vulkan mobile |
| ASTC 4x4 → 12x12 | 8 → 0.89 | RGB(A), LDR or HDR | Everything on mobile; size chosen per texture | Modern mobile, Apple GPUs |

ASTC's block size is the quality dial: 4x4 (8 bpp) for normals and UI, 6x6 (3.56 bpp) a common albedo default, 8x8 (2 bpp) for low-detail content. Apple Silicon Macs support both BCn and ASTC.

## KTX2

KTX 2.0 is Khronos' GPU texture container, ratified 14 Aug 2020, that holds any Vulkan-style format plus optional supercompression — most notably Basis Universal, which transcodes at load time to whatever the device supports ([spec](https://github.khronos.org/KTX-Specification/ktxspec.v2.html)).

**File layout**

1. Identifier — 12-byte magic.
2. Header — `vkFormat`, `typeSize`, width/height/depth, `layerCount`, `faceCount`, `levelCount`, `supercompressionScheme`.
3. Index — offsets to the DFD, key/value data and supercompression global data.
4. Level index — byte offset, length and uncompressed length per mip.
5. Data Format Descriptor (DFD) — describes the texel layout and color model (e.g. ETC1S, UASTC) and transfer function (sRGB vs linear).
6. Key/value data — `KTXorientation`, `KTXwriter`, `KTXswizzle` and similar.
7. Supercompression global data — e.g. BasisLZ codebooks.
8. Mip level data — stored smallest mip first, so a streamer can show a low-res version before the rest arrives.

**Supercompression schemes**

| Value | Scheme | Used with |
| --- | --- | --- |
| 0 | None | Raw BCn/ASTC/ETC2, or plain UASTC |
| 1 | BasisLZ | Basis ETC1S only |
| 2 | Zstandard | UASTC, or any native format (BCn, ASTC) |
| 3 | ZLIB | Any format; less common |
| 4 | UASTC HDR 6x6 intermediate | Basis UASTC HDR 6x6i (assigned by Khronos early 2026) |
| 5 | XUASTC LDR | Basis XUASTC (Khronos assignment in progress as of early 2026) |
| 6 | XUBC7 | Basis XUBC7 ([Binomial KTX2 notes](https://github.com/BinomialLLC/basis_universal/wiki/KTX2-File-Format-Support-Technical-Details)) |

Basis-encoded files use `vkFormat = VK_FORMAT_UNDEFINED` and signal the codec through the DFD color model. Native-format KTX2 (e.g. BC7 + Zstd, or RDO-encoded BC7 + Zstd) is equally valid and skips transcoding.

**Basis Universal codecs inside KTX2**

| Codec | Rough quality / size | Transcodes to | Since |
| --- | --- | --- | --- |
| ETC1S (BasisLZ) | Smallest files, visibly lossy; avoid for normals and UI | BC1/3/4/5/7, ETC, ASTC, PVRTC, raw | KTX2 launch |
| UASTC LDR 4x4 (+RDO, Zstd) | Near-BC7 quality, larger files | Same LDR list | KTX2 launch |
| UASTC HDR 4x4 | HDR, 4x4 blocks | ASTC HDR, BC6H, RGB9E5, half-float | v1.50, Sep 2024 |
| ASTC HDR 6x6 / UASTC HDR 6x6 intermediate | Smaller HDR, lambda-based RDO | ASTC HDR 6x6, BC6H, RGB9E5, half | v1.60, Jan 2025 |
| ASTC LDR (all block sizes) | Native ASTC, fast transcode elsewhere | All LDR formats | v2.0, Jan 2026 |
| XUASTC LDR | Supercompressed ASTC with weight-grid DCT; arithmetic, Zstd or hybrid entropy; in-loop deblocking from v2.50 | All LDR formats | v2.0, Jan 2026 |
| XUBC7 | Supercompressed full BC7; lossless by default, lossy with RDO | BC7 directly, near-lossless ASTC 4x4, other LDR formats | v2.50, mid-2026 |

**Recent Basis Universal releases** ([release notes](https://github.com/BinomialLLC/basis_universal/wiki/Release-Notes))

| Date | Version | Change |
| --- | --- | --- |
| Jul–Aug 2026 | [v2.50](https://github.com/BinomialLLC/basis_universal/releases) | XUBC7, in-loop deblocking for XUASTC/ASTC plus an optional deblocking shader, universal .DDS transcoding (Binomial's wiki dates it 1 Jul; the GitHub release is tagged 3 Aug) |
| May 2026 | v2.10 | KTX-Software compatibility changes for HDR and XUASTC LDR KTX2 files |
| 19 Jan 2026 | v2.0 | ASTC LDR, XUASTC LDR, unified `-quality` 1–100 / `-effort` 0–10, fast analytical bc7f/etc1f encoders, pure C API, Python, WASM64, WASI |
| 21 Jan 2025 | v1.60 | UASTC HDR 6x6 + RDO; WASM multithreading; C++17 required |
| 10 Sep 2024 | v1.50 | UASTC HDR 4x4; EXR/HDR/QOI/DDS input; CLI defaults to .KTX2 |
| 12 Apr 2021 | v1.15 | Full KTX2 + Zstd support |

**glTF: `KHR_texture_basisu`.** Allows ETC1S with BasisLZ (scheme 1) or UASTC with no supercompression or Zstd (scheme 0 or 2); images must be 2D ([extension spec](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)).

**Tooling**

- `basisu` CLI — reference encoder; writes .KTX2 by default and Zstd-compresses UASTC/ASTC output unless told not to.
- KTX-Software `ktx` suite — `create`, `encode`, `deflate`, `transcode`, `info`, `validate`, `compare`, `convert`; replaced `toktx`/`ktxsc`/`ktx2check` from v4.3 ([repo](https://github.com/KhronosGroup/KTX-Software)). `ktx create` does no resizing, so pre-size inputs (WebGL size limits matter).
- gltf-transform — `etc1s` / `uastc` commands re-encode a glTF's textures to KTX2.
- Loaders — three.js `KTX2Loader`, Babylon.js, KtxUnity, gdx-basis-universal (libGDX); the transcoder ships as WASM.
- Inspection — Basis WebGL testbed, Kram, PVRTexTool, Mali/Adreno texture tools.

## Texture optimization playbook

Optimize two budgets separately: the GPU format and resolution set VRAM, while RDO and supercompression set disk and download size. Most wasted memory comes from oversized textures and wrong formats, not encoder settings.

**Format by texture role**

| Role | Desktop / console | Mobile | Web (KTX2 + Basis) |
| --- | --- | --- | --- |
| Albedo, opaque | BC7 (or BC1 for bulk) | ASTC 6x6 | ETC1S for bulk, UASTC for hero |
| Albedo + alpha | BC7 | ASTC 4x4–6x6 | UASTC |
| Tangent-space normal | BC5, low or no RDO | ASTC 4x4–5x5 or EAC RG11 | UASTC, low RDO |
| Single-channel mask (roughness, AO, height) | BC4 | ASTC or EAC R11 | ETC1S or UASTC, R channel |
| Packed ORM / masks | BC7 or BC1 | ASTC | UASTC |
| HDR (sky, env, lightmap) | BC6H | ASTC HDR | UASTC HDR 4x4 or 6x6 |
| UI, text, sprites | BC7 or uncompressed | ASTC 4x4 | UASTC; never ETC1S |

**Checklist**

1. Right-size first. Set texel-density targets per asset class; dropping 4K → 2K cuts memory by 4×, more than any encoder setting.
2. Generate full mip chains with a good filter (Kaiser or Lanczos), filtering in linear space for sRGB content.
3. Renormalize normal-map mips; preserve alpha coverage on cutout textures (foliage, fences) so they don't thin out at distance.
4. Pack channels: roughness/metal/AO into one texture; keep data maps linear, not sRGB.
5. Use the narrowest format per role (BC4 for one channel, BC5 for normals) instead of BC7 everywhere.
6. Apply RDO by content class: start at the engine default (UE λ 30), lower it for normals, UI and hero assets, raise it for terrain detail and distant props.
7. Add BC7Prep (Oodle) or Zstd supercompression (KTX2) for extra lossless savings on BC7-heavy sets.
8. Stream: let the engine load low mips first (KTX2 stores smallest mips first; Unreal uses mip streaming or virtual textures).
9. Measure: compare against source with PSNR plus a perceptual metric such as SSIM or NVIDIA FLIP, then check in-engine at real viewing distances.
10. Pin encoder versions (Oodle Texture SDK version, Basis Universal release) and keep encodes deterministic, so rebuilds don't change every texture and inflate patches.
11. Validate KTX2 output with `ktx validate` before shipping.

## Comparison: which approach when

Oodle Texture is the pick for shipped PC/console BCn content, especially on Unreal or PS5; KTX2 + Basis is the pick when one file must run on many GPU families, especially the web and mobile.

| Approach | On the GPU | Runtime step | Platforms | License |
| --- | --- | --- | --- | --- |
| Oodle Texture RDO + Kraken | Standard BCn | LZ decompress only (hardware on PS5) | PC, consoles | Free in Unreal; commercial otherwise |
| + BC7Prep | Standard BC7 | Decompress + fast untransform (GPU or CPU) | PC, consoles | Same |
| Native BCn/ASTC in KTX2 + Zstd | Standard BCn/ASTC | Zstd decompress | Any, one build per format family | Open (Apache 2.0 tools) |
| Basis ETC1S in KTX2 | Transcoded BCn/ETC/ASTC/PVRTC | BasisLZ decode + transcode | Universal, incl. web | Open (Apache 2.0) |
| Basis UASTC (+RDO, Zstd) in KTX2 | Transcoded, near-BC7 quality | Zstd + transcode | Universal | Open |
| Basis XUASTC LDR in KTX2 (v2.0+) | ASTC native or transcoded | Entropy decode + transcode | Universal | Open |

**Pick by situation**

- Unreal project shipping on PC/console: Oodle Data (Kraken) + Oodle Texture RDO are on by default — tune λ per LOD group and verify the log lines.
- PS5: Kraken + Oodle Texture is the hardware-accelerated path; Sony licenses Oodle Texture for all developers.
- Custom engine, PC/console, no Oodle license: RDO-encode BCn with an open encoder (e.g. bc7enc_rdo) and store in KTX2 with Zstd, or use DirectStorage's GDeflate path.
- Web/WebGL/WebGPU or glTF delivery: KTX2 + Basis via `KHR_texture_basisu` — ETC1S for bulk, UASTC for normals and hero assets.
- Mobile-only: native ASTC in KTX2 + Zstd if one ASTC build covers your devices; Basis when you also need ETC2 or PVRTC fallbacks.
- Highest quality at any cost: non-RDO (λ 0) BC7, or Oodle near-lossless RDO + BC7Prep.

## How good they are

Published results put BCn RDO at roughly 25–45% smaller packages at small visible cost, and Basis' newer latent-space codecs well below 3 bits per texel. No two sources use the same images, metric or back end, so treat these as ballparks and measure on your own corpus (next section).

| Encoder / setup | Content | Result | Source |
| --- | --- | --- | --- |
| Oodle Texture BC7, λ 0 → λ 40, Kraken | "my soup" photo | Kraken saving rises from 3.5% to 45% | [RAD](https://www.radgametools.com/oodletexture.htm) |
| Oodle Texture BC7, λ 0 → λ 40, Kraken | one sample texture | 1.06:1 → 2.06:1 (table in the Oodle Texture section) | [Bloom](https://cbloomrants.blogspot.com/2020/06/oodle-texture-sample-run.html) |
| Kraken + Oodle Texture vs Zip | a shipped game's texture set | 3.16:1 vs 1.64:1 | [Bloom](https://cbloomrants.blogspot.com/2020/09/how-oodle-kraken-and-oodle-texture.html) |
| bc7enc_rdo BC7 (bc7e base), λ 1.0, Deflate | xmen_1024 | 4.53 bits/texel at 37.70 dB PSNR (raw BC7 = 8) | [Geldreich](https://richg42.blogspot.com/2021/02/) |
| bc7enc_rdo BC7, no RDO, Deflate | kodim08 | 7.69 bits/texel — plain BC7 barely compresses | [Geldreich](https://richg42.blogspot.com/2021/02/) |
| BC7 RDO λ 0.5 + Zstd in KTX2 | kodim23 / kodim01 | 362 → 224 KB and 357 → 225 KB vs Zstd alone (~38–40% smaller) | [third-party playground](https://walcht.github.io/walcht/guides/ktx-bcn-playground.html) |
| Basis ETC1S | general | ~0.3–3 bpp, low–medium quality | [Basis README](https://github.com/BinomialLLC/basis_universal) |
| Basis UASTC LDR 4x4 + RDO + Zstd | general | ~5–8 bpp, near-BC7 quality | basisu help |
| Basis XUASTC LDR (4x4–12x12) | general | ~0.3–5.7 bpp depending on block size and quality | basisu help |
| Basis latent codecs, lossy sweet spot | general | ~1.15–2.5 bpp (lossless ~3.5–5.6 bpp) | [Basis README](https://github.com/BinomialLLC/basis_universal) |
| NVIDIA RTX NTC (neural) | SDK sample material set | 2.5 MB vs 12 MB BCn vs 32 MB raw | [RTXNTC](https://github.com/NVIDIA-RTX/RTXNTC) |

**Baseline encoder quality (no RDO).** Aras Pranckevičius' 2020 survey measured Luma-PSNR across a mixed Unity texture set: BC7 and ASTC 4x4 above 42 dB; BC1/BC3, ASTC 6x6 and ETC2 at 35–40 dB; ASTC 8x8 below 35 dB ([Aras](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/)). bc7e was the best BC7 encoder, reaching above 45 dB at 10–20 Mpix/s. Your RDO starts from whatever base encoder you pick, so a weak base caps every λ.

**Neural compression cost.** NTC's inference-on-sample mode cuts texture VRAM up to ~85% but adds roughly 0.5–2 ms per frame on RTX 50-series GPUs; inference-on-load transcodes to BCn at load and costs nothing per frame, but saves only disk ([Tom's Hardware](https://tomshardware.com/pc-components/gpus/benchmarking-nvidias-rtx-neural-texture-compression-tech-that-can-reduce-vram-usage-by-over-80-percent)).

## Samples to compare against

You can benchmark against everything here except Oodle without asking anyone: Basis and bc7enc_rdo are open source and run locally, and Basis has a browser testbed. Oodle needs a free evaluation SDK from RAD, or its built-in copy inside Unreal Engine.

**Galleries and live tools**

| Resource | What you get | How to use it |
| --- | --- | --- |
| [Oodle Texture RDO examples](https://www.radgametools.com/oodletextureexamples.htm) | 8 crops (512×512) stepped through λ levels with Kraken sizes: Painted metal BC7, Medieval modular BC1 and BC7, kodim18 BC1, Old saddle BC7 + BC5 normals, Tree stump BC7, Cliffside BC7 | Visual targets. Painted metal (CC0, ambientCG) and kodim18 (Kodak set) have public sources, so you can encode the same images; the Quixel crops are not redistributable |
| [Oodle ratio/RMSE chart](https://www.radgametools.com/images/oodle_texture_ratio_and_rmse.svg) and [BC7Prep chart](https://www.radgametools.com/images/oodle_texture_bc7prep_ratios.svg) | RAD's R-D curves | Shape of the curve to beat |
| [Basis KTX2 encode/transcode testbed](https://subquantumtech.com/xu/ktx2_encode_test/) | Encode your own image in the browser to ETC1S, UASTC, ASTC, XUASTC and more; view the transcoded result | Fastest way to see Basis on your content |
| [Basis README R-D graph](https://github.com/BinomialLLC/basis_universal) | XUASTC LDR 4x4 rate vs distortion over a 151-image corpus, scored with PSNR-HVS-M | Published curve to place yours against |
| [UASTC HDR 4x4 examples](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-HDR-Examples), [ASTC HDR 6x6 examples](https://github.com/BinomialLLC/basis_universal/wiki/ASTC-HDR-6x6-Example-Images) | HDR comparison images | HDR baseline |
| [Aras' interactive charts](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/) | Quality vs encode speed for BC1/3, BC7, ASTC, ETC2 encoders | Choosing a base encoder |
| [Geldreich's bc7enc_rdo posts](https://richg42.blogspot.com/2021/02/) | Command lines, mode histograms, bits/texel and PSNR per run | Reproducible open RDO numbers |

**Encoders to run locally**

- [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo) — open RDO for BC1–BC7; the closest free stand-in for Oodle Texture. λ via `-z`, LZ lookback window via `-zc`.
- [Basis Universal v2.50](https://github.com/BinomialLLC/basis_universal) — every Basis codec; `basisu -compare` prints PSNR and SSIM with delta images.
- [astcenc](https://github.com/ARM-software/astc-encoder) — reference ASTC encoder; reports PSNR.
- Oodle Texture — free evaluation SDK on request from RAD, or the copy that ships with Unreal (check the Unreal EULA before using it to benchmark a competing product).

**Test images**

- Kodak 24 (kodim01–24) — the de facto common set; both RAD and Geldreich publish results on it.
- CC0 PBR sets from ambientCG and Poly Haven — real albedo, normal, roughness and AO maps, freely redistributable.
- Your own shipped textures — the only set that answers "is this better for us"; keep a frozen copy.

**Metrics**

- Rate: bits per texel *after* the real back end (zstd, zlib, Kraken, GDeflate), never raw block size.
- Distortion: RGB PSNR and Luma-PSNR for comparability; SSIM; PSNR-HVS-M (what Basis reports); NVIDIA FLIP for perceptual differences.
- Normal maps: light the map and score the lit image, as Aras does, or measure angular error in degrees.
- Packed masks: score each channel separately; a combined RGB PSNR hides a ruined roughness channel.

## Building your own RDO + transcoder

The all-in-one design you describe already exists in open form: Basis Universal v2.50 (Apache 2.0) pairs offline RDO with latent-space supercompression and transcodes to every GPU format. Oodle Texture is the closed bar for no-runtime BCn RDO. A new tool wins by beating them on R-D curves for specific content, or on speed, determinism and integration — not by being all-in-one.

**Diagram — suggested encoder architecture (two outputs, one harness):**

```text
Source textures → Analyze → Base encode → RDO core (λ + rate model of the back end)
(albedo, normals,  (role, perceptual   (BCn / ASTC,           │
 masks, HDR)        weights, smooth     multi-target aware)   ├─ Output A (no runtime step):
                    blocks)                                   │    Native BCn/ASTC (RDO-tuned, Oodle-style)
                                                              │    → LZ back end (zstd, Kraken, GDeflate) → GPU texture
                                                              └─ Output B (transcode at load):
                                                                   Latent coding (DCT, DPCM, entropy coder)
                                                                   → Runtime transcode (to BCn, ASTC, ETC per device) → GPU texture
GPU texture → Benchmark harness: scores both outputs, retunes λ and the rate model
              (frozen corpus · bits/texel after the real back end · PSNR, SSIM, FLIP · BD-rate per texture role)
```

Share analysis, base encoding and the RDO core; output A matches Oodle (no runtime cost), output B matches Basis (one file, any GPU).

**State of the art you are competing with**

| Tool | Approach | Runtime work | License |
| --- | --- | --- | --- |
| Oodle Texture | Perceptual BCn RDO aimed at any LZ back end; BC7Prep transform | None (BC7Prep: fast untransform) | Commercial |
| bc7enc_rdo | Open BCn RDO with an LZ bit estimate over a lookback window | None | Open source |
| Basis UASTC + RDO + Zstd | Dual-target 4x4 format restricted to partitions common to ASTC and BC7 | Zstd + cheap transcode | Apache 2.0 |
| Basis XUASTC LDR (v2.0, Jan 2026) | "JPEG for ASTC": DCT on ASTC weight grids, endpoint DPCM, block prediction, arithmetic or Zstd coding | Entropy decode + optional transcode | Apache 2.0 |
| Basis XUBC7 (v2.50, mid-2026) | Supercompressed full BC7 via weight-grid DCT and DPCM; lossless by default, lossy with RDO; transcodes near-losslessly to ASTC 4x4 | Entropy decode + optional transcode | Apache 2.0 |
| NVIDIA RTX NTC | Learned latents + tiny MLP per material, up to 16 channels | Shader inference, or decode to BCn on load | NVIDIA SDK license |

**Techniques worth reusing**

1. Model the real back end's rate. bc7enc_rdo estimates LZ bits against a sliding lookback window (128–32,768 bytes) and tries to copy byte runs from earlier blocks; RAD optimizes for size after any LZ.
2. Protect smooth blocks. Scale distortion up on low-variance blocks (bc7enc_rdo auto-computes a factor around 19) or RDO turns gradients into banding.
3. Allocate perceptually. Oodle spends more bits where errors are visible and lets RDO decline to change a block when the result would be unacceptable — that is what makes one λ safe across a whole data set.
4. Reduce entropy in the base encoder. Restricting modes and quantizing choices gives 5–10% LZ gains at no encode cost (bc7enc `-e`).
5. Add lossless transforms before LZ. BC7Prep-style bit-field reordering gains 5–15% on BC7 and stacks with RDO.
6. Code in the latent space. Transform-code the block parameters (weights, endpoints) instead of hoping LZ finds repeats; this is where Basis now gets ~1–2.5 bpp.
7. Deblock in the loop. Basis v2.50 optimizes large ASTC blocks knowing a shader deblocking filter will run at sample time.
8. Optimize for several targets at once. UASTC chooses blocks that are good as both ASTC and BC7 (about 0.75 dB average loss when converted to BC7), so one encode serves every platform.

**Where there is room**

- Content-aware tuning per texture role: normals scored on lit output, packed masks per channel, alpha coverage for cutouts.
- Rate models matched to a specific back end (Kraken, zstd with dictionaries, GPU GDeflate) and its chunk size.
- Joint optimization across mips and across textures in one package, e.g. shared Zstd dictionaries.
- Fast, deterministic encoding (GPU or SIMD) so builds cache and patches stay small.
- A transcoder that also emits RDO'd BCn/ASTC for platforms with hardware LZ decoders, combining both families.

**Benchmark harness plan**

1. Freeze a corpus: Kodak 24, ~100 CC0 PBR maps grouped by role, and a slice of your own textures.
2. Run baselines: bc7e/astcenc (no RDO), bc7enc_rdo across a λ sweep, Basis v2.50 across `-quality` for each codec, and Oodle if you get the eval SDK.
3. Pass every output through the same back ends: zstd -19, zlib -9, and Kraken or GDeflate if available.
4. Score with the metrics above, per role.
5. Plot rate-distortion curves per role and summarize with BD-rate (Bjøntegaard delta: average % size change at equal quality), the standard figure video codecs use.
6. Also log encode Mpix/s, transcode/decode time and run-to-run determinism.

## Sources

- [Oodle Texture product page — RAD Game Tools](https://www.radgametools.com/oodletexture.htm)
- [Oodle Texture RDO examples — RAD Game Tools](https://www.radgametools.com/oodletextureexamples.htm)
- [Oodle Texture in Unreal Engine — Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/oodle-texture-in-unreal-engine)
- [Oodle Data — Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/oodle-data)
- [Oodle now free to use in Unreal Engine — Epic blog](https://www.unrealengine.com/en-US/blog/oodle-now-free-to-use-in-unreal-engine-via-github)
- [How Oodle Kraken and Oodle Texture supercharge the PS5 I/O — Charles Bloom](https://cbloomrants.blogspot.com/2020/09/how-oodle-kraken-and-oodle-texture.html)
- [Oodle Texture BC7Prep data flow — Charles Bloom](https://cbloomrants.blogspot.com/2020/06/oodle-texture-bc7prep-data-flow.html)
- [Oodle Texture sample run — Charles Bloom](https://cbloomrants.blogspot.com/2020/06/oodle-texture-sample-run.html)
- [KTX 2.0 specification — Khronos](https://github.khronos.org/KTX-Specification/ktxspec.v2.html)
- [KHR_texture_basisu — Khronos glTF](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)
- [KTX-Software v4.3 release notes — Khronos](https://github.com/KhronosGroup/KTX-Software/releases/tag/v4.3.2)
- [Basis Universal release notes — Binomial](https://github.com/BinomialLLC/basis_universal/wiki/Release-Notes)
- [Basis Universal repository and README — Binomial](https://github.com/BinomialLLC/basis_universal)
- [Basis Universal releases (v2.50) — GitHub](https://github.com/BinomialLLC/basis_universal/releases)
- [KTX2 file format support technical details — Binomial wiki](https://github.com/BinomialLLC/basis_universal/wiki/KTX2-File-Format-Support-Technical-Details)
- [XUBC7 usage guide — Binomial wiki](https://github.com/BinomialLLC/basis_universal/wiki/XUBC7-Usage-Guide)
- [JPEG for ASTC — Binomial wiki](https://github.com/BinomialLLC/basis_universal/wiki/JPEG-for-ASTC)
- [bc7enc_rdo — Rich Geldreich](https://github.com/richgel999/bc7enc_rdo)
- [bc7enc_rdo posts, Feb 2021 — Rich Geldreich's blog](https://richg42.blogspot.com/2021/02/)
- [KTX BCn playground — walcht](https://walcht.github.io/walcht/guides/ktx-bcn-playground.html)
- [Texture Compression in 2020 — Aras Pranckevičius](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/)
- [RTX Neural Texture Compression SDK — NVIDIA](https://github.com/NVIDIA-RTX/RTXNTC)
- [Benchmarking RTX NTC — Tom's Hardware](https://tomshardware.com/pc-components/gpus/benchmarking-nvidias-rtx-neural-texture-compression-tech-that-can-reduce-vram-usage-by-over-80-percent)

The 2.9.15–2.9.16 Oodle build numbers come from third-party game-mod release notes, not RAD; confirm against your SDK.
