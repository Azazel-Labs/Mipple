# bc7f harness

`bc7f_encode` runs Binomial's analytical **bc7f** BC7 encoder (from Basis Universal, Apache 2.0) on a PNG and writes
a DX10 DDS, so `mipple score` and `scripts/head2head.py --bc7f` can grade it like any other encoder.

```sh
git clone --depth 1 https://github.com/BinomialLLC/basis_universal ~/src/basis_universal
git clone --depth 1 https://github.com/richgel999/bc7enc_rdo ~/src/bc7enc_rdo      # for lodepng
BASIS_DIR=~/src/basis_universal BC7ENC_DIR=~/src/bc7enc_rdo tools/bc7f/build.sh
tools/bc7f/bc7f_encode in.png out.dds default        # fastest|faster|fast|default|partial|nonanalytical
```

It prints `secs=…`, the single-threaded time of the block loop only (no PNG load or file write).

Presets map to `bc7f::cPackBC7FlagDefault*` in `basisu_transcoder_internal.h`:

| Preset | What bc7f does |
| --- | --- |
| fastest | Mode 6 only, trivial-block path |
| faster | Mode 6 with p-bit optimization; modes 4/5/6 for alpha |
| fast | + 2-subset modes (1/3 RGB, 7 RGBA) |
| default | + 3-subset modes (0/2) and dual-plane (4/5) for RGB |
| partial | default + real SSE checks on hard blocks |
| nonanalytical | brute force at the top level (slow) |

Not built in the mipple dev sandbox (compiling third-party code was blocked there); build it locally.

**bc7g** is announced but unreleased; `scripts/head2head.py --bc7g PATH` is ready for it.
