# mipple in the browser

A static page that runs mipple compiled to WebAssembly. Pick a sample texture or drop your own
image, move the sliders, and compare the original with the decoded blocks: split, side by side, or
the amplified difference. The result panel shows Zstd-19 bits per texel, PSNR, RMSE and SSIM (the
numbers `mipple score` prints), the encode time, the equivalent `mipple encode` command and a
`.ktx2` download. That command run through the native CLI gives a byte-identical file.

Everything runs in the visitor's browser, in a Web Worker on one thread. Nothing is uploaded.
When the browser has WebGPU, the base blocks come from mipple's GPU kernels (the same blocks as the
CPU encoder: each encode checks a 256-block sample against the CPU and drops to the CPU for good on
a mismatch). Every mip level goes in one submission with one readback. The first GPU encode
compiles the shaders, which takes 1-3 s once. RDO, Zstd and the metrics stay on the CPU.

## Build

You need Rust with the `wasm32-unknown-unknown` target, [wasm-pack](https://github.com/drager/wasm-pack),
and a clang that can target wasm32 (zstd is C). Ubuntu's `clang` package works. On Windows the
LLVM or Swift toolchain clang works.

```sh
rustup target add wasm32-unknown-unknown
CARGO_PROFILE_RELEASE_DEBUG=false RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build web --target web --release --no-typescript --no-pack --out-dir www/pkg
python -m http.server -d web/www 8000   # then open http://localhost:8000
```

The page must be served over HTTP. Module workers do not load from `file://`.
`.github/workflows/pages.yml` runs the same build and deploys `web/www` to GitHub Pages.

## Layout

- `src/lib.rs`: the wasm-bindgen API. It decodes images with the same `image` crate the CLI
  uses, and it encodes with the same options, metrics and KTX2 writer as `mipple encode`.
- `www/`: the page. Plain HTML, CSS and ES modules, no framework or bundler. `worker.js` runs the
  encoder; the page terminates and respawns the worker to cancel a stale encode.
- `make_samples.py`: regenerates `www/samples/` from Poly Haven originals in `test-images/polyhaven`
  (512×512 Lanczos PNGs plus JPEG thumbnails).

## Credits

The sample textures are from [Poly Haven](https://polyhaven.com/textures) and are CC0 (public
domain). They are downscaled to 512×512.
