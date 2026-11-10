# Competitive benchmark

Corpus: 24 images from `test-images\kodak`; threads per encoder: 1. Sizes are corpus totals after each back end; PSNR is the per-image mean over the format's channels; time is total encode seconds (see scripts/compete.py for what each encoder's time covers).


## BC1

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e0 | 36.05 | 2.994 | 3.118 | 0.02 | 467.42 | — |
| mipple e1 | 36.14 | 2.992 | 3.117 | 0.04 | 245.35 | — |
| mipple e2 | 36.54 | 3.013 | 3.129 | 0.26 | 35.85 | — |
| mipple e3 | 36.64 | 3.004 | 3.122 | 0.44 | 21.63 | — |
| mipple e4 | 36.69 | 3.012 | 3.123 | 0.70 | 13.44 | — |
| mipple e5 | 36.71 | 3.010 | 3.121 | 1.13 | 8.34 | — |
| mipple e6 | 36.74 | 3.000 | 3.113 | 2.97 | 3.17 | — |
| mipple e7 | 36.74 | 2.997 | 3.111 | 3.77 | 2.50 | — |
| mipple e8 | 36.74 | 2.996 | 3.110 | 4.38 | 2.15 | — |
| rgbcx L0 | 36.00 | 2.988 | 3.114 | 0.10 | 91.62 | — |
| rgbcx L6 | 36.43 | 3.040 | 3.144 | 0.62 | 15.27 | — |
| rgbcx L12 | 36.67 | 3.010 | 3.121 | 4.93 | 1.91 | — |
| rgbcx L18 | 36.74 | 2.994 | 3.106 | 16.39 | 0.58 | — |
| icbc:1 | 36.05 | 2.994 | 3.119 | 0.28 | 34.14 | — |
| icbc:5 | 36.57 | 2.938 | 3.030 | 0.72 | 13.09 | — |
| icbc:8 | 36.68 | 2.997 | 3.113 | 1.86 | 5.07 | — |
| icbc:9 | 36.71 | 3.007 | 3.120 | 4.14 | 2.28 | — |
| stb_bc1 | 35.80 | 3.009 | 3.133 | 0.23 | 40.42 | — |
| stb_bc1hq | 36.03 | 2.995 | 3.120 | 0.25 | 37.29 | — |
| ispc_bc1 | 35.71 | 3.001 | 3.123 | 0.02 | 450.12 | — |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx L0 | 36.00 | 91.62 | mipple e0 (36.05 dB, 467.42 Mpix/s) | ✓ |
| rgbcx L6 | 36.43 | 15.27 | mipple e2 (36.54 dB, 35.85 Mpix/s) | ✓ |
| rgbcx L12 | 36.67 | 1.91 | mipple e4 (36.69 dB, 13.44 Mpix/s) | ✓ |
| rgbcx L18 | 36.74 | 0.58 | mipple e6 (36.74 dB, 3.17 Mpix/s) | ✓ |
| icbc:1 | 36.05 | 34.14 | mipple e0 (36.05 dB, 467.42 Mpix/s) | ✓ |
| icbc:5 | 36.57 | 13.09 | mipple e3 (36.64 dB, 21.63 Mpix/s) | ✓ |
| icbc:8 | 36.68 | 5.07 | mipple e4 (36.69 dB, 13.44 Mpix/s) | ✓ |
| icbc:9 | 36.71 | 2.28 | mipple e5 (36.71 dB, 8.34 Mpix/s) | ✓ |
| stb_bc1 | 35.80 | 40.42 | mipple e0 (36.05 dB, 467.42 Mpix/s) | ✓ |
| stb_bc1hq | 36.03 | 37.29 | mipple e0 (36.05 dB, 467.42 Mpix/s) | ✓ |
| ispc_bc1 | 35.71 | 450.12 | mipple e0 (36.05 dB, 467.42 Mpix/s) | ✓ |

## BC4

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e0 | 42.97 | 3.782 | 3.790 | 0.01 | 1572.86 | — |
| mipple e1 | 43.60 | 3.784 | 3.792 | 0.02 | 558.41 | — |
| mipple e2 | 44.32 | 3.780 | 3.787 | 0.05 | 200.36 | — |
| mipple e5 | 44.65 | 3.781 | 3.788 | 0.09 | 104.86 | — |
| mipple e6 | 44.81 | 3.781 | 3.788 | 0.15 | 64.37 | — |
| mipple e8 | 44.98 | 3.781 | 3.788 | 0.36 | 26.49 | — |
| rgbcx hl | 43.13 | 3.782 | 3.790 | 0.03 | 377.49 | — |
| rgbcx hr5 | 44.79 | 3.769 | 3.779 | 12.18 | 0.77 | — |
| stb_bc4 | 43.13 | 3.779 | 3.787 | 0.05 | 195.39 | — |
| ispc_bc4 | 42.86 | 3.784 | 3.792 | 0.01 | 1179.65 | — |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hl | 43.13 | 377.49 | mipple e1 (43.60 dB, 558.41 Mpix/s) | ✓ |
| rgbcx hr5 | 44.79 | 0.77 | mipple e6 (44.81 dB, 64.37 Mpix/s) | ✓ |
| stb_bc4 | 43.13 | 195.39 | mipple e1 (43.60 dB, 558.41 Mpix/s) | ✓ |
| ispc_bc4 | 42.86 | 1179.65 | mipple e0 (42.97 dB, 1572.86 Mpix/s) | ✓ |

## BC5

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e0 | 42.93 | 7.255 | 7.282 | 0.01 | 925.21 | — |
| mipple e1 | 43.56 | 7.260 | 7.287 | 0.03 | 307.40 | — |
| mipple e2 | 44.27 | 7.266 | 7.291 | 0.09 | 102.69 | — |
| mipple e5 | 44.60 | 7.270 | 7.294 | 0.18 | 53.74 | — |
| mipple e6 | 44.76 | 7.273 | 7.297 | 0.29 | 32.29 | — |
| mipple e8 | 44.93 | 7.276 | 7.300 | 0.72 | 13.16 | — |
| rgbcx hl | 43.10 | 7.258 | 7.285 | 0.05 | 192.60 | — |
| rgbcx hr5 | 44.74 | 7.260 | 7.288 | 24.11 | 0.39 | — |
| stb_bc5 | 43.10 | 7.255 | 7.281 | 0.09 | 108.10 | — |
| ispc_bc5 | 42.82 | 7.256 | 7.283 | 0.02 | 593.53 | — |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hl | 43.10 | 192.60 | mipple e1 (43.56 dB, 307.40 Mpix/s) | ✓ |
| rgbcx hr5 | 44.74 | 0.39 | mipple e6 (44.76 dB, 32.29 Mpix/s) | ✓ |
| stb_bc5 | 43.10 | 108.10 | mipple e1 (43.56 dB, 307.40 Mpix/s) | ✓ |
| ispc_bc5 | 42.82 | 593.53 | mipple e0 (42.93 dB, 925.21 Mpix/s) | ✓ |

## BC7

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e0 | 42.22 | 6.676 | 6.839 | 0.03 | 299.59 | — |
| mipple e1 | 43.48 | 6.757 | 6.906 | 0.07 | 137.57 | — |
| mipple e2 | 44.11 | 6.904 | 7.031 | 0.18 | 53.41 | — |
| mipple e3 | 44.78 | 7.072 | 7.183 | 0.40 | 23.36 | — |
| mipple e4 | 45.30 | 7.157 | 7.259 | 1.13 | 8.34 | — |
| mipple e5 | 45.37 | 7.176 | 7.277 | 1.31 | 7.21 | — |
| mipple e6 | 45.64 | 7.318 | 7.402 | 3.73 | 2.53 | — |
| mipple e7 | 45.66 | 7.321 | 7.405 | 5.03 | 1.88 | — |
| mipple e8 | 46.02 | 7.329 | 7.413 | 33.44 | 0.28 | — |
| mipple e9 | 46.03 | 7.332 | 7.415 | 44.09 | 0.21 | — |
| bc7e u0 | 42.42 | 6.721 | 6.876 | 0.14 | 66.93 | — |
| bc7e u2 | 45.32 | 7.355 | 7.451 | 2.32 | 4.07 | — |
| bc7e u4 | 45.63 | 7.341 | 7.433 | 4.55 | 2.08 | — |
| bc7e u6 | 46.01 | 7.352 | 7.444 | 38.72 | 0.24 | — |
| bc7f:fastest | 41.90 | 6.575 | 6.749 | 0.12 | 80.87 | — |
| bc7f:faster | 41.90 | 6.575 | 6.749 | 0.11 | 84.56 | — |
| bc7f:fast | 43.07 | 6.707 | 6.857 | 0.17 | 56.34 | — |
| bc7f:default | 43.89 | 6.934 | 7.063 | 0.29 | 32.85 | — |
| bc7f:partial | 44.19 | 7.022 | 7.140 | 0.48 | 19.86 | — |
| ispc_bc7:ultrafast | 42.41 | 6.719 | 6.874 | 0.11 | 84.11 | — |
| ispc_bc7:veryfast | 45.11 | 7.289 | 7.394 | 1.74 | 5.41 | — |
| ispc_bc7:fast | 45.29 | 7.357 | 7.456 | 2.95 | 3.20 | — |
| ispc_bc7:basic | 45.51 | 7.399 | 7.486 | 7.20 | 1.31 | — |
| ispc_bc7:slow | 45.63 | 7.444 | 7.527 | 24.62 | 0.38 | — |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| bc7e u0 | 42.42 | 66.93 | mipple e1 (43.48 dB, 137.57 Mpix/s) | ✓ |
| bc7e u2 | 45.32 | 4.07 | mipple e5 (45.37 dB, 7.21 Mpix/s) | ✓ |
| bc7e u4 | 45.63 | 2.08 | mipple e6 (45.64 dB, 2.53 Mpix/s) | ✓ |
| bc7e u6 | 46.01 | 0.24 | mipple e8 (46.02 dB, 0.28 Mpix/s) | ✓ |
| bc7f:fastest | 41.90 | 80.87 | mipple e0 (42.22 dB, 299.59 Mpix/s) | ✓ |
| bc7f:faster | 41.90 | 84.56 | mipple e0 (42.22 dB, 299.59 Mpix/s) | ✓ |
| bc7f:fast | 43.07 | 56.34 | mipple e1 (43.48 dB, 137.57 Mpix/s) | ✓ |
| bc7f:default | 43.89 | 32.85 | mipple e2 (44.11 dB, 53.41 Mpix/s) | ✓ |
| bc7f:partial | 44.19 | 19.86 | mipple e3 (44.78 dB, 23.36 Mpix/s) | ✓ |
| ispc_bc7:ultrafast | 42.41 | 84.11 | mipple e1 (43.48 dB, 137.57 Mpix/s) | ✓ |
| ispc_bc7:veryfast | 45.11 | 5.41 | mipple e4 (45.30 dB, 8.34 Mpix/s) | ✓ |
| ispc_bc7:fast | 45.29 | 3.20 | mipple e4 (45.30 dB, 8.34 Mpix/s) | ✓ |
| ispc_bc7:basic | 45.51 | 1.31 | mipple e6 (45.64 dB, 2.53 Mpix/s) | ✓ |
| ispc_bc7:slow | 45.63 | 0.38 | mipple e6 (45.64 dB, 2.53 Mpix/s) | ✓ |
