# Competitive benchmark

Corpus: 24 images from `test-images\kodak`; threads per encoder: 22. Sizes are corpus totals after each back end; PSNR is the per-image mean over the format's channels; time is total encode seconds (see scripts/compete.py for what each encoder's time covers).


## BC1

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e5 | 36.71 | 3.010 | 3.121 | 0.36 | 25.87 | +0.0% |
| rgbcx L18 | 36.74 | 2.994 | 3.106 | 2.07 | 4.55 | — |
| bc7enc_rdo rgbcx L18 λ=0 | 36.74 | 2.994 | 3.106 | 2.17 | 4.35 | — |
| bc7enc_rdo rgbcx L18 λ=0.1 | 36.72 | 2.957 | 3.066 | 13.13 | 0.72 | — |
| bc7enc_rdo rgbcx L18 λ=0.25 | 36.67 | 2.908 | 3.012 | 13.34 | 0.71 | +5.3% |
| bc7enc_rdo rgbcx L18 λ=0.5 | 36.51 | 2.828 | 2.922 | 13.14 | 0.72 | +11.2% |
| bc7enc_rdo rgbcx L18 λ=1 | 36.09 | 2.682 | 2.763 | 12.73 | 0.74 | +22.4% |
| bc7enc_rdo rgbcx L18 λ=2 | 35.14 | 2.480 | 2.550 | 12.71 | 0.74 | +40.7% |
| bc7enc_rdo rgbcx L18 λ=4 | 33.44 | 2.253 | 2.312 | 11.91 | 0.79 | — |
| mipple e5 λ=0 | 36.71 | 3.010 | 3.121 | 0.19 | 48.63 | +0.0% |
| mipple e5 λ=0.25 | 36.72 | 2.890 | 3.014 | 9.86 | 0.96 | +0.0% |
| mipple e5 λ=0.5 | 36.68 | 2.787 | 2.914 | 11.69 | 0.81 | -0.0% |
| mipple e5 λ=1 | 36.55 | 2.578 | 2.705 | 10.09 | 0.94 | +0.0% |
| mipple e5 λ=2 | 36.14 | 2.222 | 2.344 | 9.56 | 0.99 | +0.0% |
| mipple e5 λ=4 | 35.26 | 1.802 | 1.929 | 8.88 | 1.06 | +0.0% |
| mipple e5 λ=8 | 33.76 | 1.392 | 1.519 | 7.82 | 1.21 | +0.0% |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx L18 | 36.74 | 4.55 | ✗ | best is mipple e5, 0.02 dB short |
| bc7enc_rdo rgbcx L18 λ=0 | 36.74 | 4.35 | ✗ | best is mipple e5, 0.02 dB short |
| bc7enc_rdo rgbcx L18 λ=0.1 | 36.72 | 0.72 | ✗ | best is mipple e5, 0.01 dB short |
| bc7enc_rdo rgbcx L18 λ=0.25 | 36.67 | 0.71 | mipple e5 (36.71 dB, 25.87 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx L18 λ=0.5 | 36.51 | 0.72 | mipple e5 (36.71 dB, 25.87 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx L18 λ=1 | 36.09 | 0.74 | mipple e5 (36.71 dB, 25.87 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx L18 λ=2 | 35.14 | 0.74 | mipple e5 (36.71 dB, 25.87 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx L18 λ=4 | 33.44 | 0.79 | mipple e5 (36.71 dB, 25.87 Mpix/s) | ✓ |

BD-rate of **mipple RDO** vs each RDO family (negative = mipple needs fewer bits at equal PSNR):

| vs family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| bc7enc_rdo rgbcx | -26.8% | -23.9% |

## BC4

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e5 | 44.65 | 3.781 | 3.788 | 0.07 | 135.76 | +0.0% |
| rgbcx hr5 | 44.79 | 3.769 | 3.779 | 1.66 | 5.67 | — |
| bc7enc_rdo rgbcx hr5 λ=0 | 44.79 | 3.769 | 3.779 | 1.62 | 5.83 | — |
| bc7enc_rdo rgbcx hr5 λ=0.1 | 44.72 | 3.713 | 3.723 | 9.41 | 1.00 | — |
| bc7enc_rdo rgbcx hr5 λ=0.25 | 44.32 | 3.529 | 3.538 | 9.41 | 1.00 | +0.4% |
| bc7enc_rdo rgbcx hr5 λ=0.5 | 43.42 | 3.251 | 3.263 | 8.98 | 1.05 | +5.8% |
| bc7enc_rdo rgbcx hr5 λ=1 | 41.69 | 2.939 | 2.952 | 8.89 | 1.06 | +15.0% |
| bc7enc_rdo rgbcx hr5 λ=2 | 38.87 | 2.636 | 2.650 | 8.41 | 1.12 | +31.6% |
| bc7enc_rdo rgbcx hr5 λ=4 | 35.52 | 2.357 | 2.370 | 8.28 | 1.14 | +60.4% |
| mipple e5 λ=0 | 44.65 | 3.781 | 3.788 | 0.02 | 415.68 | +0.0% |
| mipple e5 λ=0.25 | 44.10 | 3.346 | 3.371 | 5.12 | 1.84 | +0.0% |
| mipple e5 λ=0.5 | 42.83 | 2.854 | 2.898 | 5.66 | 1.67 | -0.0% |
| mipple e5 λ=1 | 40.92 | 2.373 | 2.428 | 5.89 | 1.60 | +0.0% |
| mipple e5 λ=2 | 38.14 | 1.887 | 1.949 | 5.65 | 1.67 | +0.0% |
| mipple e5 λ=4 | 34.50 | 1.333 | 1.402 | 4.95 | 1.91 | +0.0% |
| mipple e5 λ=8 | 31.33 | 0.888 | 0.965 | 4.11 | 2.30 | +0.0% |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hr5 | 44.79 | 5.67 | ✗ | best is mipple e5, 0.14 dB short |
| bc7enc_rdo rgbcx hr5 λ=0 | 44.79 | 5.83 | ✗ | best is mipple e5, 0.14 dB short |
| bc7enc_rdo rgbcx hr5 λ=0.1 | 44.72 | 1.00 | ✗ | best is mipple e5, 0.07 dB short |
| bc7enc_rdo rgbcx hr5 λ=0.25 | 44.32 | 1.00 | mipple e5 (44.65 dB, 135.76 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=0.5 | 43.42 | 1.05 | mipple e5 (44.65 dB, 135.76 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=1 | 41.69 | 1.06 | mipple e5 (44.65 dB, 135.76 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=2 | 38.87 | 1.12 | mipple e5 (44.65 dB, 135.76 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=4 | 35.52 | 1.14 | mipple e5 (44.65 dB, 135.76 Mpix/s) | ✓ |

BD-rate of **mipple RDO** vs each RDO family (negative = mipple needs fewer bits at equal PSNR):

| vs family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| bc7enc_rdo rgbcx | -20.0% | -18.4% |

## BC5

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e5 | 44.60 | 7.270 | 7.294 | 0.08 | 113.12 | +0.0% |
| rgbcx hr5 | 44.74 | 7.260 | 7.288 | 3.41 | 2.77 | — |
| bc7enc_rdo rgbcx hr5 λ=0 | 44.74 | 7.260 | 7.288 | 3.35 | 2.82 | — |
| bc7enc_rdo rgbcx hr5 λ=0.1 | 44.69 | 7.189 | 7.215 | 12.19 | 0.77 | — |
| bc7enc_rdo rgbcx hr5 λ=0.25 | 44.35 | 6.889 | 6.917 | 12.26 | 0.77 | +0.2% |
| bc7enc_rdo rgbcx hr5 λ=0.5 | 43.56 | 6.390 | 6.429 | 11.73 | 0.80 | +8.0% |
| bc7enc_rdo rgbcx hr5 λ=1 | 42.07 | 5.833 | 5.882 | 10.99 | 0.86 | +19.5% |
| bc7enc_rdo rgbcx hr5 λ=2 | 39.47 | 5.316 | 5.375 | 10.79 | 0.87 | +42.9% |
| bc7enc_rdo rgbcx hr5 λ=4 | 36.00 | 4.828 | 4.896 | 10.84 | 0.87 | +84.4% |
| mipple e5 λ=0 | 44.60 | 7.270 | 7.294 | 0.04 | 218.82 | +0.0% |
| mipple e5 λ=0.25 | 43.84 | 6.146 | 6.244 | 9.78 | 0.96 | +0.0% |
| mipple e5 λ=0.5 | 42.39 | 5.058 | 5.215 | 9.65 | 0.98 | +0.0% |
| mipple e5 λ=1 | 40.41 | 4.068 | 4.247 | 9.43 | 1.00 | +0.0% |
| mipple e5 λ=2 | 37.78 | 3.166 | 3.348 | 9.22 | 1.02 | +0.0% |
| mipple e5 λ=4 | 34.32 | 2.191 | 2.355 | 8.34 | 1.13 | +0.0% |
| mipple e5 λ=8 | 31.18 | 1.422 | 1.554 | 6.60 | 1.43 | +0.0% |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| rgbcx hr5 | 44.74 | 2.77 | ✗ | best is mipple e5, 0.14 dB short |
| bc7enc_rdo rgbcx hr5 λ=0 | 44.74 | 2.82 | ✗ | best is mipple e5, 0.14 dB short |
| bc7enc_rdo rgbcx hr5 λ=0.1 | 44.69 | 0.77 | ✗ | best is mipple e5, 0.08 dB short |
| bc7enc_rdo rgbcx hr5 λ=0.25 | 44.35 | 0.77 | mipple e5 (44.60 dB, 113.12 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=0.5 | 43.56 | 0.80 | mipple e5 (44.60 dB, 113.12 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=1 | 42.07 | 0.86 | mipple e5 (44.60 dB, 113.12 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=2 | 39.47 | 0.87 | mipple e5 (44.60 dB, 113.12 Mpix/s) | ✓ |
| bc7enc_rdo rgbcx hr5 λ=4 | 36.00 | 0.87 | mipple e5 (44.60 dB, 113.12 Mpix/s) | ✓ |

BD-rate of **mipple RDO** vs each RDO family (negative = mipple needs fewer bits at equal PSNR):

| vs family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| bc7enc_rdo rgbcx | -26.1% | -23.8% |

## BC7

| Encoder config | PSNR (dB) | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s | Zstd size vs mipple RDO curve at equal PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| bc7e u6 | 46.01 | 7.352 | 7.444 | 10.25 | 0.92 | — |
| mipple e5 | 45.37 | 7.176 | 7.277 | 0.29 | 32.03 | +0.0% |
| bc7enc_rdo bc7e u6 λ=0 | 46.01 | 7.352 | 7.444 | 8.69 | 1.09 | — |
| bc7enc_rdo bc7e u6 λ=0.05 | 45.80 | 7.269 | 7.355 | 57.80 | 0.16 | — |
| bc7enc_rdo bc7e u6 λ=0.1 | 45.39 | 7.136 | 7.215 | 57.12 | 0.17 | — |
| bc7enc_rdo bc7e u6 λ=0.25 | 43.80 | 6.698 | 6.763 | 55.78 | 0.17 | +23.0% |
| bc7enc_rdo bc7e u6 λ=0.5 | 41.26 | 6.122 | 6.178 | 54.07 | 0.17 | +50.5% |
| bc7enc_rdo bc7e u6 λ=1 | 37.80 | 5.441 | 5.490 | 51.78 | 0.18 | +97.2% |
| bc7enc_rdo bc7enc.cpp -e λ=0 | 43.83 | 6.408 | 6.579 | 1.65 | 5.73 | +17.3% |
| bc7enc_rdo bc7enc.cpp -e λ=0.05 | 43.66 | 6.316 | 6.473 | 44.35 | 0.21 | +17.9% |
| bc7enc_rdo bc7enc.cpp -e λ=0.1 | 43.31 | 6.160 | 6.308 | 43.44 | 0.22 | +20.0% |
| bc7enc_rdo bc7enc.cpp -e λ=0.25 | 41.99 | 5.715 | 5.846 | 43.16 | 0.22 | +29.4% |
| bc7enc_rdo bc7enc.cpp -e λ=0.5 | 39.90 | 5.181 | 5.300 | 41.42 | 0.23 | +48.7% |
| bc7enc_rdo bc7enc.cpp -e λ=1 | 37.06 | 4.585 | 4.687 | 38.08 | 0.25 | +79.6% |
| mipple e5 λ=0 | 45.37 | 7.176 | 7.277 | 0.23 | 41.47 | +0.0% |
| mipple e5 λ=0.25 | 44.56 | 5.970 | 6.099 | 28.46 | 0.33 | +0.0% |
| mipple e5 λ=0.5 | 43.32 | 5.135 | 5.297 | 26.91 | 0.35 | +0.0% |
| mipple e5 λ=1 | 40.69 | 3.816 | 4.015 | 23.90 | 0.39 | +0.0% |
| mipple e5 λ=2 | 38.72 | 3.045 | 3.232 | 20.61 | 0.46 | +0.0% |
| mipple e5 λ=4 | 36.39 | 2.377 | 2.542 | 19.47 | 0.48 | +0.0% |
| mipple e5 λ=8 | 33.25 | 1.638 | 1.763 | 17.11 | 0.55 | +0.0% |

Base-encoder dominance: the cheapest mipple effort that is at least as good and as fast as each competitor config (✗ = none yet; the gap shows the best mipple trade-off):

| Competitor | PSNR | Mpix/s | Beaten by | Gap |
| --- | --- | --- | --- | --- |
| bc7e u6 | 46.01 | 0.92 | ✗ | best is mipple e5, 0.64 dB short |
| bc7enc_rdo bc7e u6 λ=0 | 46.01 | 1.09 | ✗ | best is mipple e5, 0.64 dB short |
| bc7enc_rdo bc7e u6 λ=0.05 | 45.80 | 0.16 | ✗ | best is mipple e5, 0.43 dB short |
| bc7enc_rdo bc7e u6 λ=0.1 | 45.39 | 0.17 | ✗ | best is mipple e5, 0.02 dB short |
| bc7enc_rdo bc7e u6 λ=0.25 | 43.80 | 0.17 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7e u6 λ=0.5 | 41.26 | 0.17 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7e u6 λ=1 | 37.80 | 0.18 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=0 | 43.83 | 5.73 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=0.05 | 43.66 | 0.21 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=0.1 | 43.31 | 0.22 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=0.25 | 41.99 | 0.22 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=0.5 | 39.90 | 0.23 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |
| bc7enc_rdo bc7enc.cpp -e λ=1 | 37.06 | 0.25 | mipple e5 (45.37 dB, 32.03 Mpix/s) | ✓ |

BD-rate of **mipple RDO** vs each RDO family (negative = mipple needs fewer bits at equal PSNR):

| vs family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| bc7enc_rdo bc7e | -31.5% | -29.2% |
| bc7enc_rdo bc7enc.cpp -e | -30.5% | -28.6% |
