Corpus: 24 images from `/tmp/claude-0/-home-claude/34355fae-205f-5cbf-ab6d-85cea8d05a60/scratchpad/kodak`. Sizes are corpus totals after each back end; PSNR/SSIM are per-image means; time is total encode seconds.

| Encoder config | Mean PSNR (dB) | Mean SSIM | Zstd-19 bpt | Deflate-9 bpt | Time (s) | Mpix/s |
| --- | --- | --- | --- | --- | --- | --- |
| mipple e1 λ=0.0 | 45.03 | 0.9973 | 7.266 | 7.358 | 9.43 | 1.00 |
| mipple e1 λ=0.25 | 44.81 | 0.9970 | 6.914 | 6.994 | 53.18 | 0.18 |
| mipple e1 λ=0.5 | 44.25 | 0.9960 | 6.454 | 6.533 | 54.55 | 0.17 |
| mipple e1 λ=1.0 | 43.00 | 0.9927 | 5.703 | 5.792 | 56.79 | 0.17 |
| mipple e1 λ=2.0 | 41.26 | 0.9862 | 4.919 | 5.032 | 58.03 | 0.16 |
| mipple e1 λ=4.0 | 39.91 | 0.9807 | 4.506 | 4.638 | 59.22 | 0.16 |
| mipple e1 λ=8.0 | 39.06 | 0.9768 | 4.364 | 4.497 | 58.82 | 0.16 |
| bc7enc_rdo u4 λ=0.0 | 45.33 | 0.9975 | 7.194 | 7.296 | 7.82 | 1.21 |
| bc7enc_rdo u4 λ=0.05 | 45.14 | 0.9973 | 7.084 | 7.177 | 182.69 | 0.05 |
| bc7enc_rdo u4 λ=0.1 | 44.75 | 0.9968 | 6.902 | 6.989 | 185.05 | 0.05 |
| bc7enc_rdo u4 λ=0.25 | 43.30 | 0.9944 | 6.429 | 6.503 | 179.81 | 0.05 |
| bc7enc_rdo u4 λ=0.5 | 40.94 | 0.9895 | 5.890 | 5.953 | 177.13 | 0.05 |
| bc7enc_rdo u4 λ=1.0 | 37.59 | 0.9788 | 5.253 | 5.307 | 174.35 | 0.05 |
| bc7enc_rdo -C -e λ=0.0 | 43.83 | 0.9967 | 6.408 | 6.579 | 8.20 | 1.15 |
| bc7enc_rdo -C -e λ=0.05 | 43.66 | 0.9965 | 6.315 | 6.472 | 173.38 | 0.05 |
| bc7enc_rdo -C -e λ=0.1 | 43.31 | 0.9959 | 6.158 | 6.305 | 173.10 | 0.05 |
| bc7enc_rdo -C -e λ=0.25 | 41.98 | 0.9934 | 5.709 | 5.840 | 168.21 | 0.06 |
| bc7enc_rdo -C -e λ=0.5 | 39.89 | 0.9882 | 5.170 | 5.288 | 161.25 | 0.06 |
| bc7enc_rdo -C -e λ=1.0 | 37.03 | 0.9777 | 4.568 | 4.670 | 150.62 | 0.06 |

BD-rate vs **bc7enc_rdo** (negative = fewer bits at equal PSNR):

| Family | Zstd-19 | Deflate-9 |
| --- | --- | --- |
| mipple | -13.4% | -12.6% |
| bc7enc_rdo -e | -7.9% | -6.7% |
