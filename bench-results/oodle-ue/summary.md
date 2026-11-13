# mipple vs Oodle Texture 2.9.16 (Unreal Engine 5.8)

Oodle's blocks come from a normal UE 5.8.4 cook (Final encode speed, RDO on, lambda from Lossy Compression Amount; dumped with `-OodleDebugDumpFilter`, bit-exact with the cooked texture data). Both encoders are scored on the same source pixels, sizes are Zstd level 19 over the block data. Each cell is mipple's size at Oodle's quality relative to Oodle's size (negative = mipple smaller), from a mipple -e 5 lambda sweep. Lambda 0 is Oodle without RDO.

kodim18_original rows are from mipple 0.9.16 (smooth weight 5x, linear ramp to std 14): that 512x512 crop is not in the repo, so --scores cannot re-run it.

## BC1 Kodak-24 (24 images)

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.998 | 6.839 | 0.97219 | -18.9% | -9.4% |
| 1 | 2.908 | 6.816 | 0.97215 | -15.1% | -6.8% |
| 10 | 2.480 | 7.011 | 0.96588 | -11.5% | -7.5% |
| 20 | 2.249 | 7.233 | 0.96093 | -11.0% | -7.3% |
| 30 | 2.100 | 7.448 | 0.95707 | -11.3% | -6.9% |
| 40 | 1.986 | 7.659 | 0.95366 | -11.5% | -6.7% |

## BC1 kodim18_original

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.464 | 8.027 | 0.96750 | -7.3% | -5.3% |
| 1 | 3.420 | 7.994 | 0.96773 | -4.7% | -3.4% |
| 10 | 3.126 | 8.154 | 0.96192 | -1.8% | -4.8% |
| 20 | 2.898 | 8.381 | 0.95626 | -0.9% | -4.7% |
| 30 | 2.715 | 8.633 | 0.95150 | -0.8% | -3.7% |
| 40 | 2.564 | 8.908 | 0.94707 | -1.4% | -2.6% |

## BC4 Kodak-24 (24 images)

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.789 | 1.630 | 0.99674 | -13.4% | -5.4% |
| 1 | 3.389 | 1.626 | 0.99478 | -4.0% | -1.7% |
| 10 | 2.517 | 2.217 | 0.98089 | -6.6% | -2.1% |
| 20 | 2.251 | 2.673 | 0.97263 | -9.6% | -3.2% |
| 30 | 2.128 | 3.021 | 0.96737 | -12.6% | -4.3% |
| 40 | 2.046 | 3.306 | 0.96331 | -15.1% | -5.1% |

## BC4 kodim18_original

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.971 | 1.954 | 0.99598 | -5.2% | -1.5% |
| 1 | 3.896 | 1.871 | 0.99567 | n/a | -0.2% |
| 10 | 3.044 | 2.486 | 0.98092 | +2.8% | +4.8% |
| 20 | 2.744 | 3.015 | 0.97245 | +1.2% | +6.2% |
| 30 | 2.559 | 3.434 | 0.96592 | +0.2% | +6.7% |
| 40 | 2.432 | 3.788 | 0.96046 | -0.6% | +6.6% |

## BC5 Kodak-24 (24 images)

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.427 | 2.756 | 0.99713 | -31.1% | -11.3% |
| 1 | 5.691 | 2.687 | 0.99507 | -4.5% | -1.8% |
| 10 | 4.379 | 3.231 | 0.98590 | -10.7% | -4.2% |
| 20 | 4.048 | 3.672 | 0.98079 | -15.7% | -5.7% |
| 30 | 3.852 | 4.035 | 0.97723 | -19.0% | -6.3% |
| 40 | 3.710 | 4.346 | 0.97437 | -21.2% | -6.7% |

## BC5 kodim18_original

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.682 | 2.559 | 0.99705 | -28.9% | -8.7% |
| 1 | 5.584 | 2.516 | 0.99468 | +5.6% | +3.8% |
| 10 | 4.369 | 2.988 | 0.98818 | -7.3% | +2.3% |
| 20 | 4.054 | 3.388 | 0.98479 | -12.2% | +1.0% |
| 30 | 3.867 | 3.715 | 0.98229 | -14.8% | +0.4% |
| 40 | 3.723 | 4.023 | 0.98030 | -17.2% | +0.7% |

## BC7 Kodak-24 (24 images)

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.492 | 2.276 | 0.99571 | -6.5% (15/24) | -2.6% (4/24) |
| 1 | 7.261 | 2.266 | 0.99574 | -2.9% (12/24) | -0.9% (4/24) |
| 10 | 5.228 | 2.807 | 0.98897 | -1.9% | -3.9% |
| 20 | 4.306 | 3.354 | 0.98310 | -1.7% | -2.5% |
| 30 | 3.821 | 3.803 | 0.97855 | -2.3% | -1.8% |
| 40 | 3.510 | 4.188 | 0.97461 | -3.1% | -1.5% |

## BC7 kodim18_original

| Oodle λ | Oodle bpt | Oodle RMSE | Oodle SSIM | at equal RMSE | at equal SSIM |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.825 | 2.971 | 0.99426 | -3.9% | -2.3% |
| 1 | 7.759 | 2.961 | 0.99430 | -2.1% | -1.1% |
| 10 | 6.796 | 3.205 | 0.99050 | +0.1% | -6.5% |
| 20 | 5.819 | 3.749 | 0.98342 | +2.7% | -9.7% |
| 30 | 5.127 | 4.319 | 0.97743 | +3.2% | -7.4% |
| 40 | 4.683 | 4.789 | 0.97279 | +3.7% | -5.1% |

