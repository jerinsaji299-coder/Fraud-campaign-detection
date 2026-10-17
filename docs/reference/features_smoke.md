# Feature smoke test on the full dataset

- edge features: 54 dims (15 payment currencies, 15 receiving, 7 formats)
- node features: 6 dims
- lookback candidates: [24, 48]; excluded but measured: [72]
- spec fitted on: `{'n_windows': 1, 'n_transactions': 1114921, 'first_t': '2022-09-02 00:00:00', 'last_t': '2022-09-02 00:00:00', 'max_timestamp_seen': '2022-09-01 23:59:00'}`

## Usable training detection times (rule 3: full lookback)

Unfiltered training grid: 16.

| Lookback | Usable | Dropped |
|---|---:|---:|
| 24h | 13 | 3 |
| 48h | 9 | 7 |
| 72h | 5 | 11 |

## Window sizes across ALL usable detection times

Counts only, no features. This is the authoritative source for
the largest window: sampling a few detection times is not
enough to find it.

| lookback_h | excluded | split | n_times | edges_min | edges_median | edges_max | accounts_min | accounts_median | accounts_max | laundering_max | t_of_max_edges | accounts_at_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | False | stress | 12 | 208325 | 482564 | 654467 | 148023 | 232221 | 346684 | 545 | 2022-09-10 00:00:00 | 346684 |
| 24 | False | test | 20 | 208325 | 482335 | 654467 | 148023 | 205484 | 346684 | 545 | 2022-09-10 00:00:00 | 346684 |
| 24 | False | train | 13 | 207382 | 479911 | 1114921 | 125564 | 258492 | 437073 | 423 | 2022-09-02 00:00:00 | 437073 |
| 24 | False | val | 12 | 276521 | 482003 | 483199 | 159302 | 204775 | 225074 | 531 | 2022-09-07 18:00:00 | 205184 |
| 48 | False | stress | 12 | 862792 | 983556 | 1137240 | 234893 | 311538 | 354635 | 1053 | 2022-09-10 00:00:00 | 354229 |
| 48 | False | test | 20 | 758835 | 964789 | 1137240 | 234609 | 235273 | 354635 | 1053 | 2022-09-10 00:00:00 | 354229 |
| 48 | False | train | 9 | 414812 | 961831 | 1869370 | 162388 | 360196 | 509993 | 822 | 2022-09-03 00:00:00 | 509993 |
| 48 | False | val | 12 | 484761 | 862118 | 965117 | 186298 | 234878 | 235299 | 1029 | 2022-09-07 18:00:00 | 235038 |
| 72 | True | stress | 12 | 1345565 | 1465601 | 1619991 | 244309 | 318769 | 361685 | 1567 | 2022-09-10 00:00:00 | 360879 |
| 72 | True | test | 20 | 967075 | 1446300 | 1619991 | 244043 | 244597 | 361685 | 1567 | 2022-09-10 00:00:00 | 360879 |
| 72 | True | train | 5 | 1169261 | 1475014 | 2076752 | 366440 | 414201 | 510772 | 1217 | 2022-09-04 00:00:00 | 510772 |
| 72 | True | val | 12 | 897462 | 1102819 | 1447490 | 244043 | 244511 | 331819 | 1499 | 2022-09-08 00:00:00 | 244344 |

## Featurised sample windows

Training probes that would violate rule 3 are skipped, not
silently clamped to a shorter window.

| split | day | t | lookback_h | excluded | n_edges | n_accounts | n_laundering | laundering_rate | class_weight | slice_s | build_s | tensors_mb | xgb_matrix_mb | peak_rss_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | weekday (Fri) | 2022-09-02 12:00 | 24 | False | 787448 | 340795 | 341 | 0.000433 | 2308.2 | 0.2 | 6.15 | 197.2 | 207.9 | 3484.0 |
| train | weekend (Sun) | 2022-09-04 12:00 | 24 | False | 207655 | 125564 | 395 | 0.001902 | 524.7 | 0.08 | 1.84 | 52.9 | 54.8 | 3484.0 |
| train | weekend (Sun) | 2022-09-04 12:00 | 48 | False | 687566 | 278755 | 818 | 0.00119 | 839.5 | 0.18 | 5.05 | 171.7 | 181.5 | 3484.0 |
| train | weekend (Sun) | 2022-09-04 12:00 | 72 | True | 1475014 | 414201 | 1159 | 0.000786 | 1271.7 | 0.31 | 10.58 | 363.9 | 389.4 | 4423.0 |
| val | weekday (Mon) | 2022-09-05 12:00 | 24 | False | 346325 | 178526 | 420 | 0.001213 | 823.6 | 0.14 | 2.64 | 87.4 | 91.4 | 4423.0 |
| val | weekday (Mon) | 2022-09-05 12:00 | 48 | False | 553980 | 205798 | 815 | 0.001471 | 678.7 | 0.15 | 3.75 | 137.9 | 146.3 | 4423.0 |
| val | weekday (Mon) | 2022-09-05 12:00 | 72 | True | 1033891 | 298589 | 1238 | 0.001197 | 834.1 | 0.24 | 6.93 | 255.3 | 272.9 | 4423.0 |
| test | weekday (Wed) | 2022-09-07 12:00 | 24 | False | 482398 | 197516 | 519 | 0.001076 | 928.5 | 0.14 | 3.32 | 120.5 | 127.4 | 4423.0 |
| test | weekday (Wed) | 2022-09-07 12:00 | 48 | False | 964040 | 234810 | 1020 | 0.001058 | 944.1 | 0.22 | 5.69 | 237.0 | 254.5 | 4423.0 |
| test | weekday (Wed) | 2022-09-07 12:00 | 72 | True | 1310365 | 244411 | 1440 | 0.001099 | 909.0 | 0.27 | 7.46 | 320.4 | 345.9 | 4423.0 |
| test | weekend (Sat) | 2022-09-10 12:00 | 24 | False | 431971 | 250377 | 492 | 0.001139 | 877.0 | 0.13 | 3.25 | 109.7 | 114.0 | 4423.0 |
| test | weekend (Sat) | 2022-09-10 12:00 | 48 | False | 1001589 | 354635 | 1020 | 0.001018 | 981.0 | 0.22 | 6.56 | 248.9 | 264.4 | 4423.0 |
| test | weekend (Sat) | 2022-09-10 12:00 | 72 | True | 1483590 | 361685 | 1522 | 0.001026 | 973.8 | 0.31 | 8.91 | 364.7 | 391.7 | 4423.0 |

## Estimated full-batch GPU memory

3-layer GINEConv, hidden 64, from the true largest valid window
per lookback. Estimated from tensor shapes, not measured.

| lookback_h | excluded | scope | split_of_max | t_of_max | n_edges | n_accounts | forward_activations_mb | with_backward_mb | recommended_headroom_mb |
|---|---|---|---|---|---|---|---|---|---|
| 24 | False | inference (any split) | train | 2022-09-02 00:00:00 | 1114921 | 437073 | 4067.9 | 8135.9 | 10576.6 |
| 24 | False | training | train | 2022-09-02 00:00:00 | 1114921 | 437073 | 4067.9 | 8135.9 | 10576.6 |
| 48 | False | inference (any split) | train | 2022-09-03 00:00:00 | 1869370 | 509993 | 6301.9 | 12603.7 | 16384.8 |
| 48 | False | training | train | 2022-09-03 00:00:00 | 1869370 | 509993 | 6301.9 | 12603.7 | 16384.8 |
| 72 | True | inference (any split) | train | 2022-09-04 00:00:00 | 2076752 | 510772 | 6871.1 | 13742.1 | 17864.8 |
| 72 | True | training | train | 2022-09-04 00:00:00 | 2076752 | 510772 | 6871.1 | 13742.1 | 17864.8 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 32.4 | 3,159 MB | yes |
| fit feature spec on the 2022-09-02 00:00:00 24h training window | 7.4 | 3,484 MB | yes |
| scan every usable detection time (counts only, no features) | 101.4 | 3,484 MB | yes |
