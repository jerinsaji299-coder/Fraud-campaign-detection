# Stage 3.4.1 evidence: real windows through the real model

GINE 3x64 + edge readout MLP on `[h_src, h_dst, edge_features]`, bidirectional message
passing with a direction flag, class-weighted loss, Adam. One
GPU (Tesla T4, 15.6 GB). Real features from a
spec fitted on every rule-3 training window, as training fits
it.

Peaks are reported allocated **and** reserved; reserved is what
the allocator holds and therefore what runs out.

## Windows

| lookback_h | kind | t | n_edges | n_accounts | n_positive | class_weight | edge_dims | node_dims | matches_features_smoke | ok | mode | first_step_s | median_step_s | loss | peak_allocated_mb | peak_reserved_mb | mb_per_1k_edges | fits_full_batch_budget | seconds | n_scores |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | train | 2022-09-02 00:00:00 | 1114921 | 437073 | 322 | 3461.49 | 54 | 6 | True | True | full-batch train step | 0.517 | 0.276 | 10615.46582 | 4843.1 | 6387.9 | 5.729 | True | nan | nan |
| 48 | train | 2022-09-03 00:00:00 | 1869370 | 509993 | 730 | 2559.78 | 54 | 6 | True | True | full-batch train step | 0.506 | 0.445 | 356394.1875 | 7638.9 | 10701.8 | 5.725 | True | nan | nan |
| 24 | eval | 2022-09-10 00:00:00 | 654467 | 346684 | 514 | 1272.28 | 54 | 6 | True | True | full-window inference | nan | nan | nan | 1920.9 | 2705.3 | 4.134 | nan | 0.115 | 654467.0 |
| 48 | eval | 2022-09-10 00:00:00 | 1137240 | 354229 | 1053 | 1079.0 | 54 | 6 | True | True | full-window inference | nan | nan | nan | 3255.5 | 4750.0 | 4.177 | nan | 0.177 | 1137240.0 |

## Timing and host memory

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 25.5 | 3,902 MB | yes |
| scan every usable detection time | 37.0 | 3,902 MB | yes |
| fit feature spec on all training windows, L=24h | 30.2 | 10,644 MB | yes |
| build + featurise largest train window, L=24h | 5.5 | 10,644 MB | yes |
| move largest train window, L=24h to the GPU | 0.1 | 10,644 MB | yes |
| build + featurise largest eval window, L=24h | 3.5 | 10,644 MB | yes |
| move largest eval window, L=24h to the GPU | 0.0 | 10,644 MB | yes |
| fit feature spec on all training windows, L=48h | 39.3 | 14,136 MB | yes |
| build + featurise largest train window, L=48h | 8.7 | 14,136 MB | yes |
| move largest train window, L=48h to the GPU | 0.1 | 14,136 MB | yes |
| build + featurise largest eval window, L=48h | 5.1 | 14,136 MB | yes |
| move largest eval window, L=48h to the GPU | 0.1 | 14,136 MB | yes |
