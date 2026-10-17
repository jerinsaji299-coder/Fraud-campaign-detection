# Why the oracle misses campaigns

The oracle scores every edge with its true label, so a miss is
never a scoring failure. It is a property of the campaign's shape
in time, or of the matching rule meeting a graph that does not
cooperate. **Diagnosis only — the matching rule is unchanged.**

## Coverage

| lookback_h | split | n_campaigns | n_detected | n_missed | miss_rate |
|---|---|---|---|---|---|
| 24 | val | 26 | 19 | 7 | 0.2692 |
| 24 | test | 56 | 46 | 10 | 0.1786 |
| 24 | stress | 76 | 55 | 21 | 0.2763 |
| 48 | val | 26 | 20 | 6 | 0.2308 |
| 48 | test | 56 | 50 | 6 | 0.1071 |
| 48 | stress | 76 | 56 | 20 | 0.2632 |

## Paired lead-time comparison between lookbacks

Earliness is compared **only on the campaigns both lookbacks
detect**. Two median lead times cannot be compared directly:
each median is over a different set of campaigns, and the extra
campaigns a longer lookback reaches are the slow ones detected
late, so its median falls even when no campaign got later.
Coverage is a separate question, reported as the campaigns each
lookback finds that the other misses. A positive difference
means the second lookback was earlier.

| label_a | label_b | split | n_both | n_only_a | n_only_b | n_neither | only_a_campaigns | only_b_campaigns | median_difference_h | median_normalized_difference | median_frac_observed_difference | n_b_earlier | n_a_earlier | n_tied | ci_low_h | ci_high_h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L=24h | L=48h | val | 19 | 0 | 1 | 6 |  | 173 | 0.0 | 0.0 | 0.0 | 1 | 2 | 16 | 0.0 | 0.0 |
| L=24h | L=48h | test | 46 | 0 | 4 | 6 |  | 187,194,238,245 | 0.0 | 0.0 | 0.0 | 5 | 1 | 40 | 0.0 | 0.0 |
| L=24h | L=48h | stress | 54 | 1 | 2 | 19 | 295 | 261,282 | 0.0 | 0.0 | 0.0 | 1 | 0 | 53 | 0.0 | 0.0 |

- On the 19 campaigns both detect, L=48h - L=24h = 0.0h (95% CI [0.0, 0.0]); the interval includes zero, so no earliness difference is demonstrated. Separately, L=48h detects 1 that L=24h misses, and L=24h detects 0 that L=48h misses.
- On the 46 campaigns both detect, L=48h - L=24h = 0.0h (95% CI [0.0, 0.0]); the interval includes zero, so no earliness difference is demonstrated. Separately, L=48h detects 4 that L=24h misses, and L=24h detects 0 that L=48h misses.
- On the 54 campaigns both detect, L=48h - L=24h = 0.0h (95% CI [0.0, 0.0]); the interval includes zero, so no earliness difference is demonstrated. Separately, L=48h detects 2 that L=24h misses, and L=24h detects 1 that L=48h misses.

## Misses by reason

| reason | meaning |
|---|---|
| `temporal_sparsity` | at no window before the deadline did 3+ of its accounts connect through its own laundering edges, and none formed afterwards either |
| `purity_failure` | a 3+ fragment did form, but the cluster it sat in was under half the campaign's own accounts |
| `late_only` | a matching cluster existed, but only after the deadline — censoring, not sparsity |
| `unexplained` | none of the above; would mean the diagnosis and the evaluation disagree |

| lookback_h | split | reason | n |
|---|---|---|---|
| 24 | stress | temporal_sparsity | 21 |
| 24 | test | late_only | 2 |
| 24 | test | temporal_sparsity | 8 |
| 24 | val | temporal_sparsity | 7 |
| 48 | stress | purity_failure | 1 |
| 48 | stress | temporal_sparsity | 19 |
| 48 | test | late_only | 1 |
| 48 | test | temporal_sparsity | 5 |
| 48 | val | temporal_sparsity | 6 |

## By base_type

| lookback_h | base_type | reason | n |
|---|---|---|---|
| 24 | BIPARTITE | temporal_sparsity | 15 |
| 24 | CYCLE | late_only | 1 |
| 24 | CYCLE | temporal_sparsity | 5 |
| 24 | FAN-IN | temporal_sparsity | 2 |
| 24 | FAN-OUT | temporal_sparsity | 4 |
| 24 | GATHER-SCATTER | temporal_sparsity | 3 |
| 24 | RANDOM | late_only | 1 |
| 24 | RANDOM | temporal_sparsity | 2 |
| 24 | SCATTER-GATHER | temporal_sparsity | 1 |
| 24 | STACK | temporal_sparsity | 4 |
| 48 | BIPARTITE | temporal_sparsity | 15 |
| 48 | CYCLE | purity_failure | 1 |
| 48 | CYCLE | temporal_sparsity | 4 |
| 48 | FAN-IN | temporal_sparsity | 2 |
| 48 | FAN-OUT | temporal_sparsity | 2 |
| 48 | GATHER-SCATTER | temporal_sparsity | 3 |
| 48 | RANDOM | late_only | 1 |
| 48 | RANDOM | temporal_sparsity | 2 |
| 48 | SCATTER-GATHER | temporal_sparsity | 1 |
| 48 | STACK | temporal_sparsity | 1 |

## By group

| lookback_h | group | reason | n |
|---|---|---|---|
| 24 | fragmentable | late_only | 2 |
| 24 | fragmentable | temporal_sparsity | 27 |
| 24 | hub | temporal_sparsity | 9 |
| 48 | fragmentable | late_only | 1 |
| 48 | fragmentable | purity_failure | 1 |
| 48 | fragmentable | temporal_sparsity | 23 |
| 48 | hub | temporal_sparsity | 7 |

## Every missed campaign

| campaign_id | split | lookback_h | base_type | group | reason | detail | n_txn | n_accounts | duration_h | deadline | max_connected_own_accounts | max_txn_gap_h | gap_exceeds_lookback | best_cluster_purity | best_cluster_size | merged_with_campaigns | merged_with_unassigned | windows_checked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 | val | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 24 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 33.0h, largest gap 10.5h | 12 | 24 | 32.96666666666667 | 2022-09-06 23:49:00 | 2 | 10.47 | False | nan | nan | 0 | 0 | 7 |
| 162 | val | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 30 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 15 transactions over 34.5h, largest gap 7.1h | 15 | 30 | 34.53333333333333 | 2022-09-06 18:45:00 | 2 | 7.13 | False | nan | nan | 0 | 0 | 7 |
| 166 | val | 24 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 88.8h, largest gap 78.8h (exceeds L=24h) | 3 | 3 | 88.75 | 2022-09-09 05:17:00 | 2 | 78.83 | True | nan | nan | 0 | 0 | 12 |
| 167 | val | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 24.2h, largest gap 13.8h | 3 | 6 | 24.183333333333334 | 2022-09-06 17:45:00 | 2 | 13.77 | False | nan | nan | 0 | 0 | 6 |
| 171 | val | 24 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 68.1h, largest gap 65.1h (exceeds L=24h) | 3 | 3 | 68.06666666666666 | 2022-09-08 12:22:00 | 2 | 65.08 | True | nan | nan | 0 | 0 | 12 |
| 173 | val | 24 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 74.1h, largest gap 45.8h (exceeds L=24h) | 4 | 4 | 74.11666666666666 | 2022-09-08 19:18:00 | 2 | 45.8 | True | nan | nan | 0 | 0 | 12 |
| 180 | val | 24 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 7 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 6 transactions over 142.4h, largest gap 83.4h (exceeds L=24h) | 6 | 7 | 142.36666666666667 | 2022-09-11 00:00:00 | 2 | 83.42 | True | nan | nan | 0 | 0 | 12 |
| 179 | test | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 10 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 16.9h, largest gap 11.1h | 5 | 10 | 16.883333333333333 | 2022-09-07 12:02:00 | 2 | 11.12 | False | nan | nan | 0 | 0 | 6 |
| 187 | test | 24 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 62.0h, largest gap 34.0h (exceeds L=24h) | 3 | 4 | 62.03333333333333 | 2022-09-08 16:04:00 | 2 | 34.03 | True | nan | nan | 0 | 0 | 10 |
| 194 | test | 24 | STACK | fragmentable | temporal_sparsity | never more than 2 of its 9 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 6 transactions over 89.3h, largest gap 24.0h | 6 | 9 | 89.31666666666666 | 2022-09-10 06:42:00 | 2 | 23.98 | False | nan | nan | 0 | 0 | 17 |
| 203 | test | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 8 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 8.8h, largest gap 3.9h | 4 | 8 | 8.766666666666667 | 2022-09-07 16:39:00 | 2 | 3.88 | False | nan | nan | 0 | 0 | 6 |
| 206 | test | 24 | RANDOM | fragmentable | late_only | a matching cluster existed, but only after the deadline (2022-09-07 16:24:00); detection after the deadline counts as missed. Reached 2 connected accounts before the deadline, over 6 window(s) | 3 | 4 | 26.166666666666668 | 2022-09-07 16:24:00 | 2 | 23.48 | False | nan | nan | 0 | 0 | 6 |
| 233 | test | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 16 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 44.4h, largest gap 11.5h | 8 | 16 | 44.416666666666664 | 2022-09-09 05:28:00 | 2 | 11.53 | False | nan | nan | 0 | 0 | 12 |
| 238 | test | 24 | STACK | fragmentable | temporal_sparsity | never more than 2 of its 21 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 14 transactions over 95.3h, largest gap 16.6h | 14 | 21 | 95.3 | 2022-09-11 00:00:00 | 2 | 16.62 | False | nan | nan | 0 | 0 | 20 |
| 241 | test | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 26 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 39.1h, largest gap 13.7h | 13 | 26 | 39.06666666666667 | 2022-09-09 03:05:00 | 2 | 13.72 | False | nan | nan | 0 | 0 | 12 |
| 244 | test | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 24 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 47.4h, largest gap 15.5h | 12 | 24 | 47.43333333333333 | 2022-09-09 14:11:00 | 2 | 15.52 | False | nan | nan | 0 | 0 | 14 |
| 245 | test | 24 | CYCLE | fragmentable | late_only | a matching cluster existed, but only after the deadline (2022-09-09 03:58:00); detection after the deadline counts as missed. Reached 2 connected accounts before the deadline, over 12 window(s) | 3 | 3 | 35.6 | 2022-09-09 03:58:00 | 2 | 21.65 | False | nan | nan | 0 | 0 | 12 |
| 261 | stress | 24 | STACK | fragmentable | temporal_sparsity | never more than 2 of its 21 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 14 transactions over 79.6h, largest gap 17.4h | 14 | 21 | 79.61666666666666 | 2022-09-11 00:00:00 | 2 | 17.4 | False | nan | nan | 0 | 0 | 12 |
| 264 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 26 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 35.6h, largest gap 10.1h | 13 | 26 | 35.65 | 2022-09-09 16:53:00 | 2 | 10.07 | False | nan | nan | 0 | 0 | 6 |
| 277 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 42.8h, largest gap 20.8h | 9 | 18 | 42.8 | 2022-09-10 11:53:00 | 2 | 20.78 | False | nan | nan | 0 | 0 | 9 |
| 282 | stress | 24 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 8 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 7 transactions over 89.6h, largest gap 39.8h (exceeds L=24h) | 7 | 8 | 89.63333333333334 | 2022-09-11 00:00:00 | 2 | 39.83 | True | nan | nan | 0 | 0 | 12 |
| 290 | stress | 24 | RANDOM | fragmentable | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 85.2h, largest gap 56.9h (exceeds L=24h) | 5 | 6 | 85.16666666666667 | 2022-09-11 00:00:00 | 2 | 56.88 | True | nan | nan | 0 | 0 | 12 |
| 301 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 42.5h, largest gap 10.6h | 9 | 18 | 42.483333333333334 | 2022-09-11 00:00:00 | 2 | 10.57 | False | nan | nan | 0 | 0 | 12 |
| 307 | stress | 24 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 9 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 146.2h, largest gap 63.9h (exceeds L=24h) | 8 | 9 | 146.18333333333334 | 2022-09-11 00:00:00 | 2 | 63.88 | True | nan | nan | 0 | 0 | 12 |
| 313 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 8 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 23.3h, largest gap 20.0h | 4 | 8 | 23.283333333333335 | 2022-09-11 00:00:00 | 2 | 19.98 | False | nan | nan | 0 | 0 | 12 |
| 326 | stress | 24 | RANDOM | fragmentable | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 46.0h, largest gap 36.3h (exceeds L=24h) | 3 | 4 | 46.0 | 2022-09-11 00:00:00 | 2 | 36.28 | True | nan | nan | 0 | 0 | 12 |
| 336 | stress | 24 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 64.8h, largest gap 50.5h (exceeds L=24h) | 4 | 4 | 64.75 | 2022-09-11 00:00:00 | 2 | 50.47 | True | nan | nan | 0 | 0 | 12 |
| 337 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 12 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 6 transactions over 34.5h, largest gap 21.2h | 6 | 12 | 34.55 | 2022-09-11 00:00:00 | 2 | 21.25 | False | nan | nan | 0 | 0 | 12 |
| 339 | stress | 24 | FAN-IN | hub | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 60.5h, largest gap 34.9h (exceeds L=24h) | 5 | 6 | 60.46666666666667 | 2022-09-11 00:00:00 | 2 | 34.85 | True | nan | nan | 0 | 0 | 12 |
| 343 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 16 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 33.6h, largest gap 9.2h | 8 | 16 | 33.65 | 2022-09-11 00:00:00 | 2 | 9.2 | False | nan | nan | 0 | 0 | 12 |
| 347 | stress | 24 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 46.0h, largest gap 17.5h | 9 | 18 | 45.95 | 2022-09-11 00:00:00 | 2 | 17.47 | False | nan | nan | 0 | 0 | 12 |
| 352 | stress | 24 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 14 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 95.2h, largest gap 29.0h (exceeds L=24h) | 13 | 14 | 95.15 | 2022-09-11 00:00:00 | 2 | 29.03 | True | nan | nan | 0 | 0 | 12 |
| 353 | stress | 24 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 11 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 10 transactions over 95.7h, largest gap 23.9h | 10 | 11 | 95.71666666666667 | 2022-09-11 00:00:00 | 2 | 23.85 | False | nan | nan | 0 | 0 | 12 |
| 355 | stress | 24 | SCATTER-GATHER | fragmentable | temporal_sparsity | never more than 2 of its 9 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 14 transactions over 95.5h, largest gap 29.2h (exceeds L=24h) | 14 | 9 | 95.5 | 2022-09-11 00:00:00 | 2 | 29.23 | True | nan | nan | 0 | 0 | 12 |
| 358 | stress | 24 | STACK | fragmentable | temporal_sparsity | never more than 2 of its 15 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 10 transactions over 91.7h, largest gap 36.7h (exceeds L=24h) | 10 | 15 | 91.68333333333334 | 2022-09-11 00:00:00 | 2 | 36.68 | True | nan | nan | 0 | 0 | 12 |
| 359 | stress | 24 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 27 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 26 transactions over 183.8h, largest gap 24.8h (exceeds L=24h) | 26 | 27 | 183.83333333333334 | 2022-09-11 00:00:00 | 2 | 24.83 | True | nan | nan | 0 | 0 | 12 |
| 366 | stress | 24 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 66.6h, largest gap 40.5h (exceeds L=24h) | 3 | 3 | 66.56666666666666 | 2022-09-11 00:00:00 | 2 | 40.5 | True | nan | nan | 0 | 0 | 12 |
| 367 | stress | 24 | FAN-IN | hub | temporal_sparsity | never more than 2 of its 13 accounts connected via its own edges in any 24h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 94.8h, largest gap 22.5h | 12 | 13 | 94.81666666666666 | 2022-09-11 00:00:00 | 2 | 22.53 | False | nan | nan | 0 | 0 | 12 |
| 150 | val | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 24 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 33.0h, largest gap 10.5h | 12 | 24 | 32.96666666666667 | 2022-09-06 23:49:00 | 2 | 10.47 | False | nan | nan | 0 | 0 | 7 |
| 162 | val | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 30 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 15 transactions over 34.5h, largest gap 7.1h | 15 | 30 | 34.53333333333333 | 2022-09-06 18:45:00 | 2 | 7.13 | False | nan | nan | 0 | 0 | 7 |
| 166 | val | 48 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 88.8h, largest gap 78.8h (exceeds L=48h) | 3 | 3 | 88.75 | 2022-09-09 05:17:00 | 2 | 78.83 | True | nan | nan | 0 | 0 | 12 |
| 167 | val | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 24.2h, largest gap 13.8h | 3 | 6 | 24.183333333333334 | 2022-09-06 17:45:00 | 2 | 13.77 | False | nan | nan | 0 | 0 | 6 |
| 171 | val | 48 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 68.1h, largest gap 65.1h (exceeds L=48h) | 3 | 3 | 68.06666666666666 | 2022-09-08 12:22:00 | 2 | 65.08 | True | nan | nan | 0 | 0 | 12 |
| 180 | val | 48 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 7 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 6 transactions over 142.4h, largest gap 83.4h (exceeds L=48h) | 6 | 7 | 142.36666666666667 | 2022-09-11 00:00:00 | 2 | 83.42 | True | nan | nan | 0 | 0 | 12 |
| 179 | test | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 10 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 16.9h, largest gap 11.1h | 5 | 10 | 16.883333333333333 | 2022-09-07 12:02:00 | 2 | 11.12 | False | nan | nan | 0 | 0 | 6 |
| 203 | test | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 8 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 8.8h, largest gap 3.9h | 4 | 8 | 8.766666666666667 | 2022-09-07 16:39:00 | 2 | 3.88 | False | nan | nan | 0 | 0 | 6 |
| 206 | test | 48 | RANDOM | fragmentable | late_only | a matching cluster existed, but only after the deadline (2022-09-07 16:24:00); detection after the deadline counts as missed. Reached 2 connected accounts before the deadline, over 6 window(s) | 3 | 4 | 26.166666666666668 | 2022-09-07 16:24:00 | 2 | 23.48 | False | nan | nan | 0 | 0 | 6 |
| 233 | test | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 16 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 44.4h, largest gap 11.5h | 8 | 16 | 44.416666666666664 | 2022-09-09 05:28:00 | 2 | 11.53 | False | nan | nan | 0 | 0 | 12 |
| 241 | test | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 26 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 39.1h, largest gap 13.7h | 13 | 26 | 39.06666666666667 | 2022-09-09 03:05:00 | 2 | 13.72 | False | nan | nan | 0 | 0 | 12 |
| 244 | test | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 24 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 47.4h, largest gap 15.5h | 12 | 24 | 47.43333333333333 | 2022-09-09 14:11:00 | 2 | 15.52 | False | nan | nan | 0 | 0 | 14 |
| 264 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 26 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 35.6h, largest gap 10.1h | 13 | 26 | 35.65 | 2022-09-09 16:53:00 | 2 | 10.07 | False | nan | nan | 0 | 0 | 6 |
| 277 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 42.8h, largest gap 20.8h | 9 | 18 | 42.8 | 2022-09-10 11:53:00 | 2 | 20.78 | False | nan | nan | 0 | 0 | 9 |
| 290 | stress | 48 | RANDOM | fragmentable | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 85.2h, largest gap 56.9h (exceeds L=48h) | 5 | 6 | 85.16666666666667 | 2022-09-11 00:00:00 | 2 | 56.88 | True | nan | nan | 0 | 0 | 12 |
| 295 | stress | 48 | CYCLE | fragmentable | purity_failure | reached 5 connected accounts, but the best cluster was 0.46 its own accounts — merged with 9 other campaign(s) | 7 | 7 | 84.65 | 2022-09-11 00:00:00 | 5 | 21.12 | False | 0.4615 | 13.0 | 9 | 0 | 12 |
| 301 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 42.5h, largest gap 10.6h | 9 | 18 | 42.483333333333334 | 2022-09-11 00:00:00 | 2 | 10.57 | False | nan | nan | 0 | 0 | 12 |
| 307 | stress | 48 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 9 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 146.2h, largest gap 63.9h (exceeds L=48h) | 8 | 9 | 146.18333333333334 | 2022-09-11 00:00:00 | 2 | 63.88 | True | nan | nan | 0 | 0 | 12 |
| 313 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 8 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 23.3h, largest gap 20.0h | 4 | 8 | 23.283333333333335 | 2022-09-11 00:00:00 | 2 | 19.98 | False | nan | nan | 0 | 0 | 12 |
| 326 | stress | 48 | RANDOM | fragmentable | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 46.0h, largest gap 36.3h | 3 | 4 | 46.0 | 2022-09-11 00:00:00 | 2 | 36.28 | False | nan | nan | 0 | 0 | 12 |
| 336 | stress | 48 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 4 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 4 transactions over 64.8h, largest gap 50.5h (exceeds L=48h) | 4 | 4 | 64.75 | 2022-09-11 00:00:00 | 2 | 50.47 | True | nan | nan | 0 | 0 | 12 |
| 337 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 12 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 6 transactions over 34.5h, largest gap 21.2h | 6 | 12 | 34.55 | 2022-09-11 00:00:00 | 2 | 21.25 | False | nan | nan | 0 | 0 | 12 |
| 339 | stress | 48 | FAN-IN | hub | temporal_sparsity | never more than 2 of its 6 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 5 transactions over 60.5h, largest gap 34.9h | 5 | 6 | 60.46666666666667 | 2022-09-11 00:00:00 | 2 | 34.85 | False | nan | nan | 0 | 0 | 12 |
| 343 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 16 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 8 transactions over 33.6h, largest gap 9.2h | 8 | 16 | 33.65 | 2022-09-11 00:00:00 | 2 | 9.2 | False | nan | nan | 0 | 0 | 12 |
| 347 | stress | 48 | BIPARTITE | fragmentable | temporal_sparsity | never more than 2 of its 18 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 9 transactions over 46.0h, largest gap 17.5h | 9 | 18 | 45.95 | 2022-09-11 00:00:00 | 2 | 17.47 | False | nan | nan | 0 | 0 | 12 |
| 352 | stress | 48 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 14 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 13 transactions over 95.2h, largest gap 29.0h | 13 | 14 | 95.15 | 2022-09-11 00:00:00 | 2 | 29.03 | False | nan | nan | 0 | 0 | 12 |
| 353 | stress | 48 | FAN-OUT | hub | temporal_sparsity | never more than 2 of its 11 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 10 transactions over 95.7h, largest gap 23.9h | 10 | 11 | 95.71666666666667 | 2022-09-11 00:00:00 | 2 | 23.85 | False | nan | nan | 0 | 0 | 12 |
| 355 | stress | 48 | SCATTER-GATHER | fragmentable | temporal_sparsity | never more than 2 of its 9 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 14 transactions over 95.5h, largest gap 29.2h | 14 | 9 | 95.5 | 2022-09-11 00:00:00 | 2 | 29.23 | False | nan | nan | 0 | 0 | 12 |
| 358 | stress | 48 | STACK | fragmentable | temporal_sparsity | never more than 2 of its 15 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 10 transactions over 91.7h, largest gap 36.7h | 10 | 15 | 91.68333333333334 | 2022-09-11 00:00:00 | 2 | 36.68 | False | nan | nan | 0 | 0 | 12 |
| 359 | stress | 48 | GATHER-SCATTER | hub | temporal_sparsity | never more than 2 of its 27 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 26 transactions over 183.8h, largest gap 24.8h | 26 | 27 | 183.83333333333334 | 2022-09-11 00:00:00 | 2 | 24.83 | False | nan | nan | 0 | 0 | 12 |
| 366 | stress | 48 | CYCLE | fragmentable | temporal_sparsity | never more than 2 of its 3 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 3 transactions over 66.6h, largest gap 40.5h | 3 | 3 | 66.56666666666666 | 2022-09-11 00:00:00 | 2 | 40.5 | False | nan | nan | 0 | 0 | 12 |
| 367 | stress | 48 | FAN-IN | hub | temporal_sparsity | never more than 2 of its 13 accounts connected via its own edges in any 48h window before the deadline, and no matching cluster formed afterwards either; 12 transactions over 94.8h, largest gap 22.5h | 12 | 13 | 94.81666666666666 | 2022-09-11 00:00:00 | 2 | 22.53 | False | nan | nan | 0 | 0 | 12 |

## What the false alarms are

| t | split | lookback_h | cluster_size | accounts_in_some_campaign | campaigns_touched | accounts_unassigned_laundering | best_overlap_fraction |
|---|---|---|---|---|---|---|---|
| 2022-09-09 00:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 00:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 18:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | test | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 00:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 00:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 18:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | stress | 24 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-08 12:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 00:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 12:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 12:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 18:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 18:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 00:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 00:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 06:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 06:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 18:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | test | 48 | 25 | 25 | 13 | 0 | 0.48 |
| 2022-09-11 00:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | test | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-08 12:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 00:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 06:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 12:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 12:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 18:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-09 18:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 00:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 00:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 06:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 06:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-10 18:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | stress | 48 | 25 | 25 | 13 | 0 | 0.48 |
| 2022-09-11 00:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |
| 2022-09-11 00:00:00 | stress | 48 | 3 | 3 | 2 | 0 | 0.6667 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 38.8 | 3,159 MB | yes |
| oracle on val, L=24h | 2.6 | 3,159 MB | yes |
| diagnose 7 misses on val, L=24h | 10.7 | 3,159 MB | yes |
| characterise false alarms on val, L=24h | 1.5 | 3,159 MB | yes |
| oracle on test, L=24h | 4.0 | 3,159 MB | yes |
| diagnose 10 misses on test, L=24h | 24.1 | 3,159 MB | yes |
| characterise false alarms on test, L=24h | 2.8 | 3,159 MB | yes |
| oracle on stress, L=24h | 2.3 | 3,159 MB | yes |
| diagnose 21 misses on stress, L=24h | 35.7 | 3,159 MB | yes |
| characterise false alarms on stress, L=24h | 1.8 | 3,159 MB | yes |
| oracle on val, L=48h | 3.6 | 3,159 MB | yes |
| diagnose 6 misses on val, L=48h | 15.0 | 3,159 MB | yes |
| characterise false alarms on val, L=48h | 2.5 | 3,159 MB | yes |
| oracle on test, L=48h | 6.7 | 3,159 MB | yes |
| diagnose 6 misses on test, L=48h | 25.4 | 3,159 MB | yes |
| characterise false alarms on test, L=48h | 5.0 | 3,159 MB | yes |
| oracle on stress, L=48h | 4.4 | 3,159 MB | yes |
| diagnose 20 misses on stress, L=48h | 57.0 | 3,159 MB | yes |
| characterise false alarms on stress, L=48h | 3.0 | 3,159 MB | yes |
