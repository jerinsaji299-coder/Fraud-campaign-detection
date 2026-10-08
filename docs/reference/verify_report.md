# Phase 2 verification on the full dataset

- pytest exit code: `0`
- pandas: `2.3.3`
- transactions file: `/kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv`
- date range: `['2022-09-01T00:00:00', '2022-09-18T16:18:00']`
- institution loads (K=8): `[1269310, 1269310, 1269309, 1269309, 1269309, 1269309, 1269309, 1269309]`
- group counts after the unfragmentable relabel: `{'fragmentable': 228, 'hub': 139, 'unfragmentable': 3}`
- unfragmentable campaigns: **3**

## Regression numbers

| Metric | Expected | Actual | Result | Evidence |
|---|---:|---:|---|---|
| Total transaction rows | 5,078,345 | 5,078,345 | PASS | notebook cell b42159f4 |
| Rows on/after CUTOFF (2022-09-11) | 1,108 | 1,108 | PASS | notebook cell 280d9179 |
| Campaigns in the pattern file | 370 | 370 | PASS | notebook cell cd70c51d |
| Pattern-file transactions | 3,209 | 3,209 | PASS | notebook cell cd70c51d |
| Pattern transactions matched into the transaction table | 3,209 | 3,209 | PASS | notebook cell cd70c51d (0 unmatched) |
| Laundering rows (Is Laundering == 1) | 5,177 | 5,177 | PASS | notebook cell b42159f4 |
| Unassigned laundering rows (label 1, no campaign) | 1,968 | 1,968 | PASS | README decision log (5,177 - 3,209) |
| eval_ok campaigns | 258 | 258 | PASS | notebook cell 280d9179 |
| eval_ok campaigns in train | 100 | 100 | PASS | notebook cell b7e46dec |
| eval_ok campaigns in val | 26 | 26 | PASS | notebook cell b7e46dec |
| eval_ok campaigns in test | 56 | 56 | PASS | notebook cell b7e46dec |
| eval_ok campaigns in stress | 76 | 76 | PASS | notebook cell b7e46dec |
| eval_ok fragmentable (reassignable) campaigns | 152 | 152 | PASS | notebook cell b7e46dec |
| eval_ok fragmentable campaigns in test | 37 | 37 | PASS | notebook cell b7e46dec |
| Fragmentable test campaigns sharing a train account | 7 | 7 | PASS | notebook cell cb0caa40 |
| base_type CYCLE | 54 | 54 | PASS | notebook cell 11875435 |
| base_type GATHER-SCATTER | 51 | 51 | PASS | notebook cell 11875435 |
| base_type BIPARTITE | 49 | 49 | PASS | notebook cell 11875435 |
| base_type FAN-OUT | 48 | 48 | PASS | notebook cell 11875435 |
| base_type SCATTER-GATHER | 44 | 44 | PASS | notebook cell 11875435 |
| base_type STACK | 43 | 43 | PASS | notebook cell 11875435 |
| base_type RANDOM | 41 | 41 | PASS | notebook cell 11875435 |
| base_type FAN-IN | 40 | 40 | PASS | notebook cell 11875435 |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| pytest -m fulldata | 39.3 | 3,207 MB | yes |

**All regression numbers match.**
