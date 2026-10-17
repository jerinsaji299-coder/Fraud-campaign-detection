# Pipeline run

- artifacts: `/kaggle/temp/repo/backend/artifacts`
- archive: `/kaggle/working/artifacts.zip`
- rows: 5,078,345

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
| Unfragmentable campaigns | 2 | 2 | PASS | Kaggle, commit 55b645e, 2 sessions x 3 processes, 2026-10-09 |

## Visibility file digests (SHA-256)

Compare these across two independent sessions; the seeded
search must reproduce identical bytes.

| File | SHA-256 |
|---|---|
| `seed_0.parquet` | `8953a9d77bf694e72e18fada8bd3065dfc06b4b1589e327dee6901c8adb9b0ff` |
| `seed_1.parquet` | `36a322ad1cfc9eadd2a4c554a9497a11ac4c9aa6936071b76acc1dc87ff56204` |
| `seed_2.parquet` | `d5d519ecb24001f853de2eaee2928813dda308fb5a96894d2d98bd0ca3eaed16` |
| `seed_3.parquet` | `f56ad7e33736a9824630f472a4fd090625e24194083bfde19a0ef27fcf3643bc` |
| `seed_4.parquet` | `df3139abbc850b46ecf2d499b31d55f47bffdb05ab53fc8bbdac29cd9d5ac63a` |

## Timing

| Step | Seconds | Memory | True peak |
|---|---:|---:|---|
| build pipeline | 33.2 | 3,160 MB | yes |
| export artifacts | 18.6 | 3,160 MB | yes |
| zip artifacts | 0.1 | 3,160 MB | yes |
