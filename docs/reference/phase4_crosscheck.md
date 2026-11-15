# Phase 4 cross-check: every README figure, its source, and the command

One table. Every figure the README states for Phase 4, the file and row it
comes from, and the command that reproduces it.

**The one command is:**

```
cd backend
python scripts/verify_phase4_crosscheck.py
```

It re-derives every **committed** figure from its source file and prints PASS
or FAIL per figure with the value it recomputed. Exit 0 if all match. Add
`--json` for the machine-readable form.

**Current state: 70 passed, 0 failed, 9 skipped.** The nine skipped are the
only Phase 4 figures in the README that are **not** reproducible from a
committed file — they are listed in their own section at the bottom, and they
are the things to re-check when their evidence lands. A skipped figure is one
the README asserts on trust; that is a smaller list than the passing one on
purpose.

Last updated 2026-11-15.

---

## 1. The centralized comparator and the seed-noise floor (O1)

Source for all of these: **`docs/reference/evaluate_baseline.csv`**, rows with
`split=test`, `variant=xgboost_centralized`, `lookback_h=48`,
`test_subset=all`. Phase 3's seeds are **model** seeds.

| Figure | Value | How to locate it |
|---|---:|---|
| centralized recall, per model seed | 0.696 / 0.661 / 0.696 / 0.661 / 0.679 | `campaign_recall`, sorted by `seed` |
| centralized relative, per model seed | 0.78 / 0.74 / 0.78 / 0.74 / 0.76 | `recall_relative_to_ceiling`, sorted by `seed` |
| **Phase 3 headline** (five-seed mean) | **0.679 / 0.760** | mean of the five rows |
| **Phase 4 comparator** (`matched_seed0`) | **0.696 / 0.780** | the `seed == 0` row alone |
| rejected: three seeds as model seeds | 0.6845 / 0.7667 | mean of `seed <= 2` |
| centralized ceiling at L = 48 | 0.8928571 (= 50/56) | `oracle_recall`; `nunique() == 1` across seeds |
| tau chosen | 0.99 | `tau`, identical at every seed |
| **model-seed spread** | 0.0357 recall, 0.040 relative | `max - min` of the two columns |
| the spread as campaigns | **2.00 of 56** | `0.0357 × n_campaigns` |
| spread ÷ (union → centralized gap) | 1/7 to 1/4.5 | `0.040 / (0.780 − 0.600)` and `/ (0.780 − 0.500)` |
| spread ÷ (lone institution → centralized gap) | ~1/11 to ~1/10 | `0.040 / (0.780 − 0.369)` and `/ (0.780 − 0.306)` |
| room for 4.2 against the matched comparator | **0.18 – 0.28** | `0.780 − 0.600` and `0.780 − 0.500` |

**The two centralized numbers are both correct and measure different things.**
0.679 / 0.760 is the five-model-seed mean and remains the Phase 3 baseline
headline. 0.696 / 0.780 is model seed 0, the Phase 4 comparator, because every
federated booster is model seed 0 and Phase 4 studies fragmentation rather
than model seed. See the README section "The two centralized numbers, and why
both are right".

## 2. Stage 4.1 session A — the 96 boosters

Source: **`docs/reference/train_isolated.csv`** (96 rows), with
`docs/reference/train_isolated.{md,json}` beside it.

| Figure | Value | How to locate it |
|---|---:|---|
| institution-cells | 96 | `len(frame)` |
| cells at `tier == normal` | 96 of 96 | `(tier == "normal").sum()` |
| untrained cells | 0 | `(~trained).sum()` |
| thinnest cell | 1,003 positives | `n_positive.min()` |
| thinnest cell identity | target 1.0, seed 1, institution 2 | the row at `n_positive.idxmin()` |
| mean positives per cell | 1,598.1 | `n_positive.mean()` |
| mean rows per cell | 1,999,169 | `n_rows.mean()` |
| class weights | 435.23 – 4,014.69 | `class_weight_min.min()`, `class_weight_max.max()` |
| windows per cell | 9 | `n_windows`, every row |
| booster time | 83.8s mean, 79.1 – 111.1s | `train_seconds` mean/min/max |
| cross-institution spread, full data | 2.42 / 1.93 / 1.89 / 1.91 | `max/min` of `n_positive` within each `visibility_target` |
| the three fragmented targets' span | 2.12% | `max/min` of those last three ratios |
| target 1.0 above the three | 25.1% and 27.8% (quoted as "25-28%") | `2.42/1.93 − 1`, `2.42/1.89 − 1` |
| institution 0 vs the cross-institution mean | **1.442× positives on 1.020× rows** | `groupby("institution").sum()`, index 0 ÷ mean |
| institution 0 tops the positives in | **12 of 12 cells** | per `(target, seed)`, `argmax` of `n_positive` |

Timing figures quoted in prose and not in the CSV — 2.25 h total, 8,094.0s
instrumented against 8,130s in the log, 5,353 MB peak — come from
`docs/reference/train_isolated.json`'s timing block and the session log.

## 3. The tau grid's top, and why it cannot be extended

Source: **`fraudcamp.evaluation.MODEL_TAU_GRID`** plus IEEE 754 binary32.

| Figure | Value | How |
|---|---:|---|
| grid points | 22 | `len(MODEL_TAU_GRID)` |
| grid top | 0.9999999 | `grid[-1]` |
| 1.0 excluded | yes | `1.0 not in grid` |
| float32 values strictly between the top and 1.0 | **exactly 1** | `np.nextafter` walk from `grid[-1]` to 1.0 |
| that value | 1 − 6.0e-8 | the first `nextafter` step |
| 1 − 1e-8 in float32 | **is 1.0** | `np.float32(1 - 1e-8)` |

So the one excluded cell in session B is a **representability** limit, not a
short grid.

## 4. The 1% sample figures — development only

These are regenerable but **not committed**, because artifacts are not
committed. The verifier checks them when the artifacts are present locally and
reports them as skipped otherwise.

| Figure | Value | Source (uncommitted) | Command |
|---|---:|---|---|
| stage 4.0 spread at L = 48 | 3.13 / 2.03 / 1.76 / 1.60 | `backend/artifacts/federation_dev/institution_split.csv` | `python scripts/run_institution_split.py --config configs/local.yaml` |
| stage 4.1 spread, seed 0 | 2.01 / 1.86 / 1.44 / 1.53 | `backend/artifacts/isolated_dev/phase4_isolated_institutions.csv` | `python scripts/run_isolated.py --config configs/local.yaml` |
| institution composition, bank 70 at 33.6% of institution 0's load | — | `backend/artifacts/federation_dev/institution_composition.csv` | `python scripts/run_institution_composition.py --config configs/local.yaml` |
| chain ratios: fedavg 1.41×, evidence 1.15×, fedavg_evidence 0.44×, evidence training 4.2× | — | `backend/artifacts/phase4_dev/phase4_chain.json` | `python scripts/run_phase4_chain.py --config configs/local.yaml` |
| ceilings agree across all four conditions | max disagreement **0.0** | same | same |
| union ceiling equals centralized ceiling | difference **0.0** | same | same |

**A caution the README repeats and this table restates:** the sample thins the
background to 1%, so its positive rate is ~3.3% against the full data's
0.102%. Precision, false alarms per day, every budget-chosen tau and every
binding count are properties of the thinned sample. Two figures already
reversed between the sample and the full data — the budget binding (0 of 8 →
all) and the near-ceiling reading (0.62-0.89 → 0.306-0.369).

## 5. SKIPPED — the nine figures not reproducible from a committed file

All nine are **session B's**, reported into the README before its CSVs were
committed. Session B covered targets 1.00 and 0.75 only; the complete report
lands in `docs/reference/` after stage 4.1 session C, and these are to be
re-checked against it then.

| Figure | Value as reported |
|---|---:|
| union test oracle recall | 0.8928571 (50 of 56) |
| lone institution relative, seeds 0 / 1 | 0.306 / 0.369 |
| union relative, seeds 0 / 1 | 0.500 / 0.600 |
| union raw recall, seeds 0 / 1 | 0.446 / 0.536 |
| `binding_fraction` at `per_inst_*` | 1.0 |

What **can** be checked about them now, and does pass:

| Check | Result |
|---|---|
| 50/56 equals the reported ceiling | 0.8928571 ✓ |
| raw ÷ ceiling reproduces each relative value | 0.446/(50/56) = 0.4995; 0.536/(50/56) = 0.6003; 0.679/(50/56) = 0.7605 ✓ |
| the ceiling matches the committed Phase 3 record | identical to `evaluate_baseline.csv`'s 0.8928571 ✓ |

Two further gaps in session B's hand-off, recorded rather than guessed: the
**operating point** those recall figures were read at was not named, and
**seed 2** was not included.

## 6. Figures that are definitions, not measurements

Stated here so they are not looked for in a CSV. Each is checked by a test
rather than by this script.

| Claim | Where it is pinned |
|---|---|
| `total_B = per_inst_B = B` for centralized | `test_total_and_per_institution_budgets_coincide_for_one_institution` — it follows from `budget_share(B, 1)` |
| boundary evidence cannot raise a ceiling | `test_evidence_moves_knowledge_not_rows_so_the_ceiling_cannot_move`, and empirically by the chain's ceiling check |
| the union of views holding an account = the whole window, for that account | `test_union_of_views_node_features_equal_whole_window_node_features` |
| substituting boundary accounts = substituting every account in a view | `test_substituting_boundary_accounts_equals_substituting_every_account` |
| score averaging's fast path is bit-identical to the naive one | `test_shared_featurisation_is_bit_identical_to_one_spec_at_a_time`, `test_the_fast_and_naive_averaged_scorers_agree_exactly` |
| the three model-seed policies give different numbers | `test_the_model_seed_policy_must_be_named_and_the_three_disagree` |
| 0.679 / 0.760 is the five-seed mean | `test_the_phase3_headline_is_the_five_seed_mean` |
