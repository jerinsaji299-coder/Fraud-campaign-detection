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

**Current state: 146 passed, 0 failed, 10 skipped.** The ten skipped are the
only Phase 4 figures in the README that are **not** reproducible from a
committed file — they are listed in their own section at the bottom, and they
are the things to re-check when their evidence lands. A skipped figure is one
the README asserts on trust; that is a smaller list than the passing one on
purpose.

Last updated 2026-11-18.

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
| chain ratios: fedavg **1.21×**, evidence **0.85×**, fedavg_evidence **1.06×**, evidence training **1.46×** | — | **`docs/reference/phase4_chain_sample.json`** (committed) | `python scripts/run_phase4_chain.py --config configs/local.yaml` |
| ceilings agree across all four conditions, 256 ceiling cells each | max disagreement **0.0** | same | same |
| union ceiling equals centralized ceiling, 32 comparisons | difference **0.0** | same | same |
| whole chain, 4 targets, seed 0 | 2,609s, exit 3 (12 low-edge taus) | same | same |

| backfill reconstructs the measured F1 value to | **≤ 2.22e-16** over 32 (cell, institution) pairs | `artifacts/isolated_dev/phase4_isolated.json` + its combined CSVs | `python scripts/verify_f1_backfill.py` |

That last row has its **own** command because it is the evidence for a
decision not to spend ~1.5 h re-running session B's two cells. It needs a run
where both the tau curve and the summary row are on disk, which is a local
isolated run: the JSON carries `tau_curves`, the Kaggle per-cell CSVs do not.
The script stages the run's cells with the new columns stripped, runs the real
`backfill-f1-value`, and compares. It is **one ULP, not bit-exact** - the
per-cell CSV stores precision and recall at 16 significant digits, so `1/26`
round-trips as `0.0384615384615384`.

The dry run's own figures **are** committed, as
`docs/reference/phase4_chain_sample.{md,json}` and
`phase4_compare_sample{,_headline}.csv`, fenced as development-only. The
verifier checks the ratios and both ceiling checks against them.

> **The ratios above replaced a first set taken from one cell** — 1.41 / 1.15
> / 0.44 / 4.2 — every one of which was wrong. The one-cell `train-evidence`
> had the cold boundary-account pass inside it; the one-cell `fedavg` was
> measured under different load. Fixing the verifier's own baseline matcher
> was what surfaced it: matching `" isolated"` also matched
> `"train isolated"`, so it first reported a fedavg ratio of **20.45** by
> dividing an evaluation by a training time.

**A caution the README repeats and this table restates:** the sample thins the
background to 1%, so its positive rate is ~3.3% against the full data's
0.102%. Precision, false alarms per day, every budget-chosen tau and every
binding count are properties of the thinned sample. Two figures already
reversed between the sample and the full data — the budget binding (0 of 8 →
all) and the near-ceiling reading (0.62-0.89 → 0.306-0.369).

## 5. Stage 4.1 session C1 — the gate evidence (2026-11-18)

Source for all of these: **`docs/reference/phase4_isolated_c1_t100_s2*`**,
the nine artifacts of the one cell C1 evaluated. **One cell of the full data,
not a pooled Phase 4 result** — `t100_s2` alone, target 1.0, seed 2.

| Figure | Claimed | Source row |
|---|---|---|
| low-edge F1 taus in the cell | 1, institution 1 | `_taus.csv`, `operating_point == f1`, `tau_edge == low` |
| institution 1 `f1_best_value` | 0.0027, degenerate | the same row |
| institution 1 curve maximum, and 0 from tau 0.3 up | 0.0027, then 0.0 | `tau_curves["1"]` in the cell JSON |
| high-edge F1 taus in the cell | 1, institution 2 | `_taus.csv`, `tau_edge == high` |
| institution 2 `f1_best_value`, curve maximum, campaigns ever detected on validation | 0.0, 0.0, 0 over all 22 points | `_taus.csv` and `tau_curves["2"]` |
| institution 2 budget points **defined** | 6 of 6 | `_taus.csv`, its six non-`f1` rows |
| institution 2 validation oracle recall and reachable campaigns | 0.0 and 0 — the only such institution in the cell | `_ceilings.csv`, `eval_split == val`, `test_subset == all` |
| F1 pooled mean over the 7 remaining, per split | val 0.082418, test 0.094388, stress 0.058271 | `_summary.csv`, the `f1` / subset-`all` rows |
| the same with institution 2 entered at recall 0 | val 0.072115, test 0.082589, stress 0.050987 | the same rows plus one zero |
| the relative change that would be | **−12.5% on every split** | arithmetic on the two above: adding a zero to seven values is a 7/8 scaling |
| institution 1 test recall at `f1` — the highest of the seven | 0.2321 | `_summary.csv`, test / `f1` |
| the F1 mean if institution 1 were dropped too (rejected option C) | 0.0714 | the same rows without institution 1 |
| institution 2 test recall at `total_10`, and institutions beating it | 0.0714, and 1 — so second of the eight | `_summary.csv`, test / `total_10` |
| degenerate curves in the cell, and the panel | **2 of 8** | `_taus.csv`, the eight `f1` rows |
| the same count **as C1 shipped it** | 1 of 7 | `_pooled.csv` — the figure the 2026-11-18 denominator fix changes, pinned here so the change is visible rather than silent |

The last two rows are the only place this script checks a figure **and** the
superseded version of it. C1's artifacts were written before the denominator
fix, so they still say 1 of 7; every cell from C2 onward says 2 of 8 for a
cell shaped like this one. Both are correct for what they are, and keeping
both checkable is what stops the change looking like a drift.

*Added 2026-11-18, after the fix was found not to reach them:* a cell written
before the fix is reloaded verbatim, so the count is re-taken at assembly by
`_recount_degenerate_on_reload`. Run against these artifacts it moves C1's 12
pooled and 12 union rows from `1 of 7` to `2 of 8` and leaves C2's alone.

## 5b. Stage 4.1 session C2 — targets 0.75 and 0.50 (2026-11-18)

Source for all of these: **`docs/reference/phase4_isolated_c2*`**, the nine
artifacts of the six cells C2 evaluated.

| Figure | Claimed | Source row |
|---|---|---|
| cells, and institution-cells | 6 and 48, targets 0.5 and 0.75 | `_taus.csv`, distinct `(visibility_target, seed)` and the `f1` rows |
| degenerate F1 curves | **0** — and 0 rows with no flag at all | `_taus.csv`, the 48 `f1` rows |
| undefined F1 points, and tau rows at a grid edge | 0 and 0 | `_taus.csv`, `defined` and `tau_edge` |
| the exit code, from the artifacts rather than the expectation | 0 edge taus over 6 cells, so **exit 0** | `tau_at_grid_edge` and `cells` in `phase4_isolated_c2.json` |
| the F1 tau range | 0.9 to 0.995 — the interior of the grid | `_taus.csv`, the 48 `f1` rows |
| the `f1_best_value` range | 0.0317 to 0.3226 | the same rows |
| budget points, and how many bind | 288 of 288 | `_taus.csv`, the non-`f1` rows |
| union recall at `f1`, and what it costs | 0.500-0.625 at **39.0-86.7 /day** | `_union.csv`, test / subset `all` |
| union recall at `total_10`, and what it costs | 0.268-0.446 at **1.9-4.0 /day** | the same rows |
| the same at target 1.0 | 0.571 at **2,119.2 /day** | `phase4_isolated_c1_t100_s2_union.csv`, test / `all` |
| the factor between them | **24.4x** against C2's dearest cell, **54.4x** against its cheapest | arithmetic on the two above |
| the mechanism: institution 1's tau and its own volume | 0.05, and **1,956.0 /day** — 92.3% of the union's summed total | `phase4_isolated_c1_t100_s2_summary.csv`, test / `f1` |
| union oracle recall at 0.75 and 0.50 | **0.8928571** at every cell — the centralized ceiling | `_union.csv`, test / `all` |
| the counterfactual behind the proposal: union F1 without institution 1 | 32 campaigns to **23 of 56**, 0.571 to **0.411**, at a cost of **≤163.4 /day** | recomputed from `phase4_isolated_c1_t100_s2_outcomes.csv`, the per-institution `f1` / test rows |
| the rule the README states about quoting any of this | *"Cross-target comparison at the F1 point is confounded by cost"* | `README.md` — a form of words, checked as one |

The last row is checked because the cost figures only mean something
alongside the rule they were recorded for. The counterfactual row is checked
because a recommendation nobody can re-derive is a recommendation taken on
trust.

## 6. SKIPPED — the ten figures not reproducible from a committed file

All ten are **session B's**, reported into the README before its CSVs were
committed. Session B covered targets 1.00 and 0.75 only; the complete report
lands in `docs/reference/` after stage 4.1 session C, and these are to be
re-checked against it then.

| Figure | Value as reported |
|---|---:|
| `t100_s0` wall time, sum(`eval_seconds`) × 1.08 | 2,568.5s (46.2 min) |
| `t100_s1` wall time, sum(`eval_seconds`) × 1.08 | 2,465.0s (44.4 min) |
| union test oracle recall | 0.8928571 (50 of 56) |
| lone institution relative, seeds 0 / 1 | 0.306 / 0.369 |
| union relative, seeds 0 / 1 | 0.500 / 0.600 |
| union raw recall, seeds 0 / 1 | 0.446 / 0.536 |
| `binding_fraction` at `per_inst_*` | 1.0 |

(Nine were skipped before the dry run was committed; its ratios are now
checked, and session B's two derived wall times were added, leaving these
ten.)

What **can** be checked about them now, and does pass:

| Check | Result |
|---|---|
| 50/56 equals the reported ceiling | 0.8928571 ✓ |
| raw ÷ ceiling reproduces each relative value | 0.446/(50/56) = 0.4995; 0.536/(50/56) = 0.6003; 0.679/(50/56) = 0.7605 ✓ |
| the ceiling matches the committed Phase 3 record | identical to `evaluate_baseline.csv`'s 0.8928571 ✓ |

Two further gaps in session B's hand-off, recorded rather than guessed: the
**operating point** those recall figures were read at was not named, and
**seed 2** was not included — because B stopped before reaching it.

The two wall times are the figures that settled **why** B stopped: at 44-46
minutes a cell, two cells plus one in flight is ~2.3 h, which excludes the
12 h session limit and leaves the low-edge F1 tau rule as the only cause that
fits. They were derived from B's own `*_institutions.csv` rather than
re-measured, and the 1.08 scale factor between a cell's wall time and the sum
of its per-institution `eval_seconds` is itself from the committed dry run,
where it ranged 1.05-1.11 across four cells. Both become checkable here when
B's CSVs are committed after session C3.

## 7. Figures that are definitions, not measurements

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
| the degeneracy epsilon is **0.01**, with an inclusive boundary | `test_the_degeneracy_epsilon_is_fixed_and_its_boundary_is_inclusive` — fixed 2026-11-16, before any cell ran under the recording policy |
| `f1_best_value` is the value **at** the chosen tau | `test_f1_best_value_is_the_value_at_the_chosen_tau` |
| a cell with no F1 curve is out of both terms of the degenerate count | `test_degenerate_summary_excludes_cells_with_no_curve` |
| `2PR/(P+R)` of the validation `f1` row is the curve maximum | `test_the_backfill_identity_holds_for_the_f1_point_on_validation`, and empirically by `scripts/verify_f1_backfill.py` |
| a low-edge F1 tau is recorded, not a stop | `test_a_low_edge_f1_tau_is_recorded_under_the_2026_11_16_policy` |
| a high-edge F1 tau gets one of three reasons, ceiling first | `test_the_three_reasons_a_high_edge_f1_tau_can_happen` — and `test_a_high_edge_on_a_budget_point_keeps_the_original_reason` for the scope |
| the degenerate count is over all institutions, not the survivors | `test_a_degenerate_count_is_over_all_institutions_not_the_survivors`, end to end in `test_a_cell_excluded_for_degeneracy_is_still_counted_in_its_own_cell` |
| a reloaded cell's high-edge reason is re-derived at assembly | `test_a_reloaded_cells_high_edge_reason_is_re_derived_at_assembly` |
| a zero validation ceiling makes `f1_best_value` derivable as exactly 0.0 | `test_a_zero_validation_ceiling_backfills_f1_best_value_as_zero` |
