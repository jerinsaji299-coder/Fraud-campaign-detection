# Phase 4: choices D1-D6 did not cover

Every choice made while implementing stages 4.2, 4.3 and 4.4 that the research
decisions D1-D6 did not settle, one line each, with the alternative rejected.
Anything that **changes a pooled number** and is not covered by D1-D6 is a
stop condition, so it appears in the OPEN section at the top rather than being
decided here.

Last updated 2026-11-15.

---

## O1 — was open, DECIDED 2026-11-15 by the user: `matched_seed0`

| # | Choice | Decision and reason of record | Rejected |
|---|---|---|---|
| **O1** | Which model seed(s) the centralized arm of 4.4 uses | **`matched_seed0`** — model seed 0 alone, comparator **0.696 / 0.780**, constant across all 12 cells. *Phase 4 studies fragmentation, not model seed.* Every federated booster is model seed 0, so the centralized comparator must be model seed 0 or the contrast carries a model-seed difference inside it | **`phase4_seeds_as_model_seeds`** (0.6845 / 0.7667) rejected **explicitly**: it would make centralized the only arm whose model varies per cell, so its spread would be **model noise** against every other arm's **visibility noise** — not the same quantity. **`phase3_five_seeds`** (0.679 / 0.760) is outside the 3-seed scope; it remains the Phase 3 baseline headline, reported beside the comparator and never as it |

Three consequences, all recorded in the README:

1. **Two centralized numbers now coexist and are explained, not reconciled** —
   0.679 / 0.760 is the five-model-seed mean (Phase 3 headline); 0.696 / 0.780
   is the model-seed-0 comparator (Phase 4). The note sits next to both.
2. **Bracket notation is gone.** "0.679 [0.661, 0.696]" read as a confidence
   interval and is a seed range; it now reads "0.679 (five-model-seed mean;
   per-seed range 0.661-0.696)" at all five sites.
3. **A floor on interpretable differences: ~0.04 relative recall, ~2 campaigns
   of 56.** Phase 4's seeds vary visibility only, so model-seed noise is held
   fixed and invisible; Phase 3 measures it at centralized as 0.661-0.696 /
   0.74-0.78. That is a seventh to a fifth of the union → centralized gap, so
   smaller differences between arms are not reportable as real.

## Decided — naming and reporting

| # | Choice | Alternative rejected |
|---|---|---|
| N1 | Condition labels are `isolated`, `fedavg`, `evidence`, `fedavg_evidence`, `centralized` | `constants.CONDITIONS`'s placeholders `fedavg_only` / `fedavg_embedding`. "Embedding" is wrong for a tree model - there is no embedding anywhere in 4.3, only node features - and "only" names nothing. The README had already recorded this as a question 4.2 would have to settle |
| N2 | 4.3's two evaluations are separate **conditions** (`evidence`, `fedavg_evidence`) rather than one condition with a variant column | A variant column would let a pooled mean be taken across 4.3a and 4.3b by accident. Separate condition labels make every comparison name both sides |
| N3 | Dated README changelog entries that **record the 2026-11-14 rename** keep `uncapped_*` as that day's name, with the supersession noted; entries that merely *use* the name carry the current `per_inst_*` | Renaming globally, which would make the 2026-11-14 entry claim a name that did not exist that day; or restoring `uncapped_*` everywhere, which leaves a name in the README that no longer exists in any report |
| N4 | Session C's input cell rewrites `uncapped_` → `per_inst_` in session B's **staged copies** of its cell CSVs | Leaving the rename to the report assembly, which would put two names for one operating point into one report and split every groupby; or re-running session B, which costs hours to change a string |

## Decided — stage 4.2 (score averaging)

| # | Choice | Alternative rejected |
|---|---|---|
| F1 | Scores are averaged as **probabilities**, equally weighted | Averaging log-odds. The frozen 22-point tau grid lives on probabilities in `[0.05, 0.9999999]`; averaging in logit space and thresholding on the same grid would compare an isolated probability against a federated quantity that is not one |
| F2 | An institution with **no booster** is left out of the mean; it does not enter at score 0 | Entering 0, which claims the institution predicted "not laundering" when it did not predict at all, and pulls every averaged score down by `1/K` per missing model - changing the scores of the institutions that did train. Never fires on full data (96/96 trained); recorded because the 1% sample can starve an institution |
| F3 | The raw featurisation is **shared** across the K boosters scoring one view (`features.build_window_features_multi`), falling back to per-spec featurisation when the specs' one-hot vocabularies differ | Featurising once per booster always. That is 8x the expensive half of the work and makes 4.2 unaffordable on full data. Soundness is not assumed: the raw features depend on a spec only through the three vocabularies, and `test_shared_featurisation_is_bit_identical_to_one_spec_at_a_time` plus `test_the_fast_and_naive_averaged_scorers_agree_exactly` pin the two paths to bit equality |
| F4 | 4.2 reuses stage 4.1's boosters rather than training its own | Retraining identical boosters, which would differ by nothing but would break the pairing between 4.1 and 4.2 if any training detail drifted |

## Decided — stage 4.3 (boundary-account evidence)

| # | Choice | Alternative rejected |
|---|---|---|
| E1 | The shared node features are computed in **one pass over the whole window** rather than per subset of institutions | Building the union of views per account. The two are *identical*: visibility is an endpoint rule, so an account's owner sees every edge it is an endpoint of, and every node feature in the spec is an aggregate over incident edges alone. Proved rather than argued, by `test_union_of_views_node_features_equal_whole_window_node_features` |
| E2 | The override substitutes only **boundary** accounts, per D3 | Substituting every account in a view. The two give the identical matrix, because for an account its owner holds the view-local features already equal the shared ones. Pinned by `test_substituting_boundary_accounts_equals_substituting_every_account`, which is also the statement that no non-boundary account's features were shared |
| E3 | The shared-feature cache holds **float64** | float32, which halves the memory and was the first implementation. It made the substitution lossy by ~1e-7 on the two log-amount columns for *every* account, including accounts where it must be an exact no-op - so 4.3a would have differed from 4.1 by float noise in every node column instead of only where the evidence is new. Caught by `test_an_owned_accounts_features_are_unchanged` |
| E4 | One `BoundaryEvidence` per cell, shared by training, 4.3a and 4.3b, and refusing a second assignment | A provider per institution or per evaluation. Sharing is where nearly all of its value is, and it makes "all three used identical evidence" true by construction. Refusing a second assignment stops one cell being scored with another cell's evidence from a warm cache |
| E5 | The spec is refitted with the override applied inside `fit_feature_spec`, not only at scoring time | Fitting on un-augmented values and scoring augmented ones, which shifts every node column by exactly the partial-vs-complete-neighbourhood difference that 4.3 exists to measure. `test_a_spec_fitted_unaugmented_and_scored_augmented_really_differs` shows the failure is real |

## Decided — stage 4.4 and the shared machinery

| # | Choice | Alternative rejected |
|---|---|---|
| C1 | Centralized is evaluated as the **K = 1 case** of a per-institution condition, through a degenerate assignment putting every account in institution 0 | A separate centralized evaluation path. Reuse means tau selection, the grid-edge policy, the ceiling, the subsets and the summariser are the same code as every other arm - which is what rule 1 requires - and D4's "for centralized, `total_B = per_inst_B = B`" then falls out of `budget_share(B, 1)` instead of being special-cased |
| C2 | The centralized detection run happens **once per model seed** and is replicated across the `(target, seed)` cells, with the cell's seed used as the bootstrap seed | Re-running 12 times. A centralized model cannot depend on a visibility assignment - the assignment moves accounts between institutions without touching a row - so 11 of 12 runs would reproduce the first by construction. Every row carries `replicated_from_model_seed` and `n_independent_runs` so no reader can take 12 cells for 12 measurements |
| C3 | `isolated.run_cell` and `evaluate_institution` take `condition`, `scorer_for` and `trainer` parameters; 4.2 and 4.3 supply a scorer and nothing else | Copying `run_cell` per condition. Two copies are two chances for the arms to diverge silently, and a paired comparison over cells measured by different code is not a paired comparison |

## Decided — the open questions this closes

| Was open in the README | Now |
|---|---|
| "`constants.CONDITIONS` and the frontend still spell the middle two conditions `fedavg_only` / `fedavg_embedding` where this README says ... 4.2 has to settle it" | Settled by N1 |
| Whether evidence sharing raises an institution's **ceiling** | Under D3 it cannot: node features are shared, rows are not, so the visibility mask and therefore the oracle ceiling are unchanged. Pinned by `test_evidence_moves_knowledge_not_rows_so_the_ceiling_cannot_move`. The earlier expectation that exchange would raise a ceiling was about a form that moves transactions, which is not this one |

## Decided - the comparison report (stage 4.5)

| # | Choice | Alternative rejected |
|---|---|---|
| P1 | Five contrasts are reported: `fedavg - isolated`, `evidence - isolated`, `fedavg_evidence - fedavg`, `fedavg_evidence - isolated`, `fedavg_evidence - evidence`, plus each arm against centralized | Reporting only the two D3 names. `fedavg_evidence - isolated` is the only one that answers "both mechanisms together", and `fedavg_evidence - evidence` the only one that asks whether averaging still adds anything once evidence is shared. Both are cheap, and would otherwise be reconstructed by hand from rows that were never paired |
| P2 | Two pairing granularities, named in every row as `unit`: `per_institution` (paired per `(target, seed, institution)`) and `union` (per `(target, seed)`), plus `union_vs_centralized` | One granularity. The per-institution mean and the union move in **opposite** directions across the sweep, so collapsing them would average two different quantities |
| P3 | A cell undefined on **one** side removes the **pair** | Keeping the defined side, which would compare arm A over 8 institutions against arm B over 7. The number of pairs dropped this way is reported per contrast |
| P4 | A cell **absent** from one side raises `NonIdenticalCells`; a cell **present but undefined** does not | Treating both alike. An absent cell means a run did not cover it, which is a different failure from one the grid-edge policy excluded, and must not be absorbed silently |
| P5 | The ceilings are checked for agreement across all four per-institution conditions, and a disagreement **raises** | Reporting it as a warning. A moved ceiling means a condition changed visibility, which invalidates every contrast built on it, so there is nothing left to report |
| P6 | Each difference is summarised by mean, median, min, max and the counts of each sign - never a standard error or a t-statistic across institutions | A confidence interval across the 8. They are not exchangeable (institution 0 deterministically holds the largest bank), so a spread across them describes this fixed panel rather than estimating anything |
| P7 | Every row carries `below_seed_noise_floor`, true when the mean difference is under the Phase 3 model-seed spread | Stating the floor only in the README. A per-row flag is what stops a 0.02 difference being written up three sections later |
| P8 | `compare-conditions` runs over whatever arms are present and **names** the contrasts it could not form | Requiring all five. Naming the gaps is what makes stage 4.3 droppable without the report quietly becoming a different report. Verified by running it with the 4.3 reports removed |

## Decided - the live demo panel (stage 4.6)

| # | Choice | Alternative rejected |
|---|---|---|
| L1 | The demo campaign is chosen by: test-split, `eval_ok`, at least `MATCH_MIN_ACCOUNTS` accounts -> rank by **distinct institutions holding its accounts** descending, `campaign_id` ascending -> first one the centralized oracle detects in the split horizon | Picking at random, or picking the campaign the union detects best. Random would often pick a campaign no detector reaches, demonstrating nothing; "best detected" would select on the outcome being shown. Ranking on **spread across institutions** selects on the structure the panel is about, and the disclaimer says so. It chose campaign 214 at rank 0 of 56: 22 accounts across all 8 institutions |
| L2 | `/demo/*` is a separate router that **computes**, reading a bundle rather than the raw dataset | Extending an existing router, or letting the frontend compute. The API's contract is "serves, never analyses", so a computing router has to be visible as the exception; and the frontend rule forbids it computing research metrics |
| L3 | The **featurised windows** are cached, four at a time, and the predicts stay live | Caching per-model scores, which would make a flip ~2 ms. Rejected for now because it would stop the predicts being live, which is what "computed on the spot" means, and the measured flip is already 70-130 ms. Kept as the third fallback if full-data latency demands it, with the caveat that it would have to be said on the page |
| L4 | `/demo/aggregate` returns **200 with `available: false`** when there is no comparison artifact; `manifest` and `run` return 404 | Making all three 404. The aggregate does not depend on the bundle, and the panel's aggregate section must render a specific empty state rather than an error, because that section ships whether or not the numbers exist |
| L5 | Purity and coverage come from one new function, `clusters.cluster_match_detail`, which `cluster_hits_campaign` now delegates to | Computing them in the panel. Two definitions would let it show 60% purity for a cluster the frozen rule counted as a hit at some other threshold |
| L6 | The bundle script scans every offered `(window, institution)` and records **all four** outcomes | Listing only the settings where averaging wins. On the sample that is 2 of 64, with 0 where the own model wins and 58 where neither finds the campaign - and the panel prints all four counts |
| L7 | The bundle exports the **8 boosters of one cell** (~14 MB on full data) rather than all 96 (~170 MB) | Shipping all 96. The toggle needs exactly the K models of one `(target, seed)` |

## Decided - the Kaggle sessions D to J

| # | Choice | Alternative rejected |
|---|---|---|
| K1 | Six sessions: 4.2 across two, 4.3 training in one, 4.3 evaluation across two, plus one short session for 4.4 and the comparison | One session per stage. The measured per-cell ratios applied to session B's bracket put 4.2 at 7-15.5 h and 4.3 evaluation at 8-17.5 h, both over the 12 h limit at the pessimistic end |
| K2 | Each evaluation session prints a **CALIBRATION** line after its first cell, projecting the session from that measurement | Relying on the runbook's estimates. The full-data per-cell baseline is the one quantity nobody has measured, so every estimate inherits that uncertainty; a line printed at minute 30 replaces it with a fact |
| K3 | Sessions D/E/F come before G/I/J, so a complete `isolated`/`fedavg`/`centralized` result with its comparison exists before 4.3 starts | Running 4.3 earlier. "4.3 can be dropped cleanly" is only true if everything it would be compared against is already finished |
| K4 | `evaluate-centralized` **refuses to train** without `--retrain` | Falling back to training silently. A retrained comparator is a different model from the one the Phase 3 record is about, and the report would carry that record's provenance for it |
| K5 | Sessions F and J re-run the D6 `uncapped_* -> per_inst_*` rename over the staged reports | Assuming every input was produced after the rename. Sessions B and C ran before it, and one report carrying two names for one operating point would split every groupby |
| K6 | `--max-cached-windows 12` in the runbook's 4.3 cells, not unbounded | Unbounded, which is fastest and is what the sample uses. A full-data window's shared node features are tens of MB and a cell walks ~33 windows, so an unbounded cache risks the session being killed for memory - which costs more than the recomputation |

## Decided - found while measuring, and worth keeping

| # | Choice | Why it is here |
|---|---|---|
| M1 | `featurise_was_cached` is read **before** the cache lookup | It was read after, so it always said `True` and reported nothing. Found only because the measurement run printed it beside a 134 ms featurisation. It now reads False/True with 134 ms/0 ms |
| M2 | The `EvidenceRunner` cell guard compares assignments **by value**, not identity | Identity rejected the normal path: `run_cell` rebuilds the assignment from the pipeline, so the trainer and the scorers receive equal-but-distinct objects for the same cell, and 4.3a could not be evaluated at all. A test pins both directions now |
| M3 | Two cross-check tolerances tightened from 20% to 0.5% | They were loose enough to pass a drifted figure. Tightening them corrected a README claim: the three fragmented targets are within **2.2%** of each other, not 2% |
