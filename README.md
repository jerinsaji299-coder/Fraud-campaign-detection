# Early Discovery of Emerging Fraud Campaigns from Fragmented Cross-Institution Evidence

**Repository:** https://github.com/jerinsaji299-coder/Fraud-campaign-detection

A final-year B.Tech (AI & Data Science) major project (journal paper also
planned) studying how the benefit of cross-bank collaboration for early
money-laundering-campaign discovery changes as each bank's visibility of a
campaign shrinks, and whether that depends on the campaign's topology. The
repository has two parts: a Python research pipeline that is the single
source of truth for every number, and a React frontend that presents the
dataset, campaigns, visibility, and (later) experiment results — the
frontend never invents or computes research results itself.

```bash
git clone https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git
```

## Table of contents

1. [Research question, hypothesis, and contribution](#research-question-hypothesis-and-contribution)
2. [Project status](#project-status)
3. [Dataset](#dataset)
4. [Frozen research definitions](#frozen-research-definitions)
5. [Decision log](#decision-log)
6. [Verified numbers](#verified-numbers)
7. [Repository structure](#repository-structure)
8. [Setup](#setup)
9. [How to run](#how-to-run)
10. [Backend: modules](#backend-modules)
11. [Artifacts](#artifacts)
12. [API reference](#api-reference)
13. [Frontend](#frontend)
14. [Results schema](#results-schema)
15. [Testing](#testing)
16. [Known limitations and open questions](#known-limitations-and-open-questions)
17. [Future phases](#future-phases)
18. [Phase 3 definitions](#phase-3-definitions-centralized-baselines)
    - [Oracle ceiling](#oracle-ceiling)
    - [Stage 3.3: the XGBoost baseline](#stage-33-the-xgboost-baseline)
    - [Stage 3.4: the centralized GNN](#stage-34-the-centralized-gnn)
    - [3.4.1 design](#341-design--approved-2026-11-10-with-four-additions)
    - [3.4.4 the full run](#344-the-full-run--done)
    - [3.4.5 the gate report: why the GNN loses](#345-the-gate-report--why-the-gnn-loses-at-extraction-not-at-ranking)
    - [Comparing two configurations: the paired method](#comparing-two-configurations-the-paired-method)
    - [Recall against the ceiling](#recall-against-the-ceiling)
19. [Phase 4 definitions](#phase-4-definitions-the-federated-conditions)
    - [Frozen scope](#phase-4-frozen-scope)
    - [The four conditions](#the-four-conditions)
    - [Stage 4.0: institution-split data](#stage-40-institution-split-data--done)
    - [Stage 4.1: the isolated condition](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14)
        - [Grid-edge policy](#12-grid-edge-policy-and-a-proposed-extension-of-model_tau_grid)
        - [What was built](#what-was-built)
        - [Measured on the 1% sample — development only](#measured-on-the-1-sample--development-only-not-for-quotation)
20. [Version history](#version-history)
21. [Changelog](#changelog)

## Research question, hypothesis, and contribution

**Research question:** How does the benefit of collaboration between banks
for early laundering-campaign discovery change as each bank's visibility of
a campaign decreases, and does this depend on campaign topology?

**Experimental conditions** (models and experiments are a later phase, not
built yet):

1. **Isolated** — each institution trains and detects alone.
2. **FedAvg-only** — institutions share model weights (the "knowledge
   advantage").
3. **FedAvg + boundary embedding exchange** — institutions additionally
   share embeddings of boundary accounts at detection time (the "evidence
   advantage").
4. **Centralized** — full-visibility upper bound.

**Related work this project positions against:**

- Ceydeli et al. 2026 — federated cross-client subgraph pattern detection
  with embedding exchange.
- Zheng et al. 2025, "Networked Markets, Fragmented Data" — federated GNN
  on bank-partitioned IBM AML data.
- FedGraph-VASP 2026.

**Our distinct contribution:** campaign-level evaluation (not transaction-
level), discovery lead time as a first-class metric, and a graded
visibility dose-response curve, split by campaign topology (fragmentable
campaigns vs. a hub control group that is always fully visible to one
bank).

## Project status

**Phase 3 is COMPLETE** (2026-11-14, tag `v0.3-phase3`). The gate question is
answered: a centralized model **can** discover campaigns early — the XGBoost
baseline reaches **0.679 test recall at L = 48, 0.760 of the oracle
ceiling** — and **the graph model does not beat it**. Over five seeds the GNN
matches XGBoost's recall at the F1 operating point while raising ~5× the
alerts, and at a matched budget of 10 false alarms a day it reaches **0.525
against XGBoost's 0.654**. The *reason* is measured rather than guessed: the
shortfall is at **cluster extraction, not edge ranking** — see
§[why the GNN loses](#345-the-gate-report--why-the-gnn-loses-at-extraction-not-at-ranking).
The stage 3.3 retrain reproduced every recorded stage 3.3 number exactly,
so the baseline is a confirmed regression check rather than a one-off.

**Current stage:** **Phase 4**, stage **4.0 done** — the institution-split
data layer is built and tested, and the per-institution training loads are
measured on the 1% sample, and stage **4.1 — the `isolated` condition — is
approved, built and tested** (2026-11-14). Each institution alone is the floor
the federated conditions must beat, and it is measured on the sample as a
development check; the full-data run is a Kaggle session via `train-isolated`
and `evaluate-isolated`. The frozen scope stands: **3 seeds, L = 48**, 96
institution-cells. **Next: stage 4.2**, which has to define what "FedAvg"
means for a tree model. See
§[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14).

Phase 4 is unaffected by the Phase 3 result: it uses XGBoost only, a decision
taken before the five-seed run and now reinforced by it (see the Decision
log, 2026-11-14).

**How Phase 3 finished.** The 3.4.4 full run went through as planned — two
GPU sessions (one per lookback, 20 GNN models) and one CPU session in commit
mode that retrained the 20 XGBoost boosters, evaluated everything, and ran
the direct model-versus-model comparison. Evidence is committed under
`docs/reference/`. The 3.4.3 quick check's one-seed reading held up under
five seeds: the GNN is behind at a comparable alert volume, it was **not
tuned** in response (equal tuning effort stands), and 3.4.5 diagnoses *where*
it loses. See §[stage 3.3 results](#stage-33-results-full-data-five-seeds--complete),
§[the 3.4.4 results](#344-the-full-run--done) and
§[the gate report](#345-the-gate-report--why-the-gnn-loses-at-extraction-not-at-ranking).

**Phase 2 — pipeline, artifacts, API, frontend (complete)**

| Stage | Description | Status |
|---|---|---|
| 0 | Notebook relocation, CLAUDE.md, README, repo skeleton, plan | done |
| 1 | Research pipeline (loading, campaigns, splits, institutions, visibility), unit tests, `make_sample.py`, `run_pipeline.py` | done |
| 2 | Artifact export | done |
| 3 | FastAPI server + API tests | done |
| 4 | Frontend scaffold, theme, layout, API client, types, Home, Dataset, Campaign Explorer | done |
| 5 | Campaign Detail, Visibility Lab, Results Dashboard (empty-state), Methodology page, frontend tests | done |

**Phase 3 — baselines and campaign-level evaluation (COMPLETE, 2026-11-14)**

Gate question: can a **centralized** model (full visibility, natural data, no
reassignment) discover campaigns early, and does a graph model beat a
non-graph baseline at it? Nothing federated, no institutions in training, and
no visibility reassignment is used in this phase.

| Stage | Description | Status |
|---|---|---|
| 3.0 | Kaggle runner + Phase 2 full-data verification | done — all 24 numbers PASS, determinism confirmed |
| 3.1 | Window builder + features + leakage tests | done |
| 3.2 | Evaluation module (extraction, matching, lead time, bootstrap CIs) + oracle and random scorers | done, **validated on full data**; ceiling, miss diagnosis and the paired lookback comparison all measured and recorded |
| 3.3 | XGBoost baseline: train, select tau on validation, evaluate on test | **done** — 20 boosters on the full data at commit `de48b35`, exit 0, no edge taus. Selected: the full model at L = 48 (validation recall 0.531 vs 0.385); test recall 0.679 [0.661, 0.696], 0.760 of the ceiling. Format ablation settled; one campaign beats the ceiling |
| 3.4 | GNN centralized (GINEConv edge classification) | **done** — all six sub-stages. 20 models on the full data (2 variants × 2 lookbacks × 5 seeds), both GPU sessions and the CPU session exit 0. **The GNN does not beat the baseline**: at a matched 10 FA/day budget it reaches 0.525 test recall against XGBoost's 0.654, and the gate report locates the shortfall at **cluster extraction, not edge ranking** |
| 3.5 | Gate report | **done** as §3.4.5. The results files and the frontend Results page stay **deferred to Phase 5**, which is what populates `artifacts/results/` — Phase 3 produces no `runs.parquet`/`detections.parquet` |

**Phase 4 — federated conditions (in progress)**

Scope deliberately cut to fit a two-week finish: **XGBoost only, the full
feature set only, three seeds, one lookback**. See
§[Phase 4 frozen scope](#phase-4-frozen-scope).

| Stage | Description | Status |
|---|---|---|
| 4.0 | Institution-split data: per-institution views of every window, and the account features each institution can compute from its own view alone | **done** — `federation/views.py`, 31 tests, per-institution training loads measured on the sample, visibility cross-check exact on all 370 campaigns |
| 4.1 | **The `isolated` condition**: each institution trained and scoring on its own view alone, with its **own** oracle ceiling per target, and the two pooled numbers (per-institution mean, and the union over institutions) | **built and tested** — `federation/isolated.py`, 40 tests, backend 402 passed / 25 skipped; `train-isolated` and `evaluate-isolated` runner subcommands, resumable. Measured on the sample; the full-data run is a Kaggle session. §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) |
| 4.2 | `fedavg`: what "FedAvg" means for a tree model, then implementation and tests, reusing the Phase 3 evaluation code unchanged | not started — the design question moved here from 4.1 |
| 4.3 | `fedavg_exchange`: what crosses the institution boundary, then the gate — centralized vs `fedavg_exchange` at the 100 bin, full data. A failed gate stops the sweep | not started |
| 4.4 | Kaggle runner subcommands, resumable per (condition, bin, seed) | not started |

**Later phases (not started)**

| Phase | Description |
|---|---|
| 5 | Core experiment: all four conditions across visibility bins and seeds |
| 6 | Extensions: imperfect alignment, adversarial fragmentation, DP noise |
| 7 | Analysis and paper |

## Dataset

- **Source:** Kaggle — `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.
- **Files used:** `HI-Small_Trans.csv` (5,078,345 transactions) and
  `HI-Small_Patterns.txt` (laundering-campaign ground truth). The larger
  `HI-Medium`/`HI-Large`/`LI-*` files exist in the same dataset but are not
  used now; `HI-Medium` is kept as an optional later replication (see
  Decision log).
- **Date range used:** September 1–10, 2022 (`2022/09/01 00:00` through
  the cutoff `2022/09/11 00:00`). The raw file actually spans through
  2022/09/18, but only 1,108 rows (0.02% of the file) fall on or after
  September 11, and 59.1% of those are laundering — far above the ~0.1%
  overall laundering rate — so including that tail would let a model
  cheat by learning "late in the file = laundering". See the Decision log
  and `CUTOFF` in Frozen research definitions.

## Frozen research definitions

These definitions must not change without explicit user approval (see
`CLAUDE.md`). Each includes why it is defined this way.

**Columns:** `Timestamp, From Bank, Account, To Bank, Account.1, Amount
Received, Receiving Currency, Amount Paid, Payment Currency, Payment
Format, Is Laundering`. Account columns are loaded as strings (not ints)
because leading zeros are meaningful in some IDs. Bank IDs are loaded as
integers because the pattern file writes some banks with leading zeros
(e.g. `"021174"`) that must compare equal to the integer bank IDs in the
transaction file.

**Account ID:** `f"{bank_int}_{account_str}"`, computed separately for the
sender (`src`) and receiver (`dst`) of each transaction. *Why:* raw
account strings are reused across banks, so the ID must be scoped to a
bank to be globally unique.

**Timestamp format:** `"%Y/%m/%d %H:%M"`.

**CUTOFF:** `2022-09-11 00:00`. Model input = transactions with
`timestamp < CUTOFF`. The uncut data is still used to compute true
campaign completion time (ground truth), since a campaign's real end may
fall after the cutoff. *Why:* see Dataset section above — the post-cutoff
tail is a label leak.

**Campaign parsing:** `HI-Small_Patterns.txt` blocks start with `BEGIN
LAUNDERING ATTEMPT - <TYPE...>` and end with `END LAUNDERING ATTEMPT`;
interior lines are transactions in the same columns as the CSV, without a
header. `campaign_id` is the 0-indexed order of appearance in the file
(370 campaigns, IDs 0–369). `base_type` is the text after `"- "` and
before any `":"`, stripped — one of `FAN-OUT, FAN-IN, CYCLE,
SCATTER-GATHER, GATHER-SCATTER, BIPARTITE, STACK, RANDOM` (the raw `type`
field also carries a free-text suffix like `"Max 16-degree Fan-Out"`,
which is dropped for `base_type`).

**Matching pattern transactions to the full transaction table:** join on
`[Timestamp, From Bank, src account, To Bank, dst account, Amount Paid]`.
All 3,209 pattern-file transactions must match (asserted); 0 are
unmatched.

**`campaign_id` on the main transaction table:** `-1` for rows that are not
part of any campaign. Rows with `Is Laundering == 1` and `campaign_id ==
-1` are "unassigned laundering" — used as training labels, excluded from
campaign-level evaluation (1,968 of 5,177 laundering rows; see Decision
log).

**Campaign table columns:** `campaign_id, base_type, n_txn, n_accounts,
n_banks, start, end` (true completion time, from the *uncut* data),
`duration_h`, `crosses_cutoff` (`end >= CUTOFF`), `deadline = min(end,
CUTOFF)`, `eval_ok = (n_txn >= 3 and n_accounts >= 3)`. *Why `eval_ok`:*
about a quarter of campaigns have only 1–2 transactions, where "lead time"
is not a meaningful concept.

**Splits, by campaign start date:**
- `train`: Sept 1–4
- `val`: Sept 5
- `test`: Sept 6–7
- `stress`: Sept 8–10

*Why Sept 6–7 for test, not Sept 8–10:* with test = Sept 8–10, 74 of 76
eval_ok test campaigns were truncated by the cutoff before completing —
Sept 6–7 keeps most test campaigns fully observed while Sept 8–10 becomes
a deliberate "stress" split of heavily-censored campaigns.

Later-phase time boundaries derived from the same splits: training data is
everything before `2022-09-05 00:00`; validation windows cover Sept 5;
test windows cover Sept 6 up to `CUTOFF`.

**Groups:** `hub = {FAN-IN, FAN-OUT, GATHER-SCATTER}`; all other base
types = `fragmentable`. *Why:* in every hub-type campaign, one hub account
funnels or gathers all transactions, and that account's own bank
necessarily sees 100% of the campaign — hub campaigns are always at
visibility 1.0 under any institution assignment, so they can't be used to
study the visibility dose-response and instead serve as a control group.

Fragmentable campaigns whose lowest achievable visibility (the
local-search run at target 0.25, seed 42) is `>= 0.9` are relabeled
**"unfragmentable"**. The **control group** = `hub` + `unfragmentable`.
*Why:* a handful of small fragmentable campaigns can't be pulled below
~0.9 visibility no matter how their accounts are assigned (e.g. too few
distinct accounts to spread across institutions), so they behave like hub
campaigns for this study and should not be analyzed as if visibility could
be manipulated.

**Shared accounts:** a test campaign is flagged `shares_train_account` if
it shares any account with any train-split campaign. *Why:* an account
reused across splits could let a model "remember" it from training rather
than generalize, so results must be reportable with and without these
campaigns.

**Institutions:** `K = 8`. Bank volume = count of pre-cutoff transactions
where the bank is sender or receiver. Banks are sorted by volume
descending and assigned greedily, each to the institution with the
currently smallest total load — deterministic, no randomness, and the
mapping is saved. Two clarifications added when this was implemented
(2026-10-06, see Decision log): ties in volume are broken by ascending
bank id so the ordering is fully reproducible, and banks with **zero**
pre-cutoff volume are still assigned (sorted last) so that every bank in
the dataset has an institution — a campaign crossing the cutoff can
otherwise reference a bank that never appears before it. *Why K = 8, not 3:* every transaction is visible to
*both* of its endpoint banks' institutions, so under random assignment the
lowest visibility any campaign can structurally reach is about `2/K`; with
`K=3` that floor is ~67%, far above the lowest intended target of 25%.
`K=8` lowers the floor enough to probe the 25% target, which matters
because median eval_ok campaigns only touch 8 accounts.

An institution **sees** a transaction if the sender's or the receiver's
bank belongs to that institution.

**Visibility:** `campaign visibility = max over institutions of (fraction
of the campaign's transactions that institution sees)`. Targets `[1.0,
0.75, 0.5, 0.25]`.

For each `(seed, target)`, **only** the accounts belonging to `eval_ok`
**fragmentable** campaigns are reassigned to institutions; every other
account keeps its bank-based (natural) institution. Reassigning an account
moves **all** of its transactions (not just the ones in a given campaign).

"Fragmentable" here is a test on **topology only** — `eval_ok` and a
non-hub `base_type` — recorded as the `reassignable` column. Campaigns
later relabelled `"unfragmentable"` (below) stay in this set and keep
receiving exactly the same treatment; that relabel is an analysis
grouping, not an exemption (decided 2026-10-06, see Decision log).

- **Target 1.0:** all of a campaign's accounts are sent to one randomly
  chosen institution.
- **Other targets:** local search — random start, then repeatedly pick one
  random account, move it to a random institution, and keep the move only
  if `|visibility - target|` does not get worse; 400 iterations.

**Conflict resolution:** an account that belongs to several campaigns is
assigned according to its **earliest** campaign (by start time). After all
assignments for a given `(seed, target)` are finalized, every campaign's
**achieved** visibility is **recomputed** from the final global
assignment (it may differ from the value found during that campaign's own
local search, because a shared account may have been moved again by a
later-processed campaign).

**Visibility bins** (analysis uses bins, not raw targets, because achieved
visibility often misses its target): `"100"` if `>= 0.9`; `"75"` if in
`[0.65, 0.9)`; `"50"` if in `[0.45, 0.65)`; `"low"` if `< 0.45`.

**Seeds:** `[0, 1, 2, 3, 4]`; the reassignment seed equals the experiment
seed. The `"unfragmentable"` floor run is a separate, fixed run at seed
`42`, target `0.25`, and is **not** one of the five experiment seeds.

A function must be provided that, given an assignment, returns each
institution's visible transaction subset
(`visibility.visible_transactions`).

**Planned evaluation rules** (documented now, **implemented in Phase 3**,
not this session):

- Detection windows: 6-hour steps, 24–72 hour lookback, causal (a detector
  never sees future transactions relative to its window's end).
- Matching: a detected cluster "hits" a campaign if at least 50% of the
  cluster's accounts belong to the campaign **and** the cluster covers at
  least 3 of the campaign's accounts.
- Lead time vs. true completion time, reported three ways: absolute hours,
  normalized by campaign duration, and fraction of campaign transactions
  observed at detection time. A campaign not detected by its `deadline` is
  "missed".
- Test results reported on four subsets: `all`, `fully_observed`
  (`crosses_cutoff == False`), `no_shared_accounts`
  (`shares_train_account == False`), and `oracle_reachable` (the campaigns
  the oracle detects at that lookback and split — see
  §[recall against the ceiling](#recall-against-the-ceiling)). The first
  three are predicates on a campaign; the fourth depends on an oracle run,
  so `constants.STATIC_TEST_SUBSETS` names the three that do not.

## Phase 3 definitions (centralized baselines)

Additive to the frozen Phase 2 definitions above; nothing there changed.
Phase 3 answers one gate question: **can a centralized model, with full
visibility and no reassignment, discover campaigns early — and does a graph
model beat a non-graph baseline at it?** Nothing federated, no institutions
in training, no visibility sweep.

**Detection times.** Every 6 hours, aligned to 00:00 / 06:00 / 12:00 /
18:00, over `(2022-09-01 00:00, CUTOFF]` — 40 times. Sept 1 00:00 is
excluded because its window is empty by construction.

**Causal windows.** At detection time `t` a model sees only transactions
with `t - L <= timestamp < t`. The upper bound is strict: a transaction
stamped exactly `t` is already the future. Lookback `L ∈ {24, 48}` hours,
chosen on validation; development uses 24. **L = 72 was dropped** on
2026-10-10 (see Decision log); `features-smoke` still measures it once and
labels it `[EXCLUDED]`, so the evidence for dropping it stays on the
record.

**Detection-time splits** (distinct from the campaign splits, which are
about when a campaign *starts*): train `t <= 2022-09-05 00:00`, val
`(2022-09-05, 2022-09-06]`, test `(2022-09-06, CUTOFF]`. With any lookback,
the last training window therefore ends exactly at Sept 5 00:00, so no
training transaction is from Sept 5 or later.

**Labels.** Edge label = `Is Laundering`, including laundering transactions
that belong to no campaign. The positive rate is ~0.1%, so training uses a
class-weighted loss with weight `n_negative / n_positive` per window — about
**1,000** at that rate (measured: 1,014 on a 400k-transaction window at a
0.098% positive rate).

**Edge features** (32 dims): `log1p` of amount paid and amount received, a
same-currency flag, hour-of-day as sin/cos, a self-transfer flag
(`src == dst`), a cross-bank flag (`From Bank != To Bank`), day-of-week
one-hot (7), and one-hot payment currency, receiving currency and payment
format against vocabularies fitted on training windows, each with a
trailing `__other__` column for categories unseen in training.

**Node features** (6 dims), computed from the current window only:
in-degree, out-degree, `log1p` total in-amount (sum of amount received when
the account is the receiver), `log1p` total out-amount (amount paid when it
is the sender), distinct counterparties, and distinct currencies — both
pooled over incoming and outgoing transactions.

**What is never a feature:** account identity, bank identity, institution,
anything label-derived, and anything from at or after `t`. Banks enter only
as the equality `From Bank != To Bank` and accounts only as `src == dst`;
nothing identifies *which* bank or account is involved. Each of these is a
property test, not a convention — see Testing.

**Expected GPU constraint (estimated, pending the `features-smoke` run).**
From tensor shapes, a 3-layer GINEConv with hidden 64 trained **full-batch**
needs roughly: ~2.3 GB on a quiet 24h window (210k edges), ~10 GB on a busy
24h window (1.1M edges), ~16.6 GB at L = 48 on the busiest days (1.9M
edges), and ~18.6 GB at L = 72 (2.1M edges). Kaggle offers 16 GB GPUs
(T4 ×2, or P100), so **full-batch training is feasible at L = 24 and
marginal at L = 48 on the busiest windows**; L = 48 will likely need the
neighbour sampling the design already permits ("train with
LinkNeighborLoader-style sampling if full windows do not fit in GPU
memory"). The L = 72 figure is one of the reasons that lookback was dropped.
These are estimates, not measurements — `features-smoke` on the full data
replaces them with real figures, reporting the largest **candidate** window
separately from the excluded one.

### Oracle ceiling

The oracle scorer (score = true label) is the best achievable under the
frozen matching rule **using only the laundering edges**: a cluster cannot
hit a campaign until three of its accounts are visible, and the oracle sees
every laundering edge the moment it appears. Its median lead time and
median `frac_observed` are the ceiling every later result is read
against — a model at 60% of the oracle's lead time is doing well, whereas
60% of a theoretical 100% would be meaningless.

> **The ceiling is a reference, not a strict upper bound** (corrected
> 2026-11-09). The README previously said no detector could beat it. That
> is wrong, and the stage 3.3 run found the counterexample. The oracle
> keeps *exactly* the laundering edges; a model keeps whatever scores above
> tau, which is **neither a subset nor a superset** of them. Keeping every
> laundering edge is not always best: it can merge a campaign into a
> cluster too impure for the 50% rule, where a sparser edge set would have
> matched.
>
> **Measured.** At L = 48 on stress, `xgboost_centralized` detects
> **campaign 295** — a 7-transaction, 7-account CYCLE — in **all five
> seeds**, and the oracle does not. The oracle's own diagnosis says why:
> 295 is the single `purity_failure` in the entire full-data run. Its own
> laundering edges do connect 5 accounts, comfortably past the 3 the rule
> needs, but at L = 48 the full laundering edge set merges it with **9
> other campaigns** into a 13-account cluster at purity **0.4615**, just
> under the 0.5 required. The oracle detects it at L = 24, where the
> shorter window does not reach the bridging edges.
>
> The model, thresholded at tau ≈ 0.99, keeps only the edges it scores
> most confidently, drops part of that bridge, and is left with a cluster
> that clears both conditions. That the campaign is won back by keeping
> *fewer* edges is what the evidence supports: a cluster can only lose
> purity by gaining accounts, so a model firing on background edges could
> not have rescued this one. (The run does not export the model's cluster
> membership, so the selectivity step is an inference from the oracle's
> diagnosis rather than a direct observation; exporting the cluster for 295
> at its detection time would settle it.)
>
> A second route exists in principle and has **not** been observed here: a
> campaign whose own laundering edges never connect 3 accounts
> (`structurally_disconnected`) could in theory be bridged into a matchable
> cluster by background edges a model scores above tau. Nothing in this run
> shows it.
>
> Consequences, all now implemented: `recall_relative_to_ceiling` **can
> exceed 1** and is never clamped; on the `oracle_reachable` subset it is
> bounded by 1 only because that subset counts reachable campaigns alone; a
> detection outside the reachable set shows up as `all` detections
> exceeding `oracle_reachable` detections, which is how this was found.
> `tests/test_evaluation.py` pins the case and fails if the ratio is
> clamped. In the stage 3.3 run the ratio never actually exceeded 1 — the
> model misses 17–18 reachable stress campaigns for the one it adds — but
> the measure must not assume it cannot.

**Measured on the full data, 2026-10-15** (subset `all`):

| L | Split | Oracle recall | Median lead time | Median `frac_observed` |
|---|---|---:|---:|---:|
| 24 | val | 0.731 | 68.5h | — |
| 24 | test | 0.821 | 74.7h | **0.25** |
| 24 | stress | 0.724 | 81.7h | — |
| 48 | val | 0.769 | 67.3h | — |
| 48 | test | **0.893** | 72.1h | **0.25** |
| 48 | stress | 0.737 | 81.5h | — |

Each median lead time in this table is taken **over that row's own detected
campaigns**, so the medians in different rows are not comparable with each
other. A row's median is a description of that configuration, not a term in a
subtraction. Comparisons between rows use the paired method below.

Read these as the budget every later result is spent against. Two things
stand out:

- **A detector should not aim for 100% recall.** The oracle itself reaches
  only 0.82 (L = 24) / 0.89 (L = 48) on test, so roughly 11–18% of evaluable
  test campaigns are out of reach *of the laundering edges* under the frozen
  matching rule. Reporting a model's recall as a fraction of 1.0 would
  understate it; the honest denominator is the oracle — bearing in mind that
  a model keeping a different edge set can occasionally exceed it, as one
  does on stress. §[Why the oracle misses](#why-the-oracle-misses-campaigns)
  explains where that gap comes from.
- **L = 48 buys recall and costs no earliness. Tested, 2026-10-17.** The
  paired comparison settles what the two medians could not: on the
  campaigns both lookbacks detect, the per-campaign lead-time difference is
  **0.0h with a 95% CI of [0.0, 0.0] on every split**, and the same for
  normalized lead time and `frac_observed`. There was no earliness
  trade-off; the apparent 74.7h → 72.1h drop was entirely the changed
  composition of the detected set. Coverage moves one way only on test:
  L = 48 detects **4 campaigns L = 24 misses** (187, 194, 238, 245) and
  misses none that L = 24 finds. On stress it gains 2 (261, 282) and loses
  1 (295); on val it gains 1 (173) and loses none.

| Split | Campaigns both detect | Paired difference (L48 − L24) | 95% CI | Only L = 48 | Only L = 24 |
|---|---:|---:|---|---|---|
| val | 19 | 0.0h | [0.0, 0.0] | 1 (173) | 0 |
| test | 46 | 0.0h | [0.0, 0.0] | 4 (187, 194, 238, 245) | 0 |
| stress | 54 | 0.0h | [0.0, 0.0] | 2 (261, 282) | 1 (295) |

  Evidence: `docs/reference/oracle_diagnose.{md,json}`, Kaggle, commit
  `eacb6d3`. Of the 46 paired test campaigns 40 tied exactly, 5 were
  earlier under L = 48 and 1 under L = 24 — so even the sign counts carry
  no useful direction, and the median difference of 0.0h is not an artefact
  of a tie-heavy distribution being summarised away.

Median `frac_observed` of 0.25 on test means the oracle fires when about a
quarter of a campaign's transactions have happened — earlier than the
matching rule's three-account floor might suggest, because campaigns tend to
front-load their account activity.

Synthetic development figures, kept only as a code check: median lead 32.9h
(val) / 25.0h (test) / 27.1h (stress), `frac_observed` 0.40 / 0.33 / 0.33.

### Comparing two configurations: the paired method

**Frozen.** Whenever two configurations are compared — lookbacks here, and
the four conditions in Phase 4–5 — earliness and coverage are reported as two
separate things, and neither is allowed to stand in for the other.

1. **Earliness, paired.** Restrict to the campaigns **both** configurations
   detect, take the per-campaign difference in lead time, and report its
   median with a seeded bootstrap CI over campaigns, plus the counts of
   campaigns where each side was earlier and where they tied. The sign
   convention is `b − a`, so **positive means `b` detected earlier**. The
   same pairing is applied to `normalized_lead_time` and `frac_observed`.
2. **Coverage, separately.** Report how many campaigns each configuration
   detects that the other misses, **by campaign id**, so the discordant sets
   can be inspected rather than assumed to be random.
3. **Directly, against each other.** To compare A and B, pair **A against
   B**. Never infer an A-versus-B difference from `A vs C` and `B vs C`,
   however natural the common reference C is — the oracle ceiling, the
   centralized condition, anything. Each of those comparisons is taken over
   *its own* shared set, so the two numbers are not on a common scale and
   their difference means nothing. Pairing fixes the denominator for one
   comparison; it does not make two comparisons subtractable.

Comparing two medians of `median_lead_time_h` is invalid and is not done
anywhere in this project. Each median is over a different set of campaigns,
so a configuration that detects strictly more campaigns usually shows a
*lower* median lead time — the extra campaigns it reaches are the slow ones
detected late. A real improvement then reads as a regression. What changed is
the composition of the set, not the earliness.

Implemented once, in `evaluation.paired_lead_time_comparison`, and reported
by `oracle-diagnose` as `oracle_diagnose_paired.csv` plus a section in
`oracle_diagnose.{md,json}`. `evaluation.describe_paired_comparison` writes
the sentence, and refuses to state a direction when the CI includes zero.
`tests/test_paired_comparison.py` pins it, including the two cases where
the shortcuts mislead: unpaired medians saying the better detector got
worse by 45 hours while the paired difference is zero with one extra
campaign detected, and two comparisons against a common reference implying
a 25-hour difference where the direct comparison is a tie with one side
simply covering more.

Both failures have happened here, which is why rule 3 is written down. The
first produced the withdrawn claim that L = 48 lost earliness
(2026-10-17); the second produced the withdrawn claim that dropping payment
format cost 15 hours of earliness (2026-10-18, corrected 2026-10-19). The
same shape twice: a difference read off two numbers that were never on the
same scale.

### Stage 3.3: the XGBoost baseline

The non-graph control for the gate question. Same windows, same causal
bounds, same features; one transaction per row, so no message passing and
no neighbourhood beyond the two endpoints.

**Two variants, frozen 2026-10-17.** Identical in every respect except the
columns the second drops:

| Variant | Features | Dropped |
|---|---|---|
| `xgboost_centralized` | all | — |
| `xgboost_centralized_no_format` | all but the payment-format one-hots | every `fmt_*`, including `fmt___other__` |

**Why the ablation exists.** In the IBM synthetic data, laundering is
strongly linked to payment format — on the 1% sample `fmt_ACH` alone
carries 46% of the model's gain and the format columns together carry ~87%.
Payment format is an **allowed** input under the frozen design: it is a
property of the transaction, not an identity or a label. But a feature that
dominant in *synthetic* data is a shortcut a model can take instead of
learning transactional structure, and a result resting on it would say more
about the data generator than about campaign discovery. So **every model
from here on — the GNN and the federated conditions included — is reported
with and without it**, and the gap between the two is part of the result.
The variants are built from one matrix (the ablated one is a column slice),
so the two see byte-identical windows, labels and weights.

**Five seeds.** Every configuration is trained at seeds 0–4, and results
are reported per seed and as mean [min, max] across seeds, so a difference
is read against seed noise rather than taken from a single run.

**Hyperparameters, frozen and identical for every variant, lookback and
seed** (tuning per configuration would make a comparison a comparison of
tuning effort):

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 1.0 |
| `reg_lambda` | 1.0 |
| `tree_method` | `hist` |
| `max_bin` | 256 |
| `objective` / `eval_metric` | `binary:logistic` / `aucpr` |
| `seed` | 0–4 (the only source of randomness) |

**Training.** One model per (variant, lookback, seed), over every
rule-3-valid training detection time pooled into a single matrix — 13
windows at L = 24, 9 at L = 48. Pooling is what makes one scorer usable at
every detection time; rule 3 keeps a silently clamped short window out of
the training distribution. The class weight `n_negative / n_positive` is
applied **per window**, as a per-row weight: a positive carries its own
window's weight and a negative carries 1.0. Pooling the weight instead
would let a busy window's balance set the weight for a quiet one.

**What the model does not have.** No tau — it emits probabilities, and the
threshold is chosen on validation only by `evaluation.select_tau`. No
campaign knowledge: labels are per-transaction `Is Laundering`, including
laundering that belongs to no campaign. No features of its own: they come
from `features.py` unchanged, so the leakage properties already tested
there apply without a second implementation to keep honest.

#### The model tau grid (frozen 2026-10-18)

Model scorers — XGBoost now, the GNN and the federated conditions later —
all use **one** grid:

```
0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
0.995, 0.999, 0.9995, 0.9999
```

The oracle and random scorers keep `DEFAULT_TAU_GRID`
(`0.5, 0.75, 0.9, 0.95, 0.99`). The split exists because the two kinds of
scorer are genuinely different: a sanity scorer's scores are exactly 0 or
1, so every threshold in (0, 1] keeps the same edges and the choice is
vacuous. A class-weighted model's scores are not merely spread over the
unit interval — they **pile up against 1**.

Both ends of this grid were found by running into them, which is why it has
an odd shape:

- **The low end**, added 2026-10-17: on the 1% sample validation F1 peaked
  at 0.3, below `DEFAULT_TAU_GRID`'s floor of 0.5, so that grid was
  truncating the optimum rather than finding it (0.821 against 0.803).
- **The near-1 tail**, added 2026-10-18: on the full data (L = 24, seed 0)
  F1 then rose from **0.128 at tau 0.95 to 0.543 at 0.99** and the sweep
  chose 0.99, the top of the grid. A jump that large across the final
  interval means the optimum lies beyond it. With per-window class weights
  of **495–3,461**, scores concentrate so close to 1 that a linear grid
  cannot resolve the region where the decision actually happens. The tail
  halves the remaining distance to 1 at each step, so the resolution is in
  `1 − tau` rather than in tau.

One grid for every model and condition means a comparison between them is
never a comparison of how far each was allowed to search.

Cost is asymmetric. Clustering scales with the edges kept, so the low
thresholds are the expensive ones and the tail is nearly free — almost
nothing survives 0.999. `select_tau` records **the seconds each tau took**
and the reports carry them. The score cache is warmed before the sweep, so
those seconds are clustering alone and comparable across taus; without it
the first tau would silently absorb every window's featurisation. Caching
cannot change the curve, since a scorer's output does not depend on tau,
and `tests/test_tau_and_budgets.py` pins the cached and uncached curves
equal.

**If a chosen tau lands on an edge of the grid, `evaluate` writes every
report and then exits 3 with a STOP message.** An edge tau is a boundary,
not a maximum; that is how the tail came to be added. The grid is frozen
for every model and condition, so widening it is a decision to take
deliberately, not something to do mid-run. If a chosen tau ever reaches
**0.9999**, more nines are not the answer: that would mean the scores are
saturated, and the thing to examine is the class weighting itself — a
design decision, not a grid detail.

#### Stage 3.3 results (full data, five seeds) — complete

**Run:** Kaggle CPU notebook at commit `de48b35`, 20 boosters (2 variants ×
2 lookbacks × 5 seeds), **exit 0 with no tau at a grid edge**. Evidence:
`docs/reference/train_baseline.*` and `evaluate_baseline.*`.

Test split, `all` subset, mean [min, max] over seeds 0–4:

| Variant | L | Recall | vs ceiling | FA/day | Median lead | Seed sd |
|---|---:|---:|---:|---:|---:|---:|
| `xgboost_centralized` | 24 | 0.654 [0.643, 0.661] | 0.796 | 3.8 | 67.2h | 0.009 |
| `xgboost_centralized` | **48** | **0.679 [0.661, 0.696]** | 0.760 | 10.5 | 68.2h | 0.016 |
| `xgboost_centralized_no_format` | 24 | 0.479 [0.393, 0.554] | 0.583 | 4.2 | 62.6h | 0.052 |
| `xgboost_centralized_no_format` | 48 | 0.532 [0.500, 0.571] | 0.596 | 8.6 | 64.7h | 0.026 |

**Selected configuration: the full model at L = 48**, chosen by the frozen
protocol — on **validation**, recall 0.531 against 0.385 for L = 24. Both
lookbacks are reported, because the validation margin did not survive to
test:

- On validation the gap is large: 0.531 against 0.385, about **3.8 of 26**
  campaigns.
- On test it is about **1.4 of 56** — paired per seed, L = 48 adds 1–3
  campaigns and loses 0–1, at **0.0h earliness on the campaigns both
  detect** (CI [0.0, 0.0] in every seed).
- And it costs roughly **3× the false alarms**, 10.5 a day against 3.8.

So the selection follows the protocol, but the honest summary is that the
longer lookback buys one or two extra campaigns for three times the alarm
load, and the validation estimate of its benefit was optimistic. Stage 3.4
should carry both lookbacks rather than treat L = 48 as settled on this
margin.

**The format ablation is settled: format buys coverage, not earliness.**

- **Earliness:** the direct paired ablation's CI includes zero in **all ten
  test cells** (2 lookbacks × 5 seeds). Nothing is demonstrated, which
  retires the withdrawn 15-hour claim for good.
- **Coverage:** the full model detects **6–14 test campaigns per seed** that
  the no-format variant misses; the reverse is **0–1**. On the ceiling
  scale that is 0.760–0.796 against 0.583–0.596.
- **Stability:** no-format is markedly noisier across seeds — sd **0.052**
  at L = 24 against **0.009** for the full model, a five-fold difference.
  Without the format columns the model is leaning on weaker, more
  seed-sensitive signal.

Read together: payment format is doing a large share of the work, and what
it contributes is *finding* campaigns rather than finding them *sooner*.
Since the link between format and laundering is an artefact of the IBM
generator, the no-format column is the one that should be believed to
transfer, and stage 3.4's GNN has to be read with both in view.

**One campaign beats the ceiling.** At L = 48 on stress the full model
detects campaign 295 in all five seeds, which the oracle declines — see
§[Oracle ceiling](#oracle-ceiling) for the mechanism and what it changes.

**Cost, measured against the estimate** (CPU notebook, 30 GB):

| | Estimated | Measured |
|---|---:|---:|
| `train`, 20 boosters | ~1.5 h | **1.45 h** |
| `evaluate`, 20 models | ~2 h | **1.59 h** |
| Per booster, L = 24 / L = 48 | 215s / ~290s | 207–217s / 282–289s |
| Peak memory, L = 24 | 9.9 GB | **9.9 GB** |
| Peak memory, L = 48 | 12–13 GB | **15.9 GB** |

About **3 hours** end to end. The L = 48 memory estimate was ~3 GB low —
scaling the L = 24 peak by row count under-counts, because the pooled
matrix and the per-window matrices coexist during `np.vstack` and both
grow. 15.9 GB fits a 30 GB CPU notebook comfortably. *(Corrected 2026-11-10:
this previously said 15.9 GB would not fit a GPU notebook's "13–16 GB".
That was wrong — it confused the GPU's **device** memory with the
notebook's **system** RAM. `gpu-check` measured the Kaggle GPU notebook at
**33.7 GB of system RAM**, so XGBoost's matrix would fit there too. Stage
3.3 still belongs on CPU, for the reason that always applied: its
bottleneck is pandas and scipy, and 3.4 needs the GPU quota.)*

*(Two earlier single-seed quick checks are superseded by this run. The
first, at commit `5664a6b`, chose tau at the top of the grid and exited 3,
which is what prompted the near-1 tail; the second, at `6a6d1be`, exited 0
with interior taus. Their numbers — 0.661 and 0.464 test recall at L = 24,
seed 0 — are consistent with the five-seed means above.)*

### Stage 3.4: the centralized GNN

Not trained yet. The design was frozen on 2026-10-10 and nothing here
changes it: GINEConv edge classification, 2–3 layers, hidden 64, message
passing **in both directions** (every edge also inserted reversed, with a
direction flag), and an edge readout MLP on `[h_src, h_dst, edge_features]`.
**Full-batch training at both lookbacks** — one whole window per step —
decided 2026-11-10 on the measurement below, replacing the neighbour
sampling frozen on 2026-10-10 when the memory estimate was twice the truth.
One regime for both lookbacks either way, so the L comparison is never
confounded by the training method. Inference scores full windows
**unchunked**: the largest evaluation window needs 4.75 GB of a 15.6 GB
card.

Everything around the model is shared with stage 3.3 rather than
reimplemented — the same window features, labels, rule-3 training times,
class-weighting principle, campaign extraction, matching rule, model tau
grid, alert budgets, subsets and paired comparisons. **No account identity
and no learned per-account embeddings**: a node's only inputs are its six
window features. Graphs are built per window `[t − L, t)` and nothing
else. Seeds 0–4, both variants (full and no-format), both lookbacks.

It runs in sub-stages, each stopping for approval:

| | Sub-stage | Status |
|---|---|---|
| 3.4.0 | GPU environment check — no model, no data | **done** (Kaggle, `2755139`, exit 0) |
| 3.4.1 | Design | **approved 2026-11-10** with four additions — see §[the design](#341-design--approved-2026-11-10-with-four-additions) |
| 3.4.2 | Implementation, tests, resumability, local CPU smoke run | **done** — `models/gnn.py`, 25 tests, `train-gnn` / `evaluate-gnn`, smoke run on the real sample |
| 3.4.3 | Kaggle quick check: 1 seed, L = 24, full variant — GPU training then CPU evaluation | **done** (Kaggle, `8d8929c`, both sessions exit 0) — see §[the quick check](#343-quick-check--done-provisional-one-seed) |
| 3.4.4 | The full run: 20 GNN models, the XGBoost retrain, the direct comparison | **done** (2026-11-14) — three sessions, all exit 0; the retrain reproduced every stage 3.3 number. See §[the results](#344-the-full-run--done) |
| 3.4.5 | Phase 3 gate report: GNN vs XGBoost, paired comparisons only | **done** — and it locates the shortfall. See §[why the GNN loses](#345-the-gate-report--why-the-gnn-loses-at-extraction-not-at-ranking) |

#### 3.4.0: what the check does

`kaggle_runner.py gpu-check` reports what a GPU notebook actually is,
before any design is sized against guesses. It reports the accelerator
(name, total and free memory, compute capability, count), the host (Python
version, CPU count, total and available RAM), and the torch / CUDA / cuDNN
/ PyG versions.

It then installs `torch_geometric` and attempts each optional extension —
`pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster` — from the PyG
wheel index matching the image's torch and CUDA, reporting each as
installed or failed **without aborting**. The install itself lives in
`ensure_torch_geometric`, which **every** GPU subcommand calls, so none of
them depends on another having run first in the session; only `gpu-check`
asks it for the optional extensions, since a command that needs to train
should not spend minutes installing wheels it can run without. None is required: GINEConv and
neighbour sampling work without them, they are only faster, and their
wheels often lag the newest Python. Which of them lands is worth knowing
before a training session rather than during one.

Finally it runs a GINEConv forward and backward **at the real feature
widths** — 3 layers, hidden 64, 54 edge features plus the direction flag,
6 node features, every edge also reversed — at 50k, 200k and 500k edges,
reporting seconds and peak GPU memory for each. An out-of-memory is caught
and reported as a result rather than a crash, and the probe stops at the
first size that fails. Those numbers are the input to 3.4.1.

Writes `gpu_check.{md,json}`; **exits 2** if there is no usable CUDA device
or the smoke test fails, after reporting everything it could still
determine. `--no-install` skips the pip steps and `--probe-edges` changes
the sizes.

**Stage 3.3's 15.9 GB does not constrain the GNN.** That figure is
XGBoost's pooled **host** training matrix — 9.1M rows × 66 features held at
once in system RAM. The GNN processes one window at a time and never
materialises that pool, and its constraint is **device** memory on a 15.6
GB card, which is a different quantity in a different place. Measured in
3.4.0 and `gpu-probe-windows`, never inherited from 3.3.

**torch and torch_geometric are still not in `requirements.txt`.** The
check installs PyG itself on Kaggle, and local development does not need
the CUDA stack until the model exists in 3.4.2. They land with that
sub-stage, the same rule every other dependency has followed.

#### 3.4.0: measured environment (Kaggle, commit `2755139`, exit 0)

| | Measured |
|---|---|
| Python | 3.13.16 |
| torch / CUDA | 2.11.0+cu130 |
| torch_geometric | 2.8.1 |
| PyG extensions | **all four installed** — `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster` |
| GPU | **2 × Tesla T4, 15.6 GB each** |
| System RAM | **33.7 GB** |

GINEConv 3×64 forward+backward on synthetic graphs at the real feature
widths:

| Edges | Peak GPU | Seconds |
|---:|---:|---:|
| 50,000 | 251 MB | 0.07–0.16 |
| 200,000 | 936 MB | 0.07–0.16 |
| 500,000 | 2,302 MB | 0.07–0.16 |

**That is ~4.6 MB per 1,000 edges, and it changes the stage 3.4 design
question.** Projected onto the real largest training windows:

| | Edges | Projected | Earlier shape-based estimate |
|---|---:|---:|---:|
| Largest L = 24 training window | 1,114,921 | **~5.1 GB** | — |
| Largest L = 48 training window | 1,869,370 | **~8.6 GB** | ~16.6 GB |

The shape-based estimate that justified neighbour sampling was roughly
**twice** the measured scaling. If the projection holds, **full-batch
training fits both lookbacks on one T4**, and the sampling decision of
2026-10-10 was taken on a number that no longer stands. It is therefore
reopened, and 3.4.1 decides it on a measurement rather than a projection —
`gpu-probe-windows` below runs the real windows, because a synthetic graph
with random edges has a different account-degree distribution from a real
window and the projection is only linear until something is not.

#### 3.4.1 evidence: `gpu-probe-windows`

Before any design is proposed, this runs **the real thing**: the largest
rule-3 training window at each lookback (L = 24: 2022-09-02 00:00,
1,114,921 edges; L = 48: 2022-09-03 00:00, 1,869,370 edges) and the largest
evaluation window (2022-09-10 00:00 at L = 48, 1,137,240 edges), with real
features from a spec fitted on **every** rule-3 training window exactly as
training fits it — a spec fitted on one window would have narrower
vocabularies, fewer edge features, and would understate the memory this
exists to measure.

Each training window goes through GINE 3×64 + the edge readout MLP + the
class-weighted loss + an Adam step, forward and backward, on **one** T4 —
one card, because a window has to fit on a single device and using both
would measure something no training step reproduces. Three steps are
timed, with the first reported separately since it pays for allocator
warm-up. The evaluation window is scored unchunked, to find out whether
chunking is needed at all.

It reports peak GPU memory **allocated and reserved** — reserved is what
the allocator holds and therefore what runs out — plus MB per 1,000 edges,
step time, and the time to build and featurise each window. The largest
windows are *found* by a full scan of every usable detection time rather
than remembered, and checked against what `features-smoke` recorded, so a
mismatch is visible instead of silent. An out-of-memory is caught and
reported as a measurement; the command exits 3 if any probe failed.

**The decision rule, fixed in advance:** full-batch is proposed only if the
largest L = 48 window peaks below **12 GB reserved**, leaving headroom on a
15.6 GB card for real-data variation and allocator fragmentation.
Whichever regime is chosen applies to **both** lookbacks, so the lookback
comparison is never confounded by the training method.

`--allow-cpu` runs the whole path off a GPU. It **measures nothing** — host
and device memory are different quantities and no CPU timing resembles a
T4's — and says so in its output; it exists so the code can be verified
before a GPU session is spent on it, which is how the unconditional
`torch.cuda.reset_peak_memory_stats()` call was found, and how the
self-install path above was tested (by uninstalling PyG locally and
watching the probe put it back).

#### 3.4.1 evidence: measured (Kaggle, one T4, 2026-11-10)

All four probes completed, and every window matched what `features-smoke`
recorded. Peaks are **reserved** — what the allocator holds, so what runs
out:

| | Edges | Accounts | Positives | Class weight | Peak reserved | MB/1k edges | Step |
|---|---:|---:|---:|---:|---:|---:|---:|
| L = 24 train | 1,114,921 | 437,073 | 322 | 3,461 | **6.39 GB** | 5.73 | 0.28s |
| L = 48 train | 1,869,370 | 509,993 | 730 | 2,560 | **10.70 GB** | 5.73 | 0.45s |
| L = 24 eval | 654,467 | 346,684 | 514 | 1,272 | 2.71 GB | 4.13 | 0.12s |
| L = 48 eval | 1,137,240 | 354,229 | 1,053 | 1,079 | 4.75 GB | 4.18 | 0.18s |

**Full-batch training fits both lookbacks.** The largest L = 48 window
peaks at 10.70 GB against the 12 GB rule fixed before the run, with 4.9 GB
left on a 15.6 GB card. The 2026-10-10 sampling decision was taken on an
estimate of ~16.6 GB; the measurement is 10.70.

**The real windows are ~25% heavier than the synthetic ones**: 5.73 MB per
1,000 edges against 4.6 from `gpu-check`, which projected 8.6 GB for L = 48
where the truth is 10.70 GB. Still inside the budget, but with
meaningfully less headroom than the projection implied — which is why the
projection was not treated as the answer.

**Inference needs no chunking.** The largest evaluation window scores
unchunked in 0.18s at 4.75 GB, less than half the card.

**Host memory peaked at 14.1 GB**, during the L = 48 feature-spec fit — the
same pooling cost stage 3.3 pays — against 33.7 GB available. The GPU is
not the binding constraint; fitting the spec over 9 or 13 training windows
is the heaviest host step.

**One result needs explaining before training, and changes what 3.4.1 must
decide.** The reported losses are **10,615 at L = 24 and 356,394 at
L = 48** after a single step from random initialisation. A class-weighted
BCE on these windows should start near 1–2. Numbers that size mean the
logits are enormous, and the likely cause is structural rather than a bug
in the probe, which ran the frozen architecture faithfully: **GINEConv
aggregates by sum**, and these windows contain accounts of very high
degree. Summing hundreds or thousands of 64-dimensional messages, three
layers deep, with no normalisation between them, compounds into activations
far outside the range a linear readout can map to sensible logits — and the
largest window, which has the most extreme hub degrees, is 34× worse than
the smaller one, which is the pattern that explanation predicts.

The frozen design specifies GINEConv, 2–3 layers and hidden 64, and says
nothing about normalisation. **3.4.1 therefore has to propose one** — layer
normalisation between convolutions, degree-normalised aggregation, or
another remedy — with the reasoning, as a change to be approved rather than
assumed. Nothing is trained until that is settled: a model whose activations
explode on the busiest windows would train, report numbers, and mean
nothing.

#### 3.4.1 design — **approved 2026-11-10, with four additions**

Approved as proposed: the balanced-mean loss, layer normalisation after
each convolution, sequential training on one GPU with parallelism as a
contingency, one feature spec per lookback fitted once and shared, and
window-major evaluation in a CPU session. Four additions came with the
approval and are part of the design:

1. **`log1p` on the heavy-tailed node counts, inside the GNN only** — the
   root cause of the extreme logits, below.
2. **Early stopping on validation edge-level average precision**, not
   campaign-F1.
3. **One frozen hyperparameter configuration**, as XGBoost had.
4. The parameters below are stated explicitly so nothing is implicit.

Every number is measured unless it says otherwise.

##### 1. The loss scale, and why it was 34× worse at L = 48

The probe reported initial losses of **10,615 (L = 24)** and **356,394
(L = 48)** from random initialisation. Three candidate causes, two of which
the arithmetic rules out:

- **Not the reduction.** It is already `mean`. A `sum` reduction would have
  given ~6.7 × 10¹¹, not 3.6 × 10⁵.
- **Not the class weight.** With `pos_weight = n_neg / n_pos`, the positive
  block's weights sum to `n_pos × pos_weight = n_neg ≈ N`, so after mean
  reduction it contributes ≈ 1, whatever the window. Measured: 322 × 3,461
  = 1,114,600 against N = 1,114,921 (ratio **0.9997**), and 730 × 2,560 =
  1,868,639 against 1,869,370 (**0.9996**). The weighting is self-normalising
  and is not the cause.
- **The logits.** With both of the above neutral, the loss is close to the
  mean |logit| of the mis-signed side, so **logits are of order 10⁴–10⁵**.

Why they get that large: **GINEConv aggregates by sum**, and the inputs are
already extreme. Measured on the 1% sample's largest windows, *after* the
feature spec's standardisation, node features reach **|x| = 110 (L = 24)
and 159 (L = 48)** — hub accounts sit a hundred-plus standard deviations
out, because degree and amount are heavy-tailed and standardising a
heavy-tailed column does not make it bounded. Edge features are well
behaved (max 4.6). Three layers of summing 64-dimensional messages over
high-degree accounts then compound that, with nothing in between to rescale
it. The ratio between the two windows follows: the sample's maximum degree
is 231 at L = 24 and 463 at L = 48, and a cube of that ratio is ~8, in the
same direction as the measured 34× on full data, where the degrees are far
larger.

**Proposed fix, two parts.**

*(a) A balanced-mean loss*, so every window contributes comparably by
construction rather than approximately:

```
L = ½ · mean over positives of BCE  +  ½ · mean over negatives of BCE
```

**How the class weight enters:** each positive carries weight
`1 / (2 · n_pos)` and each negative `1 / (2 · n_neg)`, so the ratio of
per-positive to per-negative weight is exactly **`n_neg / n_pos`** — the
frozen class-weighting principle, unchanged. This is the same objective as
`pos_weight = n_neg/n_pos` under mean reduction, up to the factor
`n_neg / N` measured above at 0.9996–0.9997 and a constant 2 that the
learning rate absorbs. It is adopted not to change the objective but to
make window-invariance exact and the number readable: at initialisation it
is **ln 2 ≈ 0.693** for any window, so a loss that is not ≈ 0.69 on the
first step is a bug rather than a mystery.

*(b) `log1p` on the heavy-tailed node counts, inside the GNN only* —
added on approval, and the root cause rather than a mitigation. The shared
spec standardises **raw** counts, and standardising a heavy-tailed column
does not bound it. The GNN therefore applies `log1p` to `in_degree`,
`out_degree`, `n_counterparties` and `n_currencies` **before** standardising,
with its own statistics fitted on training windows only (the shared spec's
were computed on raw counts and do not describe a transformed column).
`log_in_amount` and `log_out_amount` are already `log1p` in the shared
features and are left alone.

**The shared feature spec is untouched, and so is XGBoost.** A tree splits
identically under any monotone transform of an input, so this changes
nothing for stage 3.3 even in principle, and the stage 3.3 results stand.

Measured on the 1% sample's largest training windows, maximum |x| per
column, shared spec against GNN:

| | Raw max | Shared spec | GNN | |
|---|---:|---:|---:|---|
| `in_degree` | 5 / 7 | 8.4 / 11.6 | **4.0 / 4.7** | log1p |
| `out_degree` | 231 / 457 | 108.4 / 156.6 | **13.7 / 15.3** | log1p |
| `log_in_amount` | 23 / 23 | 4.1 / 4.1 | 4.1 / 4.1 | — |
| `log_out_amount` | 22 / 23 | 3.9 / 4.2 | 3.9 / 4.2 | — |
| `n_counterparties` | 228 / 455 | 109.9 / 158.5 | **40.4 / 42.2** | log1p |
| `n_currencies` | 3 / 7 | 18.0 / 48.3 | **16.1 / 29.8** | log1p |
| **overall** | | **109.9 / 158.5** | **40.4 / 42.2** | |

(L = 24 / L = 48.) Degree improves ~8–10×, and the overall maximum falls
~3–4×. It does **not** bound the inputs: `n_counterparties` still reaches
40, because standardising divides by a standard deviation that a tightly
concentrated bulk makes small, and `n_currencies` barely moves at all since
it takes only a handful of values and standardisation normalises away any
monotone transform of a near-binary column. **That is why LayerNorm is kept
as well rather than instead.**

*(c) Layer normalisation after each GINEConv.* Alternatives considered:

| Option | Why not |
|---|---|
| **LayerNorm per node, after each conv** (recommended) | Degree- and size-independent, no train/test statistics, keeps sum aggregation and so keeps GINEConv's expressiveness. Smallest change that removes the compounding. |
| BatchNorm over nodes (as the original GIN paper) | Its running statistics are fitted on training windows and applied to evaluation windows whose size and degree distribution differ by up to 3×. A train/test mismatch introduced by the fix. |
| Mean instead of sum aggregation | Fixes the scale but discards the injectivity that makes GIN a GIN; a different model, not a normalisation. |
| Clamping logits or shrinking the init | Treats the symptom; the activations still compound. |

The training loop will log **max |logit| and the per-layer activation norm
on the first step of each epoch**, so if the explanation is wrong that is
visible immediately rather than inferred from a loss curve.

##### 2. Memory safety

10.70 GB of 15.6 is a 4.9 GB margin, held across many epochs rather than
one step, so:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set by the runner
  before torch is imported, so the allocator grows a segment instead of
  fragmenting into unusable blocks over hundreds of differently-sized
  windows.
- `torch.cuda.empty_cache()` between windows, and `reset_peak_memory_stats()`
  at each epoch boundary.
- **Peak allocated and reserved logged per epoch**, written into the
  training report, so drift is visible rather than discovered by an OOM in
  hour three.
- **On OOM: fail loudly.** Report the window, its edge count, the epoch and
  the peak, write the partial report, exit non-zero. **Never fall back to
  sampling, smaller batches or chunking silently** — a run that quietly
  changed training regime partway would produce numbers that cannot be
  compared with anything, including themselves.

##### 3. Two GPUs: stagger the expensive part, then parallelise safely

The 14.1 GB host peak is **the feature-spec fit**, not training: it pools
every rule-3 training window's transactions to build vocabularies and
scaler statistics. Two concurrent processes doing that would need ~28 GB of
33.7 before counting the dataframes, which is why naive two-process
parallelism is the risk here.

**Recommended:** fit each lookback's spec **once** in a short step, save it
to disk (it is a small JSON of vocabularies and scaler statistics — the
format `FeatureSpec.save` already writes), and have every training process
**load** it. A training process then holds the dataframe (~3.9 GB) plus one
window's features (~0.25 GB) rather than a 14.1 GB pool, so two processes
fit comfortably and `CUDA_VISIBLE_DEVICES=0` / `=1` can run two seeds at
once.

**But it is probably unnecessary**, which is the more useful finding. At
the measured step times, an epoch is 13 × 0.28s = **3.6s** at L = 24 and
9 × 0.45s = **4.1s** at L = 48. Even at 100 epochs that is under 7 minutes
of compute per model, and the whole 20-model matrix is ~2 hours on **one**
GPU — inside a single 9-hour session. The proposal is therefore: **run
sequentially on one GPU, with the spec pre-fitted and shared**, and keep
two-process parallelism as a contingency if the epoch count turns out much
larger. Parallelism that is not needed is a host-RAM risk taken for
nothing.

##### 4. Train on GPU, evaluate on CPU — and featurise once

Training needs the GPU; evaluation does not. Measured on this laptop (not
Kaggle, so the absolute number will differ), full-window CPU inference runs
at **~1.7 µs per edge** — 0.51s for 300k edges — which puts the largest
evaluation window (1,137,240 edges) at roughly **2 seconds** and all 44
evaluation windows at L = 48 (41.0M edges) at **~70 seconds per model**.
GPU inference is 0.18s for that window, so the GPU buys nothing here worth
a session's quota.

**Evaluation runs window-major, not model-major.** Stage 3.3 re-featurised
every evaluation window for every model, which — at the measured 7s per
million edges — cost roughly **1.2 of its 1.59 hours**. Instead: for each
(lookback, detection time), featurise the window **once**, score it with
**all ten models** at that lookback, and keep only the scores. Scores are
one float per edge per model: 41.0M × 10 × 4 B = **1.6 GB** for L = 48,
against ~14 GB to cache the features themselves. The existing per-model tau
sweep, budgets and paired comparisons then run from cached scores exactly
as they do now.

Estimated evaluation cost: ~5 minutes of featurisation per lookback,
~20 minutes of GNN inference across all 20 models, and the clustering the
tau sweep already pays — call it **under an hour on CPU**, against 1.59
hours for stage 3.3's XGBoost.

##### 5. Everything else

| | Proposed | Why |
|---|---|---|
| **Architecture** | Linear 6→64 + LayerNorm + ReLU, then 3 × GINEConv at 64 each followed by LayerNorm + ReLU, then readout MLP `[h_src, h_dst, edge_features]` → 64 → 1. Dropout 0.0 | The input projection is an implementation choice worth stating: PyG's `GINEConv(edge_dim=…)` projects edge features down to the *node* width to add them to messages, so a first layer at the raw node width would squeeze 55 edge features into 6 dimensions and discard most of the edge information before it was used. Dropout is zero because the model is ~50k parameters and regularisation comes from early stopping. |
| **Optimizer** | Adam, lr 1e-3, **weight decay 0.0**, no schedule | Frozen across variants, lookbacks and seeds for the same reason stage 3.3's hyperparameters were: otherwise a comparison becomes a comparison of tuning effort. With the balanced loss starting at 0.69, 1e-3 is a conventional starting point. Weight decay is zero: at ~50k parameters with early stopping already regularising, a decay term would be one more knob nobody tuned. |
| **Epochs** | Max 100, patience 10, both fixed in advance | At ~4s an epoch the ceiling costs minutes, so it is set by patience, not by budget. Fixed before any full-data training, like everything else here. |
| **Tuning** | **One configuration, never tuned on validation or test** | The condition that makes the GNN-versus-XGBoost comparison a comparison of models. XGBoost got one fixed configuration; so does this. Any later tuning would have to be declared and would change what the comparison means. |
| **Early-stopping metric** | **Validation edge-level average precision (AUPRC)**, computed every epoch | Threshold-free, so it does not depend on a tau that has not been chosen yet, and it is the right summary when positives are rare — at a 0.1% positive rate ROC-AUC flatters almost anything. Campaign-F1 was considered and **rejected**: it needs cluster extraction at several thresholds every epoch, and over 26 validation campaigns it moves in steps of 1/26, so it would stop on noise. Validation *loss* is also rejected: under a 2,500× class weight it is dominated by a few hundred positives and can fall while campaign-level recall does not move. Tau is still chosen **once at the end** on validation campaign-F1 from the model grid, exactly as for XGBoost. |
| **Windows early stopping may see** | The **validation detection split** only, `(Sept 5, Sept 6]` — four times at either lookback | The leakage trap in this sub-stage. Edge-level average precision reads *every* label in a window, and the validation **evaluation horizon** that tau selection uses runs to Sept 8, so it contains detection times inside the test detection split. Selecting epochs on it would be selecting on test-period edges. Tau selection can use that horizon because only validation campaigns' outcomes enter it (rule 2); this cannot. Pinned in `tests/test_gnn.py`. |
| **Checkpoint kept** | The epoch with the best validation campaign-F1, not the last | Standard, and it makes early stopping a selection rather than just a stop. |
| **Window order** | Shuffled each epoch by `numpy.random.default_rng(seed * 1000 + epoch)` | Full-batch means one window per step, so order is the only stochasticity besides initialisation; seeding it per (seed, epoch) keeps a run reproducible and the shuffle independent across seeds. |
| **Seed control** | `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy` and Python seeds, all from the run seed; `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)` **attempted** and the result recorded | Honest limit: GPU scatter accumulates in nondeterministic order and floating-point addition is not associative, so if PyG's kernel has no deterministic variant the call raises and determinism is reported as unavailable rather than claimed. Two mitigations make this acceptable: the five seeds' spread is reported and will dwarf last-bit drift, and **evaluation is exactly reproducible regardless**, because it runs from saved weights. |
| **Resumability** | Each finished (variant, lookback, seed) saved to `/kaggle/working/models/gnn/<variant>_L<L>_s<seed>/` immediately; on restart, existing models are skipped. `evaluate` writes a partial report per configuration | An interrupted session loses at most the model in flight. Evaluation runs in a later CPU session from the saved models, so the two halves never have to fit in one session. |

##### Estimated cost of the full run

| | Estimate | Basis |
|---|---|---|
| Training, 20 models | **~2 h GPU**, one session | 3.6s/epoch (L = 24) and 4.1s (L = 48) measured × up to 100 epochs, + ~1 min per model of setup |
| Featurising training windows | ~2 min per lookback, cached in ~2 GB host RAM and reused across all epochs and seeds | 7s per million edges measured; without the cache, re-featurising each epoch would cost 70s against 4s of compute and would dominate training by ~20× |
| Evaluation, 20 models | **< 1 h CPU**, separate session | ~1.7 µs/edge inference + window-major featurisation, against stage 3.3's 1.59 h |

Both halves fit single sessions with room, and the GPU session is the only
one spending quota.

#### 3.4.2 implementation — done

`models/gnn.py`, `tests/test_gnn.py` (25 tests), and the
`train-gnn` / `evaluate-gnn` subcommands. `torch` and `torch_geometric` are
in `requirements.txt`, unpinned, because the Kaggle GPU image ships its own
CUDA-matched torch and the GPU subcommands install PyG themselves.

**The scorer has the same interface as the XGBoost scorer** —
`scorer(window_transactions, t) → one probability per transaction in row
order` — so `evaluation.evaluate_split`, `select_tau`, the budgets and the
paired comparisons are the *same code* for both models, with no
model-specific path. `score_graph` exists beside it so a caller evaluating
several models can build each window once; the five seeds of a variant
consume byte-identical graphs.

**Resumability.** Each finished model is saved immediately under
`models/gnn/<variant>_L<L>_s<seed>/` and skipped on a restart;
`is_trained` requires **all four** files, so a run interrupted mid-save is
retrained rather than skipped and then loaded broken. The feature spec and
node preprocessor are fitted once per lookback into `models/gnn/spec_L<L>/`
and reused. `evaluate-gnn` writes a partial report after every
configuration and can run in a later session from the saved models.

**Tests** cover the things that would quietly invalidate a result: that a
window graph contains no edge at or after `t` (a transaction stamped
exactly `t` is added to the data and the graph does not grow); that early
stopping never sees a test detection time; that no account, bank or
campaign name reaches the inputs and no parameter is account-vocabulary
sized; that scores line up positionally with the window's rows and number
`E` rather than the doubled `2E`; that the balanced loss is `ln 2` at
initialisation for a 100-edge and a 100,000-edge window alike and that its
per-sample weight ratio is exactly `n_neg / n_pos`, checked through the
gradients; that average precision matches hand-computed values; that the
no-format variant drops exactly the `fmt_*` columns including
`fmt___other__` and changes nothing else; and that a saved model reloads to
identical scores.

**Local CPU smoke run on the real 1% sample — development numbers only.**
The 1% background makes separation artificially easy; nothing here is a
result. L = 24, full variant, seed 0: the loss starts near `ln 2` and falls
to 0.06, validation average precision reaches **0.94**, early stopping fires
at epoch 31 having peaked at 21, and **`max |logit|` stays around 10** —
against the 10⁴–10⁵ the unnormalised probe implied, which is the `log1p`
and LayerNorm working. Evaluation then gives test recall 0.786 at tau 0.99
(interior), 0.957 of the oracle, and 0.0h paired against the ceiling.
Training took 94s and evaluation 25s on a laptop CPU.

Two measurements worth carrying into 3.4.3: featurisation costs ~40 µs per
edge and a forward pass ~9 µs on these small windows, so **featurising
dominates** — which is what window-major evaluation exists to amortise, and
why the graph is built once per (window, variant) rather than once per
model. On the full data's far larger windows both per-edge costs fall
sharply.

#### 3.4.3 quick check — done, provisional, one seed

One model on the full data: L = 24, the full variant, seed 0. GPU training
in one session, CPU evaluation in another, both exit 0 (Kaggle, commit
`8d8929c`). Evidence: `docs/reference/train_gnn.{md,csv,json}`,
`evaluate_gnn.{md,csv,json}` and the four companion CSVs.

**Training behaved as designed.**

| | Measured | Expected, and what it tells us |
|---|---|---|
| First epoch's loss | ~0.69 | `ln 2`, which the balanced-mean loss gives for any window whatever its class ratio. The loss scale is right. |
| `max \|logit\|` | 1.2 → 14 over training | Single and low double digits, against the 10⁴–10⁵ the unnormalised probe implied. `log1p` on the node counts plus LayerNorm hold on the full data, not only on the sample. |
| Peak GPU memory | **7.43 GB, flat across epochs** | One window at a time: the graphs are cached on the **host** (5.92 GB there) and moved to the device per step. The probe measured 6.39 GB for the largest L = 24 window forward+backward, and training adds the optimizer state and the fragmentation of cycling windows of different sizes, so 16% above it is the expected direction and size. Flat is the part that matters: no epoch-over-epoch climb, so nothing leaks. Half of one 15.6 GB card. |
| Epoch time | ~7s | 13 full-batch steps. |
| Early stopping | best epoch 45 (val AP 0.4189), 56 epochs run | Patience 10, fired as designed. |
| Whole model | **463.4s** | ~95s to featurise and cache the 17 window graphs on the host, then ~7s an epoch. This, not the epoch time, is what sizes the full run. |
| Feature-spec + node-preprocessor fit | 54.4s, host peak 10.6 GB | Once per lookback, shared by every variant and seed. The probe measured 14.1 GB for the L = 48 fit; both are far inside the notebook's 33.7 GB. |
| Determinism | `deterministic: true` | `use_deterministic_algorithms` took effect, so the run is reproducible from the seed. |

**Validation average precision is 0.419 on the full data against 0.94 on
the 1% sample.** That is not a regression — it is the positive rate. The
sample keeps every laundering edge and 1% of the background, so positives
are ~6% of its edges against 0.102% on the full data, and average precision
is bounded below by the positive rate. The sample's 0.94 was never
comparable to anything, which is why no sample number is a bar.

##### The result, and what it does and does not say

L = 24, seed 0, test split, against the stage 3.3 XGBoost at the same
lookback and seed. **Provisional: one seed, one lookback, one variant.**

| | GNN | XGBoost | |
|---|---:|---:|---|
| Chosen tau (validation campaign-F1) | 0.995 | 0.99 | both interior to the grid |
| Test recall (`all`) | 0.661 (37/56) | 0.661 | identical |
| Share of the oracle ceiling | 0.804 | 0.804 | identical |
| **Test false alarms per day** | **33.3** | **3.4** | **~10× apart** |
| Best validation campaign-F1 | 0.418 | 0.543 | the quantity tau selection maximises |
| Validation recall at ~4 FA/day | 0.231 (tau 0.999) | 0.423 (tau 0.99) | at a comparable alert volume |
| Paired earliness vs the oracle | 0.0h, CI includes zero | 0.0h, CI includes zero | on every split |

**"Equal recall" is not the finding.** The two models reach the same test
recall at alert volumes an order of magnitude apart, and matching a
detector's recall while raising ten times as many alerts is worse, not
equal. The comparable readings all favour XGBoost: the validation
campaign-F1 that tau selection actually maximises (0.418 against 0.543),
and validation recall at a matched ~4 alerts a day (0.231 against 0.423).
On this one configuration the GNN is **behind**, and the honest summary is
that it is behind by roughly a factor of two in recall at a comparable
alert volume.

Three limits on even that statement:

1. **One seed, one lookback, one variant.** Stage 3.3 found no-format
   recall varying 5× more across seeds than the full variant's, so a single
   seed is not a measurement of a model.
2. **The alert volumes were not matched by the evaluation.** Both numbers
   above come from tau chosen by validation campaign-F1, which optimises F1
   and not a budget. The budget comparison is the fair one and did not
   exist in `evaluate-gnn` until now — see below.
3. **There is still no direct GNN-versus-XGBoost paired comparison.** Both
   "0.0h" rows above are each model against *the oracle*, and rule 3
   forbids inferring an A-versus-B difference from `A vs C` and `B vs C`.
   The earliness question between these two models is, as of this run,
   **untested** — not answered, and not answered either way.

##### The GNN is not being tuned to close this gap

**Equal tuning effort stands.** The GNN has one hyperparameter
configuration, frozen before any full-data training, exactly as XGBoost
does. A result that reflects how much search each model received is not a
result about models, and the asymmetry would be invisible in the final
table. So nothing here is widened, deepened, re-weighted or
re-scheduled because the number came out low.

The only thing that justifies changing the model is a **demonstrated bug**:
a specific, named defect with evidence that it is wrong, not an
underwhelming metric. "It could probably do better with more layers" is not
a bug. The diagnostics added below exist to *find* such a defect if there is
one, and to characterise the gap honestly if there is not.

#### 3.4.3 follow-up: four evaluation gaps closed

The quick check exposed four gaps in what the evaluation reported, not in
what the models do. No model changed, nothing was tuned, and no frozen
definition moved.

**1. Alert budgets, which the frozen design requires.** `evaluate-gnn` now
evaluates every model at 10, 50 and 100 false alarms a day, with the tau for
each budget chosen on **validation** from the same curve the F1 tau comes
from, so no test information enters the operating point. This is the fair
head-to-head: two models' F1-chosen points need not alert at comparable
rates, and at 33 against 3 alerts a day they do not.

**2. A direct model-versus-model paired comparison — and the records it
needs.** Rule 3 requires A and B be compared to each other. That needs both
models' per-campaign detection times at a matched seed, lookback and
operating point, and **stage 3.3 saved none of them**: its files hold
summaries, tau curves, budget rows and its own paired comparisons against
the oracle, but not one row per campaign. The campaign-level records existed
only inside the running process.

So from now on **every evaluation writes them** — `evaluate_outcomes.csv`
and `evaluate_gnn_outcomes.csv`, one row per campaign per operating point —
and a new `compare-models` subcommand reads two such files and nothing else:
no model, no features, no transactions. It pairs on evaluation split,
lookback, seed, operating point and ablation arm, reports earliness over
the campaigns both detect with a bootstrap CI, reports coverage each way by
campaign id, and refuses to run if two detectors ever collapse into one key.
Because it reads files, the comparison can run in a cheap CPU session long
after the runs, and the same command serves every Phase 4+
federated-versus-centralized comparison.

Stage 3.3's records are **regenerated by re-running `evaluate`, not by
retraining**: CPU only, 1.59 hours measured, and the 20 boosters are
unchanged. If the stage 3.3 training output no longer exists, retraining
them is a further 1.44 measured CPU hours — still no GPU.

**3. Two diagnostics that select nothing.** Both answer "where does the
difference live", and neither chooses a tau, stops a model or appears in a
headline.

- **Edge-level average precision**, over the raw scores before any
  threshold or clustering, computed for both models on the **same four
  validation-detection-split windows** the GNN early-stops on, and on each
  evaluation horizon. If the campaign-level gap shows up here, the
  difference is in how the edges are ranked; if it does not, the difference
  is in how the scores survive thresholding and connected-component
  extraction. The GNN's early stopping and this diagnostic call **one**
  function, so the two numbers cannot drift apart.
- **The size distribution of false-alarm clusters**, in accounts, at each
  operating point. A hundred three-account components is not the same alert
  load as a hundred forty-account ones, so "the same false alarms per day"
  is not necessarily the same work.

**4. The quick-check numbers are recorded as provisional**, which is the
section above.

What this cost in code: `evaluation/edge_metrics.py` and
`evaluation/compare.py` (both model-agnostic), per-campaign records and
cluster-size rows in `evaluate.py`, the budget loop and the two diagnostics
in both `evaluate` and `evaluate-gnn`, and the `compare-models` subcommand.
`tests/test_compare_models.py` adds 22 tests. The three that matter are the
ones whose failure would be invisible: that a record survives a CSV round
trip with `detected` intact (the string `"False"` is truthy, so a careless
parse reports every campaign detected and a recall of 1.0 — checked through
a file read as text, because pandas usually infers a real boolean column and
would hide it); that a comparison run from two files is identical to the
same comparison in process, so rule 3 holds across sessions; and that two
variants tagged into one key are **refused** rather than silently mixed into
one paired set. Each was confirmed to fail when the guard is removed.

Verified end to end on the real 1% sample — development numbers only: train
and evaluate both models at L = 24 seed 0, then `compare-models`, which
pairs all four operating points on both splits, reports coverage each way
(the GNN finds 2 campaigns XGBoost misses at the 50/day budget, and
1-for-1 at the F1 point), and produces the edge-AP and cluster-shape tables.
On the sample the GNN's `val_detection` edge AP is 0.9406 — **the same
number training reported as its best epoch's validation AP**, which is the
check that the diagnostic computes the quantity early stopping used.

#### 3.4.4 the full-run plan — executed as written

*(The plan as approved, kept for the record; the measured outcome is in
§[the full run](#344-the-full-run--done) below. The three sessions ran as
described and all exited 0.)*

Two variants × two lookbacks × five seeds = **20 GNN models**, plus the
XGBoost baseline, which has to be **retrained**: stage 3.3 ran in an
interactive session, so its `/kaggle/working` was discarded and the 20
boosters no longer exist. Three sessions — two GPU, one CPU. The cells are
§5k of `backend/scripts/kaggle/run_on_kaggle.md`.

**GPU, training.** Measured on the 3.4.3 quick check: pipeline build 30.6s,
feature-spec and node-preprocessor fit 54.4s, then **463.4s for one model
of 56 epochs** — which decomposes into ~95s to featurise and cache the 17
window graphs on the host plus ~7s an epoch. Peak reserved GPU memory
7,432 MB, flat. L = 48 has 9 training windows against 13, each about twice
the edges, so ~1.4× the edges an epoch.

| | Per model | Ten models | At the 100-epoch cap |
|---|---:|---:|---:|
| L = 24 | ~8.1 min (measured) | **~1.4 h** | ~2.2 h |
| L = 48 | ~11.4 min (×1.4) | **~1.9 h** | ~3.1 h |

Both fit a 9-hour session with hours to spare. The feature spec is fitted
**once per lookback** and shared by all ten models, because that is the
expensive step — 14.1 GB of host RAM at L = 48, 10.6 GB measured at L = 24,
against the notebook's 33.7 GB. The window graphs are rebuilt per model,
which costs ~16 minutes across ten models and is the one thing here worth
optimising later.

**One GPU session per lookback**, not one for both. Both would fit the 9
hours on these numbers, but a finished model is skipped on restart, so
splitting means a lost session costs at most the model in flight rather than
the lookback — and it does not spend the whole estimate in one place against
a weekly quota. The only thing the single session saves is one pipeline
build.

**The tightest point of the stage is L = 48's GPU memory.** The probe
measured 10.70 GB reserved for the largest L = 48 window; training added
1.16× at L = 24 (7.43 against the probe's 6.39), so expect **~12.4 GB of a
15.6 GB card**, about 80%. It is expected to fit, it is extrapolated rather
than measured, and the failure mode is explicit: on out-of-memory
`train-gnn` reports the window, epoch and peak, writes a partial report and
exits 4, with no silent fall back to sampling or chunking.

**CPU, everything else — one 12-hour session, in commit mode.** Measured on
the quick check at L = 24 over 24 windows and 11.17M edges: 92.6s to
featurise them once (8.28 µs/edge) and 56.4s for one model's forward passes
(5.08 µs/edge), with tau selection over the 16-value grid at 37.3s a model.

| Step | L = 24 | L = 48 (≈2× the edges) | Basis |
|---|---:|---:|---|
| Featurise, once per variant | ~3 min | ~6 min | measured |
| Forward passes, 10 models | ~9 min | ~19 min | measured |
| Tau selection, 10 models | ~6 min | ~12 min | measured |
| Evaluate: 3 splits × 4 operating points × 10 models | ~15–60 min | ~30–120 min | measured at the F1 tau; the range is the risk below |
| **`evaluate-gnn`** | **~35–80 min** | **~70–160 min** | |
| `train` (XGBoost, 20 boosters) | — | — | **1.44 h measured** |
| `evaluate` (XGBoost, writes the records) | — | — | **1.59 h measured** |
| `compare-models` | — | — | under a minute; it reads two CSVs |

**~4.8–7.1 hours in total**, inside a 12-hour session, so 5–7 hours of
margin. It runs in **commit mode** ("Save & Run All") rather than
interactively, which is also what cost stage 3.3 its models.

**The retrain doubles as a regression check.** Same seeds, same code, same
data, so the re-evaluated numbers must reproduce the recorded stage 3.3
results — the full model at L = 48 with test recall **0.679** and **0.760**
of the ceiling, and no tau at a grid edge. If they do not, something changed
that should not have, and that is worth more than the comparison it was run
for.

**The one real risk, named.** Clustering cost scales with the edges kept, so
a *low* budget tau is the expensive case: at L = 24 on test, stage 3.3 spent
194s on a single model at its 10/day tau against 3.7s at its F1 tau, a 50×
spread. The GNN alerts more freely at its F1 tau, so its 10/day tau will be
*higher* and cheaper — but its 50 and 100/day taus will be lower and dearer,
and they are new work no measurement covers. The upper end of each range
above assumes every budget behaves like stage 3.3's worst case. Nothing
silently degrades if it is slow: `evaluate-gnn` writes a partial report
after every configuration, and because the session saves its own output, a
later CPU session can attach it and finish from the saved models — no GPU,
no retraining.

**Sequence.**

1. **GPU session A1:** `train-gnn --lookbacks 24` — ten models. Save the
   output.
2. **GPU session A2:** `train-gnn --lookbacks 48` — ten models. Save the
   output.
3. **CPU session B, commit mode**, with A1's and A2's outputs attached as
   inputs. In order: locate both `models/gnn` trees under `/kaggle/input`
   with `find` and copy them into one directory, asserting 20 models and
   both specs are present — a one-minute check that fails before hours are
   spent; then `train`; then `evaluate`; then `evaluate-gnn`; then
   `compare-models`. XGBoost goes first because its cost is known to within
   a percent, which leaves the whole remaining margin for the one uncertain
   step.
4. **Stage 3.4.5**, the Phase 3 gate report, is written from session B's
   output — by paired comparisons only, per lookback, per seed, at the
   F1-chosen taus and at every budget, with coverage reported separately. It
   needs no further session.

#### 3.4.4 the full run — done

**Run:** three Kaggle sessions as planned — two GPU (one per lookback, ten
models each) and one CPU session in commit mode that retrained the 20
XGBoost boosters, evaluated all 40 models and ran `compare-models`. All exit
0, no tau at a grid edge (the chosen taus span 0.95–0.999 against a grid top
of 0.9999). The `models/` trees were not kept; every number below comes from
the committed reports in `docs/reference/`.

**The stage 3.3 retrain is a clean regression pass.** This was the point of
retraining rather than the comparison it was run for, so it is reported
first. Test split, `all` subset, mean [min, max] over seeds 0–4:

| Variant | L | Recorded in stage 3.3 | Re-run 2026-11-14 | Seed sd |
|---|---:|---|---|---:|
| `xgboost_centralized` | 24 | 0.654 [0.643, 0.661], 0.796 of ceiling, 3.8 FA/day | **0.654 [0.643, 0.661], 0.796, 3.79** | 0.009 |
| `xgboost_centralized` | 48 | 0.679 [0.661, 0.696], 0.760 of ceiling, 10.5 FA/day | **0.679 [0.661, 0.696], 0.760, 10.53** | 0.016 |
| `xgboost_centralized_no_format` | 24 | 0.479 [0.393, 0.554], 0.583, 4.2 FA/day | **0.479 [0.393, 0.554], 0.583, 4.17** | 0.052 |
| `xgboost_centralized_no_format` | 48 | 0.532 [0.500, 0.571], 0.596, 8.6 FA/day | **0.532 [0.500, 0.571], 0.596, 8.55** | 0.026 |

Every figure reproduces to three decimal places, including the min and max
across seeds and the five-fold seed-noise difference between the variants.
The boosters are a deterministic function of the seeds, the code and the
data, and that is now demonstrated rather than assumed.

**The GNN, five seeds, at the F1 operating point.** Test split, `all`
subset:

| Model | L | Recall | vs ceiling | FA/day | Median lead | Seed sd |
|---|---:|---:|---:|---:|---:|---:|
| `xgboost_centralized` | 24 | 0.654 | 0.796 | **3.8** | 67.2h | 0.009 |
| `gnn_centralized` | 24 | **0.661** | 0.804 | **20.3** | 69.1h | 0.046 |
| `xgboost_centralized` | 48 | **0.679** | 0.760 | **10.5** | 68.2h | 0.016 |
| `gnn_centralized` | 48 | 0.636 | 0.712 | **34.2** | 61.7h | 0.065 |
| `gnn_centralized_no_format` | 24 | 0.386 | 0.470 | 20.6 | 61.3h | 0.045 |
| `gnn_centralized_no_format` | 48 | 0.418 | 0.468 | 74.4 | 56.8h | **0.181** |

The one-seed quick check's reading survives five seeds: at L = 24 the GNN's
recall is statistically indistinguishable from XGBoost's (0.661 against
0.654, well inside the seed spread) **at more than five times the alert
volume**, and at L = 48 it is behind on both. It is also markedly noisier —
seed sd 0.046–0.065 against the baseline's 0.009–0.016, and **0.181** for
no-format at L = 48, where one seed collapses to 0.143 recall.

**At a matched alert budget, the gap is unambiguous.** Recall at 10 false
alarms a day, tau chosen on validation, test split, mean over five seeds:

| Model | L = 24 | L = 48 |
|---|---:|---:|
| `xgboost_centralized` | **0.654** | 0.479 |
| `gnn_centralized` | **0.525** | 0.475 |
| `xgboost_centralized_no_format` | 0.511 | 0.443 |
| `gnn_centralized_no_format` | **0.032** | 0.082 |

**The no-format GNN effectively cannot operate at 10 alarms a day.** At
L = 24 it produces **zero clusters in all five seeds** at that budget — there
is no threshold on the frozen grid at which it alerts inside the budget and
still detects anything. At L = 48 two of five seeds are likewise zero and a
third yields one or two clusters. Its 0.032 mean recall is not a weak
detector; it is mostly a detector that has been switched off by its own
threshold. The XGBoost ablation at the same budget reaches 0.511.

**Earliness is not where the two differ.** The direct paired comparison
(rule 3: GNN against XGBoost, never each against the oracle) gives a median
difference of **0.0h with a CI containing zero in every seed** at the F1
point, and at the 10/day budget every CI also touches or contains zero. No
earliness difference is demonstrated in either direction. What differs is
**coverage**: at 10/day on test at L = 24, seed 0, XGBoost detects 13
campaigns the GNN misses while the GNN detects 1 that XGBoost misses.

#### 3.4.5 the gate report — why the GNN loses at extraction, not at ranking

The campaign-level number is the product of two stages — rank the edges,
then threshold and extract connected components — and a single recall figure
cannot say which one costs the GNN its performance. The two diagnostics added
after the quick check answer it, and they point the same way.

**The edge ranking is near-equal at L = 24.** Edge-level average precision
over the raw scores, before any threshold or clustering, full variant, mean
over five seeds:

| Window set | L | XGBoost | GNN | Difference |
|---|---:|---:|---:|---:|
| test | 24 | **0.394** | 0.392 | −0.002 |
| `val_detection` (the 4 windows the GNN early-stops on) | 24 | 0.425 | **0.435** | **+0.010** |
| test | 48 | **0.387** | 0.350 | −0.037 |
| `val_detection` | 48 | **0.501** | 0.439 | −0.062 |

At L = 24 the two models rank edges equally well, and on the four
validation-detection windows the GNN early-stops on it ranks **slightly
better**. Yet its campaign recall at 10 FA/day there is 0.525 against 0.654.
**The ranking is not the problem at L = 24; what happens after the threshold
is.** (At L = 48 the GNN is genuinely behind on ranking too, −0.037, which is
a separate and simpler deficit.)

**The mechanism is the shape of the false alarms.** From
`false_alarm_cluster_shape` in `compare_models.json`, at the F1 operating
point on test, L = 24, seed 0:

| | Clusters | Median accounts | Mean | Max | Size distribution |
|---|---:|---:|---:|---:|---|
| XGBoost | **16** | 3 | **3.00** | **3** | 16 × 3 accounts, nothing larger |
| GNN | **158** | 3 | **3.20** | **6** | 132 × 3, 22 × 4, 3 × 5, 1 × 6–10 |

XGBoost's false alarms are *all exactly three accounts* — the smallest
component the extraction rule will emit, i.e. the minimum accident. The
GNN's are ten times as numerous and systematically larger.

**Why:** message passing **correlates the scores of adjacent edges**. A
GNN's high scores therefore arrive in connected neighbourhoods and readily
form spurious connected components, while XGBoost scores each transaction
independently, so its high scores are scattered and rarely form a connected
triple by chance. Under connected-component extraction, the very property
that is supposed to help a graph model — neighbourhood information flowing
along edges — works against it. The model pays for each false neighbourhood
in alert volume, which forces its budget tau higher, which costs it the
campaigns it would otherwise have found.

**This is a finding, not a defect to fix.** The GNN is **not** tuned in
response and the extraction rule is **not** changed: both were frozen before
any full-data training, and altering either now would make the comparison a
record of which model received more attention. The result stands as measured.
See §[future phases](#future-phases) for the follow-up this suggests — an
extraction rule less sensitive to score correlation, or per-neighbourhood
score calibration — which is further work, not a correction to this result.

Evidence: `docs/reference/compare_models.{md,json}`,
`compare_models_edge_ap.csv`, `compare_models_clusters.csv`,
`compare_models_paired.csv`, `compare_models_coverage.csv`, and the
per-campaign records in `evaluate_outcomes.csv` / `evaluate_gnn_outcomes.csv`.

#### Development checks on the 1% sample

**These are development checks, not results.** They exist to show the code
runs end to end and produces coherent numbers; none of them is evidence
about the method. The sample keeps every campaign transaction and thins the
background to 1%, which leaves a **5.99% laundering rate against the full
data's 0.102%** — separation is artificially easy, false alarms are
artificially scarce, and any threshold chosen against them is meaningless
at full scale. **No number here is a bar for stage 3.4 and none should be
quoted as a finding.** The real numbers come from the Kaggle run.

L = 24, test split, seeds 0–1, for illustration of the shape only:

| Variant | Recall (`all`) | vs ceiling | Median lead | FA/day |
|---|---:|---:|---:|---:|
| `xgboost_centralized` | 0.777 [0.768, 0.786] | 0.946 | 74.8h | 5.3 |
| `xgboost_centralized_no_format` | 0.714 [0.714, 0.714] | 0.870 | 73.8h | 5.0 |
| oracle (the ceiling) | 0.821 | 1.000 | 74.7h | 1.3 |

What these checks confirm about the *implementation*, with no claim about
the method:

- The pipeline runs: tau is chosen on validation from the model grid (0.3
  and 0.7 at the two seeds — interior, so no stop), all four subsets are
  reported, and the budgets, the ablation and the paired comparisons all
  produce output.
- The paired machinery behaves as designed: against the ceiling and across
  the ablation, the lead-time difference is 0.0h with CI [0.0, 0.0], and
  neither model ever detects a campaign its comparator misses. On a
  6-hourly grid that is the expected shape — the first detection time at
  which a campaign is matchable is the same for any detector that scores
  those edges above tau, so what differs between detectors is *whether*
  they fire, not *when*. Whether that holds on the full data is an open
  question.
- The ablation bites: dropping the format columns costs recall on the
  sample and redistributes importance across structural features
  (`self_transfer`, node degrees, counterparty and currency counts) instead
  of one dominant column. Both effects need the full data to mean anything.

### Recall against the ceiling

**Frozen 2026-10-17.** The oracle reaches 0.821 (L = 24) / 0.893 (L = 48)
on test, so a model's recall measured against 1.0 is measured against a
denominator that includes campaigns the laundering edges alone cannot
yield. Every scorer and model is therefore reported two extra ways:

- **`oracle_reachable`**, a fourth test subset: the campaigns the oracle
  detects at that lookback and split. Reported alongside `all`,
  `fully_observed` and `no_shared_accounts`, never instead of them.
- **`recall_relative_to_ceiling`**, the recall as a fraction of the
  oracle's recall on the same subset. A model at 0.60 against an 0.80
  ceiling is at 0.75 of what was achievable, and saying 0.60 understates
  it. **It is never clamped and can exceed 1**, because the ceiling is not
  strict — see §[Oracle ceiling](#oracle-ceiling). On `oracle_reachable` it
  is bounded by 1, but only because that subset counts reachable campaigns
  alone; a detection outside the reachable set appears as `all` detections
  exceeding `oracle_reachable` detections.

Both are reported; neither replaces absolute recall, because the absolute
number is what says how much of the problem remains unsolved. The subset is
not a property of a campaign, so `evaluation.summarize` requires the
oracle's detected ids to compute it and **raises rather than silently
falling back to every campaign** — a quiet fallback would divide by the
wrong denominator and overstate a model.

**Phase 5 will report both.** The visibility sweep's headline numbers are
given on `all` *and* on `oracle_reachable`. The reason is specific to that
experiment: the structurally disconnected campaigns below are concentrated
in the fragmentable group, and their own laundering edges cannot form a
matchable cluster in any condition at any visibility level. Left in the denominator they add the same dead
weight to all four conditions, compressing the differences the experiment
exists to measure; removed, the dose-response curve is read over campaigns
that could actually respond. Both are reported because the `all` number is
the one an institution would experience.

### Why the oracle misses campaigns

The oracle has perfect information, so a miss is never a scoring failure.
`kaggle_runner.py oracle-diagnose` classifies every missed `eval_ok`
campaign into one of four reasons, and changes nothing about the matching
rule:

| Reason | Meaning | What it implies |
|---|---|---|
| `structurally_disconnected` | pool **every** one of its own edges before the deadline, with no window at all, and its largest connected set of accounts is still under 3 | **no lookback can ever detect it.** The evidence does not exist in the graph; this is a property of the campaign, and the hard floor under the ceiling |
| `slow` | pooled, its own edges do connect 3+ accounts, but never inside a single window before the deadline | reachable in principle: a longer L may find it, and the reported largest transaction gap says whether it could |
| `purity_failure` | a 3+ fragment did form, but the cluster it sat in was under half the campaign's own accounts | the campaign was merged into something bigger — the report names whether that was other campaigns, unassigned laundering, or both |
| `late_only` | a matching cluster existed, but only after the deadline | censoring, not sparsity. A campaign whose transactions are an hour apart can still land here if its deadline precedes the first window that can see it |
| `unexplained` | none of the above | would mean the diagnosis and the evaluation disagree, and is reported loudly rather than bucketed away |

Precedence is deliberate and frozen: **`purity_failure` → `late_only` →
`structurally_disconnected` → `slow`**, so the most specific explanation
wins. A censored campaign is never mislabelled as sparse, which would send a
reader hunting for a longer lookback when the problem is the deadline; and a
campaign whose evidence never connects at all is never mislabelled as slow,
which would suggest a longer lookback could rescue it when nothing can.

`structurally_disconnected` and `slow` were a single reason,
`temporal_sparsity`, until 2026-10-17. They were separated because they
imply opposite actions, and pooling them hid which of the two the ceiling
was actually made of.

**Measured on the full data, Kaggle, commit `eacb6d3`** (before the split
was implemented, so these are the pooled counts):

| L | `structurally_disconnected` + `slow` | `late_only` | `purity_failure` | Total misses |
|---|---:|---:|---:|---:|
| 24 | 36 | 2 | 0 | 38 |
| 48 | 30 | 1 | 1 | 32 |

Per split at L = 24: val 7, test 8, stress 21 pooled-sparsity misses; at
L = 48: val 6, test 5, stress 19. **Purity failures are negligible** — one
case in 70 misses across both lookbacks — which is why the matching rule
stays frozen; there is no evidence that the 50% condition is costing
anything.

**The split itself, measured on the real 1% sample** (test split; the
full-data split awaits the next Kaggle run):

| L | `structurally_disconnected` | `slow` | `late_only` | Total |
|---|---:|---:|---:|---:|
| 24 | 5 | 3 | 2 | 10 |
| 48 | 5 | 0 | 1 | 6 |

All 10 structurally disconnected cases across both lookbacks are
**BIPARTITE** and all are in the `fragmentable` group. Two things follow.
First, the count does not move with L — 5 at both lookbacks — exactly as the
definition requires, since no lookback can help. Second, the 4 campaigns
L = 48 rescues are precisely the 3 `slow` ones plus 1 `late_only`, which is
the mechanism behind its higher recall, now named rather than inferred.

The sample is trustworthy for this one comparison because `make_sample`
keeps **every** campaign transaction and thins only the background, and the
sample reproduces the full-data test outcomes exactly: 56 campaigns, 46
detected at L = 24 and 50 at L = 48, the same 4 rescued ids, and the same
pooled sparsity totals (5 + 3 = 8, 5 + 0 = 5). It is **not** trustworthy for
false-alarm rates or anything involving the background, which is 1% of it.

The command also characterises the oracle's **false alarms** (1–6 per day on
the full data). Since the oracle keeps only laundering edges, every cluster
it forms is real laundering; a false alarm is therefore laundering that the
matching rule could not attribute to a campaign of the split being scored
and that was not majority-unassigned either.

**Which of the rule's two conditions failed is now reported separately**,
with counts. The rule requires *purity* (≥ 50% of the **cluster's** accounts
belong to one campaign) **and** *coverage* (≥ 3 of that **campaign's**
accounts are in the cluster), and a single "best overlap" number cannot say
which bound was binding. The earlier summary line read "best overlap 0.667
(below the 0.5 the rule requires)", which contradicts itself: 0.667 is above
0.5. Those clusters had passed purity and failed coverage. Each condition is
now checked against its own threshold and the binding one named:

| `fails` | Meaning |
|---|---|
| `coverage` | pure enough, but holds under 3 of the campaign's accounts |
| `purity` | holds 3+ of the campaign's accounts, but is too diluted by others |
| `both` | neither condition met by any campaign it touches |
| `no_campaign_accounts` | touches no campaign account at all: unattributable laundering |

**On the real 1% sample** (test, L = 24; the background is thinned, so the
*rate* is not full-data, but the mix is informative): of 22 sampled
clusters, **21 fail coverage and 1 fails purity**, median best purity 0.667
against a 0.50 threshold and median best coverage 2 accounts against a
threshold of 3. The oracle's false alarms are therefore almost entirely
*fragments of real campaigns that are too small to claim*, not clusters
polluted by unrelated traffic — a different problem from the one the old
wording implied, and one the three-account floor creates by design.

### Alert-budget metrics (frozen 2026-10-15)

Recall alone can be bought by alerting more. On the full data the random
scorer reached **0.27 recall at L = 48 while raising about 47,000 false
alarms per day** — a detector no analyst could use, scoring a recall a
reader might mistake for signal. So alongside the primary metrics, every
scorer and model also reports:

- **campaign recall and median lead time at false-alarm budgets of 10, 50
  and 100 per day**, taking the *lowest* tau whose validation alarm rate
  stays inside the budget — within a fixed budget the most sensitive
  detector is the one worth having;
- the **full recall vs false-alarms-per-day curve**, saved for plotting.

A budget unreachable on the tau grid is reported as unreachable, with the
cheapest rate the grid achieves, rather than silently falling back to a
threshold that breaks it. Campaign-F1 on validation remains the **primary**
tau rule; the budget figures are reported alongside it, not instead of it.

### Sanity scorer results (stage 3.2, synthetic data)

The oracle and random scorers exist to validate the evaluation code before
any model is built. Run on a **seeded synthetic dataset** (120 campaigns,
300k transactions, 0.37% laundering) because the 1% research sample has not
been downloaded locally — so these numbers check the code, and are **not
results**. `scripts/run_sanity_scorers.py` reproduces them; the full-data
version runs on Kaggle in stage 3.3.

| Split | Scorer | Campaigns | Recall | Precision | False alarms/day | Other-split hits | Ambiguous | Median lead | Frac observed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | oracle | 14 | 1.00 | 1.00 | 0 | 160 | 1 | 32.9h | 0.40 |
| val | random | 14 | 0.00 | 0.00 | 971 | 0 | 9 | — | — |
| test | oracle | 18 | 1.00 | 1.00 | 0 | 300 | 3 | 25.0h | 0.33 |
| test | random | 18 | 0.00 | 0.00 | 947 | 0 | 18 | — | — |
| stress | oracle | 46 | 0.87 | 1.00 | 0 | 86 | 3 | 27.1h | 0.33 |
| stress | random | 46 | 0.00 | 0.00 | 988 | 0 | 13 | — | — |

This is the behaviour the design predicts. The oracle finds everything it
can, with no false alarms, at a median `frac_observed` of 0.33 — i.e. about
when the campaign's third account appears, which is exactly what the
matching rule requires. Random finds nothing and raises ~950 false alarms a
day. Stress oracle recall is 0.87 rather than 1.00 because campaigns
censored by the cutoff can complete after their deadline, and a detection
after the deadline counts as missed.

The large other-split-hit counts are rule 1 doing its job: a test window
contains campaigns from every split, and the oracle finds those too. Without
rule 1 they would have been charged as false alarms and test precision would
have read ~0.06 instead of 1.00.

> **A note on how these numbers were reached.** The first synthetic dataset
> gave random a recall of 0.57–0.71, which the design says means the
> matching rule is too generous. It was not: that generator gave campaign
> accounts no background traffic at all, so randomly keeping half the edges
> left a small *pure* fragment of each campaign, which the matching rule
> correctly called a hit. The fix was to the test data — campaign accounts
> now also transact in the background (770 of 770), and the laundering rate
> dropped from 1.8% to 0.37% — not to the frozen rule. Random recall then
> went to 0.00. The real check is still the full-data run.

### Evaluation rules (frozen 2026-10-10, implemented in stage 3.2)

**Rule 1 — other-split hits.** A detected cluster that hits a real campaign
belonging to a **different** split than the one being evaluated is an
*other-split hit*: neither a hit nor a false alarm, counted and reported
separately, and excluded from both precision terms. It is a correct
detection of something not currently being measured, so counting it either
way would be wrong. Applies to val, test and stress evaluation. A cluster
hitting a campaign that falls outside every split (one starting on/after the
cutoff) is also an other-split hit. The outcome vocabulary is therefore
`hit`, `other_split_hit`, `ambiguous`, `false_alarm`.

**Rule 2 — evaluation horizons.** Which detection times a split's campaigns
are scored over, each `(exclusive start, inclusive end]` and additionally
capped per campaign by its own `deadline`:

| Split | Horizon | Detection times |
|---|---|---|
| val | `(2022-09-05 00:00, 2022-09-08 00:00]` | 12 |
| test | `(2022-09-06 00:00, CUTOFF]` | 20 |
| stress | `(2022-09-08 00:00, CUTOFF]` | 12 |

Validation campaigns start on Sept 5 but need until Sept 8 to run their
course, so the validation horizon is deliberately longer than the Sept 5–6
*training-data* split and overlaps the test horizon (8 shared times, Sept 6
06:00 → Sept 8 00:00). Only **validation-campaign labels** inform tau
selection, so the overlap leaks nothing into test. The stress horizon
(frozen 2026-10-10) begins exactly where validation's ends, so those two
abut without overlapping, and sits strictly inside the test horizon — which
is precisely why rule 1 matters there: at a shared detection time a cluster
can hit a test campaign while stress is being scored, and that must count as
an other-split hit rather than a false alarm. Training has no evaluation
horizon; it is scored on validation, and asking for one raises.

**Rule 3 — full-history training windows.** A training detection time is
used only when `t - L >= DATA_START`, so no training window is silently
short (a "24h" feature computed over 6 hours would train the model on a
distribution it never sees again). Evaluation horizons are unaffected — they
all begin days after the data starts. The cost is training windows:

| Lookback | Usable training times | Dropped (of 16 on the grid) | First usable |
|---|---:|---:|---|
| 24h | 13 | 3 | 2022-09-02 00:00 |
| 48h | 9 | 7 | 2022-09-03 00:00 |
| 72h (excluded) | **5** | 11 | 2022-09-04 00:00 |

L = 72 leaving only five training detection times was one of the three
reasons it was dropped as a candidate (see Decision log).
`features-smoke` reports these counts on every run, for the excluded
lookback too.

**Scaling.** Mean/std standardisation of the continuous columns only
(one-hot, cyclical and binary features are left alone), fitted on training
windows **only** and saved as JSON so validation, test and every later
condition replay exactly the training transformation. Fitting is *refused*
if any non-training window is passed in. Arithmetic is float64; matrices are
stored float32, which matters because a busy 24h window holds ~1M
transactions.

## Phase 4 definitions (the federated conditions)

Additive to the frozen Phase 2 and Phase 3 definitions; nothing in either
changed. Phase 4 answers the research question itself: **how does the benefit
of collaboration change as each bank's visibility of a campaign shrinks, and
does that depend on topology?**

### Phase 4 frozen scope

Cut deliberately to fit a two-week finish, and **frozen on 2026-11-14**. Each
cut is a stated limitation, not an oversight:

| | Phase 4 uses | Not used, and why |
|---|---|---|
| **Model** | **XGBoost only** | No GNN in any federated condition. On the full data at L = 24, seed 0, the GNN matched XGBoost's test recall (0.661) while raising **33.3 false alarms a day against 3.4** — ten times the alert volume for the same recall, and at a matched ~4 alarms a day it reached 0.231 validation recall against XGBoost's 0.423. The federated comparison therefore uses the baseline model; the GNN stays as the Phase 3 **centralized** comparison only |
| **Variant** | **the full feature set only** | No `no_format` arm in federated conditions. The payment-format ablation stands as a **centralized** result (format buys coverage, not earliness); re-running it four times over would double the matrix for a question already answered |
| **Seeds** | **0, 1, 2** | Not 0-4. Three seeds give **wider confidence intervals** than Phase 3's five, and that is recorded as a limitation rather than hidden |
| **Lookback** | **one, L = 48** | Chosen by the frozen Phase 3 protocol, which selects on **validation**: recall 0.531 at L = 48 against 0.385 at L = 24. Carrying both lookbacks would double a matrix that is already 4 conditions × 4 bins × 3 seeds. **Stated honestly:** that validation margin shrank to ~1.4 of 56 campaigns on test and cost 3× the false alarms, so L = 48 is the protocol's answer rather than a comfortable one |
| **Institutions** | **K = 8, unchanged** | Reducing K would raise the structural visibility floor from ~`2/8` to ~`2/4` — about 25% to 50% — and flatten the very dose-response the experiment measures |

Everything else is frozen exactly as Phase 2 and Phase 3 left it: campaign
definition, splits, detection horizons, windows, features, cluster
extraction, the matching rule, the model tau grid, alert budgets, the four
test subsets, paired-comparison rules 1-3, and equal tuning effort.

**The table above is final; an amendment was proposed and declined.** Stage
4.1's instruction first asked for five seeds and both lookbacks, and it was
**withdrawn** on 2026-11-14 — the cost is in
§[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) (96 cells
against 320, and ~25-35 CPU hours against ~90-110 for the rest of Phase 4).
The standing rule it produced: if resolution is ever needed, **seeds widen
and lookbacks do not**, because L was settled by the Phase 3 selection rule
on validation and re-opening a decided rule inside a later phase is drift.
Any widening also binds 4.2 and 4.3, because two conditions are only pairable
over identical cells.

### The four conditions

| | Condition | Training | At detection time | What its difference measures |
|---|---|---|---|---|
| 1 | `isolated` | each institution on its own visible transactions only | each institution scores only its **own** visible edges; a cluster can only form inside one institution's view | the baseline every advantage is measured against |
| 2 | `fedavg` | institutions train locally, model state combined each round | each institution still scores only its **own** visible edges | the **knowledge** advantage: `fedavg − isolated` |
| 3 | `fedavg_exchange` | as `fedavg` | plus boundary-account information shared at detection time, so a cluster can **span** institutions | the **evidence** advantage: `exchange − fedavg` |
| 4 | `centralized` | the existing Phase 3 model, full visibility | scores everything | the upper bound |

XGBoost is not a neural network, so conditions 2 and 3 need concrete
definitions for a tree model — "averaging the weights" of a forest is not a
meaningful operation. **Stages 4.2 and 4.3 propose them and stop for
approval** — the design question moved there when 4.1 became the `isolated`
condition.
Whatever is approved, the **same** extraction, matching and evaluation code
runs for all four conditions, so a difference between them can only come
from the scores.

### Stage 4.0: institution-split data — done

The substrate every condition stands on. Given a `(seed, target)` assignment
from Phase 2, `federation/views.py` produces each institution's **view**: the
transactions it can see, and the account features it can compute from that
view alone.

**One definition of "sees", implemented once.** An institution sees a
transaction if the sender's **or** the receiver's account belongs to it —
the frozen Phase 2 rule, in `views.visible_mask`, and nothing else in Phase 4
decides visibility. A test requires it to agree row for row with Phase 2's
own `visibility.visible_transactions`, so Phase 4 cannot drift into a second
notion of visibility.

**A view is a `Window`, so no feature code is duplicated.**
`institution_view` returns the same `windows.Window` type the central
pipeline uses, which means `features.build_window_features(view, spec)`
works on it **unchanged** and computes that institution's node features from
its own view alone. There is no second featuriser to keep honest, and the
Phase 3 leakage properties carry over untouched. A view is a *row subset*:
which institution produced it is never written into the data, so a model
trained on a view cannot learn "I am institution 3".

**Both endpoints see a cross-institution transaction**, which is why the
achievable visibility floor is ~`2/K` and not `1/K`. A transaction is
therefore seen by exactly **two** institutions when its endpoints sit in
different ones and exactly **one** when they do not — pinned as a test, and
the reason the per-institution row counts below sum to ~188% of the central
count rather than 100%.

**The three properties the stage is defined by**, each checked over the whole
fixture rather than a hand-picked row, and each **proven able to fail** by
mutating the real source and watching the suite break:

| Property | Mutation injected | Result |
|---|---|---|
| A view holds exactly the rows that institution is an endpoint of | `visible_mask` reduced to sender-only | **4 tests fail** |
| The union of all views covers every transaction | `all_views` loops `range(k - 1)` | **2 tests fail** |
| A campaign's visibility computed from the views equals the recorded achieved visibility | `max` over institutions replaced by `mean` | **13 tests fail** |

The third is a genuine cross-check rather than a restatement: the pipeline
computes achieved visibility from each campaign's **pattern edge list**,
while the test counts rows of the **tagged transaction table** that land in
each institution's view. Two independent paths to the same number, required
to agree to within floating point for every campaign at every
`(seed, target)`. Measured: **0 mismatches across all 370 campaigns in all
26 runs** (3 seeds × 4 targets × 2 lookbacks, plus the natural split).

#### What each institution has to train on

**Measured on the real 1% sample, 2026-11-14 — a development check, not a
result.** The sample keeps every campaign transaction and thins the
background (and with it the unassigned laundering) to 1%, so its positive
rate is **3.55%** against the full data's **0.102%**. Absolute row counts and
class weights therefore do **not** transfer; the *relative* distribution
across institutions does, because it is a property of the assignment rather
than of the background density.

Pooled over the 9 rule-3 training windows at L = 48, per institution:

| Target | Rows per institution | Positives per institution (min / median / max) | Spread | Class weight | Starved institutions |
|---|---|---|---:|---|---:|
| natural (no reassignment) | 19,984-21,397 | 603 / — / 1,048 | 1.74× | 19-32 | 0 |
| 1.00 | 19,763-21,424 | **274** / 536 / 857 | **3.13×** | 24-71 | 0 |
| 0.75 | 20,041-21,514 | 538 / 761 / 1,095 | 2.04× | 18-37 | 0 |
| 0.50 | 20,119-21,533 | 589 / 793 / 1,039 | 1.76× | 19-33 | 0 |
| 0.25 | 20,120-21,494 | 599 / 762 / 959 | 1.60× | 20-33 | 0 |

**No institution is starved of positives at any target, on the sample.** The
thinnest cell is institution 2 at seed 1, target 1.0: **274 positives in
19,803 rows**. Every institution has at least one positive in every training
window, at every target and seed.

**The finding worth carrying into 4.1: target 1.0 is the *most* imbalanced
condition, not the easiest.** The positive spread across institutions is
**3.1× at target 1.0 against 1.6× at target 0.25**, and each institution's
share of the central positives falls to 0.082-0.256 from 0.179-0.287. The
mechanism follows from the frozen definition: at target 1.0 all of a
campaign's accounts are sent to **one** randomly chosen institution, so its
laundering edges become intra-institution and are seen by that institution
alone; at low targets the campaign is deliberately scattered, its edges
cross institutions, and **more** institutions see each positive. So the
isolated condition's weakest institution is weakest at the **100 bin** — the
end of the sweep where collaboration is supposed to matter least. 4.1 has to
say what the isolated condition does about that, and the effect is a
candidate explanation for any flattening at the top of the dose-response
curve.

**Full-data figures are not yet measured.** Scaling the sample's positives by
~1.6× (it retains all 3,209 campaign transactions but ~1% of the 1,968
unassigned laundering rows) and its rows by ~97× puts each institution's
pooled L = 48 training matrix at roughly 2M rows with several hundred to a
couple of thousand positives, i.e. a class weight in the same 495-3,461 band
stage 3.3 measured per window. That is an **extrapolation**; the real numbers
land with the first Kaggle run in stage 4.4.

Reproduce with:

```
cd backend
python scripts/run_institution_split.py --config configs/local.yaml
```

It writes `artifacts/federation_dev/institution_split.csv` and
`institution_split_checks.csv` (gitignored, development only) and takes about
two minutes on the sample.

### Stage 4.1: the isolated condition — approved and implemented 2026-11-14

**Approved on 2026-11-14 with six decisions and implemented the same day.**
The design below is what was built — §1-11 are the approved design, and
"What was built" at the end records the code, the tests and what the sample
run shows. The six decisions, folded into the subsections below: the frozen scope is **kept** (3 seeds,
L = 48) and the earlier five-seed/two-lookback instruction is **withdrawn**;
the thin-positive tiers are accepted **plus a second pooled mean over trained
institutions only**; the recall-0-versus-undefined split is accepted
unchanged; the union's matched-total budget is accepted, with **whose number
the union is** now stated explicitly; the structural ceiling is recorded as a
**result**; and a **null result for 4.2/4.3 is declared an expected possible
outcome** in advance.

**What 4.1 is.** Each institution alone: trained on its own visible
transactions, scoring its own visible transactions, forming clusters inside
its own view. No model averaging, no boundary-account sharing, no
coordination of any kind. Its purpose is to produce **the floor the later
conditions must beat.**

**Renumbering.** 4.1 was previously "define FedAvg and embedding exchange
for a tree model". Those definitions move to **4.2** (`fedavg`) and **4.3**
(`fedavg_exchange`, with the gate), and the Kaggle runner work stays at
**4.4**. The sub-stage table above is updated accordingly. The reservations
already recorded about calling probability-averaging "FedAvg" are unaffected
and are raised when 4.2 is designed.

#### 1. Scope: the frozen scope is kept, and the wider instruction is withdrawn

**Decided: 3 seeds (0, 1, 2) and L = 48 only.** The stage 4.1 instruction
first asked for five seeds and both lookbacks; it was withdrawn as written
"from Phase 3 habit", and the frozen Phase 4 scope stands unchanged. If
resolution is ever needed, **seeds** widen and **lookbacks do not**: L was
settled by the Phase 3 selection rule on validation, and re-opening a decided
rule inside a later phase is drift.

The argument that settled it is recorded because it applies again to any
later widening — the choice was never local to 4.1:

> **Whatever is chosen binds 4.2 and 4.3 as well.** Rule 1 pairs two
> conditions only over identical cells. A floor measured at five seeds and
> two lookbacks cannot be compared against a `fedavg` arm measured at three
> seeds and one lookback: the paired comparison would have no matching
> keys, and the cross-condition difference, which is the entire research
> result, would be unavailable.

The cost, from the stage 3.3 measurements (207-217s per booster at L = 24
and 282-289s at L = 48; `evaluate` 1.59h for 20 models across 3 splits and
4 operating points), scaled by the measured per-institution view size of
~23.5% of the central row count:

| | Cells (institution × target × seed × L) | Estimated CPU | 12h sessions |
|---|---:|---:|---:|
| **Chosen: frozen scope** — 3 seeds, L = 48 | 8 × 4 × 3 × 1 = **96** | **7-12 h** | 1-2 |
| Withdrawn — 5 seeds, both lookbacks | 8 × 4 × 5 × 2 = **320** | **20-34 h** | 3-4 |

Per cell that is one booster (~70-110s on a view of ~2.1M rows), one
evaluation across 3 splits and 4 operating points (~120-200s), and one
oracle pass for that institution's own ceiling (~40-90s). The same
multiplier then applies to `fedavg` and `fedavg_exchange`, so the real
decision is **~25-35 CPU hours for the rest of Phase 4 against ~90-110** —
and `fedavg_exchange` is the more expensive of the two, because its clusters
span institutions.

The reason to prefer the frozen scope was never that five seeds are worse —
they are better — but that the remaining three conditions, the gate, the
runner and the write-up have to fit the same two weeks, and a floor is only
useful if the conditions it floors actually get measured. The limitation this
leaves is the one already stated: three seeds give wider CIs than Phase 3's
five, and any Phase 4 difference smaller than the seed spread is reported as
undemonstrated rather than claimed.

#### 2. Model: the XGBoost baseline, frozen configuration, full feature set

Already a recorded Phase 4 research decision (Decision log, 2026-11-14) and
unchanged here: 300 trees, depth 6, eta 0.1, subsample and colsample 0.8,
`hist`, `binary:logistic` with `aucpr`, the per-window class weight
`n_negative / n_positive`, and the full 66-feature variant. The reason
specific to the federated conditions is recorded with it: federating the
weaker model would confound a federation result with a model result, so the
model is held fixed and only visibility and coordination vary.

#### 3. How an institution is restricted: the scorer, not the data

There are two ways to make `evaluate_split` see only one institution's
edges. The proposal takes the second.

**Rejected: pass the institution's view as `full_df`.** It restricts the
right thing for the wrong reason, and it silently moves the ground truth.
`evaluate_split` derives the unassigned-laundering account set — the set
that makes a cluster `ambiguous` rather than a `false_alarm` — from its
`full_df` argument. Hand it a view and laundering the institution cannot
see stops counting as laundering, so an alert landing on it is recorded as
a false alarm instead. The institution's false-alarm rate, and with it every
tau chosen against a budget, would then depend on which rows were filtered
out.

**Proposed: a masking scorer.** `full_df`, `camp`, `pattern_df`,
`accounts_by_campaign`, the detection grid, the horizons and the deadlines
all stay **global and identical to Phase 3**. Institution *i*'s scorer
receives the global window, computes `views.visible_mask` over its rows,
builds features from `institution_view(window, i, assignment)` alone, and
returns

```
scores = zeros(len(window))
scores[mask] = booster.predict(view_features)
```

Because a view is a row subset in the window's own order, the alignment is
positional and needs no join. Because the smallest tau on `MODEL_TAU_GRID`
is 0.05 > 0, an invisible edge can never be kept, so a cluster can only form
inside the view — which is the definition of the isolated condition. **No
line of the Phase 3 evaluation code changes**, so the extraction, the purity
≥ 0.5 and coverage ≥ 3 matching rule, the false-alarm-clusters-per-day
accounting and the four test subsets are the same code that produced the
centralized numbers, and a difference between conditions can only come from
the scores.

*Visibility restricts what can be scored. It never restricts what is true.*

#### 4. The ceiling, per institution per target — and a measured surprise

The centralized ceiling (0.821 at L = 24, 0.893 at L = 48) is a property of
the centralized view and is **not** reused for an institution's number.
Instead, for every `(institution, target, seed, L, split)` the oracle runs
**through the same masking scorer**: `oracle_scores` on the visible rows, 0
elsewhere. That institution's own oracle recall is its ceiling, and
`recall_relative_to_ceiling` is taken against it, **never clamped** — a
model detecting a campaign its own oracle misses is a real event at the
institution level for the same reason it is centrally.

Two consequences for the comparison rules:

- The `oracle_reachable` subset is defined by **that institution's own**
  oracle, so its campaign set differs per institution. **Two institutions
  are therefore never paired on `oracle_reachable`** — that would pair over
  different sets, which rule 1 forbids. All pairing happens on `all`, and
  `oracle_reachable` carries the ratio only.
- An institution whose own ceiling is 0 on a subset has an **undefined**
  ratio, not 0 and not infinity. `recall_relative_to_ceiling` already
  returns `None` when the oracle recall is 0; those cells are counted and
  listed, never averaged in as zeros.

**Measured now, on the 1% sample, as evidence for this design** — a
*structural* proxy for the per-institution ceiling: does institution *i*'s
view contain a connected component of ≥ 3 of a campaign's accounts, over
that campaign's own laundering edges? It ignores the 6-hour detection grid
and the deadline, so it is an **upper bound** on the real per-institution
ceiling. Over all 258 `eval_ok` campaigns, seeds 0-2:

| Target | Per-institution **mean** | Weakest | Strongest | Union of all 8 | Institutions reaching a campaign |
|---|---:|---:|---:|---:|---:|
| 1.00 | **0.255** | 0.182 | 0.353 | **0.884** | 2.04 |
| 0.75 | 0.445 | 0.353 | 0.519 | **0.884** | 3.56 |
| 0.50 | 0.502 | 0.411 | 0.574 | **0.884** | 4.01 |
| 0.25 | **0.546** | 0.461 | 0.609 | **0.884** | 4.37 |

**The per-institution column is a mean over the 8 institutions, not a max.**
Stated because it changes how the reversal is read: the mean is dragged down
by every institution that sees nothing, so a low value is a statement about
*concentration* as much as about reach. The strongest-institution column is
given beside it, and the reversal survives there too — 0.353 at target 1.0
against 0.609 at 0.25 — so it is not an artefact of averaging.

Two things in that table matter more than their size.

**(a) The per-institution mean runs backwards.** A single bank's structural
reach is *worst* at the 100 bin (0.255) and *best* at the 25 bin (0.546) —
the opposite of the intuition the sweep is built on. The mechanism is
concentration, and it is **measured** rather than assumed (seed 0, the
`reached by` line the script prints):

| At target | Campaigns reached by exactly 1 institution | by 2+ | by none | Campaigns with **all accounts** in one institution |
|---|---:|---:|---:|---:|
| 1.00 | **128-134** | 94-100 | 30 | **144-147** of 258 |
| 0.75 | 14-21 | 207-214 | 30 | **0** of 258 |
| 0.50 | 14-21 | 207-214 | 30 | **0** of 258 |
| 0.25 | 13-21 | 207-215 | 30 | **0** of 258 |

(Ranges are the three seeds; 30 unreached is identical in all twelve runs.)

So at target 1.0 roughly half the campaigns are concentrated into a single
holder (144-147 of 258), that holder is usually the only institution that can
see a connected triple (128-134 campaigns reached by exactly one), and the
other seven contribute zeros to the mean. At the lower
targets no campaign sits wholly inside one institution at all, every
cross-institution edge is seen by **both** endpoints, and ~4.4 institutions
each hold a connected fragment. Fragmenting a campaign spreads partial
evidence more widely than it hides it.

*One detail worth keeping straight:* 145 campaigns have all their accounts in
one institution at target 1.0 (seed 0), yet 256 of 258 reach visibility 1.0
exactly.
The two are not the same condition. Visibility needs **one institution to be
an endpoint of every campaign edge**, which for a hub topology means holding
the hub, not the whole campaign — so the search reaches full visibility for
113 campaigns whose accounts are still spread across institutions. The 30
campaigns reached by nobody are exactly the 30 that are not structurally
reachable centrally either (258 − 228), not a fragmentation loss.

**(b) The union's structural ceiling is 0.884 at every target, exactly.**
Not approximately — identically, and identical to the centralized structural
value: 228 of 258 campaigns, in all twelve `(target, seed)` runs. This is a
theorem, and the measurement confirms it: a 3-account connected component
contains a vertex with ≥ 2 distinct neighbours, and that vertex's
institution is an endpoint of both of its edges, so it sees that component
**whatever the assignment does with the other accounts**. The campaigns with
such a vertex number 228 of 258 — the same 228.

So the isolated-**union** condition is **not structurally handicapped at
all**. Whatever the federation buys in 4.2 and 4.3, it cannot be "the
cluster cannot be assembled from one view", because structurally it always
can. It has to come from one of three places, and 4.1 is what separates
them: the two edges of that degree-2 vertex must fall in the **same lookback
window** before the deadline; the institution must hold enough positives to
**learn** (stage 4.0: thinnest at the 100 bin); and the features an
institution computes from a partial view are **weaker inputs** than the
central ones. This sharpens what the stage is for, and it is a reason to
expect the union floor to be **high** — which makes 4.2's job harder and the
result more interesting either way.

*Caveat, stated plainly:* the 1% sample thins the background and not the
campaign edges, and these numbers use only campaign edges and the
assignment, so they should transfer — but they are a proxy measured on the
sample, they ignore time entirely, and the real per-institution ceilings are
whatever 4.1 measures on the full data.

Reproduce with:

```
cd backend
python scripts/run_structural_ceiling.py --config configs/local.yaml
```

Under 10 seconds on the sample; it prints the centralized value and the
degree-2 count beside the union so the identity can be seen rather than
taken on trust.

#### 5. The two pooled numbers, and the alert budget they are read at

Both are reported, kept separate, and never averaged into one figure.

**(a) The per-institution mean** — "what does a typical bank achieve alone".
The mean over the 8 institutions of that institution's recall, each against
**its own** ceiling. A bank operating alone has its own analysts, so each
institution is operated at the **full** budget B (10, 50, 100 false-alarm
clusters a day), exactly as the centralized model is.

**It is reported twice: over all 8 institutions, and over trained
institutions only.** An `untrained` institution entered at recall 0 is
honest, but a single mean conflates **"cannot see the campaign"** with **"had
no labels to learn from"**, and those are different claims about federation:
the first says fragmentation hides evidence, the second says it starves
learning, and 4.2 addresses them by different mechanisms. Both means carry
their denominators, and when no institution is `untrained` the two are
identical and are still both printed, so a reader never has to infer which
one a number is.

**(b) The union** — "8 banks each reporting separately, nobody
coordinating". A campaign counts as discovered if **any** institution
discovers it alone, and the union's detection time is the **earliest** across
institutions, so earliness is reported for it too. Its ceiling is the union
of the per-institution oracle detections, which by (4b) is expected to sit at
or near the centralized ceiling.

**Whose number this is, stated explicitly: a reader who receives all eight
alert streams — a regulator or FIU — and not a bank.** No individual bank can
compute it, sees the other seven streams, or benefits from the deduplication,
and the total alert volume it is budgeted at is the one that lands on that
single recipient's desk. Deduplication only means anything for an entity that
sees everything. The union is therefore the honest floor for 4.2 at the
**supervisory** level: it is what fragmented evidence already achieves with
no coordination beyond mandatory reporting.

**The budget problem, which is the substantive objection to reading the
union naively.** Eight institutions each at 10 false alarms a day put up to
80 a day in front of whoever receives the reports. Against a centralized
model at 10 a day, that union is not a floor — it has bought its coverage
with eight times the alert volume, and 4.2 would then be measured against a
floor that outspends it. The proposal:

- **The union's primary operating point matches total volume.** Each
  institution's tau is chosen on validation so the **summed** false-alarm
  rate meets B, instantiated as the equal split **B / K** per institution.
  Equal rather than proportional to volume, and **equal specifically because
  it is untuned**: allocating the budget across institutions by anything
  data-dependent would be a tuning step, and the equal-effort constraint that
  makes conditions comparable does not cover tuning the *allocation*. An
  optimised allocation would also hand the floor an advantage the federated
  conditions were never given.
- **Duplicate alerts are deduplicated.** Two institutions raising the same
  account set at the same detection time is one case on a desk, so the union
  counts a cluster identity (the sorted account tuple at a detection time)
  once, for hits and false alarms alike.
- **A second union row is reported, clearly labelled**, at per-institution
  budget B, with its **measured total** false-alarm rate printed beside it.
  That is the honest "eight banks with eight sets of analysts" reading, and
  printing its volume is what stops it being quoted as if it were matched.

**The per-institution mean is a statistic, not a detector.** It has no
per-campaign outcome records, so it **cannot** be the subject of a paired
comparison against the centralized model or against `fedavg`. The union
**is** a detector — it has a per-campaign detected flag and an earliest
detection time — so it is what rule-1 pairing uses. Recorded here so that a
later session cannot mistake the mean for something pairable.

#### 6. Tau, per institution, on validation

Tau is chosen per `(institution, target, seed, L)` by the frozen rule:
validation only, campaign-level F1, `MODEL_TAU_GRID` with its near-1 tail,
ties broken by `best_tau`. Eight institutions therefore hold eight different
thresholds, which is correct — each bank calibrates its own detector on what
it can see — and **equal tuning effort is preserved**, because every
institution gets the identical protocol, not identical attention.

A tau at the edge of the grid is recorded per cell (`tau_at_grid_edge`) and
**does not abort the run**: all cells finish, every report is written, and
the runner returns **exit 3** at the end if any cell selected at an edge.
That is the pattern `evaluate` and `evaluate-gnn` already use — reports
complete, then stop loudly — and with 96 cells, aborting on the first edge
tau would throw away hours of finished work.

#### 7. Class weights

No new rule: the frozen per-window weight `n_negative / n_positive`,
computed **within the institution's own view window**, positives carrying
their window's weight and negatives 1.0. Stage 4.0 measured the consequence
on the sample — the weight band widens to 24-71 at target 1.0 against 18-37
at the other targets — and that is reported per cell, not corrected.

#### 8. First decision requested: institutions with too few positives

**The premise needs one correction first.** The instruction asks about "too
few positive edges to train on at the 25% target". Stage 4.0 measured the
opposite ordering: on the 1% sample the positives-per-institution spread is
**3.13× at target 1.0 and 1.60× at target 0.25**, the thinnest cell of all is
at target 1.0 (institution 2, seed 1: 274 positives), and **no institution is
starved at any target**. Scarcity of *training* positives is a **100-bin**
risk, for the reason in (4a). What is genuinely thin at the 25 bin is not
training data but **window-level reachability** — the institution has
positives to learn from, but the three accounts it needs may not be connected
inside any one window — and that is measured as its ceiling, not patched as a
training problem.

**Proposed rule — three tiers, nothing silently dropped:**

| Positives in the pooled training matrix | What happens | How it is reported |
|---|---|---|
| **0** | Cannot train: XGBoost needs both classes | cell recorded `untrained`, recall **0**, contributes nothing to the union, listed by `(institution, target, seed)` |
| **1-9** | **Trains**, flagged `thin` | a normal cell, carrying the flag and its positive count |
| **≥ 10** | Trains | a normal cell |

`MIN_POSITIVES_TO_TRAIN = 10` therefore stays what stage 4.0 made it — a
**reporting threshold, not a gate**. Refusing to train at 9 positives and
training at 10 would delete a real outcome on an arbitrary boundary, and what
matters about a thin institution is what it achieves, which is measurable.

An `untrained` cell is recall **0 and not undefined**, because "this bank has
nothing to learn from and therefore catches nothing" is the honest answer to
"what does a bank achieve alone" — an outcome of the condition, not missing
data. The count of `untrained` cells is printed beside every pooled number
that includes them, and per (5a) the per-institution mean is reported **both**
over all 8 institutions and over trained institutions only, so that
"cannot see it" and "had nothing to learn from" are never read as one
number.

**And no institution is ever rescued by another's data.** Pooling to fix a
thin institution is federation, which is exactly what 4.2 measures; doing it
inside the floor would destroy the comparison.

#### 9. Second decision requested: zero qualifying clusters at a budget

**Recall 0 and undefined are both needed, for two different events** — a
detector that found nothing, and a detector that cannot be operated there at
all:

| Event | Recorded as | Why |
|---|---|---|
| A tau exists at the budget; clusters form; none matches a campaign | **recall 0** | A real outcome. Phase 3 precedent: the no-format GNN produces **zero qualifying clusters at 10 FA/day in all five seeds at L = 24**, and that is reported as a finding, not as missing data |
| A tau exists at the budget; **no cluster forms at all** | **recall 0** | Same reason: the detector was operable and detected nothing |
| **No tau on the grid meets the budget** — even the highest exceeds B false alarms a day | **undefined (empty)** | The operating point does not exist. `select_tau_for_budget` already returns `None` for exactly this case, deliberately, rather than silently falling back. Recording 0 would conflate "detects nothing" with "cannot be run at this volume", and would drag the pooled mean down with a cell that was never measured |
| The institution's **own ceiling is 0** on a subset | ratio **undefined**, recall still reported | Existing behaviour of `recall_relative_to_ceiling` |

**The reporting rule that makes this safe:** every pooled number is printed
with its denominator — how many cells it averages, out of 8 institutions ×
the seed count, so 24 under the frozen scope — and every excluded cell is
listed by `(institution, target, seed, budget)` with its reason. A mean over
21 cells is never allowed to look like a mean over 24. This matters most at the
union's matched-total budget, where B / K is **1.25 false alarms a day per
institution** at the 10/day budget — the likeliest place on the whole matrix
for the grid to hold no affordable tau.

#### 10. Outputs

| File | Contents |
|---|---|
| `phase4_isolated_outcomes.csv` | Per-campaign records, the **same columns** as `evaluate_outcomes.csv` plus `condition`, `institution`, `visibility_target` and `visibility_bin`, so `compare-models` can pair 4.2 and 4.3 against it. The **union** is written as its own detector rows (`institution = "union"`), since it has genuine per-campaign outcomes |
| `phase4_isolated.json` / `.md` | The evidence report: per-institution summaries, per-institution ceilings, the two pooled numbers with their denominators, the `untrained` / `thin` / no-affordable-tau lists, tau curves, timings |
| `phase4_isolated_ceilings.csv` | The oracle ceiling per `(institution, target, seed, L, split, subset)` — the thing that must never be confused with the centralized ceiling |
| `phase4_isolated_summary.csv` | One row per `(target, seed, L, split, operating_point, institution)`, plus the two pooled rows |

All four go to `docs/reference/`. One integration point to settle in 4.2
rather than now: `compare-models` refuses to run when two detectors collapse
into one pairing key, so it needs `institution` and `visibility_target` in
its key before it can read these files — a small additive change to a command
that already exists, and no change to what it computes.

New runner subcommands, `train-isolated` and `evaluate-isolated`, resumable
per `(target, seed, institution)` so a lost session costs one cell.

#### 11. What 4.1 will not do

No model averaging, no boundary-account sharing, no cross-institution
clusters — those are 4.2 and 4.3. No tuning of anything in response to a weak
institution, and no change to the extraction rule, the matching rule or the
budget accounting. If the floor comes out higher than expected, that is the
result.

#### 12. Grid-edge policy, and a proposed extension of `MODEL_TAU_GRID`

Decided **before** the full-data run, on 2026-11-14, so that no cell's
treatment depends on having seen which cells hit an edge.

**The two edges mean opposite things, and only one of them invalidates a
cell.** `isolated.tau_grid_edge` names the direction, and the asymmetry is
the point:

| The chosen tau sits at | What it means | What happens |
|---|---|---|
| the **top** of the grid | the optimum lies **beyond** the grid, so this operating point was **not found** | the cell is **undefined**: excluded from both pooled means, listed by `(institution, operating point)` with its **rejected tau** and the reason, and **never scored 0**. The run finishes and exits 3 |
| the **bottom**, for the **F1** point | the most sensitive threshold on the grid is still the best one, so the optimum may lie **below** it | on the full data this **should not happen** at 0.102% prevalence. The runner **stops and does not record that cell** (`--allow-low-edge-tau` overrides, for the sample only). On the sample it is expected, and `run_isolated.py` records it and exits 3 |
| the **bottom**, for a **budget** point | the most sensitive threshold on the grid already fits the budget | **nothing.** This is a loose budget, not a missing optimum, and it is neither flagged nor excluded |

That last row is a refinement of the rule as first stated, and it is not a
small one: on the sample **166 of 224** `(institution, operating point)` cells
chose tau = 0.05, of which **163 are budget points** and only **3 are F1
points**. Treating every low edge alike would have excluded or stopped on
three quarters of the matrix for the uninteresting reason that `budget_100`
is easy to satisfy.

A high-edge cell is excluded for exactly the reason an unaffordable budget is,
and is reported the same way: *a detector that could not be operated at a
point was not measured at that point, and a mean that silently absorbs it as
0 is reporting a measurement that was never made.*

**Proposed: six more points at the top of the grid — awaiting approval.**
`MODEL_TAU_GRID` is a frozen definition, so this is a proposal and the code is
unchanged until it is approved. The extension continues the existing tail's
construction exactly — each step halves the remaining distance to 1, in the
alternating 5/1 pattern the current tail already uses:

```
current (16):  0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
               0.99 0.995 0.999 0.9995 0.9999
proposed (22): ... 0.9999 | 0.99995 0.99999 0.999995 0.999999 0.9999995 0.9999999
1 - tau:       ... 1e-4   | 5e-5    1e-5    5e-6     1e-6     5e-7      1e-7
```

Three measured reasons, none of them a response to a result:

1. **The current headroom is one grid point.** On the full data the highest
   tau ever chosen was **0.9995** (the GNN at a budget) and **0.999** for
   XGBoost; the F1-chosen taus were **0.99-0.995**. The top of the grid,
   0.9999, was never reached — by one step. Stage 4.1 trains each institution
   on ~23.5% of the rows at a similar prevalence, and 96 institution-cells ×
   7 operating points is 672 chances to need the step that is not there.
2. **It is nearly free, measured.** On the full data a validation sweep step
   costs 9.1s at tau 0.05 and **2.6s (max 3.4s)** at every point from 0.99
   upward, because almost nothing survives those thresholds and clustering
   cost scales with the edges kept. Six more points is **~16s per
   institution-cell**, about **26 minutes** across the whole 96-cell run.
3. **It stops where float32 does.** XGBoost returns float32 probabilities, and
   the largest float32 below 1.0 is **0.99999994**, i.e. `1 - 6.0e-8`. So
   `1 - 1e-7` is the last threshold still comfortably distinguishable from
   1.0, while `1 - 1e-8` **is** 1.0 in float32. Beyond that a finer grid
   cannot separate anything.

**And 1.0 itself is deliberately not in the grid.** Because the rule is
`score >= tau`, a tau of 1.0 would keep whichever scores happen to round to
exactly 1.0 in float32 — which depends on the arithmetic, the library build
and the hardware, and so is not reproducible. Stopping at `1 - 1e-7` keeps
every threshold meaningful *and* reproducible, and keeps "the chosen tau is at
the top of the grid" meaning what the policy above says it means: the optimum
is beyond what can be resolved, so the operating point was not found.

#### What was built

| | |
|---|---|
| `federation/isolated.py` | the condition: the masking scorer, per-institution training, the per-institution oracle ceiling, the union (outcomes, and the deduplicating cluster pass), the two pooled means, the seven operating points, and `run_cell`, which both runners call so the local and Kaggle reports come from one code path |
| `constants.PHASE4_SEEDS`, `constants.PHASE4_LOOKBACK_H` | the frozen scope **in code** — `[0, 1, 2]` and `48` — so a runner cannot drift from this README. `evaluate-isolated` refuses more than one lookback outright |
| `tests/test_federation_isolated.py` | **40 tests**, including the grid-edge policy; backend suite **402 passed, 25 skipped** |
| `scripts/run_isolated.py` | the condition on the 1% sample, for development |
| `train-isolated` | one booster per institution per `(target, seed)`, saved as it finishes, **resumable per institution** |
| `evaluate-isolated` | tau, ceilings, the per-institution and union reports, **resumable per `(target, seed)`**; it **never trains**, so a lost evaluation session cannot silently retrain on different boosters; exit 3 on an edge tau after every report is written |

Eight files come out of a run: `phase4_isolated_outcomes.csv` (per campaign,
the `evaluate_outcomes.csv` columns plus `condition`, `institution`,
`visibility_target`, `operating_point` and `tier`, with the union written as
its own detector rows), then `_summary`, `_pooled`, `_union`, `_ceilings`,
`_taus`, `_institutions`, and `phase4_isolated.json` / `.md`.

**Seven operating points, not four.** `f1` and `budget_10/50/100` give each
institution its own full budget — a bank alone has its own analysts — and
`total_10/50/100` give each institution `B / K` so the union's summed volume
is the one being budgeted. The sample run shows why both are needed: at
target 0.25 the union's deduplicated volume is **39.8** false alarms a day at
`budget_10` and **18.3** at `total_10`. Eight institutions each inside 10 a
day are not together inside 10 a day.

*A budget is a **validation-chosen** operating point, as in Phase 3, so the
realized test volume is not the budget: at `total_10` the union realizes
12.4-18.3 a day across the four targets. That is reported, not corrected.*

**Verified across processes.** Training and evaluation are separate Kaggle
sessions, so the two paths have to agree. For target 1.0 seed 0 they do
exactly: the eight taus chosen in one process (0.9995, 0.7, 0.7, 0.3, 0.995,
0.95, 0.995, 0.2) are the same eight chosen from the saved boosters by
`evaluate-isolated`.

**The guards were shown able to fail.** Three bugs are reproduced in the suite
against the same fixtures the tests use: a masked scorer that fills positions
instead of rows (the classic misalignment), a union that sums per-institution
false alarms instead of deduplicating cluster identities, and an untrained
institution recorded as undefined instead of 0 — which silently lifts the
pooled mean from 0.40 to 0.80 in the fixture.

**One cross-check worth naming.** The union's alert volume comes from a second
loop over the detection grid, because `EvaluationResult` keeps cluster *sizes*
and not account sets, so there is nothing in it to deduplicate by. That loop
calls the same frozen `extract_clusters` and `classify_cluster`, and a test
requires its **per-institution** counts to equal `evaluate_split`'s
`n_false_alarms` and `n_hits_clusters` for every institution at two taus. If
they ever disagree, the union's volume is being measured by a different rule
from every per-institution number beside it, and the suite fails.

#### Measured on the 1% sample — DEVELOPMENT ONLY, not for quotation

> **Every figure in this subsection is a development check on the 1% sample
> and none of them may be quoted in the report, the deck or the paper.** They
> exist to show the code runs and behaves as designed, not to say what the
> system achieves. The tau behaviour diagnosed in Reading 4 is exactly what
> will not transfer: the sample's positive rate is **3.3%** against the full
> data's **0.102%**, which moves the class weights by an order of magnitude
> and with them every chosen threshold, and therefore every recall, precision
> and alert rate downstream of one. The quotable numbers come from
> `train-isolated` / `evaluate-isolated` on the full data, and do not exist
> yet.
>
> One claim here is **not** a sample measurement: that the isolated union's
> oracle ceiling equals the centralized one rests on an argument about the
> frozen visibility rule, not on this run. Its *figures* below are still
> sample figures, and its confirmation on the full data is still pending.

The sample thins the background to 1%, so **precision, false alarms per day
and every budget-chosen tau do not transfer**. The **ceilings** do, because
they are a property of the campaign edges and the assignment, which the
sample preserves exactly. Seed 0, L = 48, `--splits val test`.

**Result: the isolated union's oracle ceiling equals the centralized oracle
ceiling exactly — with the 6-hour detection grid, the lookback and the
deadlines in force.**

| Split | Centralized oracle | Union of the 8 institutions' own oracles |
|---|---:|---:|
| val | **0.7692** (20/26) | **0.7692** at every one of the four targets |
| test | **0.8929** (50/56) | **0.8929** at every one of the four targets |

The 4.1 proposal predicted this structurally, ignoring time. It holds *with*
time, and the mechanism says why it must: if the centralized oracle detects a
campaign at time `t`, some window-local component held three of its accounts;
that component contains an account with two campaign counterparties; and
**that account's institution is an endpoint of both of those edges**, so it
sees the same component, in the same window, at the same `t`. Not merely the
same campaign — the same detection time, hence the same lead time and the
same side of the deadline.

*(The test figure 0.8929 is the full data's recorded ceiling of 0.893 at
L = 48 to three decimals. Not a coincidence either: the sample keeps every
campaign transaction and thins only the background, and the oracle scores
laundering edges alone.)*

**Reading 1: an individual institution's ceiling is far lower, and it moves
the opposite way.** Mean over the 8 institutions of each institution's own
oracle recall, subset `all`:

| Target | val | test |
|---|---:|---:|
| 1.00 | **0.139** (0.038-0.192) | **0.194** (0.143-0.232) |
| 0.75 | 0.303 (0.192-0.385) | 0.350 (0.268-0.411) |
| 0.50 | 0.327 (0.269-0.462) | 0.413 (0.357-0.589) |
| 0.25 | **0.346** (0.231-0.500) | **0.442** (0.411-0.518) |

The same reversal as the structural proxy, now with time in force: a single
bank's reach is **worst** where the campaign is concentrated and **best**
where it is scattered, while the union's is flat. The dose-response the sweep
exists to measure therefore runs in **opposite directions** for the two
pooled numbers, which is exactly why they are reported separately and never
averaged into one figure.

**Reading 2 — the one that reframes the stage: each institution already
achieves most of its *own* ceiling.** Test split, pooled per-institution:

| Target | recall | relative to its **own** ceiling | union recall | union, relative to ceiling |
|---|---:|---:|---:|---:|
| 1.00 | 0.125-0.170 | **0.62-0.87** | 0.679-0.804 | 0.76-0.90 |
| 0.75 | 0.250-0.313 | **0.71-0.89** | 0.786-0.857 | 0.88-0.96 |
| 0.50 | 0.286-0.368 | **0.70-0.89** | 0.804-0.839 | 0.90-0.94 |
| 0.25 | 0.328-0.404 | **0.74-0.91** | 0.839-0.857 | 0.94-0.96 |

(Ranges are across the seven operating points.) An institution's problem is
**not** that it is a weak detector on what it can see — it reaches 0.6-0.9 of
its own ceiling. It is that it cannot see. That is a prediction about 4.2 and
4.3 worth writing down before either runs: sharing **model knowledge**
(`fedavg`) addresses the part that is already near its ceiling, while sharing
**evidence** (`fedavg_exchange`) addresses the part that is actually missing.
If the two conditions separate anywhere, that is where.

**Reading 3: both of the approved reporting rules fire on real data.** At
target 0.75, institution 6 has **no affordable tau** at `total_10` — no
threshold on the grid meets 1.25 false alarms a day — so that cell is
**undefined**, excluded from the pooled mean, and listed: the mean is over
**7** institutions, not 8, and says so. And deduplication is not cosmetic:
the union sheds **80-138** duplicate clusters per cell at targets 0.75 and
below, against **3-8** at target 1.0 — exactly as the mechanism predicts,
since a fragmented campaign is seen by more institutions at once and a
concentrated one by about two.

**Reading 4: the sample run exits 3, at the LOW edge of the tau grid.**
Three of the 32 institution-cells chose **tau = 0.05**, the most sensitive
threshold on the grid — target 0.50 institution 2, and target 0.25
institutions 0 and 3. That is the **opposite** direction from stage 3.3's
problem, which was taus pinned near 1 and prompted the near-1 tail, and the
mechanism is the sample itself: a 3.3% positive rate against the full data's
0.102% puts the per-window class weights around 30 instead of 495-3,461, so
scores are not crushed against 1 and campaign F1 is maximised by keeping
almost everything. On the sample this is expected and is one more reason its
taus cannot transfer. On the full data **either** edge has to be explained
before the numbers are used, and both runners now print **which** edge each
cell hit rather than only that it hit one.

*(No institution is `untrained` anywhere on the sample, so the two
per-institution means are identical here. Both are still printed, so a reader
never has to work out which one a number is. A fixture in the test suite
builds the case where they differ — and in it the only institution that can
see the campaign is one that had no positives to train on, which is the
conflation the second mean exists to prevent.)*

#### Runtime on the full data

Revised now that the code exists and a performance defect was found in it:
the union's cluster pass was being handed fresh scorers, so it re-featurised
every window for every institution at every operating point, and a cell that
should take minutes hung. Each institution now owns one warm `CachingScorer`
that the per-institution evaluation and the union pass **share**; the cell
that hung now finishes in **151s** on the sample.

Per **institution-cell** on the full data: one booster (~70-110s on a view of
~2.1M rows), one evaluation across 3 splits and 7 operating points with the
cache warm (~150-280s), and one oracle pass for its own ceiling (~40-90s).
Across the frozen scope's **96 institution-cells** (8 institutions × 4 targets
× 3 seeds, L = 48) that is **7-14 CPU hours** — one or two 12-hour sessions,
resumable per institution in `train-isolated` and per `(target, seed)` in
`evaluate-isolated`, so a lost session costs one institution or one cell
rather than the run.

## Decision log

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 2026-11-14 | **Grid-edge policy, fixed before the full-data run. A chosen tau at the TOP of the model grid makes that operating point UNDEFINED** — excluded from both pooled means, listed with its rejected tau, **never scored 0** — because the optimum lies beyond the grid and the point was therefore not found | Identical in kind to an unaffordable budget, and reported the same way: a detector that could not be operated at a point was not measured at that point, and a mean that absorbs it as 0 reports a measurement that was never made. Decided in advance so that no cell's treatment can depend on having seen which cells hit the edge | user decision, 2026-11-14; §[stage 4.1 §12](#12-grid-edge-policy-and-a-proposed-extension-of-model_tau_grid); `tests/test_federation_isolated.py` |
| 2026-11-14 | **A chosen F1 tau at the BOTTOM of the grid stops the full-data run and is NOT recorded** (`--allow-low-edge-tau` overrides, for the thinned sample only). **A bottom-edge *budget* tau is neither flagged nor excluded** | At the full data's 0.102% prevalence the most sensitive threshold on the grid should not be the best one; if it is, something is wrong with the data, the weighting or the grid, and recording the cell would bury that. The budget exception is measured, not assumed: on the sample **166 of 224** (institution, operating point) cells chose tau = 0.05, of which **163 are budget points** — a low budget tau says only that the budget is loose, not that an optimum is missing, so treating every low edge alike would have excluded three quarters of the matrix for no reason | user decision, 2026-11-14, with the budget refinement measured; §[stage 4.1 §12](#12-grid-edge-policy-and-a-proposed-extension-of-model_tau_grid) |
| 2026-11-14 | **Proposed, awaiting approval: `MODEL_TAU_GRID` gains six points at the top** — 0.99995, 0.99999, 0.999995, 0.999999, 0.9999995, 0.9999999 — **before** the full-data run, not in response to results. The grid is a frozen definition, so the code is unchanged until this is approved | Three measured reasons. The current headroom is **one grid point**: on the full data the highest tau ever chosen was 0.9995 (GNN at a budget) and 0.999 for XGBoost, against a top of 0.9999, and 4.1 has 672 (institution, operating point) chances to need the next step. It is nearly free: a sweep step costs 9.1s at tau 0.05 but **2.6s** from 0.99 up, so six points is ~16s per institution-cell, ~26 min across the run. And it stops where float32 does: the largest float32 below 1.0 is 0.99999994, so `1 - 1e-7` is the last distinguishable threshold while `1 - 1e-8` **is** 1.0. **1.0 itself is excluded on purpose** — `score >= 1.0` would keep whatever rounds to exactly 1.0, which depends on the arithmetic and the build and is not reproducible | §[stage 4.1 §12](#12-grid-edge-policy-and-a-proposed-extension-of-model_tau_grid); `docs/reference/evaluate_baseline_tau_curve.csv`, `evaluate_baseline_budgets.csv` |
| 2026-11-14 | **MEASURED RESULT, recorded as a result: the isolated union's oracle ceiling equals the centralized oracle ceiling exactly** — not structurally but **with the 6-hour detection grid, the lookback and the deadlines in force**. Sample, L = 48: val 0.7692 against 0.7692, test 0.8929 against 0.8929, at targets 1.0 and 0.25 alike | The proof is the same degree-2 argument, carried through time: if the centralized oracle detects a campaign at `t`, some window-local component held three of its accounts; that component contains an account with two campaign counterparties; that account's institution is an endpoint of **both** of those edges, so it sees the same component in the same window at the same `t` — the same detection time, hence the same lead time and the same side of the deadline. Consequence, declared in advance: a null result for 4.2 or 4.3 is an expected possible outcome, because there is no structural headroom for coordination to recover | `scripts/run_isolated.py`, `scripts/run_structural_ceiling.py`; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) |
| 2026-11-14 | **Stage 4.1 evaluates SEVEN operating points, not four**: `f1` and `budget_10/50/100` at each institution's own full budget, plus `total_10/50/100` at `B / K` each so the union's summed volume meets `B` | Measured on the sample at target 1.0: the union's deduplicated volume at `budget_10` is **23.3 false alarms a day**, because eight institutions each inside 10/day are not together inside 10/day, while at `total_10` it is **4.0**. Comparing a 23.3/day union against a centralized model at 10/day would measure a floor that outspends what it floors. Both readings are kept because they answer different questions — eight banks with their own analysts, and one supervisory desk — and the per-institution-budget union prints its realized total beside it so it cannot be quoted as matched | `phase4_isolated_union.csv`; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §5 |
| 2026-11-14 | **The union's alert volume is counted by a second pass over the detection grid, and that pass is cross-checked against `evaluate_split` per institution** | `EvaluationResult` keeps cluster **sizes** and not account sets — deliberately, since it never stores account names — so there is nothing in it to deduplicate by. The second pass calls the same frozen `extract_clusters` and `classify_cluster`, and a test requires its per-institution false-alarm and hit counts to equal `evaluate_split`'s for every institution at two taus. Without that check the union's volume could be measured by a different rule from every per-institution number in the same table | `tests/test_federation_isolated.py` |
| 2026-11-14 | **Stage 4.1 approved, 2026-11-14.** The `isolated` condition is implemented as designed: an institution is restricted by its **scorer**, not by its data, so the Phase 3 evaluation code runs unchanged, and each institution gets **its own oracle ceiling** per target | The centralized ceiling (0.821 / 0.893) is a property of the centralized view; dividing an institution's recall by it would measure against a bound that institution could never reach. Restricting by the scorer rather than by `full_df` keeps the ground truth global - the unassigned-laundering set that makes a cluster `ambiguous` is derived from that argument, so filtering rows would silently reclassify ambiguous clusters as false alarms and move every budget-chosen tau | user approval, 2026-11-14; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) |
| 2026-11-14 | **The earlier "five seeds, both lookbacks" instruction for Phase 4 is WITHDRAWN. The frozen scope stands: 3 seeds, L = 48.** Standing rule for any future widening: **seeds widen, lookbacks do not** | The instruction was written from Phase 3 habit. L = 48 was settled by the Phase 3 selection rule on validation, and re-opening a decided rule inside a later phase is drift. Cost measured: 96 cells against 320, ~7-12 CPU hours against ~20-34 for 4.1 alone and ~25-35 against ~90-110 for the rest of Phase 4, inside a two-week finish. Any widening binds 4.2 and 4.3 too, since two conditions are only pairable over identical cells | user decision, 2026-11-14; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §1 |
| 2026-11-14 | **The per-institution mean is reported TWICE: over all 8 institutions, and over trained institutions only.** An institution with 0 positives is `untrained` and enters the all-8 mean at recall 0; 1-9 positives trains and is flagged `thin`; `MIN_POSITIVES_TO_TRAIN = 10` stays a reporting label, never a gate | A single mean conflates **"cannot see the campaign"** with **"had no labels to learn from"**, and those are different claims about federation - the first says fragmentation hides evidence, the second says it starves learning, and 4.2 addresses them by different mechanisms. Recall 0 rather than undefined for an `untrained` cell because "this bank has nothing to learn from and therefore catches nothing" is a real outcome of the condition, not missing data. Refusing to train at 9 positives and training at 10 would delete a real outcome on an arbitrary boundary | user decision, 2026-11-14; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §8 |
| 2026-11-14 | **Zero qualifying clusters at an existing tau is recall 0; no affordable tau on the grid is UNDEFINED**, excluded from the pooled mean and listed by cell with its reason | Two different events. A detector that was operable and found nothing is a measurement - the Phase 3 precedent is the no-format GNN producing zero clusters at 10 FA/day in all five seeds at L = 24, reported as a finding. A budget no tau can meet is not an operating point at all; `select_tau_for_budget` already returns `None` there deliberately, and recording 0 would drag the pooled mean down with a cell that was never measured. Every pooled number therefore prints its denominator | user decision, 2026-11-14; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §9 |
| 2026-11-14 | **The union's primary operating point matches TOTAL alert volume** - B / K per institution, duplicate cluster identities deduplicated, realized volume printed beside it, and the per-institution-B union reported separately. **The union is a regulator's or FIU's number, not a bank's** | Eight institutions each at 10 false alarms a day spend 80 a day; compared against a centralized model at 10 that is not a floor but a detector with eight times the budget. Equal split specifically **because it is untuned**: allocating the budget by anything data-dependent would be a tuning step the equal-effort constraint does not cover, and would hand the floor an advantage the federated conditions never got. The union is attributed to a supervisory reader because no individual bank can compute it, see the other seven streams, or benefit from the deduplication - dedup only means anything for an entity that sees everything | user decision, 2026-11-14; §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §5 |
| 2026-11-14 | **A null result for `fedavg` or `fedavg_exchange` is declared IN ADVANCE to be an expected possible outcome, reported as a finding and not treated as a failure to tune** | Measured and recorded as a result: the isolated **union** already reaches the centralized structural ceiling - 0.884 at every visibility target, 228 of 258 `eval_ok` campaigns, identical to the centralized value - because a three-account component always contains an account with two campaign counterparties, and that account's institution is an endpoint of both of its edges whatever the assignment does with the rest. There is therefore no structural headroom for coordination to recover, and any benefit must come from timing, learning or feature quality. Declared before the runs for the same reason equal tuning effort was: a result that only counts when positive is not a measurement | user decision, 2026-11-14; `scripts/run_structural_ceiling.py`, §[stage 4.1](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) §4 |
| 2026-11-14 | **Stage 4.1 is the `isolated` condition**, not the FedAvg design proposal; the tree-model definitions of `fedavg` and `fedavg_exchange` move to **4.2** and **4.3**, and the Kaggle runner stays at 4.4 | 4.1 produces the floor every later condition is measured against, so it comes first: a federation that does not beat uncoordinated banks each reporting separately has bought nothing. The `fedavg` definition is not needed to measure the floor, and deciding it later keeps the reservations about calling probability-averaging "FedAvg" open until the condition that depends on them is designed | user decision, 2026-11-14; §[the 4.1 proposal](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14) |
| 2026-11-14 | **A per-institution ceiling is computed per institution per target, and the centralized ceiling is never reused for an institution's number**. The oracle runs through the same masking scorer as the model | An institution that sees a quarter of a campaign's edges has its own, much lower ceiling, because the three-account component may not exist in its slice at all; dividing its recall by 0.893 would understate it against a bound it was never able to reach. Measured consequence of getting this wrong: the per-institution **structural** reach is 0.255 at target 1.0 against 0.546 at target 0.25, so the ceiling moves in the opposite direction to the intuition the sweep is built on | user instruction, 2026-11-14; `scripts/run_structural_ceiling.py` |
| 2026-11-14 | **Phase 3 is COMPLETE. The gate question is answered: a centralized model can discover campaigns early, and the graph model does not beat the non-graph baseline.** XGBoost reaches 0.679 test recall at L = 48 (0.760 of the ceiling); at a matched 10 FA/day budget the GNN reaches **0.525 against XGBoost's 0.654** | Five seeds, two lookbacks, two variants, on the full data, with the stage 3.3 retrain reproducing every recorded number to three decimals — so the baseline is a confirmed regression check, not a single run. The GNN's F1-point recall at L = 24 (0.661) is inside seed noise of XGBoost's (0.654) but costs **5× the alerts** (20.3 against 3.8 a day), and it is noisier across seeds (sd 0.046–0.065 against 0.009–0.016) | `docs/reference/evaluate_baseline.*`, `evaluate_gnn.*`, `compare_models.*`; tag `v0.3-phase3` |
| 2026-11-14 | **The GNN's shortfall is located at CLUSTER EXTRACTION, not at edge ranking** — and this is recorded as a **finding, not a defect to fix**. Neither the GNN nor the extraction rule is changed in response | Edge-level AP on test at L = 24 is **0.394 (XGBoost) against 0.392 (GNN)**, and on the four validation-detection windows the GNN early-stops on the **GNN ranks better**, 0.435 against 0.425 — yet its campaign recall at 10 FA/day is 0.525 against 0.654. The mechanism is in the false-alarm cluster shape: at the F1 point (test, L = 24, seed 0) XGBoost produces **16** false-alarm clusters, **all exactly 3 accounts**, while the GNN produces **158**, mean **3.20** accounts and up to **6**. Message passing correlates the scores of adjacent edges, so the GNN's high scores arrive in connected neighbourhoods and form spurious connected components; XGBoost scores transactions independently, so its high scores are scattered and rarely form a connected triple. The property that should help a graph model works against it under connected-component extraction. Tuning the model or relaxing the extraction rule now would make the comparison a record of how much attention each model received — equal tuning effort was the condition that made it a comparison of models at all | `compare_models_edge_ap.csv`, `false_alarm_cluster_shape` in `compare_models.json` |
| 2026-11-14 | **At L = 48 the GNN is behind on edge ranking too** — 0.350 against 0.387 — which is a separate deficit from the extraction effect and is reported as such | Reported rather than merged into the extraction finding, because the two have different causes and different implications: at L = 24 the ranking is equal and only extraction costs the GNN, while at L = 48 it loses at both stages. Conflating them would overstate how general the extraction mechanism is | `compare_models_edge_ap.csv` |
| 2026-11-14 | **Recorded: the no-format GNN cannot operate at the 10 FA/day budget.** At L = 24 it yields **zero clusters in all five seeds**; at L = 48, two of five are zero and a third yields one or two | There is no threshold on the frozen grid at which it alerts inside the budget and still detects anything, so its 0.032 mean recall is not a weak detector but one switched off by its own threshold. Reported rather than worked around: widening the grid for one arm would break the "one grid for every model and condition" rule that keeps a comparison from being a comparison of search effort. The XGBoost ablation reaches 0.511 at the same budget | `evaluate_gnn_budgets.csv`, `compare_models_clusters.csv` |
| 2026-11-14 | **Phase 4 uses XGBoost only. No GNN in any federated condition**; the GNN remains the Phase 3 centralized comparison | On the full data at L = 24, seed 0, the GNN matched XGBoost's test recall (**0.661**, 0.804 of the ceiling) while raising **33.3 false alarms a day against 3.4** — ten times the alert volume for the same recall — and at a matched ~4 alarms a day reached **0.231** validation recall against XGBoost's **0.423**. Equal recall at ten times the alert load is worse, not equal, so the federated comparison is built on the baseline model. This also keeps Phase 4 inside a two-week finish: four conditions × four bins × three seeds is already 48 cells, and a GNN arm would need a GPU session per cell | user decision, 2026-11-14; §[3.4.3 quick check](#343-quick-check--done-provisional-one-seed) |
| 2026-11-14 | **Phase 4 runs the full feature set only** — no `no_format` arm | The payment-format ablation is settled as a **centralized** result: format buys coverage, not earliness, with the direct paired earliness CI including zero in all ten test cells. Re-running it across four federated conditions would double the matrix for a question already answered. The limitation this leaves is explicit: Phase 4's absolute numbers inherit whatever the format shortcut contributes, so they are read as *differences between conditions*, which the shortcut affects equally | user decision, 2026-11-14 |
| 2026-11-14 | **Phase 4 uses seeds 0, 1, 2 — three, not five** | Timeline. Recorded as a limitation: three seeds give wider confidence intervals than Phase 3's five, and any Phase 4 difference smaller than the seed spread is reported as undemonstrated rather than claimed | user decision, 2026-11-14 |
| 2026-11-14 | **Phase 4 uses one lookback, L = 48**, chosen by the frozen Phase 3 protocol on validation (recall 0.531 against 0.385) | Carrying both lookbacks would double a 48-cell matrix. The protocol selects on validation and that is what was followed, but the margin is thin and is stated as such: it shrank to ~1.4 of 56 campaigns on test and cost ~3× the false alarms. Using the protocol's answer rather than re-litigating it keeps the selection rule consistent with Phase 3 | user decision, 2026-11-14; §[stage 3.3 results](#stage-33-results-full-data-five-seeds--complete) |
| 2026-11-14 | **K = 8 institutions is unchanged in Phase 4** | Every transaction is visible to both endpoint banks' institutions, so the structural visibility floor is ~`2/K`. Halving K to 4 would raise that floor from ~25% to ~50% and flatten the dose-response curve the experiment exists to measure | user decision, 2026-11-14 |
| 2026-11-14 | **Stage 4.0: an institution's view is a `windows.Window`, and the Phase 3 featuriser is reused on it unchanged** rather than given a federated variant | A second featuriser would be a second place for the leakage properties to be wrong, and the whole point of the comparison is that conditions differ only in what each party can see. Returning the same `Window` type means `features.build_window_features` computes an institution's node features from its own view by exactly the code that computes them centrally, and a view stays a pure row subset, so no institution identity can reach the model inputs | implementation, stage 4.0; `tests/test_federation_views.py` |
| 2026-11-14 | **Stage 4.0 reporting threshold: an institution with fewer than 10 positive rows in its pooled training matrix is reported as "starved"** (`federation.MIN_POSITIVES_TO_TRAIN`) | Not a frozen research definition — a reporting threshold, so that an institution that cannot learn the positive class at all is a finding before a training session discovers it. What the `isolated` condition should *do* about a starved institution is a 4.1 decision, deliberately not pre-empted here. On the 1% sample no institution is starved at any target; the thinnest cell has 274 positives | implementation, stage 4.0 |
| 2026-10-05 | Exclude data on/after Sept 11 (`CUTOFF`) from model input | Only 1,108 rows remain after the cutoff and 59.1% are laundering (vs. ~0.1% overall) — a model could learn "late = laundering" instead of real signal | notebook cell `280d9179` |
| 2026-10-05 | Test split moved from Sept 8–10 to Sept 6–7 | With the old split, 74 of 76 eval_ok test campaigns were truncated by the cutoff before completing | notebook cells `280d9179`, `6b9b3892` |
| 2026-10-05 | Size filter: `eval_ok` requires `n_txn >= 3` and `n_accounts >= 3` | About a quarter of campaigns have only 1–2 transactions, where lead time is not a meaningful concept | notebook cell `280d9179` |
| 2026-10-05 | `K` (institutions) changed from 3 to 8; the originally-considered 10% visibility target was dropped | Every transaction is visible to both of its endpoint banks, so the achievable visibility floor is ~`2/K`; `K=3` can't reach low visibility, and median campaigns have only 8 accounts to spread around | notebook cells `77e5003e`, `ed484d7c` |
| 2026-10-05 | Hub types (`FAN-IN`, `FAN-OUT`, `GATHER-SCATTER`) used as a control group, always at visibility 1.0 | The hub account's own bank sees every transaction in the campaign by construction | notebook cell `77e5003e` |
| 2026-10-05 | Analysis uses achieved-visibility bins, not raw targets | Local search only reaches a median of 0.33 at target 0.25 (not 0.25), and small campaigns bottom out around 0.5 | notebook cells `ed484d7c`, `52107d17` |
| 2026-10-05 | Results reported with and without campaigns sharing accounts across splits | 100 accounts appear in more than one campaign (83 of them across different splits); 7 of 37 fragmentable test campaigns are affected | notebook cells `52107d17`, `cb0caa40` |
| 2026-10-05 | Unassigned laundering rows used for training labels, excluded from campaign evaluation | 1,968 of 5,177 laundering rows belong to no campaign in the pattern file | derived from notebook cells `cd70c51d`, Frozen definitions |
| 2026-10-05 | `HI-Medium` kept as an optional later replication dataset, not used now | Keeps scope to one verified dataset for the main experiment while leaving a path to test generalization later | project spec |
| 2026-10-06 | Campaigns relabelled `"unfragmentable"` are **still reassigned** in the five experiment runs; reassignment is decided by topology (`eval_ok` + non-hub `base_type`), recorded as the `reassignable` column | They receive the same visibility treatment as every other fragmentable campaign and simply land in the `"100"` bin at every target — which is itself the finding. Excluding them would leave them at whatever the natural bank split happens to give, which is not necessarily >= 0.9, so they would not sit consistently in the control group | user decision, 2026-10-06 |
| 2026-10-06 | Bank volume ties broken by ascending bank id; banks with zero pre-cutoff volume still get an institution (sorted last) | The frozen definition only says "sort by volume descending", leaving ties unspecified — pinning the tie-break makes the mapping fully reproducible. Zero-volume banks must still be assigned because a campaign crossing the cutoff can reference a bank with no pre-cutoff activity, which would otherwise raise a lookup error during visibility recomputation | implementation, Stage 2 |
| 2026-10-06 | Achieved visibility is exported for **every** campaign (all 370), not only the reassignable ones | The frozen definition says to recompute "every campaign's" achieved visibility, and the frontend must be able to show visibility for any campaign the user opens, including hub and non-`eval_ok` ones. Rows carry `campaign_reassigned` so analysis can still restrict to the treated set | README frozen definitions, Stage 2 |
| 2026-10-07 | Phase 3 gate: verify every Phase 2 regression number on the full data **before** any model work | The 24 full-data tests have never run — the dataset only exists on Kaggle. Every Phase 3 result would silently inherit any discrepancy in the campaign table, splits or group assignment, and a model built on wrong ground truth is worse than no model | user instruction, stage 3.0 |
| 2026-10-07 | `pandas>=2.2` pinned in requirements.txt | `campaigns.build_campaign_table` passes `include_groups=` to `groupby.apply`, which pandas added in 2.2 and older versions reject with a `TypeError`. Unpinned, a Kaggle image with pandas 2.1 would fail deep in the pipeline with a confusing error | implementation, stage 3.0 |
| 2026-10-08 | Every collection is sorted before any seeded random step. **Bug:** a campaign's accounts were held in a Python `set`, and per-process string-hash randomisation made set iteration order differ between runs, so the seeded search followed a different trajectory each process — same seed, different assignment. **Evidence:** two Kaggle runs of the same pipeline disagreed on the unfragmentable count (3 vs 2); reproduced locally by varying `PYTHONHASHSEED`. **Fix:** accounts are now sorted tuples, and `sorted()` is applied at every point where order reaches the RNG. The algorithm, targets, iteration count and seeds are unchanged — only enumeration order is pinned, so no frozen definition moved. **Why now:** Phase 3 is centralized-only and never touches these assignments, but Phase 4-5 *is* the visibility sweep and the `unfragmentable` label defines the control group; fixing it after results existed would have invalidated them | `tests/test_determinism.py`, stage 3.0 follow-up |
| 2026-10-15 | **Alert-budget metrics added to the evaluation**: campaign recall and median lead time at false-alarm budgets of 10, 50 and 100 per day, taking the lowest tau whose validation alarm rate stays inside the budget, plus the full recall-vs-false-alarms curve saved for plotting. Campaign-F1 on validation stays the primary tau rule | On the full data the random scorer reached 0.27 recall at L = 48 while raising ~47,000 false alarms per day. Recall alone cannot distinguish a detector from one that simply alerts more, and a reader could mistake that 0.27 for signal. A budget makes the comparison operational: what recall can you get for an alarm rate an analyst could actually work through | user decision, 2026-10-15; `tests/test_tau_and_budgets.py` |
| 2026-10-15 | **Tau tie-break made fully deterministic**: among thresholds with equal campaign-F1, prefer higher precision, then fewer false alarms, then the higher tau | Ties are common — a binary-scoring detector ties everywhere — and the previous rule (lowest tau) still left the outcome sensitive to what else tied. The new order expresses a preference rather than an accident: of two equally-scoring thresholds, take the one that cries wolf less, and failing that the cheaper, more conservative one. Tested to be independent of grid order | user decision, 2026-10-15 |
| 2026-10-15 | `features-smoke` training probes are rule-3 filtered and the largest window comes from a full scan; the oracle's misses are diagnosed rather than accepted | The oracle misses 18% (L = 24) and 11% (L = 48) of test campaigns despite perfect information, so the ceiling is well below 1.0 and the reason matters: a campaign unreachable because of temporal sparsity says something different about the method than one lost to a purity failure. `oracle-diagnose` classifies each, and deliberately changes no rule | user decision, 2026-10-15; `tests/test_oracle_diagnose.py` |
| 2026-10-17 | **Two configurations are compared by the paired method, and never by subtracting two median lead times.** Earliness is the per-campaign lead-time difference over the campaigns **both** detect — median, seeded bootstrap CI, and per-side earlier/tied counts, sign convention `b − a` so positive means `b` was earlier — applied equally to `normalized_lead_time` and `frac_observed`. Coverage is reported separately as the campaigns each side detects that the other misses, by id. Applies to the lookback comparison now and to the four conditions in Phase 4-5. The earlier claim that L = 48 loses earliness (74.7h → 72.1h) is **withdrawn as untested** | Those two medians are over different sets of campaigns. L = 48 additionally detects the harder, slower campaigns, which are detected late, so its median falls even if nothing was detected later than under L = 24 — a genuine improvement reads as a regression, and the bootstrap CIs overlapped almost entirely anyway. What changed is the composition of the set, not the earliness | user-reported, 2026-10-17; `tests/test_paired_comparison.py` |
| 2026-10-17 | **The matching rule stays frozen.** The oracle's misses on the full data are 36 sparsity + 2 `late_only` + 0 `purity_failure` at L = 24, and 30 + 1 + 1 at L = 48 | One purity failure in 70 misses. The 50% purity condition is not what the ceiling is made of, so relaxing it would buy almost nothing and would cost the frozen definition every earlier result depends on | user decision, 2026-10-17; `docs/reference/oracle_diagnose.md` |
| 2026-10-17 | **`temporal_sparsity` split into `structurally_disconnected` and `slow`**, precedence `purity_failure` → `late_only` → `structurally_disconnected` → `slow`. The first pools **every** own edge before the deadline with no window at all and still cannot connect 3 accounts; the second can, but never inside one window | The two imply opposite actions — nothing can be done about a disconnected campaign, while a slow one is a lookback question — and pooling them hid which of the two the ceiling was made of. On the sample all 10 disconnected cases are BIPARTITE and fragmentable, and the count does not move with L, which is the definition working | user decision, 2026-10-17; `tests/test_oracle_diagnose.py` |
| 2026-10-17 | **Fourth test subset `oracle_reachable`** (the campaigns the oracle detects at that lookback and split) and **`recall_relative_to_ceiling`** reported for every scorer and model. Phase 5 reports the visibility analysis on both `all` and `oracle_reachable` | The oracle reaches 0.821/0.893 on test, so recall against 1.0 counts campaigns no detector could find. For Phase 5 specifically: the structurally undetectable campaigns are concentrated in the fragmentable group and are undetectable in every condition at every visibility level, so leaving them in the denominator adds the same dead weight to all four conditions and compresses the very differences the experiment measures. `all` is still reported, because that is what an institution would experience | user decision, 2026-10-17; `tests/test_evaluation.py` |
| 2026-10-17 | **Stage 3.3 hyperparameters frozen and identical for every variant, lookback and seed**: 300 trees, depth 6, eta 0.1, subsample 0.8, colsample 0.8, `hist`, `binary:logistic` with `aucpr`, seed 0. The class weight `n_negative / n_positive` is applied **per window** as a per-row weight (positives carry their own window's weight, negatives 1.0) | Tuning per lookback would make the L comparison a comparison of tuning effort rather than of lookback length, and the same applies to the four conditions in Phase 4-5. The values are deliberately unremarkable: this is a baseline, not a contender. Per-window weighting rather than pooled because the windows differ by an order of magnitude in size, and a pooled weight would let a busy window set the balance for a quiet one | implementation, stage 3.3; `tests/test_xgb_baseline.py` |
| 2026-10-17 | `xgboost>=2.1` added to requirements.txt; torch and torch_geometric still wait for 3.4 | Each dependency lands with the stage that uses it, so a baseline session does not pay for the CUDA stack. Local development used xgboost 3.4.1; the Kaggle image ships its own, hence the loose pin | implementation, stage 3.3 |
| 2026-11-09 | **Stage 3.3's 15.9 GB peak does not constrain the GNN**, and stage 3.4 runs in six sub-stages (3.4.0-3.4.5), each stopping for approval. torch and torch_geometric still stay out of requirements.txt until 3.4.2 | The 15.9 GB is XGBoost's pooled training matrix, 9.1M rows x 66 features held at once. A neighbour-sampled GNN processes one window at a time and never materialises that pool, so carrying the figure over would size the design against the wrong quantity. The GNN's memory is measured in 3.4.0 and 3.4.3 instead. Sub-staging keeps a GPU session from being spent discovering what an environment check answers in minutes | user decision, 2026-11-09 |
| 2026-11-09 | **The oracle ceiling is a reference, not a strict upper bound.** Every "no detector can beat these" claim is corrected: the oracle is the best achievable **using only the laundering edges**, and `recall_relative_to_ceiling` can exceed 1 and is never clamped | The oracle keeps exactly the laundering edges; a model keeps whatever scores above tau, which is neither a subset nor a superset of them. Keeping *every* laundering edge is not always best: at L = 48 on stress it merges campaign 295 with 9 other campaigns into a 13-account cluster at purity 0.4615, just under the 0.5 the rule needs, while a model at tau ~ 0.99 keeps a sparser set and matches it. **Measured in all five seeds**, and 295 is the single `purity_failure` in the whole full-data diagnosis | user-reported, 2026-11-09; `tests/test_evaluation.py` |
| 2026-11-09 | **Stage 3.3 selected configuration: the full model at L = 48**, by the frozen protocol (validation recall 0.531 against 0.385). Both lookbacks are carried into 3.4 rather than treating L = 48 as settled | The protocol selects on validation and that is what was done, but the margin did not survive: ~3.8 of 26 campaigns on validation against ~1.4 of 56 on test, for ~3x the false alarms (10.5/day against 3.8). Paired per seed, L = 48 adds 1-3 test campaigns and loses 0-1 at 0.0h earliness. A one-campaign edge bought with triple the alarm load is not a settled answer | implementation, stage 3.3 |
| 2026-11-09 | **The payment-format ablation is settled: format buys coverage, not earliness.** Every later model still reports both variants | The direct paired earliness CI includes zero in **all ten** test cells (2 lookbacks x 5 seeds), so no earliness effect is demonstrated. Coverage is unambiguous: the full model finds 6-14 test campaigns per seed that no-format misses, against 0-1 the other way, and 0.76-0.80 of the ceiling against 0.58-0.60. No-format is also five times noisier across seeds (sd 0.052 against 0.009), which says it is leaning on weaker signal. Since the format-laundering link is a generator artefact, the no-format column is the one expected to transfer | implementation, stage 3.3 |
| 2026-10-19 | **Rule 3 of the paired method, frozen: two configurations are compared DIRECTLY against each other.** An A-versus-B difference is never inferred from `A vs C` and `B vs C`, whatever the shared reference C — the oracle ceiling, the centralized condition, anything | Each of those comparisons is taken over its own shared set of campaigns, so the two numbers are not on a common scale and their difference is not a quantity. Pairing fixes the denominator for one comparison; it does not make two comparisons subtractable. **Evidence:** this produced the withdrawn claim that dropping payment format cost 15h of earliness — 0.0h for the full model against the oracle, -15h for the ablated one, read as a 15h ablation effect. The direct ablation is -3.0h with CI [-12, 0], so nothing is demonstrated. Structurally the same error as subtracting two medians (2026-10-17), one level up, which is why it is now written down rather than assumed | user-reported, 2026-10-19; `tests/test_paired_comparison.py` |
| 2026-10-18 | **`MODEL_TAU_GRID` extended upward with a near-1 tail**: `0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999`, the same grid for every model and condition from now on. The edge check stays: a chosen tau of 0.9999 exits 3 and stops | The full-data quick check (L = 24, seed 0) chose **0.99, the top of the grid**, for both variants, with F1 rising **0.128 at tau 0.95 to 0.543 at 0.99** - a jump that large across the final interval means the optimum lies beyond it. Per-window class weights of **495-3,461** push scores so close to 1 that a linear grid cannot resolve the region where the decision happens; the tail halves the remaining distance to 1 at each step, putting the resolution in `1 - tau`. The extra thresholds are nearly free, since almost no edges survive 0.999. If a tau ever reaches 0.9999 the answer is not more nines but the class weighting itself | user decision, 2026-10-18; `docs/reference` quick check, `tests/test_tau_and_budgets.py` |
| 2026-10-17 | **A separate tau grid for model scorers** (extended 2026-10-18, above): `0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99`, used by every model — XGBoost, the GNN, and the federated conditions. The oracle and random scorers keep `DEFAULT_TAU_GRID`. The edge warning stays, and if a model's chosen tau is still at an edge on full data the run reports it and **stops** rather than extending silently. `select_tau` records the seconds each tau cost | The two kinds of scorer are genuinely different: a sanity scorer's scores are 0 or 1, so every threshold keeps the same edges and the low end would only cost clustering time, while a class-weighted model spreads its scores across the unit interval and the old floor of 0.5 truncated its optimum (validation F1 0.821 at tau 0.3 against 0.803 at 0.5 on the sample). One grid for every model means a comparison between models or conditions is never a comparison of how far each was allowed to search. Per-tau timing is recorded because clustering cost scales with the edges kept, so the low thresholds are the expensive ones and that cost should be visible rather than inferred | user decision, 2026-10-17; `tests/test_tau_and_budgets.py` |
| 2026-10-17 | **Payment-format ablation, frozen**: alongside `xgboost_centralized`, a variant `xgboost_centralized_no_format` trained without every `fmt_*` column (including `fmt___other__`), identical in all else. Every later model, the GNN and the federated conditions included, is reported **with and without** payment format | In the IBM synthetic data laundering is strongly linked to format: on the sample `fmt_ACH` alone carries 46% of the model's gain and the format columns ~87%. Format is an allowed input — a property of the transaction, not an identity or a label — but a feature that dominant in synthetic data is a shortcut a model can take instead of learning transactional structure, and a result resting on it would say more about the generator than about campaign discovery. The two variants are built from one matrix (the ablated one is a column slice), so they see byte-identical windows, labels and weights | user decision, 2026-10-17; `tests/test_xgb_baseline.py` |
| 2026-10-17 | **Every model configuration is trained at seeds 0-4**, and results are reported per seed and as mean [min, max] across seeds | A single run cannot distinguish an effect from seed noise. XGBoost's `subsample` and `colsample_bytree` make the fit genuinely stochastic, so the spread is real and worth carrying | user decision, 2026-10-17 |
| 2026-10-17 | A **score cache** is used for the tau sweep and warmed before it | A scorer's output does not depend on tau, so a 16-point grid would otherwise re-featurise every million-edge window sixteen times for sixteen identical answers. It cannot change the curve and the cached and uncached curves are pinned equal in a test. Warming it first keeps the per-tau seconds comparable instead of letting the first tau absorb all the featurisation | implementation, stage 3.3 |
| 2026-10-17 | **False alarms report which of the rule's two conditions failed** — purity < 0.5 vs coverage < 3 campaign accounts — with counts, replacing a single "best overlap" number | The old summary line said "best overlap 0.667 (below the 0.5 the rule requires)", which contradicts itself. Those clusters passed purity and failed coverage. On the sample 21 of 22 fail coverage, so the oracle's false alarms are mostly fragments of real campaigns too small to claim — a different problem from pollution by unrelated traffic, and one the three-account floor creates by design | user-reported, 2026-10-17 |
| 2026-11-13 | **Session B of the 3.4.4 run retrains the XGBoost baseline** (`train` then `evaluate`, 1.44h + 1.59h measured, CPU), because the stage 3.3 boosters no longer exist: that run was an **interactive** session, so its `/kaggle/working` was discarded. Session B therefore runs in **commit mode** | Nothing is lost that cannot be recreated - the models are a deterministic function of the seeds, the code and the data - but it has to be recreated in the same session that evaluates them, because `evaluate` reads the boosters `train` writes. The retrain is also worth having as a **regression check**: the re-evaluated numbers must reproduce the recorded stage 3.3 results (L = 48 full model, test recall 0.679, 0.760 of the ceiling, no edge taus), and a disagreement would matter more than the comparison it was run for | user-reported, 2026-11-13; `backend/scripts/kaggle/run_on_kaggle.md` section 5k |
| 2026-11-10 | **The GNN is not tuned in response to the 3.4.3 quick check.** It matched XGBoost's test recall at ~10x the alert volume and reached a lower validation campaign-F1 (0.418 against 0.543); the configuration stays frozen anyway. Only a **demonstrated bug** - a named defect with evidence - justifies a change | Equal tuning effort is the condition that makes the comparison a comparison of models. Tuning the GNN because its number came out low, and only then, would make the final table a record of how much search each model received, and the asymmetry would be invisible in it. An underwhelming metric is not evidence of a defect | user decision, 2026-11-10 |
| 2026-11-10 | **Alert budgets (10/50/100 a day) added to `evaluate-gnn`**, tau per budget chosen on validation from the same curve the F1 tau comes from | Required by the frozen design, and the only fair head-to-head: the two models' F1-chosen operating points sit an order of magnitude apart in alert volume (33.3 against 3.4 a day), so comparing recall at those points compares a cheap detector with an expensive one. No test information enters the operating point | user decision, 2026-11-10 |
| 2026-11-10 | **Every evaluation now writes per-campaign records (`*_outcomes.csv`), and the new `compare-models` subcommand is the only place a cross-model earliness claim may come from.** Stage 3.3 saved none, so its records are regenerated by re-running `evaluate` - CPU only, 1.59h measured, no retraining | Rule 3 requires A and B be compared directly, which needs both models' per-campaign detection times at a matched seed, lookback and operating point. Those existed only inside a running process, so the GNN-versus-XGBoost comparison could not be assembled from stage 3.3's files at all. Reading two files rather than two live models also means the comparison runs in a cheap CPU session across runs days apart, which is what Phase 4+ needs for the federated conditions | user decision, 2026-11-10; `tests/test_compare_models.py` |
| 2026-11-10 | **Two diagnostics added that select nothing**: edge-level average precision for both models on the same windows (including the four validation-detection-split windows the GNN early-stops on), and the size distribution of false-alarm clusters at each operating point | A campaign-level number is the product of two stages - rank the edges, then cluster and match - and on its own it does not say which stage two models differ in. The edge metric isolates the ranking; a gap present there is a ranking gap, a gap absent there lives in what survives clustering. The cluster sizes answer the second question a matched alert rate leaves open: a hundred three-account components is not the same work as a hundred forty-account ones. Neither chooses a tau, stops a model or appears in a headline, and the GNN's early stopping and the diagnostic call one function so they cannot drift | user decision, 2026-11-10; `tests/test_compare_models.py` |
| 2026-11-10 | **The GNN gets ONE fixed hyperparameter configuration**, frozen before any full-data training and never tuned on validation or test | This is the condition that makes the GNN-versus-XGBoost comparison fair. XGBoost was given one configuration for the same reason, so neither model's number reflects how much search it received. Any later tuning of either would have to be declared, because it would change what the comparison means | user decision, 2026-11-10 |
| 2026-11-10 | **`log1p` on the heavy-tailed node counts (`in_degree`, `out_degree`, `n_counterparties`, `n_currencies`), applied INSIDE THE GNN ONLY**, before standardising, with the GNN's own statistics fitted on training windows. The shared feature spec is unchanged and the stage 3.3 results stand | The root cause of the extreme logits, not a mitigation: the shared spec standardises *raw* counts, and standardising a heavy-tailed column does not bound it. Measured on the sample's largest windows, maximum \|x\| falls from 109.9 to 40.4 (L = 24) and 158.5 to 42.2 (L = 48), and on `out_degree` alone from 156.6 to 15.3. A tree splits identically under any monotone transform, so XGBoost is unaffected even in principle. It reduces but does not *bound* the inputs - `n_counterparties` still reaches 40 - so LayerNorm is kept as well | user decision, 2026-11-10; `tests/test_gnn.py` |
| 2026-11-10 | **Early stopping on validation edge-level average precision**, computed every epoch over the **validation detection split** only. Campaign-F1 is not used for stopping; tau is still chosen once at the end by validation campaign-F1 on the model grid | AUPRC is threshold-free, so it does not need a tau that has not been chosen, and it is the right summary at a 0.1% positive rate. Campaign-F1 per epoch needs cluster extraction at several thresholds and moves in steps of 1/26 over 26 validation campaigns, so it would be both slow and noisy. The detection-split restriction matters: edge-level AUPRC reads every label in a window, and the validation *evaluation horizon* reaches into the test detection split, so stopping on it would select on test-period edges | user decision, 2026-11-10; `tests/test_gnn.py` |
| 2026-11-10 | **DECIDED: full-batch training for both lookbacks**, closing the sampling decision reopened earlier the same day. Inference runs unchunked | Measured, not estimated: the largest L = 48 training window peaks at **10.70 GB reserved** of 15.6 GB and L = 24 at 6.39 GB, both inside the 12 GB rule fixed before the run; the largest evaluation windows need 2.71 and 4.75 GB. Steps take 0.28s and 0.45s. Full-batch removes three parameters the comparison does not need - fanouts, batch size and a sampler's bias - and sees every edge of a window at every step, which is what the frozen matching rule evaluates against. The 2026-10-10 sampling choice rested on a shape-based ~16.6 GB that the measurement halved | user decision, 2026-11-10; `gpu_probe_windows.*` |
| 2026-11-10 | **Measured: full-batch training fits both lookbacks** - the largest L = 48 window peaks at **10.70 GB reserved** on a 15.6 GB T4 against the 12 GB rule fixed before the run, and L = 24 at 6.39 GB. Inference needs no chunking (4.75 GB, 0.18s for the largest evaluation window). The regime choice itself is 3.4.1's to propose on this evidence | The 2026-10-10 estimate was ~16.6 GB, the measurement is 10.70. Real windows came in ~25% heavier than `gpu-check`'s synthetic graphs (5.73 vs 4.6 MB per 1,000 edges), so the synthetic projection of 8.6 GB would have understated it - which is exactly why the probe ran real windows before anything was decided. Host memory peaks at 14.1 GB during the feature-spec fit, not on the GPU | `gpu_probe_windows.*`, 2026-11-10 |
| 2026-11-10 | **The probe exposed an activation explosion that 3.4.1 must resolve**: losses of 10,615 (L = 24) and 356,394 (L = 48) after one step from random init, where class-weighted BCE should start near 1-2. No training happens until a normalisation is proposed and approved | GINEConv aggregates by **sum**, and these windows hold accounts of very high degree; summing thousands of 64-dim messages three layers deep with nothing between them compounds beyond what a linear readout can map to sensible logits. The largest window being 34x worse than the smaller one is the pattern that explanation predicts. The frozen design names GINEConv, 2-3 layers and hidden 64 but says nothing about normalisation, so adding one is a change to approve rather than assume. A model whose activations explode on the busiest windows would still train and still report numbers | `gpu_probe_windows.*`, 2026-11-10 |
| 2026-11-10 | **Every GPU subcommand installs `torch_geometric` itself** when the session lacks it, via one shared `ensure_torch_geometric`. Only `gpu-check` also attempts the optional extensions | A Kaggle session starts clean, and `gpu-probe-windows` failed on a fresh one with "No module named 'torch_geometric'" because only `gpu-check` installed it. The alternative was a documented rule that `gpu-check` must run first, which is a rule that gets forgotten - as it just was. The extensions stay exclusive to `gpu-check` because they are not required and a training command should not spend session minutes on wheels it can run without | user-reported, 2026-11-10 |
| 2026-11-10 | **The stage 3.4 sampling decision is REOPENED**, and `gpu-probe-windows` added to settle it on a measurement. Full-batch will be proposed only if the largest L = 48 training window peaks **below 12 GB reserved** on one T4; whichever regime wins applies to **both** lookbacks | The 2026-10-10 decision rested on a shape-based estimate of ~16.6 GB for the busiest L = 48 window against 16 GB cards. `gpu-check` measured GINEConv 3x64 at **~4.6 MB per 1,000 edges** (251 MB at 50k, 936 MB at 200k, 2,302 MB at 500k), which projects **~5.1 GB** for the largest L = 24 window and **~8.6 GB** for the largest L = 48 one - roughly half the estimate. If that holds on real windows, sampling would add fanouts, a batch size and a sampler's bias to a comparison that does not need them. It is only a projection, though: a random graph's account-degree distribution is not a real window's, and linearity holds until it does not. Hence the probe before the proposal | user decision, 2026-11-10; `gpu_probe_windows.*` |
| 2026-11-10 | **Correction: a Kaggle GPU notebook has 33.7 GB of system RAM**, so stage 3.3's 15.9 GB peak would fit there. Every "too large for a GPU notebook" claim is corrected | The claim confused the GPU's **device** memory (15.6 GB per T4) with the notebook's **host** RAM, which is a different quantity in a different place. Stage 3.3 still belongs on CPU, for the reason that always applied: its bottleneck is pandas and scipy, which a GPU does not touch, and stage 3.4 needs the weekly GPU quota. The 15.9 GB also does not constrain the GNN, which never materialises a pooled matrix | user-reported, 2026-11-10 |
| 2026-10-10 | **Stage 3.4 training regime frozen: both candidate lookbacks train with neighbour sampling** (LinkNeighborLoader-style mini-batches of target edges with sampled multi-hop neighbourhoods). Inference scores full windows, chunked if needed. Fanouts, batch size and negative/positive handling are chosen in stage 3.4 and reported before training | If L = 24 trained full-batch and L = 48 trained with sampling, the L comparison would confound lookback length with training method, and the chosen L could not be attributed to the lookback. Using one regime for both keeps that comparison clean. It also sidesteps the capacity question: the busiest L = 48 window is estimated at ~16.6 GB full-batch against 16 GB GPUs | user decision, 2026-10-10. **The capacity half of this was reopened on 2026-11-10**: measured scaling is ~4.6 MB/1k edges, about half the estimate, so full-batch may fit. The *one regime for both lookbacks* half stands either way |
| 2026-10-10 | `features-smoke` samples only rule-3-valid training detection times, and a cheap full scan of **every** usable detection time supplies the largest-window figures | The first full-data run reported Sept 2 12:00 at L = 48 as the largest candidate window (1.5M edges), but that time violates rule 3: its lookback starts before `DATA_START`, so the window was silently clamped — which is also why its L = 48 and L = 72 rows were identical. Sampling five detection times cannot find the true maximum anyway, so the scan replaces guesswork with a full enumeration (counts only, no features) | user-reported, 2026-10-10 |
| 2026-10-10 | "Mostly involved in unassigned-laundering edges", for the ambiguous rule, is read as **at least 50% of the cluster's accounts** touching a laundering edge that belongs to no campaign | The design says "mostly" without a number; 0.5 mirrors the matching rule's own threshold, so the two use one consistent notion of majority. Exposed as `constants.AMBIGUOUS_MIN_FRACTION` rather than buried in the code | interpretation, stage 3.2 |
| 2026-10-10 | Stress evaluation horizon frozen at `(2022-09-08 00:00, CUTOFF]`, capped per campaign by its deadline, with other-split hits applying as in rule 1 | Stress campaigns start Sept 8-10, so the horizon opens where the validation horizon closes: the two abut without overlapping, and stress sits strictly inside the test horizon. Previously left unspecified and raising rather than guessed | user decision, 2026-10-10 |
| 2026-10-10 | Lookback candidates reduced to **L ∈ {24, 48}**; L = 72 dropped | Three independent reasons: (1) rule 3 leaves only 5 full-history training detection times at L = 72 versus 13 at L = 24, and those 5 windows overlap each other heavily, so they carry far less independent signal than the count suggests; (2) estimated full-batch GINE memory is ~18.6 GB against Kaggle's 16 GB GPUs, so it could not be trained the same way as the other candidates; (3) a 3-day lookback is large relative to a 10-day dataset, making the window a substantial fraction of the whole study period. L = 72 is still measured once by `features-smoke` and labelled `[EXCLUDED]`, so the evidence stays on the record instead of only in prose | user decision, 2026-10-10 |
| 2026-10-10 | Three evaluation rules frozen: other-split hits counted separately, the validation detection horizon extended to Sept 8, and training restricted to detection times with a full lookback | (1) A cluster that finds a real campaign from another split is neither a success for the split being scored nor a false alarm; counting it either way would misstate precision. (2) Validation campaigns start on Sept 5 but need until Sept 8 to complete, so a horizon ending Sept 6 would score them before they had happened; only validation-campaign labels inform tau, so the overlap with the test horizon leaks nothing. (3) A silently short window would train the model on a distribution it never meets at evaluation time | user decision, 2026-10-10; `tests/test_evaluation_rules.py` |
| 2026-10-10 | Stage 3.1 feature deviations and interpretations accepted as documented: `__other__` one-hot columns; `log1p` rather than `log`; in-amount from *Amount Received* and out-amount from *Amount Paid*; counterparties and currencies pooled across both directions; scaling applied to continuous columns only; float32 storage with float64 arithmetic; numpy instead of scikit-learn for the scaler | Each is either forced (log(0) is undefined; unseen categories must go somewhere) or an interpretation of wording the design left open, and each is cheaper to keep documented than to re-litigate. scikit-learn still arrives with stage 3.3, where it is actually needed | user approval of the Stage 3.1 review, 2026-10-10 |
| 2026-10-09 | "Unfragmentable campaigns" pinned at **2**, and the five visibility SHA-256 digests recorded as reference values | Determinism confirmed on Kaggle across 2 independent sessions and 3 separate pipeline processes at commit `55b645e` (Python 3.13.15, pandas 2.3.3): all produced byte-identical visibility files and the same count. That is what makes the number trustworthy — before the fix, runs disagreed (3 vs 2), so the value could not be pinned from a single run. The notebook's exploratory 2 is consistent but was computed without conflict resolution, so it was never sufficient evidence on its own | `docs/reference/visibility_hashes_session1.json`, Verified numbers |
| 2026-10-07 | torch, torch_geometric, xgboost and scikit-learn are **not** added to requirements.txt yet | They are only needed from stages 3.3-3.4. Installing ~2GB of unused ML dependencies now would slow every Kaggle session and risk disturbing the pre-installed CUDA stack during a run whose only job is verifying Phase 2. Each lands with the stage that uses it | implementation, stage 3.0 |

## Verified numbers

These are also used as the full-data regression test suite
(`@pytest.mark.fulldata`, skipped when the full dataset is absent).

| Metric | Value | Notebook cell |
|---|---|---|
| Total transaction rows | 5,078,345 | `b42159f4` |
| Rows on/after CUTOFF (2022-09-11) | 1,108 | `280d9179` |
| Campaigns (pattern-file attempts) | 370 | `cd70c51d` |
| Pattern-file transactions | 3,209 (all matched, 0 unmatched) | `cd70c51d` |
| Laundering rows (`Is Laundering == 1`) | 5,177 | `b42159f4`, `cd70c51d` |
| `eval_ok` campaigns (total) | 258 | `280d9179` |
| `eval_ok` by split | train 100, val 26, test 56, stress 76 | `b7e46dec` |
| `eval_ok` fragmentable campaigns | 152 (test: 37) | `b7e46dec` |
| Fragmentable test campaigns with `shares_train_account` | 7 of 37 | `cb0caa40` |
| `base_type` counts | CYCLE 54, GATHER-SCATTER 51, BIPARTITE 49, FAN-OUT 48, SCATTER-GATHER 44, STACK 43, RANDOM 41, FAN-IN 40 | `11875435` |
| Unfragmentable campaigns | 2 | Kaggle, commit `55b645e`, 2 sessions × 3 processes, 2026-10-09 |

The first 23 were reproduced on the full dataset on 2026-10-08
(`docs/reference/verify_report.md`). The unfragmentable count was pinned
separately on 2026-10-09, after the determinism fix, once two independent
sessions agreed — before the fix it varied between runs.

### Reference visibility digests

The visibility search is seeded, so the same commit and environment must
produce byte-identical files. These are the reference digests; any future run
that disagrees means something has become order-dependent again, and the
`test_determinism.py` suite should be the first place to look.

| File | SHA-256 |
|---|---|
| `visibility/seed_0.parquet` | `8953a9d77bf694e72e18fada8bd3065dfc06b4b1589e327dee6901c8adb9b0ff` |
| `visibility/seed_1.parquet` | `36a322ad1cfc9eadd2a4c554a9497a11ac4c9aa6936071b76acc1dc87ff56204` |
| `visibility/seed_2.parquet` | `d5d519ecb24001f853de2eaee2928813dda308fb5a96894d2d98bd0ca3eaed16` |
| `visibility/seed_3.parquet` | `f56ad7e33736a9824630f472a4fd090625e24194083bfde19a0ef27fcf3643bc` |
| `visibility/seed_4.parquet` | `df3139abbc850b46ecf2d499b31d55f47bffdb05ab53fc8bbdac29cd9d5ac63a` |

**Environment these were produced in** (digests are only comparable within
it, since a different pyarrow could change the bytes without changing the
data): Kaggle, commit `55b645e`, Python 3.13.15, pandas 2.3.3. Confirmed
across 2 independent sessions and 3 separate pipeline processes, all
byte-identical.

The copy installed in `backend/artifacts/` was verified against these
digests on 2026-10-09, before installation and again afterwards, so the
local artifacts are known to be the deterministic ones.
Machine-readable copy: `docs/reference/visibility_hashes_session1.json`. Regenerate
and re-compare with `backend/scripts/kaggle/run_on_kaggle.md` §5b.

Note on "eval_ok fragmentable = 152": this is the **topology** count —
`eval_ok` campaigns with a non-hub `base_type`, i.e. the `reassignable`
column — counted *before* the `"unfragmentable"` relabel, which is how the
reference notebook counted it (cell `b7e46dec`, which has no relabel step).
After the floor run relabels some of them, the three-valued `group` column
shows fewer `fragmentable`; the set of campaigns actually reassigned is
unchanged. `scripts/run_pipeline.py` prints both.

## Repository structure

```
project-root/
  README.md                        this document
  CLAUDE.md                        short pointer + update rule + core rules
  .gitignore                       data/, artifacts/ contents, node_modules, .venv, __pycache__, .env
  docs/
    PROJECT_OVERVIEW.md           readable summary of the whole project + improvement roadmap
    demo_checklist.html           standalone run sheet for a live demo: startup commands,
                                    a 10-step browser checklist with the expected value at
                                    each step, and the campaigns to show. No dependencies,
                                    no network, opens from the filesystem
    reference/
      01_data_verification.ipynb   read-only: verified Kaggle exploration behind every definition above
      verify_report.md             stage 3.0 Kaggle run: all 24 regression numbers
      regression_actuals.json      the same, machine-readable
      visibility_hashes_session1.json  reference digests, session 1
      visibility_hashes.json       the same digests, third independent session (identical)
      pipeline_report.md           full-data pipeline run, commit eacb6d3
      features_smoke.{md,csv,json} window sizes, feature dims, GPU estimates
      features_smoke_scan_full.csv every usable detection time, scanned
      evaluate_scorers.{md,csv,json}  oracle vs random on the full data
      evaluate_scorers_tau_curve.csv  precision/recall/FA per tau
      oracle_diagnose.{md,csv,json}   every oracle miss, diagnosed, + the paired comparison
      train_baseline.{md,csv,json}    stage 3.3: 20 boosters, windows, feature importance
      train_baseline_windows.csv      per training window: edges, positives, class weight
      train_baseline_importance.csv   gain by feature, one seed per variant
      evaluate_baseline.{md,csv,json} stage 3.3 results: every subset, seed and lookback
      evaluate_baseline_across_seeds.csv  mean/min/max/sd over the five seeds
      evaluate_baseline_paired.csv    model vs ceiling, lookbacks, and the format ablation
      evaluate_baseline_tau_curve.csv per-tau precision/recall/F1 and the seconds each cost
      evaluate_baseline_budgets.csv   recall and lead time at 10/50/100 false alarms a day
      evaluate_baseline_operating.csv recall vs false alarms, for plotting
      gpu_check.{md,json}             stage 3.4.0: GPU, RAM, PyG install, GINEConv smoke test
      gpu_probe_windows.{md,json}     stage 3.4.1: real windows through the real model
      train_gnn.{md,csv,json}         stage 3.4.3: the trained GNN, epochs, memory, determinism
      evaluate_gnn.{md,csv,json}      stage 3.4.3 quick check: one GNN, L = 24, seed 0
      evaluate_gnn_paired.csv         that model against the ceiling, paired
      evaluate_gnn_tau_curve.csv      its per-tau precision/recall/F1 on validation
      evaluate_gnn_inference.csv      measured featurise and forward cost per edge
      evaluate_gnn_partial.csv        the partial report, written per configuration
      --- stage 3.4.4, the full run (2026-11-14) ---
      evaluate_outcomes.csv           XGBoost: one row per campaign per operating point
      evaluate_gnn_outcomes.csv       the GNN: the same, and what compare-models pairs on
      evaluate_clusters.csv           XGBoost false-alarm cluster sizes per operating point
      evaluate_gnn_clusters.csv       the GNN: the same
      evaluate_edge_metrics.csv       XGBoost edge-level average precision per window set
      evaluate_gnn_edge_metrics.csv   the GNN: the same
      evaluate_gnn_budgets.csv        GNN recall and lead time at 10/50/100 alarms a day
      evaluate_gnn_operating.csv      GNN recall vs false alarms, for plotting
      compare_models.{md,json}        stage 3.4.5: the direct GNN-vs-XGBoost gate report
      compare_models_paired.csv       earliness, paired per seed and operating point
      compare_models_coverage.csv     which campaigns each model finds that the other misses
      compare_models_edge_ap.csv      edge ranking compared: where the GNN is and is not behind
      compare_models_clusters.csv     false-alarm cluster shape: the extraction mechanism
  backend/
    requirements.txt               Python dependencies: pandas, numpy, pyarrow, pyyaml, pytest, fastapi, uvicorn, pydantic
    pyproject.toml                 setuptools packaging for `fraudcamp` (src/ layout) + pytest `fulldata` marker
    configs/
      local.yaml                  paths for the downloaded sample
      kaggle.yaml                 paths for the full dataset on Kaggle
    src/fraudcamp/
      __init__.py
      constants.py                all frozen definitions, single source of truth for pipeline + API
      load_data.py                CSV loading, account IDs, cutoff split
      campaigns.py                pattern parsing, matching, campaign table
      splits.py                   train/val/test/stress, hub/fragmentable/unfragmentable groups, shared accounts
      institutions.py             bank -> institution volume-balanced assignment
      visibility.py               visibility targets, local search, conflict resolution, bins
      pipeline.py                 end-to-end orchestration shared by the runner and the exporter
      export.py                   writes backend/artifacts/
      regression.py               the full-data expected numbers + actual/expected comparison
      windows.py                  detection-time grid, causal window slicing, detection-time splits
      features.py                 edge + node features, training-only vocabularies and scaler
      models/
        __init__.py               the Phase 3 baselines
        xgb_baseline.py           stage 3.3: the non-graph control (no tau, no campaign knowledge)
        gnn.py                    stage 3.4: GINEConv edge classification, full-batch, resumable
      federation/
        __init__.py               Phase 4: the public federation API
        views.py                  stage 4.0: per-institution views of every window,
                                    boundary accounts, and the per-institution
                                    training-load report
      evaluation/
        __init__.py               the public evaluation API
        clusters.py               extraction, the matching rule, outcome classification
        evaluate.py               the detection-time loop, lead time, metrics, bootstrap CIs,
                                    per-campaign records, cluster-size distributions
        edge_metrics.py           edge-level average precision: the diagnostic, and the GNN's
                                    early-stopping metric, one definition for both
        compare.py                two detectors compared directly from their saved records
        scorers.py                the oracle and random sanity scorers
        diagnose.py               why the oracle misses a campaign; what its false alarms are
    app/
      __init__.py                 puts src/ on sys.path so the app can import fraudcamp uninstalled
      main.py                     create_app(): FastAPI app, CORS, router registration, artifact load
      schemas.py                  pydantic response models for every endpoint
      routers/
        meta.py                   /health, /summary, /methodology
        campaigns.py              /campaigns (filter, sort, search, paginate), /campaigns/{id}
        visibility.py             /campaigns/{id}/visibility, /visibility/distribution
        results.py                /results/summary, /results/detections (404 until Phase 5)
      services/
        artifacts.py              ArtifactStore: loads+caches artifacts, guarded accessors, JSON coercion
        methodology.py            frozen definitions as structured JSON, values read from constants
    scripts/
      make_sample.py              run ON KAGGLE: all pattern txns + random 1% of the rest, seed 42
      make_demo_data.py           seeded SYNTHETIC dev dataset (not the research sample)
      run_sanity_scorers.py       runs the oracle and random scorers through the real evaluation code
      run_xgb_baseline.py         stage 3.3 locally on the 1% sample: train, tau, evaluate, paired
      run_institution_split.py    stage 4.0: per-institution training rows and positives at
                                    every (seed, target), + the visibility cross-check
      run_structural_ceiling.py   evidence for the 4.1 proposal: the per-institution STRUCTURAL
                                    ceiling (upper bound on its oracle ceiling), and the union
      run_isolated.py             stage 4.1 locally on the 1% sample: the isolated condition
                                    end to end, per institution and pooled two ways
      run_pipeline.py             runs the pipeline end to end; prints every Verified-numbers value; --export writes artifacts
      kaggle/
        run_on_kaggle.md          the exact notebook cells for running this repo on Kaggle
        kaggle_runner.py          verify / pipeline / features-smoke / evaluate-scorers /
                                  oracle-diagnose / make-sample / train / evaluate /
                                  gpu-check / gpu-probe-windows / train-gnn / evaluate-gnn /
                                  compare-models
    tests/
      conftest.py                 src/ on sys.path + the `mini_dataset` synthetic-dataset fixture
      test_load_data.py           leading-zero banks, account IDs, timestamp parsing, cutoff
      test_campaigns.py           pattern parsing, base_type normalization, matching, eval_ok
      test_splits.py              split boundaries, hub/fragmentable, unfragmentable, shared accounts
      test_institutions.py        volume-balanced assignment, determinism, natural institution lookup
      test_visibility.py          visibility_of, local search, conflict resolution, bins
      test_pipeline.py            end-to-end build() on the synthetic dataset; determinism; reassignable set
      test_export.py              every artifact's shape, contents, determinism, empty results/
      test_api.py                 every endpoint via TestClient against a real exported fixture artifact set
      test_determinism.py         cross-process reproducibility: subprocess runs under differing PYTHONHASHSEED
      test_windows.py             detection-time grid, splits, strict causal window bounds
      test_evaluation_rules.py    the three frozen evaluation rules + their violation checks
      test_evaluation.py          extraction, matching, outcomes, lead time, metrics, bootstrap CIs
      test_tau_and_budgets.py     the deterministic tau tie-break and the alert budgets
      test_oracle_diagnose.py     the four oracle-miss reasons, each on a hand-built case
      test_compare_models.py      the saved records, the round trip, the direct comparison,
                                    edge-level AP, false-alarm cluster shape
      test_paired_comparison.py   the paired lead-time method, incl. the misleading-medians case
      test_xgb_baseline.py        the baseline: rule 3, score alignment, identity, save/load
      test_gnn.py                 the GNN: causality, identity, log1p, the loss, resumability
      test_federation_views.py    stage 4.0: a view is exactly what an institution sees,
                                    the views cover everything, and they reproduce the
                                    recorded achieved visibility
      test_federation_isolated.py stage 4.1: the mask lives in the scorer, the ground truth
                                    stays global, each institution's own ceiling, the union
                                    (earliest wins, deduplicated), the two pooled means
      test_features.py            feature shapes + the leakage properties (future, labels, identity)
      test_fulldata.py            @pytest.mark.fulldata, one test per regression number; skip if no data
    data/                         gitignored: raw/, sample/, processed/
    artifacts/                    gitignored except .gitkeep; summary.json, campaigns.parquet,
                                  campaign_transactions.parquet, institutions.json, visibility/, results/
  frontend/                       Vite + React + TypeScript app
    README.md                     pointer to this document + the npm commands
    .env.example                  VITE_API_URL template
    vite.config.ts                react + tailwind plugins, vitest (jsdom) config
    src/
      main.tsx                    entry: QueryClient, ThemeProvider, BrowserRouter
      App.tsx                     routes; Stage 5 pages render an honest placeholder
      index.css                   Tailwind v4 import, dark variant, chrome/ink tokens
      theme.ts                    THE colour source: institutions (8), conditions (4), bins (4)
      api/
        types.ts                  TS mirrors of every pydantic response model
        client.ts                 fetch wrapper, ApiError (isNotExported / isMissing)
        hooks.ts                  TanStack Query hooks per endpoint
      lib/
        themeContext.ts           theme context + useThemeMode hook
        ThemeProvider.tsx         light/dark state, persisted to localStorage
        glossary.ts               plain-language definitions for technical terms
        visibility.ts             the ONE frontend-computed metric: mirror of visibility_of
        visibility.test.ts        chain = 1.0, fan-out = 1.0, bins, per-institution subsets
      components/
        Layout.tsx                sidebar nav, top bar, API status, theme toggle, skip link
        FourStepStory.tsx         the four-picture explanation of the research problem
        cards.tsx                 StatCard, ChartCard, InstitutionBadge, Pill
        states.tsx                Skeleton, LoadingBlock, EmptyState, ErrorState
        ConditionLegend.tsx       the 4 conditions, each with colour AND dash pattern
        GlossaryTerm.tsx          hover/focus tooltip on a technical term
        CampaignGraph.tsx         Cytoscape graph: accounts as nodes, transactions as arrows
        Timeline.tsx              time replay slider with play/pause, deadline + cutoff marks
        VisibilityToy.tsx         the drag-accounts-between-banks sandbox
        states.test.tsx           empty-state and error-state tests
      pages/
        HomePage.tsx              research question, four-step story, key stats, pipeline status
        DatasetPage.tsx           stat cards + 4 charts + split/group breakdowns + cutoff card
        CampaignExplorerPage.tsx  filter/sort/search/paginate table
        CampaignDetailPage.tsx    graph, replay, visibility controls, bank's-eye view, metadata
        VisibilityLabPage.tsx     the toy + real achieved-visibility distributions per target
        ResultsPage.tsx           empty state + labelled outlines of the planned figures
        MethodologyPage.tsx       /api/methodology rendered as cards
        campaignFilters.ts        pure filter logic (query building, chips, sorting)
        campaignFilters.test.ts   filter logic tests
        pages.render.test.tsx     render smoke tests for Home, Dataset, Explorer
        stage5.render.test.tsx    render tests for Detail, Lab, Results, Methodology
      test/setup.ts               jest-dom matchers + Testing Library cleanup
```

## Setup

- **Backend:** Python 3.11+ virtual environment, then `pip install -r
  backend/requirements.txt` (pandas **>= 2.2**, numpy, pyarrow, pyyaml,
  pytest, fastapi, uvicorn, pydantic). The ML dependencies for Phase 3
  (torch, torch_geometric, xgboost, scikit-learn) are added by the stages
  that use them, not up front — see the Decision log. Tests also run directly against any Python
  interpreter that already has those packages — `backend/tests/conftest.py`
  puts `backend/src` on `sys.path`, so an editable install is convenient
  but not required.
  > Note (this machine, 2026-10-05): the C: drive had ~65MB free, too
  > little to create a venv or install packages. The system Python
  > already had every required package, so Stage 1 was developed and
  > tested against it directly. Free up disk space before relying on a
  > project-local venv.
- **Frontend:** Node.js (LTS; developed on Node 22); `npm install` inside
  `frontend/`. Copy `frontend/.env.example` to `frontend/.env` if the API is
  not on `http://localhost:8000`.
- **Environment variables:**
  - `FRAUDCAMP_ARTIFACTS_DIR` — which artifacts directory the API server
    reads. Defaults to `backend/artifacts`.
  - `VITE_API_URL` — backend base URL for the frontend (Stage 4).

  Pipeline **data** paths come from `backend/configs/local.yaml` or
  `backend/configs/kaggle.yaml`, not from environment variables.

## How to run

- **Run the pipeline on sample data** (after `make_sample.py` output has
  been downloaded into `backend/data/sample/`):
  ```
  cd backend
  python scripts/run_pipeline.py --config configs/local.yaml
  ```
- **Run the pipeline on the full Kaggle data** (on Kaggle, with the
  dataset mounted):
  ```
  cd backend
  python scripts/run_pipeline.py --config configs/kaggle.yaml
  ```
  Prints every number in the [Verified numbers](#verified-numbers) table
  so a run can be checked against it.
- **Write the artifacts** (add `--export` to either command above):
  ```
  cd backend
  python scripts/run_pipeline.py --config configs/local.yaml --export
  ```
  Writes everything listed in [Artifacts](#artifacts) into the
  `artifacts_dir` from the config, and prints each file's size. Runtime is
  dominated by the visibility local search — 6 runs (5 experiment seeds +
  the floor run) × 4 targets × ~152 campaigns × 400 iterations — expect a
  few minutes on the full dataset, seconds on the sample.
- **Generate the sample on Kaggle and copy it locally:** on Kaggle, run
  `python backend/scripts/make_sample.py` (writes
  `/kaggle/working/HI-Small_Trans.csv` and `HI-Small_Patterns.txt`); then
  download those two files into `backend/data/sample/`.
- **Train and evaluate the baseline locally, on the 1% sample:**
  ```
  cd backend
  python scripts/run_xgb_baseline.py --config configs/local.yaml
  ```
  Trains one model per lookback, picks tau on validation, evaluates every
  split on all four subsets against the oracle ceiling, and writes
  `artifacts/baseline_dev/`. Development only — the sample's background is
  thinned to 1%, so precision and false alarms per day do not transfer. A
  couple of minutes end to end.
- **Report the Phase 4 institution split (stage 4.0):**
  ```
  cd backend
  python scripts/run_institution_split.py --config configs/local.yaml
  ```
  For every `(seed, target)` it builds each institution's view of the rule-3
  training windows and reports its pooled rows, positives, positive rate and
  class weight, flagging any institution with fewer than 10 positives; it
  also cross-checks every campaign's visibility recomputed from the views
  against the pipeline's own figure and reports the mismatch count, which
  must be 0. `--lookbacks`, `--seeds` and `--targets` narrow the run. Writes
  `artifacts/federation_dev/` (gitignored, development only). About two
  minutes on the sample.
- **Measure the per-institution structural ceiling (evidence for the 4.1
  proposal):**
  ```
  cd backend
  python scripts/run_structural_ceiling.py --config configs/local.yaml
  ```
  For every `(target, seed)` and institution it asks whether that
  institution's view holds a connected component of 3+ of a campaign's
  accounts, over the campaign's own laundering edges — an **upper bound** on
  that institution's oracle ceiling, since it ignores the detection grid and
  the deadline. It also prints the centralized value and the count of
  campaigns holding an account with 2+ distinct campaign counterparties,
  which must be equal (see
  §[the 4.1 proposal](#stage-41-the-isolated-condition--approved-and-implemented-2026-11-14)).
  Writes `artifacts/federation_dev/structural_ceiling.csv`. Under 10 seconds
  on the sample.
- **Run the isolated condition on the sample (stage 4.1):**
  ```
  cd backend
  python scripts/run_isolated.py --config configs/local.yaml
  python scripts/run_isolated.py --config configs/local.yaml --targets 1.0 --seeds 0 --splits val
  ```
  For each `(target, seed)` it trains one booster per institution on that
  institution's own view, picks each institution's tau on validation, computes
  each institution's **own** oracle ceiling, and reports the two pooled
  numbers and the union at all seven operating points. Writes
  `artifacts/isolated_dev/` (gitignored, development only). **Development
  check, not a result:** the sample's thinned background makes precision,
  false alarms per day and every budget tau untransferable. Exit 3 if any
  chosen tau is at a grid edge, after the reports are written. A few minutes
  per `(target, seed)` on the sample.
- **Run the isolated condition on the full data (on Kaggle):**
  ```
  python scripts/kaggle/kaggle_runner.py train-isolated    --config configs/kaggle.yaml
  python scripts/kaggle/kaggle_runner.py evaluate-isolated --config configs/kaggle.yaml
  ```
  The frozen scope is the default: targets 1.0/0.75/0.5/0.25, seeds 0-2,
  L = 48 — 96 institution-cells. `train-isolated` saves each booster as it
  finishes and skips it on a restart; `evaluate-isolated` writes each
  `(target, seed)` cell's rows as it finishes and resumes past them. It
  **never trains**, so a lost evaluation session can never silently retrain
  on different boosters. `--targets`, `--seeds` and `--force` narrow or redo a
  run; more than one `--lookbacks` is refused, because Phase 4 runs one. It
  **stops without recording a cell** whose F1 tau lands at the bottom of the
  tau grid, which should not happen at full-data prevalence;
  `--allow-low-edge-tau` overrides that and is only for the thinned sample.
- **Train and evaluate the baseline on the full data (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py train
  python scripts/kaggle/kaggle_runner.py evaluate
  ```
  Covers both variants, both lookbacks and seeds 0–4 — 20 boosters — and
  writes `train_baseline.{md,json}` + 3 CSVs and
  `evaluate_baseline.{md,json}` + 6 CSVs. **CPU is enough**: the dominant
  cost is pandas featurisation and scipy clustering, neither of which a GPU
  touches, and stage 3.4 needs the GPU quota. Expect **~3.5–4 hours** end to
  end — training ~1.5 h (215s per booster, measured at L = 24) and
  evaluation ~2 h (~3.5 min per model at L = 24, measured) — with memory
  peaking at **9.9 GB at L = 24, measured**, and an estimated 12–13 GB at
  L = 48. `evaluate` exits 2 if no model is there and **exits 3 if a chosen
  tau lands on an edge of the model grid**, after writing every report. Full
  details in §5g of `run_on_kaggle.md`.
- **Check a GPU notebook before stage 3.4 (on Kaggle, Accelerator: GPU):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py gpu-check
  ```
  Reports the GPU, the host RAM, the torch/CUDA/PyG versions and which PyG
  extensions install, then runs a GINEConv forward+backward at the real
  feature widths. Writes `gpu_check.{md,json}` and exits 2 if there is no
  usable GPU. This is the **only** command here that needs a GPU; §5h of
  `run_on_kaggle.md` has the detail.
- **Verify Phase 2 against the full data (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py verify
  ```
  Runs `pytest -m fulldata` and prints a PASS/FAIL table of every number in
  [Verified numbers](#verified-numbers), expected against actual. Writes
  `/kaggle/working/verify_report.md` and `regression_actuals.json`. Exits
  non-zero if anything fails or if the dataset is not where
  `configs/kaggle.yaml` expects it.
- **Run the pipeline and zip the artifacts (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py pipeline
  ```
  Builds the pipeline, exports the artifacts, writes
  `/kaggle/working/artifacts.zip` plus `pipeline_report.md`, and re-checks
  the same regression numbers so the run is self-verifying. Download the zip
  and unpack it into `backend/artifacts/` to give the API and frontend real
  data.

- **Measure window sizes and GNN feasibility (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py features-smoke
  ```
  Builds windows at L = 24, 48 and 72 hours at five detection times — a
  weekday and a weekend day from train and from test, plus the one weekday
  validation has (validation spans only Mon Sept 5 → Tue Sept 6, so it has
  no weekend time) — and reports per window: edges, accounts, laundering
  edges, laundering rate, class weight, slice and build time, process
  memory, the in-memory tensor footprint, and the analytic size of the
  XGBoost matrix. It also reports, per lookback, how many training detection
  times survive rule 3 (full-history windows). Then it estimates full-batch
  GPU memory for a 3-layer GINEConv with hidden 64 on the largest window.
  Writes `features_smoke.{csv,json,md}` to `/kaggle/working/`.

  The feature spec for this command is fitted on a **single** training
  window, deliberately: the command measures sizes and timing, and fitting
  across all training windows would concatenate overlapping windows into
  tens of millions of rows for no benefit here.

  All three commands print elapsed time and memory per step, take `--config`
  and `--out`, and work locally too (pointed at a sample config, the regression
  table is skipped with an explanation rather than reporting 23 meaningless
  failures). The full notebook walkthrough is
  `backend/scripts/kaggle/run_on_kaggle.md`.
- **Validate the evaluation on the full data (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py evaluate-scorers
  ```
  Oracle and random at L = 24 and 48 over the val, test and stress
  horizons, with tau chosen on validation by campaign-level F1 (the full
  precision/recall-vs-tau curve is saved), bootstrap CIs, and the oracle
  broken down by `base_type` and group. Writes
  `evaluate_scorers.{md,csv,json}` plus the tau-curve and breakdown CSVs.
  `--lookbacks`, `--splits`, `--tau-grid`, `--bootstrap` and `--seed` narrow
  the run.
- **Build the real 1% sample (on Kaggle):**
  ```
  cd backend
  python scripts/kaggle/kaggle_runner.py make-sample
  ```
  Or `python scripts/make_sample.py --config configs/kaggle.yaml --out /kaggle/working`
  — both call the same `build_sample`. Download the two files into
  `backend/data/sample/`; after that local development uses the real
  sample via `--config configs/local.yaml`, and the synthetic generator is
  for unit tests only.
- **Generate synthetic development data and run the sanity scorers:**
  ```
  cd backend
  python scripts/make_demo_data.py --out data/demo --seed 7
  python scripts/run_sanity_scorers.py --config data/demo/demo.yaml --lookback 24
  ```
  The generated data is **synthetic and not the research sample** — it
  exists so the evaluation code can be exercised locally at a size where
  recall and lead time mean something. It is gitignored. The scorer run
  prints per-split metrics for all three test subsets, bootstrap CIs, a
  breakdown by topology group, and a pass/fail verdict comparing oracle
  against random.
- **Start the backend server:**
  ```
  cd backend
  python -m uvicorn app.main:app --reload --port 8000
  ```
  Serves `http://localhost:8000/api/...`, with interactive docs at
  `http://localhost:8000/docs`. Reads `backend/artifacts` unless
  `FRAUDCAMP_ARTIFACTS_DIR` says otherwise. It starts fine with no
  artifacts at all — `/api/health` reports what is missing and the data
  endpoints return 503 telling you to run the export.
- **Start the frontend dev server:**
  ```
  cd frontend
  npm run dev
  ```
  Serves `http://localhost:5173`, which the API already allows via CORS. Run
  the backend at the same time; without it the app still loads and every page
  shows an honest "API unreachable" or "not exported yet" state.
- **Build the frontend:** `npm run build` (type-checks with `tsc -b`, then
  bundles). **Lint:** `npm run lint` (oxlint).
- **Run backend tests:**
  ```
  cd backend
  python -m pytest tests/
  ```
  Full-data regression tests (`@pytest.mark.fulldata`) run automatically
  once `backend/data/sample/` or the Kaggle paths in
  `backend/configs/kaggle.yaml` exist; otherwise they skip.
- **Run frontend tests:**
  ```
  cd frontend
  npm run test
  ```

### Installing artifacts from Kaggle

After a Kaggle `pipeline` run, download `/kaggle/working/artifacts.zip` and
unpack it so that these sit **directly** in `backend/artifacts/`:

```
backend/artifacts/
  summary.json
  campaigns.parquet
  campaign_transactions.parquet
  institutions.json
  visibility/seed_0.parquet ... seed_4.parquet
  results/          (empty until Phase 5)
  .gitkeep          (tracked - leave it alone)
```

On Windows, **do not** use Explorer's "Extract All" directly onto
`backend/artifacts/`: it creates a folder named after the archive, giving
`backend/artifacts/artifacts/summary.json`. That is one level too deep, and
the API then reports every artifact as missing. Either extract elsewhere and
move the *contents* in, or use PowerShell, which lets you control the
destination exactly:

```powershell
cd "C:\Users\ASUS\Downloads\FRAUD CAMPAIGN DETECTION PROJECT"
Expand-Archive -Path "$HOME\Downloads\artifacts.zip" -DestinationPath "backend\artifacts" -Force
# if it still landed nested, flatten it:
if (Test-Path "backend\artifacts\artifacts\summary.json") {
  Move-Item "backend\artifacts\artifacts\*" "backend\artifacts" -Force
  Remove-Item "backend\artifacts\artifacts"
}
Get-ChildItem backend\artifacts        # summary.json must be listed here
```

**Restart the backend afterwards.** Artifacts are read once at startup and
cached, so a running server will keep reporting them missing no matter what
is on disk. `--reload` only restarts on *code* changes, not data.

If `/api/health` still shows `core_artifacts_available: false`, read its
`hint` field: the server detects the nesting mistake and names the offending
directory, both in that response and in a warning logged at startup.

The artifacts stay gitignored — they are derived data and must never be
committed.

## Backend: modules

- **`constants.py`** — every frozen definition (cutoff, split boundaries,
  hub types, K, visibility targets/seeds/bins, planned evaluation
  constants) as plain Python values. Nothing else in the backend hardcodes
  these.
- **`load_data.py`** — `load_config`/`resolve_path` (YAML config
  handling), `load_transactions` (reads the CSV with correct dtypes,
  parses timestamps, derives `src`/`dst` global account IDs),
  `apply_cutoff` (pre-cutoff subset), `account_bank` (extracts the bank
  int from a global account ID).
- **`campaigns.py`** — `parse_patterns` (BEGIN/END block parsing),
  `build_pattern_df` (flattens attempts into one row per pattern
  transaction, with `campaign_id`, `base_type`), `match_patterns` (joins
  pattern rows onto the full transaction table, asserts all match, tags
  `campaign_id` onto every transaction, -1 for non-campaign rows),
  `build_campaign_table` (aggregates to one row per campaign), `add_eval_fields`
  (`crosses_cutoff`, `deadline`, `eval_ok`), `campaign_accounts`/`campaign_edges`
  (per-campaign account set / transaction edge list, used by visibility).
- **`splits.py`** — `assign_split` (train/val/test/stress by start date,
  `None` for campaigns starting on/after `CUTOFF`), `assign_group`
  (hub/fragmentable by `base_type`), `reassignable_mask` (the topology
  test that decides which campaigns get reassigned: `eval_ok` + non-hub
  `base_type`), `mark_unfragmentable` (relabels fragmentable campaigns
  whose floor visibility can't drop below 0.9), `control_group_mask`
  (hub + unfragmentable), `mark_shared_accounts` (flags test campaigns
  sharing an account with any train campaign).
- **`institutions.py`** — `bank_volume` (pre-cutoff sender+receiver
  counts), `assign_institutions` (deterministic greedy volume balancing
  across `K` institutions), `natural_institution` (bank-based institution
  of an account).
- **`visibility.py`** — **determinism:** every account collection is sorted
  before anything random touches it, because sets iterate in a
  hash-dependent order that varies per process and would otherwise make the
  seeded search irreproducible (see Decision log, 2026-10-08).
  `visibility_of` (max-over-institutions seen
  fraction), `assign_campaign`/`_local_search_free` (per-campaign
  institution assignment toward a target, respecting already-locked
  shared accounts), `build_global_assignment` (processes eval_ok
  fragmentable campaigns in start-time order, resolving shared-account
  conflicts in favor of the earliest campaign), `institution_of` (final
  institution of an account, falling back to natural), `recompute_achieved`
  (final achieved visibility per campaign from the global assignment),
  `bin_of` (visibility -> `"100"/"75"/"50"/"low"`), `visible_transactions`
  (the required "subset an institution can see" function).
- **`pipeline.py`** — `build(config_path)` runs every deterministic stage
  (load → parse → match → campaign table → splits → groups → institutions
  → floor run → unfragmentable relabel → shared-account flags) and returns
  a `PipelineResult` holding the transaction table, pattern table, campaign
  table, bank→institution map, loads, and per-campaign edges/accounts.
  `ordered_reassignable` returns the reassignable campaigns in start-time
  order (which is what makes conflict resolution favour the earliest
  campaign); `run_visibility(seed, target)` builds one global assignment and
  recomputes achieved visibility for every campaign. Shared by
  `scripts/run_pipeline.py` and `export.py` so the two cannot drift apart.
- **`export.py`** — `build_summary`, `build_campaign_transactions`,
  `build_institutions`, `build_visibility_table(seed)`, and `export_all`,
  which writes every file described in [Artifacts](#artifacts) and creates
  an empty `results/`.
- **`windows.py`** — the only place allowed to slice a window, so the
  causality rule lives in one place: `detection_times` (the aligned grid),
  `split_of_detection_time`, `detection_times_for_split`, `window_slice`
  (`t - L <= ts < t`, upper bound strict), `build_window`/`iter_windows`
  (which skip empty windows), and `window_summary` for checking a
  configuration's size and label balance before spending GPU time on it.
  It also holds the three evaluation rules' time logic: `has_full_lookback`
  and `training_detection_times` /
  `usable_training_times_per_lookback` (rule 3), `evaluation_horizon` and
  `evaluation_detection_times` (rule 2, with `"stress"` deliberately
  raising), and `hit_kind` (rule 1, the hit vs other-split-hit decision).
- **`features.py`** — `fit_feature_spec(train_windows)` learns the
  category vocabularies and the scaler and **refuses** anything that is not
  a training window; `FeatureSpec.save/load` persists them;
  `build_window_features(window, spec)` returns a `WindowFeatures` holding
  node features, edge features, `edge_index`, labels, campaign ids and
  timestamps, with `xgboost_matrix()` flattening to edge + both endpoints'
  features for the non-graph baseline and `class_weight()` giving the
  imbalance weight.
- **`evaluation/`** — the fixed, non-learned half of Phase 3, identical for
  every model and condition so that differences in results come from the
  scores alone.
  - `clusters.py` — `extract_clusters` (connected components of the edges
    scoring >= tau, components of >= 3 accounts, accounts enumerated sorted
    so the output does not depend on hash order), `cluster_hits_campaign`
    (the frozen >= 50% / >= 3 rule), `find_hit_campaigns` (indexed by
    account, so a cluster is only compared against campaigns it touches),
    `is_ambiguous`, and `classify_cluster` returning one of
    `hit` / `other_split_hit` / `ambiguous` / `false_alarm`.
  - `evaluate.py` — `evaluate_split` walks a split's horizon, scores each
    window, classifies every cluster, and records each campaign's first
    detection *if* it beats the deadline; `CampaignOutcome` carries the
    three lead-time forms; `summarize` gives recall, precision, false
    alarms per day and medians per test subset, with `bootstrap_ci`
    (seeded percentile bootstrap over campaigns) and `summarize_by` for
    group and base_type breakdowns; `subset_filter` resolves a subset name,
    including `oracle_reachable`, which it refuses to guess the denominator
    for; `recall_relative_to_ceiling` divides a recall by the oracle's on the
    same subset; `select_tau` chooses the threshold on
    **validation only** by campaign-level F1 and returns the full
    precision/recall-vs-tau curve, with `campaign_f1` as the cheap
    statistic the sweep uses instead of a bootstrap at every candidate;
    `best_tau` applies the deterministic tie-break, and
    `select_tau_for_budget` / `recall_vs_false_alarms` provide the
    alert-budget view. `paired_lead_time_comparison` is the **only**
    sanctioned way to compare two configurations' earliness — per-campaign
    differences over the campaigns both detect, with coverage reported
    separately — and `describe_paired_comparison` renders it in words,
    declining to state a direction when the CI includes zero. See
    §[the paired method](#comparing-two-configurations-the-paired-method).
  - `diagnose.py` — `diagnose_misses` classifies every missed campaign into
    `purity_failure` / `late_only` / `structurally_disconnected` / `slow` /
    `unexplained`, in that precedence, with the evidence behind the verdict:
    the largest fragment reached inside a window, the largest reached with
    every own edge pooled and no window at all (which is what separates the
    last two), transaction gaps against the lookback, best cluster purity
    and what the campaign merged with. `diagnose_false_alarms` characterises
    the oracle's false-alarm clusters and names which half of the matching
    rule each one fails, purity or coverage, against its own threshold.
    Diagnosis only: nothing here scores, and nothing changes a rule.
  - `scorers.py` — the oracle (score = true label) and random scorers.
  The scorer is called as `scorer(window_transactions, t)` and only ever
  receives the window, so a scorer cannot reach past `t` even if it tries.
- **`models/xgb_baseline.py`** — the non-graph baseline. `XGBParams`
  holds the frozen hyperparameters; `build_training_data` pools every
  rule-3-valid training window into one matrix and carries each row's
  per-window class weight; `train_baseline` fits one booster per lookback;
  `TrainedBaseline.scorer()` returns a `scorer(window, t)` that
  `evaluate_split` accepts unchanged, so a model is scored by exactly the
  code that scored the oracle. `save`/`load` round-trip the booster, the
  feature spec and the parameters together, and scoring refuses a spec that
  does not match the model rather than producing numbers that look fine.
  `feature_importance` is report-only — selecting features from it would
  make the baseline's feature set differ from the GNN's, which is what the
  comparison holds fixed.
- **`federation/`** — Phase 4. `views.py` (stage 4.0) turns a `(seed,
  target)` visibility assignment into **per-institution views**:
  `visible_mask` is the one implementation of the frozen "sees" rule (sender
  **or** receiver), `institution_view` wraps a view as a
  `windows.Window` so the Phase 3 featuriser runs on it unchanged,
  `all_views` builds all `K` from a single endpoint pass,
  `boundary_accounts` names the accounts visible to more than one
  institution (what stage 4.3's exchange condition reasons about),
  `campaign_visibility_from_views` recomputes achieved visibility from the
  views as an independent check on the pipeline's own figure, and
  `view_load_report` reports each institution's pooled training rows and
  positives so a starved institution is a finding before training. Nothing
  in `views.py` trains or scores.

  `isolated.py` (stage 4.1) is the **`isolated` condition**.
  `masked_scorer` is where an institution's visibility is applied and the
  only place it is: it hands an inner scorer the visible rows and scatters
  the result back into the global window's own row order, so the evaluation
  keeps the **global** transactions, campaigns, deadlines and horizons and no
  line of the Phase 3 evaluation code changes. `oracle_view_scorer` is the
  oracle through the same mask, which is how each institution gets **its
  own** ceiling. `train_isolated` trains on the institution's view — the one
  place the view *is* passed as the frame, because training labels are
  per-row and no cluster or ambiguity logic is involved — and returns an
  `untrained` record rather than raising when there are no positives.
  `union_outcomes` combines per-institution outcomes (any detects it,
  earliest wins, ties to the lowest institution), `union_cluster_accounting`
  counts the union's clusters deduplicated by account set,
  `pooled_mean` returns the per-institution mean twice with its
  denominators, `operating_points` defines the seven points, and `run_cell`
  runs one `(target, seed, lookback)` so the local script and the Kaggle
  runner report from one code path.
- **`regression.py`** — the single source for the full-data expectations:
  `METRICS` (each with its expected value and the notebook cell it came
  from), `compute_actuals(result)`, `compare`, `format_table` /
  `format_markdown`, and `is_full_dataset`. Both `tests/test_fulldata.py`
  and the Kaggle runner read it, so the test suite and the PASS/FAIL report
  cannot disagree about what is expected.

### Scripts

- **`scripts/run_pipeline.py`** — local/Kaggle pipeline run, prints every
  regression number, `--export` writes artifacts.
- **`scripts/make_sample.py`** — run on Kaggle to build the 1% sample.
- **`scripts/kaggle/kaggle_runner.py`** — the Kaggle entry point.
  `verify` runs the full-data tests and prints the expected-vs-actual table;
  `pipeline` builds, exports and zips the artifacts; `features-smoke` builds
  detection windows at L = 24/48/72 on the full data and reports per-window
  size, timing, memory and tensor footprint plus a full-batch GNN memory
  estimate; `evaluate-scorers` runs the oracle and random scorers on the
  full data at both candidate lookbacks over all three horizons, with tau
  chosen on validation, bootstrap CIs and recall at fixed false-alarm
  budgets; `oracle-diagnose` explains every campaign the oracle misses and
  characterises its false alarms; `make-sample` builds the 1%
  development sample; `train` and `evaluate`
  exist but exit with a message until stages 3.3 and 3.4. Every step reports
  elapsed time and memory (a true peak via `resource` on Linux; on Windows it
  falls back to current RSS and says so rather than mislabelling it).
- **`scripts/kaggle/run_on_kaggle.md`** — the notebook cells, what to send
  back, expected runtimes, and the failure modes worth recognising.

### API server (`backend/app/`)

- **`main.py`** — `create_app(artifacts_dir=None)` builds the FastAPI app,
  adds CORS for the Vite dev server, loads the artifacts into
  `app.state.store`, and mounts every router under `/api`. The
  module-level `app` is what uvicorn serves; tests call `create_app()`
  with a temporary directory instead of touching environment variables.
- **`services/artifacts.py`** — `ArtifactStore` loads and caches every
  artifact at startup, tolerating missing files so `/health` can report
  them; `require_*` accessors raise 503 (not exported yet) or 404
  (campaign not found) with actionable messages. `to_records` converts
  DataFrame rows to JSON-safe dicts (numpy scalars unwrapped, `NaN`/`NaT`
  → `null`). When core artifacts are missing it logs a warning naming each
  one and the directory searched, and `find_nested_artifacts` detects the
  common case of artifacts extracted one directory too deep
  (`artifacts/artifacts/summary.json`) and says so explicitly — in the log
  and in `/api/health`'s `nested_artifacts_dir` and `hint` fields.
- **`services/methodology.py`** — builds the `/methodology` document,
  reading every value from `fraudcamp.constants` so the published
  definitions cannot disagree with the pipeline.
- **`schemas.py`** — pydantic response models mirroring the artifact
  schemas, so an artifact change that is not reflected here fails the API
  tests.
- **`routers/`** — one module per endpoint group (`meta`, `campaigns`,
  `visibility`, `results`).

## Artifacts

The full transaction data is too large to serve directly, so `export.py`
writes these small derived files to the config's `artifacts_dir`
(`backend/artifacts/` by default). Write them with
`python scripts/run_pipeline.py --config <cfg> --export`.

**`summary.json`** — everything the Dataset page needs without touching
the raw data:

| Field | Contents |
|---|---|
| `generated_at` | ISO timestamp of the export |
| `source` | `trans_csv`, `patterns_txt` paths the run used |
| `rows` | `total`, `pre_cutoff`, `post_cutoff` |
| `date_range` | `min`, `max` transaction timestamps |
| `banks` | `total` distinct banks |
| `laundering` | `rows`, `in_campaign`, `unassigned` |
| `campaigns` | `total`, `eval_ok`, `pattern_transactions`, `reassignable` |
| `by_base_type` | campaign count per `base_type` |
| `by_split` | per split: `campaigns`, `eval_ok`, `crosses_cutoff`, `shares_train_account` |
| `by_group`, `by_group_eval_ok` | campaign count per `group` (hub / fragmentable / unfragmentable) |
| `cutoff` | `timestamp`, `rows_excluded`, `laundering_rows_excluded`, `laundering_share_excluded`, `laundering_share_overall` |
| `transactions_per_day` | date → row count (includes the excluded tail, so the Dataset chart can shade it) |
| `campaigns_starting_per_day` | date → campaign count |
| `institutions` | `k` |
| `visibility` | `targets`, `seeds`, `bins` |

**`campaigns.parquet`** — one row per campaign (370 on full data):
`campaign_id`, `base_type`, `n_txn`, `n_accounts`, `n_banks`, `start`,
`end`, `duration_h`, `crosses_cutoff`, `deadline`, `eval_ok`, `split`,
`group`, `shares_train_account`, `reassignable`.

**`campaign_transactions.parquet`** — only transactions belonging to a
campaign (~3.2k rows on full data): `campaign_id`, `txn_index`,
`timestamp`, `from_bank`, `to_bank`, `src`, `dst`, `amount_paid`,
`payment_currency`, `amount_received`, `receiving_currency`,
`payment_format`, `is_laundering`, `src_natural_institution`,
`dst_natural_institution`. Sorted by `campaign_id`, then `timestamp`.
`timestamp` is a real datetime (not the raw `2022/09/01 00:10` string) so
the frontend can sort and replay on it. `txn_index` is the 0-based
position within the campaign's time-ordered transactions — a stable
transaction id assigned by the pipeline, which the API's visibility
endpoint uses to say which transactions an institution can see. The two
`*_natural_institution` columns are what lets the API answer "each
account's natural institution" without shipping the whole
bank→institution map.

**`institutions.json`** — `k`, `total_banks`, a `note` describing the
assignment rule, and `institutions`: one entry per institution with
`institution` (0..K-1), `load` (pre-cutoff transaction endpoints), and
`n_banks`.

**`visibility/seed_{s}.parquet`** — one file per experiment seed
(`0..4`), in long format: one row per `(target, campaign_id, account)`
with columns `target`, `campaign_id`, `account`, `institution`,
`account_reassigned`, `achieved_visibility`, `bin`,
`campaign_reassigned`. `achieved_visibility` and `bin` are constant
within a `(target, campaign_id)` group and are recomputed from the final
global assignment. **Every** campaign appears, not just the reassignable
ones — hub and non-`eval_ok` campaigns carry their natural-assignment
visibility and `campaign_reassigned = False`. Long format (rather than a
nested per-campaign structure) keeps the file flat enough to filter
directly in pandas and to serve per-campaign slices cheaply. The
seed-42 floor run is not exported; it exists only to decide the
`"unfragmentable"` relabel, which is already baked into
`campaigns.parquet`'s `group`.

**`results/`** — created, but **deliberately empty** (only a `.gitkeep`).
Later phases write it; see [Results schema](#results-schema). The API
404s and the frontend shows an empty state until then.

## API reference

All endpoints are `GET` under the `/api` prefix. Interactive docs at
`/docs`, raw schema at `/openapi.json`. CORS allows
`http://localhost:5173` and `http://127.0.0.1:5173` (the Vite dev server).
The server reads the artifacts **once at startup** and caches them; it
never touches the raw dataset.

Status codes used throughout:

| Code | Meaning |
|---|---|
| 404 | the campaign does not exist, or results have not been generated yet |
| 422 | an invalid parameter (unknown sort key, seed, or visibility target) |
| 503 | the artifact needed has not been exported yet, with the command to run |

**`/api/health`** — `status`, `artifacts_dir`, `artifacts` (a map of every
expected artifact path to whether it exists), `core_artifacts_available`,
`results_available`, plus `missing_artifacts`, `nested_artifacts_dir` and a
human-readable `hint` when something is wrong (see
[Installing artifacts from Kaggle](#installing-artifacts-from-kaggle)).
Works even with no artifacts at all, so the frontend's API-status indicator
can always render.

**`/api/summary`** — `summary.json` verbatim, validated against the
response model (see [Artifacts](#artifacts) for every field).

**`/api/campaigns`** — the campaign table, filtered, sorted and paginated.

| Parameter | Values | Default |
|---|---|---|
| `page` | >= 1 | 1 |
| `page_size` | 1–500 | 25 |
| `base_type`, `split`, `group` | exact match | — |
| `eval_ok`, `crosses_cutoff`, `shares_train_account` | bool | — |
| `search` | campaign id, or part of an account id | — |
| `sort_by` | `campaign_id`, `base_type`, `n_txn`, `n_accounts`, `n_banks`, `start`, `end`, `duration_h` | `campaign_id` |
| `sort_dir` | `asc`, `desc` | `asc` |

Returns `items`, `total`, `page`, `page_size`, `pages`. **Search rule:** a
digits-only query means the campaign id, plus any account whose account
part (after the `bank_` prefix) starts with those digits — so `3` finds
campaign 3 rather than every account containing a 3, while `800737690`
still finds that account. Any other query is a case-insensitive substring
match on the full account id, e.g. `1_A`.

**`/api/campaigns/{id}`** — `campaign` (the full row), `transactions`
(time-ordered, each with its `txn_index`), and `accounts` (each account
with its `natural_institution`). 404 if the campaign does not exist.

**`/api/campaigns/{id}/visibility?seed=&target=`** — `achieved_visibility`,
`bin`, `campaign_reassigned`, `n_transactions`, `accounts` (each with
`institution` and whether it was `reassigned`), and `institutions` — one
entry per institution with `visible_txn_indices` and `n_visible`, which is
exactly what the frontend's bank's-eye view ("this bank sees X of Y")
needs. Transactions are referenced by `txn_index`, the 0-based position in
the campaign's time-ordered list from `/api/campaigns/{id}`; the index is
assigned by the pipeline and stored in the artifact, not invented by the
API. `seed` must be one of the five experiment seeds and `target` one of
the four visibility targets, else 422.

**`/api/visibility/distribution?seed=`** — for each target: every
campaign's `achieved_visibility`, `bin`, `group` and `reassigned` flag,
plus `bin_counts` and `bin_counts_by_group`. Also returns
`unfragmentable_campaigns`. All campaigns are included, not only the
reassignable ones, so hub and control-group distributions can be drawn.

**`/api/methodology`** — the frozen definitions as structured `entries`,
each with `key`, `title`, `definition`, `value` and `reason`. Every
*value* is read from `fraudcamp.constants` — the same module the pipeline
uses — so the published methodology cannot drift from what the code did.
Served even when no artifacts exist.

**`/api/results/summary`** and **`/api/results/detections?campaign_id=`** —
both 404 with a clear message until Phase 5 writes
`artifacts/results/`. Shape once populated is in
[Results schema](#results-schema).

## Frontend

**Stack:** Vite + React + TypeScript, React Router, TanStack Query,
Tailwind CSS v4, Recharts, Cytoscape.js, lucide-react, Vitest + React
Testing Library.

> Deviation from the original spec: the spec said React 18, but the current
> Vite template ships **React 19** (with Vite 8, Tailwind 4, TypeScript 6).
> Nothing here depends on 18-only behaviour, so the newer versions were kept
> rather than pinned backwards. This is a tooling choice, not a research
> definition.

**Why Cytoscape.js over react-force-graph-2d:** the research question is
*about topology*, so a viewer must see at a
glance that a FAN-OUT is a star and a CYCLE is a ring. Cytoscape's
deterministic layouts (`circle`, `concentric`, `breadthfirst`) make that
structure legible and render the same campaign identically every time, which
also matters for figures in the paper. Force-directed layouts look livelier
but can settle a ten-node cycle into an ambiguous blob and vary run to run.
In practice the layout is chosen per `base_type`: `circle` for CYCLE,
`concentric` for the hub shapes (which puts the hub account in the middle by
degree), and `breadthfirst` for chains, bipartite and random patterns.

> **Half of that motivation is not delivered, measured 2026-11-14.** The
> FAN-OUT claim holds: `concentric` puts the hub account in the centre and the
> campaign reads as an unmistakable star. **The CYCLE claim does not.**
> `circle` places nodes in collection order, which here is
> `[...accounts].sort()` — alphabetical by account id, not cycle traversal —
> so consecutive accounts in the cycle land on opposite sides of the circle and
> most edges cross the interior as chords instead of following the rim. The
> determinism claim is unaffected, and nothing is broken; the graph is simply
> not self-describing for this one topology. See
> §[Known limitations](#known-limitations-and-open-questions) for the counts.

### Pages

| Page | Route | What it shows | Status |
|---|---|---|---|
| Home ("Start here") | `/` | The research question in plain language, the four-step visual story (campaign → split across banks → each bank sees a fragment → collaboration recovers it), key stat cards, a live pipeline-status panel, guided-tour buttons | done |
| Dataset | `/dataset` | Stat cards; transactions per day with the cutoff line and shaded excluded tail; campaigns per pattern type (hub types marked); campaign duration distribution; split breakdown with evaluable/censored/shared-account counts; treatment vs control group split; a "why we excluded data after Sept 10" card using the real numbers | done |
| Campaign Explorer | `/campaigns` | Searchable, filterable, sortable, paginated table with removable filter chips and a live result count; rows open a campaign | done |
| Campaign Detail | `/campaigns/:id` | The demo centrepiece. Interactive Cytoscape graph (accounts as nodes labelled by institution, transactions as directed arrows, hover for amount/time, drag/zoom/pan, click a node for its details); time replay with play/pause and deadline + cutoff marks; visibility controls (seed, and either the natural bank split or a target) with the achieved-visibility and bin badge; bank's-eye-view tabs that fade everything the selected bank cannot see and report "this bank sees X of Y"; a detection panel that is an explicit empty state until Phase 5; and a metadata sidebar explaining every flag in plain language | done |
| Visibility Lab | `/visibility` | A sandbox where accounts of a fan-out and a cycle are dragged (or clicked) between three bank zones while visibility updates live, then the real achieved-visibility histogram per target, bins per group, and the unfragmentable count | done |
| Results | `/results` | Empty state plus labelled outlines of all five planned figures, each with a "how to read this" note and no numbers at all; the condition legend with its fixed colours and dash patterns | done |
| Methodology | `/methodology` | `/api/methodology` rendered as cards: each definition, the reason for it, and its machine-readable values | done |

### Where numbers come from

Every number in the app is served by the API, with exactly one exception:
the **Visibility Lab toy**, where the user invents a scenario the backend has
never seen. That number is computed by `src/lib/visibility.ts`, a direct
mirror of `visibility_of` in `backend/src/fraudcamp/visibility.py`, and it is
tested against the same cases as the Python function. Nothing else on any
page is computed locally, and no page ever displays a placeholder value as
though it were a result.

### Theme and colour system

`src/theme.ts` is the single source for every series colour: one fixed colour
per institution (8), per condition (4), and per visibility bin (4). Chart
chrome and ink live as CSS custom properties in `src/index.css`, redefined
under `.dark` so the light/dark swap happens in one place.

Colours come from a validated categorical palette, with both light and dark
steps checked against the chart surface for lightness band, chroma floor,
colour-vision-deficiency separation, normal-vision separation, and contrast.

**Colour is never the only cue**, as the spec requires — and in one case it
*cannot* be. Eight categorical hues cannot be told apart reliably under
colour-vision deficiency when any two of them may appear side by side (the
validator fails that case for any ordering of eight hues). So institution
identity is carried **label-first**: every institution mark also shows its
label (`I0`–`I7`) in badges, legends and graph nodes, with colour as the
secondary cue. Conditions likewise each carry a line dash pattern plus a
label, and visibility bins use an ordered single-hue ramp (which reverses
direction in dark mode so the "more visible" end always reads strongest
against its own background) plus their written range.

### Accessibility

Keyboard navigation throughout (including a skip-to-content link and
keyboard-activatable table rows), visible focus rings, `aria-sort` on
sortable columns, `aria-live` on the result count and API status, table
captions, `role="alert"` on errors, and `prefers-reduced-motion` honoured.
Layout is responsive from phone width up, with the sidebar collapsing to a
toggle below `lg`.

## Results schema

Later experiment phases (4–5) must write exactly this shape into
`backend/artifacts/results/`. Nothing in this section is generated yet —
the Results Dashboard frontend page renders an empty state until these
files exist.

**`results/runs.parquet`** — one row per `(condition, seed, visibility_bin,
group, test_subset)`:

| Column | Meaning |
|---|---|
| `condition` | one of `isolated`, `fedavg_only`, `fedavg_embedding`, `centralized` |
| `seed` | experiment seed, one of `0,1,2,3,4` |
| `visibility_bin` | one of `"100", "75", "50", "low"` |
| `group` | `hub` or `fragmentable` (control group = `hub` + `unfragmentable`) |
| `test_subset` | `"all"`, `"fully_observed"`, `"no_shared_accounts"`, or `"oracle_reachable"` |
| `campaign_recall` | fraction of eligible campaigns detected by deadline |
| `campaign_precision` | fraction of detections that are true positives |
| `false_alarms` | count of false-positive detections |
| `median_lead_time_h` | median lead time in hours, among detected campaigns |
| `median_normalized_lead_time` | median lead time normalized by campaign duration |
| `median_frac_observed_at_detection` | median fraction of campaign transactions observed at detection |
| `oracle_recall` | the oracle's recall on the same split, lookback and subset: the ceiling |
| `recall_relative_to_ceiling` | `campaign_recall / oracle_recall`, the honest fraction achieved |

**`results/detections.parquet`** — one row per `(condition, seed, target,
campaign_id)`:

| Column | Meaning |
|---|---|
| `condition` | as above |
| `seed` | as above |
| `target` | visibility target used for this run (`1.0, 0.75, 0.5, 0.25`) |
| `campaign_id` | campaign identifier |
| `detected` | bool |
| `detection_time` | timestamp of detection, if any |
| `lead_time_h` | hours between detection and true completion |
| `normalized_lead_time` | `lead_time_h` normalized by campaign duration |
| `frac_observed` | fraction of the campaign's transactions observed at detection time |

## Testing

- **Backend unit tests** (`backend/tests/test_{load_data,campaigns,splits,institutions,visibility}.py`)
  on tiny hand-made fixtures: pattern parsing, base_type normalization,
  leading-zero banks, cutoff, eval_ok, splits, volume balancing,
  visibility (`A->B->C` chain = 1.0, fan-out = 1.0), conflict resolution,
  recomputation.
- **Backend leakage tests** (`test_features.py`, `test_windows.py`), written
  as properties rather than spot checks, because a leak is easier to prove
  by invariance than by inspection:
  - **No future.** Deleting every transaction from `t` onward must not
    change a single number in either matrix — and a sharper version sets the
    post-`t` rows to absurd values (1e12 amounts, an invented currency, all
    labels flipped to 1) and requires the features to be bit-identical.
  - **No labels.** Flipping every `Is Laundering` leaves the features
    identical while the targets differ.
  - **No identity.** Relabelling every account leaves the edge features
    identical and the set of node rows unchanged (only their order moves);
    renumbering banks while preserving which pairs differ leaves everything
    identical, since only the `!=` equality is permitted.
  - **No training data from Sept 5 onward**, checked over every training
    window; and no window of any lookback reaches at or past its `t`.
  - **Scaler provenance.** Fitting on a validation or test window raises;
    fitting is unchanged by whether later data exists in the dataframe at
    all; and the scaler's identity columns are exactly the non-continuous
    features.
  - Plus the grid's alignment, 6-hour spacing, inclusive split boundaries,
    partitioning, that longer lookbacks are supersets of shorter ones, and
    that unseen categories land in the `__other__` column without shifting
    the known ones.

  **Verified by deliberately introducing leaks** (2026-10-09), the same way
  the determinism suite was checked, because a leakage test that cannot fail
  is worthless:

  | Injected leak | Caught by |
  |---|---|
  | `Is Laundering` added to the edge features | `test_labels_do_not_leak_into_features` |
  | window upper bound `ts <= t` instead of `ts < t` | `test_window_excludes_its_own_detection_time`, and `test_a_transaction_stamped_exactly_at_t_is_treated_as_future` |
  | `From Bank` added as a numeric feature | `test_no_identity_feature_names`, `test_bank_identity_does_not_leak_beyond_the_equality_flag` |

  The off-by-one leak initially escaped the *feature-level* future test,
  because the fixture had no transaction stamped exactly on the detection
  time being used, so the boundary was never exercised where it matters.
  `test_a_transaction_stamped_exactly_at_t_is_treated_as_future` was added
  to close that, and confirmed to fail under the same injected leak.
- **Backend tau and budget tests** (`test_tau_and_budgets.py`, 22 tests):
  each level of the tie-break in isolation, that precision outranks the
  tau preference rather than being overridden by it, and that the result is
  **independent of grid order** under four different orderings — the whole
  reason the tie-break was fixed. Then the budget selection at each frozen
  budget, the inclusive boundary, an unreachable budget returning `None`
  rather than approximating, the curve sorted for plotting, and the
  motivating case written as a test: of two detectors the profligate one has
  the higher recall and is the one the budget rejects. Then the model grid:
  that it is frozen and reaches below the sanity grid's floor, that a tau at
  either end is reported as an edge, that **caching the scorer cannot change
  the curve** — the cached and uncached curves are compared row by row, and
  the cached run is shown to do exactly one scoring call per window instead
  of one per (window, tau) — that the cache raises if a window changes size
  under a key it has already answered, and that every curve row records what
  that tau cost.
- **Backend GNN tests** (`test_gnn.py`, 25 tests): causality — a
  transaction stamped exactly `t` is added to the data and the window graph
  does not grow — and that early stopping never sees a test detection time,
  which is the leakage trap of this sub-stage since edge-level AUPRC reads
  every label in a window. Then that no account, bank or campaign name
  reaches the inputs and no parameter is account-vocabulary sized; that
  `log1p` is applied to the four count columns and not to the already-logged
  amounts, with the mechanism tested on a power-law column where it cuts the
  maximum z-score five-fold but demonstrably does **not** bound it, which is
  why LayerNorm is kept; that the balanced loss is `ln 2` at initialisation
  for a 100-edge and a 100,000-edge window alike, that its per-sample weight
  ratio is exactly `n_neg / n_pos` (checked through the gradients), and that
  a window with no positives still has a finite loss; that average precision
  matches hand-computed values and is NaN without positives; that scores
  number `E` rather than the doubled `2E` and line up positionally; that the
  no-format variant drops exactly the `fmt_*` columns including
  `fmt___other__` and leaves labels, graph and node features untouched; that
  a saved model reloads to identical scores while a half-written one is not
  counted as finished; and that `max |logit|` stays under 100 through
  training, which is the fix working.
- **Backend federation-view tests** (`test_federation_views.py`, 31 tests,
  stage 4.0): the three properties the stage is defined by — a view holds
  **exactly** the rows that institution is an endpoint of (checked both ways:
  every row inside the view qualifies, and every row outside does not), the
  union of the views is the whole dataset, and a campaign's visibility
  recomputed **from the views** equals the achieved visibility the pipeline
  records, at every one of the 12 `(seed, target)` combinations. The last is a
  genuine cross-check, because the pipeline reaches that number through a
  campaign's pattern edge list and the test reaches it by counting rows of the
  tagged transaction table. Then: that a view is a pure row subset adding no
  column, so institution identity cannot become a feature; that a transaction
  is seen by exactly two institutions when its endpoints differ and one when
  they do not; that the fast `all_views` path and the per-institution path
  agree frame for frame; that Phase 4 agrees row for row with **Phase 2's own**
  `visibility.visible_transactions`, so there is only one notion of "sees" in
  the project; that the Phase 3 featuriser accepts a view unchanged and that a
  boundary account really does lose degree in a view, which is the deficiency
  stage 4.1's exchange condition exists to repair; that `all_views` returns
  all `K` entries even when some are empty; that views are reproducible from
  the seed and that different seeds give different views; and that the load
  report's per-institution rows sum back between 1× and 2× the central total.
  **Proven able to fail** by mutating the real source, not a stand-in:
  reducing `visible_mask` to sender-only breaks 4 tests, looping
  `range(k - 1)` in `all_views` breaks 2, and replacing the `max` over
  institutions with a `mean` breaks 13.
- **Backend isolated-condition tests** (`test_federation_isolated.py`, 40
  tests, stage 4.1): that a masked scorer scores the visible rows and
  **zeroes the rest**, that the scores land on the rows they describe (the
  values are row-identifying, so a misalignment shows), and that the clusters
  it produces are **identical** to the clusters extracted from the view alone
  — which is what "a cluster can only form inside the view" means
  operationally. That the ground truth stays **global**: on a fixture where a
  cluster touches unassigned laundering the institution cannot see, the
  classification is `ambiguous` under the global truth and `false_alarm`
  under the view's, so the rejected design is shown getting it wrong rather
  than argued about. That `oracle_view_scorer` is the oracle on the view and
  nothing more, and that on a four-account chain spread one account per
  institution the centralized oracle detects the campaign while only the two
  institutions holding a **middle** account can — four of eight have a
  ceiling of 0 on it, which is why a ceiling is never shared. That the union
  detects what any institution detects, at the **earliest** time, with ties
  broken to the lowest institution so a dict order cannot change a result,
  and that it **deduplicates** a cluster two institutions both raise. That
  the pooled mean carries both denominators, enters an untrained institution
  at 0, and excludes an undefined cell from both means while listing it. And
  on a whole cell: that `run_cell` emits every report, that the union never
  detects less than its best institution, that the pooled rows are a function
  of the per-institution rows above them, that the outcome rows carry every
  key a later paired comparison needs, and that two runs at one seed agree —
  including the union's credited institutions, which a non-deterministic
  tie-break would change. **Proven able to fail** against the same fixtures:
  a scorer that fills positions instead of rows leaves non-zero scores on
  invisible rows, a union that sums instead of deduplicating disagrees with
  the deduplicated count, and an untrained institution recorded as undefined
  lifts the pooled mean from 0.40 to 0.80. And the grid-edge policy, with tau
  selection **stubbed** so the tests are about the policy rather than about
  coaxing a fixture into choosing a particular threshold: that a tau at the
  top of the grid makes every operating point undefined, carrying its rejected
  tau and its reason and leaving the pooled means empty rather than 0; that a
  bottom-edge F1 tau is flagged while a bottom-edge **budget** tau is not,
  because a loose budget is not a missing optimum; that `tau_grid_edge`
  follows whatever grid is in force, so extending the grid moves the edge
  instead of leaving the check behind; and that a tau beyond the grid is filed
  under its own reason and never as an unaffordable budget.
- **Backend cross-model comparison tests** (`test_compare_models.py`, 22
  tests): the three that matter are the ones whose failure would be
  invisible. That a per-campaign record survives a CSV round trip with
  `detected` intact — the string `"False"` is truthy, so a careless parse
  reports every campaign detected and a recall of 1.0, and because pandas
  usually infers a real boolean column and would hide that, the guard is
  tested through a file read as text and through records whose flag is
  already a string. That a comparison run from two files is **identical** to
  the same comparison in process, median and CI bound for bound, so rule 3
  holds across sessions and machines rather than only inside one run. And
  that two detectors tagged into one match key are **refused** rather than
  silently mixed into one paired set, which is how an ablation arm or a
  variant would otherwise disappear into a mixture. Then the sign
  convention (`b − a`, so positive means the second detected earlier,
  checked by swapping the inputs), that unmatched seeds are not compared and
  are reported instead, that the two ablation arms are compared arm by arm,
  and that coverage is reported by campaign id each way. Then the edge
  metrics: average precision against hand-computed values, NaN rather than
  0.0 where a window holds no positive, a lift of about 1 for a random
  ranker at a 0.1% positive rate, that the pooled row is **recomputed**
  rather than averaged over windows, that a scorer returning the wrong
  number of scores raises, and that the GNN's early-stopping metric and this
  diagnostic are the *same function object*. Then cluster shape: that it
  counts only its own kind, that two runs matching on false-alarm count can
  differ entirely in size distribution — the case the measure exists for —
  and that an empty distribution reports no clusters rather than failing.
  Each of the three leading guards was confirmed to fail with the guard
  removed.
- **Backend baseline tests** (`test_xgb_baseline.py`, 20 tests): the
  things that would quietly invalidate a result rather than the model's
  accuracy, which is measured on the full data and not asserted in a unit
  test. That training uses only rule-3-valid training detection times (13
  at L = 24) and that a feature spec cannot be fitted outside training;
  that the class weight is per window and not pooled; that no account,
  bank, institution, campaign or label name reaches the feature matrix;
  that scores line up positionally with the window's rows, checked against
  a column whose relationship to the score is known rather than by
  comparing a reversed input to a reversed output, which would cancel; that
  an empty window scores to an empty array; that the same seed gives the
  same model; that save/load round-trips to identical scores; that a
  mismatched spec is refused instead of scored; that the model chooses no
  threshold of its own; and that it plugs into `evaluate_split` with no
  model-specific path. Then the ablation: that it drops every `fmt_*`
  column **including `fmt___other__`** and nothing else (currencies, the
  one-hot block's neighbours, survive); that it changes only the columns,
  with labels, weights, windows and per-window stats identical and the kept
  columns equal to a slice of the original matrix rather than recomputed;
  that the full variant is a no-op view and slicing twice is free, so five
  seeds share one matrix without copying it; that an ablated model trains,
  scores and round-trips carrying its variant; that an unknown variant is
  refused; and that the variants and seeds are frozen and the variant names
  are not the four federated conditions. **Proven able to fail:** dropping
  the rule-3 filter fails the training-times test, and permuting the
  scorer's output fails the alignment test.
- **Backend paired-comparison tests** (`test_paired_comparison.py`, 15
  tests): two are constructed so that a shortcut gives the wrong answer.
  The second pins rule 3: C is a reference both detectors are scored
  against, A ties with C on two campaigns and is 50h later on two more that
  only it reaches, B ties with C on the same two and misses the rest — so
  `A vs C` is −25h and `B vs C` is 0h, and subtracting them says B is 25h
  earlier, while the direct comparison is a tie with A simply covering more.
  The first is constructed so that the *unpaired* medians give
  the wrong answer — B detects everything A detects at exactly the same time,
  plus one campaign only detectable late, so medians make B look 45 hours
  worse while the paired difference is 0.0h with one extra detection. Then
  the sign convention in both directions, coverage counted separately and by
  id, campaigns neither side detects excluded but counted, campaigns missing
  from one evaluation ignored rather than treated as misses, no shared
  detections yielding `None` instead of a number, the normalized and
  `frac_observed` pairings, a missing normalized value not sinking the hours
  comparison, identical detectors giving a zero interval, the interval being
  a function of the seed alone, `describe_paired_comparison` refusing a
  direction when zero is inside the interval and stating one when it is not,
  and `as_dict` staying flat enough for a CSV row. One test runs the real
  oracle through `evaluate_split` at both lookbacks on a built world where a
  campaign's 30-hour gaps put it beyond any 24h window, checking it surfaces
  as coverage rather than as a lead-time change. **Proven able to fail:**
  making the comparison unpaired fails 4 of the 14.
- **Backend oracle-diagnosis tests** (`test_oracle_diagnose.py`, 14 tests):
  each of the four reasons on a hand-built case where the answer follows
  from the construction — 40-hour gaps for `temporal_sparsity`, a campaign
  swallowed by an unassigned-laundering chain for `purity_failure`, a
  deadline before the first usable window for `late_only` — plus that a
  longer lookback rescues a sparse campaign, that a *detected* campaign is
  never diagnosed as sparse, that a genuine miss is never `unexplained`, and
  that running a diagnosis leaves the evaluation's verdict untouched. Then
  the two sparsity reasons: three disjoint pairs that no lookback can ever
  connect are `structurally_disconnected`, a 40-hour-gap chain is `slow`,
  and one test isolates the single thing that separates them by holding the
  miss fixed and varying only the pooled connectivity. For false alarms,
  that a cluster can fail coverage while *passing* purity — the case the old
  summary line contradicted itself over — each failing condition named for
  the binding bound, and that the classification agrees with
  `cluster_hits_campaign` on five hand-built clusters rather than
  re-implementing the thresholds.
- **Backend evaluation tests** (`test_evaluation.py`, 37 tests): cluster
  extraction (tau threshold, transitive joining, the >= 3 account minimum,
  direction-independence, determinism); the matching rule at and around both
  of its thresholds; classification of all four outcome kinds, including
  that a cluster hitting both an evaluated-split and another-split campaign
  is credited only to the evaluated split; and metrics, with the bootstrap
  CI shown to be seeded and to bracket its statistic. The end-to-end cases
  are hand-built with timestamps chosen so the answer is checkable on
  paper — one campaign is asserted to be detected at Sept 6 12:00 with
  lead time exactly 32.0h, normalized 32/37, and `frac_observed` 0.75; the
  same campaign with no later transactions has its deadline pass first and
  is asserted **missed**; an unassigned-laundering cluster is asserted
  `ambiguous` with false alarms still zero; and a train campaign inside a
  test window is asserted to produce an other-split hit rather than a false
  alarm.
- **Backend evaluation-rule tests** (`test_evaluation_rules.py`, 44 tests)
  cover the three rules frozen on 2026-10-10: that a same-split hit is a
  hit and a different-split (or no-split) hit is an `other_split_hit`; that
  the validation horizon runs to Sept 8 with an exclusive start and
  inclusive end, reaches past the Sept 5–6 data split, and overlaps the test
  horizon by exactly 8 times; that the test horizon is unchanged; that the
  stress horizon runs `(Sept 8, CUTOFF]` over 12 times, abuts validation
  without overlapping it and sits strictly inside test; that training has no
  horizon and asking raises; that the lookback candidates are exactly
  {24, 48} with 72 excluded-but-measured and the two sets disjoint; and that
  every training detection time has its full lookback, with the counts
  13 / 9 / 5 pinned for L = 24 / 48 / 72, longer lookbacks giving strict
  subsets, evaluation times unaffected, and the *unfiltered* grid shown to
  contain short windows so the filter is demonstrably doing work.

  **Verified by deliberate violation** (2026-10-10), as with the leakage and
  determinism suites:

  | Injected violation | Tests that failed |
  |---|---|
  | `hit_kind` returns `"hit"` for every match | 7 (`test_a_hit_on_a_different_split_is_an_other_split_hit`, all cases) |
  | validation horizon capped at Sept 6 | 4, incl. the overlap and past-the-split tests |
  | full-lookback filter removed | 8, incl. the pinned 13/9/5 counts |
  | stress horizon starting Sept 6 instead of Sept 8 | 4, incl. the abuts-validation and inside-test tests |
  | L = 72 left in as a candidate | 3, incl. the minimum-training-windows guard |

- **Backend determinism tests** (`test_determinism.py`): the visibility build
  must be reproducible from its seed alone. Because one process has one hash
  seed, in-process tests cannot detect hash-order dependence, so these run
  the build in **subprocesses** with `PYTHONHASHSEED=1/2/3` and require the
  account order, the account-to-institution map, the achieved visibilities,
  the unfragmentable count and the whole serialised payload to be identical.
  Also covers two in-process runs at each of the four targets, and asserts
  the accounts invariant (sorted sequence, never a set) directly. Verified to
  *fail* when the old set-iteration is reintroduced — 4 of the 10 break,
  including the byte-identical check.
- **Backend pipeline and export tests** (`test_pipeline.py`,
  `test_export.py`) run the whole pipeline against the `mini_dataset`
  fixture — a synthetic 18-row, 4-campaign dataset built in `conftest.py`
  that deliberately covers a hub campaign, a fragmentable campaign, a
  too-small (non-`eval_ok`) campaign, a test campaign sharing an account
  with a train campaign, an unassigned laundering row, a post-cutoff row,
  and a bank that appears only after the cutoff. They assert every
  artifact's columns and contents, that `results/` stays empty, that two
  runs produce byte-identical tables, and that different seeds actually
  produce different assignments.
- **Backend API tests** (`test_api.py`) with FastAPI `TestClient` against a
  fixture artifact set produced by the **real** export, so the API is
  tested against artifacts of exactly the shape the pipeline writes.
  Covers health with and without artifacts, the 503-with-instructions path
  when nothing is exported, summary, campaign filtering/sorting/paging/
  search, campaign detail, per-campaign visibility (including that
  `visible_txn_indices` agree with the detail endpoint and that achieved
  visibility equals the best institution's share), the distribution
  endpoint, methodology values matching `constants`, the results 404s,
  OpenAPI/docs, and the CORS header.
- **Backend full-data regression tests** (`backend/tests/test_fulldata.py`,
  `@pytest.mark.fulldata`): one parametrized test per number in
  [Verified numbers](#verified-numbers), so a failure names the exact metric
  rather than one assertion failing for all of them. Expectations come from
  `fraudcamp.regression`. The suite builds the pipeline once and, when
  `FRAUDCAMP_REGRESSION_OUT` is set, writes the computed numbers as JSON so
  the Kaggle runner can print its table without a second full build. Skips
  automatically when the dataset is absent.
  > **Confirmed on 2026-10-08:** a Kaggle `verify` run reproduced all 23
  > numbers (pytest exit 0, pandas 2.3.3, 39.3s, 3,207 MB peak). The report
  > is kept at `docs/reference/verify_report.md`. They still skip on any
  > machine without the full dataset, which is every local machine.
- **Frontend tests** (Vitest + React Testing Library), in `frontend/src`:
  - `pages/campaignFilters.test.ts` — explorer filter logic: query building
    (including that a `false` boolean survives rather than being dropped as
    empty), chip labelling, clearing a filter back to `null` not `false`, and
    sort toggling.
  - `components/states.test.tsx` — empty states, and that a 503 is presented
    as "not exported yet" with the API's own message rather than as a crash.
  - `pages/pages.render.test.tsx` — render smoke tests mounting the real
    Home, Dataset and Campaign Explorer pages against a stubbed API, checking
    the research question, real statistics, pipeline status, cutoff
    explanation, split/group breakdowns, the result count, and that every
    table row is a keyboard-reachable link.
  - `lib/visibility.test.ts` — the toy visibility function against the same
    cases as the Python original: chain `A->B->C` = 1.0, fan-out = 1.0, the
    no-overlap case = 1/3, the bin thresholds, and per-institution visible
    subsets. It also pins two results that are easy to get wrong: a
    four-account cycle **cannot** go below 0.75 across two banks (the
    boundary transactions stay visible to both), and reaches 0.5 only once
    there are four banks — which matches what the backend pipeline produces
    for the same cycle in the test fixture.
  - `pages/stage5.render.test.tsx` — Campaign Detail (loads, defaults to the
    natural split with the seed selector disabled, shows achieved visibility
    and bin after choosing a target, reports "this bank sees 2 of 3" when a
    bank is selected, explains each metadata flag, and shows a detection
    empty state), Visibility Lab (live recomputation as accounts move, the
    hub shape staying at 100%, and the real distributions rendering), Results
    (the not-available-yet state, five figure outlines each with a
    how-to-read note, the four named conditions, and that it does *not* claim
    results are missing when the API has some), and Methodology (each
    definition with its reason and values, plus the not-a-production-system
    note). Cytoscape needs a real canvas, so the graph component itself is
    stubbed in these tests.

**Current pass status (2026-10-17):**
- Backend: `python -m pytest tests/` from `backend/` → 331 passed, 25
  skipped, 0 failed. The skips are the full-data suite, which needs the
  dataset that only exists on Kaggle; all 24 of its numbers passed there on
  2026-10-08/09.
- Frontend: `npm run test` from `frontend/` → 48 passed, 0 failed.
  `npm run build` (tsc + bundle) and `npm run lint` both clean.

**Browser verification (2026-11-14).** The pages have now been opened in a
real browser — headless Chrome driven over the DevTools protocol, which needs
no new dependency — and the **Cytoscape graph has been seen rendering**. What
was checked, and what it found:

- All six pages render with **no exceptions and no failed requests**. First
  render: Campaign Explorer with no filters **153 ms**, the heaviest campaign
  (239 — 45 accounts, 30 transactions) **198 ms**, Visibility Lab **272 ms**.
- The graph was rendered for eight hand-picked cases: a cycle, a fan-out, the
  largest campaign, the smallest (1 transaction, 2 accounts), a post-cutoff
  campaign whose `split` is `null`, and campaigns carrying a **self-transfer**
  (`src == dst`) and a **duplicate account pair**. All drew correctly;
  self-loops render as loop arcs and reciprocal edges separate under
  `curve-style: bezier`.
- Separately, **all 370 campaigns** were built headlessly through the same
  element construction and per-`base_type` layout the component uses: no
  throw, no non-finite position, no overlapping nodes, correct node and edge
  counts, mean layout 1.16 ms and worst 18.6 ms. The three layouts are
  discrete and synchronous with `animate: false`, so there is no force
  simulation that could fail to settle.
- Error and empty states were exercised against a live backend: a 404 campaign
  gives a clean alert in ~1.1 s, a **fully unreachable backend** puts "Could
  not reach the API at http://localhost:8000" on every page within ~150 ms with
  the status chip reading "API unreachable", the results 404 gives the empty
  state, and a slow API shows "Loading campaign...".
- **Two defects found**, both recorded under
  §[Known limitations](#known-limitations-and-open-questions): a CYCLE does not
  read as a ring, and a non-numeric id in the URL (`/campaigns/abc`) leaves the
  main panel empty with no message, because `Number('abc')` is `NaN`, the query
  is disabled, and `CampaignDetailPage` returns `null`.

Still not verified: human visual judgement on a real display (projector colour,
and edge contrast in light mode), and any browser other than Chrome. The jsdom
tests continue to stub the graph because jsdom has no canvas, so this evidence
is recorded here rather than in the suite.

## Known limitations and open questions

- **Some campaigns are undetectable by construction, and they are not
  randomly distributed.** On the sample's test split, 5 of 56 campaigns are
  `structurally_disconnected`: pooling every one of their own edges before
  the deadline, their largest connected set of accounts is 2, and the rule
  needs 3. All are BIPARTITE and all are `fragmentable`. No model, lookback
  or condition can ever detect them, so they are a fixed floor under every
  result — and because they sit in the group whose fragmentation Phase 5
  studies, they would dilute that comparison if left in the denominator.
  Hence `oracle_reachable`. The full-data split of this floor across all
  three splits needs the next Kaggle run; only the pooled totals (36/30) are
  measured on the full data so far.
- **The baseline's full-data numbers are provisional.** One seed, at
  L = 24 only — so the 0.661 recall, the 0.804-of-ceiling figure and the
  ablation's 11-campaign coverage gap are all subject to change under five
  seeds and the second lookback. Nothing about earliness is established at
  all: the direct paired ablation is −3.0h with a CI of [−12, 0], which
  includes zero.
- **Everything else reported for stage 3.3 comes from the 1% sample** and is
  a **development check, not a result**: the thinned background makes separation artificially easy and
  false alarms artificially scarce, so neither the recall, the
  recall-against-ceiling, the payment-format dominance nor the ablation gap
  is evidence about the method. They are not a bar for stage 3.4.
- **Payment format may be doing most of the work, and it is a synthetic
  artefact.** The IBM generator links laundering strongly to format. That
  is why every model is reported with and without it; if the gap on full
  data is large, the honest reading is that the dataset is more separable
  than the problem, and the no-format numbers are the ones that transfer.
- **The 1% sample keeps every campaign transaction and thins only the
  background**, so it has 53,960 rows at a 5.99% laundering rate against the
  full data's 0.1%. Campaign-level recall and lead time transfer closely —
  it reproduces the full-data test outcomes exactly — but anything involving
  the background does not: false-alarm rates, precision, FA/day and any
  threshold chosen against them are ~60x off and must come from a full-data
  run.
- The pipeline currently targets only `HI-Small`; generalization to
  `HI-Medium`/`HI-Large` or the `LI-*` (low-illicit-ratio) files is
  unverified and deferred to a possible later replication.
- Institution assignment is bank-volume-balanced but not otherwise
  realistic (no real-world bank identities or geography).
- The local-search visibility procedure is a heuristic, not an optimal
  solver; achieved visibility can miss its target, which is why analysis
  uses bins rather than raw targets.
- Detection-window evaluation rules (Phase 3) are specified but not yet
  implemented or validated against the data.
- No model, training, federated-learning or experiment code exists yet. The
  four experimental conditions are defined and the result schema is fixed,
  but nothing has been run, so the project has no findings of any kind.
- **The Phase 3 comparison holds the extraction rule fixed, and that choice
  is not neutral between the two models.** The gate report shows the GNN's
  shortfall is at connected-component extraction rather than edge ranking:
  message passing correlates adjacent edges' scores, so its high scores form
  spurious components where XGBoost's scattered ones do not. The honest
  reading is **"the GNN does not beat the baseline under this extraction
  rule"**, not "graph models do not help here" — a rule less sensitive to
  score correlation might change the verdict, and testing that is future
  work. The rule stays frozen because every earlier result depends on it.
- **The Phase 3 models no longer exist.** The 3.4.4 `models/` trees were not
  kept, so every Phase 3 number is reproducible only by retraining from the
  seeds, the code and the data. That path is demonstrated — the stage 3.3
  retrain reproduced all four configurations to three decimals — but it costs
  a GPU session per lookback plus a CPU session.
- **Phase 4 is scoped down, and every cut is a limitation.** XGBoost only
  (no federated GNN), the full feature set only (no `no_format` arm), **three
  seeds rather than five**, and **one lookback**. The three-seed choice in
  particular gives **wider confidence intervals** than Phase 3's five, so a
  Phase 4 difference smaller than the seed spread must be reported as
  undemonstrated rather than claimed. The single lookback rests on a
  validation margin that did not survive to test. See
  §[Phase 4 frozen scope](#phase-4-frozen-scope).
- **Phase 4's per-institution training loads are measured on the 1% sample
  only.** The *relative* distribution across institutions transfers (it is a
  property of the assignment), but the absolute row counts, positive rates
  and class weights do not — the sample's positive rate is 3.55% against the
  full data's 0.102%. The full-data figures land with the first Kaggle run in
  stage 4.4; until then, "no institution is starved" is a sample result with
  a stated extrapolation behind it.
- **The isolated condition's union is not structurally handicapped, which
  constrains what Phase 4 can claim.** Measured on the sample: the union over
  institutions of "this institution's own view holds a connected triple of
  this campaign" is **0.884 at every visibility target**, identical to the
  centralized value (228 of 258 `eval_ok` campaigns) — because a three-account
  component always contains an account with two campaign counterparties, and
  that account's institution sees both of its edges whatever the assignment
  does. So any benefit `fedavg_exchange` shows over `isolated` cannot be
  attributed to "the cluster could not be assembled from one view"; it has to
  be attributed to **timing** (both edges inside one lookback window),
  **learning** (thin positives) or **weaker features** from a partial view.
  This is a structural proxy on the sample that ignores time, not the
  measured ceiling, and stage 4.1 is what measures the real one.
- **No stage 4.1 figure may be quoted yet.** Everything measured so far is
  on the 1% sample, whose 3.3% positive rate against the full data's 0.102%
  moves the class weights by an order of magnitude and with them every chosen
  tau — and three of the 32 institution-cells chose the most sensitive
  threshold on the grid, which at full-data prevalence should not happen. The
  ceilings are the one family of figures expected to transfer, because they
  are a property of the campaign edges and the assignment, which the sample
  preserves. Until the Kaggle run, stage 4.1 has **no** reportable result
  except the union-ceiling argument, which rests on the visibility rule rather
  than on data.
- **The condition names in code and in this README do not match yet, and
  4.2 has to settle it.** `constants.CONDITIONS` and the frontend's theme use
  `isolated`, `fedavg_only`, `fedavg_embedding`, `centralized`, while this
  README and the Phase 4 sections use `fedavg` and `fedavg_exchange` for the
  middle two. Only `isolated` is in use so far, and it agrees, so nothing is
  wrong today — but renaming touches the results schema and the frontend, so
  it is done once, deliberately, when 4.2 defines the condition it names,
  rather than silently now.
- **Declared before 4.2 runs: a null result for `fedavg` or
  `fedavg_exchange` is an expected possible outcome and is reported as a
  finding, not treated as a failure to tune.** It follows from the line
  above: since the isolated union already reaches the centralized structural
  ceiling, there is no structural headroom for coordination to recover, and
  any benefit has to come from timing, learning or feature quality — none of
  which is guaranteed to be large. This is written down **in advance**, on
  2026-11-14, for the same reason equal tuning effort was: a result that only
  counts as a result when it is positive is not a measurement. If a condition
  shows no benefit, nothing is tuned in response and the mechanism is
  diagnosed instead, exactly as the Phase 3 GNN verdict was.
- **Stage 4.1's scope is not settled.** Its instruction asks for five seeds
  and both lookbacks where the frozen Phase 4 scope says three seeds and
  L = 48; the amendment is costed in the proposal (96 cells against 320, and
  ~25-35 CPU hours against ~90-110 for the rest of Phase 4) and **awaits an
  explicit decision**. Whichever is chosen binds 4.2 and 4.3, because two
  conditions are only pairable over identical cells.
- The **1% research sample is present locally** (`backend/data/sample/`), so
  local development runs against the real sample rather than the synthetic
  generator. Stage 3.2's sanity-scorer numbers in this README still come from
  the seeded synthetic dataset and validate the evaluation *code*, not the
  dataset; they have not been re-run on the sample, because the full-data
  `evaluate-scorers` run superseded them.
- The first full-data `features-smoke` run (commit `caae35f`) reported a
  "largest candidate window" that violated rule 3 and was silently clamped.
  Its other figures — 54 edge dims, 13/9/5 usable training times, build
  times — stand; its largest-window line is **superseded** by the corrected
  scan. The original report is kept in `docs/reference/` as the first
  measurement.
- Reproducibility is verified for the visibility search but **only within
  one environment**. The reference digests hold for Python 3.13 / pandas
  2.3.3 / the Kaggle image of 2026-10-09; a different pyarrow could change
  the parquet bytes without changing the data, so a digest mismatch should
  be read as "investigate", not automatically "the search broke". The
  `unfragmentable` count and the achieved-visibility values are the
  environment-independent things to compare.
- **A CYCLE does not read as a ring, so the topology rationale for Cytoscape
  is only half met.** Measured 2026-11-14 on the real artifacts: `circle`
  layout orders nodes by collection order — `[...accounts].sort()`, i.e.
  alphabetical by account id — rather than by cycle traversal, so most of a
  cycle's edges cut across the interior as chords. Counting edges that join
  two *circle-adjacent* nodes: **campaign 192 5 of 8, campaign 219 5 of 7,
  campaign 181 4 of 10, campaign 37 3 of 7, campaign 62 2 of 6** (steps apart
  per edge for 192: `[1,1,1,1,1,3,3,3]`). The **FAN-OUT star claim holds** —
  `concentric` centres the hub account and the shape is unmistakable — and so
  does the determinism claim, since the same campaign still draws identically
  every time. Fixing it means ordering a cycle's nodes by traversal before
  handing them to `circle`; until then, a demo should not claim the ring is
  visible. Recorded rather than fixed, by decision of 2026-11-14.
- The frontend **has** now been inspected in a real browser (headless Chrome
  over CDP, 2026-11-14): all six pages and eight hand-picked campaign graphs
  render with no exceptions, no failed requests and no layout collapse, and
  all 370 campaigns build finite, non-overlapping Cytoscape layouts headlessly.
  What remains unverified is **human visual judgement** on a real display —
  colour rendition on a projector, and the low-contrast edge styling in light
  mode — plus every browser other than Chrome. The jsdom tests still stub the
  graph, so the browser evidence lives in this README, not in the suite.
- `backend/artifacts/` is empty on a fresh clone. Both the API and the
  frontend handle that deliberately (503 with the command to run, and an
  honest empty state), but the app shows nothing real until the pipeline has
  been run on Kaggle and the artifacts copied down.

## Future phases

- **Phase 3 — Baselines and evaluation (in progress):** see the Phase 3
  table in [Project status](#project-status). Implements the
  detection-window / matching / lead-time rules above, plus oracle and
  random sanity scorers, an XGBoost baseline and a centralized GNN. Its
  gate question is whether a full-visibility model can discover campaigns
  early at all.
- **Phase 4 — Federated conditions:** FedAvg-only and FedAvg + boundary
  embedding exchange.
- **Phase 5 — Core experiment:** run all four conditions across visibility
  bins, seeds, groups, and test subsets; populate `results/`.
- **Phase 6 — Extensions:** imperfect alignment across institutions,
  adversarial fragmentation, differential-privacy noise on shared
  embeddings/weights.
- **Phase 7 — Analysis and paper:** write up findings against the related
  work positioning in the Research question section.

**Follow-up suggested by the Phase 3 gate report, not attempted here.** The
GNN loses at cluster extraction rather than at edge ranking, because message
passing correlates the scores of adjacent edges and connected-component
extraction turns those correlated neighbourhoods into spurious clusters. The
natural follow-up is therefore **an extraction rule less sensitive to score
correlation** — for example requiring a minimum score margin across a
component, or a density/size prior — or **per-neighbourhood score
calibration** that removes the local correlation before thresholding. Either
would change a frozen definition every existing result depends on, so it is
future work and explicitly **not** a correction to the Phase 3 result: the
comparison reported here is between two models under one fixed extraction
rule, which is what makes it a comparison of models.

## Version history

Tagged points in the repository, newest first. Check one out with
`git checkout <tag>`.

| Tag | Date | What it contains |
|---|---|---|
| `v0.3-phase3` | 2026-11-14 | **Phase 3 complete.** The centralized gate question answered on the full data: the XGBoost baseline reaches 0.679 test recall at L = 48 (0.760 of the oracle ceiling), and the centralized GNN does **not** beat it — 0.525 against 0.654 at a matched 10 false alarms a day. The gate report locates the GNN's shortfall at **cluster extraction rather than edge ranking**, with the mechanism measured. 20 boosters and 20 GNN models trained on the full data across five seeds, two lookbacks and two variants; the stage 3.3 retrain reproduced every recorded number. Phase 4 (federated) has its stage 4.0 data layer built. Still no federated results and no visibility sweep. |
| `v0.2-verified` | 2026-10-09 | Phase 2 verified end to end on the full dataset, and reproducible. All 24 regression numbers pinned and confirmed on Kaggle; the visibility search made deterministic across processes, with reference SHA-256 digests recorded and confirmed across 2 sessions × 3 processes. Still no models and no experiment results. |
| `v0.1-data-pipeline` | 2026-10-08 | Phase 2 complete, plus the stage 3.0 Kaggle runner. The research pipeline (campaign ground truth, splits, topology groups, institutions, visibility), the artifact export, the read-only FastAPI server, and the full React frontend — 69 backend tests and 48 frontend tests passing. **No models and no experiment results**: the four conditions are defined and the result schema is fixed, but nothing has been trained or run, and the full-data regression numbers are still unverified (that is what stage 3.0's Kaggle run is for). |

## Changelog

- **2026-11-14 (stage 4.1 — grid-edge policy, and the sample figures fenced
  off)** — both decided before the full-data run, so no cell's treatment can
  depend on having seen which cells hit an edge.

  *Every 4.1 figure is now marked development-only and not quotable*, in the
  subsection itself and under Known limitations. The reason is the tau
  behaviour diagnosed yesterday: the sample's 3.3% positive rate against the
  full data's 0.102% moves the class weights by an order of magnitude and with
  them every chosen threshold, and so every recall, precision and alert rate
  downstream of one. The exception is noted rather than quietly relied on —
  the union-ceiling claim rests on an argument about the frozen visibility
  rule, not on this run, but its figures are still sample figures and its
  full-data confirmation is pending.

  *A tau at the TOP of the grid now makes that operating point undefined* —
  excluded from both pooled means, listed with its **rejected tau** and the
  reason, never scored 0 — because the optimum lies beyond the grid and the
  point was not found. It is reported exactly like an unaffordable budget, and
  for the same reason: a detector that could not be operated at a point was
  not measured there.

  *A bottom-edge F1 tau stops the full-data run and is not recorded*, with
  `--allow-low-edge-tau` for the sample only. Both paths were verified on the
  sample: the default stops with nothing written, the override writes the cell
  and still exits 3.

  *One refinement to the rule as stated, and it is not small.* A bottom-edge
  **budget** tau is neither flagged nor excluded, because it says only that
  the budget is loose. Measured: **166 of 224** (institution, operating point)
  cells on the sample chose tau = 0.05, of which **163 are budget points** and
  only **3 are F1 points**. Treating every low edge alike would have stopped
  on or excluded three quarters of the matrix for the uninteresting reason
  that `budget_100` is easy to satisfy.

  *Proposed, awaiting approval:* six more grid points at the top — 0.99995
  through 0.9999999 — continuing the existing tail's halving pattern.
  `MODEL_TAU_GRID` is frozen, so the code is unchanged. Evidence: the current
  headroom is **one** grid point (the highest tau ever chosen on full data was
  0.9995), the extension is nearly free (**2.6s** per sweep step above 0.99
  against 9.1s at 0.05, so ~26 minutes across the 96-cell run), and it stops
  where float32 does (the largest float32 below 1.0 is 0.99999994, and
  `1 - 1e-8` **is** 1.0). **1.0 itself is excluded on purpose**: `score >= 1.0`
  would keep whatever rounds to exactly 1.0, which is not reproducible across
  builds.

  40 tests in `test_federation_isolated.py`; the policy is pinned by tests
  that stub tau selection, so they are about the policy rather than about
  coaxing a fixture into producing a particular threshold.

- **2026-11-14 (stage 4.1 — BUILT)** — **the isolated condition exists, is
  tested, and has already produced one result.**

  *What was built.* `federation/isolated.py`: the masking scorer,
  per-institution training on the institution's own view, the per-institution
  oracle ceiling through the same mask, the union (earliest detection wins,
  clusters deduplicated by account set), the two pooled means with their
  denominators, seven operating points, and `run_cell`, which both runners
  call so the local and Kaggle reports cannot drift. The frozen scope is now
  **in code** as `constants.PHASE4_SEEDS` and `constants.PHASE4_LOOKBACK_H`,
  and `evaluate-isolated` refuses more than one lookback outright.
  `tests/test_federation_isolated.py`, **36 tests**; backend **397 passed, 25
  skipped**. New runner subcommands `train-isolated` and `evaluate-isolated`,
  resumable per institution and per `(target, seed)`; the evaluator never
  trains, so a lost session cannot silently retrain on different boosters.
  Verified across processes: the eight taus chosen in one process are the
  eight chosen from the saved boosters.

  *A result, not a development reading.* The isolated **union's** oracle
  ceiling equals the **centralized** ceiling exactly, with the 6-hour
  detection grid, the lookback and the deadlines in force: val 0.7692 against
  0.7692 and test 0.8929 against 0.8929, at all four visibility targets. The
  degree-2 argument carries through time — the institution holding an account
  with two campaign counterparties sees the same component in the same window
  at the same detection time, so it matches not only the campaign but the lead
  time and the side of the deadline. Recorded in the Decision log, with the
  consequence declared in advance: **a null result for 4.2 or 4.3 is an
  expected possible outcome**, to be reported as a finding.

  *Three development readings* (sample; the thinned background makes
  precision, alert rates and budget taus untransferable). An individual
  institution's own ceiling is far lower and moves the **opposite** way to the
  union's — test mean 0.194 at target 1.0 against 0.442 at 0.25 — so the
  dose-response runs in opposite directions for the two pooled numbers. Each
  institution nevertheless achieves **0.6-0.9 of its own ceiling**: its
  problem is not that it detects badly on what it sees, it is that it cannot
  see. That is a prediction about 4.2 versus 4.3 worth having on the record
  before either runs. And both approved reporting rules fired on real data —
  institution 6 at target 0.75 has **no affordable tau** at `total_10`, so
  that cell is undefined and the pooled mean is over 7 institutions and says
  so, while deduplication sheds **80-138** duplicate clusters a cell at
  targets 0.75 and below against **3-8** at target 1.0.

  *The sample run exits 3, at the LOW edge of the tau grid* — three of 32
  institution-cells chose tau = 0.05, the opposite direction from stage 3.3's
  near-1 problem, because the sample's 3.3% positive rate puts class weights
  near 30 instead of 495-3,461. Expected on the sample; on full data either
  edge must be explained. Both runners now print **which** edge was hit, and
  the explanation of the two directions lives in `_edge_tau_rows`.

  *Seven operating points, because four would have flattered the floor.* At
  target 0.25 the union's deduplicated volume is 39.8 false alarms a day at
  `budget_10` against 18.3 at `total_10` — eight institutions each inside 10 a
  day are not together inside 10 a day. Both readings are kept, and the
  per-institution-budget union prints its realized total beside it.

  *One performance defect found and fixed during the run.* The union's cluster
  pass was given fresh scorers, so it re-featurised every window for every
  institution at every operating point, and a cell that should take minutes
  hung. Each institution now owns one warm `CachingScorer` that the
  per-institution evaluation and the union pass share; the cell that hung now
  finishes in 151s. The full-data estimate is revised to **7-14 CPU hours**
  for the 96 institution-cells.

  *Also corrected:* `boundary_accounts`' description pointed at "stage 4.1's
  exchange condition", which is now stage 4.3. And a limitation is recorded:
  `constants.CONDITIONS` and the frontend still spell the middle two
  conditions `fedavg_only` / `fedavg_embedding` where this README says
  `fedavg` / `fedavg_exchange`. Only `isolated` is in use so far and it
  agrees; the rename touches the results schema and the frontend, so 4.2 does
  it once, deliberately.

- **2026-11-14 (stage 4.1 — APPROVED, with one measured correction)** — the
  design is approved with six decisions and implementation begins.

  *Scope settled.* The five-seed/two-lookback instruction is **withdrawn**;
  the frozen scope stands at **3 seeds, L = 48**, 96 cells. Standing rule
  recorded for any future widening: **seeds widen, lookbacks do not**.

  *Two reporting rules fixed.* The per-institution mean is now reported
  **twice** — over all 8 institutions and over **trained** institutions only
  — because one mean conflates "cannot see the campaign" with "had no labels
  to learn from", which are different claims about federation. Recall 0
  versus undefined is accepted unchanged.

  *The union now has an owner.* Its matched-total budget (B / K, deduplicated)
  is accepted, and the README states **whose number it is: a regulator or FIU
  receiving all eight alert streams, not a bank** — no individual bank can
  compute it or benefit from the deduplication. Equal split is primary
  **because it is untuned**: an optimised allocation would be a tuning step
  the equal-effort constraint does not cover.

  *Declared in advance:* a **null result for 4.2/4.3 is an expected possible
  outcome**, to be reported as a finding and not treated as a failure to tune,
  because the isolated union already reaches the centralized structural
  ceiling and leaves no structural headroom to recover.

  *One correction to what was written yesterday.* The reversal mechanism in
  §4(a) was stated as "at target 1.0 a campaign's accounts are all moved to
  one institution". That is true of only **144-147 of 258** campaigns.
  Measured and rewritten: full visibility needs **one institution to be an
  endpoint of every campaign edge**, which for a hub topology means holding
  the hub rather than the whole campaign, so 113 campaigns reach visibility
  1.0 with their accounts still spread across institutions. The per-institution
  figure is also now stated to be a **mean over the 8 institutions, not a
  max**, with the strongest-institution column added — the reversal survives
  there too (0.353 at target 1.0 against 0.609 at 0.25), so it is not an
  artefact of averaging. `run_structural_ceiling.py` now prints the reach
  distribution and the single-holder count that back these claims.

- **2026-11-14 (stage 4.1 — PROPOSAL ONLY, awaiting approval)** — **the
  isolated condition is designed, nothing is implemented.**

  *What changed in the plan.* Stage 4.1 is now the **`isolated` condition** —
  each institution trained and scoring on its own view alone — and the
  tree-model definitions of `fedavg` and `fedavg_exchange` move to **4.2** and
  **4.3**. The floor comes first because a federation that cannot beat
  uncoordinated banks each reporting separately has bought nothing.

  *The design.* An institution is restricted by its **scorer**, not by its
  data: `full_df`, `camp`, the detection grid, the horizons and the deadlines
  stay global and identical to Phase 3, and institution *i*'s scorer returns 0
  outside `views.visible_mask` and its booster's probability inside, with
  features built from the view alone. **No line of the Phase 3 evaluation code
  changes.** Passing the view as `full_df` is rejected and why is recorded: it
  would move the ground truth, because the unassigned-laundering set that
  makes a cluster `ambiguous` is derived from that argument. Each institution
  gets **its own oracle ceiling** per target, through the same mask, never
  clamped and never the centralized 0.821/0.893.

  *Measured for the proposal*, `scripts/run_structural_ceiling.py` (new,
  under 10s on the sample): the per-institution **structural** reach runs
  **backwards** — 0.255 at target 1.0 against 0.546 at 0.25, because at
  target 1.0 a campaign sits inside one institution while at 0.25 its
  scattered edges are each seen by both endpoints — and the **union is 0.884
  at every target, exactly the centralized value** (228 of 258 campaigns).
  That identity is a theorem the script checks: a three-account component
  contains an account with two campaign counterparties, whose institution is
  an endpoint of both edges. Consequence recorded under Known limitations:
  any `exchange`-over-`isolated` benefit cannot be attributed to "the cluster
  could not be assembled from one view".

  *Two decisions requested, with the premise of one corrected.* The thin-data
  risk is at the **100 bin**, not the 25 bin (stage 4.0: 3.13× positive spread
  at target 1.0 against 1.60× at 0.25). Proposed: 0 positives → `untrained`
  and recall **0** (an outcome, not missing data); 1-9 → trains, flagged
  `thin`; `MIN_POSITIVES_TO_TRAIN` stays a reporting threshold, not a gate.
  And zero qualifying clusters at an existing tau → recall **0** (Phase 3
  precedent: the no-format GNN at 10 FA/day), while **no affordable tau on the
  grid** → **undefined**, excluded from the pooled mean and listed, because
  `select_tau_for_budget` already returns `None` for exactly that and 0 would
  conflate "found nothing" with "cannot be run here".

  *Raised as an objection.* Eight institutions each at 10 false alarms a day
  spend 80 a day, so the **union's primary operating point matches total
  volume** (B / K per institution, duplicate cluster identities
  deduplicated), with the per-institution-B union reported separately and its
  measured total volume printed beside it. The per-institution mean is a
  **statistic, not a detector**, so it is never the subject of a paired
  comparison; the union is.

  *Awaiting a decision.* The instruction's **five seeds and both lookbacks**
  are a change to the frozen Phase 4 scope and are **not applied**: 96 cells
  against 320, ~7-12 CPU hours against ~20-34, and ~25-35 hours against
  ~90-110 for the rest of Phase 4. Recommendation: keep three seeds and
  L = 48; if one widens, widen seeds, not lookbacks. Whatever is chosen binds
  4.2 and 4.3, since two conditions are only pairable over identical cells.
  **(Resolved the same day — see the entry above: the frozen scope was kept
  and the wider instruction withdrawn.)**

- **2026-11-14 (stage 3.4.4 + 3.4.5 — PHASE 3 COMPLETE, tag `v0.3-phase3`)** —
  **the full run is in, the GNN does not beat the baseline, and the reason is
  located.**

  *The run.* Three Kaggle sessions as planned — two GPU (ten models each) and
  one CPU session in commit mode that retrained the 20 boosters, evaluated all
  40 models and ran `compare-models`. All exit 0, **no tau at a grid edge**.
  The `models/` trees were not kept; the reports are committed under
  `docs/reference/`.

  *The regression check passed exactly*, which was the point of retraining
  rather than the comparison it was run for. All four XGBoost configurations
  reproduce the recorded stage 3.3 numbers **to three decimal places**,
  including the per-seed min and max and the five-fold seed-noise difference
  between the variants: L = 48 full at **0.679 [0.661, 0.696], 0.760 of the
  ceiling**.

  *The result.* At the F1 operating point the GNN's L = 24 recall (0.661) sits
  inside seed noise of XGBoost's (0.654) but costs **20.3 false alarms a day
  against 3.8**; at L = 48 it is behind on both (0.636 at 34.2/day against
  0.679 at 10.5). **At a matched 10 FA/day budget the gap is unambiguous:
  0.525 against 0.654.** The GNN is also 3–5× noisier across seeds. No
  earliness difference is demonstrated in either direction — the direct paired
  comparison is 0.0h with a CI containing zero in every seed — so what differs
  is coverage.

  *The diagnosis, and it is the useful part.* **The shortfall is at cluster
  extraction, not at edge ranking.** Edge-level AP on test at L = 24 is
  **0.394 (XGBoost) against 0.392 (GNN)**, and on the four
  validation-detection windows the GNN early-stops on the **GNN ranks
  better**, 0.435 against 0.425 — while its recall at 10 FA/day is 0.525
  against 0.654. The mechanism is in the false-alarm cluster shape: at the F1
  point (test, L = 24, seed 0) XGBoost produces **16** false-alarm clusters,
  **all exactly 3 accounts**, and the GNN produces **158**, mean **3.20**
  accounts, up to **6**. Message passing correlates adjacent edges' scores, so
  the GNN's high scores arrive in connected neighbourhoods and form spurious
  components; XGBoost scores transactions independently, so its high scores
  are scattered and rarely form a connected triple. **The property that should
  help a graph model works against it under connected-component extraction.**
  At L = 48 the GNN is additionally behind on ranking (0.350 against 0.387),
  recorded separately because it is a different deficit.

  *Also recorded:* the **no-format GNN cannot operate at 10 alarms a day** —
  zero clusters in all five seeds at L = 24, and two of five at L = 48 — so
  its 0.032 mean recall is a detector switched off by its own threshold rather
  than a weak one. The XGBoost ablation reaches 0.511 there.

  *This is a finding, not a defect.* The GNN is **not** tuned and the
  extraction rule is **not** changed: both were frozen before any full-data
  training, and changing either now would make the comparison a record of
  which model got more attention. The follow-up it suggests — an extraction
  rule less sensitive to score correlation, or per-neighbourhood score
  calibration — is written into §[Future phases](#future-phases) as further
  work, and the limitation that the verdict is "under this extraction rule"
  is stated under Known limitations.

  *Phase 3 is marked COMPLETE* and tagged **`v0.3-phase3`**. Stage 3.5's
  results files and the frontend Results page stay deferred to Phase 5, which
  is what populates `artifacts/results/`.

- **2026-11-14 (stage 4.0)** — **Phase 4 begins: the institution-split data
  layer is built, tested and measured. Nothing federated is implemented yet.**

  *Scope frozen first.* Phase 4 is cut to fit a two-week finish: **XGBoost
  only** (no federated GNN — on full data it matched XGBoost's recall at ten
  times the alert volume), **the full feature set only** (the format ablation
  stands as a centralized result), **seeds 0-2** rather than 0-4, **one
  lookback, L = 48** by the frozen Phase 3 protocol on validation, and
  **K = 8 unchanged**, since halving K would raise the visibility floor from
  ~25% to ~50% and flatten the sweep. Five Decision-log entries record each
  cut with its reason, and the narrowing is listed under Known limitations —
  three seeds mean wider confidence intervals, and that is reported rather
  than hidden.

  *Stage 4.0.* `federation/views.py` turns a `(seed, target)` assignment into
  per-institution **views**. Two choices matter. The frozen "sees" rule —
  sender **or** receiver — is implemented **once**, and a test requires it to
  agree row for row with Phase 2's own `visibility.visible_transactions`, so
  the project cannot acquire a second notion of visibility. And a view is
  returned as a `windows.Window`, so `features.build_window_features` runs on
  it **unchanged** and computes an institution's node features from its own
  view alone — no second featuriser, and the Phase 3 leakage properties carry
  over. A view is a pure row subset, so institution identity never reaches
  the model inputs.

  *Tests:* `tests/test_federation_views.py`, 31 tests, backend **362 passed,
  25 skipped**. The three defining properties were each **proven able to fail
  by mutating the real source**: sender-only visibility breaks 4 tests,
  looping `range(k - 1)` breaks 2, and a `mean` in place of the `max` over
  institutions breaks 13. The visibility check is a real cross-check — the
  pipeline computes achieved visibility from each campaign's pattern edge
  list, the test counts rows of the tagged transaction table landing in each
  view — and it agrees exactly: **0 mismatches across all 370 campaigns in all
  26 runs**.

  *Measured on the 1% sample (a development check, not a result).* Pooled over
  the 9 rule-3 training windows at L = 48, **no institution is starved of
  positives at any target**; the thinnest cell is institution 2 at seed 1,
  target 1.0, with **274 positives in 19,803 rows**. The finding worth
  carrying into 4.1: **target 1.0 is the most imbalanced condition, not the
  easiest** — the positive spread across institutions is **3.1× at target 1.0
  against 1.6× at target 0.25**, because at target 1.0 a campaign's accounts
  all go to one institution and its laundering edges stop crossing
  boundaries, so fewer institutions see each positive. The isolated
  condition's weakest institution is therefore weakest at the **100 bin**, and
  that is a candidate explanation for any flattening at the top of the
  dose-response curve.

  *Added:* `federation/__init__.py`, `federation/views.py`,
  `tests/test_federation_views.py`, `scripts/run_institution_split.py`, and a
  new README section, §[Phase 4 definitions](#phase-4-definitions-the-federated-conditions).
  Two stale Known-limitations entries removed on the evidence of this work:
  the 1% sample **is** present locally, and the evaluation module **has** been
  run on full data.

  *Next:* stage **4.1**, the design proposal — what "FedAvg" and "embedding
  exchange" mean for a tree model. No code until it is approved.

- **2026-11-14 (demo readiness)** — **the frontend was audited in a real
  browser for the first time; the campaign graph works, and the "a CYCLE reads
  as a ring" claim does not.**

  *The audit.* Artifacts re-verified against the reference digests (370
  campaigns, 258 eval_ok, 152 reassignable, 3,209 campaign transactions, five
  seeds × 14,384 rows, **all five SHA-256 digests byte-identical** to
  §[Verified numbers](#verified-numbers), nothing nested). Every `/api`
  endpoint exercised over HTTP across 45 calls: no unexpected status, every
  non-200 an expected 404 or 422, and the slowest response **50 ms**. Every
  TypeScript response type in `src/api/types.ts` diffed field-by-field against
  the **live** responses rather than the schema — 17 interfaces, no drift.
  `npm run build`, `tsc -b`, `oxlint` and the 48 frontend tests are all clean
  (and oxlint was confirmed able to fail, by injecting a constant condition).

  *The browser.* Headless Chrome over the DevTools protocol — no new
  dependency — rendered all six pages and eight hand-picked campaign graphs
  with **no exceptions**, and all 370 campaigns were built headlessly through
  the component's own element construction and layout choice without a single
  throw, non-finite position or collapsed layout. Self-transfers (11 rows in
  11 campaigns, all GATHER-SCATTER) draw as loop arcs; the 68 campaigns with
  reciprocal pairs separate under `curve-style: bezier`; the heaviest campaign
  reaches first render in 198 ms. Error states were exercised too: a dead
  backend puts an honest message on every page in ~150 ms. Details in
  §[Testing](#testing).

  *What it found.* **A CYCLE does not read as a ring** — `circle` layout orders
  nodes alphabetically by account id rather than by cycle traversal, so most
  edges cross the interior as chords (campaign 192 joins 5 of 8 circle-adjacent
  pairs, 219 5 of 7, 181 4 of 10, 37 3 of 7, 62 2 of 6). The **FAN-OUT star
  claim holds** and so does determinism. **Decided 2026-11-14: recorded, not
  fixed** — this is a presentation flaw, not a correctness one, and it is
  handled by not claiming the ring. Also found: `/campaigns/<non-numeric>`
  leaves the main panel empty with no message. Both are in
  §[Known limitations](#known-limitations-and-open-questions); the remaining
  findings were cosmetic and are deferred.

  *Added:* `docs/demo_checklist.html` — a standalone, dependency-free run sheet
  with the startup commands, the campaigns to show (211, 241, 219) and the
  expected value at every step. **No code changed.**

- **2026-11-13 (3.4.4 plan revised)** — **the stage 3.3 boosters are gone,
  so the full run needs three sessions and session B retrains them.**

  The stage 3.3 run was an *interactive* Kaggle session, so its
  `/kaggle/working` was discarded when the session ended: the 20 boosters do
  not exist any more. Session B of the 3.4.4 plan therefore runs `train`
  (1.44h measured) before `evaluate` (1.59h), because `evaluate` reads the
  models `train` writes in the same session, and the whole session runs in
  **commit mode** rather than interactively. Nothing is lost that cannot be
  recreated — the boosters are a deterministic function of the seeds, the
  code and the data — and the retrain doubles as a **regression check**: the
  re-evaluated numbers must reproduce test recall 0.679 and 0.760 of the
  ceiling for the L = 48 full model, with no tau at a grid edge.

  *Session B, in order:* locate both GPU sessions' `models/gnn` trees under
  `/kaggle/input` with `find`, copy them into one directory and **assert**
  20 models and both specs are there — a one-minute check that fails before
  hours are spent, rather than `evaluate-gnn` exiting 2 after the XGBoost
  work; then `train`, `evaluate`, `evaluate-gnn`, `compare-models`. XGBoost
  first because its cost is known to within a percent, which leaves the
  whole remaining margin for the one uncertain step. **~4.8–7.1 hours** of a
  12-hour session. The copy cell was run against a fake `/kaggle/input` tree
  in all three states — both sessions attached, one attached, none — and
  stops with a named assertion in the last two.

  *The GPU estimates are now anchored on measurement* rather than on the
  live log's epoch time. `train_gnn.md` from the quick check gives **463.4s
  for one model of 56 epochs** — ~95s to featurise and cache the 17 window
  graphs on the host, then ~7s an epoch — plus a 54.4s feature-spec fit per
  lookback at a 10.6 GB host peak, and `deterministic: true`. So ten models
  is **~1.4h at L = 24** and **~1.9h at L = 48** (~2.2h and ~3.1h if every
  one runs the 100-epoch cap), each inside a 9-hour session. The tightest
  point of the stage is L = 48's GPU memory: the probe's 10.70 GB for the
  largest window plus the 1.16× training added at L = 24 projects **~12.4 GB
  of a 15.6 GB card**, which is extrapolated, not measured, and fails loudly
  (exit 4 with the window, epoch and peak) rather than silently.

  *Evidence committed:* `docs/reference/train_gnn.{md,csv,json}`.

- **2026-11-10 (3.4.3 + the evaluation gaps)** — **the quick check ran; the
  GNN is behind; four gaps in the evaluation closed and the full-run plan
  written.**

  *The quick check* (Kaggle, `8d8929c`, both sessions exit 0; evidence in
  `docs/reference/evaluate_gnn.*`). One model, L = 24, full variant, seed 0.
  Training behaved as designed: first epoch's loss ~0.69 (`ln 2`, so the
  loss scale is right), `max |logit|` 1.2 → 14 against the 10⁴–10⁵ the
  unnormalised probe implied, peak GPU memory **7.43 GB flat across epochs**
  against the probe's 6.39 GB for the largest window alone — 16% above it,
  in the expected direction, and flat, so nothing leaks — ~7s an epoch, best
  epoch 45, stopped at 55 with patience 10. Validation AP is 0.419 on the
  full data against 0.94 on the 1% sample, which is the positive rate
  (0.102% against ~6%), not a regression: the sample's figure was never
  comparable.

  *The result, provisional on one seed.* Test recall 0.661 (37/56, 0.804 of
  the ceiling) at tau 0.995 — the same recall as XGBoost at the same
  lookback and seed, **at 33.3 false alarms a day against 3.4**. Equal
  recall at ten times the alert volume is not equal. The comparable readings
  favour XGBoost: best validation campaign-F1, which is what tau selection
  maximises, 0.418 against 0.543; validation recall at ~4 alerts a day 0.231
  against 0.423. So the GNN is behind by roughly a factor of two in recall
  at a comparable alert volume on this configuration. **It is not being
  tuned to close that gap** — equal tuning effort is what makes the
  comparison a comparison of models, and only a demonstrated bug, a named
  defect with evidence, justifies a change.

  *Gap 1, the alert budgets.* `evaluate-gnn` was missing them, and they are
  required by the frozen design. Added: 10, 50 and 100 a day, tau per budget
  chosen on **validation** from the same curve the F1 tau comes from. This
  is the fair head-to-head, exactly because the F1-chosen points sit an
  order of magnitude apart in alert volume.

  *Gap 2, no direct comparison was possible at all.* Rule 3 requires A and B
  be compared to each other, which needs both models' per-campaign detection
  times at a matched seed, lookback and operating point — and **stage 3.3
  saved none of them**: summaries, tau curves, budget rows and its own
  comparisons against the oracle, but not one row per campaign. So every
  evaluation now writes `*_outcomes.csv`, and a new **`compare-models`**
  subcommand reads two such files and nothing else — no model, no features,
  no transactions — pairing on split, lookback, seed, operating point and
  ablation arm, reporting earliness over the campaigns both detect with a
  bootstrap CI and coverage each way by campaign id, and **refusing** to run
  if two detectors collapse into one key. Stage 3.3's records are
  regenerated by re-running `evaluate`: CPU only, 1.59h measured, no
  retraining. Reading files rather than live models is also what lets the
  Phase 4+ federated conditions be compared across sessions.

  *Gap 3, two diagnostics that select nothing.* Edge-level average precision
  over the raw scores, for both models on the **same** four
  validation-detection-split windows the GNN early-stops on and on each
  horizon — if the campaign-level gap is there too it is a ranking gap, and
  if it is not it lives in what survives clustering. And the size
  distribution of false-alarm clusters at each operating point, because a
  hundred three-account components is not the same alert load as a hundred
  forty-account ones. The GNN's early stopping and the diagnostic call one
  function, so the two numbers cannot drift.

  *Gap 4.* The quick-check numbers are recorded above as provisional.

  *Code.* New `evaluation/edge_metrics.py` and `evaluation/compare.py`, both
  model-agnostic; per-campaign records and cluster-size distributions in
  `evaluate.py`; the budget loop and both diagnostics in `evaluate` and
  `evaluate-gnn`; `windows.full_lookback_detection_times` so the GNN's
  early-stopping window set has one definition; the `compare-models`
  subcommand; runbook §5k for the full run.

  *Verified end to end on the real 1% sample* — development numbers only:
  both models trained and evaluated at L = 24 seed 0, then `compare-models`
  pairing all four operating points on both splits. The GNN's
  `val_detection` edge AP there is 0.9406, **the same number training
  reported as its best epoch's validation AP**, which is the check that the
  diagnostic computes the quantity early stopping used.

  *Tests:* backend 331 passed, 25 skipped (`test_compare_models.py`, 22 new).
  The three leading guards — the boolean round trip, in-process-versus-file
  identity, and the refusal to mix two detectors under one key — were each
  confirmed to fail with the guard removed.

- **2026-11-10 (3.4.2)** — **3.4.1 approved with four additions; the GNN is
  implemented, tested and smoke-run.**

  *Approved as proposed:* the balanced-mean loss, LayerNorm after each
  convolution, sequential training on one GPU, one shared feature spec per
  lookback, window-major CPU evaluation.

  *Addition 1, the root cause.* `log1p` on the heavy-tailed node counts —
  `in_degree`, `out_degree`, `n_counterparties`, `n_currencies` — applied
  **inside the GNN only**, before standardising, with the GNN's own
  statistics fitted on training windows. The shared spec is untouched and
  stage 3.3 stands: a tree splits identically under a monotone transform.
  Measured on the sample's largest windows, maximum |x| falls **109.9 → 40.4**
  (L = 24) and **158.5 → 42.2** (L = 48), and on `out_degree` alone
  156.6 → 15.3. It reduces but does not bound — `n_counterparties` still
  reaches 40, and `n_currencies` barely moves because standardisation
  normalises away any monotone transform of a near-binary column — so
  LayerNorm stays.

  *Addition 2.* Early stopping on validation edge-level average precision,
  every epoch, over the **validation detection split only**. That
  restriction is the leakage trap here: AUPRC reads every label in a window,
  and the validation *evaluation horizon* reaches into the test detection
  split, so stopping on it would select on test-period edges. Campaign-F1 is
  not used for stopping (slow per epoch, and 1/26 steps over 26 campaigns);
  tau is still chosen once at the end by validation campaign-F1 on the model
  grid.

  *Additions 3 and 4.* One frozen hyperparameter configuration, never tuned
  on validation or test — recorded as the condition that makes the
  GNN-versus-XGBoost comparison fair — and every parameter stated
  explicitly: 3 layers, hidden 64, dropout 0.0, Adam 1e-3, **weight decay
  0.0**, max 100 epochs, patience 10, per-epoch window order shuffled from
  `seed * 1000 + epoch`.

  *Implemented:* `models/gnn.py`, `tests/test_gnn.py` (25 tests, backend 309
  passing), the `train-gnn` and `evaluate-gnn` subcommands, and `torch` /
  `torch_geometric` in requirements unpinned. The scorer matches the XGBoost
  scorer's interface, so the evaluation code is shared rather than
  duplicated. Resumability: each model saved the moment it finishes and
  skipped on restart, `is_trained` requiring all four files so a
  mid-save interruption is retrained rather than loaded broken, the spec
  fitted once per lookback, and partial reports written per configuration.

  *Local CPU smoke run on the real 1% sample — development numbers only.*
  Loss starts near `ln 2` and falls to 0.06, validation AP reaches 0.94,
  early stopping fires at epoch 31 having peaked at 21, and **`max |logit|`
  stays near 10** against the 10⁴–10⁵ the unnormalised probe implied.
  Evaluation gives test recall 0.786 at tau 0.99 (interior), 0.957 of the
  oracle, 0.0h paired against the ceiling.

  *One implementation bug found and fixed while measuring:* the first
  evaluation loop built each window's graph once **per model**, which would
  have featurised every window ten times and defeated the window-major
  design. It now builds once per (window, variant) — the five seeds of a
  variant consume byte-identical graphs — and the numbers are unchanged
  either way, which is the invariance that restructuring had to preserve.
  Featurisation costs ~40 µs/edge against ~9 µs for a forward pass on the
  sample's small windows, so it is indeed the dominant cost.

- **2026-11-10 (3.4.1 proposal)** — **Full-batch training decided; the
  design proposal written, awaiting approval.**

  *Decided:* full-batch at **both** lookbacks, closing the sampling
  decision reopened the same day. The largest L = 48 window peaks at 10.70
  GB of 15.6 against the 12 GB rule, L = 24 at 6.39 GB, and inference runs
  unchunked at 4.75 GB. Full-batch removes three parameters the comparison
  does not need — fanouts, batch size, sampler bias — and the 2026-10-10
  estimate that argued for sampling was twice the measurement.

  *The loss scale is explained, and two of the three candidate causes are
  ruled out by arithmetic.* The reduction is already `mean` (a `sum` would
  have given ~6.7 × 10¹¹, not 3.6 × 10⁵), and the class weight is
  self-normalising: `n_pos × pos_weight = n_neg ≈ N`, measured at 0.9997
  and 0.9996 of N, so it contributes ≈ 1 whatever the window. What is left
  is the logits, of order 10⁴–10⁵. Measured support: after the feature
  spec's standardisation, node features on the sample's largest windows
  still reach **|x| = 110 and 159** — standardising a heavy-tailed degree
  column does not bound it — and three layers of **sum** aggregation over
  high-degree accounts compound from there. Edge features are fine (max
  4.6).

  *Proposed:* a **balanced-mean loss** (½ mean over positives + ½ mean over
  negatives), which keeps the frozen class weight exactly — per-positive to
  per-negative weight is `n_neg / n_pos` — makes window-invariance exact
  rather than approximate, and starts at ln 2 ≈ 0.693 for any window, so a
  wrong first step is obvious. And **layer normalisation** after each
  GINEConv, the proposal's one addition to the frozen design: chosen over
  BatchNorm (its running statistics would be fitted on training windows and
  applied to evaluation windows up to 3× different in size) and over mean
  aggregation (which would discard what makes a GIN a GIN).

  *Also proposed:* `expandable_segments:True` and per-epoch peak logging
  with a loud failure on OOM and **no silent fallback**; pre-fitting each
  spec once and sharing it, since the 14.1 GB host peak is the spec fit
  rather than training — which makes two-GPU parallelism safe but also
  unnecessary, as the whole 20-model matrix is ~2 h on one GPU; evaluation
  in a separate CPU session, **window-major** so each evaluation window is
  featurised once and scored by all ten models at that lookback, keeping
  1.6 GB of scores instead of ~14 GB of features and saving roughly 1.2 of
  the 1.59 hours stage 3.3 spent; early stopping on **validation
  campaign-F1** rather than loss, with the reasons; seeded window shuffling;
  and an honest statement that GPU scatter may not be bit-reproducible,
  mitigated by reporting the five-seed spread and by evaluation running from
  saved weights.

  Nothing is implemented. 3.4.2 starts once the proposal is approved or
  amended.

- **2026-11-10 (3.4.1 evidence)** — **`gpu-probe-windows` ran on one T4,
  and full-batch training fits both lookbacks.**

  The largest L = 48 training window — 1,869,370 edges, 509,993 accounts —
  peaks at **10.70 GB reserved** against the 12 GB rule fixed before the
  run, leaving 4.9 GB on a 15.6 GB card; L = 24 peaks at **6.39 GB**. Steps
  take 0.28s and 0.45s. Inference needs **no chunking**: the largest
  evaluation window scores unchunked in 0.18s at 4.75 GB. Host memory peaks
  at 14.1 GB during the feature-spec fit, against 33.7 GB — the GPU is not
  the binding constraint. Every window matched what `features-smoke`
  recorded.

  **Running real windows mattered.** They came in **~25% heavier** than
  `gpu-check`'s synthetic graphs — 5.73 against 4.6 MB per 1,000 edges — so
  the synthetic projection of 8.6 GB for L = 48 understated the truth of
  10.70. Still inside the budget, but with meaningfully less headroom than
  the projection implied, which is why the projection was never treated as
  the answer.

  **The probe also exposed something that has to be fixed before any
  training.** The losses are **10,615 at L = 24 and 356,394 at L = 48**
  after one step from random initialisation, where a class-weighted BCE on
  these windows should start near 1–2. GINEConv aggregates by **sum**, these
  windows hold very high-degree accounts, and three layers of summing
  64-dimensional messages with no normalisation between them compounds past
  what a linear readout can turn into sensible logits — the largest window
  being 34× worse than the smaller one is the pattern that predicts. The
  frozen design names GINEConv, 2–3 layers and hidden 64 and says nothing
  about normalisation, so **3.4.1 must propose one** and have it approved. A
  model whose activations explode on the busiest windows would still train
  and still report numbers.

- **2026-11-10 (later)** — **Every GPU subcommand now installs PyG
  itself.** `gpu-probe-windows` failed on a fresh Kaggle session with "No
  module named 'torch_geometric'", because only `gpu-check` installed it.

  The install logic is now one function, `ensure_torch_geometric`, which
  every GPU subcommand calls and any later one should too. The first GPU
  command of a session pays about 30 seconds for it; the rest find it
  already present. Only `gpu-check` asks for the optional extensions, since
  they are not required and a command that needs to train should not spend
  session minutes installing wheels it can run without.

  The alternative — documenting that `gpu-check` must run first — is a rule
  that gets forgotten, as it just was.

  Verified by uninstalling `torch_geometric` locally and running the probe:
  it installed it in 29s, re-imported it and completed all four windows.
  `gpu-check`'s own report is unchanged apart from a new `installed_now`
  field.

- **2026-11-10** — **3.4.0 measured, and the sampling decision reopened.**

  `gpu-check` passed on Kaggle at commit `2755139`, exit 0: Python 3.13.16,
  torch 2.11.0+cu130, PyG 2.8.1, **all four optional extensions installed**,
  **2 × Tesla T4 (15.6 GB each)**, **33.7 GB system RAM**. GINEConv 3×64
  forward+backward took 0.07–0.16s at every size, using 251 MB at 50k edges,
  936 MB at 200k and 2,302 MB at 500k.

  **That is ~4.6 MB per 1,000 edges, about half the estimate the sampling
  decision rested on.** Projected onto the real windows: ~5.1 GB for the
  largest L = 24 training window (1.11M edges) and ~8.6 GB for the largest
  L = 48 one (1.87M), against the shape-based ~16.6 GB of 2026-10-10. So
  full-batch may fit both lookbacks, and the 2026-10-10 regime decision is
  **reopened** — its *one regime for both lookbacks* half stands regardless,
  but which regime is now an open question.

  It is still only a projection, so nothing is decided on it: a random
  graph's account-degree distribution is not a real window's, and linearity
  holds until it does not. `gpu-probe-windows` runs **the real thing** — the
  largest rule-3 training window at each lookback and the largest evaluation
  window, with real features from a spec fitted on every training window as
  training fits it, through GINE 3×64 + the edge readout MLP + the
  class-weighted loss + an Adam step on **one** T4. It reports peak memory
  allocated *and* reserved (reserved is what runs out), MB per 1,000 edges,
  step time and build+featurise time, finds the largest windows by a full
  scan rather than remembering them, and checks them against what
  `features-smoke` recorded so a mismatch shows. **The decision rule is
  fixed in advance:** full-batch only if the largest L = 48 window peaks
  below 12 GB reserved, leaving headroom on a 15.6 GB card.

  *Correction:* the claim that stage 3.3's 15.9 GB peak was "conclusively
  too large for a GPU notebook" was **wrong** — it put a host-memory figure
  against the GPU's device memory. A GPU notebook has 33.7 GB of system RAM,
  so the matrix would fit there. Stage 3.3 stays on CPU for the reason that
  always applied: pandas and scipy dominate it, and 3.4 needs the quota.
  Corrected in all four places it appeared.

  *Verified locally before the GPU run:* `--allow-cpu` exercises the whole
  probe path off a GPU — it measures nothing, and says so — which caught a
  real bug, `torch.cuda.reset_peak_memory_stats()` being called
  unconditionally and raising on a CPU-only build. Every `torch.cuda` call
  is now guarded. On the 1% sample all four probes complete: real class
  weights (119.2 and 70.1), 54 edge dims, finite losses, and one score per
  real transaction rather than per directed edge.

- **2026-11-09 (stage 3.4.0)** — **The GPU environment check**, the first
  piece of stage 3.4. No model, no data, no training.

  `kaggle_runner.py gpu-check` reports the accelerator (name, memory, compute
  capability, count), the host (Python, CPUs, total and available RAM) and
  the torch / CUDA / cuDNN / PyG versions; installs `torch_geometric` and
  attempts `pyg-lib`, `torch-scatter`, `torch-sparse` and `torch-cluster`
  from the wheel index matching the image, reporting each as installed or
  failed **without aborting** since none is required; then runs a GINEConv
  forward and backward **at the real feature widths** — 3 layers, hidden 64,
  54 edge features plus the direction flag, 6 node features, every edge also
  inserted reversed — at 50k, 200k and 500k edges, reporting seconds and
  peak GPU memory. An OOM is caught and reported as a result. Writes
  `gpu_check.{md,json}` and exits 2 without a usable GPU.

  Using the project's real widths rather than round numbers means the check
  exercises the shape 3.4 will actually run, and the probe sizes are what
  3.4.1 will size fanouts and chunking against — measurements rather than
  guesses.

  Verified locally on CPU end to end (torch 2.7.1, PyG 2.8.1): the stack
  builds, forward and backward run at 20k and 50k edges, the report is
  written, and it exits 2 for want of a GPU, which is the designed
  behaviour.

  *Carried forward, at the user's correction:* stage 3.3's 15.9 GB peak is
  **XGBoost's pooled training matrix** and says nothing about the GNN, which
  processes one window at a time under neighbour sampling. The GNN's memory
  is measured in 3.4.0 and 3.4.3, not inherited. torch and torch_geometric
  stay out of `requirements.txt` until 3.4.2, when the model exists.

  Stage 3.4 now has its own README section with the frozen design and the
  six sub-stages, and `run_on_kaggle.md` has §5h — the only section in that
  document that needs **Accelerator: GPU**.

- **2026-11-09** — **Stage 3.3 is complete**, and the oracle ceiling turns
  out not to be a ceiling.

  The full run went through on Kaggle CPU at commit `de48b35`: 20 boosters,
  **exit 0, no tau at a grid edge**. Evidence committed under
  `docs/reference/`.

  *Results (test, `all`, mean over five seeds).* The full model reaches
  **0.654 at L = 24** and **0.679 at L = 48** — 0.796 and 0.760 of the
  oracle. The no-format variant reaches 0.479 and 0.532. **Selected by the
  frozen protocol: the full model at L = 48**, on validation recall 0.531
  against 0.385.

  *The selection is honest but thin.* That validation margin, about 3.8 of
  26 campaigns, came to roughly **1.4 of 56** on test — 1–3 campaigns
  gained and 0–1 lost per seed, at 0.0h earliness on the campaigns both
  detect — and it costs **3× the false alarms**, 10.5 a day against 3.8. So
  both lookbacks go into 3.4 rather than L = 48 being treated as settled.

  *The format ablation is settled.* The direct paired earliness CI includes
  zero in **all ten** test cells, so the withdrawn 15-hour claim is retired
  for good. What format buys is **coverage**: 6–14 test campaigns per seed
  against 0–1 the other way. It also buys stability — no-format's seed sd
  is **0.052** against **0.009**, five times noisier. Since the
  format-laundering link is a generator artefact, the no-format column is
  the one expected to transfer.

  **The oracle ceiling is not strict, and the README said it was.** At
  L = 48 on stress the full model detects **campaign 295 in all five
  seeds** and the oracle does not. The oracle's own diagnosis explains it:
  295 is the single `purity_failure` in the entire full-data run — its own
  laundering edges connect 5 accounts, well past the 3 required, but
  keeping *every* laundering edge merges it with 9 other campaigns into a
  13-account cluster at purity **0.4615**, just under 0.5. A model at
  tau ≈ 0.99 keeps a sparser edge set, drops part of that bridge, and
  matches. The oracle is therefore the best achievable *from the laundering
  edges*, not the best achievable, and every "no detector can beat these"
  claim is corrected in the README, in `scorers.py`, in
  `recall_relative_to_ceiling` and in the runner's printed and written
  reports. The ratio **can exceed 1 and is never clamped**;
  `tests/test_evaluation.py` pins the case and fails if it is. Worth
  noting what the evidence does *not* show: a model firing on background
  edges to bridge a disconnected campaign is possible in principle but did
  not happen here — a cluster only loses purity by gaining accounts, so
  295 was won back by keeping fewer edges, not more.

  *Cost, measured against the estimate:* train 1.45 h (estimated ~1.5 h),
  evaluate 1.59 h (~2 h), about **3 hours** end to end. Peak memory at
  L = 24 was 9.9 GB, exactly as estimated; at L = 48 it was **15.9 GB**
  against an estimate of 12–13, because scaling by row count under-counts
  the `np.vstack` transient where the pooled and per-window matrices
  coexist. Still comfortable in a 30 GB CPU notebook, and conclusively too
  large for a GPU notebook's 13–16 GB. *(Corrected 2026-11-10: wrong —
  that is the GPU's device memory, not the notebook's 33.7 GB of system
  RAM.)*

  *One reported detail did not hold up.* The run notes said the no-format
  variant also beat the oracle once on test at L = 48 (seeds 3, 4). It does
  not: its `all` and `oracle_reachable` detections are equal in every seed
  there (28/28, 29/29, 32/32, 29/29, 31/31). What those two seeds show is
  no-format detecting one campaign the **full model** misses — 225 and 207
  — which is in the `format_ablation` rows, not `model_vs_oracle`. The
  ceiling is beaten in exactly one place: `xgboost_centralized`, L = 48,
  stress, campaign 295.

  *Tests:* backend 284 passed, 25 skipped.

- **2026-10-19** — **Quick check 2 recorded, and the earliness framing
  corrected.**

  At commit `6a6d1be` the extended grid did its job: the run **exited 0**
  with both taus interior. The full model chose **0.99** (validation F1
  0.543, falling to 0.490 at 0.995) and the no-format variant **0.995**
  (F1 0.343, falling to 0.205 at 0.999) — maxima rather than boundaries, so
  the near-1 tail reaches past where the decision happens and the grid
  question is settled.

  *Provisional, one seed, L = 24 only:* the full model reaches test recall
  **0.661 (0.804 of the ceiling)** at 3.4 false alarms a day; the no-format
  variant **0.464 (0.565 of the ceiling)** at 1.7. The ablated model's
  recall moved from quick check 1's 0.500 because its tau moved from 0.99
  to 0.995; the full model's tau did not move, so its row is unchanged.

  **The earliness claim is withdrawn.** The previous entry said dropping
  payment format cost 15 hours of earliness. That came from comparing each
  variant against the **oracle** separately — 0.0h for the full model,
  −15h for the ablated one — and treating the gap between those two numbers
  as the ablation's effect. They are taken over different shared sets of
  campaigns, so their difference is not a quantity. The criterion is the
  **direct** paired ablation, no-format against the full model, and it is
  **−3.0h with a 95% CI of [−12, 0]**: the interval includes zero, so no
  earliness difference is demonstrated. What the ablation clearly costs is
  **coverage** — 37 test campaigns against 26, eleven fewer. The five-seed
  run over both lookbacks decides it.

  *Rule 3 of the paired method, now frozen:* compare two configurations
  **directly against each other**; never infer an A-versus-B difference
  from `A vs C` and `B vs C`. Pairing fixes the denominator for one
  comparison; it does not make two comparisons subtractable. This is the
  same error as subtracting two medians (2026-10-17) one level up — the
  second time the project has produced a claim by putting two numbers on
  different scales side by side — so it is written into the method with a
  test that fails on the shortcut: two reference comparisons implying a
  25-hour difference where the direct one is a tie.

  *Tests:* backend 283 passed, 25 skipped.

- **2026-10-18 (later)** — **The model tau grid gets a near-1 tail, and the
  first full-data numbers are in — provisionally.**

  The quick check (L = 24, seed 0, both variants) ran cleanly and
  `evaluate` **exited 3**, as designed: the sweep chose **tau = 0.99, the
  top of the grid**. The reason is visible in the curve — F1 rises from
  **0.128 at 0.95 to 0.543 at 0.99** — and in the weights: at 495–3,461 per
  window, a class-weighted model's scores pile up against 1, and a linear
  grid cannot resolve the region where the decision happens. The grid is
  therefore extended to `…, 0.99, 0.995, 0.999, 0.9995, 0.9999`, halving
  the remaining distance to 1 at each step so the resolution sits in
  `1 − tau`. Same grid for every model and condition. The edge check
  stands: a chosen tau of 0.9999 exits 3 again, and at that point the thing
  to examine would be the class weighting, not the grid.

  *Provisional results, recorded as such:* the full model reaches test
  recall **0.661 (0.804 of the ceiling)** at 3.4 false alarms a day and is
  exactly as early as the oracle on the campaigns both detect; the
  no-format variant reaches **0.500 (0.609 of the ceiling)** and is 15
  hours later than the oracle, 95% CI [−27, −6]. *(Corrected 2026-10-19:
  this entry went on to call that the first non-zero paired earliness
  result and to read it as the ablation's effect. It is neither. The
  ablation's effect is the **direct** comparison between the two variants,
  which is −3.0h with CI [−12, 0] — nothing demonstrated. Comparing each
  variant against the oracle separately and subtracting is invalid, because
  the two comparisons are over different shared sets; see rule 3 of the
  paired method and the 2026-10-19 entry.)* One seed and an edge tau, so
  neither reading is a result.

  *Runtime and memory, now measured rather than estimated:* **215s per
  booster** at L = 24 (the earlier guess of ~90s was scaled from the sample
  and wrong) and **~3.5 min per model** to evaluate, with the training
  matrix peaking at **9.9 GB** against the ~7 GB first written here —
  `np.vstack` holds the per-window matrices and the pooled copy at once.
  Scaling the parts that grow (1.36× the rows, 1.97× the evaluation edges)
  puts L = 48 at ~290s per booster and ~12–13 GB. The full 20-booster run
  is **~3.5–4 hours**: training ~1.5 h, evaluation ~2 h. Inside a CPU
  notebook's 30 GB and 12 hours; the L = 48 memory would *not* be
  comfortable on a GPU notebook's 13–16 GB, which is one more reason this
  stage is CPU. *(Corrected 2026-11-10: that comparison was wrong — a GPU
  notebook has 33.7 GB of system RAM, measured. The CPU case rests on the
  bottleneck being pandas and scipy, and on 3.4 needing the quota.)*

  Also fixed `run_on_kaggle.md` §6, which still said `train` and `evaluate`
  were not implemented.

- **2026-10-18** — **Three decisions taken on stage 3.3 before the
  full-data run, and the sample numbers demoted to development checks.**

  *A separate tau grid for model scorers, frozen:* `0.05, 0.1, 0.2, 0.3,
  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99`, used by every model — XGBoost
  now, the GNN and the federated conditions later — while the oracle and
  random scorers keep `DEFAULT_TAU_GRID`. One grid for every model means a
  comparison between models or conditions is never a comparison of how far
  each was allowed to search. `select_tau` now records **the seconds each
  tau cost**, because clustering scales with the edges kept and the low
  thresholds are the expensive ones; the score cache is warmed before the
  sweep so those seconds are clustering alone. If a chosen tau still lands
  on an edge of the grid, `evaluate` writes every report and **exits 3**
  rather than extending the grid silently.

  *A payment-format ablation, frozen:* `xgboost_centralized_no_format`
  drops every `fmt_*` column (including `fmt___other__`) and is identical in
  all else, built as a column slice of the same matrix so the two variants
  see byte-identical windows, labels and weights. Reported alongside the
  full model, and **every later model is reported with and without payment
  format**. The IBM generator links laundering strongly to format, which is
  an allowed input but a shortcut-prone one: a result resting on it would
  say more about the generator than about campaign discovery.

  *Five seeds:* every configuration is trained at seeds 0–4 and reported
  per seed and as mean [min, max], so a difference is read against seed
  noise. With both variants and both lookbacks that is 20 boosters.

  *Sample numbers relabelled.* Everything measured on the 1% sample is now
  presented as a **development check, not a result**, in the stage 3.3
  section, the limitations and the earlier changelog entry: the thinned
  background makes separation artificially easy and false alarms
  artificially scarce, so none of it is a bar for stage 3.4.

  *Runtime, for the record:* the full run is **CPU, about 3.5–4.5 hours**.
  A GPU would shorten only the ~35 minutes of training — featurisation
  (pandas) and clustering (scipy) dominate and neither uses one — and stage
  3.4 needs the GPU quota. Estimated from rates measured on this dataset:
  0.7s per 100k edges to featurise and ~14s per million edges clustered,
  against the pooled edge counts (6.71M training rows at L = 24, 9.11M at
  L = 48). §5g of `run_on_kaggle.md` has the breakdown.

  *Tests:* backend 281 passed, 25 skipped. The decisive new one is that
  **caching the scorer cannot change the tau curve** — cached and uncached
  curves compared row by row, with the cached run doing one scoring call
  per window instead of one per (window, tau) — plus that the ablation drops
  every format column including `fmt___other__` and nothing else, and
  changes only the columns.

- **2026-10-17 (stage 3.3)** — **The XGBoost baseline is implemented and
  validated on the real 1% sample.** Ready for the full-data Kaggle run;
  nothing here is a full-data result.

  `models/xgb_baseline.py` trains one booster per lookback over every
  rule-3-valid training detection time pooled into one matrix, with the
  class weight `n_negative / n_positive` applied per window as a per-row
  weight. Hyperparameters are frozen and identical for both lookbacks. The
  model emits probabilities and chooses no threshold: tau comes from
  `evaluation.select_tau` on validation only, and `TrainedBaseline.scorer()`
  plugs into the same `evaluate_split` that scored the oracle, so there is
  no model-specific evaluation path to keep honest.

  *As a development check on the 1% sample* — not a result, and not a bar
  for anything: test recall 0.768 at L = 24 and 0.821 at L = 48, which is
  0.935 and 0.920 of the oracle's on the same split. Paired against the
  oracle it is **0.0h, CI [0.0, 0.0]** on every split and both lookbacks,
  and it never finds a campaign the oracle misses, so what differs is
  coverage rather than lateness; whether that holds on the full data is
  open. Gain is dominated by payment format (`fmt_ACH` 46%, formats
  together ~87%) — an allowed feature, but a shortcut-prone one, which is
  what the 2026-10-18 ablation decision answers. *(Corrected 2026-10-17:
  this entry originally called these numbers a high bar for 3.4. The 1%
  background makes separation artificially easy, so they are a check that
  the code runs, nothing more.)*

  *Reported, not changed:* on the sample the chosen tau landed on
  `DEFAULT_TAU_GRID`'s floor of 0.5, and a wider sweep put the validation
  optimum at 0.3 (F1 0.821 vs 0.803). The grid was designed for scorers
  whose scores are 0 or 1. It is an implementation default rather than a
  frozen definition, so it is left alone; `evaluate` now warns when the
  chosen tau sits at an edge, `--tau-grid` can widen it, and the options are
  written up for a decision **before** the full-data run.

  *New:* `kaggle_runner.py train` and `evaluate` (no longer stubs),
  `scripts/run_xgb_baseline.py` for local development, `xgboost>=2.1` in
  requirements.txt, and `tests/test_xgb_baseline.py` (13 tests, backend 269
  passed / 25 skipped), with the rule-3 filter and the score alignment each
  proven able to fail.

- **2026-10-17 (later)** — **Full-data results recorded, and three
  refinements before stage 3.3.** Kaggle run at commit `eacb6d3`; the
  evidence files are committed under `docs/reference/`, and the real 1%
  sample is in `backend/data/sample/` (gitignored, as always).

  *The earliness question is answered.* The paired comparison ran on the
  full data: **0.0h difference with a 95% CI of [0.0, 0.0] on every split**,
  for absolute, normalized and `frac_observed` lead time alike. L = 48 costs
  no earliness and detects 4 test campaigns L = 24 misses (187, 194, 238,
  245) while missing none. The withdrawn claim was not merely unsupported;
  it was wrong, and the paired method is what showed that.

  *Also measured:* 24 of 24 pinned regression numbers pass again, and the
  five visibility digests are byte-identical in a **third** independent
  session. Oracle misses: 36 sparsity + 2 `late_only` + 0 `purity_failure`
  at L = 24, and 30 + 1 + 1 at L = 48. **Decision: the matching rule stays
  frozen** — one purity failure in 70 misses means the 50% condition is not
  what the ceiling is made of.

  *`temporal_sparsity` is now two reasons*, `structurally_disconnected` and
  `slow`, with precedence `purity_failure` → `late_only` →
  `structurally_disconnected` → `slow`. The first pools **every** own edge
  before the deadline, with no window at all, and still cannot connect 3
  accounts, so no lookback can ever detect it; the second can, and a longer
  lookback may reach it. They were separated because they imply opposite
  actions. On the sample's test split the 8 pooled sparsity misses at L = 24
  split 5 / 3, all 10 disconnected cases across both lookbacks are BIPARTITE
  and fragmentable, the count does not move with L, and the 4 campaigns
  L = 48 rescues are exactly the 3 `slow` ones plus 1 `late_only` — the
  mechanism behind its recall, now named rather than inferred.

  *A fourth test subset, `oracle_reachable`*, and
  `recall_relative_to_ceiling`, reported for every scorer and model
  alongside `all` / `fully_observed` / `no_shared_accounts`. `summarize`
  **raises** if asked for the subset without the oracle's detected ids,
  rather than dividing by the wrong denominator. Phase 5 will report the
  visibility analysis on both `all` and `oracle_reachable`, because the
  structurally undetectable campaigns sit in the fragmentable group and
  would add the same dead weight to all four conditions, compressing the
  effect the experiment exists to measure.

  *The false-alarm summary is corrected.* It read "best overlap 0.667
  (below the 0.5 the rule requires)", which contradicts itself. Purity and
  coverage are now checked against their own thresholds and the binding one
  named: on the sample, 21 of 22 clusters fail **coverage**, median purity
  0.667 against 0.50 and median coverage 2 accounts against 3. The oracle's
  false alarms are mostly fragments of real campaigns too small to claim,
  which the three-account floor creates by design.

  *Tests:* backend 256 passed, 25 skipped (`test_oracle_diagnose.py` 14,
  `test_evaluation.py` 36). All three refinements were also run end to end
  through `oracle-diagnose` and `evaluate-scorers` on the real sample.

- **2026-10-17** — **Lookback and condition comparisons are now paired, and
  the earliness claim about L = 48 is withdrawn as untested.**

  The README claimed that L = 48 buys recall but loses earliness, 74.7h →
  72.1h. That was not supported by the measurement. The two medians are taken
  over *different sets of campaigns*: L = 48 additionally detects the harder,
  slower campaigns, and those are detected late, so its median falls even if
  no campaign was detected later than it would have been under L = 24. The
  bootstrap CIs also overlap almost entirely. The claim is removed. The
  recall numbers stand; the earliness side now reads **untested**, pending
  the paired comparison on the full data.

  *The paired method is frozen* as the way two configurations are ever
  compared, lookbacks now and the four conditions in Phase 4–5. Earliness is
  the per-campaign lead-time difference over the campaigns **both** detect,
  as a median with a seeded bootstrap CI and the per-side earlier/tied
  counts, sign convention `b − a` so positive means `b` was earlier; the same
  pairing is applied to `normalized_lead_time` and `frac_observed`. Coverage
  is reported **separately**, as how many campaigns each side detects that
  the other misses, named by id. Two medians of `median_lead_time_h` are
  never subtracted.

  *New:* `evaluation.paired_lead_time_comparison` and
  `evaluation.describe_paired_comparison`, the latter refusing to state a
  direction when the CI includes zero. `oracle-diagnose` gained a paired
  section, printed and written to `oracle_diagnose_paired.csv` and into
  `oracle_diagnose.{md,json}`, so it lands with the next Kaggle run. The
  ceiling table now says in words that its medians are each over that row's
  own campaigns and are not comparable across rows.

  *Tests:* `tests/test_paired_comparison.py`, 14 tests, backend now 248
  passing (25 full-data tests skipped locally). The decisive one is built so
  the unpaired medians are *wrong*: B detects everything A detects at exactly
  the same time, plus one campaign only detectable late, which makes B look
  45 hours worse by medians while the paired difference is 0.0h with one
  extra detection. Proven able to fail: making the comparison unpaired
  (`a.detected or b.detected`) fails 4 of the 14. One test runs the real
  oracle through `evaluate_split` at both lookbacks on a built world where a
  campaign's 30-hour gaps put it out of reach of any 24h window, and checks
  it appears as coverage rather than as a lead-time change. The paired
  section was also run end to end through the runner on the local demo data.

- **2026-10-15** — **The evaluation is validated on the full data** (oracle
  far above random on every split), and three follow-ups landed before stage
  3.3 begins.

  *The oracle ceiling is now a measured number, and it is not 1.0.* Recorded
  in its own section: test recall 0.821 at L = 24 and 0.893 at L = 48, with
  median lead times of 74.7h and 72.1h and median `frac_observed` 0.25. Two
  readings worth keeping: 11–18% of evaluable test campaigns are
  **unreachable** under the frozen matching rule whatever the model, so the
  honest denominator for a model's recall is the oracle rather than 1.0; and
  L = 48 raises test recall from 0.821 to 0.893. *(Corrected 2026-10-17: this
  entry originally read that L = 48 buys recall while losing earliness,
  74.7h → 72.1h. That comparison was invalid — see the 2026-10-17 entry.)*

  *`oracle-diagnose`* explains every missed campaign as
  `temporal_sparsity`, `purity_failure`, `late_only` or `unexplained`, with
  the evidence — largest connected fragment reached, transaction gaps against
  the lookback, best cluster purity, and what the campaign merged with —
  summarised by reason, `base_type` and group, and also characterises the
  oracle's false-alarm clusters. Writing the tests surfaced a precedence
  error in my first version: a campaign with transactions an hour apart whose
  deadline falls before the first usable window was being reported as
  *temporal sparsity*, which would send a reader looking for a longer
  lookback when the real cause is censoring. Precedence is now
  purity → late → sparsity, so the most specific explanation wins.

  *Alert-budget metrics*, frozen: recall and median lead time at 10, 50 and
  100 false alarms per day, taking the lowest tau that stays inside each
  budget on validation, plus the recall-vs-alarm-rate curve for plotting.
  Prompted by random reaching 0.27 recall at L = 48 on ~47,000 alarms a day.
  An unreachable budget is reported as unreachable with the cheapest rate the
  grid achieves — verified on the demo data, where random cannot get below
  48.4/day and the 10/day budget correctly declines to pretend otherwise.

  *The tau tie-break is now deterministic*: equal F1 → higher precision →
  fewer false alarms → higher tau, tested to be independent of grid order.
  This reverses the previous "prefer the lower tau", so the oracle now
  selects the top of the grid; one existing test was updated accordingly.

  Backend: 234 passed, 25 skipped.
- **2026-10-11** — Added `docs/PROJECT_OVERVIEW.md`: a readable account of
  the whole project for someone arriving fresh — the problem, the research
  question and its positioning, the data, the frozen design in condensed
  form, what has been built, what is verified versus assumed, the four bugs
  worth writing up, and an improvement roadmap separating immediate
  unblocking actions from research and engineering suggestions. It is a
  summary that defers to this README on every point of fact, and it states
  plainly in two places that the project has no results yet, since a reader
  skimming a repository this size would otherwise assume it does.
- **2026-10-11** — Full-data validation of the evaluation, ahead of stage
  3.3. Added `kaggle_runner.py evaluate-scorers`: oracle and random at both
  candidate lookbacks over the val, test and stress horizons, reporting
  recall, precision, false alarms per day, other-split hits, ambiguous
  clusters and all three lead-time forms with 95% bootstrap CIs, plus the
  oracle broken down by `base_type` and by topology group, into
  `evaluate_scorers.{md,csv,json}` with separate tau-curve and breakdown
  CSVs. Tau is chosen **on validation only** by campaign-level F1 with the
  curve saved beside it — the same procedure stage 3.3 will use for a real
  model, so random is tuned exactly as a model would be rather than handed a
  convenient threshold. Added `select_tau` and `campaign_f1` to the
  evaluation module (the latter so the sweep does not pay for a 1000-sample
  bootstrap at every candidate), and a `DEFAULT_TAU_GRID` weighted toward
  high thresholds because a low tau keeps most of a million-edge window.
  Added `kaggle_runner.py make-sample`, with `make_sample.py` refactored
  around a reusable `build_sample` so the script and the subcommand share
  one implementation. Two performance defects fixed before they reached full
  data: `evaluate_split` was doing an O(campaigns) dataframe filter per hit
  to look up a deadline and another per campaign when building outcomes, both
  now dictionary and row lookups; and `extract_clusters` gained a scipy
  `connected_components` fast path with the pure-Python union-find kept as a
  fallback and a test proving the two partition identically. Recorded the
  **oracle ceiling** as its own README section — the best lead time
  achievable under the frozen matching rule, currently filled with the
  synthetic figures and marked pending the full-data run. `run_on_kaggle.md`
  gained an explicit run order table and sections §5d and §5e. Backend: 208
  passed, 25 skipped.
- **2026-10-10** — **Stage 3.2: the evaluation module, plus a
  `features-smoke` correctness fix.**

  *The fix first.* The first full-data `features-smoke` run reported
  Sept 2 12:00 at L = 48 as the largest candidate window, but that
  detection time violates rule 3 — its lookback starts before `DATA_START`,
  so the window was silently clamped, which is also why its L = 48 and
  L = 72 rows came out identical. `features-smoke` now (a) skips rule-3-
  invalid training probes and prints why, naming the first valid time at
  that lookback, (b) performs a cheap full scan of **every** usable
  detection time per (lookback, split) — counts only, located by binary
  search over sorted timestamps, laundering via a prefix sum — reporting
  min/median/max edges and accounts and the time of the maximum, and (c)
  computes the GPU estimate from the true maximum valid window, split into
  an inference figure (any split) and a training figure, and reports
  candidates separately from the excluded lookback. The scan is tested to
  agree exactly with the real `window_slice` and to include no invalid
  training time.

  *The module.* Added `fraudcamp/evaluation/` — `clusters.py` (extraction,
  the frozen >= 50% / >= 3 matching rule, and classification into
  hit / other-split hit / ambiguous / false alarm), `evaluate.py` (the
  horizon loop, first-detection-before-deadline logic, the three lead-time
  forms, per-subset metrics, a seeded percentile bootstrap, and group /
  base_type breakdowns) and `scorers.py` (oracle and random). 28 tests,
  with the end-to-end cases hand-built so the expected detection time and
  lead time are checkable on paper. `AMBIGUOUS_MIN_FRACTION = 0.5` makes
  the "mostly" in the ambiguous rule explicit.

  *Validation.* The 1% research sample has never been downloaded locally,
  so there was nothing to run the scorers on; added
  `scripts/make_demo_data.py` (seeded, synthetic, gitignored, clearly not
  the research sample) and `scripts/run_sanity_scorers.py`. The oracle
  reaches recall 1.00 / precision 1.00 / zero false alarms at a median
  `frac_observed` of 0.33, and random reaches 0.00 with ~950 false alarms a
  day — the behaviour the design predicts. Getting there required fixing
  the *generator*, not the rule: its first version gave campaign accounts no
  background traffic, so random scored 0.57–0.71 recall by keeping pure
  fragments of campaigns. Full numbers and that diagnosis are in
  "Sanity scorer results".

  Also froze the **stage 3.4 training regime**: both candidate lookbacks
  train with neighbour sampling so the L comparison is not confounded by
  different training methods; inference scores full windows, chunked if
  needed. Backend: 203 passed, 25 skipped.
- **2026-10-10** — Two further decisions frozen. **Stress evaluation
  horizon** set to `(2022-09-08 00:00, CUTOFF]`, capped per campaign by its
  deadline, with other-split hits applying as in rule 1; it abuts the
  validation horizon without overlapping and sits strictly inside the test
  horizon, both pinned by tests. The previous "raises rather than guess"
  behaviour and its Known-limitations entry are gone, replaced by nine tests
  covering the horizon, its boundaries, its relationship to the other two,
  and that training still has no horizon. **Lookback candidates reduced to
  L ∈ {24, 48}**, dropping L = 72 for three reasons recorded in the Decision
  log: only 5 heavily-overlapping full-history training windows under rule 3,
  ~18.6 GB estimated full-batch GINE memory against 16 GB GPUs, and a 3-day
  window being large relative to a 10-day dataset. `constants` now separates
  `LOOKBACKS_H` (candidates) from `EXCLUDED_LOOKBACKS_H` and
  `MEASURED_LOOKBACKS_H`, so `features-smoke` still measures L = 72 once and
  labels it `[EXCLUDED]`, and reports the largest **candidate** window's GPU
  estimate separately from the excluded one. Tests that assert "no lookback
  we ever build reaches past `t`" now cover the measured set, not just the
  candidates. Both decisions verified by deliberate violation (a stress
  horizon starting Sept 6 fails 4 tests; leaving L = 72 as a candidate fails
  3). Backend: 173 passed, 25 skipped.
- **2026-10-10** — Stage 3.1 **approved**. Recorded the accepted feature
  deviations and interpretations in the Decision log, and froze three new
  Phase 3 evaluation rules (documented now, mostly implemented in stage
  3.2): other-split hits counted separately from hits and false alarms; the
  validation detection horizon extended to `(Sept 5, Sept 8]`, overlapping
  the test horizon by 8 times with only validation labels informing tau; and
  training restricted to detection times whose full lookback lies inside the
  data. The time logic for all three went into `windows.py` now
  (`hit_kind`, `evaluation_horizon`/`evaluation_detection_times`,
  `has_full_lookback`/`training_detection_times`/
  `usable_training_times_per_lookback`), with the clustering and matching
  that feed rule 1 left to stage 3.2. 35 new tests, each rule verified by
  deliberate violation (see Testing). **Rule 3 has a real cost worth
  noting: L = 72 leaves only 5 usable training detection times out of 16,
  versus 13 at L = 24** — `features-smoke` now reports these counts per
  lookback, in its console output and both report files. One gap surfaced
  and is deliberately left open rather than guessed: rule 1 mentions stress
  evaluation but no stress horizon was specified, so
  `evaluation_horizon("stress")` raises with a pointer to Known limitations.
  Backend: 164 passed, 25 skipped.
- **2026-10-09** — Stage 3.1 review follow-ups, plus a `features-smoke`
  runner subcommand. Proved the leakage tests can actually fail by injecting
  three leaks in turn — the label into the edge features, an `ts <= t`
  off-by-one in the window bound, and `From Bank` as a numeric feature — and
  confirming which tests break in each case (see Testing). The off-by-one
  exposed a real gap: the feature-level future test used a detection time
  with no transaction exactly on it, so only the window test caught it.
  Added `test_a_transaction_stamped_exactly_at_t_is_treated_as_future`,
  which anchors on a `t` a transaction really sits on, and verified it fails
  under that same leak. Backend tests: 129 passed.
  Added `kaggle_runner.py features-smoke`, which builds windows at
  L = 24/48/72 at five detection times (weekday and weekend from train and
  test, plus validation's single weekday — validation spans only Mon Sept 5
  → Tue Sept 6, so it has no weekend time) and reports edges, accounts,
  laundering edges and rate, class weight, timings, memory and tensor
  footprints, then estimates full-batch GPU memory for a 3-layer GINEConv
  with hidden 64. It writes `features_smoke.{csv,json,md}`, fits its spec on
  a single training window on purpose, and hand-rolls its markdown table
  because `pandas.to_markdown` needs `tabulate`, which this project does not
  depend on.
  **The estimate already matters:** full-batch at L = 48/72 on the busiest
  days comes to ~17–19 GB against Kaggle's 16 GB GPUs, so stage 3.4 will
  likely need neighbour sampling for the longer lookbacks — recorded under
  Phase 3 definitions, to be replaced by real figures from the run.
- **2026-10-09** — Installed the verified deterministic artifacts locally.
  The files previously in `backend/artifacts/` were from a pre-fix run —
  their visibility digests matched none of the references — so they were
  removed (`.gitkeep` kept) and replaced from the Kaggle archive. The
  archive was extracted to a temporary folder and checked **before**
  installation against both `hashes_session1.json` and the confirmed
  reference digests, which also agree with each other; the installed copies
  were re-hashed afterwards and are byte-identical. Validated through the
  API's own `ArtifactStore`: 370 campaigns, 258 eval_ok, 152 reassignable,
  3,209 campaign transactions, 5 visibility seeds × 14,384 rows, exactly 2
  unfragmentable, splits 100/26/56/76, and no nested-folder false positive.
  Renamed the committed evidence to
  `docs/reference/visibility_hashes_session1.json` (updating the three
  README references), deleted the downloaded archive, and added `*.zip` to
  `.gitignore` so an archive sitting in the project root can never be
  committed.
- **2026-10-09** — **Stage 3.1: detection windows, features, leakage
  tests.** Added `windows.py` (the aligned 6-hourly detection grid over
  `(DATA_START, CUTOFF]` = 40 times, detection-time splits, and
  `window_slice` as the single place a window is ever cut, with a strict
  `ts < t` upper bound) and `features.py` (32 edge features, 6 window-local
  node features, training-only category vocabularies with an `__other__`
  column, and a mean/std scaler over the continuous columns that refuses to
  be fitted on anything but training windows). `WindowFeatures` carries
  node/edge matrices, `edge_index`, labels, campaign ids and timestamps,
  with `xgboost_matrix()` for the non-graph baseline and `class_weight()`
  for the imbalance. New detection-time constants (`DATA_START`,
  `LOOKBACKS_H`, `DETECTION_{TRAIN,VAL,TEST}_END`) are additive — no frozen
  Phase 2 definition changed — and are written up in a new "Phase 3
  definitions" README section. 43 new tests: the leakage properties are
  invariance checks (delete the future, poison the future with absurd
  values, flip every label, relabel every account, renumber every bank) plus
  the grid, split boundaries, lookback nesting and unseen-category handling.
  Two implementation notes: one-hot encoding uses a dict map rather than
  `pd.Categorical`, whose out-of-category behaviour is deprecated and is
  exactly the `__other__` path; and matrices are stored float32 after
  float64 arithmetic, which halves memory without affecting
  bit-reproducibility. Measured on a synthetic 400k-transaction window:
  32 + 6 dims, a 44-column XGBoost matrix, ~8s to build, and a class weight
  of 1,014 at a 0.098% positive rate. Backend: 128 passed, 25 skipped.
- **2026-10-09** — **Determinism confirmed; unfragmentable count pinned at
  2.** Two independent Kaggle sessions and three separate pipeline processes
  at commit `55b645e` (Python 3.13.15, pandas 2.3.3) produced byte-identical
  `visibility/seed_*.parquet` files and the same unfragmentable count, so
  the metric moved from pending to pinned — the regression table is now 24
  of 24 checked, with nothing pending. Recorded the five reference SHA-256
  digests in Verified numbers together with the environment they are valid
  for, and committed `docs/reference/visibility_hashes_session1.json` as the
  machine-readable copy; the README digests were verified against that file
  programmatically rather than by eye, which caught one transcription slip.
  Removed the stale-artifacts warning from Known limitations (artifacts
  regenerated and reinstalled), replacing it with the narrower caveat that
  digests are only comparable within one environment, since a different
  pyarrow could change parquet bytes without changing the data. Tagged
  `v0.2-verified`.
- **2026-10-08** — **Fixed a reproducibility defect in the visibility
  search.** A campaign's accounts were held in a Python `set`, and
  per-process string-hash randomisation made set iteration order vary, so
  the seeded search followed a different trajectory in every process:
  identical seed, different account-to-institution map. It surfaced as two
  Kaggle runs disagreeing on the unfragmentable count (3 vs 2) and was
  reproduced locally by varying `PYTHONHASHSEED`. Fixed by sorting at every
  point where order reaches the RNG — `visibility.build_global_assignment`
  (accounts sorted before the locked/free split), `visibility.assign_campaign`
  and `visibility._local_search_free` (free list sorted), plus
  `pipeline._edges_and_accounts` and `campaigns.campaign_accounts` now
  return sorted **tuples** instead of sets, with `PipelineResult` typed
  accordingly and `groupby(sort=True)` made explicit. An audit of the rest
  of the backend found the other order-sensitive spots already pinned in
  earlier work (`institutions.bank_volume` tie-breaks by bank id,
  `pipeline.ordered_reassignable` sorts by `[start, campaign_id]`,
  `export.build_visibility_table` sorts campaigns and accounts,
  `find_nested_artifacts` sorts `iterdir`) and the remaining set usage to be
  membership tests or `len()` only, where order cannot leak. No frozen
  definition changed: same algorithm, targets, iteration count and seeds —
  only enumeration order is pinned. Added `tests/test_determinism.py` (10
  tests) which runs the build in subprocesses under `PYTHONHASHSEED=1/2/3`
  and requires byte-identical results; confirmed it *fails* when the bug is
  reintroduced. Added "Unfragmentable campaigns" as a **pending** regression
  metric — reported every run, never failed — since the notebook's
  exploratory 2 came from a procedure without conflict resolution and the
  pre-fix runs disagreed; `regression.py` now supports pending entries and
  the full-data suite skips them with the actual value in the skip reason.
  `kaggle_runner pipeline` now prints a SHA-256 for each
  `visibility/seed_*.parquet` and writes `visibility_hashes.json`, so two
  independent sessions can be compared directly; verified locally that
  parquet bytes are stable across processes, which is what makes that
  comparison meaningful. Added §5b to `run_on_kaggle.md` with the
  two-session regeneration and comparison procedure. Backend: 85 passed, 25
  skipped. **The local `visibility/seed_*.parquet` files are now stale** and
  must be regenerated — see Known limitations.
- **2026-10-08** — **Phase 2 verified on the full dataset.** A Kaggle
  `verify` run reproduced all 23 regression numbers (pytest exit 0, pandas
  2.3.3, 39.3s, 3,207 MB peak); the report is in
  `docs/reference/verify_report.md`. Phase 2's agreement with the reference
  notebook is now a measured fact rather than an assumption, which clears
  the stage 3.0 gate.
  Installed the exported artifacts locally: they had been unpacked into
  `backend/artifacts/artifacts/`, one directory too deep, which is why
  `/api/health` reported every core artifact missing. Moved the contents up,
  restored the tracked `backend/artifacts/.gitkeep` (the extraction had
  removed it), and validated every file through the API's own
  `ArtifactStore`: 370 campaigns, 258 eval_ok, 152 reassignable, 3,209
  campaign transactions, 14,384 rows per visibility seed file, splits
  100/26/56/76 — all matching the verified numbers.
  Made that failure self-diagnosing: the server now logs a warning at
  startup naming each missing core artifact and the directory it searched,
  and `find_nested_artifacts` detects the extracted-too-deep case and
  reports it in both the log and `/api/health` (new `missing_artifacts`,
  `nested_artifacts_dir` and `hint` fields, mirrored in the frontend's
  `Health` type). Six new API tests cover detection, the health response,
  that `visibility/` and `results/` are never mistaken for an extra nesting
  level, and the all-present and absent-directory cases — backend is now 75
  passed, 24 skipped. Added an "Installing artifacts from Kaggle"
  subsection to How to run with the PowerShell commands, the Explorer
  "Extract All" pitfall and the restart requirement.
- **2026-10-08** — Added `.gitattributes` with `* text=auto eol=lf`, so text
  files are stored and checked out as LF on every platform and a Windows
  clone sees the same bytes as a Linux/Kaggle one. `git add --renormalize .`
  changed **nothing**: the system-wide `core.autocrlf=true` had already been
  normalising content to LF on commit, so the index was LF throughout (an
  audit with `git ls-files --eol` showed 76 LF, 7 empty, 0 binary files).
  What the file actually fixes is *checkout* — 12 working-tree files had
  picked up CRLF, which is where the "LF will be replaced by CRLF" warnings
  came from. Those 12 were restored from the index so the local tree now
  matches a fresh clone: all 89 text files are LF on both sides. Backend
  (69 passed) and frontend (48 passed, lint clean) re-run afterwards, since
  source files were rewritten in place.
- **2026-10-08** — Published to GitHub at
  https://github.com/jerinsaji299-coder/Fraud-campaign-detection and tagged
  `v0.1-data-pipeline`. Initialised the repository (branch `main`), filled
  the gaps in `.gitignore` (`*.env`, `kaggle.json`, `*.pem`, `*.key`,
  `credentials.json`, `Thumbs.db`, `desktop.ini`, `.vscode/`, `.idea/`,
  `*.swp`, `venv/`, `build/`, `dist/`) and changed `backend/data/` to
  `backend/data/*` with `!backend/data/.gitkeep`, because the directory-wide
  rule also hid the placeholder and a fresh clone therefore lost the folder.
  Verified before staging that no secrets exist in the project (the only
  `.env`-shaped file is `frontend/.env.example`, holding just
  `VITE_API_URL`), that nothing over 5 MB would be committed (largest:
  `package-lock.json` at 123 KB), and that the evidence notebook (32.9 KB)
  and `.env.example` are **not** ignored while data, artifacts,
  `node_modules`, `dist` and credential files are. First commit: 95 files,
  14,027 insertions. Added the repository URL and clone command to the top
  of this README, a Version history section, and the real clone URL in
  `backend/scripts/kaggle/run_on_kaggle.md` (it previously carried a
  `<YOUR-USERNAME>/<YOUR-REPO>` placeholder).
- **2026-10-07** — Phase 3 begins. Stage 3.0: Kaggle runner and the means to
  verify Phase 2 on the full data. Added
  `backend/src/fraudcamp/regression.py` (every expected number with the
  notebook cell it came from, plus `compute_actuals`, comparison and table
  formatting) and rewrote `tests/test_fulldata.py` on top of it — the suite
  is now one parametrized test per number (24 instead of 10), so a failure
  names the exact metric, and it can dump the computed numbers to JSON via
  `FRAUDCAMP_REGRESSION_OUT`. Added
  `backend/scripts/kaggle/kaggle_runner.py` (`verify` runs the full-data
  tests then prints an expected-vs-actual PASS/FAIL table and writes
  `verify_report.md`; `pipeline` builds, exports, zips to
  `artifacts.zip` and re-checks the same numbers; `train`/`evaluate` exit
  with a not-implemented message until stages 3.3-3.4) and
  `scripts/kaggle/run_on_kaggle.md` (notebook cells, what to send back,
  expected runtimes, failure modes). Pinned `pandas>=2.2` in
  requirements.txt: the pipeline passes `include_groups=` to
  `groupby.apply`, which older pandas rejects, so an older Kaggle image would
  have failed confusingly. Both runner commands were exercised locally
  against the synthetic fixture, and the comparison table was proven to
  render both all-PASS and FAIL rows, so the first Kaggle run is not also
  the first test of the reporting code. Two things fixed while doing that:
  the regression table is now skipped with an explanation when the run is
  not the full dataset (it previously printed 23 meaningless FAILs on sample
  data), and memory is labelled honestly — a true peak via `resource` on
  Linux, explicitly marked "current, not peak" on Windows. **Note:** the
  full-data tests still have not run anywhere; that is what the Kaggle run
  is for. Updated README: Project status (restructured into Phase 2 / Phase
  3 / later phases), Repository structure, Setup, How to run, Backend
  modules (new Scripts subsection), Testing, Decision log.
- **2026-10-07** — Stage 5: the remaining four pages, completing this phase
  of work. Added `lib/visibility.ts` (a direct mirror of the backend's
  `visibility_of`, the only research metric the frontend computes, and only
  for the user-built Lab scenario), `components/CampaignGraph.tsx`
  (Cytoscape, layout chosen per `base_type` so topology is legible and
  reproducible), `components/Timeline.tsx` (time replay with deadline and
  cutoff markers), `components/VisibilityToy.tsx` (drag or click accounts
  between three bank zones), and the Campaign Detail, Visibility Lab,
  Results and Methodology pages; removed the Stage 4 placeholder page and
  installed `cytoscape`. The Results page deliberately ships as the
  empty-state version: labelled outlines of all five planned figures with
  their how-to-read notes and **no numbers**, since Phase 5 has not run.
  Three things found while building: two `setState`-in-effect patterns in
  Campaign Detail were replaced with derived values (the bank filter now
  resets correctly when a view change moves every account, instead of
  flashing a stale view); and **two of my own test expectations were wrong
  in ways worth recording** — a four-account cycle cannot be pushed below
  0.75 across two banks, and moving a single account out of a six-account
  cycle does not reduce visibility at all, because the bank holding the
  remaining five still touches every transaction. Both are now pinned as
  tests, and the second is used as the teaching point in the Lab. Added 25
  frontend tests (48 total); build and lint clean; every Stage 5 endpoint
  verified over HTTP. The Cytoscape graph is stubbed in tests and has not
  been seen rendering in a browser — see Testing. Updated README: Project
  status, Repository structure, Frontend (pages table, graph layouts, a new
  "where numbers come from" note), Testing.
- **2026-10-06** — Stage 4: frontend scaffold and the first three pages.
  Created `frontend/` with Vite + React + TypeScript, React Router, TanStack
  Query, Tailwind v4, Recharts, lucide-react and Vitest. Added the theme
  system (`src/theme.ts` as the single colour source, chrome tokens in
  `index.css` with a `.dark` override), the API layer (TS types mirroring
  every pydantic model, a fetch wrapper with a typed `ApiError`, and query
  hooks), the app shell (sidebar, top bar with live API status and a
  light/dark toggle, skip link, responsive collapse), reusable components
  (StatCard, ChartCard, EmptyState, ErrorState, Skeleton, GlossaryTerm,
  InstitutionBadge, ConditionLegend, FourStepStory), and the Home, Dataset
  and Campaign Explorer pages. Routes for the Stage 5 pages exist but render
  an explicit "not built yet" placeholder rather than a mock-up. Decisions:
  (1) **Cytoscape.js** chosen over react-force-graph-2d for the Stage 5
  campaign graph, because deterministic layouts make campaign *topology*
  legible and reproducible — the thing the research question is actually
  about; (2) React 19 / Vite 8 / Tailwind 4 kept as shipped by the current
  template instead of pinning back to the spec's React 18 — a tooling
  choice, not a research definition; (3) institution identity is carried
  label-first, because the palette validator confirms eight categorical hues
  cannot clear colour-vision-deficiency separation when any two may sit side
  by side, so `I0`–`I7` labels accompany colour everywhere. Fixed while
  building: Recharts 3's stricter tooltip formatter types; a Windows
  case-only filename collision between `ThemeContext.tsx` and
  `themeContext.ts` (provider renamed to `ThemeProvider.tsx`); and missing
  Testing Library auto-cleanup (it does not self-register when Vitest globals
  are off, so the DOM accumulated across tests and queries found duplicates —
  now handled centrally in `src/test/setup.ts`). Added 23 frontend tests;
  `npm run build` and `npm run lint` are clean. Verified both servers start
  and communicate over HTTP including CORS, but the UI has **not** been
  visually inspected in a browser — see Testing. Updated README: Project
  status, Repository structure, Setup, How to run, Frontend (new full
  section), Testing.
- **2026-10-06** — Stage 3: FastAPI server. Added `backend/app/` —
  `main.py` (`create_app`, CORS for the Vite dev server, routers under
  `/api`, artifacts loaded once at startup), `schemas.py` (pydantic
  response models for every endpoint), `services/artifacts.py`
  (`ArtifactStore` with caching, guarded accessors and JSON-safe record
  conversion), `services/methodology.py` (frozen definitions as structured
  JSON, every value read from `fraudcamp.constants`), and routers for
  meta/campaigns/visibility/results. Two artifact changes made to support
  it, with `campaign_transactions.parquet` updated accordingly: the
  `timestamp` column is now a real datetime instead of the raw
  `2022/09/01 00:10` string, and a `txn_index` column (0-based position
  within a campaign's time-ordered transactions) gives each transaction a
  stable id that the pipeline — not the API — defines, which is what
  `/api/campaigns/{id}/visibility` uses to report what each institution
  can see. Campaign search was tightened after a test caught that a bare
  `1` matched every account containing a 1: digits-only queries now mean
  the campaign id plus accounts whose account part starts with those
  digits, and everything else is a substring match on the account id.
  Added `tests/test_api.py` (24 tests) run against artifacts produced by
  the real export; also booted the server under uvicorn and exercised
  `/api/health`, `/api/campaigns`, `/api/campaigns/{id}/visibility`,
  `/api/visibility/distribution`, `/api/methodology` and the results 404
  over HTTP. Test suite: 69 passed, 10 skipped. Updated README: Project
  status, Repository structure, Setup (new `FRAUDCAMP_ARTIFACTS_DIR` env
  var), How to run (server command), Backend: modules (new API server
  subsection), Artifacts (`timestamp`/`txn_index`), API reference (full
  contract), Testing.
- **2026-10-06** — Stage 2: artifact export. Added
  `backend/src/fraudcamp/pipeline.py` (end-to-end orchestration, shared by
  the runner and the exporter so they cannot drift) and `export.py`
  (writes `summary.json`, `campaigns.parquet`,
  `campaign_transactions.parquet`, `institutions.json`,
  `visibility/seed_{0..4}.parquet`, and an empty `results/`). Rewrote
  `scripts/run_pipeline.py` on top of `pipeline.build()` and gave it an
  `--export` flag. Decisions settled this stage and recorded in the
  Decision log: (1) campaigns relabelled `"unfragmentable"` are **still
  reassigned** — reassignment is now an explicit topology test exposed as
  the `reassignable` column, which also makes the "152 eval_ok
  fragmentable" verified number reproducible after the relabel; (2) bank
  volume ties are broken by bank id and zero-pre-cutoff-volume banks still
  get an institution, fixing a latent `KeyError` for campaigns that cross
  the cutoff into a bank with no earlier activity; (3) achieved visibility
  is exported for all 370 campaigns, with a `campaign_reassigned` flag,
  since the frontend must be able to open any campaign. Added
  `tests/test_pipeline.py` and `tests/test_export.py` (20 new tests) plus a
  `mini_dataset` synthetic-dataset fixture in `conftest.py`; verified the
  `--export` CLI path end to end against that fixture outside pytest. Test
  suite: 45 passed, 10 skipped. Updated README: Project status, Frozen
  research definitions (reassignment-is-topology, institution tie-break and
  zero-volume banks, floor-run seed note), Decision log, Verified numbers
  (footnote on the 152 figure), Repository structure, How to run,
  Backend: modules, Artifacts (full schema of every file), Testing.
- **2026-10-05** — Stage 1: implemented the research pipeline —
  `backend/src/fraudcamp/{constants,load_data,campaigns,splits,institutions,visibility}.py`
  — covering column/timestamp/account-ID loading, cutoff, pattern
  parsing + matching (asserts all pattern transactions match), the
  campaign table (`eval_ok`, `crosses_cutoff`, `deadline`), splits,
  hub/fragmentable/unfragmentable groups, `shares_train_account`,
  volume-balanced institution assignment (K=8), and visibility (local
  search, start-time-ordered conflict resolution with recomputation,
  bins). Added `backend/requirements.txt`, `backend/pyproject.toml`
  (src-layout packaging + `fulldata` pytest marker),
  `backend/scripts/make_sample.py` (Kaggle-side 1%-sample generator) and
  `run_pipeline.py` (prints every Verified-numbers value for a given
  config). Added 25 unit tests across 5 test files plus
  `test_fulldata.py` (10 regression tests, `@pytest.mark.fulldata`,
  auto-skip without the full dataset) and `conftest.py`. Fixed two issues
  found while running the suite: `splits.assign_split` had to build its
  `None`-valued column via a plain list instead of `.apply()`/`np.select`,
  because pandas 3.x's default string-dtype inference silently turns
  `None` into `<NA>` before an `.astype(object)` can recover it; and an
  initial `test_visibility_no_overlap_is_low` fixture was constructed
  wrong (gave 2/3 visibility, not 1/3) and was corrected. All 25 unit
  tests pass; the 10 fulldata tests skip on this machine (no dataset
  downloaded locally, and the C: drive had too little free space —
  ~65MB — to create a venv; verified the system Python already has
  pandas/numpy/pyarrow/pyyaml/pytest/fastapi/pydantic and ran the suite
  against it directly). Updated README: Project status, Setup, How to
  run, Backend: modules, Repository structure, Testing.
- **2026-10-05** — Stage 0: moved `01_data_verification.ipynb.ipynb` →
  `docs/reference/01_data_verification.ipynb` (read-only evidence, not
  run or modified locally); created repo skeleton (`backend/`,
  `frontend/`, `docs/reference/`, placeholder `README.md`s in
  `models/`, `federation/`, `evaluation/`); created `.gitignore`; created
  `CLAUDE.md`; created this `README.md` with all required sections,
  frozen research definitions, decision log, and verified numbers
  transcribed from the notebook's cell outputs.
