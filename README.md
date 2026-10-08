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
18. [Version history](#version-history)
19. [Changelog](#changelog)

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

**Current stage:** Phase 3, stage 3.0 — **done and verified**. All 23
regression numbers reproduced on the full dataset via Kaggle on 2026-10-08
(`docs/reference/verify_report.md`), and the exported artifacts are
installed locally and serving.
**Next step:** stage 3.1 (window builder + features + leakage tests). One
open question first: the visibility assignment is not reproducible across
processes (see [Known limitations](#known-limitations-and-open-questions)) —
it does not affect Phase 3, which is centralized-only, but it should be
settled before Phase 4.

**Phase 2 — pipeline, artifacts, API, frontend (complete)**

| Stage | Description | Status |
|---|---|---|
| 0 | Notebook relocation, CLAUDE.md, README, repo skeleton, plan | done |
| 1 | Research pipeline (loading, campaigns, splits, institutions, visibility), unit tests, `make_sample.py`, `run_pipeline.py` | done |
| 2 | Artifact export | done |
| 3 | FastAPI server + API tests | done |
| 4 | Frontend scaffold, theme, layout, API client, types, Home, Dataset, Campaign Explorer | done |
| 5 | Campaign Detail, Visibility Lab, Results Dashboard (empty-state), Methodology page, frontend tests | done |

**Phase 3 — baselines and campaign-level evaluation (in progress)**

Gate question: can a **centralized** model (full visibility, natural data, no
reassignment) discover campaigns early, and does a graph model beat a
non-graph baseline at it? Nothing federated, no institutions in training, and
no visibility reassignment is used in this phase.

| Stage | Description | Status |
|---|---|---|
| 3.0 | Kaggle runner + Phase 2 full-data verification | done — all 23 numbers PASS (2026-10-08) |
| 3.1 | Window builder + features + leakage tests | not started |
| 3.2 | Evaluation module (extraction, matching, lead time, bootstrap CIs) + oracle and random scorers | not started |
| 3.3 | XGBoost baseline: train, select tau on validation, evaluate on test | not started |
| 3.4 | GNN centralized (GINEConv edge classification) | not started |
| 3.5 | Gate report + results files + frontend Results page wired to real baseline results | not started |

**Later phases (not started)**

| Phase | Description |
|---|---|
| 4 | Federated conditions: FedAvg-only, FedAvg + boundary embedding exchange |
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
- Test results reported on three subsets: `all`, `fully_observed`
  (`crosses_cutoff == False`), `no_shared_accounts`
  (`shares_train_account == False`).

## Decision log

| Date | Decision | Reason | Evidence |
|---|---|---|---|
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
    reference/
      01_data_verification.ipynb   read-only: verified Kaggle exploration behind every definition above
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
      models/README.md            placeholder — Phase 3+ (not this session)
      federation/README.md        placeholder — Phase 4+ (not this session)
      evaluation/README.md        placeholder — Phase 3+ (not this session)
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
      run_pipeline.py             runs the pipeline end to end; prints every Verified-numbers value; --export writes artifacts
      kaggle/
        run_on_kaggle.md          the exact notebook cells for running this repo on Kaggle
        kaggle_runner.py          verify / pipeline (train / evaluate land in stages 3.3-3.4)
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

  Both commands print elapsed time and memory per step, take `--config` and
  `--out`, and work locally too (pointed at a sample config, the regression
  table is skipped with an explanation rather than reporting 23 meaningless
  failures). The full notebook walkthrough is
  `backend/scripts/kaggle/run_on_kaggle.md`.
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
- **`visibility.py`** — `visibility_of` (max-over-institutions seen
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
  `pipeline` builds, exports and zips the artifacts; `train` and `evaluate`
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
| `test_subset` | `"all"`, `"fully_observed"`, or `"no_shared_accounts"` |
| `campaign_recall` | fraction of eligible campaigns detected by deadline |
| `campaign_precision` | fraction of detections that are true positives |
| `false_alarms` | count of false-positive detections |
| `median_lead_time_h` | median lead time in hours, among detected campaigns |
| `median_normalized_lead_time` | median lead time normalized by campaign duration |
| `median_frac_observed_at_detection` | median fraction of campaign transactions observed at detection |

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

**Current pass status (2026-10-08):**
- Backend: `python -m pytest tests/` from `backend/` → 75 passed, 24 skipped
  (fulldata, no dataset present on this machine), 0 failed. The 24 skipped
  ones passed on Kaggle against the full data on 2026-10-08.
- Frontend: `npm run test` from `frontend/` → 48 passed, 0 failed.
  `npm run build` (tsc + bundle) and `npm run lint` both clean.

**Not verified:** the pages have not been opened in a real browser from this
environment — there is no browser tooling available here. Rendering is
covered by the jsdom render tests above, every endpoint the pages depend on
was confirmed to answer correctly over HTTP (including the results 404 that
drives the empty state), and both servers were confirmed to talk to each
other including the CORS header. But visual layout, chart appearance, dark
mode, and in particular the **Cytoscape graph** (which the tests stub,
because jsdom has no canvas) have not been seen running. The campaign graph
is the one piece that most warrants a look in a browser before the demo.

## Known limitations and open questions

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
- **The visibility assignment is not reproducible across processes.** The
  pipeline seeds its RNG explicitly, but `build_global_assignment` iterates
  a campaign's accounts from a Python **set**, and string hash
  randomisation makes that order differ between processes. The seeded draws
  therefore follow a different trajectory each run. Demonstrated by running
  the same build under `PYTHONHASHSEED=1/2/3`: the resulting
  account-to-institution maps differ. It also explains why the Kaggle
  `verify` run reported 3 unfragmentable campaigns while the `pipeline` run
  that produced the shipped artifacts reported 2. No regression number is
  affected (none depends on the search), and Phase 3 is centralized-only so
  it does not use these assignments at all — but Phase 4-5 and the
  `unfragmentable` control-group label do. The fix is a one-line
  determinisation (sort the accounts before searching), which would change
  the exported visibility values, so it needs an explicit decision before
  anyone builds on the current `visibility/seed_*.parquet`. **Open.**
- The frontend has not been visually inspected in a browser from the
  development environment used so far (no browser tooling available); the
  Cytoscape campaign graph in particular is covered only by stubbed tests.
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

## Version history

Tagged points in the repository, newest first. Check one out with
`git checkout <tag>`.

| Tag | Date | What it contains |
|---|---|---|
| `v0.1-data-pipeline` | 2026-10-08 | Phase 2 complete, plus the stage 3.0 Kaggle runner. The research pipeline (campaign ground truth, splits, topology groups, institutions, visibility), the artifact export, the read-only FastAPI server, and the full React frontend — 69 backend tests and 48 frontend tests passing. **No models and no experiment results**: the four conditions are defined and the result schema is fixed, but nothing has been trained or run, and the full-data regression numbers are still unverified (that is what stage 3.0's Kaggle run is for). |

## Changelog

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
