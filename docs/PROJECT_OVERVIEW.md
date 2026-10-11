# Early Discovery of Emerging Fraud Campaigns from Fragmented Cross-Institution Evidence

**A complete overview of the project, and where it should go next.**

Final-year B.Tech (AI & Data Science) major project · journal paper planned
Repository: https://github.com/jerinsaji299-coder/Fraud-campaign-detection

> This document is a readable account of the whole project for someone
> coming to it fresh — a supervisor, a reviewer, a collaborator, or the
> author of the paper six months from now. It is a **summary**, not the
> specification. [`README.md`](../README.md) at the repository root is the
> authoritative living document: every frozen definition, decision, verified
> number and API contract lives there, and where the two ever disagree, the
> README is right.

---

## Table of contents

1. [The problem in one page](#1-the-problem-in-one-page)
2. [Research question and contribution](#2-research-question-and-contribution)
3. [The data](#3-the-data)
4. [The research design](#4-the-research-design)
5. [What has been built](#5-what-has-been-built)
6. [Where the project stands](#6-where-the-project-stands)
7. [What is verified vs. what is assumed](#7-what-is-verified-vs-what-is-assumed)
8. [How the engineering protects the research](#8-how-the-engineering-protects-the-research)
9. [Four bugs worth writing about](#9-four-bugs-worth-writing-about)
10. [Improvements and roadmap](#10-improvements-and-roadmap)
11. [Known weaknesses and open questions](#11-known-weaknesses-and-open-questions)
12. [Reproducing everything](#12-reproducing-everything)
13. [Glossary](#13-glossary)

---

## 1. The problem in one page

Money laundering is deliberately spread across institutions. A scheme that
looks obvious when you can see all of it — money cycling through a ring of
accounts, or fanning out from one account to twenty — looks like ordinary
business from inside any single bank, because each bank only sees the
transactions that touch its own customers.

Banks could, in principle, collaborate: share what their models have
learned, or share evidence about the accounts sitting on the boundary
between them. Both are being actively researched. But two questions have
not been answered:

1. **Does collaboration help most exactly when it is needed most?** If each
   bank can see 90% of a scheme, collaboration is nearly pointless. If each
   can see 20%, it may be the only thing that works. The *shape* of that
   relationship — the dose-response curve — is unknown.
2. **Does it depend on the shape of the scheme?** Some topologies cannot be
   hidden at all. In a fan-out, every transaction touches the central
   account, so that account's own bank always sees 100% of the scheme no
   matter how the other accounts are distributed. Other topologies — cycles,
   chains, bipartite structures — *can* be split so that nobody sees much.

This project measures both, and it measures them in the units that matter
operationally: not "did the model label this transaction correctly", but
**did it discover the campaign, and how early**.

---

## 2. Research question and contribution

> **How does the benefit of collaboration between banks for early
> laundering-campaign discovery change as each bank's visibility of a
> campaign decreases, and does this depend on campaign topology?**

Four conditions are compared:

| Condition | What is shared | What it measures |
|---|---|---|
| **Isolated** | nothing | each bank alone |
| **FedAvg-only** | model weights | the *knowledge* advantage |
| **FedAvg + boundary embedding exchange** | weights, plus embeddings of boundary accounts at detection time | the *evidence* advantage |
| **Centralized** | everything (one pooled view) | the full-visibility upper bound |

### Positioning against related work

| Work | What it does | What this project adds |
|---|---|---|
| Ceydeli et al. 2026 | federated cross-client subgraph pattern detection with embedding exchange | campaign-level evaluation and lead time, rather than pattern/transaction accuracy |
| Zheng et al. 2025, *Networked Markets, Fragmented Data* | federated GNN on bank-partitioned IBM AML | a **graded** visibility dose-response, not a single partitioning |
| FedGraph-VASP 2026 | federated graph learning for VASPs | topology split: fragmentable campaigns vs. a hub control group |

**The distinct contribution, stated plainly:** campaign-level evaluation,
discovery lead time as a first-class metric, and a graded visibility
dose-response curve split by campaign topology, with a control group that
*cannot* be fragmented and therefore should show no effect. That control is
the part most easily skipped and the part that makes the result credible: if
the hub campaigns show the same curve as the fragmentable ones, the effect
is not about visibility at all.

---

## 3. The data

**Source:** IBM's synthetic anti-money-laundering transaction set
(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml` on Kaggle),
`HI-Small` variant. It ships a ground-truth file naming every transaction
that belongs to a laundering scheme — which is what makes campaign-level
evaluation possible at all.

| Quantity | Value |
|---|---|
| Transactions | 5,078,345 |
| Date range used | Sept 1–10, 2022 (cutoff `2022-09-11 00:00`) |
| Banks | 30,470 |
| Laundering transactions | 5,177 (~0.1%) |
| Campaigns (ground truth) | 370 |
| Campaign transactions | 3,209 — all matched back to the main table |
| Laundering with no campaign | 1,968 — used as training labels, excluded from campaign scoring |
| Evaluable campaigns (`eval_ok`) | 258 |

**Why Sept 11 onward is excluded.** Only 1,108 rows fall on or after the
cutoff, and **59.1%** of them are laundering, against ~0.1% overall. A model
trained through that tail could score well by learning "late in the file
means laundering" and nothing about behaviour. The excluded data is still
used for exactly one thing: knowing when a campaign *truly* finished, so
lead time is measured against reality rather than against a truncated view.

---

## 4. The research design

Every definition below is **frozen** — changeable only by explicit decision,
with the reason recorded. This is the discipline that makes the eventual
numbers mean something; the full text and the reason for each lives in the
README.

### Campaigns

A campaign is one `BEGIN … END LAUNDERING ATTEMPT` block of the pattern
file. Its `base_type` is one of FAN-OUT, FAN-IN, CYCLE, SCATTER-GATHER,
GATHER-SCATTER, BIPARTITE, STACK, RANDOM. A campaign is **evaluable**
(`eval_ok`) with at least 3 transactions and 3 accounts — below that, "how
early was it discovered" has no meaningful answer, and about a quarter of
campaigns are that small.

### Splits, by campaign start date

| Split | Dates | Evaluable campaigns |
|---|---|---|
| train | Sept 1–4 | 100 |
| val | Sept 5 | 26 |
| test | Sept 6–7 | 56 |
| stress | Sept 8–10 | 76 |

Test was moved from Sept 8–10 to Sept 6–7 because under the old split **74
of 76** test campaigns were still unfinished at the cutoff. Sept 8–10 became
a deliberate "stress" split of heavily censored campaigns instead of being
quietly dropped.

### Topology groups

`hub = {FAN-IN, FAN-OUT, GATHER-SCATTER}` — always fully visible to one bank
by construction, so they form a **control group** rather than a treatment
group. Everything else is `fragmentable`. A fragmentable campaign that in
practice still cannot be pushed below 90% visibility is relabelled
`unfragmentable` and joins the control group; on the full data there are
exactly **2** such campaigns.

### Institutions and visibility

The 30,470 banks are grouped into **K = 8** institutions, balanced by
pre-cutoff transaction volume, deterministically. An institution *sees* a
transaction if either endpoint's bank belongs to it.

> **Campaign visibility** = the largest fraction of a campaign's
> transactions that any single institution can see.

Visibility is the independent variable. For each (seed, target) with targets
`{1.0, 0.75, 0.5, 0.25}` and seeds `{0,1,2,3,4}`, the accounts of
fragmentable campaigns are reassigned to institutions by local search toward
the target. Because achieved visibility often misses its target — the
median at target 0.25 is about 0.33 — **analysis is by achieved bin, not by
requested target**. Reporting by the target you asked for rather than what
you got would be dishonest.

Why K = 8 and not 3: every transaction is visible to *both* endpoint banks,
so the structural visibility floor is about `2/K`. With K = 3 that floor is
~67%, far above the lowest target.

### Phase 3: detection, evaluation, and lead time

Detection runs on a **causal** 6-hourly grid. At detection time `t` a model
sees only transactions with `t − L ≤ timestamp < t` — the upper bound
strict, so a transaction stamped exactly `t` is already the future. Lookback
`L ∈ {24, 48}` hours, chosen on validation.

**Extraction and matching** are fixed and non-learned, identical for every
model and condition, so differences in results come from the scores alone:
edges scoring ≥ τ form an undirected graph; connected components with ≥ 3
accounts are clusters; a cluster **hits** a campaign if ≥ 50% of the
cluster's accounts belong to it *and* it covers ≥ 3 of the campaign's
accounts. τ is chosen on validation only, by campaign-level F1, and frozen
for test.

**Four outcomes**, and the two in the middle are what make the metric
honest:

| Outcome | Meaning | Counted as |
|---|---|---|
| `hit` | matched a campaign of the split being scored | success |
| `other_split_hit` | matched a real campaign from a *different* split | neither — reported separately |
| `ambiguous` | mostly accounts doing laundering that belongs to no campaign | neither — reported separately |
| `false_alarm` | everything else | failure |

**Lead time**, measured against true completion, in three forms: absolute
hours, normalized by campaign duration, and the fraction of the campaign's
transactions observed at detection. A campaign not detected by its deadline
is **missed**, full stop.

---

## 5. What has been built

Roughly **10,500 lines** across 65 source files, with **256 tests**.

```
          Kaggle (full data, 5M rows)
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  research pipeline  (fraudcamp/)        │   ← single source of truth
  │  load → campaigns → splits → groups     │
  │       → institutions → visibility       │
  │  + windows, features, evaluation        │
  └──────────────┬──────────────────────────┘
                 │ export.py
                 ▼
      small artifacts (~400 KB, not the raw data)
      summary.json · campaigns.parquet
      campaign_transactions.parquet
      institutions.json · visibility/seed_{0..4}.parquet
                 │
                 ▼
  ┌─────────────────────────────┐      ┌──────────────────────────┐
  │  FastAPI  (read-only)       │─────▶│  React frontend          │
  │  reads artifacts at startup │      │  7 pages, never computes │
  │  never touches raw data     │      │  research metrics        │
  └─────────────────────────────┘      └──────────────────────────┘
```

### The backend pipeline

| Module | Responsibility |
|---|---|
| `constants.py` | every frozen definition, in one place |
| `load_data.py` | CSV loading, global account IDs, the cutoff |
| `campaigns.py` | pattern parsing, matching to the transaction table, the campaign table |
| `splits.py` | splits, topology groups, shared-account flags |
| `institutions.py` | deterministic volume-balanced bank → institution assignment |
| `visibility.py` | targets, local search, conflict resolution, bins |
| `windows.py` | the detection grid and the **only** place a causal window is ever sliced |
| `features.py` | 54 edge + 6 node features, training-only vocabularies and scaler |
| `evaluation/` | extraction, matching, outcomes, lead time, metrics, bootstrap CIs, sanity scorers |
| `pipeline.py` | end-to-end orchestration, shared by the runner and the exporter |
| `regression.py` | the 24 expected full-data numbers, in one place |

### The API and frontend

A read-only FastAPI server serves the artifacts (never the raw data) with
pydantic models for every response, and a React + TypeScript frontend
presents the dataset, a campaign explorer, an interactive campaign graph
(Cytoscape, with deterministic layouts so a FAN-OUT always looks like a star
and a CYCLE always looks like a ring), a time replay, a "bank's-eye view"
that fades everything a chosen institution cannot see, a visibility
sandbox, a methodology page rendered from the same constants the pipeline
uses, and a results page that shows **labelled empty outlines** rather than
invented numbers.

One rule governs the frontend: **it never computes a research metric**. The
single exception is the visibility sandbox, where the user invents a
scenario the backend has never seen; that number is computed by a direct
mirror of the backend function, tested against the same cases.

### The Kaggle runner

The full data only exists on Kaggle, so the heavy steps run there through
one entry point with five subcommands: `verify` (reproduce all 24 regression
numbers), `pipeline` (export and zip the artifacts), `features-smoke`
(window sizes, build cost, GPU feasibility), `evaluate-scorers` (validate
the evaluation module), and `make-sample` (build the 1% development sample).

---

## 6. Where the project stands

**Phase 2 — pipeline, artifacts, API, frontend: complete and verified.**

| Stage | Status |
|---|---|
| 0 · repo, docs, plan | done |
| 1 · research pipeline + unit tests | done |
| 2 · artifact export | done |
| 3 · FastAPI server + API tests | done |
| 4 · frontend scaffold, Home / Dataset / Explorer | done |
| 5 · Campaign Detail, Visibility Lab, Results, Methodology | done |

**Phase 3 — centralized baselines. Gate question: can a full-visibility
model discover campaigns early at all, and does a graph model beat a
non-graph one?**

| Stage | Status |
|---|---|
| 3.0 · Kaggle runner + full-data verification | **done** — all 24 numbers reproduced |
| 3.1 · windows, features, leakage tests | **done** |
| 3.2 · evaluation module, oracle/random scorers | **done** (code); full-data validation pending |
| 3.3 · XGBoost baseline | not started |
| 3.4 · centralized GNN (GINEConv) | not started |
| 3.5 · gate report + frontend results | not started |

**Phases 4–7** (federated conditions, the core experiment, extensions,
analysis and paper) have not begun.

### There are no results yet

This is worth stating as flatly as possible, because a reader skimming a
repository this size will assume otherwise: **no model has been trained and
the project has no findings.** What exists is the measurement apparatus, and
evidence that the apparatus is correct. The numbers that appear anywhere in
this repository are dataset properties, code checks, or sanity-scorer
behaviour — never results.

---

## 7. What is verified vs. what is assumed

Being precise about this is the difference between a project that can be
defended and one that cannot.

### Verified on the full data

| Claim | Evidence |
|---|---|
| All 24 regression numbers match the reference notebook | Kaggle run, 2026-10-08, `docs/reference/verify_report.md` |
| The visibility search is reproducible across processes and sessions | 2 independent Kaggle sessions × 3 processes, byte-identical files, 2026-10-09 |
| The unfragmentable count is 2 | same, pinned in the regression table |
| 13 / 9 / 5 usable training detection times for L = 24 / 48 / 72 | time-grid arithmetic, confirmed on the full-data run |
| 54 edge feature dimensions (15 payment currencies, 15 receiving, 7 formats) | full-data `features-smoke` |

### Verified on synthetic or fixture data only

| Claim | Status |
|---|---|
| Features contain no leakage from the future, labels, or identity | property tests — strong, but the properties are checked on small data |
| Extraction, matching, lead time and the four outcome kinds behave correctly | hand-built cases with answers computable on paper |
| Oracle finds nearly everything; random finds nothing | synthetic data; **full-data check pending** |

### Assumed

- That the matching rule's 50% / 3-account threshold is appropriate on the
  real data. The `evaluate-scorers` run tests this: if random scores well,
  the rule is too generous.
- That a centralized model can detect anything at all. That is Phase 3's
  gate question and nobody knows the answer yet.

---

## 8. How the engineering protects the research

These practices are not incidental; several of them have already caught
errors that would have invalidated results.

**One source of truth per fact.** The frozen definitions live in one
constants module used by the pipeline *and* by the API's methodology
endpoint, so the published methodology cannot drift from what the code did.
The 24 expected numbers live in one module read by both the test suite and
the Kaggle report.

**Leakage as invariance, not inspection.** Rather than reading the feature
code and declaring it clean, the tests assert that the feature matrices are
*bit-identical* when the future is deleted, when the future is replaced with
absurd values, when every label is flipped, when every account is relabelled
and when every bank is renumbered. A leak cannot survive those.

**Deliberate violation.** Every protective test has been proven able to
fail, by breaking the thing it protects and checking that it breaks: a label
added to the features, an off-by-one `≤ t` window bound, a bank ID used as a
feature, the determinism fix reverted, the stress horizon moved, the
full-lookback filter removed. A test that has never failed is a test nobody
has reason to trust.

**Seeded and reproducible.** Every stochastic step takes an explicit seed,
and reproducibility is verified *across processes* — which is how the one
genuine reproducibility defect was found (§9).

**Honest empty states.** The frontend's results page shows labelled outlines
of the planned figures and refuses to display a number it does not have.
Artifacts, data and credentials are gitignored and verified so.

**A decision log.** Every design choice carries a date, a reason, and the
evidence behind it. When a reviewer asks "why K = 8?", the answer is a row
in a table, not a memory.

---

## 9. Four bugs worth writing about

Each of these is a candidate paragraph in the paper's methods or
threats-to-validity section, because each is a mistake that would have
produced plausible-looking wrong numbers.

**1 · The visibility search was not reproducible.** Every random step was
seeded, yet two Kaggle runs of the same commit disagreed on how many
campaigns were unfragmentable (3 vs 2). Cause: a campaign's accounts were
held in a Python `set`, and string hashing is randomised per process, so the
seeded draws followed a different trajectory in every process. The fix was
to sort before drawing. *The lesson: a seed only reproduces a run if the RNG
is asked for things in the same order, and in-process tests cannot detect
this — one process has one hash seed.* The test suite now runs the build in
subprocesses under differing `PYTHONHASHSEED`.

**2 · A window was silently clamped.** The feature smoke test reported a
1.5M-edge window as the largest, at a detection time whose 48-hour lookback
began before the data did. The slice clamped at the data start, so a "48-hour
window" was really shorter — and the 48- and 72-hour rows came out
identical, which is what gave it away. *The lesson: a window that runs off
the edge of the data fails silently unless something checks.* Hence the rule
that training uses only detection times with a full lookback, and a full
enumeration of every window rather than sampling five.

**3 · The random scorer scored too well.** On the first synthetic dataset,
the random baseline reached 0.57–0.71 campaign recall. The frozen design
anticipated exactly this symptom and says it means the matching rule is too
generous. It did not: that generator gave campaign accounts no background
traffic at all, so randomly keeping half the edges left a small *pure*
fragment of each campaign, which the matching rule correctly called a hit.
*The lesson: when a sanity check fails, the test data is a suspect too.* The
generator was fixed — never the frozen rule — and random recall went to
0.00. The real check is still the full-data run.

**4 · Performance defects that only bite at scale.** An `O(campaigns)`
dataframe filter per detection, and a pure-Python union-find over
million-edge graphs. Both invisible on fixtures, both material on 5M rows.
*The lesson: profile against the real shape before the real run, not during
it.*

---

## 10. Improvements and roadmap

### Immediate (unblocks everything else)

| # | Action | Why |
|---|---|---|
| 1 | Run `evaluate-scorers` on the full data | The evaluation module has never met real data. If random scores well, the matching rule needs revisiting *before* any model is trained on top of it. |
| 2 | Run `make-sample`, download to `backend/data/sample/` | Local development is currently on synthetic data. The real 1% sample makes local iteration meaningful. |
| 3 | Re-run `features-smoke` | The corrected version supersedes the first run's largest-window figure. |
| 4 | Record the oracle ceiling | The oracle's median lead time is the best achievable under the matching rule. Every later result should be read as a fraction of it, not of a theoretical 100%. |

### Phase 3 proper

| # | Action | Notes |
|---|---|---|
| 5 | XGBoost baseline (3.3) | Decide the negative-sampling strategy explicitly: all training windows concatenated at full scale is ~3 GB, and the class weight is ~1,000. |
| 6 | Centralized GNN (3.4) | Regime already frozen: neighbour sampling at **both** lookbacks, so the L comparison is not confounded by training method. Fanouts and batch size to be reported before training. |
| 7 | Gate report (3.5) | A written, honest answer to "can this detect anything early?" — including the option that the answer is no. |

### Phases 4–5

| # | Action | Notes |
|---|---|---|
| 8 | FedAvg-only, then embedding exchange | Separating the knowledge advantage from the evidence advantage is the contribution; keep them separately measurable. |
| 9 | The visibility sweep | 4 conditions × 4 bins × 5 seeds × 2 groups. Budget the compute *before* starting; this is where the project could run out of time. |
| 10 | Wire real results into the frontend | The results page is built and waiting; the schema is fixed. |

### Research improvements worth considering

These are suggestions, not decisions.

1. **A learned-extraction ablation.** Clustering is currently fixed and
   non-learned — the right choice, because it keeps the comparison clean.
   But a reviewer will ask whether a learned clustering changes the
   conclusion. Even one ablation at a single visibility bin would answer it.
2. **Report the per-campaign curve, not only the aggregate.** With 56 test
   campaigns, medians hide a lot. A per-campaign scatter of lead time
   against achieved visibility would show whether collaboration helps *every*
   campaign a little or *a few* campaigns enormously — a materially
   different claim.
3. **Sensitivity of the matching rule.** The 50% / 3-account rule is
   defensible but arbitrary at the margin. Recomputing the headline result
   at 40% and 60% would convert "we chose a threshold" into "the result does
   not depend on the threshold" — cheap, and worth a paragraph.
4. **A second dataset.** `HI-Medium` is already noted as an optional
   replication. One replication, even partial, moves the paper from "an
   effect on HI-Small" to "an effect".
5. **An operational cost axis.** False alarms per day is already recorded.
   Framing results as "lead time gained per false alarm per day" would speak
   to practitioners far more directly than recall at a fixed threshold.
6. **State the privacy claim precisely.** The federated conditions share
   weights and boundary embeddings. What an adversary could reconstruct from
   those is a known research question, and Phase 6's differential-privacy
   noise extension is the place to address it. Say what is and is not
   claimed.

### Engineering improvements worth considering

1. **A results-schema validator.** The results schema is documented but
   nothing enforces it. A small validator run at write time would stop a
   malformed results file from reaching the API or the frontend.
2. **Dependency pinning.** `pandas>=2.2` is pinned for a known reason;
   nothing else is. A lockfile would make "the environment the digests were
   produced in" reproducible rather than described.
3. **Frontend visual verification.** No page has been opened in a browser
   from the development environment; the Cytoscape graph in particular is
   covered only by stubbed tests. One pass with a real browser before the
   demo is overdue.
4. **Artifact size budget as a test.** Artifacts are small by design. A test
   asserting they stay small would prevent the raw data creeping in by
   accident.
5. **Clustering performance headroom.** A scipy fast path exists, but the
   visibility sweep multiplies every cost by 4 conditions × 4 bins × 5
   seeds. Measure before Phase 5, not during.

---

## 11. Known weaknesses and open questions

Stated plainly, because a project that lists its own weaknesses is easier to
trust than one that does not.

| Weakness | Status |
|---|---|
| No model, no results | expected at this stage; Phase 3 |
| The evaluation module has never seen full data | next action |
| The 1% sample has never been generated | next action |
| Visibility digests are only comparable within one environment (a different pyarrow could change parquet bytes without changing data) | documented; compare the unfragmentable count and achieved visibilities instead |
| Institution assignment is volume-balanced but otherwise unrealistic — no geography, no real bank behaviour | inherent to the design; worth naming in the paper |
| The local search for visibility is a heuristic, not an optimiser | why analysis is by achieved bin, not target |
| Only one dataset | `HI-Medium` replication is optional and unstarted |
| L = 72 was dropped | 5 training windows, ~18.6 GB full-batch memory, and a 3-day window on a 10-day dataset |
| Frontend never visually inspected | needs a browser pass |

---

## 12. Reproducing everything

```bash
git clone https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git
cd Fraud-campaign-detection
```

**Backend** (Python 3.11+):
```bash
pip install -r backend/requirements.txt
cd backend && python -m pytest tests/        # 208 pass, 25 skip without full data
```

**Frontend** (Node LTS):
```bash
cd frontend && npm install && npm run test && npm run dev
```

**Full data** runs on Kaggle in this order — see
[`backend/scripts/kaggle/run_on_kaggle.md`](../backend/scripts/kaggle/run_on_kaggle.md):

```
verify → pipeline → features-smoke → evaluate-scorers → make-sample
```

All five are CPU-only. Training (3.3–3.4) is the first step needing a GPU.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Campaign** | One laundering attempt from the dataset's ground truth: a set of transactions belonging to the same scheme. |
| **Visibility** | The largest share of a campaign's transactions that any single institution can see. The study's independent variable. |
| **Lead time** | How far ahead of a campaign's true completion it was discovered — in hours, as a share of its duration, and as the share of its transactions seen. |
| **Hub pattern** | A campaign built around one central account (fan-in, fan-out, gather-scatter). Always fully visible to one bank, so it acts as a control. |
| **Fragmentable** | A campaign whose accounts can be spread so no single institution sees most of it. |
| **Unfragmentable** | Fragmentable in shape, but in practice never below 90% visibility — usually too small. Joins the control group. |
| **eval_ok** | A campaign large enough to evaluate: ≥ 3 transactions and ≥ 3 accounts. |
| **Detection time (t)** | A 6-hourly moment at which a model is asked whether a campaign is forming. |
| **Lookback (L)** | How far back from `t` a model may look: 24 or 48 hours. |
| **τ (tau)** | The score threshold above which an edge enters the clustering. Chosen on validation only. |
| **Other-split hit** | A cluster that correctly found a real campaign from a different split than the one being scored. Neither success nor false alarm. |
| **Ambiguous cluster** | A cluster made mostly of accounts doing laundering that belongs to no campaign. Neither success nor false alarm. |
| **Oracle ceiling** | The oracle scorer's lead time: the best achievable under the matching rule, since no cluster can hit a campaign before 3 of its accounts are visible. |
| **FedAvg** | Federated averaging: institutions train locally and share only model weights. |
| **Embedding exchange** | Sharing compact numeric summaries of boundary accounts at detection time, on top of shared weights. |

---

*This overview summarises the state of the project as of 2026-10-11, at
commit `0aa5a27` (15 commits, tags `v0.1-data-pipeline` and
`v0.2-verified`). For anything authoritative — frozen definitions, the
decision log, verified numbers, API contracts — see
[`README.md`](../README.md).*
