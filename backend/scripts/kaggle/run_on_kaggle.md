# Running this repository on Kaggle

The full dataset is too large for a laptop, so the heavy steps run in a
Kaggle notebook. This document is the exact sequence of cells to paste.

Everything the runner writes goes to `/kaggle/working/`, which is the only
directory Kaggle keeps after the session ends.

---

## 0. Source

The code lives at:

```
https://github.com/jerinsaji299-coder/Fraud-campaign-detection
```

Pushed and tagged `v0.1-data-pipeline` on 2026-10-08. The notebook clones
`main` by default; to pin a session to the tagged state instead, add
`--branch v0.1-data-pipeline` to the clone in Cell 2.

The repository contains no dataset copies and no credentials — `.gitignore`
excludes `backend/data/`, the contents of `backend/artifacts/`,
`node_modules/`, `.venv/`, `.env`, `*.env` and `kaggle.json`. If you later
need Kaggle API credentials in a notebook, use Kaggle's own Secrets feature;
never commit `kaggle.json`.

After pushing new work locally, remember the notebook clones a **fresh copy**
each run, so a Kaggle session only ever sees what has been pushed.

---

## Run order

Each step assumes the ones above it have passed. All of them are CPU-only;
`train` and `evaluate` (stages 3.3–3.4) are the first that need a GPU.

| # | Command | Purpose | Section |
|---|---|---|---|
| 1 | `verify` | reproduce all 24 Phase 2 regression numbers | §2 |
| 2 | `pipeline` | export the artifacts and zip them | §2 |
| 3 | `features-smoke` | window sizes, build cost, GPU estimate | §5c |
| 4 | `evaluate-scorers` | validate the evaluation module on full data | §5d |
| 5 | `oracle-diagnose` | explain why the oracle misses what it misses | §5f |
| 6 | `make-sample` | build the 1% sample for local development | §5e |
| 7 | `train-isolated` | stage 4.1: one booster per institution per (target, seed) | §5l |
| 8 | `evaluate-isolated` | stage 4.1: taus, per-institution ceilings, the two pooled numbers and the union | §5l |

Steps 1–2 are done (2026-10-08/09), step 3 run and corrected, step 4 run
(2026-10-15) — the evaluation is validated. Step 5 is new; step 6 is done and
the sample is present locally. Steps 7–8 are **proposed and not yet run**:
the stage 4.1 session plan is in §5l.

## 1. Notebook settings

1. Kaggle → **Create** → **New Notebook**.
2. Right-hand panel → **Session options**:
   - **Internet: ON** — required for `git clone` and `pip install`.
   - **Accelerator: None** — `verify` and `pipeline` are CPU-only. A GPU is
     needed from stage 3.4 (GNN training) onwards, not before.
3. **Input** → **Add Input** → search for
   `ealtman2019 IBM Transactions for Anti Money Laundering` → **Add**.

---

## 2. Cells

### Cell 1 — confirm the dataset path

The pipeline config expects this exact path. Check it before anything else,
because a path mismatch is the single most common failure here.

```python
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

You should see `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` among the
files. If the path differs (Kaggle sometimes mounts without the `datasets/`
segment), **stop and report the actual path** — the fix is one line in
`backend/configs/kaggle.yaml`, and that file is the single place paths live.

### Cell 2 — clone the repo

Cloned into `/kaggle/temp` on purpose: it keeps the notebook's saved output
small, since only reports and the artifact archive need to persist.

```python
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
```

### Cell 3 — install dependencies

Kaggle already ships most of these; this mainly guarantees `pandas >= 2.2`.

```python
!pip install -q -r requirements.txt
!python -c "import pandas, numpy, pyarrow; print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('pyarrow', pyarrow.__version__)"
```

### Cell 4 — verify Phase 2 against the full data

This is the gate. It runs the full-data regression tests and prints a
PASS/FAIL table of every number in README "Verified numbers".

```python
!python scripts/kaggle/kaggle_runner.py verify
```

Writes `/kaggle/working/verify_report.md` and
`/kaggle/working/regression_actuals.json`.

### Cell 5 — run the pipeline and export the artifacts

```python
!python scripts/kaggle/kaggle_runner.py pipeline
```

Writes `/kaggle/working/artifacts.zip` and
`/kaggle/working/pipeline_report.md`. It re-checks the same regression
numbers, so a clean pipeline run is self-verifying too.

### Cell 6 — list what to download

```python
!ls -lh /kaggle/working/
```

---

## 3. What to send back

1. The **PASS/FAIL table** printed by Cell 4 (or the contents of
   `verify_report.md`, which has the same table plus the timing).
2. The **timing table** — elapsed seconds and peak memory per step. These
   numbers decide how Phase 3 training has to be structured, so they matter
   even when everything passes.
3. Whether any step was killed (out of memory) or took much longer than the
   estimates below.

Then download `artifacts.zip` and unpack it into `backend/artifacts/` locally
so the API and frontend have real data to serve.

---

## 4. What to expect

Rough estimates only — report the numbers the runner actually prints.

| Step | Expected time | Notes |
|---|---|---|
| `pip install` | under a minute | mostly already satisfied |
| `verify` | a few minutes | dominated by reading the 5M-row CSV, then the seed-42 floor visibility run |
| `pipeline` | a few minutes more | adds 5 seeds x 4 targets of visibility search (target 1.0 skips the local search) |

Memory: the transaction table is ~5M rows with string account columns, so
expect a few GB of RAM. A standard Kaggle CPU session has far more than that.
On Linux the runner reports a true peak (via `resource`); the "True peak"
column in its tables says so explicitly.

`artifacts.zip` should be small — a few MB — because the artifacts are
deliberately summaries plus the ~3.2k campaign transactions, never the raw
data.

---

## 5. If something goes wrong

- **"No regression numbers were produced"** — the full-data tests skipped,
  which means the dataset is not where `configs/kaggle.yaml` expects it. Go
  back to Cell 1 and report the real path.
- **A regression number FAILs** — stop and send the table. A mismatch means
  either the pipeline changed behaviour or the dataset version differs from
  the one the reference notebook used. Do not start model work until this is
  resolved; every Phase 3 number would inherit the discrepancy.
- **`include_groups` TypeError** — pandas is older than 2.2. Cell 3 should
  prevent this; check the version it printed.
- **Kernel died** — out of memory. Report which step, and the memory figures
  printed for the steps that did finish.

---

## 5b. Regenerating the visibility artifacts after the determinism fix

The visibility search was not reproducible across processes until
2026-10-08: a campaign's accounts were iterated from a Python set, and
per-process string hashing made the seeded draws follow a different
trajectory each run. Two earlier Kaggle runs disagreed about how many
campaigns were `unfragmentable` (3 vs 2), which is how it was found.

Any `visibility/seed_*.parquet` produced before that fix is stale. Regenerate
it, and **confirm reproducibility by running twice in two fresh sessions**.

### Session 1

```python
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
!git log --oneline -1          # confirm you have the determinism fix
!pip install -q -r requirements.txt
!python scripts/kaggle/kaggle_runner.py verify
!python scripts/kaggle/kaggle_runner.py pipeline
```

Record from the `pipeline` output:

- the **Unfragmentable campaigns** row of the regression table (it prints as
  `PENDING` with an actual value — that value is what gets pinned), and
- the **SHA-256 block** for the five visibility files.

Both are also saved to `/kaggle/working/visibility_hashes.json`, which is the
easiest thing to send back.

### Session 2 — a genuinely fresh session

Stop the first session (**Run → Stop session**, not just "restart kernel"), start a
new notebook, and run the *same* cells again. A new session means a new
Python process with a new hash seed, which is exactly what the old code was
sensitive to.

### Compare

```python
import json
a = json.load(open("/kaggle/input/<session-1-output>/visibility_hashes.json"))
b = json.load(open("/kaggle/working/visibility_hashes.json"))
print("unfragmentable:", a["unfragmentable_campaigns"], "vs", b["unfragmentable_campaigns"])
print("hashes identical:", a["visibility_sha256"] == b["visibility_sha256"])
for name in sorted(b["visibility_sha256"]):
    same = a["visibility_sha256"].get(name) == b["visibility_sha256"][name]
    print(f"  {'OK  ' if same else 'DIFF'} {name}")
```

(Or simply paste both JSON files back and I will compare them.)

**Expected:** the unfragmentable count is the same in both sessions and every
digest matches. Parquet writing itself is byte-stable for identical data
within one environment, which was verified locally, so differing digests mean
the *search* differed — i.e. something is still order-dependent. Report that
rather than picking one of the two runs.

Once both sessions agree, send me the unfragmentable count and I will pin it
in the regression table (it is deliberately `PENDING` until then, so it is
reported but never fails a run).

## 5c. Measuring window sizes and GNN feasibility

Stage 3.1 built the detection windows and features; this measures what they
actually cost on the full data, which decides how stage 3.4 has to train.

```python
!python scripts/kaggle/kaggle_runner.py features-smoke
```

No GPU needed — this builds features on CPU and only *estimates* GPU
memory. It covers L = 24 and 48 (the candidates) plus L = 72, which is measured
once for the record and labelled `[EXCLUDED]` — it was dropped as a
candidate on 2026-10-10. Five detection times: a weekday
and a weekend day from train and from test, plus validation's single weekday
(validation spans only Mon Sept 5 → Tue Sept 6, so there is no weekend time
to sample; the report says so).

Per window it reports edges, accounts, laundering edges, laundering rate,
class weight, slice and build time, process memory, the in-memory tensor
footprint, and the analytic size of the XGBoost matrix. It then estimates
full-batch GPU memory for a 3-layer GINEConv with hidden 64 on the largest
window.

It also prints, per lookback, how many **training** detection times have
their full lookback inside the data (evaluation rule 3). Expect 13 / 9 / 5
for L = 24 / 48 / 72 out of the 16 on the grid (the 5 at L = 72 being one of
the reasons it was dropped) — the counts are
time-grid arithmetic, so the run should confirm them rather than discover
them, and a disagreement means something about the grid has changed.

Writes `features_smoke.csv`, `.json` and `.md` to `/kaggle/working/`. Send
back the `.md` (or the `.csv`).

**What to look for.** The estimate matters because Kaggle's GPUs have 16 GB.
The command reports the largest **candidate** (L = 24/48) window separately
from the excluded L = 72 one, because only the former decides how stage 3.4
trains. Predicted before the run: comfortable at L = 24 on a quiet day,
around 10 GB on a busy 24h window, and ~16.6 GB at L = 48 on the busiest
days — i.e. at the edge of a single 16 GB GPU. If the real numbers confirm
that, stage 3.4 uses neighbour sampling rather than full-batch at L = 48,
which the Phase 3 design already allows. Report the numbers either way; do
not start changing the design.

## 5d. Validating the evaluation module on the full data

Before any model is trained, the evaluation code is checked with two
scorers that have known expected behaviour. If these look wrong, the
evaluation is at fault — not a detector, because there isn't one yet.

```python
!python scripts/kaggle/kaggle_runner.py evaluate-scorers
```

Runs the **oracle** (score = true label) and **random** scorers at both
candidate lookbacks (L = 24, 48) over the val, test and stress horizons,
reporting campaigns, recall, precision, false alarms per day, other-split
hits, ambiguous clusters and all three lead-time forms with 95% bootstrap
CIs, plus the oracle broken down by `base_type` and by topology group.

Tau is chosen **on validation only**, maximising campaign-level F1, exactly
as a real model's will be in stage 3.3 — the full precision/recall-vs-tau
curve is saved alongside the choice. The oracle's scores are 0/1, so every
tau in (0, 1] is equivalent for it; it goes through the same selection
anyway so both scorers are treated identically.

Writes `evaluate_scorers.{md,csv,json}` plus
`evaluate_scorers_tau_curve.csv` and `evaluate_scorers_breakdown.csv`.

It also reports **recall and median lead time at false-alarm budgets of 10,
50 and 100 per day**, taking the lowest tau whose validation alarm rate
stays inside each budget, plus the recall-vs-alarm-rate curve for plotting.
A budget the tau grid cannot reach is reported as unreachable with the
cheapest rate available, rather than quietly using a threshold that breaks
it — if that happens often, pass a denser `--tau-grid`.

Useful options: `--lookbacks 24` to do one lookback, `--splits test` for one
horizon, `--tau-grid 0.5,0.9,0.99` to change the sweep, `--bootstrap 200`
for faster CIs, `--seed` for the random scorer.

**What to look for.**

- Oracle recall should be high (near 1.0 on val and test) with precision
  near 1.0 and essentially no false alarms. Stress will be lower, because
  campaigns censored by the cutoff can finish after their deadline and a
  detection after the deadline counts as missed.
- Random recall should be near zero with many false alarms per day. **If
  random scores well, say so and stop** — that would mean the matching rule
  is too generous, and no model result built on it would be trustworthy.
- Large other-split-hit counts are expected and correct: every window holds
  campaigns from several splits, and the oracle finds those too. They are
  excluded from both precision terms by rule 1.
- The oracle's median lead time and median `frac_observed` are the
  **ceiling** — the earliest any detector can possibly fire under a rule
  that needs three of a campaign's accounts. Send these back; they go in the
  README and every later result is read against them.

Runtime is dominated by the tau sweep on validation, since a low tau keeps
most of a one-to-two-million-edge window. Expect minutes rather than
seconds; report the timing table.

## 5e. Building the real 1% sample

```python
!python scripts/kaggle/kaggle_runner.py make-sample
```

Keeps every transaction named in the pattern file plus a seeded (42) random
1% of the rest, and copies the pattern file unchanged, so the campaign
ground truth stays complete and only the background is thinned. Writes
`HI-Small_Trans.csv` and `HI-Small_Patterns.txt` to `/kaggle/working/`.

Equivalent direct command, if you prefer:

```python
!python scripts/make_sample.py --config configs/kaggle.yaml --out /kaggle/working
```

Download **both** files into `backend/data/sample/` locally. After that,
local development uses the real sample:

```
cd backend
python scripts/run_pipeline.py --config configs/local.yaml
python scripts/run_sanity_scorers.py --config configs/local.yaml
```

The synthetic generator (`make_demo_data.py`) stays for unit tests only.

## 5f. Diagnosing the oracle's misses

The oracle has perfect information and still misses 18% of test campaigns at
L = 24 and 11% at L = 48. Those campaigns are unreachable under the frozen
matching rule whatever the model, so it matters *why*.

```python
!python scripts/kaggle/kaggle_runner.py oracle-diagnose
```

Classifies every missed `eval_ok` campaign (val, test, stress; both
lookbacks) as:

- `structurally_disconnected` — pool **every** one of its own edges before
  the deadline, with no window at all, and its largest connected set of
  accounts is still under 3. No lookback can ever detect it; this is the
  hard floor under the ceiling. Reported by `base_type` and group.
- `slow` — pooled, its own edges do connect 3+ accounts, but never inside a
  single window before the deadline. A longer lookback may reach it, and the
  reported largest transaction gap says whether it could. Also reported by
  `base_type` and group.
- `purity_failure` — a 3+ fragment did form, but the cluster was under half
  the campaign's own accounts. Reported with what it merged into: other
  campaigns, unassigned laundering, or both.
- `late_only` — a matching cluster existed, but only after the deadline.
  Censoring rather than sparsity.
- `unexplained` — none of the above, which would mean the diagnosis and the
  evaluation disagree. Should never appear; report it if it does.

Summarised by reason, `base_type` and group, and it also characterises the
oracle's false-alarm clusters (1–6 per day): since the oracle keeps only
laundering edges, each one is real laundering the rule could not attribute.
For each cluster it names **which of the rule's two conditions failed** —
purity (≥ 0.5 of the cluster belonging to one campaign) or coverage (≥ 3 of
that campaign's accounts in the cluster) — each against its own threshold,
with counts. A cluster that is two thirds one campaign has passed purity and
failed coverage, so one "best overlap" number cannot describe it.

It also reports the **paired lead-time comparison between consecutive
lookbacks**: for the campaigns both lookbacks detect, the per-campaign
lead-time difference as a median with a seeded bootstrap CI (positive means
the longer lookback was earlier), and separately how many campaigns each
lookback detects that the other misses, by id. Two median lead times are
never subtracted — each is over a different set of campaigns, so a longer
lookback that reaches more of the slow campaigns shows a lower median even if
nothing got later. This is the method for every lookback and condition
comparison; see the README section "Comparing two configurations".

Writes `oracle_diagnose.{md,json}` plus up to four CSVs, the paired one being
`oracle_diagnose_paired.csv`. `--false-alarm-limit` caps how many clusters
are characterised.

**This changes no rule.** It is diagnosis, so that the ceiling can be
explained rather than merely reported.

## 5g. Stage 3.3: the XGBoost baseline

**CPU is enough. Do not use a GPU for this.** Use a CPU notebook (4 vCPU,
30 GB RAM). Reasons, in order of weight:

1. The dominant cost is not training. It is building window features
   (pandas) and extracting clusters (scipy), both CPU-only. A GPU does not
   touch either, so it would shorten perhaps 30-60 minutes of a ~4 hour run.
2. A CPU session runs for 12 hours against a GPU session's 9, and the run
   needs ~4.
3. GPU quota is limited and stage 3.4's GNN genuinely needs it. Spending it
   here would be spending it twice.

Run both cells in **one session**, in this order. `train` writes the models
into `/kaggle/working/models/`, and `evaluate` reads them back; running
`evaluate` first exits 2 rather than guessing.

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py train
```

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py evaluate
```

Both default to the full matrix: **2 variants x 2 lookbacks x 5 seeds = 20
boosters**, evaluated on val / test / stress.

### What `train` does

Pools every rule-3-valid training detection time into one matrix - 13
windows at L = 24, 9 at L = 48 - and fits one booster per (variant,
lookback, seed) with frozen hyperparameters, identical everywhere.

The matrix is built **once per lookback**; the no-format variant is a
column slice of it, and the slice is reused across seeds. So the two
variants see byte-identical windows, labels and weights, which is what an
ablation has to hold fixed, and nothing is featurised twice.

Expect the matrix build to be the memory peak. **Measured at L = 24:
9.9 GB** for 6.71M rows x 66 features - more than the ~7 GB first estimated
here, because `np.vstack` holds the per-window matrices and the pooled copy
at the same time. L = 48 has 9.11M rows, 1.36x as many, so scaling the part
that grows gives **~12-13 GB**. Both are inside a CPU notebook's 30 GB;
neither would be comfortable on a GPU notebook's ~13-16 GB, which is one
more reason this step is CPU. **Corrected 2026-11-10:** that comparison was
wrong - it put XGBoost's host-memory peak against the GPU's *device*
memory. A Kaggle GPU notebook has **33.7 GB of system RAM** (measured by
`gpu-check`), so the matrix would fit there. This step stays on CPU for the
reason that always applied: its bottleneck is pandas and scipy, and stage
3.4 needs the GPU quota. If L = 48 does run out, the fix is to
preallocate the pooled matrix and fill it per window instead of stacking -
report the number rather than working around it.

### What `evaluate` does

Tau comes from the **model grid** - `0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999` - chosen on
validation only by campaign-level F1. The sanity scorers keep their own
grid; this one exists because a class-weighted model's scores do not just
spread over the unit interval, they **pile up against 1**. The near-1 tail
was added on 2026-10-18 after the quick check chose 0.99, the old top, with
F1 rising 0.128 -> 0.543 across that last interval. The extra taus are
nearly free: clustering cost scales with the edges kept, and almost nothing
survives a threshold of 0.999.

It reports, for every model and the oracle:

- all four subsets, `all` / `fully_observed` / `no_shared_accounts` /
  `oracle_reachable`, with `recall_relative_to_ceiling` beside each;
- the spread across seeds, mean and [min, max] and sd, so a difference is
  read against seed noise rather than taken from one run;
- recall and lead time at the 10 / 50 / 100 per-day alert budgets;
- three **paired** comparisons: each model against the oracle ceiling, each
  variant's L = 24 against its L = 48, and the format ablation against the
  full model at matched seeds. Earliness is only ever compared this way;
- the **seconds each tau cost**, clustering only - the score cache is warmed
  first, so featurisation is a separate step and the taus are comparable.

**If a chosen tau lands on an edge of the grid, `evaluate` writes every
report and then exits 3 with a STOP message.** An edge tau is a boundary,
not a maximum. Send the reports back and decide the grid; it is frozen for
every model and condition, so widening it is a decision, not a detail.

### Expected runtime

Now **measured at L = 24** by the quick check (one variant pair, seed 0)
rather than estimated: **215s per booster** to train and **~3.5 min per
model** to evaluate. The L = 48 column scales those by what actually grows:
1.36x the training rows (9.11M against 6.71M) and 1.97x the evaluation
edges (41.0M against 20.8M), from `features_smoke_scan_full.csv`.

| Step | L = 24 | L = 48 | Total |
|---|---:|---:|---:|
| `train`: build the matrix | ~50s | ~65s | ~2 min |
| `train`: per booster | 215s (measured) | ~290s | - |
| `train`: 10 boosters each | ~36 min | ~49 min | **~1.5 h** |
| `evaluate`: per model | ~4 min | ~7 min | - |
| `evaluate`: 10 models each | ~40 min | ~70 min | **~2 h** |

**About 3.5-4 hours end to end**, inside a 12-hour CPU session with room to
spare. The earlier estimate had training at ~35 minutes; it is ~1.5 hours,
because a booster takes 215s rather than the ~90s guessed from scaling the
sample. Evaluation came in slightly cheaper than estimated. The 16-tau grid
adds four thresholds above 0.99 that keep almost no edges, so they cost
little.

What is still an estimate: Kaggle's CPU speed on the day, and the L = 48
numbers, which are scaled rather than measured.

**If the budget is tight**, run `--seeds 0,1` first (about 1.5 hours end to
end) to confirm the numbers look sane, then the full five. The models
persist in `/kaggle/working`, so `evaluate` can also run in a later session
from a saved notebook output instead of repeating `train`.

### What to send back

`train_baseline.md` and `evaluate_baseline.md`, plus the CSVs - especially
`evaluate_baseline_across_seeds.csv`, `evaluate_baseline_paired.csv` and
`evaluate_baseline_tau_curve.csv` (which carries the per-tau cost). Also
say whether `evaluate` exited 0 or 3.

## 5h. Stage 3.4.0: GPU environment check

**This is the first step that needs a GPU.** In the notebook settings set
**Accelerator: GPU** (T4 x2 or P100 - either is fine for this check) before
running anything below. Everything in 5a-5g runs on CPU and should stay
there; GPU quota is weekly and limited.

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py gpu-check
```

No model, no data, no training. It answers the questions that would
otherwise be discovered halfway through a training session:

- **What the accelerator is**: name, total and free memory, compute
  capability, and how many there are.
- **What the host is**: Python version, CPU count, total and available
  system RAM.
- **What installs**: `torch_geometric` first — which every GPU subcommand
  does for itself, so no command depends on another having run earlier in
  the session — then each optional extension
  (`pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`) from the
  PyG wheel index matching this image's torch and CUDA. Each is attempted
  separately and reported as installed or failed **without aborting the
  run** - none of them is required, since GINEConv and neighbour sampling
  both work without them. They are faster when present, and their wheels
  often lag the newest Python, which is exactly what this check is for.
- **Whether it actually runs**: a GINEConv forward and backward at the real
  feature widths - 3 layers, hidden 64, 54 edge features plus a direction
  flag, 6 node features, every edge also inserted reversed - at 50k, 200k
  and 500k edges, reporting seconds and peak GPU memory for each. An
  out-of-memory error is caught and reported as a result rather than a
  crash, and the probe stops at the first size that fails.

It writes `gpu_check.{md,json}` and **exits 2** if there is no usable CUDA
device or the smoke test does not pass - after reporting everything it
could still determine, so a failed check is still informative.

`--no-install` reports the environment without pip installing anything.
`--probe-edges` changes the smoke-test sizes, e.g. `--probe-edges
50000,200000,500000,1000000` to push further up.

**What to send back:** `gpu_check.md`, and whether the command exited 0.

### Measured, 2026-11-10 (commit 2755139, exit 0)

Python 3.13.16, torch 2.11.0+cu130, PyG 2.8.1, **all four extensions
installed**, **2 x Tesla T4 (15.6 GB each)**, **33.7 GB system RAM**.
GINEConv 3x64 forward+backward: 50k edges 251 MB, 200k 936 MB, 500k
2,302 MB, each in 0.07-0.16s.

That is **~4.6 MB per 1,000 edges**, which projects ~5.1 GB for the largest
L = 24 training window (1.11M edges) and ~8.6 GB for the largest L = 48 one
(1.87M) - against the shape-based ~16.6 GB that justified neighbour
sampling in the first place. So full-batch training may fit both lookbacks,
and the sampling decision is reopened. Section 5i measures it on the real
windows before anything is decided.

## 5i. Stage 3.4.1 evidence: the real windows

Still **Accelerator: GPU**. No training, no saved model. **It does not
depend on 5h having been run first**: every GPU subcommand installs
`torch_geometric` itself if the session lacks it, so the first GPU command
of a session pays ~30s for the install and the rest find it already there.
A fresh session that ran this before `gpu-check` used to fail with "No
module named 'torch_geometric'"; it no longer can.

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py gpu-probe-windows
```

`gpu-check` measured synthetic graphs with random edges. This measures the
real thing, because a random graph's account-degree distribution is not a
real window's and the 4.6 MB/1k projection is only linear until it is not:

- the **largest rule-3 training window** at each lookback - L = 24 at
  2022-09-02 00:00 (1,114,921 edges) and L = 48 at 2022-09-03 00:00
  (1,869,370) - through GINE 3x64 + the edge readout MLP + the
  class-weighted loss + an Adam step, forward and backward;
- the **largest evaluation window**, 2022-09-10 00:00 at L = 48 (1,137,240
  edges), scored unchunked, to find out whether chunking is needed at all.

Features are real, from a spec fitted on **every** rule-3 training window
exactly as training fits it: a spec fitted on one window would have
narrower vocabularies and fewer edge features, and would understate the
memory. One GPU only (`cuda:0`) - a window has to fit on a single card.
Three training steps are timed, the first reported separately because it
pays for allocator warm-up.

It reports peak GPU memory **allocated and reserved** (reserved is what the
allocator holds, so it is what runs out), MB per 1,000 edges, step time,
and the build+featurise time per window. The largest windows are found by a
full scan rather than remembered, and checked against `features-smoke`'s
record so a mismatch shows. An OOM is caught and reported as a
measurement. Writes `gpu_probe_windows.{md,json,csv}`; exits 3 if any probe
failed.

**The decision rule, fixed before the run:** full-batch is proposed only if
the largest L = 48 window peaks **below 12 GB reserved**, leaving headroom
on a 15.6 GB card for real-data variation and fragmentation. Whichever
regime wins applies to **both** lookbacks, so the lookback comparison is
never confounded by the training method.

Expect a few minutes: most of it is the pipeline build (~32s), the scan,
and fitting two feature specs over 13 and 9 training windows; the GPU steps
themselves are fractions of a second.

`--allow-cpu` runs the path without a GPU and measures nothing - it is for
verifying the code off-session, not for numbers. `--steps` changes how many
training steps are timed (default 3).

**What to send back:** `gpu_probe_windows.md` and the exit code. The
proposal in 3.4.1 is written from it.

**A note on memory.** Stage 3.3's 15.9 GB peak is XGBoost's pooled
*training matrix* - 9.1M rows x 66 features held at once. It says nothing
about the GNN, which processes one window at a time under neighbour
sampling and never materialises that pool. The GNN's memory is measured
here and in 3.4.3, not inferred from 3.3.

## 5j. Stage 3.4.3: the GNN quick check - RUN, 2026-11-10

**Done** (commit `8d8929c`, both sessions exit 0). Kept here because 5k
repeats the same two-session shape at full size, and because the measured
numbers at the end are what 5k is sized from.

**Two sessions**, because training wants a GPU and evaluation does not.
Run them in this order; the second reads what the first saved.

### Session A - training, Accelerator: GPU

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py train-gnn \
    --lookbacks 24 --variants gnn_centralized --seeds 0
```

One model: L = 24, the full variant, seed 0. Full-batch - one whole window
per step, 13 steps an epoch - with GINE 3x64, layer normalisation after
each convolution, and the `log1p` of the heavy-tailed node counts applied
model-side. Up to 100 epochs, stopping when validation edge-level average
precision has not improved for 10.

It prints, per epoch: training loss, validation average precision,
**max |logit|** and **peak GPU memory**. What to watch:

- the loss should start near **0.693** (`ln 2`), which the balanced-mean
  loss gives for any window - if it starts at 10,000 the normalisation is
  not working;
- `max |logit|` should stay in single or low double digits, not the 10⁴-10⁵
  the unnormalised probe implied;
- peak reserved memory should sit near the **6.39 GB** the probe measured
  for the largest L = 24 window, and must not climb epoch over epoch, which
  would mean fragmentation.

Models are saved **the moment each finishes**, under
`/kaggle/working/models/gnn/<variant>_L<L>_s<seed>/`, and a model already
there is skipped on a re-run - so an interrupted session loses at most the
model in flight. `--retrain` forces a rebuild. The feature spec and node
preprocessor are fitted **once per lookback** into
`models/gnn/spec_L<L>/` and reused by every variant and seed, because that
fit is the 14.1 GB host-memory step.

On out-of-memory it reports the window, epoch and peak, writes a partial
report and **exits 4**. It does not fall back to sampling or chunking.

**Save the notebook output** so session B can read the models.

### Session B - evaluation, CPU (no accelerator)

Add session A's output as a notebook input, then:

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py evaluate-gnn \
    --lookbacks 24 --variants gnn_centralized --seeds 0 \
    --model-dir /kaggle/input/<session-A-output>/models/gnn
```

Window-major: each evaluation window is featurised **once per variant** and
scored by every seed of it, keeping only the scores. Tau is then chosen on
validation by campaign-level F1 from the model grid - the same grid, the
same rule, the same code as XGBoost - and everything is reported against
the oracle on all four subsets with the paired comparisons.

It writes a partial report after **every** configuration, so an interrupted
evaluation session loses at most one. If a chosen tau lands at an edge of
the grid it says so and **exits 3**.

### What to send back

From A: `train_gnn.md` and the per-epoch table - particularly the first
epoch's loss, the `max |logit|` column and the peak memory. From B:
`evaluate_gnn.md` and `evaluate_gnn_inference.csv`, which carries the
measured featurise and forward cost per edge on Kaggle's CPU. Plus both
exit codes, and the GPU time session A consumed.

Those numbers size stage 3.4.4's plan for the full 2 variants x 2 lookbacks
x 5 seeds.

### Measured, 2026-11-10 (commit 8d8929c, exits 0 and 0)

Training: first epoch's loss ~0.69 (`ln 2`), `max |logit|` 1.2 -> 14, peak
**7.43 GB flat across epochs** (the probe measured 6.39 GB for the largest
window's forward+backward alone; training adds optimizer state and
fragmentation across 13 windows), **~7s an epoch**, best epoch 45 at
validation AP 0.419, stopped at 55.

Evaluation: tau 0.995, interior. Test recall 0.661 (37/56, 0.804 of the
ceiling) at **33.3 false alarms a day**, against the stage 3.3 XGBoost's
same 0.661 at **3.4 a day**. Inference on Kaggle's CPU over 24 windows and
11.17M edges: **92.6s to featurise them once (8.28 us/edge)** and **56.4s
for one model's forward passes (5.08 us/edge)**, with tau selection over
the 16-value grid at **37.3s a model**.

Those four numbers are what section 5k's estimates are built from.

## 5k. Stage 3.4.4: the full run, and the direct comparison

**Three sessions.** Two GPU training sessions, one per lookback, then one
CPU session that does everything else: 20 GNN models evaluated, the XGBoost
baseline **retrained and re-evaluated**, and the direct comparison.

### Why session B retrains XGBoost

Stage 3.3 ran in an **interactive** session, so its `/kaggle/working` was
discarded when the session ended - the 20 boosters no longer exist. Two
things follow:

1. Session B runs `train` before `evaluate`: **1.44h + 1.59h measured**, CPU
   only. The GPU is not wanted for either; stage 3.3's reasons still hold
   (the cost is pandas and scipy, and the quota belongs to the GNN).
2. The retrain is also a **regression check**. Same seeds, same code, same
   data, so the re-evaluated numbers must reproduce the recorded stage 3.3
   results: the full model at L = 48 with test recall **0.679** and
   **0.760** of the ceiling, and no tau at a grid edge. If they differ,
   stop and report it - something changed that should not have.

Session B is run in **commit mode** ("Save & Run All"), not interactively,
so nothing depends on the browser staying open. Its own
`/kaggle/working` is then saved as a notebook output, which is what keeps
the GNN models, the XGBoost boosters and every report.

---

### Session A1 - GNN training at L = 24

**Session options:** Internet **ON**, Accelerator **GPU** (T4 x2 or P100;
only `cuda:0` is used - a window has to fit on one card). Input: the IBM
dataset, as in section 1.

```python
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

```python
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
```

```python
!pip install -q -r requirements.txt
```

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py train-gnn --lookbacks 24
```

```python
!ls /kaggle/working/models/gnn/ && du -sh /kaggle/working/models/gnn/
```

No `--variants` or `--seeds`, so this trains **all ten models** of this
lookback: both variants x five seeds. `torch_geometric` installs itself on
the first GPU command of the session (~30s).

**Expected cost**, from the 3.4.3 measurements (pipeline 30.6s, feature-spec
and node-preprocessor fit 54.4s, then 463.4s for one model of 56 epochs -
which is ~95s to featurise and cache 17 window graphs on the host plus ~7s
an epoch):

| | Ten models at 56 epochs | If every model runs the 100-epoch cap |
|---|---:|---:|
| L = 24 | **~1.4h** | ~2.2h |

Either way it is well inside the 9-hour GPU limit. The graphs are rebuilt
per model, which is ~16 minutes of the total across ten models; the feature
spec is fitted **once** and shared, because that is the expensive step
(14.1 GB of host RAM at L = 48, 10.6 GB measured at L = 24).

**What to watch**, per epoch:

- the loss starts near **0.693** (`ln 2`), which the balanced-mean loss
  gives for any window - 3.4.3 measured ~0.69;
- `max |logit|` stays in single or low double digits - 3.4.3 went 1.2 to
  14.1 over 56 epochs;
- peak reserved memory sits near **7.4 GB** (measured at L = 24) and does
  **not** climb epoch over epoch, which would mean fragmentation.

Models are saved the moment each finishes, under
`/kaggle/working/models/gnn/<variant>_L<L>_s<seed>/`, and one already there
is skipped on a re-run - so an interrupted session loses at most the model
in flight. `--retrain` forces a rebuild. On out-of-memory it reports the
window, epoch and peak, writes a partial report and **exits 4**; it does
not fall back to sampling or chunking.

**Save the notebook output.** Session B needs it.

### Session A2 - GNN training at L = 48

Identical to A1 except the lookback:

```python
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py train-gnn --lookbacks 48
```

| | Ten models at 56 epochs | At the 100-epoch cap |
|---|---:|---:|
| L = 48 | **~1.9h** | ~3.1h |

L = 48 has 9 training windows against 13, each about twice the edges, so
~1.4x the edges an epoch. Two numbers differ from A1 and both matter:

- **Peak GPU memory ~12.4 GB**, extrapolated from the probe's 10.70 GB for
  the largest L = 48 window and the 1.16x that training added at L = 24.
  That is ~80% of a 15.6 GB card. It is expected to fit and it is the
  tightest point of the whole stage, so watch the peak column; an OOM is
  reported as exit 4 with the window and epoch, not a crash.
- **Host RAM ~14.1 GB** during the feature-spec fit (measured by the
  probe), against the notebook's 33.7 GB.

**Save this output too.**

### Session B - evaluation, retraining and comparison (CPU, commit mode)

**Session options:** Internet **ON**, Accelerator **None**. Inputs: the IBM
dataset, **A1's output**, and **A2's output**. Then **Save Version → Save &
Run All (Commit)** rather than running interactively - the whole thing is
~5-7 hours.

```python
# B1 - dataset path, as always
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

```python
# B2 - clone and install
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
!pip install -q -r requirements.txt
```

```python
# B3 - find the two training sessions' model trees and put them in one place.
#      `--model-dir` takes a single path, and A1 and A2 are separate inputs.
import pathlib, shutil, subprocess

found = sorted(
    subprocess.run(
        ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
         "-path", "*/models/gnn"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
)
print("GNN model trees found under /kaggle/input:")
for path in found:
    print("  ", path)
assert found, "no models/gnn directory under /kaggle/input - attach A1 and A2"

dest = pathlib.Path("/kaggle/working/models/gnn")
dest.mkdir(parents=True, exist_ok=True)
for path in found:
    for child in sorted(pathlib.Path(path).iterdir()):
        target = dest / child.name
        if target.exists():
            print("   already present, left alone:", child.name)
            continue
        shutil.copytree(child, target)

models = sorted(d.name for d in dest.iterdir() if d.name.startswith("gnn_"))
specs = sorted(d.name for d in dest.iterdir() if d.name.startswith("spec_"))
print(f"{chr(10)}{len(models)} models, {len(specs)} specs in {dest}")
for name in models:
    print("  ", name)
print("  ", specs)

# Fail here rather than three hours in: a missing model would be retrained,
# and a CPU session cannot train one.
assert len(models) == 20, f"expected 20 GNN models, found {len(models)}"
assert specs == ["spec_L24", "spec_L48"], specs
```

```python
# B4 - retrain the XGBoost baseline. ~1.44h. The stage 3.3 boosters are gone
#      (that run was interactive), and `evaluate` needs them in this session.
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py train
```

```python
# B5 - evaluate XGBoost. ~1.59h. This is what writes evaluate_outcomes.csv,
#      the per-campaign records stage 3.3 never saved.
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py evaluate
```

```python
# B6 - evaluate all 20 GNN models, both lookbacks, from the tree B3 built.
#      ~35-80 min at L = 24 and ~70-160 min at L = 48.
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py evaluate-gnn
```

```python
# B7 - the direct comparison. Seconds: it reads the two outcome files.
!cd /kaggle/temp/repo/backend && python scripts/kaggle/kaggle_runner.py compare-models
```

```python
# B8 - what to download
!ls -lh /kaggle/working/
!du -sh /kaggle/working/models/
```

**The order is forced**, cell by cell:

- B3 before everything, because it is the cheap check. If A1 or A2 was not
  attached, this fails in a minute instead of after three hours of XGBoost.
- B4 before B5: `evaluate` reads the boosters `train` writes, and exits 2
  rather than guessing if they are missing. Both default to
  `/kaggle/working/models/`, and the GNN models sit in the `gnn/`
  subdirectory of it, so nothing collides.
- B5 and B6 in either order, but both before B7: `compare-models` reads
  `evaluate_outcomes.csv` and `evaluate_gnn_outcomes.csv` and **exits 2** if
  either is missing or if the two share no configuration. XGBoost first
  because its cost is known to 1%, which leaves the whole remaining margin
  for the one uncertain step.

**Total: ~4.8-7.1 hours** in a 12-hour session, so there is 5-7 hours of
margin. Where the range comes from, and the one real risk: clustering cost
scales with the edges kept, so a **low** budget tau is the expensive case -
stage 3.3 spent 194s on one model at its 10/day tau against 3.7s at its F1
tau, a 50x spread. The GNN alerts more freely at its F1 tau, so its 10/day
tau will be higher and cheaper, but its 50 and 100/day taus will be lower
and dearer, and those are new work no measurement covers yet.

Nothing is lost if B6 runs out of time: it writes a partial report after
every configuration, and because session B saves its own output, a later
CPU session can attach it and finish from the saved models - no GPU, no
retraining.

### What to send back

From A1 and A2: `train_gnn.md` from each - the per-epoch tables, the peak
memory column, and the GPU time each session used. From B: `evaluate_gnn.md`
with `evaluate_gnn_budgets.csv`, `evaluate_gnn_edge_metrics.csv` and
`evaluate_gnn_clusters.csv`; `train_baseline.md` and `evaluate_baseline.md`
(the regression check against stage 3.3); and the three `compare_models.*`
files. Plus every exit code.

Stage 3.4.5, the Phase 3 gate report, is written from those - by paired
comparisons only, per lookback, per seed, at the F1-chosen taus and at every
budget, with coverage reported separately.

## 6. Later stages

`train` and `evaluate` are implemented and are section 5g above: the stage
3.3 XGBoost baseline, which runs on **CPU** - see 5g for why, and for the
measured runtime and memory.

Stage 3.4's GNN needs **Accelerator: GPU**. Its environment check is
section 5h above and is runnable now; the model itself is not yet written.
The training regime is frozen (neighbour sampling at both candidate
lookbacks, so the lookback comparison is not confounded by the training
method; see the README decision log), and `features_smoke` has measured
the window sizes it has to fit.

Stage 3.4 runs in sub-stages, each stopping for approval. **3.4.0** (the
environment check, section 5h), **3.4.1** (the design, sized against
section 5i's measurements), **3.4.2** (implementation, tests and a local CPU
smoke run) and **3.4.3** (the quick check, section 5j) are all done.
**3.4.4** is section 5k above, awaiting approval of the plan in the README.
Then **3.4.5**, the Phase 3 gate report comparing the GNN with XGBoost by
paired comparisons only, which is written from 5k's output and needs no
further session.

Everything in this document other than 5h runs on CPU.

## 5l. Stage 4.1: the isolated condition on the full data — session A DONE, B and C proposed

The floor every federated condition is measured against: each institution
trained on its own view, scoring its own view, clustering inside it. **CPU
only** — no GPU, so no weekly quota is consumed. Everything here is the
frozen Phase 4 scope and is the runner's default: targets 1.00 / 0.75 / 0.50
/ 0.25, seeds 0 / 1 / 2, **L = 48 only**, which is **96 institution-cells**.

### Why three sessions and not one

Estimated **7-14 CPU hours** in total, against a 12-hour session limit. One
session would fit the optimistic end and lose everything at the pessimistic
end, so the work is split where it is already resumable:

| Session | Command | Inputs | Estimate | Status |
|---|---|---|---:|---|
| **A** | `train-isolated` (all 12 cells, 96 boosters) | IBM dataset | 2.5-4 h | **done 2026-11-15: 2.25 h, exit 0** |
| **B** | `evaluate-isolated --targets 1.0,0.75` (6 cells) | dataset + **A** | **2.5-5.5 h** | **done 2026-11-15: exit 3, one cell excluded at the grid top** |
| **C** | `evaluate-isolated` (no `--targets`: evaluates 0.50 and 0.25, **reloads** B's six) | dataset + **A** + **B** | **2.5-5.5 h** | next |

**Session B's outcome, and what it means for C.** Exit **3** with one
`(institution, operating point)` cell EXCLUDED — seed 0, institution 0, the F1
point, rejected at 0.9999999 — which is a reportable finding, not a failure.
**C will exit 3 again for that same cell**, because C assembles its report
from B's reloaded cells plus its own and reports their exclusions as its own.
That is the report describing itself. Two other things B found that change how
C is read: the per-institution budgets **do** bind on the full data
(`binding_fraction = 1.0`, against 0 of 8 on the sample), and two cells chose
an F1 tau of 0.1, one step above the grid floor — so a **low**-edge stop is a
live possibility at the lower targets C runs.

**Session A, measured.** 8,094s over the instrumented steps (8,130s in the
log), peak **5,353 MB**, 83.8s mean per booster (79.1-111.1s), 96 of 96 cells
`normal`. It beat the optimistic end of its estimate, which does **not** mean
B and C can be merged: training was the half of the estimate anchored on a
measured Phase 3 number, evaluation is the half that is not, and it runs over
seven operating points. Evidence:
`docs/reference/train_isolated.{md,csv,json}`.

All three in **commit mode** (Save Version → Save & Run All), Internet **ON**,
Accelerator **None**.

**C is the session that produces the complete report.** It runs with no
`--targets` so the loop covers all four; the two targets B already did are
found on disk and loaded in seconds, and the combined
`phase4_isolated_*.csv` it writes therefore cover all twelve cells. If B
comes in near the optimistic end, C can be started with all four targets
anyway — it will simply evaluate the remaining six cells.

**If any session runs out of time**, nothing finished is lost: `train-isolated`
saves each booster as it completes and skips it next time, and
`evaluate-isolated` writes each `(target, seed)` cell's rows as it completes
and resumes past them. Re-run the same session with the previous attempt's
output attached as an extra input and the copy cell below will pick it up.

### Where the estimates come from

Measured on the full data in stage 3.3, scaled by the measured per-institution
view size of **~23.5%** of the central row count:

| | Central, measured | Per institution, estimated | Per institution, **measured in A** |
|---|---:|---:|---:|
| build pipeline | 35.7s (3.2 GB) | once per session | **35.1s (3,168 MB)** |
| build the training matrix, L = 48 | 114.0s (15.9 GB peak) | ~30s, ~1.5-4 GB | folded into the booster time |
| train one booster, L = 48 | ~235s | ~60-110s | **83.8s mean, 79.1-111.1s** |
| evaluate one model, 3 splits | ~286s | ~150-280s, now over **7** operating points | still an estimate — this is what B and C measure |

Memory is not a constraint here: the central matrix peaked at 15.9 GB and an
institution's is roughly a quarter of it, inside a 30 GB CPU notebook with
room to spare. Session A confirmed it — **5,353 MB peak**, which is the
pipeline's 3,168 MB plus the institution's matrix, and above the 1.5-4 GB
figure above only because that figure was for the matrix alone.

### Session A — training, CPU, commit mode

**Inputs:** the IBM dataset only.

```python
# A1 - dataset path, as always
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

```python
# A2 - clone and install (~1-2 min)
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
!pip install -q -r requirements.txt
!git -C /kaggle/temp/repo rev-parse --short HEAD
```

```python
# A3 - pre-flight: fail in one minute rather than three hours in.
#      Checks the frozen Phase 4 scope and the extended tau grid are the
#      ones this run assumes. Sessions B and C run this cell too, AFTER
#      their input check (B3 / C3): inputs first, because a missing booster
#      is the likelier failure and the cheaper one to detect.
import sys
sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp import constants, evaluation

assert constants.PHASE4_SEEDS == [0, 1, 2], constants.PHASE4_SEEDS
assert constants.PHASE4_LOOKBACK_H == 48, constants.PHASE4_LOOKBACK_H
assert constants.K_INSTITUTIONS == 8
assert constants.VISIBILITY_TARGETS == [1.0, 0.75, 0.5, 0.25]
grid = evaluation.MODEL_TAU_GRID
assert len(grid) == 22, len(grid)
assert grid[-1] == 0.9999999 and 1.0 not in grid
print("scope and grid OK:", len(grid), "taus, top", grid[-1])
print("cells:", len(constants.VISIBILITY_TARGETS) * len(constants.PHASE4_SEEDS)
      * constants.K_INSTITUTIONS)
```

```python
# A4 - train 96 boosters. ~2.5-4 h. Resumable: a finished booster is skipped.
#      subprocess rather than `!` so the exit code is visible in the log:
#      0 = clean, anything else must be read before the numbers are used.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "train-isolated",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("train-isolated exit code:", code)
```

```python
# A5 - what to download, and the shape of what was produced
!ls -la /kaggle/working/*.csv /kaggle/working/*.md /kaggle/working/*.json
!find /kaggle/working/models/isolated -maxdepth 1 -type d | wc -l
!python -c "import pandas as pd; d = pd.read_csv('/kaggle/working/train_isolated.csv'); print(d.groupby('tier').size()); print(d[['n_rows','n_positive']].describe().loc[['min','mean','max']])"
```

### Session B — evaluating the first two targets, CPU, commit mode — DONE 2026-11-15

**Inputs:** the IBM dataset **and session A's output**.

**Cell order in B and C matters:** the input check comes **first**, then the
pre-flight, then the long cell. A missing booster is both the likelier failure
and the cheaper one to detect, and `evaluate-isolated` never trains, so it
cannot paper over one. Paste B1-B6 in order.

```python
# B1 - identical to A1 (dataset path)
```

```python
# B2 - identical to A2 (clone and install, ~1-2 min)
```

```python
# B3 - bring A's boosters into /kaggle/working, where the runner looks, and
#      ASSERT THE INPUTS IN THE FIRST MINUTE. `evaluate-isolated` never
#      trains, so a missing or half-written booster stops the run hours in
#      rather than being quietly recreated - this cell is what makes that
#      failure happen now instead.
#
#      Note on specs: unlike the GNN sessions there is no shared spec_L24 /
#      spec_L48 pair to check. Each institution fits its OWN feature spec on
#      its own view - that is the point of the condition - so every booster
#      directory carries its own `feature_spec.json`, and all 96 are checked
#      below.
#
#      The sys.path insert is here and not only in the pre-flight: this cell
#      loads boosters, so it needs `fraudcamp` importable, and it now runs
#      BEFORE the pre-flight that used to be the first cell to set the path.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
print("isolated model trees found:", found)
assert found, "no models/isolated under /kaggle/input - attach session A's output"

dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        target = dest / cell.name
        if not target.exists():
            shutil.copytree(cell, target)

cells = sorted(d for d in dest.iterdir() if d.is_dir())
print(len(cells), "institution-cell directories")

REQUIRED = ("xgb_model.json", "feature_spec.json", "model_meta.json")
trained, untrained, broken = [], [], []
for cell in cells:
    if (cell / "untrained.json").exists():
        untrained.append(cell.name)       # a real outcome, not a failure
        continue
    missing = [name for name in REQUIRED if not (cell / name).exists()]
    empty = [
        name for name in REQUIRED
        if (cell / name).exists() and (cell / name).stat().st_size == 0
    ]
    if missing or empty:
        broken.append((cell.name, missing, empty))
    else:
        trained.append(cell.name)

print(f"  trained: {len(trained)}   untrained (recorded): {len(untrained)}")
if untrained:
    print("  untrained institutions:", untrained)
if broken:
    for name, missing, empty in broken:
        print("  BROKEN", name, "missing:", missing, "empty:", empty)

assert len(cells) == 96, f"expected 96 institution-cells, found {len(cells)}"
assert not broken, f"{len(broken)} booster(s) incomplete - re-run session A"

# every trained booster must also load: a file that exists but will not
# parse is the failure this check is really for
from fraudcamp.models import xgb_baseline
sample = trained[:: max(1, len(trained) // 8)]
for name in sample:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print(f"  spot-loaded {len(sample)} boosters, all at L = 48 with 66 features")
```

```python
# B4 - the pre-flight, identical to A3: frozen Phase 4 scope and the 22-point
#      tau grid. It runs AFTER B3 because a missing booster is the likelier
#      and cheaper failure, and because the boosters B3 just loaded were
#      trained against this same grid.
```

```python
# B5 - evaluate targets 1.00 and 0.75 (six cells). ~2.5-5.5 h.
#      Exit 3 is NOT a crash: it means a chosen tau sat at an edge of the
#      grid. Read the message before using anything.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-isolated",
     "--config", "configs/kaggle.yaml", "--targets", "1.0,0.75"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-isolated exit code:", code)
```

```python
# B6 - what came out
!ls -la /kaggle/working/phase4_isolated*  /kaggle/working/isolated_cells | head -40
```

### Session C — SUPERSEDED 2026-11-16, do not run. See [5n](#5n-stage-41-the-10-remaining-cells--sessions-c1-to-c3-supersedes-5ls-session-c)

> Its input check asserts **six** of session B's cells. B produced
> **two**, so the cell fails in 44 seconds - correctly. The replacement
> plan is §5n; this section is kept only as the record of what was
> planned before B's outcome was known.

### Session C (superseded) — the remaining targets and the complete report

**Inputs:** the IBM dataset, **session A's output**, **session B's output**.
Six cells, C1-C6. B's staging cell and B's input assertion merge into **C3**,
because they guard the same failure and the merged check is the stronger one:
B's cells missing is not a crash but a silent **re-evaluation of all twelve**,
which does not fit in a 12-hour session.

```python
# C1, C2 - as B1 and B2 (dataset path; clone and install)
```

```python
# C3 - stage BOTH inputs and assert BOTH, before anything expensive. A's 96
#      boosters, exactly as B3, and B's six finished cells.
#
#      The assertion on B's cells is the one that earns this cell. A missing
#      booster makes the run fail; missing CELLS make it succeed slowly - it
#      re-evaluates the six targets B already did, so the session needs ~5-11 h
#      instead of ~2.5-5.5 and runs out of the 12-hour limit having written no
#      complete report. Cheap to check, expensive to discover at hour nine.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

# --- A: the boosters ---------------------------------------------------
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
print("isolated model trees found:", found)
assert found, "no models/isolated under /kaggle/input - attach session A's output"

dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        target = dest / cell.name
        if not target.exists():
            shutil.copytree(cell, target)

cells = sorted(d for d in dest.iterdir() if d.is_dir())
REQUIRED = ("xgb_model.json", "feature_spec.json", "model_meta.json")
trained, untrained, broken = [], [], []
for cell in cells:
    if (cell / "untrained.json").exists():
        untrained.append(cell.name)       # a real outcome, not a failure
        continue
    missing = [name for name in REQUIRED if not (cell / name).exists()]
    empty = [
        name for name in REQUIRED
        if (cell / name).exists() and (cell / name).stat().st_size == 0
    ]
    if missing or empty:
        broken.append((cell.name, missing, empty))
    else:
        trained.append(cell.name)

print(f"{len(cells)} institution-cells: trained {len(trained)}, "
      f"untrained (recorded) {len(untrained)}, broken {len(broken)}")
for name, missing, empty in broken:
    print("  BROKEN", name, "missing:", missing, "empty:", empty)
assert len(cells) == 96, f"expected 96 institution-cells, found {len(cells)}"
assert not broken, f"{len(broken)} booster(s) incomplete - re-run session A"

from fraudcamp.models import xgb_baseline
sample = trained[:: max(1, len(trained) // 8)]
for name in sample:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print(f"  spot-loaded {len(sample)} boosters, all at L = 48 with 66 features")

# --- B: the six finished cells -----------------------------------------
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-name", "isolated_cells"],
    capture_output=True, text=True, check=True).stdout.split())
print("finished-cell directories found:", found)
cells_dir = pathlib.Path("/kaggle/working/isolated_cells")
cells_dir.mkdir(parents=True, exist_ok=True)
for tree in found:
    for item in sorted(pathlib.Path(tree).glob("*.csv")):
        target = cells_dir / item.name
        if not target.exists():
            shutil.copy2(item, target)

SUFFIXES = ("institutions", "taus", "ceilings", "summary", "pooled",
            "union", "outcomes")
expected = [f"t{round(t * 100):03d}_s{seed}" for t in (1.0, 0.75)
            for seed in (0, 1, 2)]
done = sorted({p.name.split("_summary")[0] for p in cells_dir.glob("*_summary.csv")})
print(f"{len(done)} cells staged: {done}")
missing_cells = [name for name in expected if name not in done]
incomplete = [
    (name, suffix) for name in done for suffix in SUFFIXES
    if not (cells_dir / f"{name}_{suffix}.csv").exists()
]
for name, suffix in incomplete:
    print(f"  INCOMPLETE {name}: no _{suffix}.csv")
assert not missing_cells, (
    f"session B's cells {missing_cells} are not staged. Attach session B's "
    "output, or C re-evaluates them and will not finish inside 12 h. If B "
    "genuinely finished fewer, split C with --targets instead of running it "
    "blind."
)
assert not incomplete, f"{len(incomplete)} cell file(s) missing - re-send B's output"
print("  all six of B's cells complete; C will reload them in seconds")

# --- D6 migration: B ran before the rename -----------------------------
#     Session B wrote `uncapped_10/50/100`; the code now writes
#     `per_inst_10/50/100`. C concatenates B's rows with its own, so without
#     this the single report would carry TWO names for one operating point and
#     every groupby would split it. Rewriting the staged copies is safe - the
#     originals in B's output are read-only - and it is a pure relabelling:
#     only the operating_point string changes, never a number.
renamed = 0
for path in sorted(cells_dir.glob("*.csv")):
    text = path.read_text(encoding="utf-8")
    if "uncapped_" in text:
        path.write_text(text.replace("uncapped_", "per_inst_"), encoding="utf-8")
        renamed += 1
print(f"  D6: rewrote uncapped_* -> per_inst_* in {renamed} staged cell file(s)")
leftover = [p.name for p in cells_dir.glob("*.csv")
            if "uncapped_" in p.read_text(encoding="utf-8")]
assert not leftover, leftover
```

```python
# C4 - as B4, VERBATIM: the scope and grid pre-flight, after C3.
```

```python
# C5 - all four targets: the six from B load in seconds, the six at 0.50 and
#      0.25 are evaluated. ~2.5-5.5 h. This writes the complete report.
#
#      Expect exit 3, not 0: B's excluded cell is part of the report this
#      session writes, so this session reports it. Read the message - "cells
#      EXCLUDED" is that known finding, while "STOPPING, and NOT recording" is
#      the low-edge rule firing on a NEW cell.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-isolated",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-isolated exit code:", code)
```

```python
# C6 - the headline, read back from the files rather than from the log.
#      The mean NEVER prints alone: the eight institutions are not
#      exchangeable (institution 0 holds the dataset's largest bank by
#      construction and carried 44% more training positives), so the median
#      and the range with their institutions print beside it.
import pandas as pd
pooled = pd.read_csv("/kaggle/working/phase4_isolated_pooled.csv")
union = pd.read_csv("/kaggle/working/phase4_isolated_union.csv")
print("cells:", pooled[["visibility_target", "seed"]].drop_duplicates().shape[0], "of 12")
print(pooled[(pooled.eval_split == "test") & (pooled.metric == "campaign_recall")]
      [["visibility_target", "seed", "operating_point",
        "mean_over_all_institutions", "mean_over_trained_institutions",
        "n_institutions_in_mean", "median_over_institutions",
        "min_over_institutions", "min_institution",
        "max_over_institutions", "max_institution",
        "n_binding_budgets", "n_cells_with_a_budget"]]
      .to_string(index=False))
print(union[(union.eval_split == "test") & (union.test_subset == "all")]
      [["visibility_target", "seed", "operating_point", "campaign_recall",
        "oracle_recall", "false_alarms_per_day", "summed_false_alarms_per_day",
        "duplicate_clusters", "n_institutions_operating"]].to_string(index=False))
```

### Exit codes, and what each one means

| Code | Meaning | What to do |
|---|---|---|
| **0** | every cell finished, no tau at a grid edge | send the files back |
| **3**, with `STOPPING, and NOT recording` | **only if** `--stop-on-low-edge-tau` was passed. Since **2026-11-16** a chosen F1 tau at the bottom of the grid is **recorded and flagged**, not a stop — see [5n](#5n-stage-41-the-10-remaining-cells--sessions-c1-to-c3-supersedes-5ls-session-c) | drop the flag and re-run; the finished cells reload |
| **3**, with cells `EXCLUDED` for a tau at the top | the optimum lies beyond the grid for those `(institution, operating point)` cells. They are **undefined**, excluded from the pooled means and listed with their rejected tau; every other cell is intact. **A session that RELOADS earlier cells reports their exclusions as its own**, so session C exits 3 for session B's excluded cell as well as any of its own — the exit code describes the report that session wrote, not only the cells it evaluated | send the files back. This is a reportable finding, not a failure |
| anything else | a real error | send the log back |

### What to send back

From the final session: `phase4_isolated.md`, `phase4_isolated.json`, and the
seven `phase4_isolated_*.csv` files, plus `train_isolated.{csv,json,md}` from
session A. They go to `docs/reference/`, and
`phase4_isolated_outcomes.csv` is what 4.2 and 4.3 are paired against.

**If the run stopped on a LOW-edge tau there is no combined report**, because
that check returns before the report is written. The finished cells are still
on disk and are the valuable output — send **`isolated_cells/`** (all seven
CSVs per cell) and the log instead, and a resumed session picks up from the
cell that stopped it. Send `isolated_cells/` whenever a session ends for any
reason other than exit 0 or exit 3-with-cells-excluded.

### What to expect, and what would be surprising

* **No institution untrained — SETTLED by session A.** 96 of 96 cells came
  out `normal`, thinnest 1,003 positives on ~2.0M rows. The prediction of
  "roughly 2M rows with several hundred to a couple of thousand positives"
  held: mean 1,598 positives per cell, rate 0.080%, class weights 435-4,015.
* **The thinnest positives at the 100 bin, not the 25 bin — CONFIRMED by
  session A**, and narrower than the sample suggested. The thinnest cell of
  all 96 is target 1.0, seed 1, institution 2. But the cross-institution
  spread is 2.42x at target 1.0 and then flat at 1.93 / 1.89 / 1.91 for 0.75 /
  0.50 / 0.25, where the sample showed a monotone decline. Full visibility is
  the outlier; below it, the degree of fragmentation does not measurably
  change how unevenly positives are spread.
* **Institution 0 will top almost everything, and that is structural.** It
  held the most training positives in 12 of 12 cells because
  `assign_institutions` gives the dataset's largest bank to institution 0
  every time. Expect it at or near the top of the per-institution tables too,
  and read the pooled mean with the range beside it rather than as a typical
  bank.
* **The union's oracle ceiling equal to the centralized ceiling — CONFIRMED
  by session B at target 1.00**, exactly: 0.8928571, 50 of 56 campaigns,
  against the centralized L = 48 ceiling of 0.893. This is the stage 4.1
  argument from the frozen visibility rule and it now holds on full data.
  Expect the same at 0.75, 0.50 and 0.25. A disagreement at any target would
  mean the argument is wrong and would matter more than anything else in the
  report.
* **But NOT equal recall — and that is the result so far.** At the same
  ceiling the union recovers 0.500 / 0.600 against centralized 0.760. The
  union's deficit is **scoring**, not visibility, and it leaves 0.16-0.26 of
  relative recall for 4.2 to act on. Expect the lower targets to show the
  same shape; what would be surprising is the gap closing as fragmentation
  increases.
* **Budgets that DO bind — REVERSED from the sample by session B.** Every
  per-institution budget point came back binding, `binding_fraction = 1.0`,
  against 0 of 8 on the sample: the full background is ~100x denser, so the
  grid floor is far above 100 alarms a day. The `per_inst_*` name therefore
  says the wrong thing and becomes `own_*` after this stage completes. Read
  `budget_binding` per cell, not the column name.
* **A low-edge F1 tau is RECORDED, not a stop** (policy changed
  2026-11-16, see [5n](#5n-stage-41-the-10-remaining-cells--sessions-c1-to-c3-supersedes-5ls-session-c)). It is most
  likely at target **1.0**, not at 0.50 and 0.25: an institution's own ceiling
  *rises* as the target falls (0.139 → 0.346 on validation), so seeing more of
  a campaign moves the F1 optimum **up**, away from the floor. A cell that
  lands on 0.05 is recorded at that tau, flagged `tau_at_low_edge`, and
  carries `f1_degenerate` when its whole curve was flat at ~0.
* **Nothing from 0.50 or 0.25 is quotable before this runs.** Those figures
  are still from the 1% sample and are marked development-only, and session
  B's two targets are reported in the README without committed CSVs behind
  them until this session's reports land.

## 5m. Phase 4, stages 4.2 / 4.4 / 4.3 — sessions D to J

Six sessions, CPU, commit mode. **Stage 4.3 is last on purpose**: sessions D,
E and F produce a complete `isolated` / `fedavg` / `centralized` result with
its comparison report, so if time runs out after F the phase still has an
answer and the only thing missing is the evidence condition, whose four
contrasts `compare-conditions` then lists as skipped rather than leaving
silently absent. That is verified, not assumed — see "Dropping 4.3" below.

### Why these session boundaries

Every estimate below is a **sample ratio applied to a full-data baseline**,
and the ratio is the only part that was measured on real code:

Measured over **all four cells in one process** - 2,609s for the whole chain
on the sample - which is the only way the halves of a ratio are comparable:

| Per cell (8 institutions) | Total, 4 cells | Ratio to isolated | Per-cell spread |
|---|---:|---:|---:|
| `evaluate-isolated` | 574.6s | 1.00 | 112.7 - 173.6s |
| `evaluate-fedavg` | 698.0s | **1.21** | 1.09 - 1.48x |
| `evaluate-evidence` (4.3a) | 488.9s | **0.85** | 0.58 - 1.19x |
| `evaluate-evidence` (4.3b) | 608.5s | **1.06** | 0.79 - 1.33x |
| `train-evidence` (all 8) | 110.7s | **1.46** vs isolated training | 27.1 - 28.3s |

**The finding that matters for sizing: no condition is meaningfully more
expensive than `isolated`.** All four evaluation ratios sit between 0.85 and
1.21, with a per-cell spread of about +/-40% that swamps the differences
between them. `fedavg` applies eight boosters instead of one and still costs
1.21x, because **featurisation dominates and is shared**: the raw features of
a view do not depend on which booster scores next, so they are built once and
the eight scalers are applied to them. Two tests pin that path to bit
equality against featurising once per booster.

> **These numbers replace an earlier version of this table that read 1.41 /
> 1.15 / 0.44 / 4.2.** Those came from a **single** cell, and every one of
> them was wrong: the one-cell `fedavg` was measured under different machine
> load, and the one-cell `train-evidence` included the cold boundary-account
> pass (58.8s against 27-28s for every later cell), which inflated it to 4.2x.
> The same `fedavg` cell was separately timed at 173s and 228s in two runs - a
> 25% spread from nothing but load. **Only ratios measured in one process
> transfer, and even they carry +/-40% per cell.** Which is why every session
> prints a CALIBRATION line and the runbook says to act on that instead.

The full-data baseline is `evaluate-isolated`, whose per-cell time is the one
number nobody has yet measured — session B ran six cells inside its 2.5-5.5 h
bracket, so **25-55 min per cell**. Everything below inherits that
uncertainty, which is why **every session prints a CALIBRATION line after its
first cell**, projecting the session from a measurement instead of from this
table. Act on that line, not on these estimates.

Sized from `evaluate-isolated` at **25-55 min per cell** on the full data -
session B ran six cells inside its 2.5-5.5 h bracket, and that bracket is the
one number still unmeasured:

| Session | Command | Inputs | Estimate | Produces |
|---|---|---|---:|---|
| **D** | `evaluate-fedavg --targets 1.0,0.75` | dataset + A | 3-6.5 h | 6 of 12 fedavg cells |
| **E** | `evaluate-fedavg` (all; reloads D's six) | dataset + A + D | 3-6.5 h | all 12 fedavg cells |
| **F** | `evaluate-centralized` then `compare-conditions` | dataset + A + B/C + E + Phase 3 models | 1-2 h | **a complete 4.1/4.2/4.4 result** |
| **G** | `train-evidence` (96 augmented boosters) | dataset | **3-4 h** | the 4.3 boosters |
| **I** | `evaluate-evidence --targets 1.0,0.75` | dataset + G | 5-10 h | 6 of 12 cells, both 4.3a and 4.3b |
| **J** | `evaluate-evidence` (all) then `compare-conditions` | dataset + G + I + everything from F | 5-10 h | the full Phase 4 comparison |

**G is a 3-4 h session, not the 7-11 h an earlier version of this plan
claimed.** That estimate came from the inflated 4.2x training ratio; at the
measured 1.46x it is 96 x 1.46 x 83.8s = 3.3 h. Session **H** is therefore
not expected to be needed - G is still resumable per booster, so if its
calibration projects past ~10 h, stop it and run the remainder as H with the
same cell.

**I and J are the tight ones**, because they each run *two* conditions over
six cells: `(0.85 + 1.06) = 1.91x` isolated per cell, so 6 cells is 5-10 h and
12 would not fit. Both are resumable per `(condition, target, seed)`, so an
over-run costs the cell in flight and nothing else.

**4.4 needs the Phase 3 centralized booster.** `evaluate-centralized` loads
`models/xgboost_centralized_L48_s0` and **refuses to train** unless given
`--retrain`, so that a lost model cannot silently swap the booster the report
is about. Attach the Phase 3 training output to session F. If it is genuinely
gone, pass `--retrain` and say so when sending the files back: the report
records it, and the reproduction check against
`docs/reference/evaluate_baseline.csv` is printed either way.

### Session D — stage 4.2 on the first two targets, CPU, commit mode

**Inputs:** the IBM dataset, **session A's output** (the 96 boosters).
**Settings:** Accelerator **None**, Internet **ON**, Save Version → **Save &
Run All (Commit)**.

```python
# D1 - dataset path
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

```python
# D2 - clone and install (~1-2 min). Must report 0d33f89 or later: that
#      commit carries evaluate-fedavg and compare-conditions.
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
!pip install -q -r requirements.txt
!git -C /kaggle/temp/repo rev-parse --short HEAD
```

```python
# D3 - INPUT CHECK, first: stage 4.2 trains nothing, so a missing booster
#      stops the run hours in rather than being quietly recreated. Identical
#      in purpose to B3, and it asserts the same 96.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
print("isolated model trees found:", found)
assert found, "no models/isolated under /kaggle/input - attach session A's output"

dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        target = dest / cell.name
        if not target.exists():
            shutil.copytree(cell, target)

cells = sorted(d for d in dest.iterdir() if d.is_dir())
REQUIRED = ("xgb_model.json", "feature_spec.json", "model_meta.json")
trained, broken = [], []
for cell in cells:
    if (cell / "untrained.json").exists():
        continue
    bad = [n for n in REQUIRED
           if not (cell / n).exists() or (cell / n).stat().st_size == 0]
    (broken if bad else trained).append((cell.name, bad))
for name, bad in broken:
    print("  BROKEN", name, bad)
print(f"{len(cells)} institution-cells, {len(trained)} loadable")
assert len(cells) == 96, f"expected 96, found {len(cells)}"
assert not broken, f"{len(broken)} booster(s) incomplete - re-run session A"

from fraudcamp.models import xgb_baseline
sample = [n for n, _ in trained][:: max(1, len(trained) // 8)]
for name in sample:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print(f"  spot-loaded {len(sample)} boosters, all at L = 48 with 66 features")
```

```python
# D4 - PRE-FLIGHT, second: the frozen scope, the tau grid, and the two things
#      stage 4.2 is defined by - that the operating points are the per_inst_*
#      names (D6, not uncapped_*) and that the primary pair is unchanged.
import sys
sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp import constants, evaluation
from fraudcamp.federation import fedavg, isolated

assert constants.PHASE4_SEEDS == [0, 1, 2], constants.PHASE4_SEEDS
assert constants.PHASE4_LOOKBACK_H == 48, constants.PHASE4_LOOKBACK_H
assert constants.K_INSTITUTIONS == 8
assert constants.VISIBILITY_TARGETS == [1.0, 0.75, 0.5, 0.25]
grid = evaluation.MODEL_TAU_GRID
assert len(grid) == 22 and grid[-1] == 0.9999999 and 1.0 not in grid

points = [p["operating_point"] for p in isolated.operating_points()]
assert points == ["f1", "per_inst_10", "per_inst_50", "per_inst_100",
                  "total_10", "total_50", "total_100"], points
assert isolated.PRIMARY_POINTS == ("f1", "total_10")
assert fedavg.CONDITION == "fedavg"
print("scope, grid and operating points OK:", points)
```

```python
# D5 - evaluate fedavg at targets 1.00 and 0.75 (six cells). 3.5-8 h.
#      Read the CALIBRATION line it prints after the first cell and size the
#      rest of the plan from that, not from the estimate.
#      Exit 3 is NOT a crash: a chosen tau sat at a grid edge. Read it.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-fedavg",
     "--config", "configs/kaggle.yaml", "--targets", "1.0,0.75"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-fedavg exit code:", code)
```

```python
# D6 - what came out
!ls -la /kaggle/working/phase4_fedavg_* /kaggle/working/evaluate_fedavg_cells | head -40
```

### Session E — stage 4.2, the remaining targets, CPU, commit mode

**Inputs:** the IBM dataset, **session A's output**, **session D's output**.
Cells **E1, E2** are D1 and D2 verbatim; **E4** is D4 verbatim.

```python
# E3 - INPUT CHECK: A's boosters (as D3) AND D's finished cells, so this
#      session reloads them in seconds instead of re-evaluating six cells it
#      already has. Missing cells do not crash the run - they make it twice as
#      long and miss the 12 h limit, which is why this asserts rather than
#      prints.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

# --- A: the boosters, exactly as D3 ---
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
assert found, "no models/isolated under /kaggle/input - attach session A's output"
dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        if not (dest / cell.name).exists():
            shutil.copytree(cell, dest / cell.name)
assert len(sorted(d for d in dest.iterdir() if d.is_dir())) == 96

# --- D: the six finished fedavg cells ---
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-name", "evaluate_fedavg_cells"],
    capture_output=True, text=True, check=True).stdout.split())
print("finished-cell directories found:", found)
cells_dir = pathlib.Path("/kaggle/working/evaluate_fedavg_cells")
cells_dir.mkdir(parents=True, exist_ok=True)
for tree in found:
    for item in sorted(pathlib.Path(tree).glob("*.csv")):
        if not (cells_dir / item.name).exists():
            shutil.copy2(item, cells_dir / item.name)

SUFFIXES = ("institutions", "taus", "ceilings", "summary", "pooled",
            "union", "outcomes")
expected = [f"fedavg_t{round(t * 100):03d}_s{s}" for t in (1.0, 0.75)
            for s in (0, 1, 2)]
done = sorted({p.name.split("_summary")[0] for p in cells_dir.glob("*_summary.csv")})
print(f"{len(done)} cells staged: {done}")
missing = [n for n in expected if n not in done]
incomplete = [(n, s) for n in done for s in SUFFIXES
              if not (cells_dir / f"{n}_{s}.csv").exists()]
for n, s in incomplete:
    print(f"  INCOMPLETE {n}: no _{s}.csv")
assert not missing, (
    f"session D's cells {missing} are not staged. Attach session D's output, "
    "or this session re-evaluates them and will not finish inside 12 h."
)
assert not incomplete, f"{len(incomplete)} cell file(s) missing - re-send D"
print("  all six of D's cells complete; E will reload them in seconds")
```

```python
# E5 - all four targets: D's six load in seconds, the six at 0.50 and 0.25
#      are evaluated. 3.5-8 h. This writes the complete fedavg report.
#      Expect exit 3 if any of D's cells had an edge tau: this session
#      reports their exclusions as its own, because they are part of the
#      report it writes.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-fedavg",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-fedavg exit code:", code)
```

```python
# E6 - the headline, read back from the files rather than the log. The mean
#      never prints alone: institution 0 holds the dataset's largest bank by
#      construction, so the median and the range print beside it.
import pandas as pd
pooled = pd.read_csv("/kaggle/working/phase4_fedavg_pooled.csv")
union = pd.read_csv("/kaggle/working/phase4_fedavg_union.csv")
print("cells:", pooled[["visibility_target", "seed"]].drop_duplicates().shape[0], "of 12")
print(pooled[(pooled.eval_split == "test") & (pooled.metric == "campaign_recall")]
      [["visibility_target", "seed", "operating_point",
        "mean_over_all_institutions", "n_institutions_in_mean",
        "median_over_institutions", "min_over_institutions", "min_institution",
        "max_over_institutions", "max_institution",
        "n_binding_budgets", "n_cells_with_a_budget"]].to_string(index=False))
print(union[(union.eval_split == "test") & (union.test_subset == "all")]
      [["visibility_target", "seed", "operating_point", "campaign_recall",
        "oracle_recall", "recall_relative_to_ceiling", "false_alarms_per_day",
        "summed_false_alarms_per_day", "n_institutions_operating"]].to_string(index=False))
```

### Session F — stage 4.4 and the first complete comparison, CPU, commit mode

**Inputs:** the IBM dataset, **session A's output**, **sessions B/C's output**
(the `isolated` reports), **session E's output** (the `fedavg` reports), and
**the Phase 3 model output** holding `models/xgboost_centralized_L48_s0`.
Short session, ~1-2 h.

```python
# F1, F2 - as D1 and D2.
```

```python
# F3 - INPUT CHECK: the Phase 3 centralized booster, and the condition
#      reports F compares. `evaluate-centralized` REFUSES to train without
#      --retrain, so a missing booster must be caught here and not worked
#      around: the comparator has to be the model the Phase 3 record is about.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp.federation import centralized as centralized_mod

policy = centralized_mod.PHASE4_POLICY
seeds = centralized_mod.model_seed_policy(policy)
print(f"model-seed policy {policy!r} -> model seed(s) {list(seeds)}")
assert policy == "matched_seed0", (
    "O1 decided matched_seed0: Phase 4 studies fragmentation, not model seed, "
    "and every federated booster is model seed 0"
)

wanted = [f"xgboost_centralized_L48_s{s}" for s in seeds]
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-name", "xgboost_centralized_L48_s*"],
    capture_output=True, text=True, check=True).stdout.split())
print("centralized booster dirs found:", found)
models = pathlib.Path("/kaggle/working/models")
models.mkdir(parents=True, exist_ok=True)
for path in found:
    name = pathlib.Path(path).name
    if name in wanted and not (models / name).exists():
        shutil.copytree(path, models / name)
missing = [n for n in wanted if not (models / n / "xgb_model.json").exists()]
assert not missing, (
    f"no Phase 3 centralized booster for {missing}. Attach the Phase 3 "
    "training output. Only pass --retrain if it is genuinely gone, and say so "
    "when sending the files back."
)
from fraudcamp.models import xgb_baseline
for name in wanted:
    model = xgb_baseline.TrainedBaseline.load(models / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
    print(f"  {name}: L = {model.lookback_h}, {len(model.feature_names)} features")

# the condition reports to compare
staged = 0
for pattern in ("phase4_isolated_*.csv", "phase4_fedavg_*.csv"):
    for path in subprocess.run(
        ["find", "/kaggle/input", "-maxdepth", "6", "-name", pattern],
        capture_output=True, text=True, check=True).stdout.split():
        target = pathlib.Path("/kaggle/working") / pathlib.Path(path).name
        if not target.exists():
            shutil.copy2(path, target)
            staged += 1
print(f"  staged {staged} condition report file(s)")
for required in ("phase4_isolated_summary.csv", "phase4_fedavg_summary.csv"):
    assert (pathlib.Path("/kaggle/working") / required).exists(), (
        f"{required} is not staged: attach the session that produced it"
    )
# D6: sessions that ran before the rename wrote uncapped_*; one report must
# never carry two names for one operating point.
renamed = 0
for path in sorted(pathlib.Path("/kaggle/working").glob("phase4_*.csv")):
    text = path.read_text(encoding="utf-8")
    if "uncapped_" in text:
        path.write_text(text.replace("uncapped_", "per_inst_"), encoding="utf-8")
        renamed += 1
print(f"  D6: rewrote uncapped_* -> per_inst_* in {renamed} file(s)")
```

```python
# F4 - as D4 verbatim (scope, grid, operating points).
```

```python
# F5 - stage 4.4. ~1 h. One detection run at model seed 0, replicated across
#      the twelve cells with each cell's seed as the bootstrap seed, because
#      a centralized model cannot depend on a visibility assignment.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-centralized",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-centralized exit code:", code)
```

```python
# F6 - the comparison. Minutes. Exit 4 means a PAIRING RULE was violated,
#      which is a stop condition and not a bad number: send the message back.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "compare-conditions",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("compare-conditions exit code:", code)
```

```python
# F7 - the two checks that matter more than any difference, read from the
#      files. If either fails, nothing else in the report is usable.
import json, pandas as pd
checks = json.load(open("/kaggle/working/phase4_compare.json"))["checks"]
print(json.dumps(checks, indent=2))
rep = json.load(open("/kaggle/working/phase4_centralized.json"))["reproduction"]
print("\nreproduction against Phase 3 (per model seed):")
for row in rep["per_seed"]:
    print(" ", row)
print("\nPhase 3 headline (five-seed mean):", rep["phase3_headline"])
head = pd.read_csv("/kaggle/working/phase4_compare_headline.csv")
print("\nprimary points, test, subset all, relative to own ceiling:")
print(head[["contrast", "unit", "visibility_target", "seed", "operating_point",
            "n_pairs", "mean_difference", "median_difference",
            "min_difference", "max_difference", "n_positive", "n_negative",
            "below_seed_noise_floor"]].to_string(index=False))
```

### Session G — stage 4.3, the 96 augmented boosters, CPU, commit mode

**Inputs:** the IBM dataset only. **7-11 h.** Resumable per
`(target, seed, institution)`, so if G's calibration projects past ~10 h,
stop it and run the remainder as **session H** with the same cells — the
finished boosters are skipped.

```python
# G1, G2 - as D1 and D2.
```

```python
# G3 - INPUT CHECK: the dataset, and that the evidence definition is the one
#      the report will claim. Nothing else is needed - G trains from raw.
import sys
sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp import constants, features
from fraudcamp.federation import evidence as evidence_mod

assert evidence_mod.CONDITION_EVIDENCE == "evidence"
assert evidence_mod.CONDITION_FEDAVG_EVIDENCE == "fedavg_evidence"
# the six node features are what D3 shares, and no more
assert list(features.CONTINUOUS_NODE_FEATURES) == [
    "in_degree", "out_degree", "log_in_amount", "log_out_amount",
    "n_counterparties", "n_currencies",
], features.CONTINUOUS_NODE_FEATURES
print("evidence conditions and the shared node features OK:",
      list(features.CONTINUOUS_NODE_FEATURES))
```

```python
# G4 - as D4 verbatim (scope, grid, operating points).
```

```python
# G5 - train the 96 augmented boosters. 7-11 h. Resumable per booster.
#      --max-cached-windows caps the shared-evidence memory: 0 is unbounded
#      and fastest, and the full data's windows are large, so set a cap if
#      the session is killed for memory. Each booster is written as it
#      finishes, so stopping costs the one in flight.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "train-evidence",
     "--config", "configs/kaggle.yaml", "--max-cached-windows", "12"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("train-evidence exit code:", code)
```

```python
# G6 - what came out, and whether any institution starved on augmented views
import pandas as pd
frame = pd.read_csv("/kaggle/working/train_evidence.csv")
print(len(frame), "rows;", int((~frame.get("skipped", False).astype(bool)).sum()), "trained here")
if "tier" in frame:
    print(frame.tier.value_counts().to_string())
    print("min positives:", int(frame.n_positive.min()),
          " mean rows:", int(frame.n_rows.mean()))
!ls -la /kaggle/working/models/evidence | head -5
!du -sh /kaggle/working/models/evidence
```

### Session I — stage 4.3a and 4.3b on the first two targets, CPU, commit mode

**Inputs:** the IBM dataset, **session G's output** (the augmented boosters).
**4-9 h.** Both conditions are evaluated per cell from one set of boosters and
one shared-evidence object, so 4.3b − 4.3a differs in the scoring rule alone.

```python
# I1, I2 - as D1 and D2. I4 - as D4 verbatim.
```

```python
# I3 - INPUT CHECK: the 96 AUGMENTED boosters. These are not session A's -
#      a cell evaluated with the wrong boosters would produce a number that
#      looks fine and means nothing, so the count and the shape are asserted.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/evidence"],
    capture_output=True, text=True, check=True).stdout.split())
print("evidence model trees found:", found)
assert found, "no models/evidence under /kaggle/input - attach session G's output"

dest = pathlib.Path("/kaggle/working/models/evidence")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        if not (dest / cell.name).exists():
            shutil.copytree(cell, dest / cell.name)

cells = sorted(d for d in dest.iterdir() if d.is_dir())
REQUIRED = ("xgb_model.json", "feature_spec.json", "model_meta.json")
trained, broken = [], []
for cell in cells:
    if (cell / "untrained.json").exists():
        continue
    bad = [n for n in REQUIRED
           if not (cell / n).exists() or (cell / n).stat().st_size == 0]
    (broken if bad else trained).append(cell.name)
print(f"{len(cells)} augmented institution-cells, {len(trained)} loadable")
for name in broken:
    print("  BROKEN", name)
assert len(cells) == 96, f"expected 96, found {len(cells)}"
assert not broken, f"{len(broken)} booster(s) incomplete - resume session G"

from fraudcamp.models import xgb_baseline
sample = trained[:: max(1, len(trained) // 8)]
for name in sample:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print(f"  spot-loaded {len(sample)} augmented boosters at L = 48, 66 features")
```

```python
# I5 - evaluate 4.3a AND 4.3b at targets 1.00 and 0.75 (twelve cell-runs).
#      4-9 h. Both conditions share the boosters and the shared evidence, so
#      4.3b - 4.3a is the scoring rule and nothing else.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-evidence",
     "--config", "configs/kaggle.yaml", "--targets", "1.0,0.75",
     "--max-cached-windows", "12"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-evidence exit code:", code)
```

```python
# I6 - what came out
!ls -la /kaggle/working/phase4_evidence_* /kaggle/working/phase4_fedavg_evidence_* /kaggle/working/evaluate_evidence_cells | head -40
```

### Session J — stage 4.3, the remaining targets, and the full comparison

**Inputs:** the IBM dataset, **session G's output**, **session I's output**,
and **everything from session F** (the `isolated`, `fedavg` and `centralized`
reports). **4-9 h.** Cells **J1, J2** are D1 and D2; **J4** is D4.

```python
# J3 - INPUT CHECK: G's augmented boosters (as I3), I's finished cells, and
#      F's condition reports - because J re-runs compare-conditions over all
#      five arms and every one of them has to be present.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

# --- G: the augmented boosters, as I3 ---
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/evidence"],
    capture_output=True, text=True, check=True).stdout.split())
assert found, "no models/evidence under /kaggle/input - attach session G"
dest = pathlib.Path("/kaggle/working/models/evidence")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        if not (dest / cell.name).exists():
            shutil.copytree(cell, dest / cell.name)
assert len(sorted(d for d in dest.iterdir() if d.is_dir())) == 96

# --- I: the twelve finished cell-runs ---
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-name", "evaluate_evidence_cells"],
    capture_output=True, text=True, check=True).stdout.split())
cells_dir = pathlib.Path("/kaggle/working/evaluate_evidence_cells")
cells_dir.mkdir(parents=True, exist_ok=True)
for tree in found:
    for item in sorted(pathlib.Path(tree).glob("*.csv")):
        if not (cells_dir / item.name).exists():
            shutil.copy2(item, cells_dir / item.name)
expected = [f"{c}_t{round(t * 100):03d}_s{s}"
            for c in ("evidence", "fedavg_evidence")
            for t in (1.0, 0.75) for s in (0, 1, 2)]
done = sorted({p.name.split("_summary")[0] for p in cells_dir.glob("*_summary.csv")})
print(f"{len(done)} cell-runs staged")
missing = [n for n in expected if n not in done]
assert not missing, (
    f"session I's cell-runs {missing} are not staged. Attach session I's "
    "output, or J re-evaluates them and will not finish inside 12 h."
)

# --- F: the other three arms' reports ---
staged = 0
for pattern in ("phase4_isolated_*.csv", "phase4_fedavg_*.csv",
                "phase4_centralized_*.csv"):
    for path in subprocess.run(
        ["find", "/kaggle/input", "-maxdepth", "6", "-name", pattern],
        capture_output=True, text=True, check=True).stdout.split():
        target = pathlib.Path("/kaggle/working") / pathlib.Path(path).name
        if not target.exists():
            shutil.copy2(path, target)
            staged += 1
print(f"  staged {staged} report file(s) from F")
for required in ("phase4_isolated_summary.csv", "phase4_fedavg_summary.csv",
                 "phase4_centralized_summary.csv"):
    assert (pathlib.Path("/kaggle/working") / required).exists(), (
        f"{required} is not staged: attach session F's output"
    )
renamed = 0
for path in sorted(pathlib.Path("/kaggle/working").glob("phase4_*.csv")):
    text = path.read_text(encoding="utf-8")
    if "uncapped_" in text:
        path.write_text(text.replace("uncapped_", "per_inst_"), encoding="utf-8")
        renamed += 1
print(f"  D6: rewrote uncapped_* -> per_inst_* in {renamed} file(s)")
```

```python
# J5 - all four targets for both evidence conditions: I's twelve cell-runs
#      load in seconds, the twelve at 0.50 and 0.25 are evaluated. 4-9 h.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-evidence",
     "--config", "configs/kaggle.yaml", "--max-cached-windows", "12"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-evidence exit code:", code)
```

```python
# J6 - the full Phase 4 comparison, all five arms. Minutes.
#      Exit 4 means a pairing rule was violated: a stop condition.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "compare-conditions",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("compare-conditions exit code:", code)
```

```python
# J7 - as F7, now over five arms. The checks first: if the ceilings disagree
#      across conditions, a condition moved visibility and no difference in
#      the report means anything.
import json, pandas as pd
payload = json.load(open("/kaggle/working/phase4_compare.json"))
print(json.dumps(payload["checks"], indent=2))
print("\nconditions present:", payload["conditions_present"])
print("skipped contrasts:", [s.get("name") for s in payload["skipped"]])
head = pd.read_csv("/kaggle/working/phase4_compare_headline.csv")
print("\nprimary points, test, subset all, relative to own ceiling:")
print(head[["contrast", "unit", "visibility_target", "seed", "operating_point",
            "n_pairs", "mean_difference", "median_difference",
            "min_difference", "max_difference", "n_positive", "n_negative",
            "below_seed_noise_floor"]].to_string(index=False))
```

### Dropping 4.3

Verified rather than asserted: with the `evidence` and `fedavg_evidence`
reports absent, `compare-conditions` produces
`centralized - isolated`, `centralized - fedavg` and `fedavg - isolated`, and
**names** the four contrasts it could not form with the reason "one arm not
present". So stopping after session F leaves a complete, self-describing
4.1/4.2/4.4 result. Nothing has to be re-run to get there.

### What to send back from sessions D to J

From each session: every `phase4_*.csv`, the matching `.md` and `.json`, and
**the `*_cells/` directory**. The cells directory is what makes the next
session cheap, and it is the only output that survives a session that stopped
before writing a combined report - a 12 h kill, or `--stop-on-low-edge-tau`
firing. If the log says `STOPPING, and NOT recording`, or simply ends
mid-cell, there are no `phase4_*` files to send and the cells are the whole
output. A low-edge tau on its own no longer stops anything (policy changed
2026-11-16).

### What to expect, and what would be surprising

* **The ceilings must be identical across `isolated`, `fedavg`, `evidence`
  and `fedavg_evidence`**, to the last digit. 4.2 and 4.3 change what a model
  knows, never which rows it has. `compare-conditions` raises if they ever
  disagree, and that failure matters more than any difference in the report.
  Confirmed on the sample at max disagreement **0.0**.
* **The union ceiling must equal the centralized ceiling**, as session B
  measured and `docs/reference/evaluate_baseline.csv` records: 0.8928571, 50
  of 56. Also confirmed on the sample at difference **0.0**.
* **A difference smaller than ~0.04 relative recall is not an effect.**
  Phase 4's seeds vary visibility only; all boosters are model seed 0. Phase 3
  measures model-seed noise at centralized as 0.661-0.696 recall and 0.74-0.78
  relative — about 2 campaigns of 56, a seventh to a fifth of the
  union → centralized gap. Rows flagged `below_seed_noise_floor` must not be
  written up as results.
* **`fedavg` should cost about 1.4x `isolated`, not 8x.** If a cell takes
  eight times as long, the shared featurisation has fallen back to per-booster
  featurisation, which happens when the specs' one-hot vocabularies disagree.
  The answer is still correct — the fallback is exact — but the session will
  not finish, so send the calibration line back.
* **An institution untrained on augmented views would be new.** Session A
  trained 96 of 96 on plain views with 1,003 positives at the thinnest, and
  augmentation changes features, not rows, so the positive counts should be
  identical to session A's. If they are not, the view changed and something is
  wrong.
* **A low-edge F1 tau is no longer a stop condition.** It fired on session
  B's `t100_s2`, the mechanism turned out not to be a bug, and since
  2026-11-16 such a cell is recorded at the floor tau and flagged
  ([5n](#5n-stage-41-the-10-remaining-cells--sessions-c1-to-c3-supersedes-5ls-session-c)). What to watch instead is
  `n_f1_degenerate` on the pooled rows: a cell whose validation F1 curve was
  flat at ~0 has an arbitrary argmax, and the count says how many of a mean's
  cells were in that regime.

## 5n. Stage 4.1, the 10 remaining cells — sessions C1 to C3 (supersedes §5l's Session C)

**§5l's Session C is wrong and must not be run.** Its C3 asserts six of
session B's cells. B produced **two** — `t100_s0` and `t100_s1` — so that
assertion fires in 44 seconds, which is the cell doing its job. The plan below
replaces it.

### What B produced, and why it stopped — settled

| | |
|---|---|
| cells written | `t100_s0`, `t100_s1` — 2 of the 6 requested |
| cells missing | `t100_s2`, `t075_s0`, `t075_s1`, `t075_s2` |
| boosters | all 96, verified loadable at L = 48 with 66 features |
| **why it stopped** | **the low-edge F1 tau rule, on `t100_s2`** |
| **per-cell wall time** | `t100_s0` **2,568.5s** (46.2 min), `t100_s1` **2,465.0s** (44.4 min) |

The cause was settled from B's own output, without re-measuring anything:
summing `eval_seconds` over the eight institutions of each finished cell and
scaling by 1.08 — the cell-wall-to-eval-sum ratio measured across all four
cells of the sample dry run, where it ranged 1.05-1.11 — gives 46.2 and 44.4
minutes. Two cells plus one in flight is **~2.3 h**, so the 12 h limit is
excluded, and no other cause on the list fits a run that left two complete
cells and no combined report.

Two consequences:

* **the 25-55 min per-cell estimate was right.** Two independent measurements,
  44.4 and 46.2 min, agree to within 4%. An earlier reading of this evidence
  suggested the estimate might be 5-10× low; it was not;
* **the earlier three-cell-per-session cap is lifted.** At 46 min a cell, 10 h
  of headroom is **13 cells**, not 3.

A diagnostic session was written to tell a low-edge stop from a 12 h kill —
the two are indistinguishable on disk, because `return 3` preceded both the
cell write and the combined report. It is no longer needed and has been
removed from this plan.

### The policy in force — read this before running anything

A low-edge F1 tau is now **recorded at the floor tau, flagged, and kept in the
pooled means** (changed 2026-11-16 with explicit approval; see the README
section *DECIDED: a low-edge F1 tau is recorded and flagged*). So:

* **no flag is needed.** `--allow-low-edge-tau` is accepted and ignored,
  because what it asked for is the default. The earlier `LOW_EDGE` switch in
  this runbook is gone;
* `--stop-on-low-edge-tau` restores the pre-registered stop, for reproducing
  the old behaviour only;
* a cell that hits the floor prints `LOW-EDGE F1 TAU in <cell>. POLICY
  'record_and_flag', approved 2026-11-16`, lists each institution with its tau
  and its best validation campaign F1, and **carries on**. That line is in the
  log so the numbers can never be read without knowing which rule produced
  them.

Recording a floor tau is only meaningful beside the value at it, so every cell
now also records **`f1_best_value`** and **`f1_degenerate`** (true when the
best validation campaign F1 is `<= 0.01`, an epsilon fixed 2026-11-16 before
any cell ran under this policy). Every pooled row, union row and comparison
row carries `n_f1_degenerate` the way it already carries the binding-budget
count. The flag informs interpretation only: **`f1` and `total_10` stay
primary**, as designated in stage 4.1.

### The three sessions

Sized from the measured **46 min per cell**, with 10 h of usable headroom
against the 12 h limit:

| Session | Command | Inputs | Cells | Estimate | Produces |
|---|---|---|---:|---:|---|
| **C1** | `evaluate-isolated --targets 1.0 --seeds 2` | dataset + **A** | 1 | **~0.8 h** | `t100_s2` — the cell B stopped on — **DONE 2026-11-18: outcome (a)** |
| **C2** | `evaluate-isolated --targets 0.75,0.5` | dataset + **A** | 6 | **~4.6 h** | all of 0.75 and 0.50 — **DONE 2026-11-18: 6 of 6, 4.4 h, exit 0, no degeneracy and no edge taus anywhere** |
| **C3** | `evaluate-isolated` (plain) | dataset + **A** + **B** + **C1** + **C2** | 3 + assembly | **~2.5 h** | the three 0.25 cells **and the complete report over all twelve** |

**Why C1 is alone.** It is the cell the pre-registered rule stopped, and the
first cell to run under the changed policy. Keeping it in its own ~50-minute
session buys a checkpoint on a policy that has never executed on real data;
losing that checkpoint to save 45 minutes would be a bad trade. Its three
possible outcomes are **pre-registered below**, and two of the three say
**stop**.

**Why the sessions are decoupled.** C1 and C2 take only session A's boosters —
neither needs the other's cells, and neither needs B's. Each writes its own
per-cell CSVs as it goes, so a session that over-runs costs the cell in flight
and nothing else, and a session that fails costs nothing another session
needed. Only C3 depends on the rest, because only C3 assembles.

Each session's input check asserts **exactly what that session needs** and
nothing more. C1 and C2 assert their own boosters; only C3 asserts a complete
set of twelve cells.

### C3 merges evaluation and assembly — and that is safe, with one condition

Running the three 0.25 cells and the twelve-cell report in one session is
sound, because `evaluate-isolated` with no `--targets` already does both: it
walks all four targets in `constants.VISIBILITY_TARGETS` order
(1.0, 0.75, 0.5, 0.25), **reloads** every cell whose `_summary.csv` is already
present, evaluates the ones that are not, and writes the report over all of
them. The nine staged cells reload in seconds; 0.25 is last in the order, so
the three fresh cells are evaluated after the reloads, not interleaved.

Nothing about the merge breaks an assumption in the assembly path. What it
does is make the assembly depend on the three fresh cells finishing, so two
things must be true, and the input check below enforces both:

1. **the nine staged cells must be renamed before the run.** B ran before the
   `uncapped_*` → `per_inst_*` rename. If the staged cells still say
   `uncapped_*` while the fresh cells say `per_inst_*`, the concat produces
   **both** column sets with NaNs in each, and a pooled row would be computed
   over whichever name it found. This is the one genuine break, and it is a
   silent one;
2. **the nine staged cells must be backfilled before the run**, so every row
   carries `f1_best_value`. The assembly refuses to write a report without it
   rather than counting a missing value as not-degenerate.

The residual coupling, stated plainly: **if one of C3's three cells fails, the
assembly does not happen.** No cell is lost — each is written as it finishes —
so the fix is to re-run C3 with the finished cells staged, which then reloads
them and assembles. If that is unacceptable, split C3 into `--targets 0.25`
and a separate assembly session; the assembly needs no boosters and takes
minutes.

### C1 — `t100_s2` alone. CPU, commit mode. ~0.8 h.

**Inputs:** the IBM dataset, **session A's output**. Not B's.

```python
# C1a - dataset path
!ls /kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/ | head
```

```python
# C1b - clone and install. Must report the commit that carries the
#       2026-11-16 low-edge policy or later.
!rm -rf /kaggle/temp/repo
!git clone --depth 1 https://github.com/jerinsaji299-coder/Fraud-campaign-detection.git /kaggle/temp/repo
%cd /kaggle/temp/repo/backend
!pip install -q -r requirements.txt
!git -C /kaggle/temp/repo rev-parse --short HEAD
```

```python
# C1c - INPUT CHECK, first. Asserts EXACTLY what this session needs: A's 96
#       boosters, and nothing about any other session's cells. The old C3
#       asserted six of B's cells, which is what failed in 44s.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")

found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
print("isolated model trees found:", found)
assert found, "no models/isolated under /kaggle/input - attach session A's output"

dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        if not (dest / cell.name).exists():
            shutil.copytree(cell, dest / cell.name)

cells = sorted(d for d in dest.iterdir() if d.is_dir())
REQUIRED = ("xgb_model.json", "feature_spec.json", "model_meta.json")
broken = [
    c.name for c in cells
    if not (c / "untrained.json").exists()
    and any(not (c / n).exists() or (c / n).stat().st_size == 0 for n in REQUIRED)
]
print(f"{len(cells)} institution-cells, {len(broken)} broken")
assert len(cells) == 96, f"expected 96, found {len(cells)}"
assert not broken, f"{len(broken)} booster(s) incomplete - re-run session A"

# the 8 this session actually scores
from fraudcamp.models import xgb_baseline
from fraudcamp.federation import isolated
mine = [isolated.cell_name(1.0, 2, i) for i in range(8)]
for name in mine:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print("loaded all 8 boosters for t100_s2:", mine)

stale = sorted(pathlib.Path("/kaggle/working").glob("isolated_cells/*_summary.csv"))
assert not stale, f"unexpected cached cells in /kaggle/working: {stale}"
print("no cached cells present: this session will evaluate t100_s2 for real")
```

```python
# C1d - PRE-FLIGHT, second. The grid FLOOR is asserted, because the floor is
#       what the low-edge rule fires on: if it has moved, this session is not
#       comparable to t100_s0 and t100_s1, which were selected on a 22-point
#       grid. The degeneracy epsilon is asserted for the same reason.
import sys
sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp import constants, evaluation
from fraudcamp.federation import isolated

assert constants.PHASE4_SEEDS == [0, 1, 2]
assert constants.PHASE4_LOOKBACK_H == 48
assert constants.K_INSTITUTIONS == 8
assert constants.VISIBILITY_TARGETS == [1.0, 0.75, 0.5, 0.25]
grid = evaluation.MODEL_TAU_GRID
assert len(grid) == 22, len(grid)
assert grid[0] == 0.05, ("the grid FLOOR has moved; t100_s0/s1 were selected "
                         "on a 22-point grid and this would not be comparable",
                         grid[0])
assert grid[1] == 0.1, "the second point moved: B's near-floor taus were 0.1"
assert grid[-1] == 0.9999999 and 1.0 not in grid
assert isolated.F1_DEGENERATE_EPSILON == 0.01, (
    "the degeneracy epsilon was fixed at 0.01 on 2026-11-16, before any cell "
    "ran under the recording policy", isolated.F1_DEGENERATE_EPSILON,
)
points = [p["operating_point"] for p in isolated.operating_points()]
assert points == ["f1", "per_inst_10", "per_inst_50", "per_inst_100",
                  "total_10", "total_50", "total_100"], points
print(f"scope OK. grid floor {grid[0]}, second {grid[1]}, top {grid[-1]}, "
      f"{len(grid)} points")
print("f1_degenerate epsilon:", isolated.F1_DEGENERATE_EPSILON)
print("operating points:", points)
```

```python
# C1e - evaluate t100_s2 ONLY. Under the 2026-11-16 policy a low-edge F1 tau
#       is RECORDED and flagged, so this should exit 0 and write the cell. A
#       `LOW-EDGE F1 TAU` line in the log is the expected outcome, not a
#       failure - it is the event that stopped session B.
import subprocess
cmd = ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-isolated",
       "--config", "configs/kaggle.yaml",
       "--targets", "1.0", "--seeds", "2"]
print("running:", " ".join(cmd))
code = subprocess.run(cmd, cwd="/kaggle/temp/repo/backend").returncode
print("evaluate-isolated exit code:", code)
print("policy in force: record_and_flag (no flag passed, none needed)")
```

```python
# C1f - what came out: the taus, the value behind each one, and the flag.
#       The one-cell phase4_isolated_*.csv this writes is NOT the report -
#       C3 writes the report over all twelve. Ignore it.
import pathlib, pandas as pd
cells = pathlib.Path("/kaggle/working/isolated_cells")
!ls -la /kaggle/working/isolated_cells 2>/dev/null | head
t = pathlib.Path(cells / "t100_s2_taus.csv")
if t.exists():
    taus = pd.read_csv(t)
    f1 = taus[taus.operating_point == "f1"]
    print(f1[["institution", "tau", "tau_edge", "f1_best_value",
              "f1_degenerate", "rejected_tau", "undefined_reason"]]
          .to_string(index=False))
    print("\nat the floor (0.05):", int((f1.tau == 0.05).sum()), "of", len(f1))
    print("one step above (0.1):", int((f1.tau == 0.1).sum()))
    print("f1_degenerate:", sorted(f1.loc[f1.f1_degenerate == True, "institution"]))
    print("\nA degenerate cell is recorded and kept in the means. Its tau was "
          "settled by the tie-break, not the data, which is what the flag "
          "says - and why f1 and total_10 stay primary regardless.")
else:
    print("t100_s2 was NOT recorded - read the exit message above.")
```

### C1's three outcomes — PRE-REGISTERED 2026-11-16, before the run

C1 is the first cell ever evaluated under the recording policy, and that
policy rests on a diagnosis — a flat ~0 campaign-F1 objective whose argmax is
settled by the tie-break — which has never been checked against full data. So
the three things the run can say are written down **before** it says one of
them, and `C1g` classifies the result mechanically rather than by reading the
log afterwards.

| | What the run shows | What it means | What happens next |
|---|---|---|---|
| **(a)** | `LOW-EDGE F1 TAU` appears **and** `f1_degenerate` is **true** for those institutions | the mechanism is as diagnosed: a flat ~0 objective, an arbitrary argmax, the floor picked by the tie-break. The policy is doing what it was designed for | **proceed to C2** |
| **(b)** | `LOW-EDGE F1 TAU` appears but `f1_degenerate` is **false** | the floor is a **genuine optimum**, not a flat-curve artefact. The event is real; the explanation written into the README is **wrong** | **STOP.** Correct the write-up before nine more cells are evaluated under an explanation that does not hold. `C1g` prints `f1_best_value` and the full validation curve for those institutions |
| **(c)** | no `LOW-EDGE` line at all | session B stopped for a reason that has not been found | **STOP and investigate.** C2 does not run on the assumption that the cause is understood |

**(b) is the one this is written down for.** A low-edge tau with a real
maximum behind it would be a finding about the detector rather than about a
flat objective, and the recording policy would still be right — but for a
different reason, and the README would be asserting the wrong one. An
explanation adopted after seeing which way it fell is not worth having.

A **mixed** result — some low-edge institutions degenerate, some not — counts
as **(b)**. The diagnosis is either the explanation for the event or it is
not; a partial one has to be narrowed before anything is built on it.

`C1g` prints the validation campaign-F1 curve for every low-edge institution
in all three branches, so the classification carries its own evidence rather
than resting on a boolean. The curve comes from
`/kaggle/working/phase4_isolated.json`, which C1 writes: its per-cell
`tau_curves` hold all 22 points per institution. (That JSON is worth keeping
even though the one-cell `phase4_isolated_*.csv` beside it is **not** the
report — C3 writes the report over all twelve.)

The classifier was exercised against fabricated cells covering all four cases
before this runbook shipped — a flat curve at the floor, a real maximum at the
floor, a **mixed** cell with one of each, and no low-edge tau at all — and
reached (a), (b), (b) and (c) respectively, plus the not-recorded case. It is
71 lines and decides nothing a reader cannot check by reading it.

```python
# C1g - classify the run against the three outcomes pre-registered BEFORE it.
#       Mechanical on purpose: the branch is decided by the cell's own rows,
#       not by reading the log, and the curve is printed either way so the
#       verdict carries its evidence.
import json, pathlib
import pandas as pd

working = pathlib.Path("/kaggle/working")
taus_path = working / "isolated_cells" / "t100_s2_taus.csv"


def curves_for(institutions):
    """The validation campaign-F1 curve per institution, from C1's JSON."""
    report = working / "phase4_isolated.json"
    if not report.exists():
        print("  (no phase4_isolated.json beside the cell - cannot show the "
              "curve; send the cell directory back)")
        return
    payload = json.loads(report.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        for key, curve in (cell.get("tau_curves") or {}).items():
            if int(key) not in institutions or not curve:
                continue
            values = [(row["tau"], row["campaign_f1"]) for row in curve]
            best = max(v for _, v in values)
            spread = best - min(v for _, v in values)
            print(f"\n  institution {key}: best {best:.6f}, "
                  f"range over the 22 points {spread:.6f}")
            print("    " + "  ".join(f"{t:g}:{v:.4f}" for t, v in values))


if not taus_path.exists():
    print("OUTCOME: the cell was not recorded at all.")
    print("Neither (a), (b) nor (c) - read C1e's exit message. STOP.")
else:
    taus = pd.read_csv(taus_path)
    f1 = taus[taus.operating_point == "f1"].copy()
    f1["is_low"] = f1["tau_edge"].astype(str).str.lower().eq("low")
    f1["is_degenerate"] = f1["f1_degenerate"].astype(str).str.lower().eq("true")
    low = f1[f1.is_low]

    print(f"{len(low)} of {len(f1)} institution(s) chose an F1 tau at the LOW "
          f"edge: {sorted(low.institution)}")
    if len(low):
        print(low[["institution", "tau", "f1_best_value",
                   "f1_degenerate"]].to_string(index=False))

    if not len(low):
        print("\nOUTCOME (c): no low-edge F1 tau in t100_s2.")
        print("Session B stopped for a reason that has not been found.")
        print("STOP and investigate. Do NOT run C2.")
        print("\nFor contrast, the curves of the three lowest taus chosen:")
        curves_for(set(f1.nsmallest(3, "tau").institution))
    elif low.is_degenerate.all():
        epsilon = f1.f1_degenerate_epsilon.iloc[0]
        print(f"\nOUTCOME (a): every low-edge institution is f1_degenerate "
              f"(best validation campaign F1 <= {epsilon}).")
        print("The mechanism is as diagnosed: a flat ~0 objective, an "
              "arbitrary argmax, the floor selected by the tie-break.")
        print("The policy is doing what it was designed for. Proceed to C2.")
        print("\nThe curves behind that verdict:")
        curves_for(set(low.institution))
    else:
        real = low[~low.is_degenerate]
        print("\nOUTCOME (b): a low-edge F1 tau with a REAL maximum behind it.")
        print(real[["institution", "tau", "f1_best_value"]].to_string(index=False))
        print("\nThe floor is a genuine optimum for these cells, not a "
              "flat-curve artefact, so the explanation in the README does not "
              "hold for them.")
        print("STOP. Do NOT run C2. Send these curves back:")
        curves_for(set(real.institution))
```

### C2 — targets 0.75 and 0.50, six cells. CPU, commit mode. ~4.6 h.

**Inputs:** the IBM dataset, **session A's output**. Not B's, not C1's.

Cells **a**, **b**, **d** are C1a, C1b, C1d verbatim. Two changes:

```python
# C2c - C1c verbatim, with these lines in place of the t100_s2-specific ones
TARGETS = [0.75, 0.5]
SEEDS = [0, 1, 2]
mine = [isolated.cell_name(t, s, i)
        for t in TARGETS for s in SEEDS for i in range(8)]
for name in mine:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48, (name, model.lookback_h)
    assert len(model.feature_names) == 66, (name, len(model.feature_names))
print(f"loaded all {len(mine)} boosters for {TARGETS}")
```

```python
# C2e - six cells, ~46 min each. The CALIBRATION line prints after the first
#       one: size the rest from it, not from the estimate above. Every cell is
#       written as it finishes, so stopping early costs the cell in flight.
import subprocess
cmd = ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-isolated",
       "--config", "configs/kaggle.yaml", "--targets", "0.75,0.5"]
print("running:", " ".join(cmd))
code = subprocess.run(cmd, cwd="/kaggle/temp/repo/backend").returncode
print("evaluate-isolated exit code:", code)
```

> **Observed 2026-11-18: exit 0**, 6 of 6 cells in 4.4 h (2,604-2,755s each).
> The artifacts say so independently of the printed code —
> `phase4_isolated_c2.json` has an empty `tau_at_grid_edge` and no tau row in
> any of the six cells carries a `tau_edge`, so there was no edge for the
> session to exit 3 about.

```python
# C2f - the six cells, their F1 taus, and every degenerate one
import pathlib, pandas as pd
cells = pathlib.Path("/kaggle/working/isolated_cells")
print("cells written:",
      sorted(p.name.split("_summary")[0] for p in cells.glob("*_summary.csv")))
taus = pd.concat([pd.read_csv(p) for p in sorted(cells.glob("*_taus.csv"))],
                 ignore_index=True)
f1 = taus[taus.operating_point == "f1"]
print(f1[["visibility_target", "seed", "institution", "tau", "tau_edge",
          "f1_best_value", "f1_degenerate"]].to_string(index=False))
print("\nat the floor:", int((f1.tau == 0.05).sum()), "of", len(f1))
print("degenerate:", int((f1.f1_degenerate == True).sum()), "of", len(f1))
```

> **Observed 2026-11-18: 0 at the floor of 48, 0 degenerate of 48.** Every F1
> tau landed between 0.9 and 0.995 and `f1_best_value` ran 0.0317-0.3226. The
> degenerate regime is target 1.0 only, which is the measured form of the
> 2026-11-16 risk-direction correction, and the (A)+(D) policy therefore binds
> on 2 of the 12 cells — pending C3's three, which are predicted clean by the
> same mechanism and not yet observed.

### C3 — target 0.25, then the complete report. CPU, commit mode. ~2.5 h.

**Inputs:** the IBM dataset, **A**, **B**, **C1**, **C2**. All five.

Cells **a**, **b**, **d** are C1a, C1b, C1d verbatim.

```python
# C3c - INPUT CHECK. This is the session that writes the authoritative report,
#       so it is the only one that asserts a COMPLETE set of twelve cells -
#       and it does the two things the merge requires BEFORE the run: the D6
#       rename, and the f1_best_value backfill. Both silent if skipped.
import pathlib, shutil, subprocess, sys

sys.path.insert(0, "/kaggle/temp/repo/backend/src")
from fraudcamp import constants
from fraudcamp.federation import isolated
from fraudcamp.models import xgb_baseline

# --- the boosters this session evaluates: the 0.25 cells only -------
found = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-path", "*/models/isolated"],
    capture_output=True, text=True, check=True).stdout.split())
assert found, "no models/isolated under /kaggle/input - attach session A"
dest = pathlib.Path("/kaggle/working/models/isolated")
dest.mkdir(parents=True, exist_ok=True)
for tree in found:
    for cell in sorted(pathlib.Path(tree).iterdir()):
        if not (dest / cell.name).exists():
            shutil.copytree(cell, dest / cell.name)
mine = [isolated.cell_name(0.25, s, i) for s in [0, 1, 2] for i in range(8)]
for name in mine:
    model = xgb_baseline.TrainedBaseline.load(dest / name)
    assert model.lookback_h == 48 and len(model.feature_names) == 66, name
print(f"loaded all {len(mine)} boosters for target 0.25")

# --- the nine finished cells, from B, C1 and C2 ---------------------
trees = sorted(subprocess.run(
    ["find", "/kaggle/input", "-maxdepth", "6", "-type", "d",
     "-name", "isolated_cells"],
    capture_output=True, text=True, check=True).stdout.split())
print("\nfinished-cell directories found:")
for d in trees:
    print("  ", d)
cells_dir = pathlib.Path("/kaggle/working/isolated_cells")
cells_dir.mkdir(parents=True, exist_ok=True)
for tree in trees:
    for item in sorted(pathlib.Path(tree).glob("*.csv")):
        if not (cells_dir / item.name).exists():
            shutil.copy2(item, cells_dir / item.name)

SUFFIXES = ("institutions", "taus", "ceilings", "summary", "pooled",
            "union", "outcomes")
staged = sorted({p.name.split("_summary")[0]
                 for p in cells_dir.glob("*_summary.csv")})
expected_nine = [isolated.cell_name(t, s)
                 for t in [1.0, 0.75, 0.5] for s in constants.PHASE4_SEEDS]
print(f"\n{len(staged)} cells staged: {staged}")
incomplete = [(n, s) for n in staged for s in SUFFIXES
              if not (cells_dir / f"{n}_{s}.csv").exists()]
for n, s in incomplete:
    print(f"  INCOMPLETE {n}: no _{s}.csv")
assert not incomplete, f"{len(incomplete)} cell file(s) missing"
missing_nine = [n for n in expected_nine if n not in staged]
assert not missing_nine, (
    f"cells {missing_nine} are not staged. This session writes the report "
    "over all twelve, so every cell from B, C1 and C2 must be attached."
)
assert not [n for n in staged if n.startswith("t025")], (
    "a 0.25 cell is already staged; this session is supposed to evaluate "
    "those three. Detach it or pass --force."
)

# --- (1) the D6 rename, BEFORE the run -----------------------------
#     B ran before uncapped_* -> per_inst_*. Mixing the two names in one
#     concat produces BOTH column sets with NaNs in each, and a pooled row
#     computed over whichever name it found. Silent if skipped.
renamed = 0
for path in sorted(cells_dir.glob("*.csv")):
    text = path.read_text(encoding="utf-8")
    if "uncapped_" in text:
        path.write_text(text.replace("uncapped_", "per_inst_"), encoding="utf-8")
        renamed += 1
print(f"\nD6: rewrote uncapped_* -> per_inst_* in {renamed} staged file(s)")
leftover = [p.name for p in cells_dir.glob("*.csv")
            if "uncapped_" in p.read_text(encoding="utf-8")]
assert not leftover, leftover
```

```python
# C3c2 - (2) the f1_best_value backfill, BEFORE the run. Arithmetic on rows
#        B's cells already wrote: the validation row at the f1 point was
#        evaluated at the chosen tau, so campaign F1 there is 2PR/(P+R) of its
#        precision and recall. Re-evaluates NOTHING. Verified against the
#        measured curves on the 1% sample to within one ULP (2.22e-16).
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "backfill-f1-value"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("backfill-f1-value exit code:", code)
assert code == 0, "backfill failed - do not assemble without it"

import pathlib, pandas as pd
cells_dir = pathlib.Path("/kaggle/working/isolated_cells")
inst = pd.concat([pd.read_csv(p) for p in sorted(cells_dir.glob("*_institutions.csv"))],
                 ignore_index=True)
trained = inst[inst.trained.astype(str).str.lower() == "true"]
gaps = trained[trained.f1_best_value.isna()]
print(f"\n{len(trained)} trained institution-cells, "
      f"{len(gaps)} without f1_best_value")
print(gaps[["visibility_target", "seed", "institution",
            "f1_best_value_source"]].to_string(index=False) if len(gaps)
      else "  (none)")
print("\nEXPECT NO GAPS. t100_s0 institution 0's F1 point was rejected at "
      "the top of the grid, so no validation run exists at that tau - but "
      "its validation oracle reaches NO campaign (n_oracle_reachable 0), "
      "which forces campaign F1 to 0 at every tau for any scorer. Since "
      "2026-11-18 the backfill derives it as exactly 0.0, recorded as "
      "f1_best_value_source = derived_from_zero_val_ceiling. A gap here "
      "now means the backfill was not re-run after that change: re-run it, "
      "it re-evaluates nothing.")
```

```python
# C3e - the three 0.25 cells, then the report over all twelve. A plain
#       evaluate-isolated walks all four targets, RELOADS the nine staged
#       cells in seconds, evaluates the three that are missing, and writes
#       phase4_isolated_*.csv over all of them.
#
#       EXPECT EXIT 3. t100_s0 carries a high-edge exclusion (institution 0,
#       F1 point, rejected at 0.9999999), and exit 3 is this report
#       describing itself - every file is written first. A low-edge tau does
#       NOT cause it any more; that is recorded and flagged.
#
#       EACH RELOADED CELL PRINTS WHAT MOVED, in two lines of its own. Expect
#       "reason re-derived at assembly" for t100_s0 institution 0 and t100_s2
#       institution 2 - both from TAU_BEYOND_GRID to the zero-validation-
#       ceiling reason - and "degeneracy count re-taken over the panel" on
#       t100_s2's 12 pooled and 12 union rows, 1 of 7 -> 2 of 8 (panel 8).
#       C2's six cells should print neither: they have no exclusion and no
#       degenerate curve. A cell that prints nothing was already right.
import subprocess
code = subprocess.run(
    ["python", "scripts/kaggle/kaggle_runner.py", "evaluate-isolated",
     "--config", "configs/kaggle.yaml"],
    cwd="/kaggle/temp/repo/backend",
).returncode
print("evaluate-isolated exit code:", code)
```

```python
# C3f - the headline. The mean never prints alone: institution 0 holds the
#       dataset's largest bank by construction. Every pooled row carries the
#       spread, the binding-budget count AND the degenerate-F1 count.
import pandas as pd
pooled = pd.read_csv("/kaggle/working/phase4_isolated_pooled.csv")
union = pd.read_csv("/kaggle/working/phase4_isolated_union.csv")
print("cells:", pooled[["visibility_target", "seed"]].drop_duplicates().shape[0], "of 12")
print(pooled[(pooled.eval_split == "test") & (pooled.metric == "campaign_recall")]
      [["visibility_target", "seed", "operating_point",
        "mean_over_all_institutions", "n_institutions_in_mean",
        "median_over_institutions", "min_over_institutions", "min_institution",
        "max_over_institutions", "max_institution",
        "n_binding_budgets", "n_cells_with_a_budget",
        "n_f1_degenerate", "n_cells_with_an_f1_curve",
        "n_institutions"]].to_string(index=False))
print(union[(union.eval_split == "test") & (union.test_subset == "all")]
      [["visibility_target", "seed", "operating_point", "campaign_recall",
        "oracle_recall", "recall_relative_to_ceiling", "false_alarms_per_day",
        "summed_false_alarms_per_day", "n_institutions_operating",
        "n_f1_degenerate", "n_institutions"]].to_string(index=False))
```

> **Read the union's `f1` rows with `false_alarms_per_day` beside them, always.**
> Measured 2026-11-18: the union's F1 point costs 1,321-2,119 false alarms a
> day at target 1.0 against 39-87 at 0.75 and 0.50, so its recall is NOT
> comparable across targets. `total_10` is volume-matched by construction and
> is the point that carries a cross-target reading. The primary points do not
> change on the strength of that - they were fixed in advance.

```python
# C3g - archive. Commit mode keeps /kaggle/working, so this is the handoff.
!cd /kaggle/working && tar -czf phase4_isolated_complete.tar.gz \
    phase4_isolated_*.csv phase4_isolated*.json isolated_cells
!ls -la /kaggle/working/phase4_isolated_complete.tar.gz
```

### Exit codes for C1, C2 and C3

| Code | Meaning | What to do |
|---|---|---|
| **0** | every requested cell evaluated, no tau at a grid edge | nothing |
| **3** | the report is complete and **written**, and some chosen tau sat at a grid edge. For C3 this is **expected**: `t100_s0` institution 0 and `t100_s2` institution 2 both had their F1 point rejected at the grid top | read which cells, **and the reason printed beside each one** - since 2026-11-18 a high edge gives one of three, and only `TAU_BEYOND_GRID` means the grid is too short. Both observed so far are `VAL_UNREACHABLE`. A **high** edge is excluded and named, a **low** edge is recorded and flagged |
| **3** with `STOPPING, and NOT recording` | only if `--stop-on-low-edge-tau` was passed | drop the flag — recording is the policy since 2026-11-16 |
| non-zero, no report | an assertion in the input check or pre-flight, or the session hit 12 h | send the log; finished cells keep their files, so a re-run resumes |

### Does anything already run need redoing?

**No.** `t100_s0`, `t100_s1`, session A, C1 and C2 stand as they are.

* a cell's CSVs are written only **after** it completes, so a written cell is
  complete — the two that exist are whole;
* the boosters are untouched and were re-verified loadable at L = 48 with 66
  features;
* `t100_s0`'s high-edge exclusion travels in its own rows and is re-derived at
  assembly, so the complete report names it - and since 2026-11-18 the
  *reason* is re-derived there too, from the cell's own `*_ceilings.csv` and
  `*_taus.csv`, so B's cells are reclassified without being re-run. C3 prints
  every reason it moves;
* the D6 relabelling changes a column name, not a number;
* C1's and C2's cells carry a degeneracy count taken over the denominator in
  force when they ran. It is **re-taken at assembly** from each cell's own
  `*_taus.csv`, by `degenerate_summary` itself, so the report over all twelve
  states one denominator and prints every row it moves. Nothing is re-run;
* `f1_best_value` is **backfilled**, not re-measured — see C3c2. Re-running
  the two cells would cost ~1.5 h and produce the same numbers, differing only
  in the 16th significant digit.

**The one scenario that would have forced a re-run was rejected:** extending
`MODEL_TAU_GRID` downward. Those two cells were selected on the 22-point grid,
so a longer grid would make them incomparable with everything after them and
all twelve would have to run again. C1d asserts `grid[0] == 0.05` and
`grid[1] == 0.1` so a mixed-grid report cannot assemble silently.
