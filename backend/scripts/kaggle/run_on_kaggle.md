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
| 5 | `make-sample` | build the 1% sample for local development | §5e |

Steps 1–2 are done (2026-10-08/09). Step 3 has been run once and is being
re-run after a correctness fix. Steps 4–5 are new.

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

## 6. Later stages

`kaggle_runner.py` also accepts `train` and `evaluate`. Both currently exit
with a message saying they are not implemented: they arrive with stage 3.3
(XGBoost baseline) and stage 3.4 (GNN), together with the GPU settings and
expected runtimes they need. Nothing about them is runnable yet. Those are
the first commands that will need **Accelerator: GPU**; everything above
runs on CPU.
