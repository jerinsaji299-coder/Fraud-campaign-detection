# Running this repository on Kaggle

The full dataset is too large for a laptop, so the heavy steps run in a
Kaggle notebook. This document is the exact sequence of cells to paste.

Everything the runner writes goes to `/kaggle/working/`, which is the only
directory Kaggle keeps after the session ends.

---

## 0. One-time: put the code on GitHub

**The project is not a git repository yet** (checked 2026-10-07), so there is
nothing to clone. From the project root:

```bash
git init
git add .
git status                       # review what is staged before committing
git commit -m "Phase 2 complete: pipeline, artifacts, API, frontend"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
git push -u origin main
```

Before pushing to a **public** repo, check `git status` output: `.gitignore`
already excludes `backend/data/`, the contents of `backend/artifacts/`,
`node_modules/`, `.venv/` and `.env`, so no dataset copies and no secrets
should be included. The repo contains no credentials of any kind.

Replace `<YOUR-USERNAME>/<YOUR-REPO>` below with your actual repo.

---

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
!git clone --depth 1 https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git /kaggle/temp/repo
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

## 6. Later stages

`kaggle_runner.py` also accepts `train` and `evaluate`. Both currently exit
with a message saying they are not implemented: they arrive with stage 3.3
(XGBoost baseline) and stage 3.4 (GNN), together with the GPU settings and
expected runtimes they need. Nothing about them is runnable yet.
