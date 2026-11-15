"""Does `backfill-f1-value` reproduce the F1 value a run actually measured?

This is the evidence behind one decision: **session B's `t100_s0` and
`t100_s1` do not need re-running** to gain the `f1_best_value` column added on
2026-11-16. Re-running them would cost ~1.5 h of Kaggle time, so the claim
that the value is recoverable from what they already wrote had better be
checked rather than argued.

The argument is that the value is already in every finished cell's
`_summary.csv`, implicitly: the validation row at the `f1` operating point was
evaluated **at the chosen tau**, so its `campaign_precision` and
`campaign_recall` are the precision and recall at the argmax, and campaign F1
is `2PR / (P + R)` of them - which is exactly the quantity
`evaluation.select_tau` maximises.

Checking it needs a run where **both** the tau curve and the summary row are
on disk. A local isolated run is the only such place: its JSON report carries
`tau_curves` per cell, which the Kaggle per-cell CSVs do not. So this script
splits a local run's combined CSVs back into per-cell files, **drops** the new
columns so the fixture looks exactly like a cell written before they existed,
runs the real `backfill-f1-value` command against it, and compares every
reconstructed value with the measured curve maximum.

Usage, from `backend/`:

    python scripts/run_isolated.py --config configs/local.yaml   # if needed
    python scripts/verify_f1_backfill.py

Exit 0 if every value agrees within `--tolerance`, 1 otherwise.

**Expect agreement to one ULP, not bit-exactly.** The per-cell CSV stores
precision and recall at 16 significant digits, so `1/26` comes back as
`0.0384615384615384` rather than `0.038461538461538464` and the harmonic mean
of the pair differs from the measured one in the last place. The measured
bound on the 1% sample is **2.22e-16** over 32 (cell, institution) pairs.
Against an epsilon of 0.01 and figures reported to three decimals that is
immaterial - but "exact" would be the wrong word, so the default tolerance is
1e-12 and the worst observed difference is printed every time.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND / "src"))

from fraudcamp.federation import isolated  # noqa: E402

SUFFIXES = ("institutions", "taus", "summary")
NEW_COLUMNS = (
    "f1_best_value", "f1_degenerate", "f1_degenerate_epsilon",
    "f1_best_value_source",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", default="artifacts/isolated_dev",
        help=(
            "a directory holding phase4_isolated.json and the combined "
            "phase4_isolated_*.csv of a local isolated run "
            "(default: artifacts/isolated_dev)"
        ),
    )
    ap.add_argument("--tolerance", type=float, default=1e-12)
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = BACKEND / run
    report_path = run / "phase4_isolated.json"
    if not report_path.exists():
        print(
            f"no {report_path}. Run `python scripts/run_isolated.py --config "
            "configs/local.yaml` first - this check needs a run whose JSON "
            "carries the tau curves, which the per-cell CSVs do not."
        )
        return 1

    report = json.loads(io.open(report_path, encoding="utf-8").read())
    frames = {}
    for suffix in SUFFIXES:
        path = run / f"phase4_isolated_{suffix}.csv"
        if not path.exists():
            print(f"no {path}")
            return 1
        frames[suffix] = pd.read_csv(path)

    staging = Path(tempfile.mkdtemp(prefix="f1_backfill_"))
    try:
        expected: dict[str, dict[int, float | None]] = {}
        for cell in report["cells"]:
            name = isolated.cell_name(cell["target"], cell["seed"])
            for suffix, frame in frames.items():
                part = frame[
                    (frame["visibility_target"] - cell["target"]).abs().lt(1e-9)
                    & frame["seed"].eq(cell["seed"])
                ].drop(columns=list(NEW_COLUMNS), errors="ignore")
                if not len(part):
                    print(f"{name}: no {suffix} rows in the combined CSV")
                    return 1
                part.to_csv(staging / f"{name}_{suffix}.csv", index=False)
            expected[name] = {
                int(institution): (
                    max(row["campaign_f1"] for row in curve) if curve else None
                )
                for institution, curve in cell["tau_curves"].items()
            }

        print(f"staged {len(expected)} cell(s) from {run}")
        print(f"epsilon in force: {isolated.F1_DEGENERATE_EPSILON}\n")

        proc = subprocess.run(
            [sys.executable, "scripts/kaggle/kaggle_runner.py",
             "backfill-f1-value", "--cells-dir", str(staging)],
            cwd=BACKEND, capture_output=True, text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            print(f"backfill-f1-value exited {proc.returncode}")
            return 1

        problems = 0
        checked = 0
        worst = 0.0
        for name, by_institution in sorted(expected.items()):
            got = pd.read_csv(staging / f"{name}_institutions.csv")
            for institution, measured in sorted(by_institution.items()):
                rows = got[got["institution"].eq(institution)]
                if len(rows) != 1:
                    problems += 1
                    print(f"  {name} i{institution}: {len(rows)} rows, want 1")
                    continue
                value = rows["f1_best_value"].iloc[0]
                if measured is None:
                    # no curve at all: must come back empty, never 0
                    if pd.notna(value):
                        problems += 1
                        print(
                            f"  {name} i{institution}: no curve was measured, "
                            f"but the backfill produced {value}"
                        )
                    continue
                checked += 1
                difference = abs(float(value) - measured)
                worst = max(worst, difference)
                if difference > args.tolerance:
                    problems += 1
                    print(
                        f"  MISMATCH {name} i{institution}: backfilled "
                        f"{value!r} against measured {measured!r}"
                    )
                flag = bool(rows["f1_degenerate"].iloc[0])
                if flag is not bool(
                    measured <= isolated.F1_DEGENERATE_EPSILON
                ):
                    problems += 1
                    print(
                        f"  FLAG WRONG {name} i{institution}: "
                        f"f1_degenerate={flag} for a best value of {measured}"
                    )

        # every file the comparison reads must carry the columns, not just the
        # one that holds tau_f1: `compare.py` takes `f1_degenerate` from the
        # summary frame
        for name in sorted(expected):
            for suffix in SUFFIXES:
                columns = pd.read_csv(staging / f"{name}_{suffix}.csv").columns
                for column in NEW_COLUMNS:
                    if column not in columns:
                        problems += 1
                        print(f"  MISSING {column} in {name}_{suffix}.csv")

        print(
            f"{checked} (cell, institution) value(s) checked against the "
            f"measured curves, {problems} problem(s)."
        )
        print(
            f"worst |backfilled - measured| = {worst:.3e} "
            f"(tolerance {args.tolerance:.0e})"
        )
        print(
            "One ULP is expected: the per-cell CSV stores precision and "
            "recall at 16 significant digits, so the reconstruction is exact "
            "arithmetic on slightly rounded inputs."
        )
        return 1 if problems else 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
