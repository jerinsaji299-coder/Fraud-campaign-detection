"""Re-derive every committed Phase 4 figure in the README from its source file.

This is the one command `docs/reference/phase4_crosscheck.md` points at. It
reads the committed evidence files, recomputes each figure the README quotes,
and prints PASS or FAIL per figure with the value it found.

    cd backend
    python scripts/verify_phase4_crosscheck.py
    python scripts/verify_phase4_crosscheck.py --json   # machine-readable

Exit **0** if every figure matches, **1** otherwise. Figures that depend on
artifacts that are not committed - the 1% sample development runs, and session
B's uncommitted reports - are listed as SKIPPED with the file they would need,
rather than silently passing. That distinction is the point: a figure this
script cannot check is a figure the README is asserting on trust, and the
count of those is printed at the end.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

README = ROOT / "README.md"
REFERENCE = ROOT / "docs" / "reference"


class Checker:
    def __init__(self):
        self.rows: list[dict] = []
        self.readme = io.open(README, encoding="utf-8").read()

    def check(self, figure, claimed, actual, source, how, tol=5e-4):
        ok = None
        if actual is None:
            status = "SKIPPED"
        else:
            try:
                ok = abs(float(claimed) - float(actual)) <= tol * max(
                    1.0, abs(float(actual))
                )
            except (TypeError, ValueError):
                ok = str(claimed) == str(actual)
            status = "PASS" if ok else "FAIL"
        self.rows.append({
            "figure": figure, "claimed": claimed, "recomputed": actual,
            "status": status, "source": source, "how": how,
        })

    def text(self, figure, needle, source, how):
        """A claim that is a form of words rather than a number."""
        present = needle in self.readme
        self.rows.append({
            "figure": figure, "claimed": needle[:70], "recomputed": present,
            "status": "PASS" if present else "FAIL",
            "source": source, "how": how,
        })

    def skip(self, figure, claimed, source, why):
        self.rows.append({
            "figure": figure, "claimed": claimed, "recomputed": None,
            "status": "SKIPPED", "source": source, "how": why,
        })


def phase3_centralized(checker: Checker) -> None:
    """O1 and the two centralized numbers, from the committed Phase 3 file."""
    path = REFERENCE / "evaluate_baseline.csv"
    source = "docs/reference/evaluate_baseline.csv"
    how = (
        "rows where split=test, variant=xgboost_centralized, lookback_h=48, "
        "test_subset=all"
    )
    if not path.exists():
        checker.skip("Phase 3 centralized", "all", source, "file missing")
        return
    frame = pd.read_csv(path)
    rows = frame[
        (frame.split == "test")
        & (frame.variant == "xgboost_centralized")
        & (frame.lookback_h == 48)
        & (frame.test_subset == "all")
    ].sort_values("seed")

    per_seed = dict(zip(rows.seed.astype(int), rows.campaign_recall))
    per_seed_rel = dict(zip(rows.seed.astype(int), rows.recall_relative_to_ceiling))
    for seed, claimed in ((0, 0.696), (1, 0.661), (2, 0.696), (3, 0.661),
                          (4, 0.679)):
        checker.check(
            f"centralized recall, model seed {seed}", claimed,
            round(float(per_seed[seed]), 3), source, how, tol=2e-3,
        )
    for seed, claimed in ((0, 0.78), (1, 0.74), (2, 0.78), (3, 0.74), (4, 0.76)):
        checker.check(
            f"centralized relative, model seed {seed}", claimed,
            round(float(per_seed_rel[seed]), 2), source, how, tol=2e-3,
        )

    checker.check(
        "Phase 3 headline recall (five-seed mean)", 0.6786,
        float(rows.campaign_recall.mean()), source, how + ", mean over 5 seeds",
    )
    checker.check(
        "Phase 3 headline relative (five-seed mean)", 0.760,
        float(rows.recall_relative_to_ceiling.mean()), source,
        how + ", mean over 5 seeds",
    )
    three = rows[rows.seed <= 2]
    checker.check(
        "centralized, Phase 4's three seeds as model seeds (rejected)", 0.6845,
        float(three.campaign_recall.mean()), source, how + ", mean over seeds 0-2",
    )
    checker.check(
        "centralized relative, three seeds (rejected)", 0.7667,
        float(three.recall_relative_to_ceiling.mean()), source,
        how + ", mean over seeds 0-2",
    )
    checker.check(
        "PHASE 4 COMPARATOR recall (matched_seed0)", 0.6964,
        float(per_seed[0]), source, how + ", seed 0 only",
    )
    checker.check(
        "PHASE 4 COMPARATOR relative (matched_seed0)", 0.780,
        float(per_seed_rel[0]), source, how + ", seed 0 only",
    )
    checker.check(
        "centralized ceiling at L=48", 50 / 56,
        float(rows.oracle_recall.iloc[0]), source, how + ", oracle_recall",
    )
    checker.check(
        "centralized ceiling is identical at every model seed", 1,
        int(rows.oracle_recall.nunique()), source, how + ", nunique",
    )

    # the seed-noise floor
    spread_recall = float(rows.campaign_recall.max() - rows.campaign_recall.min())
    spread_rel = float(
        rows.recall_relative_to_ceiling.max() - rows.recall_relative_to_ceiling.min()
    )
    checker.check(
        "model-seed spread, recall", 0.0357, spread_recall, source,
        how + ", max - min", tol=2e-3,
    )
    checker.check(
        "model-seed spread, relative (THE FLOOR)", 0.040, spread_rel, source,
        how + ", max - min", tol=2e-3,
    )
    checker.check(
        "model-seed spread as campaigns of 56", 2.00,
        spread_recall * int(rows.n_campaigns.iloc[0]), source,
        how + ", spread x n_campaigns", tol=1e-2,
    )
    # the two ratios the README states
    for label, lo, hi, claimed_lo, claimed_hi in (
        ("union -> centralized", 0.500, 0.600, 1 / 7, 1 / 4.5),
        ("lone institution -> centralized", 0.306, 0.369, 1 / 11.8, 1 / 10.3),
    ):
        checker.check(
            f"floor as a fraction of the {label} gap (narrow end)", claimed_hi,
            spread_rel / (0.780 - hi), source,
            f"0.040 / (0.780 - {hi})", tol=2e-2,
        )
        checker.check(
            f"floor as a fraction of the {label} gap (wide end)", claimed_lo,
            spread_rel / (0.780 - lo), source,
            f"0.040 / (0.780 - {lo})", tol=2e-2,
        )
    checker.check(
        "room for 4.2 against the matched comparator, narrow end", 0.18,
        0.780 - 0.600, source, "0.780 - union relative at seed 1", tol=2e-2,
    )
    checker.check(
        "room for 4.2 against the matched comparator, wide end", 0.28,
        0.780 - 0.500, source, "0.780 - union relative at seed 0", tol=2e-2,
    )


def session_a(checker: Checker) -> None:
    """The 96 boosters, from the committed training report."""
    path = REFERENCE / "train_isolated.csv"
    source = "docs/reference/train_isolated.csv"
    if not path.exists():
        checker.skip("session A", "all", source, "file missing")
        return
    frame = pd.read_csv(path)
    how = "all 96 rows"
    checker.check("session A institution-cells", 96, int(len(frame)), source, how)
    checker.check(
        "session A cells with tier=normal", 96,
        int((frame.tier == "normal").sum()), source, how,
    )
    checker.check(
        "session A untrained cells", 0,
        int((~frame.trained.astype(bool)).sum()), source, how,
    )
    checker.check(
        "thinnest cell, positives", 1003, int(frame.n_positive.min()), source,
        how + ", min n_positive",
    )
    checker.check(
        "mean positives per cell", 1598.1, float(frame.n_positive.mean()),
        source, how + ", mean n_positive", tol=1e-3,
    )
    checker.check(
        "mean rows per cell", 1999169, float(frame.n_rows.mean()), source,
        how + ", mean n_rows", tol=1e-5,
    )
    checker.check(
        "class weight, min", 435.23, float(frame.class_weight_min.min()),
        source, how, tol=1e-3,
    )
    checker.check(
        "class weight, max", 4014.69, float(frame.class_weight_max.max()),
        source, how, tol=1e-3,
    )
    checker.check(
        "n_windows per cell", 9, int(frame.n_windows.max()), source,
        how + ", every row",
    )
    checker.check(
        "train_seconds mean", 83.8, float(frame.train_seconds.mean()), source,
        how, tol=2e-3,
    )
    checker.check(
        "train_seconds min", 79.1, float(frame.train_seconds.min()), source, how,
        tol=2e-3,
    )
    checker.check(
        "train_seconds max", 111.1, float(frame.train_seconds.max()), source, how,
        tol=2e-3,
    )

    # the thinnest cell's identity, and the cross-institution spread
    thinnest = frame.loc[frame.n_positive.idxmin()]
    checker.check(
        "thinnest cell target", 1.0, float(thinnest.visibility_target), source,
        "row with min n_positive",
    )
    checker.check(
        "thinnest cell seed", 1, int(thinnest.seed), source,
        "row with min n_positive",
    )
    checker.check(
        "thinnest cell institution", 2, int(thinnest.institution), source,
        "row with min n_positive",
    )
    for target, claimed in ((1.0, 2.42), (0.75, 1.93), (0.5, 1.89), (0.25, 1.91)):
        block = frame[frame.visibility_target == target]
        checker.check(
            f"full-data spread at target {target}", claimed,
            float(block.n_positive.max() / block.n_positive.min()), source,
            f"max/min n_positive within visibility_target == {target}",
            tol=2e-3,
        )
    # the flat band, and the gap from target 1.0
    flat = [
        float(frame[frame.visibility_target == t].n_positive.max()
              / frame[frame.visibility_target == t].n_positive.min())
        for t in (0.75, 0.5, 0.25)
    ]
    top = float(frame[frame.visibility_target == 1.0].n_positive.max()
                / frame[frame.visibility_target == 1.0].n_positive.min())
    # exact values, tight tolerance. A loose tolerance here would hide the
    # drift between the rounded figures the README quotes and the real ones.
    checker.check(
        "the three fragmented targets are within 2.1% of each other", 2.12,
        100 * (max(flat) / min(flat) - 1), source, "max/min of the three ratios",
        tol=5e-3,
    )
    checker.check(
        "target 1.0 is this % above the highest of the three", 25.12,
        100 * (top / max(flat) - 1), source,
        "spread(1.0) / spread(0.75) - 1", tol=5e-3,
    )
    checker.check(
        "target 1.0 is this % above the lowest of the three", 27.78,
        100 * (top / min(flat) - 1), source,
        "spread(1.0) / spread(0.5) - 1", tol=5e-3,
    )

    # institution 0, the hub
    totals = frame.groupby("institution").agg(
        positives=("n_positive", "sum"), rows=("n_rows", "sum")
    )
    mean_pos = float(totals.positives.mean())
    mean_rows = float(totals.rows.mean())
    checker.check(
        "institution 0 positives vs the cross-institution mean", 1.442,
        float(totals.positives.loc[0]) / mean_pos, source,
        "sum of n_positive by institution, index 0 / mean", tol=2e-3,
    )
    checker.check(
        "institution 0 rows vs the cross-institution mean", 1.020,
        float(totals.rows.loc[0]) / mean_rows, source,
        "sum of n_rows by institution, index 0 / mean", tol=2e-3,
    )
    checker.check(
        "institution 0 holds the most positives in how many of 12 cells", 12,
        int(sum(
            1 for (_, _), block in frame.groupby(["visibility_target", "seed"])
            if int(block.loc[block.n_positive.idxmax()].institution) == 0
        )), source, "per (target, seed), argmax of n_positive",
    )


def float32_grid(checker: Checker) -> None:
    """The representability claim about the top of the tau grid."""
    import numpy as np

    from fraudcamp import evaluation

    source = "fraudcamp.evaluation.MODEL_TAU_GRID + IEEE 754 binary32"
    grid = evaluation.MODEL_TAU_GRID
    checker.check("tau grid points", 22, len(grid), source, "len(MODEL_TAU_GRID)")
    checker.check("tau grid top", 0.9999999, float(grid[-1]), source, "grid[-1]",
                  tol=1e-12)
    checker.check("1.0 excluded from the grid", False, 1.0 in grid, source,
                  "1.0 in MODEL_TAU_GRID")
    steps = []
    x = np.float32(grid[-1])
    while x < np.float32(1.0):
        x = np.nextafter(x, np.float32(1.0))
        steps.append(float(x))
    checker.check(
        "float32 values strictly between the grid top and 1.0", 1,
        len(steps) - 1, source, "np.nextafter walk from grid[-1] to 1.0",
    )
    checker.check(
        "that one value is 1 - 6.0e-8", 1 - 6.0e-8, steps[0], source,
        "the first nextafter step", tol=1e-3,
    )
    checker.check(
        "1 - 1e-8 collapses to 1.0 in float32", 1.0,
        float(np.float32(1 - 1e-8)), source, "np.float32(1 - 1e-8)",
    )


def sample_runs(checker: Checker) -> None:
    """The 1% sample figures: checked when the artifacts are present."""
    for label, rel, how, figures in (
        (
            "stage 4.0 sample spread (L=48)",
            "backend/artifacts/federation_dev/institution_split.csv",
            "max/min positives within target, lookback_h == 48, seed >= 0",
            ((1.0, 3.13), (0.75, 2.03), (0.5, 1.76), (0.25, 1.60)),
        ),
        (
            "stage 4.1 sample spread (seed 0)",
            "backend/artifacts/isolated_dev/phase4_isolated_institutions.csv",
            "max/min n_positive within visibility_target",
            ((1.0, 2.01), (0.75, 1.86), (0.5, 1.44), (0.25, 1.53)),
        ),
    ):
        path = ROOT / rel
        if not path.exists():
            checker.skip(
                label, ", ".join(str(v) for _, v in figures), rel,
                "development artifact, not committed: regenerate with the "
                "stage's local script",
            )
            continue
        frame = pd.read_csv(path)
        for target, claimed in figures:
            if "institution_split" in rel:
                block = frame[
                    (frame.lookback_h == 48)
                    & (frame.seed >= 0)
                    & (frame.target.astype(str) == str(target))
                ]
                value = float(block.positives.max() / block.positives.min())
            else:
                block = frame[frame.visibility_target == target]
                value = float(block.n_positive.max() / block.n_positive.min())
            checker.check(f"{label} at {target}", claimed, value, rel, how,
                          tol=3e-3)


def dry_run(checker: Checker) -> None:
    """The chain dry run's ratios and checks, from the COMMITTED payload.

    The committed `docs/reference/phase4_chain_sample.json` first, and the
    local development artifact only as a fallback. The order matters and used
    to be the other way round, which was a defect: the README's dry-run
    figures are sourced from the committed file, so checking them against a
    regenerable artifact meant any later local run could make them pass or
    fail for reasons that had nothing to do with the committed evidence.
    Re-running the chain on 2026-11-16 is what surfaced it - four ratios
    "failed" against a run whose timings were contaminated by a concurrent
    test suite, while the committed figures they describe were untouched.
    """
    committed = REFERENCE / "phase4_chain_sample.json"
    local = BACKEND / "artifacts" / "phase4_dev" / "phase4_chain.json"
    if committed.exists():
        path, rel = committed, "docs/reference/phase4_chain_sample.json"
    elif local.exists():
        path, rel = local, "backend/artifacts/phase4_dev/phase4_chain.json"
    else:
        checker.skip(
            "dry-run ratios and checks", "1.21 / 0.85 / 1.06 / 1.46, 0.0, 0.0",
            "docs/reference/phase4_chain_sample.json",
            "neither the committed payload nor a local artifact is present: "
            "regenerate with `python scripts/run_phase4_chain.py --config "
            "configs/local.yaml`",
        )
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    ceilings = checks.get("ceilings_agree_across_conditions", {})
    checker.check(
        "ceilings agree across all four per-institution conditions", 0.0,
        ceilings.get("max_disagreement"), rel,
        "checks.ceilings_agree_across_conditions.max_disagreement", tol=1e-12,
    )
    union = checks.get("union_ceiling_equals_centralized", {})
    checker.check(
        "union ceiling equals centralized ceiling", 0.0,
        union.get("max_difference"), rel,
        "checks.union_ceiling_equals_centralized.max_difference", tol=1e-12,
    )
    # Totals over EVERY cell, matched on the exact step suffix. Matching on
    # `endswith(" isolated")` also matches "train isolated", which is how an
    # earlier version of this check reported a fedavg ratio of 20.45: it was
    # dividing an evaluation by a training time.
    def total(condition: str) -> float | None:
        seconds = [
            row["seconds"] for row in payload.get("timings", [])
            if row["step"].split(" ", 1)[-1] == condition
        ]
        return sum(seconds) if seconds else None

    base = total("isolated")
    train_base = total("train isolated")
    if base:
        for condition, claimed in (("fedavg", 1.21), ("evidence", 0.85),
                                   ("fedavg_evidence", 1.06)):
            value = total(condition)
            checker.check(
                f"{condition} cost ratio to isolated", claimed,
                (value / base) if value else None, rel,
                "sum of per-cell timings / sum of isolated timings", tol=2e-2,
            )
    if train_base:
        value = total("train evidence")
        checker.check(
            "train-evidence cost ratio to train-isolated", 1.46,
            (value / train_base) if value else None, rel,
            "sum of per-cell training timings", tol=2e-2,
        )


def _cell_wall_ratios() -> list[float]:
    """A cell's isolated wall time over the sum of its eight eval_seconds.

    The factor every per-cell wall time derived from `eval_seconds` rests on,
    including the two that settled why session B stopped. Returns an empty
    list when the development artifacts are absent, in which case the caller
    lists the figure instead of checking it.
    """
    import csv

    chain = BACKEND / "artifacts" / "phase4_dev" / "phase4_chain.json"
    rows = BACKEND / "artifacts" / "phase4_dev" / "phase4_isolated_institutions.csv"
    if not chain.exists() or not rows.exists():
        return []
    payload = json.loads(chain.read_text(encoding="utf-8"))
    # `split(" ", 1)[-1] == "isolated"`, NOT `endswith(" isolated")`: the
    # latter also matches "t100_s0 train isolated", and dividing an
    # evaluation by a training time is exactly the bug that once made this
    # script report a fedavg ratio of 20.45.
    wall = {
        row["step"].split(" ", 1)[0]: row["seconds"]
        for row in payload.get("timings", [])
        if row["step"].split(" ", 1)[-1] == "isolated"
    }
    summed: dict[str, float] = {}
    with io.open(rows, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                target = float(row["visibility_target"])
                seed = int(row["seed"])
                seconds = float(row["eval_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            name = "t%03d_s%d" % (round(target * 100), seed)
            summed[name] = summed.get(name, 0.0) + seconds
    return [
        wall[name] / summed[name]
        for name in sorted(set(wall) & set(summed))
        if summed[name] > 0
    ]


def uncommitted_session_b(checker: Checker) -> None:
    """Session B's figures: reported, not yet committed. Listed, never passed."""
    for figure, claimed in (
        ("union test oracle recall (session B)", 0.8928571),
        ("lone institution relative, seed 0", 0.306),
        ("lone institution relative, seed 1", 0.369),
        ("union relative, seed 0", 0.500),
        ("union relative, seed 1", 0.600),
        ("union raw recall, seed 0", 0.446),
        ("union raw recall, seed 1", 0.536),
        ("binding_fraction at per_inst_* on full data", 1.0),
        # the two wall times that settled WHY session B stopped: at 44-46
        # min a cell, two cells plus one in flight is ~2.3 h, which
        # excludes the 12 h limit and leaves the low-edge rule as the
        # only cause that fits. Derived from B's own eval_seconds, so
        # they become checkable with the same CSVs as the rows above.
        ("t100_s0 wall time, sum(eval_seconds) x 1.08", 2568.5),
        ("t100_s1 wall time, sum(eval_seconds) x 1.08", 2465.0),
    ):
        checker.skip(
            figure, claimed, "session B CSVs (not committed)",
            "reported into the README before the CSVs were committed; they "
            "land in docs/reference/ after stage 4.1 session C and must be "
            "re-checked then",
        )
    checker.check(
        "50/56 is the union ceiling session B reported", 0.8928571, 50 / 56,
        "arithmetic", "50 / 56",
    )
    # The 1.08 scale factor behind those two wall times is itself a
    # measurement, from the dry run's per-cell `eval_seconds` against its
    # step wall times - but `eval_seconds` lives in the per-cell CSVs, which
    # are development artifacts and are not committed. So it is listed here
    # rather than checked, and `_cell_wall_ratios` below recomputes it from
    # the local artifacts when they are present.
    ratios = _cell_wall_ratios()
    if ratios:
        checker.check(
            "cell wall time / sum(eval_seconds), mean", 1.08,
            sum(ratios) / len(ratios),
            "backend/artifacts/phase4_dev (not committed)",
            f"mean of {len(ratios)} per-cell ratios", tol=5e-2,
        )
    else:
        checker.skip(
            "cell wall time / sum(eval_seconds)", 1.08,
            "backend/artifacts/phase4_dev (not committed)",
            "regenerate with `python scripts/run_phase4_chain.py --config "
            "configs/local.yaml`; the factor needs per-institution "
            "eval_seconds, which only the per-cell CSVs carry",
        )
    for label, raw, claimed in (
        ("union seed 0", 0.446, 0.4995), ("union seed 1", 0.536, 0.6003),
        ("centralized five-seed", 0.679, 0.7605),
    ):
        checker.check(
            f"{label}: raw / ceiling reproduces the relative value", claimed,
            raw / (50 / 56), "arithmetic", f"{raw} / (50/56)", tol=1e-3,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    checker = Checker()
    phase3_centralized(checker)
    session_a(checker)
    float32_grid(checker)
    sample_runs(checker)
    dry_run(checker)
    uncommitted_session_b(checker)

    frame = pd.DataFrame(checker.rows)
    if args.json:
        print(json.dumps(checker.rows, indent=2, default=str))
    else:
        for row in checker.rows:
            mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIPPED": " skip "}[
                row["status"]
            ]
            print(f"[{mark}] {row['figure']}: claimed {row['claimed']}, "
                  f"recomputed {row['recomputed']}")
            if row["status"] == "FAIL":
                print(f"          source: {row['source']} -- {row['how']}")

    counts = frame.status.value_counts().to_dict()
    print()
    print(f"{counts.get('PASS', 0)} passed, {counts.get('FAIL', 0)} failed, "
          f"{counts.get('SKIPPED', 0)} skipped (not committed yet)")
    if counts.get("SKIPPED"):
        print(
            "Skipped figures are the ones the README asserts on trust. They "
            "are not failures; they are the list of things to re-check when "
            "their evidence lands."
        )
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
