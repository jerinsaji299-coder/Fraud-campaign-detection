"""Stage 4.1 locally on the 1% sample: the isolated condition, end to end.

Runs the whole condition for each ``(target, seed)`` at one lookback and
writes the four report files stage 4.1 is defined by. A **development
check**, not a result: the sample thins the background to 1%, so precision,
false alarms per day and every budget-chosen tau do not transfer. The
full-data numbers come from the Kaggle runner's ``train-isolated`` /
``evaluate-isolated`` subcommands.

    cd backend
    python scripts/run_isolated.py --config configs/local.yaml
    python scripts/run_isolated.py --config configs/local.yaml --targets 1.0 --seeds 0

Exit codes follow the Phase 3 runner: **0** normally, **3** if any
institution's chosen tau sits at an edge of the model grid — after every
report has been written, because with 96 cells aborting on the first edge tau
would throw away hours of finished work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

from fraudcamp import constants, pipeline  # noqa: E402
from fraudcamp.federation import isolated  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--lookback", type=int, default=48,
                    help="the Phase 4 frozen lookback (default: 48)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="default: seed 0 only, since a cell costs minutes")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=constants.VISIBILITY_TARGETS)
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="default: val and test; add stress for the full set")
    ap.add_argument("--out", default="artifacts/isolated_dev")
    args = ap.parse_args()

    t0 = time.time()
    config = BACKEND / args.config if not Path(args.config).is_absolute() else Path(args.config)
    print(f"building the pipeline from {config} ...", flush=True)
    result = pipeline.build(config)
    print(
        f"  {len(result.full_df):,} transactions, {len(result.camp)} campaigns, "
        f"{int(result.camp['eval_ok'].sum())} eval_ok, K = {constants.K_INSTITUTIONS}, "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    out_dir = BACKEND / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    collected: dict[str, list[dict]] = {
        "institution_rows": [], "tau_rows": [], "ceiling_rows": [],
        "summary_rows": [], "pooled_rows": [], "union_rows": [], "outcome_rows": [],
    }
    for target in args.targets:
        for seed in args.seeds:
            print(
                f"\n=== target {target}, seed {seed}, L = {args.lookback}h ===",
                flush=True,
            )
            cell = isolated.run_cell(
                result, target=target, seed=seed, lookback_h=args.lookback,
                splits=tuple(args.splits), progress=lambda line: print(line, flush=True),
            )
            for key in collected:
                collected[key].extend(cell[key])
            cells.append({k: v for k, v in cell.items() if not k.endswith("_rows")})
            flags = cell["flags"]
            print(
                f"  done in {cell['seconds']}s — untrained {flags['untrained']}, "
                f"thin {flags['thin']}, tau at grid edge {flags['tau_at_grid_edge']}, "
                f"{len(flags['no_affordable_tau'])} (institution, point) pair(s) "
                f"with no affordable tau"
            )
            _print_cell(cell, args.splits)

    files = {
        "phase4_isolated_institutions.csv": collected["institution_rows"],
        "phase4_isolated_taus.csv": collected["tau_rows"],
        "phase4_isolated_ceilings.csv": collected["ceiling_rows"],
        "phase4_isolated_summary.csv": collected["summary_rows"],
        "phase4_isolated_pooled.csv": collected["pooled_rows"],
        "phase4_isolated_union.csv": collected["union_rows"],
        "phase4_isolated_outcomes.csv": collected["outcome_rows"],
    }
    for name, rows in files.items():
        pd.DataFrame(rows).to_csv(out_dir / name, index=False)

    report = {
        "stage": "4.1",
        "condition": isolated.CONDITION,
        "development_only": True,
        "note": (
            "measured on the 1% sample: the background is thinned to 1%, so "
            "precision, false alarms per day and every budget tau do not "
            "transfer to the full data"
        ),
        "config": str(config),
        "lookback_h": args.lookback,
        "seeds": args.seeds,
        "targets": args.targets,
        "splits": args.splits,
        "k_institutions": constants.K_INSTITUTIONS,
        "operating_points": isolated.operating_points(),
        "cells": cells,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "phase4_isolated.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    edge_taus = [
        (cell["target"], cell["seed"], cell["flags"]["tau_at_grid_edge"])
        for cell in cells if cell["flags"]["tau_at_grid_edge"]
    ]
    print(f"\nwrote {len(files) + 1} files to {out_dir}")
    print(f"total {time.time() - t0:.1f}s")
    if edge_taus:
        print("\nSTOPPING: a chosen tau is at an edge of the model grid.")
        for target, seed, institutions in edge_taus:
            print(f"  target {target} seed {seed}: institutions {institutions}")
        print("Reports above are complete. On the sample this is expected —")
        print("the thinned background leaves a positive rate two orders of")
        print("magnitude too high — but on full data it must be investigated")
        print("before the numbers are used.")
        return 3
    return 0


def _print_cell(cell: dict, splits: list[str]) -> None:
    """The per-institution table and the two pooled numbers, as read in the
    README: recall against each institution's **own** ceiling."""
    institutions = pd.DataFrame(cell["institution_rows"])
    if len(institutions):
        print("\n  -- what each institution had to train on --")
        print(institutions[[
            "institution", "tier", "n_rows", "n_positive", "positive_rate",
            "class_weight_min", "class_weight_max", "tau_f1", "tau_at_grid_edge",
        ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary = pd.DataFrame(cell["summary_rows"])
    pooled = pd.DataFrame(cell["pooled_rows"])
    union = pd.DataFrame(cell["union_rows"])
    ceilings = pd.DataFrame(cell["ceiling_rows"])

    for split in splits:
        if len(ceilings):
            own = ceilings[
                (ceilings["eval_split"] == split) & (ceilings["test_subset"] == "all")
            ]
            if len(own):
                print(f"\n  -- {split}: each institution's OWN ceiling (subset all) --")
                print(own[[
                    "institution", "tier", "oracle_recall", "n_oracle_reachable",
                ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        for point in [p["operating_point"] for p in isolated.operating_points()]:
            rows = summary[
                (summary["eval_split"] == split)
                & (summary["operating_point"] == point)
                & (summary.get("test_subset") == "all")
            ] if len(summary) else summary
            if not len(rows):
                continue
            print(f"\n  -- {split}, {point} --")
            columns = [
                c for c in [
                    "institution", "tier", "defined", "tau", "campaign_recall",
                    "oracle_recall", "recall_relative_to_ceiling",
                    "false_alarms_per_day", "median_lead_time_h",
                ] if c in rows.columns
            ]
            print(rows[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

            pooled_rows = pooled[
                (pooled["eval_split"] == split) & (pooled["operating_point"] == point)
            ]
            for row in pooled_rows.to_dict("records"):
                print(
                    f"     POOLED {row['metric']}: "
                    f"over all institutions {row['mean_over_all_institutions']} "
                    f"(n = {row['n_institutions_in_mean']}), "
                    f"over trained only {row['mean_over_trained_institutions']} "
                    f"(n = {row['n_trained_in_mean']})"
                    + (
                        f", untrained {row['untrained_institutions']}"
                        if row["untrained_institutions"] else ""
                    )
                    + (
                        f", undefined {row['undefined_institutions']}"
                        if row["undefined_institutions"] else ""
                    )
                )
            union_row = union[
                (union["eval_split"] == split)
                & (union["operating_point"] == point)
                & (union["test_subset"] == "all")
            ]
            for row in union_row.to_dict("records"):
                print(
                    f"     UNION (a supervisory reader, not a bank): recall "
                    f"{row['campaign_recall']}, ceiling {row['oracle_recall']}, "
                    f"relative {row['recall_relative_to_ceiling']}, "
                    f"deduplicated {row['false_alarms_per_day']}/day from "
                    f"{row['summed_false_alarms_per_day']}/day summed "
                    f"({row['duplicate_clusters']} duplicate clusters), "
                    f"{row['n_institutions_operating']} institutions operating"
                )


if __name__ == "__main__":
    raise SystemExit(main())
