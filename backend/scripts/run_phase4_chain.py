"""The whole Phase 4 chain locally on the 1% sample: 4.1 -> 4.2 -> 4.3 -> 4.4 -> compare.

A **development check**, not a result. The sample thins the background to 1%,
so the positive rate is ~3.3% against the full data's 0.102%: precision, false
alarms per day, every budget-chosen tau and every binding count are properties
of the thinned sample and do not transfer. Its job is to prove the chain runs
end to end, that the pairing rules hold on real rows, and that the report
assembles - before any of it is run on Kaggle.

    cd backend
    python scripts/run_phase4_chain.py --config configs/local.yaml
    python scripts/run_phase4_chain.py --config configs/local.yaml --targets 1.0 --seeds 0

What it does per ``(target, seed)``, in order, reusing one set of boosters
wherever two conditions are defined to share them:

* **4.1 isolated** - train K boosters on K views, evaluate;
* **4.2 fedavg** - the *same* boosters, every one applied to every view, scores
  averaged. Nothing retrained, so 4.2 - 4.1 is a pure scoring contrast;
* **4.3a evidence** and **4.3b fedavg_evidence** - K boosters retrained on
  augmented views, then evaluated twice from one set of boosters and one
  shared-evidence object, so the two differ in the scoring rule alone.

Then once, not per cell:

* **4.4 centralized** - the ``K = 1`` case. On the full data the Phase 3
  booster is loaded; **on the sample there is none, so one is trained here**
  with the frozen configuration at model seed 0. That is a retrain, it is
  stated in the report, and sample centralized numbers are not comparable to
  Phase 3's full-data 0.679 / 0.760 for the same reason nothing else here is.
* **compare-conditions** - every paired contrast, primary points first.

Exit codes: **0** normally, **3** if any chosen tau sat at an edge of the
model grid (after every report is written), **4** if a pairing rule was
violated, which is a stop condition rather than a bad number.
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
from fraudcamp.models import xgb_baseline  # noqa: E402
from fraudcamp.federation import (  # noqa: E402
    centralized as centralized_mod,
    compare as compare_mod,
    evidence as evidence_mod,
    fedavg as fedavg_mod,
    isolated,
    views as fv,
)

CELL_KEYS = ("institution_rows", "tau_rows", "ceiling_rows", "summary_rows",
             "pooled_rows", "union_rows", "outcome_rows")
SUFFIX = {"institution_rows": "institutions", "tau_rows": "taus",
          "ceiling_rows": "ceilings", "summary_rows": "summary",
          "pooled_rows": "pooled", "union_rows": "union",
          "outcome_rows": "outcomes"}


def _say(message: str) -> None:
    print(message, flush=True)


def _merge(into: dict, cell: dict) -> None:
    for key in CELL_KEYS:
        into.setdefault(key, []).extend(cell.get(key, []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--lookback", type=int, default=constants.PHASE4_LOOKBACK_H)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--targets", type=float, nargs="+",
                    default=constants.VISIBILITY_TARGETS)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default="artifacts/phase4_dev")
    ap.add_argument("--skip-evidence", action="store_true",
                    help="run 4.1/4.2/4.4 only, as the Kaggle plan allows if "
                         "time runs out")
    ap.add_argument("--max-cached-windows", type=int, default=None,
                    help="cap on cached shared-evidence windows (memory)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lookback = int(args.lookback)
    splits = tuple(args.splits)
    started = time.perf_counter()
    timings: list[dict] = []

    _say(f"building pipeline from {args.config} ...")
    t0 = time.perf_counter()
    result = pipeline.build(Path(args.config))
    timings.append({"step": "build pipeline", "seconds": round(time.perf_counter() - t0, 1)})
    _say(f"  {len(result.full_df):,} transactions, {len(result.camp)} campaigns")

    collected: dict[str, dict] = {}
    edge_taus: list[dict] = []
    high_edge: list[dict] = []
    low_edge: list[dict] = []
    cell_reports: list[dict] = []
    evidence_stats: list[dict] = []

    conditions = [isolated.CONDITION, fedavg_mod.CONDITION]
    if not args.skip_evidence:
        conditions += [evidence_mod.CONDITION_EVIDENCE,
                       evidence_mod.CONDITION_FEDAVG_EVIDENCE]

    for target in args.targets:
        for seed in args.seeds:
            name = isolated.cell_name(target, seed)
            _say(f"\n=== cell {name}: target {target}, seed {seed}, L = {lookback}h ===")
            t_cell = time.perf_counter()

            assignment, _achieved = fv.assignment_from_pipeline(
                result, target=target, seed=seed
            )

            # --- 4.1, and the boosters 4.2 reuses --------------------
            _say("  4.1 isolated: training K boosters ...")
            t0 = time.perf_counter()
            models41 = isolated.train_all_isolated(
                result.full_df, assignment, lookback, progress=_say
            )
            timings.append({"step": f"{name} train isolated",
                            "seconds": round(time.perf_counter() - t0, 1)})

            for condition, cell_fn in (
                (isolated.CONDITION, lambda: isolated.run_cell(
                    result, target=target, seed=seed, lookback_h=lookback,
                    splits=splits, models=models41, progress=_say)),
                (fedavg_mod.CONDITION, lambda: fedavg_mod.run_cell(
                    result, target=target, seed=seed, lookback_h=lookback,
                    splits=splits, models=models41, progress=_say)),
            ):
                _say(f"  {condition}: evaluating ...")
                t0 = time.perf_counter()
                cell = cell_fn()
                timings.append({"step": f"{name} {condition}",
                                "seconds": round(time.perf_counter() - t0, 1)})
                _merge(collected.setdefault(condition, {}), cell)
                _record_flags(cell, condition, target, seed, edge_taus,
                              high_edge, low_edge, cell_reports)

            # --- 4.3: one set of augmented boosters, two evaluations -
            if not args.skip_evidence:
                _say("  4.3: training K boosters on AUGMENTED views ...")
                runner = evidence_mod.EvidenceRunner(
                    result.full_df, lookback, args.max_cached_windows
                )
                t0 = time.perf_counter()
                models43 = runner.trainer(
                    result.full_df, assignment, lookback, None, progress=_say
                )
                timings.append({"step": f"{name} train evidence",
                                "seconds": round(time.perf_counter() - t0, 1)})
                for condition in (evidence_mod.CONDITION_EVIDENCE,
                                  evidence_mod.CONDITION_FEDAVG_EVIDENCE):
                    _say(f"  {condition}: evaluating ...")
                    t0 = time.perf_counter()
                    cell = evidence_mod.run_cell(
                        result, target=target, seed=seed,
                        condition=condition, lookback_h=lookback,
                        splits=splits, models=models43, runner=runner,
                        progress=_say,
                    )
                    timings.append({"step": f"{name} {condition}",
                                    "seconds": round(time.perf_counter() - t0, 1)})
                    _merge(collected.setdefault(condition, {}), cell)
                    _record_flags(cell, condition, target, seed, edge_taus,
                                  high_edge, low_edge, cell_reports)
                evidence_stats.append({
                    "visibility_target": target, "seed": seed,
                    **runner.provider(assignment).stats(),
                })

            _say(f"  cell {name} done in {time.perf_counter() - t_cell:.1f}s")

    # --- 4.4 centralized, once ------------------------------------------
    _say(f"\n=== 4.4 centralized ({centralized_mod.PHASE4_POLICY}) ===")
    policy = centralized_mod.PHASE4_POLICY
    model_seeds = centralized_mod.model_seed_policy(policy)
    boosters = {}
    for model_seed in model_seeds:
        _say(f"  NO saved Phase 3 booster on the sample: training one at "
             f"model seed {model_seed} with the frozen configuration")
        t0 = time.perf_counter()
        boosters[model_seed] = xgb_baseline.train_baseline(
            result.full_df, lookback_h=lookback,
            params=xgb_baseline.XGBParams(seed=model_seed),
            variant="xgboost_centralized",
        )
        timings.append({"step": f"train centralized s{model_seed}",
                        "seconds": round(time.perf_counter() - t0, 1)})
    t0 = time.perf_counter()
    central = centralized_mod.run(
        result, boosters, policy, targets=args.targets, seeds=args.seeds,
        lookback_h=lookback, splits=splits, progress=_say,
    )
    timings.append({"step": "evaluate centralized",
                    "seconds": round(time.perf_counter() - t0, 1)})

    # --- write every condition's CSVs -----------------------------------
    written: list[str] = []
    for condition, rows in sorted(collected.items()):
        for key in CELL_KEYS:
            frame = pd.DataFrame(rows.get(key, []))
            path = out / f"phase4_{condition}_{SUFFIX[key]}.csv"
            frame.to_csv(path, index=False)
            written.append(path.name)
    for key, suffix in (("rows", "summary"), ("tau_rows", "taus"),
                        ("ceiling_rows", "ceilings"),
                        ("outcome_rows", "outcomes")):
        path = out / f"phase4_centralized_{suffix}.csv"
        pd.DataFrame(central[key]).to_csv(path, index=False)
        written.append(path.name)
    if evidence_stats:
        pd.DataFrame(evidence_stats).to_csv(
            out / "phase4_evidence_stats.csv", index=False
        )
        written.append("phase4_evidence_stats.csv")

    # --- compare ---------------------------------------------------------
    _say("\n=== compare-conditions ===")
    summaries = {
        c: pd.DataFrame(rows.get("summary_rows", []))
        for c, rows in collected.items()
    }
    unions = {
        c: pd.DataFrame(rows.get("union_rows", []))
        for c, rows in collected.items()
    }
    centralized_rows = pd.DataFrame(central["rows"])
    try:
        comparison = compare_mod.compare(
            summaries, unions, centralized_rows, progress=_say
        )
    except compare_mod.NonIdenticalCells as exc:
        _say(f"\nSTOPPING: a pairing rule was violated.\n  {exc}")
        return 4

    frame = pd.DataFrame(comparison["rows"])
    frame.to_csv(out / "phase4_compare.csv", index=False)
    written.append("phase4_compare.csv")
    head = compare_mod.headline(comparison["rows"])
    head.to_csv(out / "phase4_compare_headline.csv", index=False)
    written.append("phase4_compare_headline.csv")

    payload = {
        "stage": "4.1-4.4 + compare",
        "scope": {
            "targets": args.targets, "seeds": args.seeds,
            "lookback_h": lookback, "splits": list(splits),
            "conditions": conditions,
            "centralized_policy": policy,
            "centralized_model_seeds": list(model_seeds),
            "centralized_retrained_here": True,
        },
        "DEVELOPMENT_ONLY": (
            "1% sample. The positive rate is ~3.3% against the full data's "
            "0.102%, so precision, false alarms per day, budget-chosen taus "
            "and binding counts are properties of the thinned sample. Not a "
            "result; not comparable to Phase 3's full-data centralized "
            "0.679 / 0.760."
        ),
        "checks": comparison["checks"],
        "provenance": comparison["provenance"],
        "skipped_contrasts": comparison["skipped"],
        "seed_noise_floor": comparison["seed_noise_floor"],
        "centralized_reproduction": central["reproduction"],
        "evidence_stats": evidence_stats,
        "cells": cell_reports,
        "tau_at_grid_edge": edge_taus,
        "high_edge": high_edge,
        "low_edge": low_edge,
        "timings": timings,
        "total_seconds": round(time.perf_counter() - started, 1),
        "files": sorted(written),
    }
    (out / "phase4_chain.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (out / "phase4_chain.md").write_text(
        _markdown(payload, head, comparison), encoding="utf-8"
    )
    _say(f"\nwrote {len(written) + 2} files to {out}")
    _print_headline(head, comparison)

    if low_edge:
        _say(f"\n{len(low_edge)} F1 tau(s) at the LOW edge of the grid - expected "
             "on the 1% sample, where the positive rate is ~3.3%.")
    if high_edge:
        _say(f"\n{len(high_edge)} (institution, operating point) cell(s) EXCLUDED "
             "for a tau at the top of the grid.")
    return 3 if edge_taus else 0


def _record_flags(cell, condition, target, seed, edge_taus, high_edge,
                  low_edge, cell_reports):
    flags = cell.get("flags", {})
    cell_reports.append({
        "condition": condition, "target": target, "seed": seed,
        "seconds": cell.get("seconds"),
        "untrained": flags.get("untrained"), "thin": flags.get("thin"),
        "n_high_edge": len(flags.get("tau_at_high_edge", [])),
        "n_low_edge": len(flags.get("tau_at_low_edge", [])),
        "n_no_affordable_tau": len(flags.get("no_affordable_tau", [])),
    })
    for institution, point in flags.get("tau_at_high_edge", []):
        high_edge.append({"condition": condition, "visibility_target": target,
                          "seed": seed, "institution": institution,
                          "operating_point": point})
    for institution, point in flags.get("tau_at_low_edge", []):
        low_edge.append({"condition": condition, "visibility_target": target,
                         "seed": seed, "institution": institution,
                         "operating_point": point})
    for institution in flags.get("tau_at_grid_edge", []):
        edge_taus.append({"condition": condition, "visibility_target": target,
                          "seed": seed, "institution": institution})


def _print_headline(head: pd.DataFrame, comparison: dict) -> None:
    checks = comparison["checks"]
    _say("\n--- checks ---")
    for key, value in checks.items():
        _say(f"  {key}: {value}")
    if not len(head):
        _say("  (no primary-point rows to show)")
        return
    _say("\n--- primary points, test, subset all, relative to own ceiling ---")
    cols = ["contrast", "unit", "visibility_target", "seed", "operating_point",
            "n_pairs", "mean_difference", "median_difference",
            "min_difference", "max_difference", "n_positive", "n_negative",
            "below_seed_noise_floor"]
    _say(head[[c for c in cols if c in head.columns]].to_string(index=False))


def _markdown(payload: dict, head: pd.DataFrame, comparison: dict) -> str:
    lines = [
        "# Phase 4 chain on the 1% sample: 4.1 -> 4.2 -> 4.3 -> 4.4 -> compare",
        "",
        "> **DEVELOPMENT ONLY - not a result.**",
        f"> {payload['DEVELOPMENT_ONLY']}",
        "",
        "## Scope",
        "",
        "```json",
        json.dumps(payload["scope"], indent=2),
        "```",
        "",
        "## Checks that must hold before any difference is read",
        "",
        "```json",
        json.dumps(comparison["checks"], indent=2, default=str),
        "```",
        "",
        "## The seed-noise floor",
        "",
        "Phase 4's seeds vary visibility only - every booster is model seed 0 -",
        "so model-seed noise is held fixed and invisible here. Phase 3 measures",
        "it at centralized across five model seeds: 0.661-0.696 recall,",
        "0.74-0.78 relative. **A difference smaller than ~0.04 relative recall",
        "(~2 campaigns of 56) must not be reported as an effect**, and every row",
        "below carries `below_seed_noise_floor` saying whether it is.",
        "",
        "## Primary points: `f1` and `total_10`, test, subset `all`",
        "",
    ]
    if len(head):
        cols = ["contrast", "unit", "visibility_target", "seed",
                "operating_point", "n_pairs", "mean_difference",
                "median_difference", "min_difference", "max_difference",
                "n_positive", "n_negative", "below_seed_noise_floor"]
        keep = [c for c in cols if c in head.columns]
        lines.append("| " + " | ".join(keep) + " |")
        lines.append("|" + "---|" * len(keep))
        for row in head[keep].to_dict("records"):
            lines.append("| " + " | ".join(str(row[c]) for c in keep) + " |")
    else:
        lines.append("_no primary-point rows_")
    lines += [
        "",
        "## Contrasts that were skipped",
        "",
        "```json",
        json.dumps(comparison["skipped"], indent=2, default=str),
        "```",
        "",
        "## Centralized reproduction check",
        "",
        "```json",
        json.dumps(payload["centralized_reproduction"], indent=2, default=str),
        "```",
        "",
        "## Timing",
        "",
        "| Step | Seconds |",
        "|---|---:|",
    ]
    for row in payload["timings"]:
        lines.append(f"| {row['step']} | {row['seconds']} |")
    lines += ["", f"**Total {payload['total_seconds']}s.**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
