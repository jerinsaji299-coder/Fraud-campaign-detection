"""Stage 4.0 report: what each institution actually has to train on.

Builds the per-institution views for every (seed, target) and reports the
pooled training rows and positives each institution holds, so that an
institution with too few positives to train is a *finding* before stage 4.2
spends a session discovering it.

Run locally on the 1% sample (a development check, not a result — the sample
thins the background to 1%, so absolute row counts and positive rates do not
transfer) or on Kaggle against the full data:

    cd backend
    python scripts/run_institution_split.py --config configs/local.yaml
    python scripts/run_institution_split.py --config configs/kaggle.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

from fraudcamp import constants, pipeline, windows  # noqa: E402
from fraudcamp.federation import views as fv  # noqa: E402


def build_training_windows(df: pd.DataFrame, lookback_h: int):
    """The rule-3-valid training windows, from `windows.py` — the same set
    stage 3.3 trained on. Not re-derived here."""
    times = windows.training_detection_times(lookback_h=lookback_h)
    built = [windows.build_window(df, t, lookback_h=lookback_h) for t in times]
    return [w for w in built if w.n_transactions > 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[48, 24])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--targets", type=float, nargs="+", default=constants.VISIBILITY_TARGETS)
    ap.add_argument("--out", default="artifacts/federation_dev")
    args = ap.parse_args()

    t0 = time.time()
    print(f"building the pipeline from {args.config} ...", flush=True)
    result = pipeline.build(BACKEND / args.config if not Path(args.config).is_absolute() else args.config)
    print(
        f"  {len(result.full_df):,} transactions, {len(result.camp)} campaigns, "
        f"K = {constants.K_INSTITUTIONS}, {time.time() - t0:.1f}s",
        flush=True,
    )

    out_dir = BACKEND / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    campaign_checks = []

    for lookback_h in args.lookbacks:
        train = build_training_windows(result.full_df, lookback_h)
        print(
            f"\n=== L = {lookback_h}h: {len(train)} rule-3 training windows "
            f"({windows.training_detection_times(lookback_h=lookback_h)[0]} ... "
            f"{windows.training_detection_times(lookback_h=lookback_h)[-1]}) ===",
            flush=True,
        )

        # the natural split, for reference: no account reassigned at all
        natural = fv.natural_assignment(result)
        nat = fv.view_load_report(train, natural, lookback_h)
        print(
            f"\n-- natural bank split (no reassignment) -- central: "
            f"{nat.central['rows']:,} rows, {nat.central['positives']:,} positives "
            f"(rate {nat.central['positive_rate']:.4%}, class weight "
            f"{nat.central['class_weight']:.0f})"
        )
        print(nat.per_institution.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for row in nat.per_institution.to_dict("records"):
            all_rows.append({"lookback_h": lookback_h, "seed": -1, "target": "natural", **row})

        for seed in args.seeds:
            for target in args.targets:
                assignment, achieved = fv.assignment_from_pipeline(result, target=target, seed=seed)
                report = fv.view_load_report(train, assignment, lookback_h)

                # cross-check: views must reproduce the recorded visibility
                table = fv.campaign_visibility_table(result.full_df, assignment)
                mismatches = [
                    int(r.campaign_id)
                    for r in table.itertuples()
                    if abs(r.visibility_from_views - achieved[int(r.campaign_id)]) > 1e-9
                ]
                campaign_checks.append(
                    {
                        "lookback_h": lookback_h,
                        "seed": seed,
                        "target": target,
                        "campaigns_checked": len(table),
                        "mismatches": len(mismatches),
                    }
                )

                starved = report.starved
                flag = (
                    f"  <-- {len(starved)} institution(s) under "
                    f"{fv.MIN_POSITIVES_TO_TRAIN} positives: "
                    f"{sorted(starved['institution'].tolist())}"
                    if len(starved)
                    else ""
                )
                print(
                    f"\n-- seed {seed}, target {target} -- "
                    f"{assignment.n_reassigned} accounts reassigned; "
                    f"visibility cross-check: {len(table)} campaigns, "
                    f"{len(mismatches)} mismatches{flag}"
                )
                print(
                    report.per_institution.to_string(
                        index=False, float_format=lambda v: f"{v:.4f}"
                    )
                )
                for row in report.per_institution.to_dict("records"):
                    all_rows.append(
                        {"lookback_h": lookback_h, "seed": seed, "target": target, **row}
                    )

    frame = pd.DataFrame(all_rows)
    csv_path = out_dir / "institution_split.csv"
    frame.to_csv(csv_path, index=False)
    checks = pd.DataFrame(campaign_checks)
    checks.to_csv(out_dir / "institution_split_checks.csv", index=False)

    print("\n=== summary across every (seed, target) ===")
    experiment = frame[frame["seed"] >= 0]
    if len(experiment):
        grouped = experiment.groupby(["lookback_h", "target"]).agg(
            min_positives=("positives", "min"),
            max_positives=("positives", "max"),
            min_rows=("rows", "min"),
            max_rows=("rows", "max"),
            starved_cells=("positives", lambda s: int((s < fv.MIN_POSITIVES_TO_TRAIN).sum())),
        )
        print(grouped.to_string())
    print(f"\nvisibility cross-check mismatches, all runs: {int(checks['mismatches'].sum())}")
    print(f"wrote {csv_path}")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
