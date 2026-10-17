"""Train and evaluate the XGBoost baseline locally, on the 1% sample.

Development only. The sample keeps every campaign transaction and thins the
background to 1%, so campaign-level recall and lead time are close to the
full data while anything involving the background — precision, false alarms
per day, and any threshold chosen against them — is roughly 60x off. The
numbers worth believing from this script are shapes and orderings; the real
ones come from `kaggle_runner.py train` / `evaluate` on the full data.

    python scripts/run_xgb_baseline.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

import pandas as pd  # noqa: E402

from fraudcamp import constants, evaluation, pipeline  # noqa: E402
from fraudcamp.evaluation import scorers  # noqa: E402
from fraudcamp.models import xgb_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BACKEND / "configs" / "local.yaml"))
    parser.add_argument("--out", default=str(BACKEND / "artifacts" / "baseline_dev"))
    parser.add_argument("--lookbacks", default="24,48")
    parser.add_argument("--splits", default="val,test,stress")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lookbacks = [int(v) for v in args.lookbacks.split(",")]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 78)
    print("XGBoost baseline on the 1% SAMPLE - development numbers only")
    print("=" * 78)

    started = time.perf_counter()
    result = pipeline.build(Path(args.config))
    print(
        f"rows {len(result.full_df):,}  campaigns {len(result.camp):,}  "
        f"laundering {int(result.full_df['Is Laundering'].sum()):,} "
        f"({result.full_df['Is Laundering'].mean():.3%} - the full data is 0.102%)"
    )

    def oracle(window, t):
        return scorers.oracle_scores(window["Is Laundering"].to_numpy())

    rows: list[dict] = []
    paired_rows: list[dict] = []
    importance: dict[int, list[dict]] = {}
    training_meta: dict[int, dict] = {}
    model_outcomes: dict[tuple[int, str], list] = {}

    for lookback in lookbacks:
        print(f"\n{'=' * 78}\nL = {lookback}h\n{'=' * 78}")

        t0 = time.perf_counter()
        data = xgb_baseline.build_training_data(result.full_df, lookback)
        print(
            f"training matrix: {data.n_rows:,} rows x "
            f"{data.matrix.shape[1]} features over {len(data.times)} windows, "
            f"{data.n_positive:,} positive ({data.positive_rate:.3%}), "
            f"{data.memory_mb():.0f} MB, {time.perf_counter() - t0:.1f}s"
        )
        print("  per-window class weight (n_neg / n_pos):")
        for window in data.per_window:
            print(
                f"    {window['t']}  {window['n_edges']:>8,} edges  "
                f"{window['n_positive']:>5,} pos  weight {window['class_weight']:>8,.1f}"
            )

        t0 = time.perf_counter()
        model = xgb_baseline.train_baseline(
            result.full_df, lookback,
            params=xgb_baseline.XGBParams(seed=args.seed),
            data=data,
        )
        print(f"trained in {time.perf_counter() - t0:.1f}s")
        model.save(out / f"L{lookback}")
        training_meta[lookback] = model.training
        importance[lookback] = xgb_baseline.feature_importance(model)
        print("  top features by gain:")
        for row in importance[lookback][:8]:
            print(f"    {row['feature']:<28} {row['share']:>7.1%}")

        # tau on validation only, by campaign-level F1
        t0 = time.perf_counter()
        tau, curve = evaluation.select_tau(
            result.full_df, result.camp, result.pattern_df,
            result.accounts_by_campaign, model.scorer(),
            lookback_h=lookback, seed=args.seed, scorer_name="xgboost",
        )
        print(f"\nchosen tau = {tau} on validation ({time.perf_counter() - t0:.1f}s)")
        print(f"  {'tau':>6} {'prec':>8} {'recall':>8} {'F1':>8} {'FA/day':>10}")
        for point in curve:
            print(
                f"  {point['tau']:>6} {point['campaign_precision']:>8.3f} "
                f"{point['campaign_recall']:>8.3f} {point['campaign_f1']:>8.3f} "
                f"{point['false_alarms_per_day']:>10,.1f}"
            )

        for split in splits:
            ceiling = evaluation.evaluate_split(
                result.full_df, result.camp, result.pattern_df,
                result.accounts_by_campaign, split, oracle,
                tau=scorers.SANITY_TAU["oracle"], lookback_h=lookback,
                seed=args.seed, scorer_name="oracle",
            )
            reachable = [o.campaign_id for o in ceiling.outcomes if o.detected]

            evaluated = evaluation.evaluate_split(
                result.full_df, result.camp, result.pattern_df,
                result.accounts_by_campaign, split, model.scorer(),
                tau=tau, lookback_h=lookback, seed=args.seed,
                scorer_name="xgboost",
            )
            model_outcomes[(lookback, split)] = evaluated.outcomes

            for subset in constants.TEST_SUBSETS:
                oracle_recall = evaluation.summarize(
                    ceiling, subset=subset, bootstrap_seed=args.seed,
                    n_resamples=args.bootstrap, oracle_reachable=reachable,
                )["campaign_recall"]
                for name, evaluation_result in (
                    ("oracle", ceiling), ("xgboost", evaluated)
                ):
                    rows.append(
                        {
                            "lookback_h": lookback,
                            **evaluation.summarize(
                                evaluation_result, subset=subset,
                                bootstrap_seed=args.seed, n_resamples=args.bootstrap,
                                oracle_reachable=reachable,
                                oracle_recall=oracle_recall,
                            ),
                        }
                    )

            # earliness against the ceiling, paired - the only valid way
            paired = evaluation.paired_lead_time_comparison(
                ceiling.outcomes, evaluated.outcomes, "oracle", "xgboost",
                split=split, bootstrap_seed=args.seed,
            )
            paired_rows.append({"comparison": "xgboost_vs_oracle", **paired.as_dict()})
            print(f"\n  {split}: vs the ceiling, paired")
            print(f"    {evaluation.describe_paired_comparison(paired)}")

    # the model's own lookback comparison, paired
    for split in splits:
        for earlier, later in zip(lookbacks, lookbacks[1:]):
            a = model_outcomes.get((earlier, split))
            b = model_outcomes.get((later, split))
            if not a or not b:
                continue
            paired = evaluation.paired_lead_time_comparison(
                a, b, f"L={earlier}h", f"L={later}h", split=split,
                bootstrap_seed=args.seed,
            )
            paired_rows.append({"comparison": "lookbacks", **paired.as_dict()})
            print(f"\n{split}: the model's own lookbacks, paired")
            print(f"  {evaluation.describe_paired_comparison(paired)}")

    print("\n" + "=" * 118)
    print(
        f"{'L':>3} {'split':<7}{'scorer':<9}{'subset':<20}{'camp':>5}{'det':>5}"
        f"{'recall':>8}{'vs ceil':>9}{'prec':>7}{'FA/day':>9}{'lead_h':>9}{'frac':>7}"
    )
    print("-" * 118)

    def fmt(value, spec=".3f"):
        return format(value, spec) if value is not None else "-"

    for row in rows:
        print(
            f"{row['lookback_h']:>3} {row['split']:<7}{row['scorer']:<9}"
            f"{row['test_subset']:<20}{row['n_campaigns']:>5}{row['n_detected']:>5}"
            f"{fmt(row['campaign_recall']):>8}"
            f"{fmt(row['recall_relative_to_ceiling']):>9}"
            f"{fmt(row['campaign_precision']):>7}"
            f"{row['false_alarms_per_day']:>9,.1f}"
            f"{fmt(row['median_lead_time_h'], '.1f'):>9}"
            f"{fmt(row['median_frac_observed_at_detection'], '.2f'):>7}"
        )
    print("=" * 118)

    pd.DataFrame(rows).to_csv(out / "baseline_dev_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "baseline_dev_paired.csv", index=False)
    (out / "baseline_dev.json").write_text(
        json.dumps(
            {
                "note": (
                    "1% SAMPLE, development only. Every campaign transaction is "
                    "kept and the background is thinned to 1%, so the laundering "
                    "rate is ~6% against the full data's 0.102%. Precision, "
                    "false alarms per day and any threshold chosen against them "
                    "do not transfer."
                ),
                "config": str(args.config),
                "seed": args.seed,
                "params": xgb_baseline.XGBParams(seed=args.seed).__dict__,
                "training": {str(k): v for k, v in training_meta.items()},
                "feature_importance": {str(k): v for k, v in importance.items()},
                "summary": rows,
                "paired": paired_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out / 'baseline_dev.json'} (+ 2 CSVs, models under L*/)")
    print(f"Total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
