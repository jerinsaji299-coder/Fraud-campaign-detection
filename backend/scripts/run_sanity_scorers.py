#!/usr/bin/env python
"""Run the ORACLE and RANDOM sanity scorers through the real evaluation code.

These validate the evaluation module, not a model. The oracle should find
almost every evaluable campaign; random should find almost none while
raising plenty of false alarms. If either misbehaves, the evaluation code is
wrong.

Usage:
    python scripts/run_sanity_scorers.py --config data/demo/demo.yaml
    python scripts/run_sanity_scorers.py --config configs/kaggle.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND / "src") not in sys.path:
    sys.path.insert(0, str(BACKEND / "src"))

import numpy as np  # noqa: E402

from fraudcamp import constants, evaluation, pipeline  # noqa: E402
from fraudcamp.evaluation import scorers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="val,test,stress")
    parser.add_argument("--lookback", type=int, default=constants.DEFAULT_LOOKBACK_H)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    print(f"Building the pipeline from {args.config} ...")
    result = pipeline.build(args.config)
    camp, full_df = result.camp, result.full_df
    print(f"  {len(full_df):,} transactions, {len(camp)} campaigns, "
          f"{int(camp['eval_ok'].sum())} eval_ok")
    print(f"  lookback L={args.lookback}h, seed {args.seed}\n")

    def oracle(window, t):
        return scorers.oracle_scores(window["Is Laundering"].to_numpy())

    def random_scorer(window, t):
        # seeded per detection time so the run is reproducible but the
        # scores are not identical across windows
        return scorers.random_scores(len(window), args.seed + int(t.value % 100_000))

    rows = []
    for split in args.splits.split(","):
        split = split.strip()
        for name, scorer in (("oracle", oracle), ("random", random_scorer)):
            evaluated = evaluation.evaluate_split(
                full_df=full_df,
                camp=camp,
                pattern_df=result.pattern_df,
                accounts_by_campaign=result.accounts_by_campaign,
                split=split,
                scorer=scorer,
                tau=scorers.SANITY_TAU[name],
                lookback_h=args.lookback,
                seed=args.seed,
                scorer_name=name,
            )
            for subset in constants.TEST_SUBSETS:
                rows.append(evaluation.summarize(
                    evaluated, subset=subset, bootstrap_seed=args.seed,
                    n_resamples=args.bootstrap,
                ))
            if name == "oracle":
                print(f"--- {split}: oracle by topology group ---")
                for row in evaluation.summarize_by(evaluated, "group"):
                    print(f"    {row['group']:<15} {row['n_detected']:>3}/{row['n_campaigns']:<3} "
                          f"recall={row['campaign_recall'] or 0:.2f}  "
                          f"median lead={row['median_lead_time_h'] or float('nan'):.1f}h")

    print("\n" + "=" * 112)
    header = (f"{'split':<7}{'scorer':<8}{'subset':<20}{'camp':>5}{'det':>5}"
              f"{'recall':>8}{'prec':>7}{'FA':>6}{'FA/day':>8}{'other':>7}{'ambig':>7}"
              f"{'lead_h':>9}{'norm':>7}{'frac_obs':>9}")
    print(header)
    print("-" * 112)
    for row in rows:
        def fmt(value, spec=".2f"):
            return format(value, spec) if value is not None else "-"
        print(
            f"{row['split']:<7}{row['scorer']:<8}{row['test_subset']:<20}"
            f"{row['n_campaigns']:>5}{row['n_detected']:>5}"
            f"{fmt(row['campaign_recall']):>8}{fmt(row['campaign_precision']):>7}"
            f"{row['false_alarms']:>6}{row['false_alarms_per_day']:>8.1f}"
            f"{row['other_split_hits']:>7}{row['ambiguous']:>7}"
            f"{fmt(row['median_lead_time_h'], '.1f'):>9}"
            f"{fmt(row['median_normalized_lead_time']):>7}"
            f"{fmt(row['median_frac_observed_at_detection']):>9}"
        )
    print("=" * 112)

    print("\nrecall 95% bootstrap CIs (subset = all):")
    for row in rows:
        if row["test_subset"] != "all":
            continue
        low, high = row["recall_ci"]
        lead_low, lead_high = row["median_lead_time_h_ci"]
        print(
            f"  {row['split']:<7}{row['scorer']:<8} recall "
            f"{row['campaign_recall'] if row['campaign_recall'] is not None else float('nan'):.3f} "
            f"[{low if low is not None else float('nan'):.3f}, "
            f"{high if high is not None else float('nan'):.3f}]   "
            f"median lead {row['median_lead_time_h'] if row['median_lead_time_h'] is not None else float('nan'):.1f}h "
            f"[{lead_low if lead_low is not None else float('nan'):.1f}, "
            f"{lead_high if lead_high is not None else float('nan'):.1f}]"
        )

    oracle_all = [r for r in rows if r["scorer"] == "oracle" and r["test_subset"] == "all"]
    random_all = [r for r in rows if r["scorer"] == "random" and r["test_subset"] == "all"]
    print("\nSanity verdict:")
    for o, r in zip(oracle_all, random_all):
        o_recall = o["campaign_recall"] or 0.0
        r_recall = r["campaign_recall"] or 0.0
        verdict = "OK" if o_recall > 0.5 and o_recall > r_recall else "SUSPICIOUS"
        print(
            f"  {o['split']:<7} oracle recall {o_recall:.2f} vs random {r_recall:.2f}"
            f"  -> {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
