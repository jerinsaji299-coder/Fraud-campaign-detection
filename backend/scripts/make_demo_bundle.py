"""Export everything the live demo panel needs, and nothing else.

The panel recomputes detection **on the spot** when its toggle flips, so the
API has to hold the demo windows' transactions and the eight boosters for one
``(target, seed)``. This script is what puts them on disk, so that the server
still never reads the raw dataset: it reads a small, explicit bundle that this
script wrote.

    cd backend
    python scripts/make_demo_bundle.py --config configs/local.yaml
    python scripts/make_demo_bundle.py --config configs/local.yaml --target 1.0 --seed 0

**The demo campaign is chosen by a stated rule, to illustrate the mechanism.**
It is not a representative result and the README says so. The rule, applied in
this order and recorded in the manifest:

1. candidates are the **test-split** campaigns with ``eval_ok`` and at least
   ``MATCH_MIN_ACCOUNTS`` accounts - anything else cannot be matched by the
   frozen rule even in principle;
2. rank by the **number of distinct institutions holding the campaign's
   accounts**, descending, then by ``campaign_id`` ascending. The panel exists
   to show that no single institution sees a whole campaign, so the campaign
   most spread across institutions is the one that shows it;
3. walk down that ranking and take the first campaign the **centralized oracle
   detects** within the test detection horizon at this lookback. A campaign no
   detector could ever reach would demonstrate nothing;
4. record the chosen rank and every candidate that was skipped, with why.

What the bundle contains, and nothing more: the offered windows' transactions,
the account -> institution map for those windows, the campaign ground truth
the frozen matching rule needs, the per-institution taus for the primary
operating points, and the eight boosters.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudcamp import constants, evaluation, pipeline, windows  # noqa: E402
from fraudcamp.evaluation import clusters as cluster_tools  # noqa: E402
from fraudcamp.federation import isolated, views as fv  # noqa: E402


def _say(message: str) -> None:
    print(message, flush=True)


def select_campaign(
    result, assignment, lookback_h: int, split: str = "test"
) -> dict:
    """The stated rule, with its whole audit trail."""
    camp = result.camp
    eligible = camp[(camp["eval_ok"]) & (camp["split"] == split)]
    account_institutions = {}
    candidates = []
    for row in eligible.itertuples():
        cid = int(row.campaign_id)
        accounts = result.accounts_by_campaign.get(cid, ())
        if len(accounts) < constants.MATCH_MIN_ACCOUNTS:
            continue
        held = sorted({assignment.institution_of(a) for a in accounts})
        account_institutions[cid] = held
        candidates.append({
            "campaign_id": cid,
            "n_accounts": len(accounts),
            "n_institutions_holding": len(held),
            "institutions_holding": held,
            "n_txn": int(row.n_txn),
            "deadline": str(row.deadline),
        })
    candidates.sort(key=lambda c: (-c["n_institutions_holding"], c["campaign_id"]))

    # step 3: the centralized oracle over this split, once
    detection_times = windows.evaluation_detection_times(split)
    oracle = evaluation.oracle_scorer() if hasattr(evaluation, "oracle_scorer") else None
    reachable: set[int] = set()
    if oracle is None:
        def oracle(window_transactions, t):
            return evaluation.oracle_scores(
                window_transactions["Is Laundering"].to_numpy()
            )
    oracle_result = evaluation.evaluate_split(
        result.full_df, result.camp, result.pattern_df,
        result.accounts_by_campaign, split=split, scorer=oracle,
        tau=0.5, lookback_h=lookback_h, scorer_name="centralized_oracle",
    )
    reachable = {o.campaign_id for o in oracle_result.outcomes if o.detected}

    chosen = None
    skipped = []
    for rank, candidate in enumerate(candidates):
        if candidate["campaign_id"] in reachable:
            chosen = {**candidate, "rank_in_ranking": rank}
            break
        skipped.append({
            **candidate,
            "rank_in_ranking": rank,
            "skipped_because": "the centralized oracle does not detect it in "
                               f"the {split} horizon at L = {lookback_h}h",
        })

    return {
        "rule": [
            f"candidates: {split}-split campaigns with eval_ok and at least "
            f"{constants.MATCH_MIN_ACCOUNTS} accounts",
            "rank by the number of distinct institutions holding the "
            "campaign's accounts (descending), then campaign_id (ascending)",
            "take the first the centralized oracle detects within the split's "
            f"detection horizon at L = {lookback_h}h",
        ],
        "chosen": chosen,
        "skipped": skipped,
        "n_candidates": len(candidates),
        "n_oracle_reachable": len(reachable),
        "top_candidates": candidates[:8],
        "DISCLAIMER": (
            "Chosen to ILLUSTRATE THE MECHANISM, by the rule above. It is not "
            "a representative result and must never be described as one."
        ),
    }


def offered_windows(result, campaign_id: int, lookback_h: int, split: str,
                    max_windows: int) -> list[pd.Timestamp]:
    """The detection times worth offering: those around the campaign.

    Every time in the split's horizon whose window contains at least one of
    the campaign's transactions, plus the first time after its deadline so the
    panel can show a miss as well as a hit. Capped, newest-relevant first.
    """
    times = windows.evaluation_detection_times(split)
    pattern = result.pattern_df[result.pattern_df["campaign_id"] == campaign_id]
    stamps = pd.to_datetime(pattern["ts"]).to_numpy()
    keep = []
    for t in times:
        lower = pd.Timestamp(t) - pd.Timedelta(hours=lookback_h)
        if ((stamps >= np.datetime64(lower)) & (stamps < np.datetime64(pd.Timestamp(t)))).any():
            keep.append(pd.Timestamp(t))
    if not keep:
        keep = [pd.Timestamp(t) for t in times[: max_windows]]
    return keep[:max_windows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=constants.PHASE4_LOOKBACK_H)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--models", default=None,
                    help="directory holding t<TTT>_s<S>_i<I> booster folders; "
                         "if absent the boosters are trained here")
    ap.add_argument("--out", default="artifacts/demo_panel")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lookback = int(args.lookback)
    started = time.perf_counter()

    _say(f"building pipeline from {args.config} ...")
    result = pipeline.build(Path(args.config))
    assignment, achieved = fv.assignment_from_pipeline(
        result, target=args.target, seed=args.seed
    )
    _say(f"  assignment: target {args.target}, seed {args.seed}, "
         f"{assignment.n_reassigned} accounts reassigned")

    _say("selecting the demo campaign by the stated rule ...")
    selection = select_campaign(result, assignment, lookback, args.split)
    if selection["chosen"] is None:
        _say("no candidate campaign is oracle-reachable: nothing to demo")
        return 2
    campaign_id = selection["chosen"]["campaign_id"]
    _say(f"  campaign {campaign_id}: "
         f"{selection['chosen']['n_accounts']} accounts across "
         f"{selection['chosen']['n_institutions_holding']} institutions, "
         f"rank {selection['chosen']['rank_in_ranking']} of "
         f"{selection['n_candidates']}")

    times = offered_windows(result, campaign_id, lookback, args.split,
                            args.max_windows)
    _say(f"  offering {len(times)} detection window(s)")

    # --- the windows themselves ---------------------------------------
    frames = []
    window_meta = []
    for t in times:
        txns = windows.window_slice(result.full_df, t, lookback)
        frames.append(txns.assign(detection_time=pd.Timestamp(t)))
        window_meta.append({
            "detection_time": str(t),
            "n_edges": int(len(txns)),
            "n_accounts": int(len(set(txns["src"]) | set(txns["dst"]))),
            "n_laundering_edges": int(txns["Is Laundering"].sum()),
            "n_campaign_edges": int((txns["campaign_id"] == campaign_id).sum()),
        })
    bundle = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    bundle.to_parquet(out / "demo_windows.parquet", index=False)

    # --- ground truth the frozen matching rule needs -------------------
    accounts = set()
    for txns in frames:
        accounts |= set(txns["src"]) | set(txns["dst"])
    touched_campaigns = sorted(
        {int(c) for f in frames for c in pd.unique(f["campaign_id"]) if int(c) >= 0}
    )
    truth = {
        "campaign_accounts": {
            str(cid): list(result.accounts_by_campaign.get(cid, ()))
            for cid in touched_campaigns
        },
        "campaign_split": {
            str(int(r.campaign_id)): r.split for r in result.camp.itertuples()
            if int(r.campaign_id) in touched_campaigns
        },
        "deadlines": {
            str(int(r.campaign_id)): str(r.deadline)
            for r in result.camp.itertuples()
            if int(r.campaign_id) in touched_campaigns
        },
        "campaign_end": {
            str(int(r.campaign_id)): str(r.end) for r in result.camp.itertuples()
            if int(r.campaign_id) in touched_campaigns
        },
        "unassigned_laundering_accounts": sorted(
            set(result.full_df[(result.full_df["Is Laundering"] == 1)
                               & (result.full_df["campaign_id"] == -1)]["src"])
            | set(result.full_df[(result.full_df["Is Laundering"] == 1)
                                 & (result.full_df["campaign_id"] == -1)]["dst"])
        ),
    }
    (out / "demo_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8"
    )

    # --- account -> institution, for the windows' accounts only --------
    institution_of = {a: int(assignment.institution_of(a)) for a in sorted(accounts)}
    (out / "demo_institutions.json").write_text(
        json.dumps({"k": assignment.k, "account_institution": institution_of},
                   indent=2),
        encoding="utf-8",
    )

    # --- the boosters ---------------------------------------------------
    models_out = out / "models"
    models_out.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    if args.models:
        source = Path(args.models)
        for institution in range(assignment.k):
            name = isolated.cell_name(args.target, args.seed, institution)
            src_dir = source / name
            if not src_dir.exists():
                _say(f"  MISSING booster {src_dir}")
                return 2
            shutil.copytree(src_dir, models_out / f"i{institution}",
                            dirs_exist_ok=True)
    else:
        _say("no --models given: training the 8 boosters here")
        models = isolated.train_all_isolated(
            result.full_df, assignment, lookback, progress=_say
        )
        for institution, model in sorted(models.items()):
            if not model.trained:
                _say(f"  institution {institution} untrained: nothing to save")
                continue
            model.model.save(models_out / f"i{institution}")
    for path in models_out.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size

    # --- taus for the primary points, from this cell -------------------
    taus_path = Path("artifacts/phase4_dev/phase4_isolated_taus.csv")
    taus = {}
    if taus_path.exists():
        frame = pd.read_csv(taus_path)
        frame = frame[(frame["visibility_target"] == args.target)
                      & (frame["seed"] == args.seed)]
        for row in frame.to_dict("records"):
            if row.get("operating_point") in isolated.PRIMARY_POINTS:
                taus.setdefault(str(row["operating_point"]), {})[
                    str(int(row["institution"]))
                ] = row.get("tau")

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "DEVELOPMENT_ONLY_IF_SAMPLE": (
            "If this bundle was built from the 1% sample, every number the "
            "panel computes from it is a development check, not a result."
        ),
        "config": str(args.config),
        "visibility_target": args.target,
        "seed": args.seed,
        "lookback_h": lookback,
        "split": args.split,
        "k_institutions": assignment.k,
        "campaign_selection": selection,
        "demo_campaign_id": campaign_id,
        "demo_campaign_accounts": sorted(
            result.accounts_by_campaign.get(campaign_id, ())
        ),
        "demo_campaign_institutions":
            selection["chosen"]["institutions_holding"],
        "achieved_visibility": achieved.get(campaign_id),
        "windows": window_meta,
        "primary_points": list(isolated.PRIMARY_POINTS),
        "taus": taus,
        "tau_source": str(taus_path) if taus else None,
        "models_bytes": total_bytes,
        "models_mb": round(total_bytes / 1e6, 2),
        "match_rule": {
            "min_account_fraction": constants.MATCH_MIN_ACCOUNT_FRACTION,
            "min_accounts": constants.MATCH_MIN_ACCOUNTS,
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    _say(f"\nwrote demo bundle to {out}")
    _say(f"  windows parquet: {(out / 'demo_windows.parquet').stat().st_size / 1e6:.2f} MB")
    _say(f"  boosters: {manifest['models_mb']} MB across "
         f"{assignment.k} institution(s)")
    _say(f"  demo campaign {campaign_id}, "
         f"{len(manifest['demo_campaign_accounts'])} accounts, held by "
         f"institutions {manifest['demo_campaign_institutions']}")
    if not taus:
        _say("  NOTE: no taus found - run scripts/run_phase4_chain.py first, "
             "or the panel will have to be given a tau explicitly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
