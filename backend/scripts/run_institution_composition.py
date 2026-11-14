"""Why institution 0 is not like the other seven.

Stage 4.1 found institution 0 holding the most training positives in **12 of
12** full-data cells. That is not noise and not an artefact of the visibility
search: :func:`fraudcamp.institutions.assign_institutions` walks banks into
institutions largest-volume-first, always choosing ``load.index(min(load))``
-- the **lowest index** on a tie -- so with every load at 0 the dataset's
single largest bank goes to institution 0 every time. The assignment is also
seed-independent, computed from pre-cutoff volume alone.

This script is the reproducible evidence for the two sample-measured tables in
the README's "The institutions are not exchangeable" section:

1. **composition** -- banks held, load share, and the largest bank's share of
   each institution's own load. Volume is balanced by construction; structure
   is not, and institution 0 is one hub bank plus small ones.
2. **reach** -- campaign accounts each institution *owns* against campaign
   accounts it can *see*. Ownership is not biased toward institution 0;
   visible-per-owned is. Visibility is an endpoint rule, so connectivity --
   not volume, and not ownership -- sets how much of the fraud a view holds.

Sample-only, like every local script here: the full-data counterpart is
``docs/reference/train_isolated.csv``, which is what the README quotes for
the per-institution positive counts. Nothing here is a detection metric, so
the 1% background does not distort it -- bank volume ordering and endpoint
connectivity are structural.

    cd backend
    python scripts/run_institution_composition.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

from fraudcamp import constants, institutions, pipeline  # noqa: E402
from fraudcamp.federation import views  # noqa: E402


def composition(result) -> pd.DataFrame:
    """Banks held and load concentration per institution."""
    volume = institutions.bank_volume(
        result.pre_cutoff_df,
        all_banks=sorted(
            set(result.full_df["From Bank"]) | set(result.full_df["To Bank"])
        ),
    )
    total = int(volume.sum())
    held: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for bank, n in volume.items():
        held[result.bank2inst[int(bank)]].append((int(bank), int(n)))

    rows = []
    for inst in range(constants.K_INSTITUTIONS):
        banks = sorted(held[inst], key=lambda kv: -kv[1])
        load = sum(n for _, n in banks)
        top_bank, top_volume = banks[0]
        rows.append({
            "institution": inst,
            "n_banks": len(banks),
            "load": load,
            "load_share": load / total,
            "largest_bank": top_bank,
            "largest_bank_volume": top_volume,
            "largest_bank_share_of_total": top_volume / total,
            # the number that matters: equal volume, unequal concentration
            "largest_bank_share_of_own_load": top_volume / load,
        })
    return pd.DataFrame(rows)


def reach(result, targets, seeds) -> pd.DataFrame:
    """Campaign accounts owned against campaign accounts visible.

    ``owned`` is ownership under the final assignment; ``visible`` is every
    campaign account appearing on either end of a transaction the institution
    can see. The ratio is the hub effect, and it is the quantity that explains
    a 44% positive excess on 2% more rows.
    """
    eval_ok = set(
        result.camp.loc[result.camp["eval_ok"].astype(bool), "campaign_id"]
    )
    campaign_accounts = set()
    for campaign_id, accounts in result.accounts_by_campaign.items():
        if campaign_id in eval_ok:
            campaign_accounts |= set(accounts)

    df = result.full_df
    ordered = sorted(campaign_accounts)
    rows = []
    for target in targets:
        for seed in seeds:
            assignment, _ = views.assignment_from_pipeline(
                result, target=target, seed=seed
            )
            src, dst = views.endpoint_institutions(df, assignment)
            owner = assignment.account_institutions(ordered)
            for inst in range(constants.K_INSTITUTIONS):
                mask = (src == inst) | (dst == inst)
                visible = df[mask]
                seen = (
                    set(visible["src"]) | set(visible["dst"])
                ) & campaign_accounts
                owned = sum(1 for a in campaign_accounts if owner[a] == inst)
                rows.append({
                    "visibility_target": target,
                    "seed": seed,
                    "institution": inst,
                    "n_campaign_accounts": len(campaign_accounts),
                    "owned": owned,
                    "visible": len(seen),
                    "visible_per_owned": (len(seen) / owned) if owned else None,
                    "visible_rows": int(mask.sum()),
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--targets", type=float, nargs="+", default=[1.0, 0.25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="artifacts/federation_dev")
    args = ap.parse_args()

    started = time.time()
    config = (
        BACKEND / args.config
        if not Path(args.config).is_absolute()
        else Path(args.config)
    )
    print(f"building the pipeline from {config} ...", flush=True)
    result = pipeline.build(config)

    out_dir = BACKEND / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    comp = composition(result)
    comp.to_csv(out_dir / "institution_composition.csv", index=False)
    print("\n-- composition: equal volume, unequal concentration --")
    print(comp[[
        "institution", "n_banks", "load", "load_share", "largest_bank",
        "largest_bank_share_of_own_load",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    hub = comp.loc[comp.largest_bank_share_of_own_load.idxmax()]
    print(
        f"\n  institution {int(hub.institution)} holds bank "
        f"{int(hub.largest_bank)}: {hub.largest_bank_share_of_total:.2%} of all "
        f"pre-cutoff volume, {hub.largest_bank_share_of_own_load:.1%} of its "
        f"own load, against "
        f"{comp.largest_bank_share_of_own_load.drop(hub.name).max():.1%} "
        "for the next most concentrated institution."
    )
    print(
        f"  load spread: {comp.load.min():,} to {comp.load.max():,} of "
        f"{comp.load.sum():,} endpoint-slots "
        f"({comp.load.max() - comp.load.min()} apart)"
    )

    rch = reach(result, args.targets, args.seeds)
    rch.to_csv(out_dir / "institution_reach.csv", index=False)
    for (target, seed), block in rch.groupby(["visibility_target", "seed"]):
        print(f"\n-- reach: target {target}, seed {seed} --")
        print(block[[
            "institution", "owned", "visible", "visible_per_owned",
            "visible_rows",
        ]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        zero = block[block.institution == 0].iloc[0]
        others = block[block.institution != 0]
        print(
            f"     institution 0: owns {int(zero.owned)} "
            f"(rank {int((block.owned < zero.owned).sum()) + 1} of 8 by ownership), "
            f"sees {zero.visible_per_owned:.2f} per owned against "
            f"{others.visible_per_owned.min():.2f}-"
            f"{others.visible_per_owned.max():.2f} elsewhere"
        )

    print(
        f"\nwrote institution_composition.csv and institution_reach.csv to "
        f"{out_dir}"
    )
    print(f"total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
