"""Evidence for the stage 4.1 proposal: a per-institution STRUCTURAL ceiling.

For each `(target, seed)` and each institution, ask of every `eval_ok`
campaign: does that institution's view contain a connected component of at
least `MIN_CLUSTER_ACCOUNTS` of the campaign's accounts, over the campaign's
own laundering edges?

This is **not** the oracle ceiling. It ignores the 6-hour detection grid and
the deadline, so a campaign counted here may still be unreachable in time.
It is therefore an **upper bound** on an institution's oracle ceiling, and it
exists to answer one design question before 4.1 is built: does an
institution's ceiling collapse as its visibility shrinks, and does the union
over institutions collapse with it?

Measured answer on the 1% sample (2026-11-14): the per-institution mean rises
as visibility falls (0.255 at target 1.0, 0.546 at 0.25), and the union sits
at 0.884 at **every** target - identical to the centralized structural value,
228 of 258 campaigns. That identity is a theorem, not a coincidence: a
three-account connected component contains a vertex with at least two
distinct neighbours, and that vertex's institution is an endpoint of both of
its edges, so it sees the whole component whatever the assignment does with
the other accounts. The script prints the count of campaigns holding such a
vertex alongside the union, so the two can be seen to agree.

Nothing here trains, scores or selects anything, and it writes no artifact
that any later stage reads.

    cd backend
    python scripts/run_structural_ceiling.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
for path in (BACKEND / "src", BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

from fraudcamp import constants, pipeline  # noqa: E402
from fraudcamp.federation import views as fv  # noqa: E402


def max_component(edges: list[tuple[str, str]]) -> int:
    """Size of the largest connected component over `edges`, by union-find.

    The same notion of connectedness the frozen cluster extraction uses:
    direction is ignored, and an account appears once however many of its
    transactions are kept.
    """
    parent: dict[str, str] = {}

    def find(account: str) -> str:
        parent.setdefault(account, account)
        while parent[account] != account:
            parent[account] = parent[parent[account]]
            account = parent[account]
        return account

    for src, dst in edges:
        root_src, root_dst = find(src), find(dst)
        if root_src != root_dst:
            parent[root_src] = root_dst
    if not parent:
        return 0
    return max(Counter(find(account) for account in parent).values())


def has_degree_two_vertex(edges: list[tuple[str, str]]) -> bool:
    """Does some account have two or more distinct campaign counterparties?

    If one does, its institution is an endpoint of both of those edges and
    therefore sees a three-account component on its own, whatever the
    assignment did with the other accounts. This is why the union over
    institutions cannot fall below the centralized structural value.
    """
    neighbours: dict[str, set[str]] = {}
    for src, dst in edges:
        neighbours.setdefault(src, set()).add(dst)
        neighbours.setdefault(dst, set()).add(src)
    return any(len(found - {account}) >= 2 for account, found in neighbours.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--targets", type=float, nargs="+", default=constants.VISIBILITY_TARGETS)
    ap.add_argument("--out", default="artifacts/federation_dev")
    args = ap.parse_args()

    t0 = time.time()
    config = BACKEND / args.config if not Path(args.config).is_absolute() else Path(args.config)
    print(f"building the pipeline from {config} ...", flush=True)
    result = pipeline.build(config)

    camp = result.camp
    eval_ok = sorted(int(c) for c in camp.loc[camp["eval_ok"], "campaign_id"])
    k = constants.K_INSTITUTIONS
    minimum = constants.MATCH_MIN_ACCOUNTS

    central = sum(
        1 for cid in eval_ok
        if max_component(result.edges_by_campaign.get(cid, [])) >= minimum
    )
    degree_two = sum(
        1 for cid in eval_ok
        if has_degree_two_vertex(result.edges_by_campaign.get(cid, []))
    )
    n = len(eval_ok)
    print(
        f"  {n} eval_ok campaigns, K = {k}, component threshold {minimum} accounts, "
        f"{time.time() - t0:.1f}s"
    )
    print(f"\ncentralized structural reach: {central}/{n} = {central / n:.3f}")
    print(
        f"campaigns with an account holding 2+ distinct campaign counterparties: "
        f"{degree_two}/{n} = {degree_two / n:.3f}"
        + ("  <-- equal, as the theorem requires" if degree_two == central else
           "  <-- NOT equal: check the component threshold")
    )

    rows = []
    for target in args.targets:
        for seed in args.seeds:
            assignment, _ = fv.assignment_from_pipeline(result, target=target, seed=seed)
            institution_of = assignment.institution_of
            reached = {i: 0 for i in range(k)}
            union = 0
            institutions_per_campaign = []
            for cid in eval_ok:
                edges = result.edges_by_campaign.get(cid, [])
                count = 0
                for i in range(k):
                    visible = [
                        (src, dst) for src, dst in edges
                        if institution_of(src) == i or institution_of(dst) == i
                    ]
                    if max_component(visible) >= minimum:
                        reached[i] += 1
                        count += 1
                institutions_per_campaign.append(count)
                if count:
                    union += 1
            fractions = sorted(reached[i] / n for i in range(k))
            rows.append({
                "target": target,
                "seed": seed,
                "per_institution_mean": sum(fractions) / k,
                "weakest_institution": fractions[0],
                "strongest_institution": fractions[-1],
                "union": union / n,
                "institutions_per_campaign": sum(institutions_per_campaign) / n,
            })
            print(
                f"  target {target:<5} seed {seed}: per-institution mean "
                f"{rows[-1]['per_institution_mean']:.3f} "
                f"(weakest {fractions[0]:.3f}, strongest {fractions[-1]:.3f}), "
                f"union {rows[-1]['union']:.3f}, "
                f"{rows[-1]['institutions_per_campaign']:.2f} institutions reach a campaign",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    print("\n=== pooled over seeds ===")
    print(
        frame.groupby("target").agg(
            per_institution_mean=("per_institution_mean", "mean"),
            weakest_institution=("weakest_institution", "min"),
            union=("union", "mean"),
            institutions_per_campaign=("institutions_per_campaign", "mean"),
        ).to_string(float_format=lambda v: f"{v:.3f}")
    )

    out_dir = BACKEND / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "structural_ceiling.csv"
    frame.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
