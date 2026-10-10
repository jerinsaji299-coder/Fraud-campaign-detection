#!/usr/bin/env python
"""Generate a seeded synthetic dataset for local development.

**This is not the research sample.** The real data is the Kaggle HI-Small
file, and the 1% sample comes from `make_sample.py`. This generator exists
only so the evaluation code can be exercised locally, at a size where
recall, precision and lead time mean something, without the full dataset.
Any number produced from it is a code check, never a result.

It writes HI-Small-shaped files, so everything downstream treats it like
the real thing.

Usage:
    python scripts/make_demo_data.py --out data/demo [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND / "src") not in sys.path:
    sys.path.insert(0, str(BACKEND / "src"))

CURRENCIES = ["US Dollar", "Euro", "Yuan", "Rupee", "Yen", "Swiss Franc"]
FORMATS = ["ACH", "Cheque", "Wire", "Credit Card", "Reinvestment"]
TOPOLOGIES = [
    "FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER",
    "GATHER-SCATTER", "BIPARTITE", "STACK", "RANDOM",
]

STAMP = "%Y/%m/%d %H:%M"


def _campaign_edges(rng, topology: str, accounts: list[str]) -> list[tuple[str, str]]:
    """Edges with the right shape for the topology."""
    if topology == "FAN-OUT":
        return [(accounts[0], a) for a in accounts[1:]]
    if topology == "FAN-IN":
        return [(a, accounts[0]) for a in accounts[1:]]
    if topology == "GATHER-SCATTER":
        middle = accounts[0]
        half = max(1, (len(accounts) - 1) // 2)
        return [(a, middle) for a in accounts[1 : 1 + half]] + [
            (middle, a) for a in accounts[1 + half :]
        ]
    if topology == "SCATTER-GATHER":
        first, last = accounts[0], accounts[-1]
        middles = accounts[1:-1] or [accounts[0]]
        return [(first, m) for m in middles] + [(m, last) for m in middles]
    if topology == "CYCLE":
        return [(accounts[i], accounts[(i + 1) % len(accounts)]) for i in range(len(accounts))]
    if topology == "STACK":
        return [(accounts[i], accounts[i + 1]) for i in range(len(accounts) - 1)]
    if topology == "BIPARTITE":
        half = max(1, len(accounts) // 2)
        left, right = accounts[:half], accounts[half:]
        return [(a, b) for a in left for b in right][: 3 * len(accounts)]
    # RANDOM
    return [
        (accounts[rng.integers(len(accounts))], accounts[rng.integers(len(accounts))])
        for _ in range(len(accounts) + 2)
    ]


def generate(seed: int, n_campaigns: int, n_background: int):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-09-01 00:00")

    rows: list[dict] = []
    blocks: list[tuple[str, list[dict]]] = []

    for index in range(n_campaigns):
        topology = TOPOLOGIES[index % len(TOPOLOGIES)]
        size = int(rng.integers(4, 10))
        accounts = [
            f"{int(rng.integers(1, 400)):05d}_{seed}{index:03d}{i:02d}" for i in range(size)
        ]
        edges = _campaign_edges(rng, topology, accounts)

        # spread the campaign over hours to days, starting somewhere in the
        # Sept 1-10 window so every split gets campaigns
        day = int(rng.integers(0, 10))
        hour = int(rng.integers(0, 20))
        begin = start + pd.Timedelta(days=day, hours=hour)
        span_h = float(rng.uniform(2, 70))

        block: list[dict] = []
        for position, (src, dst) in enumerate(edges):
            offset = span_h * (position / max(1, len(edges) - 1))
            ts = begin + pd.Timedelta(hours=offset, minutes=int(rng.integers(0, 59)))
            amount = round(float(rng.lognormal(9, 1.5)), 2)
            currency = CURRENCIES[int(rng.integers(len(CURRENCIES)))]
            row = {
                "Timestamp": ts.strftime(STAMP),
                "From Bank": int(src.split("_")[0]),
                "Account": src.split("_")[1],
                "To Bank": int(dst.split("_")[0]),
                "Account.1": dst.split("_")[1],
                "Amount Received": amount,
                "Receiving Currency": currency,
                "Amount Paid": amount,
                "Payment Currency": currency,
                "Payment Format": FORMATS[int(rng.integers(len(FORMATS)))],
                "Is Laundering": 1,
            }
            block.append(row)
            rows.append(row)
        blocks.append((topology, block))

    # Background traffic. Campaign accounts take part in it, which matters
    # more than it looks: with campaign accounts isolated, randomly keeping
    # half the edges leaves a small *pure* fragment of each campaign, which
    # the matching rule correctly calls a hit - so the RANDOM sanity scorer
    # scored ~0.7 recall and told us nothing. Mixing them makes random
    # clusters merge into background and dilute below the 50% threshold,
    # the way they do in the real data.
    campaign_accounts = sorted({a for _, block in blocks for a in (
        f"{row['From Bank']}_{row['Account']}" for row in block
    )} | {a for _, block in blocks for a in (
        f"{row['To Bank']}_{row['Account.1']}" for row in block
    )})
    plain = [
        f"{int(rng.integers(1, 400)):05d}_B{i:05d}" for i in range(max(2000, n_background // 20))
    ]
    pool = campaign_accounts + plain
    # campaign accounts are a small slice of the pool but must be reachable,
    # so draw them a little more often than their share would suggest
    weights = np.array(
        [3.0] * len(campaign_accounts) + [1.0] * len(plain), dtype=np.float64
    )
    weights /= weights.sum()

    picks = rng.choice(len(pool), size=(n_background, 2), p=weights)
    laundering_flags = rng.random(n_background) < 0.0008
    for index in range(n_background):
        ts = start + pd.Timedelta(
            days=int(rng.integers(0, 11)),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )
        src = pool[picks[index, 0]]
        dst = pool[picks[index, 1]]
        amount = round(float(rng.lognormal(8, 1.5)), 2)
        currency = CURRENCIES[int(rng.integers(len(CURRENCIES)))]
        rows.append(
            {
                "Timestamp": ts.strftime(STAMP),
                "From Bank": int(src.split("_")[0]),
                "Account": src.split("_")[1],
                "To Bank": int(dst.split("_")[0]),
                "Account.1": dst.split("_")[1],
                "Amount Received": amount,
                "Receiving Currency": currency,
                "Amount Paid": amount,
                "Payment Currency": currency,
                "Payment Format": FORMATS[int(rng.integers(len(FORMATS)))],
                "Is Laundering": int(laundering_flags[index]),
            }
        )

    return pd.DataFrame(rows), blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BACKEND / "data" / "demo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--campaigns", type=int, default=120)
    parser.add_argument("--background", type=int, default=300_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame, blocks = generate(args.seed, args.campaigns, args.background)
    trans = out / "HI-Small_Trans.csv"
    frame.to_csv(trans, index=False)

    lines: list[str] = []
    for topology, block in blocks:
        lines.append(f"BEGIN LAUNDERING ATTEMPT - {topology}:  synthetic")
        for row in block:
            lines.append(
                f"{row['Timestamp']},{row['From Bank']:06d},{row['Account']},"
                f"{row['To Bank']:06d},{row['Account.1']},"
                f"{row['Amount Received']:.2f},{row['Receiving Currency']},"
                f"{row['Amount Paid']:.2f},{row['Payment Currency']},"
                f"{row['Payment Format']},1"
            )
        lines.append(f"END LAUNDERING ATTEMPT - {topology}")
        lines.append("")
    patterns = out / "HI-Small_Patterns.txt"
    patterns.write_text("\n".join(lines), encoding="utf-8")

    # Absolute paths: `load_data.resolve_path` resolves relative entries
    # against the config's grandparent, which assumes a config living in
    # backend/configs/. This config does not, so leave nothing to resolve.
    config = out / "demo.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "trans_csv": str(trans.resolve()),
                    "patterns_txt": str(patterns.resolve()),
                },
                "artifacts_dir": str((out / "artifacts").resolve()),
            }
        ),
        encoding="utf-8",
    )

    print(f"SYNTHETIC DEMO DATA (seed {args.seed}) - not the research sample")
    print(f"  transactions: {len(frame):,} ({int(frame['Is Laundering'].sum()):,} laundering)")
    print(f"  campaigns:    {len(blocks)}")
    print(f"  wrote: {trans}")
    print(f"         {patterns}")
    print(f"         {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
