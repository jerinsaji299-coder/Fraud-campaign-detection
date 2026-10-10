#!/usr/bin/env python
"""Build the 1% development sample from the full dataset.

Keeps every transaction listed in HI-Small_Patterns.txt, plus a seeded
random 1% of all other transactions, and copies the pattern file unchanged.
Run it where the full data is — i.e. on Kaggle — then download both output
files into backend/data/sample/.

Usable either directly or through `kaggle_runner.py make-sample`, which
calls `build_sample` below so there is only one implementation.

Usage (on Kaggle):
    python scripts/make_sample.py
    python scripts/make_sample.py --config configs/kaggle.yaml --out /kaggle/working
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND / "src") not in sys.path:
    sys.path.insert(0, str(BACKEND / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudcamp import campaigns, load_data  # noqa: E402

SEED = 42
OTHER_FRACTION = 0.01
DEFAULT_OUT = Path("/kaggle/working")


def build_sample(
    trans_csv: str | Path,
    patterns_txt: str | Path,
    out_dir: str | Path,
    seed: int = SEED,
    other_fraction: float = OTHER_FRACTION,
) -> list[Path]:
    """Write the sampled CSV and the unchanged pattern file to `out_dir`.

    Returns the paths written. Every pattern transaction is kept, so the
    campaign ground truth stays complete and the sample remains usable for
    campaign-level work; only the background is thinned.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {trans_csv} ...", flush=True)
    full_df = load_data.load_transactions(trans_csv)
    print(f"  {len(full_df):,} rows")

    attempts = campaigns.parse_patterns(patterns_txt)
    pattern_df = campaigns.build_pattern_df(attempts)
    full_df = campaigns.match_patterns(pattern_df, full_df)

    in_pattern = full_df["campaign_id"] != -1
    pattern_rows = full_df[in_pattern]
    other_rows = full_df[~in_pattern]

    rng = np.random.RandomState(seed)
    sampled_other = other_rows.sample(frac=other_fraction, random_state=rng)

    sample = pd.concat([pattern_rows, sampled_other]).sort_index()
    sample = sample.drop(columns=["ts", "src", "dst", "campaign_id"])

    out_trans = out_dir / "HI-Small_Trans.csv"
    sample.to_csv(out_trans, index=False)
    print(
        f"  wrote {len(sample):,} rows "
        f"({len(pattern_rows):,} pattern + {len(sampled_other):,} sampled) "
        f"to {out_trans}"
    )

    out_patterns = out_dir / "HI-Small_Patterns.txt"
    shutil.copyfile(patterns_txt, out_patterns)
    print(f"  copied the pattern file unchanged to {out_patterns}")
    return [out_trans, out_patterns]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BACKEND / "configs" / "kaggle.yaml"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_data.load_config(config_path)
    build_sample(
        load_data.resolve_path(config_path, config["data"]["trans_csv"]),
        load_data.resolve_path(config_path, config["data"]["patterns_txt"]),
        args.out,
        seed=args.seed,
    )
    print("\nDownload both files into backend/data/sample/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
