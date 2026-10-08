#!/usr/bin/env python
"""Run this ON KAGGLE (needs the full dataset mounted). Writes a sample of
HI-Small_Trans.csv to /kaggle/working/: every transaction listed in
HI-Small_Patterns.txt, plus a random 1% of all other transactions (seed
42). Copies the pattern file unchanged. Download the two output files
from /kaggle/working/ into backend/data/sample/ afterward.

Usage (on Kaggle):
    python make_sample.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudcamp import campaigns, load_data  # noqa: E402

BASE = Path("/kaggle/input/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
OUT = Path("/kaggle/working")
SEED = 42
OTHER_FRACTION = 0.01


def main() -> None:
    trans_csv = BASE / "HI-Small_Trans.csv"
    patterns_txt = BASE / "HI-Small_Patterns.txt"

    print(f"Loading {trans_csv} ...")
    full_df = load_data.load_transactions(trans_csv)
    print("Total rows:", len(full_df))

    attempts = campaigns.parse_patterns(patterns_txt)
    pattern_df = campaigns.build_pattern_df(attempts)
    full_df = campaigns.match_patterns(pattern_df, full_df)

    in_pattern = full_df["campaign_id"] != -1
    pattern_rows = full_df[in_pattern]
    other_rows = full_df[~in_pattern]

    rng = np.random.RandomState(SEED)
    sample_other = other_rows.sample(frac=OTHER_FRACTION, random_state=rng)

    sample = pd.concat([pattern_rows, sample_other]).sort_index()
    sample = sample.drop(columns=["ts", "src", "dst", "campaign_id"])

    out_trans = OUT / "HI-Small_Trans.csv"
    sample.to_csv(out_trans, index=False)
    print(f"Wrote {len(sample)} rows to {out_trans}")

    out_patterns = OUT / "HI-Small_Patterns.txt"
    shutil.copyfile(patterns_txt, out_patterns)
    print(f"Copied pattern file unchanged to {out_patterns}")


if __name__ == "__main__":
    main()
