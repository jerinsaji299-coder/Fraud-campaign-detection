"""Parse HI-Small_Patterns.txt, match it against the transaction table, and
build the campaign ground-truth table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import constants


def parse_patterns(patterns_txt: str | Path) -> list[dict]:
    """Parse BEGIN/END LAUNDERING ATTEMPT blocks into a list of
    {"type": str, "rows": list[list[str]]} in order of appearance."""
    attempts: list[dict] = []
    current: dict | None = None
    with open(patterns_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                ptype = line.split("-", 1)[1].strip() if "-" in line else "UNKNOWN"
                current = {"type": ptype, "rows": []}
            elif line.startswith("END LAUNDERING ATTEMPT"):
                assert current is not None, "END without matching BEGIN"
                attempts.append(current)
                current = None
            else:
                assert current is not None, "transaction row outside a BEGIN/END block"
                current["rows"].append(line.split(","))
    return attempts


def build_pattern_df(attempts: list[dict]) -> pd.DataFrame:
    """Flatten parsed attempts into a DataFrame with one row per pattern
    transaction, campaign_id = order of appearance (0-indexed)."""
    rows = []
    for cid, a in enumerate(attempts):
        for r in a["rows"]:
            # columns: Timestamp, From Bank, Account, To Bank, Account.1,
            # Amount Received, Receiving Currency, Amount Paid, Payment
            # Currency, Payment Format, Is Laundering
            rows.append(
                [cid, a["type"], r[0], r[1], r[2], r[3], r[4], r[7]]
            )
    pat = pd.DataFrame(
        rows,
        columns=[
            "campaign_id",
            "type",
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Paid",
        ],
    )
    pat["From Bank"] = pat["From Bank"].astype(int)
    pat["To Bank"] = pat["To Bank"].astype(int)
    pat["Amount Paid"] = pat["Amount Paid"].astype(float)
    pat["base_type"] = pat["type"].str.split(":").str[0].str.strip()
    pat["ts"] = pd.to_datetime(pat["Timestamp"], format=constants.TIMESTAMP_FORMAT)
    pat["src"] = pat["From Bank"].astype(str) + "_" + pat["Account"]
    pat["dst"] = pat["To Bank"].astype(str) + "_" + pat["Account.1"]
    return pat


def match_patterns(pattern_df: pd.DataFrame, full_trans_df: pd.DataFrame) -> pd.DataFrame:
    """Match pattern transactions against the full (uncut) transaction
    table on [Timestamp, From Bank, Account, To Bank, Account.1, Amount
    Paid]. Asserts all pattern rows match. Returns full_trans_df with a
    new `campaign_id` column (-1 for non-campaign rows).

    Does not mutate full_trans_df.
    """
    keys = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid"]

    full = full_trans_df.copy()
    full["_row_id"] = np.arange(len(full))

    merged = pattern_df.merge(
        full[keys + ["_row_id"]],
        on=keys,
        how="left",
        indicator=True,
    )
    unmatched = (merged["_merge"] == "left_only").sum()
    assert unmatched == 0, f"{unmatched} pattern transactions did not match the transaction table"

    # a pattern transaction could in principle match more than one row if
    # the key combination is not unique; take the first match per pattern
    # row and assert each row_id is used for only one campaign.
    merged = merged.drop_duplicates(subset=["campaign_id"] + keys, keep="first")

    full["campaign_id"] = -1
    full.loc[merged["_row_id"], "campaign_id"] = merged["campaign_id"].values
    full = full.drop(columns=["_row_id"])
    return full


def build_campaign_table(pattern_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pattern_df into one row per campaign_id: base_type, n_txn,
    n_accounts, n_banks, start, end, duration_h."""

    def _n_accounts(g: pd.DataFrame) -> int:
        return len(set(g["src"]) | set(g["dst"]))

    def _n_banks(g: pd.DataFrame) -> int:
        return len(set(g["From Bank"]) | set(g["To Bank"]))

    camp = pattern_df.groupby("campaign_id").agg(
        base_type=("base_type", "first"),
        n_txn=("ts", "size"),
        start=("ts", "min"),
        end=("ts", "max"),
    )
    camp["n_accounts"] = pattern_df.groupby("campaign_id").apply(_n_accounts, include_groups=False)
    camp["n_banks"] = pattern_df.groupby("campaign_id").apply(_n_banks, include_groups=False)
    camp["duration_h"] = (camp["end"] - camp["start"]).dt.total_seconds() / 3600
    camp = camp.reset_index()
    return camp[["campaign_id", "base_type", "n_txn", "n_accounts", "n_banks", "start", "end", "duration_h"]]


def add_eval_fields(camp: pd.DataFrame, cutoff=constants.CUTOFF) -> pd.DataFrame:
    """Add crosses_cutoff, deadline, eval_ok to the campaign table.
    Does not mutate camp."""
    camp = camp.copy()
    camp["crosses_cutoff"] = camp["end"] >= cutoff
    camp["deadline"] = camp["end"].where(~camp["crosses_cutoff"], cutoff)
    camp["eval_ok"] = (camp["n_txn"] >= constants.EVAL_OK_MIN_TXN) & (
        camp["n_accounts"] >= constants.EVAL_OK_MIN_ACCOUNTS
    )
    return camp


def campaign_accounts(pattern_df: pd.DataFrame, campaign_id: int) -> tuple[str, ...]:
    """A campaign's accounts as a sorted tuple. Sorted rather than a set
    because these feed the seeded visibility search, where iteration order
    must not vary between processes."""
    g = pattern_df[pattern_df["campaign_id"] == campaign_id]
    return tuple(sorted(set(g["src"]) | set(g["dst"])))


def campaign_edges(pattern_df: pd.DataFrame, campaign_id: int) -> list[tuple[str, str]]:
    g = pattern_df[pattern_df["campaign_id"] == campaign_id]
    return list(zip(g["src"], g["dst"]))
