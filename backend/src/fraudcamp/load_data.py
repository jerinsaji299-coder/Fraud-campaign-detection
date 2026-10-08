"""Load the raw transaction CSV and compute global account IDs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import constants


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config_path: str | Path, relative: str) -> Path:
    """Resolve a config-relative data path against the config file's directory."""
    base = Path(config_path).resolve().parent.parent  # configs/ -> backend/
    p = Path(relative)
    return p if p.is_absolute() else base / p


def load_transactions(trans_csv: str | Path) -> pd.DataFrame:
    """Load HI-Small_Trans.csv with correct dtypes and derived account IDs.

    Account columns are loaded as strings (leading zeros are meaningful).
    Bank columns are ints. Adds `ts` (parsed timestamp), `src`, `dst`
    (global account IDs of the form "{bank_int}_{account_str}").
    """
    df = pd.read_csv(
        trans_csv,
        dtype={"Account": str, "Account.1": str},
    )
    df["From Bank"] = df["From Bank"].astype(int)
    df["To Bank"] = df["To Bank"].astype(int)
    df["ts"] = pd.to_datetime(df["Timestamp"], format=constants.TIMESTAMP_FORMAT)
    df["src"] = df["From Bank"].astype(str) + "_" + df["Account"]
    df["dst"] = df["To Bank"].astype(str) + "_" + df["Account.1"]
    return df


def apply_cutoff(df: pd.DataFrame, cutoff=constants.CUTOFF) -> pd.DataFrame:
    """Return only rows with ts < cutoff (model input). Does not mutate df."""
    return df[df["ts"] < cutoff].copy()


def account_bank(account_id: str) -> int:
    """Extract the bank int from a global account ID "{bank_int}_{account_str}"."""
    return int(account_id.split("_", 1)[0])
