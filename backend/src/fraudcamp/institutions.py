"""Volume-balanced, deterministic bank -> institution assignment."""

from __future__ import annotations

import pandas as pd

from . import constants


def bank_volume(pre_cutoff_df: pd.DataFrame, all_banks: list[int] | None = None) -> pd.Series:
    """Number of pre-cutoff transactions where each bank is sender or
    receiver, sorted by volume descending.

    `all_banks` adds any bank that has no pre-cutoff activity at all with
    a volume of 0, so that every bank in the dataset still gets an
    institution (a campaign crossing the cutoff can otherwise reference a
    bank that never appears before it). Ties are broken by bank id so the
    ordering — and therefore the assignment — is fully reproducible.
    """
    vol = pd.concat([pre_cutoff_df["From Bank"], pre_cutoff_df["To Bank"]]).value_counts()
    counts = {int(bank): int(n) for bank, n in vol.items()}
    if all_banks is not None:
        for bank in all_banks:
            counts.setdefault(int(bank), 0)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return pd.Series([n for _, n in ordered], index=[b for b, _ in ordered])


def assign_institutions(volume: pd.Series, k: int = constants.K_INSTITUTIONS) -> tuple[dict[int, int], list[int]]:
    """Greedily assign each bank (largest volume first) to the institution
    with the smallest current load. Deterministic. Returns (bank -> inst,
    loads)."""
    load = [0] * k
    bank2inst: dict[int, int] = {}
    for bank, n in volume.items():
        inst = load.index(min(load))
        bank2inst[int(bank)] = inst
        load[inst] += int(n)
    return bank2inst, load


def natural_institution(account_id: str, bank2inst: dict[int, int]) -> int:
    bank = int(account_id.split("_", 1)[0])
    return bank2inst[bank]
