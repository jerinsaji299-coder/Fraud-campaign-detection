"""Train/val/test/stress splits, hub/fragmentable/unfragmentable groups,
and the shares_train_account flag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants


def _split_of(start: pd.Timestamp) -> str | None:
    if start < constants.TRAIN_END:
        return "train"
    if start < constants.VAL_END:
        return "val"
    if start < constants.TEST_END:
        return "test"
    if start < constants.CUTOFF:
        return "stress"
    return None


def assign_split(camp: pd.DataFrame) -> pd.DataFrame:
    """Add `split` (train/val/test/stress/None) by campaign start date.
    Campaigns starting on/after CUTOFF fall outside all four splits and
    get split = None. Does not mutate camp."""
    camp = camp.copy()
    # Built via a plain list, not .apply(), and forced to dtype=object:
    # pandas 3.x infers a "string" dtype for .apply() results containing
    # strings, which silently turns None into <NA> before .astype(object)
    # ever runs.
    camp["split"] = pd.Series(
        [_split_of(s) for s in camp["start"]], index=camp.index, dtype=object
    )
    return camp


def assign_group(camp: pd.DataFrame) -> pd.DataFrame:
    """Add `group` = hub | fragmentable, based on base_type. Does not
    mutate camp."""
    camp = camp.copy()
    camp["group"] = np.where(camp["base_type"].isin(constants.HUB_TYPES), "hub", "fragmentable")
    return camp


def reassignable_mask(camp: pd.DataFrame) -> pd.Series:
    """The campaigns whose accounts get reassigned for a (seed, target):
    eval_ok, with a non-hub base_type.

    This is deliberately a topology test on base_type rather than a test
    on `group`, because a campaign relabelled "unfragmentable" keeps
    receiving the same treatment — that relabel is an analysis grouping
    only (see README decision log, 2026-10-06).
    """
    return camp["eval_ok"] & ~camp["base_type"].isin(constants.HUB_TYPES)


def mark_unfragmentable(camp: pd.DataFrame, floor_achieved: dict[int, float]) -> pd.DataFrame:
    """Relabel fragmentable campaigns whose floor visibility (achieved at
    UNFRAGMENTABLE_SEED, UNFRAGMENTABLE_TARGET) is >= UNFRAGMENTABLE_THRESHOLD
    as "unfragmentable". `floor_achieved` maps campaign_id -> achieved
    visibility from that specific run. Does not mutate camp."""
    camp = camp.copy()
    is_unfrag = camp["campaign_id"].map(floor_achieved).fillna(0) >= constants.UNFRAGMENTABLE_THRESHOLD
    camp.loc[(camp["group"] == "fragmentable") & is_unfrag, "group"] = "unfragmentable"
    return camp


def control_group_mask(camp: pd.DataFrame) -> pd.Series:
    """hub + unfragmentable."""
    return camp["group"].isin(["hub", "unfragmentable"])


def mark_shared_accounts(camp: pd.DataFrame, pattern_df: pd.DataFrame) -> pd.DataFrame:
    """Add `shares_train_account`: True for test-split campaigns that
    share any account with any train-split campaign. False elsewhere
    (including for non-test campaigns). Does not mutate camp."""
    camp = camp.copy()

    acc_camp = pd.concat(
        [
            pattern_df[["campaign_id", "src"]].rename(columns={"src": "acc"}),
            pattern_df[["campaign_id", "dst"]].rename(columns={"dst": "acc"}),
        ]
    ).drop_duplicates()
    acc_camp = acc_camp.merge(camp[["campaign_id", "split"]], on="campaign_id", how="left")

    train_accounts = set(acc_camp.loc[acc_camp["split"] == "train", "acc"])
    test_campaign_ids = camp.loc[camp["split"] == "test", "campaign_id"]

    tainted = set(
        acc_camp[(acc_camp["campaign_id"].isin(test_campaign_ids)) & (acc_camp["acc"].isin(train_accounts))][
            "campaign_id"
        ]
    )

    camp["shares_train_account"] = camp["campaign_id"].isin(tainted)
    return camp
