"""/campaigns and /campaigns/{id}."""

from __future__ import annotations

import math
from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.schemas import (
    Campaign,
    CampaignAccount,
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignTransaction,
)
from app.services.artifacts import ArtifactStore, get_store, to_records

router = APIRouter()

SORTABLE = Literal[
    "campaign_id", "base_type", "n_txn", "n_accounts", "n_banks", "start", "end", "duration_h"
]


@router.get("/campaigns", response_model=CampaignListResponse, tags=["campaigns"])
def list_campaigns(
    store: ArtifactStore = Depends(get_store),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=500),
    base_type: str | None = None,
    split: str | None = None,
    group: str | None = None,
    eval_ok: bool | None = None,
    crosses_cutoff: bool | None = None,
    shares_train_account: bool | None = None,
    search: str | None = Query(None, description="Campaign id, or part of an account id"),
    sort_by: SORTABLE = "campaign_id",
    sort_dir: Literal["asc", "desc"] = "asc",
) -> CampaignListResponse:
    df = store.require_campaigns()

    if base_type is not None:
        df = df[df["base_type"] == base_type]
    if split is not None:
        df = df[df["split"] == split]
    if group is not None:
        df = df[df["group"] == group]
    if eval_ok is not None:
        df = df[df["eval_ok"] == eval_ok]
    if crosses_cutoff is not None:
        df = df[df["crosses_cutoff"] == crosses_cutoff]
    if shares_train_account is not None:
        df = df[df["shares_train_account"] == shares_train_account]

    if search:
        df = df[df["campaign_id"].isin(_search_campaign_ids(store, search))]

    total = len(df)
    df = df.sort_values(sort_by, ascending=(sort_dir == "asc"), kind="stable")

    start = (page - 1) * page_size
    items = to_records(df.iloc[start : start + page_size])

    return CampaignListResponse(
        items=[Campaign(**row) for row in items],
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, math.ceil(total / page_size)),
    )


def _search_campaign_ids(store: ArtifactStore, search: str) -> set[int]:
    """Campaigns matching a search string.

    A digits-only query means the campaign id, plus any account whose
    account part (after the "bank_" prefix) starts with those digits — so
    "3" finds campaign 3 rather than every account that happens to contain
    a 3, while "800737690" still finds that account. Any other query is a
    case-insensitive substring match on the full account id, e.g. "1_A".
    """
    matched: set[int] = set()
    needle = search.strip()
    if not needle:
        return matched

    txns = store.campaign_transactions
    if needle.isdigit():
        matched.add(int(needle))
        if txns is not None:
            account_part = lambda col: txns[col].str.split("_", n=1).str[1]  # noqa: E731
            hit = account_part("src").str.startswith(needle) | account_part("dst").str.startswith(
                needle
            )
            matched.update(int(c) for c in txns.loc[hit, "campaign_id"].unique())
    elif txns is not None:
        upper = needle.upper()
        hit = txns["src"].str.upper().str.contains(upper, regex=False) | txns[
            "dst"
        ].str.upper().str.contains(upper, regex=False)
        matched.update(int(c) for c in txns.loc[hit, "campaign_id"].unique())

    return matched


@router.get(
    "/campaigns/{campaign_id}", response_model=CampaignDetailResponse, tags=["campaigns"]
)
def campaign_detail(
    campaign_id: int, store: ArtifactStore = Depends(get_store)
) -> CampaignDetailResponse:
    campaign = store.require_campaign(campaign_id)
    txns = store.require_campaign_transactions()
    rows = txns[txns["campaign_id"] == campaign_id].sort_values("txn_index")
    transactions = to_records(rows)

    natural: dict[str, int] = {}
    for row in transactions:
        natural.setdefault(row["src"], row["src_natural_institution"])
        natural.setdefault(row["dst"], row["dst_natural_institution"])

    return CampaignDetailResponse(
        campaign=Campaign(**campaign),
        transactions=[CampaignTransaction(**row) for row in transactions],
        accounts=[
            CampaignAccount(account=account, natural_institution=institution)
            for account, institution in sorted(natural.items())
        ],
    )
