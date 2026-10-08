"""Pydantic response models for every endpoint.

These mirror the artifact schemas documented in README "Artifacts", so a
change to an artifact that is not reflected here fails the API tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

# --- health -----------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    artifacts_dir: str
    artifacts: dict[str, bool]
    core_artifacts_available: bool
    results_available: bool
    #: Core artifacts the server could not find, empty when all are present.
    missing_artifacts: list[str] = []
    #: Set when the artifacts were found one directory too deep (the Windows
    #: "Extract All" trap), so the cause is obvious from /api/health alone.
    nested_artifacts_dir: str | None = None
    #: A human-readable explanation of whatever is wrong, if anything is.
    hint: str | None = None


# --- summary ----------------------------------------------------------------


class RowCounts(BaseModel):
    total: int
    pre_cutoff: int
    post_cutoff: int


class DateRange(BaseModel):
    min: str
    max: str


class BankCounts(BaseModel):
    total: int


class LaunderingCounts(BaseModel):
    rows: int
    in_campaign: int
    unassigned: int


class CampaignCounts(BaseModel):
    total: int
    eval_ok: int
    pattern_transactions: int
    reassignable: int


class SplitCounts(BaseModel):
    campaigns: int
    eval_ok: int
    crosses_cutoff: int
    shares_train_account: int


class CutoffInfo(BaseModel):
    timestamp: str
    rows_excluded: int
    laundering_rows_excluded: int
    laundering_share_excluded: float
    laundering_share_overall: float


class InstitutionsMeta(BaseModel):
    k: int


class VisibilityMeta(BaseModel):
    targets: list[float]
    seeds: list[int]
    bins: list[str]


class SummaryResponse(BaseModel):
    generated_at: str
    source: dict[str, str]
    rows: RowCounts
    date_range: DateRange
    banks: BankCounts
    laundering: LaunderingCounts
    campaigns: CampaignCounts
    by_base_type: dict[str, int]
    by_split: dict[str, SplitCounts]
    by_group: dict[str, int]
    by_group_eval_ok: dict[str, int]
    cutoff: CutoffInfo
    transactions_per_day: dict[str, int]
    campaigns_starting_per_day: dict[str, int]
    institutions: InstitutionsMeta
    visibility: VisibilityMeta


# --- campaigns --------------------------------------------------------------


class Campaign(BaseModel):
    campaign_id: int
    base_type: str
    n_txn: int
    n_accounts: int
    n_banks: int
    start: datetime
    end: datetime
    duration_h: float
    crosses_cutoff: bool
    deadline: datetime
    eval_ok: bool
    split: str | None
    group: str
    shares_train_account: bool
    reassignable: bool


class CampaignListResponse(BaseModel):
    items: list[Campaign]
    total: int
    page: int
    page_size: int
    pages: int


class CampaignTransaction(BaseModel):
    txn_index: int
    campaign_id: int
    timestamp: datetime
    from_bank: int
    to_bank: int
    src: str
    dst: str
    amount_paid: float
    payment_currency: str
    amount_received: float
    receiving_currency: str
    payment_format: str
    is_laundering: int
    src_natural_institution: int
    dst_natural_institution: int


class CampaignAccount(BaseModel):
    account: str
    natural_institution: int


class CampaignDetailResponse(BaseModel):
    campaign: Campaign
    transactions: list[CampaignTransaction]
    accounts: list[CampaignAccount]


# --- visibility -------------------------------------------------------------


class AccountAssignment(BaseModel):
    account: str
    institution: int
    reassigned: bool


class InstitutionView(BaseModel):
    institution: int
    visible_txn_indices: list[int]
    n_visible: int


class CampaignVisibilityResponse(BaseModel):
    campaign_id: int
    seed: int
    target: float
    achieved_visibility: float
    bin: str
    campaign_reassigned: bool
    n_transactions: int
    accounts: list[AccountAssignment]
    institutions: list[InstitutionView]


class CampaignVisibilityPoint(BaseModel):
    campaign_id: int
    achieved_visibility: float
    bin: str
    group: str
    reassigned: bool


class TargetDistribution(BaseModel):
    target: float
    campaigns: list[CampaignVisibilityPoint]
    bin_counts: dict[str, int]
    bin_counts_by_group: dict[str, dict[str, int]]


class VisibilityDistributionResponse(BaseModel):
    seed: int
    targets: list[float]
    bins: list[str]
    per_target: list[TargetDistribution]
    unfragmentable_campaigns: int


# --- methodology ------------------------------------------------------------


class MethodologyEntry(BaseModel):
    key: str
    title: str
    definition: str
    value: dict[str, Any]
    reason: str


class MethodologyResponse(BaseModel):
    entries: list[MethodologyEntry]
    n_entries: int
