"""/campaigns/{id}/visibility and /visibility/distribution."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas import (
    AccountAssignment,
    CampaignVisibilityPoint,
    CampaignVisibilityResponse,
    InstitutionView,
    TargetDistribution,
    VisibilityDistributionResponse,
)
from app.services.artifacts import ArtifactStore, get_store
from fraudcamp import constants

router = APIRouter()


def _validate(seed: int, target: float | None = None) -> None:
    if seed not in constants.VISIBILITY_SEEDS:
        raise HTTPException(
            status_code=422,
            detail=f"seed must be one of {constants.VISIBILITY_SEEDS}, got {seed}.",
        )
    if target is not None and target not in constants.VISIBILITY_TARGETS:
        raise HTTPException(
            status_code=422,
            detail=f"target must be one of {constants.VISIBILITY_TARGETS}, got {target}.",
        )


@router.get(
    "/campaigns/{campaign_id}/visibility",
    response_model=CampaignVisibilityResponse,
    tags=["visibility"],
)
def campaign_visibility(
    campaign_id: int,
    seed: int = Query(0),
    target: float = Query(1.0),
    store: ArtifactStore = Depends(get_store),
) -> CampaignVisibilityResponse:
    """Per-account institutions for one campaign under one (seed, target),
    plus which transactions each institution can see.

    Transactions are identified by `txn_index`, the 0-based position in the
    campaign's time-ordered transaction list as returned by
    `/api/campaigns/{id}`.
    """
    _validate(seed, target)
    store.require_campaign(campaign_id)

    vis = store.require_visibility(seed)
    rows = vis[(vis["campaign_id"] == campaign_id) & (vis["target"] == target)]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No visibility record for campaign {campaign_id} at seed {seed}, target {target}.",
        )

    assignment = {str(r.account): int(r.institution) for r in rows.itertuples()}
    reassigned_accounts = {str(r.account) for r in rows.itertuples() if bool(r.account_reassigned)}

    txns = store.require_campaign_transactions()
    campaign_txns = txns[txns["campaign_id"] == campaign_id].sort_values("txn_index")

    institution_views = []
    for institution in range(constants.K_INSTITUTIONS):
        visible = [
            int(r.txn_index)
            for r in campaign_txns.itertuples()
            if assignment.get(str(r.src)) == institution or assignment.get(str(r.dst)) == institution
        ]
        institution_views.append(
            InstitutionView(
                institution=institution, visible_txn_indices=visible, n_visible=len(visible)
            )
        )

    first = rows.iloc[0]
    return CampaignVisibilityResponse(
        campaign_id=campaign_id,
        seed=seed,
        target=target,
        achieved_visibility=float(first["achieved_visibility"]),
        bin=str(first["bin"]),
        campaign_reassigned=bool(first["campaign_reassigned"]),
        n_transactions=len(campaign_txns),
        accounts=[
            AccountAssignment(
                account=account,
                institution=institution,
                reassigned=account in reassigned_accounts,
            )
            for account, institution in sorted(assignment.items())
        ],
        institutions=institution_views,
    )


@router.get(
    "/visibility/distribution",
    response_model=VisibilityDistributionResponse,
    tags=["visibility"],
)
def visibility_distribution(
    seed: int = Query(0), store: ArtifactStore = Depends(get_store)
) -> VisibilityDistributionResponse:
    """Achieved visibility per target for one seed, plus counts per bin
    and per topology group."""
    _validate(seed)
    vis = store.require_visibility(seed)
    campaigns = store.require_campaigns()

    group_of = dict(zip(campaigns["campaign_id"], campaigns["group"]))

    # one row per (target, campaign) — the per-account rows repeat the
    # campaign-level achieved visibility and bin
    per_campaign = vis.drop_duplicates(subset=["target", "campaign_id"])

    per_target = []
    for target in constants.VISIBILITY_TARGETS:
        rows = per_campaign[per_campaign["target"] == target]
        points = []
        bin_counts = {b: 0 for b in constants.VISIBILITY_BIN_ORDER}
        bin_counts_by_group: dict[str, dict[str, int]] = {}

        for r in rows.itertuples():
            group = str(group_of.get(int(r.campaign_id), "unknown"))
            bin_label = str(r.bin)
            points.append(
                CampaignVisibilityPoint(
                    campaign_id=int(r.campaign_id),
                    achieved_visibility=float(r.achieved_visibility),
                    bin=bin_label,
                    group=group,
                    reassigned=bool(r.campaign_reassigned),
                )
            )
            bin_counts[bin_label] += 1
            bin_counts_by_group.setdefault(
                group, {b: 0 for b in constants.VISIBILITY_BIN_ORDER}
            )[bin_label] += 1

        per_target.append(
            TargetDistribution(
                target=target,
                campaigns=sorted(points, key=lambda p: p.campaign_id),
                bin_counts=bin_counts,
                bin_counts_by_group=bin_counts_by_group,
            )
        )

    return VisibilityDistributionResponse(
        seed=seed,
        targets=constants.VISIBILITY_TARGETS,
        bins=constants.VISIBILITY_BIN_ORDER,
        per_target=per_target,
        unfragmentable_campaigns=int((campaigns["group"] == "unfragmentable").sum()),
    )
