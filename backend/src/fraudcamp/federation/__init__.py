"""Phase 4: the federated conditions.

Stage 4.0 (done) provides the data substrate every later condition stands on:
per-institution **views** of the transaction data under a given
``(seed, target)`` visibility assignment, and the account features each
institution can compute from its own view alone.

Stage 4.1 (approved 2026-11-14) is ``isolated.py``: the ``isolated``
condition, each institution trained and scoring on its own view alone. The
remaining conditions consume the same views and are evaluated by the
unchanged Phase 3 evaluation code — ``fedavg`` is 4.2, ``fedavg_exchange``
and the gate are 4.3, and ``centralized`` is the Phase 3 model already
measured.

In ``isolated``, an institution is restricted by its **scorer** and not by
its data: the evaluation keeps the global transactions, campaigns and
deadlines, and the scorer returns 0 outside ``visible_mask`` and its model's
probability inside, with features built from ``institution_view`` alone.
Visibility restricts what can be scored, never what is true. Training is the
one place the view *is* passed as the frame, because training labels are
per-row and no cluster, purity or ambiguity logic is involved.
"""

from .isolated import (
    CONDITION,
    PRIMARY_POINTS,
    TIER_NORMAL,
    TIER_THIN,
    TIER_UNTRAINED,
    IsolatedModel,
    PooledMean,
    binding_summary,
    budget_share,
    evaluate_institution,
    masked_scorer,
    operating_points,
    oracle_view_scorer,
    pooled_mean,
    run_cell,
    train_all_isolated,
    train_isolated,
    training_tier,
    undetected_outcomes,
    union_cluster_accounting,
    union_outcomes,
)
from .views import (
    MIN_POSITIVES_TO_TRAIN,
    VisibilityAssignment,
    ViewLoadReport,
    all_views,
    assignment_from_pipeline,
    boundary_accounts,
    campaign_visibility_from_views,
    campaign_visibility_table,
    endpoint_institutions,
    institution_view,
    natural_assignment,
    view_load_report,
    visible_mask,
    visible_transactions,
)

__all__ = [
    "CONDITION",
    "PRIMARY_POINTS",
    "IsolatedModel",
    "PooledMean",
    "TIER_NORMAL",
    "TIER_THIN",
    "TIER_UNTRAINED",
    "binding_summary",
    "budget_share",
    "evaluate_institution",
    "masked_scorer",
    "operating_points",
    "oracle_view_scorer",
    "pooled_mean",
    "run_cell",
    "train_all_isolated",
    "train_isolated",
    "training_tier",
    "undetected_outcomes",
    "union_cluster_accounting",
    "union_outcomes",
    "MIN_POSITIVES_TO_TRAIN",
    "VisibilityAssignment",
    "ViewLoadReport",
    "all_views",
    "assignment_from_pipeline",
    "boundary_accounts",
    "campaign_visibility_from_views",
    "campaign_visibility_table",
    "endpoint_institutions",
    "institution_view",
    "natural_assignment",
    "view_load_report",
    "visible_mask",
    "visible_transactions",
]
