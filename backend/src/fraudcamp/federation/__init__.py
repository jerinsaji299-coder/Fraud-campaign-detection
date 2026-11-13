"""Phase 4: the federated conditions.

Stage 4.0 (done) provides the data substrate every later condition stands on:
per-institution **views** of the transaction data under a given
``(seed, target)`` visibility assignment, and the account features each
institution can compute from its own view alone.

Nothing here trains or scores anything. The four conditions — isolated,
fedavg, fedavg_exchange and centralized — are defined in stage 4.1 and
implemented in 4.2, and they all consume views built here and are evaluated
by the unchanged Phase 3 evaluation code.
"""

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
