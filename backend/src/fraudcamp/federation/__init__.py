"""Phase 4: the federated conditions.

Stage 4.0 (done) provides the data substrate every later condition stands on:
per-institution **views** of the transaction data under a given
``(seed, target)`` visibility assignment, and the account features each
institution can compute from its own view alone.

Nothing here trains or scores anything. The four conditions consume views
built here and are evaluated by the unchanged Phase 3 evaluation code:
``isolated`` is stage 4.1 (proposed, awaiting approval), ``fedavg`` is 4.2,
``fedavg_exchange`` and the gate are 4.3, and ``centralized`` is the Phase 3
model already measured.

In ``isolated``, an institution is restricted by its **scorer** and not by
its data: the evaluation keeps the global transactions, campaigns and
deadlines, and the scorer returns 0 outside ``visible_mask`` and its model's
probability inside, with features built from ``institution_view`` alone.
Visibility restricts what can be scored, never what is true.
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
