"""Campaign-level evaluation: extraction, matching, lead time, metrics.

Fixed and non-learned, so it is identical for every model and every later
condition and differences in results come from the scores alone.
"""

from .clusters import (
    build_account_index,
    classify_cluster,
    cluster_hits_campaign,
    extract_clusters,
    find_hit_campaigns,
    is_ambiguous,
)
from .evaluate import (
    SUBSET_FILTERS,
    CampaignOutcome,
    EvaluationResult,
    bootstrap_ci,
    evaluate_split,
    summarize,
    summarize_by,
)
from .scorers import SANITY_TAU, oracle_scores, random_scores, score_window

__all__ = [
    "build_account_index",
    "classify_cluster",
    "cluster_hits_campaign",
    "extract_clusters",
    "find_hit_campaigns",
    "is_ambiguous",
    "CampaignOutcome",
    "EvaluationResult",
    "SUBSET_FILTERS",
    "bootstrap_ci",
    "evaluate_split",
    "summarize",
    "summarize_by",
    "SANITY_TAU",
    "oracle_scores",
    "random_scores",
    "score_window",
]
