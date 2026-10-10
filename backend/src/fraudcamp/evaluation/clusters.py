"""Turning edge scores into clusters, and clusters into outcomes.

This is the fixed, non-learned half of the evaluation: identical for every
model and every later condition, so differences in results come from the
scores alone.

At a detection time, edges scoring at or above tau are kept, an undirected
graph is built on their accounts, and connected components with at least
`MATCH_MIN_ACCOUNTS` accounts become clusters. A cluster then gets one of
four outcomes (see `constants.HIT_KINDS`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .. import constants


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def _components_python(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """Component label per node, via union-find. The portable fallback."""
    find, union = _union_find(n)
    for a, b in zip(left.tolist(), right.tolist()):
        union(a, b)
    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


def _components_scipy(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix(
        (np.ones(left.size, dtype=np.int8), (left, right)), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def connected_component_labels(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """Component labels, using scipy when it is installed.

    A busy window holds over a million edges and this runs once per
    detection time per tau, so the pure-Python union-find is the difference
    between minutes and hours over a tau sweep. The two implementations are
    tested to produce identical clusterings; only the labels themselves may
    differ, which does not matter because clusters are keyed by membership.
    """
    try:
        return _components_scipy(left, right, n)
    except ImportError:
        return _components_python(left, right, n)


def extract_clusters(
    src: Sequence[str],
    dst: Sequence[str],
    scores: np.ndarray,
    tau: float,
    min_accounts: int = constants.MATCH_MIN_ACCOUNTS,
) -> list[frozenset[str]]:
    """Connected components of the edges scoring >= tau.

    Accounts are factorised with `sort=True`, so the enumeration order is
    the sorted account order and does not depend on hash ordering — the same
    reason the visibility search sorts.
    """
    import pandas as pd

    scores = np.asarray(scores)
    keep = np.flatnonzero(scores >= tau)
    if keep.size == 0:
        return []

    kept_src = np.asarray(src, dtype=object)[keep]
    kept_dst = np.asarray(dst, dtype=object)[keep]

    codes, accounts = pd.factorize(np.concatenate([kept_src, kept_dst]), sort=True)
    half = keep.size
    labels = connected_component_labels(codes[:half], codes[half:], len(accounts))

    # group account indices by component label, vectorised
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(order, boundaries)

    accounts = np.asarray(accounts, dtype=object)
    clusters = [
        frozenset(accounts[group].tolist()) for group in groups if group.size >= min_accounts
    ]
    # sorted for a reproducible report order
    return sorted(clusters, key=lambda c: (-len(c), min(c)))


def cluster_hits_campaign(
    cluster: Iterable[str],
    campaign_accounts: Iterable[str],
    min_fraction: float = constants.MATCH_MIN_ACCOUNT_FRACTION,
    min_accounts: int = constants.MATCH_MIN_ACCOUNTS,
) -> bool:
    """The frozen matching rule: at least `min_fraction` of the cluster's
    accounts belong to the campaign, AND it covers at least `min_accounts`
    of the campaign's accounts."""
    cluster = set(cluster)
    overlap = len(cluster & set(campaign_accounts))
    if not cluster:
        return False
    return (overlap / len(cluster)) >= min_fraction and overlap >= min_accounts


def build_account_index(accounts_by_campaign: dict[int, Sequence[str]]) -> dict[str, list[int]]:
    """account -> campaigns containing it, so a cluster only has to be
    compared against campaigns it actually touches."""
    index: dict[str, list[int]] = {}
    for campaign_id, accounts in sorted(accounts_by_campaign.items()):
        for account in accounts:
            index.setdefault(account, []).append(campaign_id)
    return index


def find_hit_campaigns(
    cluster: Iterable[str],
    accounts_by_campaign: dict[int, Sequence[str]],
    account_index: dict[str, list[int]] | None = None,
) -> list[int]:
    """Every campaign this cluster hits, lowest id first.

    More than one is possible only in the knife-edge case where a cluster
    splits exactly 50/50 across two campaigns, so it is allowed rather than
    assumed away.
    """
    cluster = set(cluster)
    index = account_index or build_account_index(accounts_by_campaign)
    candidates = {cid for account in cluster for cid in index.get(account, ())}
    return sorted(
        cid
        for cid in candidates
        if cluster_hits_campaign(cluster, accounts_by_campaign[cid])
    )


def is_ambiguous(
    cluster: Iterable[str],
    unassigned_laundering_accounts: set[str],
    min_fraction: float = constants.AMBIGUOUS_MIN_FRACTION,
) -> bool:
    """Whether a cluster that hit no campaign is *ambiguous* rather than a
    false alarm.

    "Mostly involved in unassigned-laundering edges" is read as: at least
    `min_fraction` of the cluster's accounts touch an edge that is labelled
    laundering but belongs to no campaign. Those edges are real laundering
    with no campaign to match against, so counting such a cluster as a false
    alarm would punish a correct detection. The 0.5 default mirrors the
    matching rule's threshold. (Interpretation - see README decision log.)
    """
    cluster = set(cluster)
    if not cluster:
        return False
    touching = len(cluster & unassigned_laundering_accounts)
    return (touching / len(cluster)) >= min_fraction


def classify_cluster(
    cluster: Iterable[str],
    evaluated_split: str,
    accounts_by_campaign: dict[int, Sequence[str]],
    campaign_splits: dict[int, str | None],
    unassigned_laundering_accounts: set[str],
    account_index: dict[str, list[int]] | None = None,
) -> tuple[str, list[int]]:
    """One of `constants.HIT_KINDS`, plus the campaigns hit.

    A cluster hitting any campaign of the evaluated split is a hit, even if
    it also hits campaigns elsewhere. If it only hits campaigns of other
    splits it is an other-split hit (rule 1): neither a success for this
    split nor a false alarm.
    """
    hit_ids = find_hit_campaigns(cluster, accounts_by_campaign, account_index)

    if hit_ids:
        same_split = [cid for cid in hit_ids if campaign_splits.get(cid) == evaluated_split]
        if same_split:
            return "hit", same_split
        return "other_split_hit", hit_ids

    if is_ambiguous(cluster, unassigned_laundering_accounts):
        return "ambiguous", []
    return "false_alarm", []
