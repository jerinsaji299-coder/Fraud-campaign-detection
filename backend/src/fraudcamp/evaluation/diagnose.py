"""Why does the oracle miss campaigns it has perfect information about?

The oracle scores every edge with its true label, so a miss is never a
scoring failure. It is always a property of the campaign's shape in time,
or of the matching rule meeting a graph that does not cooperate. This
module says which, per campaign, and never changes the rule.

Five reasons, in the order they are tested:

`purity_failure`
    A connected fragment of 3+ accounts did form, but the cluster it sat
    in was less than half the campaign's own accounts — the campaign was
    merged into something larger, so the matching rule declined it. The
    merge partners are named: other campaigns, unassigned laundering, or
    both.

`late_only`
    A cluster that would have matched did appear, but only after the
    deadline. The information arrived; it arrived too late to count.

`structurally_disconnected`
    Pool **every** one of the campaign's own edges before its deadline,
    with no window at all, and its largest connected set of accounts is
    still smaller than the rule's 3. No lookback, however long, can ever
    detect it under the frozen rule: the evidence does not exist in the
    graph, so this is a property of the campaign, not of the detector.
    This is the ceiling's hard floor.

`slow`
    Its own edges *do* connect 3 or more accounts once the whole campaign
    is pooled, but never within a single window before the deadline. The
    evidence exists and arrives too spread out, so a longer lookback may
    reach it — the reported largest transaction gap says whether it could.

These last two were one reason, `temporal_sparsity`, until 2026-10-17. They
were separated because they say opposite things about what to do: nothing
can be done about a disconnected campaign, while a slow one is a lookback
question. Pooling them hid which of the two the ceiling was made of.

`unexplained`
    Nothing above fits. Reported loudly rather than bucketed away, because
    it would mean this diagnosis or the evaluation disagrees with itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .. import constants, windows
from . import clusters as cluster_tools

#: In precedence order: the first that fits is the reason reported.
REASONS = (
    "purity_failure",
    "late_only",
    "structurally_disconnected",
    "slow",
    "unexplained",
)


@dataclass
class MissDiagnosis:
    campaign_id: int
    split: str
    lookback_h: int
    base_type: str
    group: str
    reason: str
    detail: str
    n_txn: int
    n_accounts: int
    duration_h: float
    deadline: str
    #: Largest set of the campaign's own accounts connected through its own
    #: laundering edges, over all windows up to the deadline.
    max_connected_own_accounts: int
    #: The same, but pooling **every** own edge before the deadline with no
    #: window at all. Below `MATCH_MIN_ACCOUNTS` this campaign is
    #: undetectable at any lookback, which is what separates
    #: `structurally_disconnected` from `slow`.
    max_connected_all_edges: int
    #: Longest gap between consecutive campaign transactions, in hours, and
    #: whether it exceeds the lookback.
    max_txn_gap_h: float
    gap_exceeds_lookback: bool
    #: Best purity achieved by a cluster containing 3+ of its accounts.
    best_cluster_purity: float | None
    best_cluster_size: int | None
    merged_with_campaigns: int
    merged_with_unassigned: int
    windows_checked: int


def _own_edge_components(window: pd.DataFrame, campaign_id: int) -> int:
    """Largest number of a campaign's accounts connected via its own edges."""
    own = window[window["campaign_id"] == campaign_id]
    if len(own) == 0:
        return 0
    found = cluster_tools.extract_clusters(
        own["src"].tolist(), own["dst"].tolist(), np.ones(len(own)), tau=0.5,
        min_accounts=1,
    )
    return max((len(cluster) for cluster in found), default=0)


def diagnose_missed_campaign(
    full_df: pd.DataFrame,
    campaign_row,
    campaign_accounts: tuple[str, ...],
    accounts_by_campaign: dict[int, tuple[str, ...]],
    unassigned_accounts: set[str],
    split: str,
    lookback_h: int,
) -> MissDiagnosis:
    """Classify one missed campaign. Diagnosis only: nothing here scores."""
    campaign_id = int(campaign_row.campaign_id)
    deadline = pd.Timestamp(campaign_row.deadline)
    own_accounts = set(campaign_accounts)

    campaign_txns = full_df[full_df["campaign_id"] == campaign_id].sort_values("ts")
    stamps = campaign_txns["ts"].to_numpy()
    gaps = np.diff(stamps) / np.timedelta64(1, "h") if len(stamps) > 1 else np.array([0.0])
    max_gap = float(gaps.max()) if gaps.size else 0.0

    # The lookback-free upper bound on connectivity. Strictly before the
    # deadline: a window ends at some t <= deadline and holds ts < t, so an
    # edge at exactly the deadline can never be inside one. If this cannot
    # reach 3 connected accounts, no window can either.
    max_connected_all_edges = _own_edge_components(
        campaign_txns[campaign_txns["ts"] < deadline], campaign_id
    )

    all_times = windows.evaluation_detection_times(split)
    before = [t for t in all_times if t <= deadline]
    after = [t for t in all_times if t > deadline]

    max_connected = 0
    best_purity: float | None = None
    best_size: int | None = None
    merged_campaigns = 0
    merged_unassigned = 0
    reached_three = False

    for t in before:
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue

        connected = _own_edge_components(window, campaign_id)
        max_connected = max(max_connected, connected)
        if connected < constants.MATCH_MIN_ACCOUNTS:
            continue
        reached_three = True

        # the cluster the oracle would actually form: all laundering edges
        laundering = window[window["Is Laundering"] == 1]
        if len(laundering) == 0:
            continue
        for cluster in cluster_tools.extract_clusters(
            laundering["src"].tolist(), laundering["dst"].tolist(),
            np.ones(len(laundering)), tau=0.5,
        ):
            overlap = len(cluster & own_accounts)
            if overlap < constants.MATCH_MIN_ACCOUNTS:
                continue
            purity = overlap / len(cluster)
            if best_purity is None or purity > best_purity:
                best_purity = purity
                best_size = len(cluster)
                others = cluster - own_accounts
                merged_campaigns = len(
                    {
                        other_id
                        for account in others
                        for other_id, accounts in accounts_by_campaign.items()
                        if other_id != campaign_id and account in set(accounts)
                    }
                )
                merged_unassigned = len(others & unassigned_accounts)

    # would it have matched later?
    late = False
    for t in after:
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue
        laundering = window[window["Is Laundering"] == 1]
        if len(laundering) == 0:
            continue
        for cluster in cluster_tools.extract_clusters(
            laundering["src"].tolist(), laundering["dst"].tolist(),
            np.ones(len(laundering)), tau=0.5,
        ):
            if cluster_tools.cluster_hits_campaign(cluster, campaign_accounts):
                late = True
                break
        if late:
            break

    # Precedence matters, and the most specific explanation wins.
    #
    # `purity_failure` first: if a big-enough fragment did form before the
    # deadline and the rule still declined it, that is the reason, whatever
    # happens later.
    #
    # `late_only` next, and crucially *above* temporal sparsity: a campaign
    # whose transactions are an hour apart but whose deadline falls before
    # the first window that can see them is not sparse — it is censored.
    # Calling that "temporal sparsity" would send the reader looking for a
    # longer lookback when the problem is the deadline.
    if reached_three and (
        best_purity is None or best_purity < constants.MATCH_MIN_ACCOUNT_FRACTION
    ):
        reason = "purity_failure"
        partners = []
        if merged_campaigns:
            partners.append(f"{merged_campaigns} other campaign(s)")
        if merged_unassigned:
            partners.append(f"{merged_unassigned} unassigned-laundering account(s)")
        detail = (
            f"reached {max_connected} connected accounts, but the best cluster was "
            f"{best_purity:.2f} its own accounts"
            if best_purity is not None
            else f"reached {max_connected} connected accounts but no cluster covered 3 of them"
        )
        if partners:
            detail += " — merged with " + " and ".join(partners)
    elif late:
        reason = "late_only"
        detail = (
            "a matching cluster existed, but only after the deadline "
            f"({deadline}); detection after the deadline counts as missed. "
            f"Reached {max_connected} connected accounts before the deadline, "
            f"over {len(before)} window(s)"
        )
    elif not reached_three and max_connected_all_edges < constants.MATCH_MIN_ACCOUNTS:
        # Pooling every own edge before the deadline still cannot connect 3
        # accounts, so no lookback can. Separated from `slow` because no
        # choice of L would change it.
        reason = "structurally_disconnected"
        detail = (
            f"pooling every one of its own edges before the deadline, with no "
            f"window at all, its largest connected set is "
            f"{max_connected_all_edges} of its {campaign_row.n_accounts} accounts; "
            f"the rule needs {constants.MATCH_MIN_ACCOUNTS}, so no lookback can "
            f"detect it. {campaign_row.n_txn} transactions over "
            f"{campaign_row.duration_h:.1f}h"
        )
    elif not reached_three:
        # The evidence does connect, just never inside one window.
        reason = "slow"
        detail = (
            f"its own edges connect {max_connected_all_edges} of its "
            f"{campaign_row.n_accounts} accounts once the whole campaign is "
            f"pooled, but never more than {max_connected} inside a single "
            f"{lookback_h}h window before the deadline, and no matching cluster "
            f"formed afterwards either; {campaign_row.n_txn} transactions over "
            f"{campaign_row.duration_h:.1f}h, largest gap {max_gap:.1f}h"
            + (f" (exceeds L={lookback_h}h)" if max_gap > lookback_h else "")
        )
    else:
        reason = "unexplained"
        detail = (
            f"reached {max_connected} connected accounts with best purity "
            f"{best_purity}; this should have been detected — the diagnosis and "
            "the evaluation disagree and one of them is wrong"
        )

    return MissDiagnosis(
        campaign_id=campaign_id,
        split=split,
        lookback_h=lookback_h,
        base_type=str(campaign_row.base_type),
        group=str(campaign_row.group),
        reason=reason,
        detail=detail,
        n_txn=int(campaign_row.n_txn),
        n_accounts=int(campaign_row.n_accounts),
        duration_h=float(campaign_row.duration_h),
        deadline=str(deadline),
        max_connected_own_accounts=max_connected,
        max_connected_all_edges=max_connected_all_edges,
        max_txn_gap_h=round(max_gap, 2),
        gap_exceeds_lookback=bool(max_gap > lookback_h),
        best_cluster_purity=round(best_purity, 4) if best_purity is not None else None,
        best_cluster_size=best_size,
        merged_with_campaigns=merged_campaigns,
        merged_with_unassigned=merged_unassigned,
        windows_checked=len(before),
    )


def _failed_condition(
    cluster: set[str],
    touching: set[int],
    accounts_by_campaign: dict[int, tuple[str, ...]],
) -> dict:
    """Which half of the matching rule does this false alarm fail?

    The rule has two conditions, and reporting only one number cannot say
    which bound was binding:

    * **purity** — at least `MATCH_MIN_ACCOUNT_FRACTION` of the *cluster's*
      accounts belong to the campaign;
    * **coverage** — the cluster covers at least `MATCH_MIN_ACCOUNTS` of the
      *campaign's* accounts.

    A cluster two thirds of which belongs to one campaign has passed purity
    comfortably and failed coverage, so describing its 0.667 as "below the
    0.5 the rule requires" is simply wrong. Each is therefore reported
    against its own threshold, and `fails` names the binding one.
    """
    best_purity = 0.0
    best_coverage = 0
    passes_purity = False
    passes_coverage = False
    for cid in touching:
        overlap = len(cluster & set(accounts_by_campaign[cid]))
        purity = overlap / len(cluster) if cluster else 0.0
        best_purity = max(best_purity, purity)
        best_coverage = max(best_coverage, overlap)
        if purity >= constants.MATCH_MIN_ACCOUNT_FRACTION:
            passes_purity = True
        if overlap >= constants.MATCH_MIN_ACCOUNTS:
            passes_coverage = True

    if not touching:
        fails = "no_campaign_accounts"
    elif passes_purity:
        fails = "coverage"  # pure enough, too few of the campaign's accounts
    elif passes_coverage:
        fails = "purity"  # enough of the campaign, too diluted by others
    else:
        fails = "both"

    return {
        "best_purity_fraction": round(best_purity, 4),
        "best_coverage_accounts": best_coverage,
        "fails": fails,
    }


FALSE_ALARM_CONDITIONS = ("purity", "coverage", "both", "no_campaign_accounts")


def diagnose_false_alarms(
    full_df: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    unassigned_accounts: set[str],
    split: str,
    lookback_h: int,
    limit: int = 50,
) -> list[dict]:
    """Describe what the oracle's false-alarm clusters actually are.

    The oracle only ever keeps laundering edges, so every cluster it forms
    is made of real laundering. A false alarm therefore means: real
    laundering that the matching rule could not attribute to a campaign of
    the split being scored, and that was not majority unassigned either.
    Worth characterising rather than accepting as noise.
    """
    campaign_splits_accounts = accounts_by_campaign
    rows: list[dict] = []

    for t in windows.evaluation_detection_times(split):
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue
        laundering = window[window["Is Laundering"] == 1]
        if len(laundering) == 0:
            continue

        account_index = cluster_tools.build_account_index(campaign_splits_accounts)
        for cluster in cluster_tools.extract_clusters(
            laundering["src"].tolist(), laundering["dst"].tolist(),
            np.ones(len(laundering)), tau=0.5,
        ):
            hit_ids = cluster_tools.find_hit_campaigns(
                cluster, campaign_splits_accounts, account_index
            )
            if hit_ids:
                continue  # a hit or an other-split hit, not a false alarm
            if cluster_tools.is_ambiguous(cluster, unassigned_accounts):
                continue  # ambiguous, not a false alarm

            touching = {
                cid
                for account in cluster
                for cid in account_index.get(account, ())
            }
            in_campaigns = sum(
                1 for account in cluster if account_index.get(account)
            )
            rows.append(
                {
                    "t": str(t),
                    "split": split,
                    "lookback_h": lookback_h,
                    "cluster_size": len(cluster),
                    "accounts_in_some_campaign": in_campaigns,
                    "campaigns_touched": len(touching),
                    "accounts_unassigned_laundering": len(cluster & unassigned_accounts),
                    **_failed_condition(cluster, touching, campaign_splits_accounts),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def diagnose_misses(
    full_df: pd.DataFrame,
    camp: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    missed_campaign_ids: list[int],
    split: str,
    lookback_h: int,
) -> list[dict]:
    """Diagnose every missed campaign in one split at one lookback."""
    unassigned = full_df[(full_df["Is Laundering"] == 1) & (full_df["campaign_id"] == -1)]
    unassigned_accounts = set(unassigned["src"]) | set(unassigned["dst"])

    rows: list[dict] = []
    for campaign_id in missed_campaign_ids:
        matches = camp[camp["campaign_id"] == campaign_id]
        if matches.empty:
            continue
        row = next(matches.itertuples())
        rows.append(
            asdict(
                diagnose_missed_campaign(
                    full_df, row, accounts_by_campaign[int(campaign_id)],
                    accounts_by_campaign, unassigned_accounts, split, lookback_h,
                )
            )
        )
    return rows
