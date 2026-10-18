"""Campaign-level evaluation: when was each campaign discovered, how early,
and how much noise did it cost.

The loop walks a split's evaluation horizon in order. At each detection time
it scores the window, extracts clusters, and classifies each one. A campaign
is detected at the first detection time where a hitting cluster appears, and
only if that time is at or before its deadline; otherwise it is missed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import constants, windows
from . import clusters as cluster_tools
from .scorers import CachingScorer


@dataclass
class CampaignOutcome:
    campaign_id: int
    split: str | None
    base_type: str
    group: str
    detected: bool
    detection_time: pd.Timestamp | None
    lead_time_h: float | None
    normalized_lead_time: float | None
    frac_observed: float | None
    crosses_cutoff: bool
    shares_train_account: bool
    deadline: pd.Timestamp
    end: pd.Timestamp
    duration_h: float
    n_txn: int


@dataclass
class EvaluationResult:
    split: str
    scorer: str
    tau: float
    lookback_h: int
    seed: int
    outcomes: list[CampaignOutcome]
    n_hits_clusters: int = 0
    n_other_split_hits: int = 0
    n_ambiguous: int = 0
    n_false_alarms: int = 0
    n_clusters: int = 0
    n_detection_times: int = 0
    horizon_days: float = 0.0
    per_time: list[dict] = field(default_factory=list)

    @property
    def false_alarms_per_day(self) -> float:
        return self.n_false_alarms / self.horizon_days if self.horizon_days else 0.0


def _lead_time(
    campaign, t: pd.Timestamp, campaign_txn_times: np.ndarray
) -> tuple[float, float | None, float]:
    """Lead time in the three frozen forms, measured against true completion."""
    end = pd.Timestamp(campaign["end"])
    lead_h = (end - t).total_seconds() / 3600.0

    duration = float(campaign["duration_h"])
    normalized = lead_h / duration if duration > 0 else None

    observed = int((campaign_txn_times < np.datetime64(t)).sum())
    frac_observed = observed / int(campaign["n_txn"]) if campaign["n_txn"] else 0.0
    return lead_h, normalized, frac_observed


def evaluate_split(
    full_df: pd.DataFrame,
    camp: pd.DataFrame,
    pattern_df: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    split: str,
    scorer,
    tau: float,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    seed: int = 0,
    scorer_name: str = "custom",
) -> EvaluationResult:
    """Evaluate one split's eval_ok campaigns over its detection horizon.

    `scorer` is called as `scorer(window_transactions, t)` and must return
    one score per transaction, in the window's row order. It never receives
    anything outside the window, so a scorer cannot cheat on time even if it
    tries.
    """
    detection_times = windows.evaluation_detection_times(split)
    eligible = camp[(camp["eval_ok"]) & (camp["split"] == split)]

    campaign_splits = dict(zip(camp["campaign_id"], camp["split"]))
    account_index = cluster_tools.build_account_index(accounts_by_campaign)
    # Looked up once per hit inside the detection loop, so a dict rather
    # than a dataframe filter: the filter is O(campaigns) per hit and the
    # full data produces thousands of hits.
    deadlines = {
        int(cid): pd.Timestamp(deadline)
        for cid, deadline in zip(camp["campaign_id"], camp["deadline"])
    }

    # accounts touching laundering that belongs to no campaign - needed for
    # the ambiguous classification
    unassigned = full_df[(full_df["Is Laundering"] == 1) & (full_df["campaign_id"] == -1)]
    unassigned_accounts = set(unassigned["src"]) | set(unassigned["dst"])

    # per-campaign transaction times, for frac_observed
    txn_times = {
        int(cid): group["ts"].to_numpy()
        for cid, group in pattern_df.groupby("campaign_id", sort=True)
    }

    first_detection: dict[int, pd.Timestamp] = {}
    result = EvaluationResult(
        split=split,
        scorer=scorer_name,
        tau=tau,
        lookback_h=lookback_h,
        seed=seed,
        outcomes=[],
        n_detection_times=len(detection_times),
    )
    if detection_times:
        span = detection_times[-1] - detection_times[0]
        result.horizon_days = span.total_seconds() / 86400.0 or 1.0

    for t in detection_times:
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue

        scores = np.asarray(scorer(window, t))
        found = cluster_tools.extract_clusters(
            window["src"].tolist(), window["dst"].tolist(), scores, tau
        )

        kinds = {kind: 0 for kind in constants.HIT_KINDS}
        for cluster in found:
            kind, hit_ids = cluster_tools.classify_cluster(
                cluster,
                split,
                accounts_by_campaign,
                campaign_splits,
                unassigned_accounts,
                account_index,
            )
            kinds[kind] += 1
            if kind == "hit":
                for cid in hit_ids:
                    # first detection wins, and only if it beats the deadline
                    if cid in first_detection:
                        continue
                    if t <= deadlines[cid]:
                        first_detection[cid] = t

        result.n_clusters += len(found)
        result.n_hits_clusters += kinds["hit"]
        result.n_other_split_hits += kinds["other_split_hit"]
        result.n_ambiguous += kinds["ambiguous"]
        result.n_false_alarms += kinds["false_alarm"]
        result.per_time.append(
            {
                "t": t,
                "n_edges": len(window),
                "n_clusters": len(found),
                **{f"n_{k}": v for k, v in kinds.items()},
            }
        )

    for row in eligible.itertuples():
        cid = int(row.campaign_id)
        t = first_detection.get(cid)
        if t is None:
            result.outcomes.append(
                CampaignOutcome(
                    campaign_id=cid, split=row.split, base_type=row.base_type,
                    group=row.group, detected=False, detection_time=None,
                    lead_time_h=None, normalized_lead_time=None, frac_observed=None,
                    crosses_cutoff=bool(row.crosses_cutoff),
                    shares_train_account=bool(row.shares_train_account),
                    deadline=pd.Timestamp(row.deadline), end=pd.Timestamp(row.end),
                    duration_h=float(row.duration_h), n_txn=int(row.n_txn),
                )
            )
            continue

        lead_h, normalized, frac = _lead_time(
            {"end": row.end, "duration_h": row.duration_h, "n_txn": row.n_txn},
            t,
            txn_times.get(cid, np.array([])),
        )
        result.outcomes.append(
            CampaignOutcome(
                campaign_id=cid, split=row.split, base_type=row.base_type,
                group=row.group, detected=True, detection_time=t,
                lead_time_h=lead_h, normalized_lead_time=normalized, frac_observed=frac,
                crosses_cutoff=bool(row.crosses_cutoff),
                shares_train_account=bool(row.shares_train_account),
                deadline=pd.Timestamp(row.deadline), end=pd.Timestamp(row.end),
                duration_h=float(row.duration_h), n_txn=int(row.n_txn),
            )
        )

    return result


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


SUBSET_FILTERS = {
    "all": lambda o: True,
    "fully_observed": lambda o: not o.crosses_cutoff,
    "no_shared_accounts": lambda o: not o.shares_train_account,
}


def subset_filter(subset: str, oracle_reachable: Iterable[int] | None = None):
    """The predicate for a subset name.

    `oracle_reachable` is not a property of a campaign — it is the set the
    oracle detected at this lookback and split — so it has to be supplied.
    Asking for it without supplying the set raises rather than quietly
    falling back to every campaign, which would report a ceiling-relative
    number against the wrong denominator.
    """
    if subset != "oracle_reachable":
        return SUBSET_FILTERS[subset]
    if oracle_reachable is None:
        raise ValueError(
            "the 'oracle_reachable' subset needs the oracle's detected campaign "
            "ids for this split and lookback; pass oracle_reachable=..."
        )
    reachable = frozenset(int(c) for c in oracle_reachable)
    return lambda o: o.campaign_id in reachable


def recall_relative_to_ceiling(
    recall: float | None, oracle_recall: float | None
) -> float | None:
    """A model's recall as a fraction of the oracle's.

    The oracle is the best any detector can do under the frozen matching
    rule, and on the full data it reaches 0.821 (L = 24) / 0.893 (L = 48) on
    test — not 1.0. Reporting a model's recall against 1.0 therefore
    understates it by counting campaigns no detector could have found. This
    is the honest denominator; `campaign_recall` against `all` is kept
    beside it so the absolute number stays visible.
    """
    if recall is None or oracle_recall is None or oracle_recall == 0:
        return None
    return recall / oracle_recall


def bootstrap_ci(
    values: list[float],
    statistic=np.median,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI over campaigns. Seeded, so a reported
    interval is reproducible."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    array = np.asarray(clean, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(n_resamples, len(array)))
    stats = statistic(array[draws], axis=1)
    return (
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    )


def summarize(
    result: EvaluationResult,
    subset: str = "all",
    bootstrap_seed: int = 0,
    n_resamples: int = 1000,
    oracle_reachable: Iterable[int] | None = None,
    oracle_recall: float | None = None,
) -> dict:
    """Campaign-level metrics for one subset of a split's campaigns.

    `oracle_reachable` is required for the subset of that name. Passing
    `oracle_recall` adds `recall_relative_to_ceiling`, the recall as a
    fraction of what the oracle achieved on the same subset.
    """
    keep = subset_filter(subset, oracle_reachable)
    outcomes = [o for o in result.outcomes if keep(o)]
    detected = [o for o in outcomes if o.detected]

    def _median(values):
        clean = [v for v in values if v is not None]
        return float(np.median(clean)) if clean else None

    recall = len(detected) / len(outcomes) if outcomes else None
    # precision uses cluster-level counts: other-split hits and ambiguous
    # clusters are excluded from both terms (rules 1 and the ambiguous rule)
    denominator = result.n_hits_clusters + result.n_false_alarms
    precision = result.n_hits_clusters / denominator if denominator else None

    lead_h = [o.lead_time_h for o in detected]
    normalized = [o.normalized_lead_time for o in detected]
    observed = [o.frac_observed for o in detected]

    return {
        "split": result.split,
        "scorer": result.scorer,
        "tau": result.tau,
        "lookback_h": result.lookback_h,
        "seed": result.seed,
        "test_subset": subset,
        "n_campaigns": len(outcomes),
        "n_detected": len(detected),
        "campaign_recall": recall,
        "recall_ci": bootstrap_ci(
            [1.0 if o.detected else 0.0 for o in outcomes],
            statistic=np.mean, seed=bootstrap_seed, n_resamples=n_resamples,
        ),
        "campaign_precision": precision,
        "hits": result.n_hits_clusters,
        "other_split_hits": result.n_other_split_hits,
        "ambiguous": result.n_ambiguous,
        "false_alarms": result.n_false_alarms,
        "false_alarms_per_day": round(result.false_alarms_per_day, 2),
        "n_clusters": result.n_clusters,
        "median_lead_time_h": _median(lead_h),
        "median_lead_time_h_ci": bootstrap_ci(lead_h, seed=bootstrap_seed, n_resamples=n_resamples),
        "median_normalized_lead_time": _median(normalized),
        "median_frac_observed_at_detection": _median(observed),
        "oracle_recall": oracle_recall,
        "recall_relative_to_ceiling": recall_relative_to_ceiling(
            recall, oracle_recall
        ),
    }


def campaign_f1(
    result: EvaluationResult,
    subset: str = "all",
    oracle_reachable: Iterable[int] | None = None,
) -> dict:
    """Campaign-level precision, recall and F1, without the bootstrap.

    Used for tau selection, where the statistic is recomputed for every
    candidate threshold and a 1000-sample bootstrap each time would dominate
    the runtime.
    """
    keep = subset_filter(subset, oracle_reachable)
    outcomes = [o for o in result.outcomes if keep(o)]
    detected = [o for o in outcomes if o.detected]

    recall = len(detected) / len(outcomes) if outcomes else 0.0
    denominator = result.n_hits_clusters + result.n_false_alarms
    precision = result.n_hits_clusters / denominator if denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "campaign_precision": precision,
        "campaign_recall": recall,
        "campaign_f1": f1,
        "hits": result.n_hits_clusters,
        "false_alarms": result.n_false_alarms,
        "other_split_hits": result.n_other_split_hits,
        "ambiguous": result.n_ambiguous,
        "n_campaigns": len(outcomes),
        "n_detected": len(detected),
    }


#: Default thresholds swept when choosing tau. Deliberately weighted toward
#: the high end: a low tau keeps most of a one-to-two-million-edge window,
#: and clustering cost scales with the edges kept, so a uniform grid spends
#: almost all its time in the region no detector would choose.
DEFAULT_TAU_GRID = (0.5, 0.75, 0.9, 0.95, 0.99)

#: The grid for **model** scorers: XGBoost now, the GNN and the federated
#: conditions later. Frozen 2026-10-18 and the same for every model and
#: condition, so a comparison between them is never a comparison of how far
#: each was allowed to search.
#:
#: Models need both ends that `DEFAULT_TAU_GRID` omits. That grid was built
#: for the sanity scorers, whose scores are exactly 0 or 1, so every
#: threshold in (0, 1] behaves identically and the choice is vacuous. A
#: class-weighted probabilistic model is the opposite case, and its scores
#: are not merely spread over the unit interval — they **pile up against
#: 1**. Both ends were found by running into them:
#:
#: * On the 1% sample the baseline's validation F1 peaked at 0.3, below the
#:   old floor of 0.5, so the low end was added.
#: * On the full data (L = 24, seed 0) F1 then rose from 0.128 at tau 0.95
#:   to 0.543 at 0.99 and the sweep chose 0.99, the top of the grid. A jump
#:   that large across the last interval says the optimum is above it: with
#:   per-window class weights of 495-3,461, scores concentrate so close to 1
#:   that a linear grid cannot resolve the region where the decision
#:   actually happens. Hence the near-1 tail.
#:
#: If a chosen tau ever lands on 0.9999, more nines are not the answer —
#: that would mean the scores are saturated, and the thing to look at is the
#: class weighting, which is a design decision rather than a grid detail.
#:
#: Cost is asymmetric: clustering scales with the edges kept, so the low
#: thresholds are expensive and the tail is nearly free. `select_tau`
#: records the seconds each tau took and the reports carry them.
MODEL_TAU_GRID = (
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
    0.995, 0.999, 0.9995, 0.9999,
)


def select_tau(
    full_df: pd.DataFrame,
    camp: pd.DataFrame,
    pattern_df: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    scorer,
    tau_grid=DEFAULT_TAU_GRID,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    seed: int = 0,
    scorer_name: str = "custom",
    cache_scores: bool = True,
) -> tuple[float, list[dict]]:
    """Choose tau on **validation only**, maximising campaign-level F1.

    Returns the chosen tau and the full precision/recall-vs-tau curve, which
    the frozen design requires to be saved alongside it. Ties are broken
    deterministically by `best_tau` below.

    Validation is the only split touched here, so the threshold carries no
    information from test or stress.

    Each row carries `seconds`, the wall time that tau cost. Low thresholds
    keep most of a million-edge window and clustering scales with the edges
    kept, so the cost of searching the low end is on the record rather than
    inferred.

    `cache_scores` wraps the scorer so each validation window is featurised
    and scored **once** for the whole sweep instead of once per tau. A
    scorer's output does not depend on tau, so this cannot change the curve
    — `tests/test_tau_and_budgets.py` pins that it does not — and without it
    a 16-point grid rebuilds every window's features 16 times.
    """
    if cache_scores:
        scorer = CachingScorer(scorer, lookback_h)

    curve: list[dict] = []
    for tau in tau_grid:
        started = time.perf_counter()
        result = evaluate_split(
            full_df, camp, pattern_df, accounts_by_campaign,
            split="val", scorer=scorer, tau=float(tau),
            lookback_h=lookback_h, seed=seed, scorer_name=scorer_name,
        )
        row = {
            "tau": float(tau),
            "lookback_h": lookback_h,
            "false_alarms_per_day": result.false_alarms_per_day,
            "horizon_days": result.horizon_days,
            **campaign_f1(result),
            "seconds": round(time.perf_counter() - started, 2),
        }
        curve.append(row)

    return best_tau(curve), curve


def tau_is_at_grid_edge(tau: float, tau_grid) -> bool:
    """Is the chosen threshold the lowest or highest the sweep could pick?

    If it is, the optimum may lie outside the grid and the number reported
    is a boundary, not a maximum. Worth saying out loud rather than
    discovering later.
    """
    values = [float(v) for v in tau_grid]
    return float(tau) in (min(values), max(values))


def best_tau(curve: list[dict]) -> float:
    """The chosen tau, with a fully deterministic tie-break.

    Ties are common and must not be settled by list order, because the grid
    order is an implementation detail. In priority order:

    1. highest campaign-level F1 (the primary rule);
    2. then highest precision — of two equally-scoring thresholds, prefer the
       one that cries wolf less;
    3. then fewest false alarms — the same preference in absolute terms, for
       when precision also ties;
    4. then the **higher** tau — the more conservative detector, and the one
       that costs less to run.

    A scorer whose output is binary (the oracle) ties on everything and so
    lands on the highest tau in the grid, which is the honest answer: for
    0/1 scores every tau in (0, 1] is the same detector.
    """
    return max(
        curve,
        key=lambda row: (
            row["campaign_f1"],
            row["campaign_precision"],
            -row["false_alarms"],
            row["tau"],
        ),
    )["tau"]


#: False-alarm budgets, in alarms per day, at which recall and lead time are
#: additionally reported. Recall alone can be inflated by alerting more: the
#: random scorer reached 0.27 recall at L=48 while raising ~47,000 alarms a
#: day, which no analyst could use.
FALSE_ALARM_BUDGETS = (10, 50, 100)


def select_tau_for_budget(curve: list[dict], budget: float) -> float | None:
    """The **lowest** tau whose validation false-alarm rate stays inside
    `budget` alarms per day, or None when the grid has no such tau.

    Lowest rather than highest: within a fixed alert budget the most
    sensitive detector is the one worth having. Returning None rather than
    silently falling back makes an unreachable budget visible — with a
    coarse grid that is a property of the grid, not of the detector.
    """
    affordable = [row for row in curve if row["false_alarms_per_day"] <= budget]
    if not affordable:
        return None
    return min(affordable, key=lambda row: row["tau"])["tau"]


def recall_vs_false_alarms(curve: list[dict]) -> list[dict]:
    """The validation recall-vs-false-alarms-per-day curve, for plotting.

    Sorted by alarm rate so it can be drawn directly as an operating
    characteristic.
    """
    points = [
        {
            "tau": row["tau"],
            "lookback_h": row["lookback_h"],
            "false_alarms_per_day": row["false_alarms_per_day"],
            "campaign_recall": row["campaign_recall"],
            "campaign_precision": row["campaign_precision"],
            "campaign_f1": row["campaign_f1"],
            "hits": row["hits"],
            "false_alarms": row["false_alarms"],
        }
        for row in curve
    ]
    return sorted(points, key=lambda row: (row["false_alarms_per_day"], row["tau"]))


@dataclass
class PairedComparison:
    """Lead time compared between two detectors, campaign by campaign."""

    label_a: str
    label_b: str
    split: str
    #: Campaigns both detected — the only ones a lead-time difference is
    #: defined on.
    n_both: int
    n_only_a: int
    n_only_b: int
    n_neither: int
    only_a_campaigns: list[int]
    only_b_campaigns: list[int]
    #: `b - a` over paired campaigns, so **positive means b detected earlier**
    #: (more lead time).
    median_difference_h: float | None
    difference_ci_h: tuple[float | None, float | None]
    median_normalized_difference: float | None
    median_frac_observed_difference: float | None
    #: Sign counts, which do not depend on magnitude.
    n_b_earlier: int
    n_a_earlier: int
    n_tied: int

    def as_dict(self) -> dict:
        data = asdict(self)
        data["ci_low_h"], data["ci_high_h"] = data.pop("difference_ci_h")
        data["only_a_campaigns"] = ",".join(str(c) for c in self.only_a_campaigns)
        data["only_b_campaigns"] = ",".join(str(c) for c in self.only_b_campaigns)
        return data


def paired_lead_time_comparison(
    outcomes_a: list[CampaignOutcome],
    outcomes_b: list[CampaignOutcome],
    label_a: str,
    label_b: str,
    split: str = "",
    bootstrap_seed: int = 0,
    n_resamples: int = 1000,
) -> PairedComparison:
    """Compare two detectors' earliness **on the campaigns both detected**.

    This is the only valid way to compare lead time between two detectors,
    and it is the method to use whenever lookbacks or conditions are
    compared.

    Comparing two medians of `median_lead_time_h` is invalid, because each
    median is taken over a different set of campaigns. A detector that
    finds strictly more campaigns will usually show a *lower* median lead
    time, since the extra ones it reaches are the harder, slower campaigns
    that are detected late — so a genuine improvement reads as a
    regression. The composition of the set changes, not the earliness.

    Two things are therefore reported separately, and neither substitutes
    for the other:

    1. **Earliness**, as the per-campaign difference `b - a` over the
       campaigns both detected, with a bootstrap CI over campaigns. Positive
       means `b` was earlier. Sign counts are given too, since they hold
       regardless of magnitude.
    2. **Coverage**, as how many campaigns each detector found that the
       other missed — the discordant sets, named by id so they can be
       inspected rather than assumed to be random.
    """
    by_id_a = {o.campaign_id: o for o in outcomes_a}
    by_id_b = {o.campaign_id: o for o in outcomes_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    both, only_a, only_b, neither = [], [], [], []
    for campaign_id in shared_ids:
        a, b = by_id_a[campaign_id], by_id_b[campaign_id]
        if a.detected and b.detected:
            both.append((a, b))
        elif a.detected:
            only_a.append(campaign_id)
        elif b.detected:
            only_b.append(campaign_id)
        else:
            neither.append(campaign_id)

    differences = [
        b.lead_time_h - a.lead_time_h
        for a, b in both
        if a.lead_time_h is not None and b.lead_time_h is not None
    ]
    normalized = [
        b.normalized_lead_time - a.normalized_lead_time
        for a, b in both
        if a.normalized_lead_time is not None and b.normalized_lead_time is not None
    ]
    observed = [
        b.frac_observed - a.frac_observed
        for a, b in both
        if a.frac_observed is not None and b.frac_observed is not None
    ]

    return PairedComparison(
        label_a=label_a,
        label_b=label_b,
        split=split,
        n_both=len(both),
        n_only_a=len(only_a),
        n_only_b=len(only_b),
        n_neither=len(neither),
        only_a_campaigns=only_a,
        only_b_campaigns=only_b,
        median_difference_h=float(np.median(differences)) if differences else None,
        difference_ci_h=bootstrap_ci(
            differences, statistic=np.median, seed=bootstrap_seed, n_resamples=n_resamples
        ),
        median_normalized_difference=float(np.median(normalized)) if normalized else None,
        median_frac_observed_difference=float(np.median(observed)) if observed else None,
        n_b_earlier=sum(1 for d in differences if d > 0),
        n_a_earlier=sum(1 for d in differences if d < 0),
        n_tied=sum(1 for d in differences if d == 0),
    )


def describe_paired_comparison(comparison: PairedComparison) -> str:
    """A sentence that states what the paired numbers do and do not show."""
    if comparison.n_both == 0:
        return (
            f"{comparison.label_a} and {comparison.label_b} share no detected "
            "campaigns, so earliness cannot be compared."
        )

    low, high = comparison.difference_ci_h
    median = comparison.median_difference_h
    bounds = (
        f"[{low:.1f}, {high:.1f}]"
        if low is not None and high is not None
        else "[n/a]"
    )
    straddles_zero = low is not None and high is not None and low <= 0 <= high

    verdict = (
        "the interval includes zero, so no earliness difference is demonstrated"
        if straddles_zero
        else f"{comparison.label_b} is "
        + ("earlier" if median and median > 0 else "later")
        + " on the campaigns both detect"
    )
    return (
        f"On the {comparison.n_both} campaigns both detect, "
        f"{comparison.label_b} - {comparison.label_a} = {median:.1f}h "
        f"(95% CI {bounds}); {verdict}. "
        f"Separately, {comparison.label_b} detects {comparison.n_only_b} that "
        f"{comparison.label_a} misses, and {comparison.label_a} detects "
        f"{comparison.n_only_a} that {comparison.label_b} misses."
    )


def summarize_by(result: EvaluationResult, attribute: str) -> list[dict]:
    """Recall and median lead time broken down by `group` or `base_type`."""
    rows = []
    values = sorted({getattr(o, attribute) for o in result.outcomes})
    for value in values:
        outcomes = [o for o in result.outcomes if getattr(o, attribute) == value]
        detected = [o for o in outcomes if o.detected]
        leads = [o.lead_time_h for o in detected if o.lead_time_h is not None]
        rows.append(
            {
                attribute: value,
                "n_campaigns": len(outcomes),
                "n_detected": len(detected),
                "campaign_recall": len(detected) / len(outcomes) if outcomes else None,
                "median_lead_time_h": float(np.median(leads)) if leads else None,
            }
        )
    return rows
