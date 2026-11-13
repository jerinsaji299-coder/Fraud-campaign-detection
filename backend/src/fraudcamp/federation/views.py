"""Per-institution views of the transaction data (Phase 4, stage 4.0).

One institution's *view* is the subset of transactions it can see: those
where it holds the sender's account or the receiver's account, under a given
``(seed, target)`` visibility assignment from Phase 2. Everything a federated
condition is allowed to know about an institution is derived from its view
and nothing else.

Three things this module is careful about:

1. **One definition of "sees".** The rule is the frozen one — an institution
   sees a transaction if the sender's *or* the receiver's account belongs to
   it — and it is implemented once, in :func:`visible_mask`. Both endpoints
   see a cross-institution transaction, which is exactly why the achievable
   visibility floor is about ``2/K`` rather than ``1/K``.

2. **No duplicated feature code.** :func:`institution_view` returns a
   :class:`~fraudcamp.windows.Window`, so
   ``features.build_window_features(view, spec)`` works on it unchanged and
   computes that institution's account features *from its own view alone*.
   There is no second featuriser to keep honest.

3. **No identity leaks in, either.** A view is a row subset. Which
   institution produced it is never written into the data, so a model
   trained on a view cannot learn "I am institution 3" — the Phase 3 feature
   rules carry over untouched.

Determinism: nothing here is random. The assignment is supplied by the
caller, institutions are enumerated ``range(k)``, and accounts are sorted
wherever order is observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import constants, institutions as institutions_mod, visibility, windows


@dataclass(frozen=True)
class VisibilityAssignment:
    """The account -> institution map for one ``(seed, target)``.

    ``global_assignment`` holds only the accounts the visibility search
    actually moved; every other account falls back to its natural,
    bank-derived institution. That fallback is the same function the
    pipeline and the exporter use, so a view cannot disagree with the
    ``achieved_visibility`` recorded in the artifacts.
    """

    seed: int
    target: float
    global_assignment: dict[str, int]
    bank2inst: dict[int, int]
    k: int = constants.K_INSTITUTIONS

    def institution_of(self, account: str) -> int:
        return visibility.institution_of(account, self.global_assignment, self.bank2inst)

    def account_institutions(self, accounts) -> dict[str, int]:
        """Institution for each account, looked up once per *distinct*
        account rather than once per row."""
        return {a: self.institution_of(a) for a in accounts}

    @property
    def n_reassigned(self) -> int:
        return len(self.global_assignment)


def assignment_from_pipeline(
    result,
    target: float,
    seed: int,
    k: int = constants.K_INSTITUTIONS,
) -> tuple[VisibilityAssignment, dict[int, float]]:
    """Build the assignment for one ``(seed, target)`` from a
    :class:`~fraudcamp.pipeline.PipelineResult`.

    Returns the assignment and the achieved visibility per campaign, both
    straight from ``pipeline.run_visibility`` — this module never
    re-implements the search.
    """
    from .. import pipeline as pipeline_mod

    ordered = pipeline_mod.ordered_reassignable(
        result.camp, result.edges_by_campaign, result.accounts_by_campaign
    )
    global_assignment, achieved = pipeline_mod.run_visibility(
        ordered, result.edges_by_campaign, result.bank2inst, target, seed
    )
    return (
        VisibilityAssignment(
            seed=seed,
            target=target,
            global_assignment=global_assignment,
            bank2inst=result.bank2inst,
            k=k,
        ),
        achieved,
    )


# ---------------------------------------------------------------------------
# the view itself
# ---------------------------------------------------------------------------


def endpoint_institutions(
    df: pd.DataFrame, assignment: VisibilityAssignment
) -> tuple[np.ndarray, np.ndarray]:
    """``(src_institution, dst_institution)`` for every row of ``df``."""
    if len(df) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    accounts = pd.unique(np.concatenate([df["src"].to_numpy(), df["dst"].to_numpy()]))
    lookup = assignment.account_institutions(accounts)
    src = df["src"].map(lookup).to_numpy(dtype=np.int64)
    dst = df["dst"].map(lookup).to_numpy(dtype=np.int64)
    return src, dst


def visible_mask(
    df: pd.DataFrame, institution: int, assignment: VisibilityAssignment
) -> np.ndarray:
    """Boolean mask of the rows of ``df`` that ``institution`` can see.

    The single implementation of the frozen rule: sender's institution or
    receiver's institution. Nothing else in Phase 4 decides visibility.
    """
    src, dst = endpoint_institutions(df, assignment)
    return (src == institution) | (dst == institution)


def visible_transactions(
    df: pd.DataFrame, institution: int, assignment: VisibilityAssignment
) -> pd.DataFrame:
    """The subset of ``df`` that ``institution`` can see."""
    return df[visible_mask(df, institution, assignment)]


def institution_view(window: windows.Window, institution: int, assignment: VisibilityAssignment) -> windows.Window:
    """One institution's view of one detection window, as a ``Window``.

    Returning a ``Window`` is the point: every Phase 3 consumer —
    ``features.build_window_features`` above all — accepts it unchanged, so
    the institution's account features are computed from its own view by
    exactly the code that computes them centrally.
    """
    return windows.Window(
        t=window.t,
        lookback_h=window.lookback_h,
        split=window.split,
        transactions=visible_transactions(window.transactions, institution, assignment),
    )


def all_views(
    window: windows.Window, assignment: VisibilityAssignment
) -> dict[int, windows.Window]:
    """Every institution's view of one window, keyed by institution.

    Computed from a single endpoint-institution pass over the window rather
    than ``k`` passes, which matters on million-row windows.
    """
    src, dst = endpoint_institutions(window.transactions, assignment)
    views: dict[int, windows.Window] = {}
    for inst in range(assignment.k):
        mask = (src == inst) | (dst == inst)
        views[inst] = windows.Window(
            t=window.t,
            lookback_h=window.lookback_h,
            split=window.split,
            transactions=window.transactions[mask],
        )
    return views


def boundary_accounts(
    df: pd.DataFrame, assignment: VisibilityAssignment
) -> dict[str, tuple[int, ...]]:
    """Accounts that appear in more than one institution's view.

    An account belongs to exactly one institution, but it is *visible* to
    every institution it transacts with. These are the accounts whose
    window features differ between the views that hold them — the ones
    stage 4.1's exchange condition has to reason about. Returns, per
    account, the sorted institutions that can see it at all.
    """
    if len(df) == 0:
        return {}
    src, dst = endpoint_institutions(df, assignment)
    seen: dict[str, set[int]] = {}
    for account, own, other in zip(df["src"].to_numpy(), src, dst):
        seen.setdefault(account, set()).update((int(own), int(other)))
    for account, own, other in zip(df["dst"].to_numpy(), dst, src):
        seen.setdefault(account, set()).update((int(own), int(other)))
    return {a: tuple(sorted(v)) for a, v in sorted(seen.items()) if len(v) > 1}


# ---------------------------------------------------------------------------
# checks that the views agree with the recorded visibility
# ---------------------------------------------------------------------------


def campaign_visibility_from_views(
    df: pd.DataFrame, campaign_id: int, assignment: VisibilityAssignment
) -> float:
    """Achieved visibility of one campaign, computed *from the views*.

    Deliberately a second, independent path to a number the pipeline
    already produces: ``visibility.recompute_achieved`` works from the
    campaign's pattern edge list, this counts rows of the tagged
    transaction table that land in each institution's view. They must
    agree, and :mod:`tests.test_federation_views` requires it.
    """
    rows = df[df["campaign_id"] == campaign_id]
    if len(rows) == 0:
        return 0.0
    src, dst = endpoint_institutions(rows, assignment)
    best = 0.0
    for inst in range(assignment.k):
        frac = float(((src == inst) | (dst == inst)).sum()) / len(rows)
        if frac > best:
            best = frac
    return best


def campaign_visibility_table(
    df: pd.DataFrame, assignment: VisibilityAssignment, campaign_ids=None
) -> pd.DataFrame:
    """``campaign_visibility_from_views`` for many campaigns at once."""
    if campaign_ids is None:
        campaign_ids = sorted(int(c) for c in pd.unique(df["campaign_id"]) if int(c) >= 0)
    src, dst = endpoint_institutions(df, assignment)
    cid = df["campaign_id"].to_numpy(dtype=np.int64)
    rows = []
    for campaign_id in campaign_ids:
        in_campaign = cid == campaign_id
        n = int(in_campaign.sum())
        if n == 0:
            rows.append({"campaign_id": campaign_id, "n_txn": 0, "visibility_from_views": 0.0})
            continue
        best = max(
            float(((src[in_campaign] == inst) | (dst[in_campaign] == inst)).sum()) / n
            for inst in range(assignment.k)
        )
        rows.append(
            {
                "campaign_id": campaign_id,
                "n_txn": n,
                "visibility_from_views": best,
                "bin_from_views": visibility.bin_of(best),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the stage 4.0 report: what each institution actually has to train on
# ---------------------------------------------------------------------------


@dataclass
class ViewLoadReport:
    """Per-institution training-set sizes for one ``(seed, target)``."""

    seed: int
    target: float
    lookback_h: int
    n_windows: int
    per_institution: pd.DataFrame = field(default_factory=pd.DataFrame)
    central: dict = field(default_factory=dict)

    @property
    def starved(self) -> pd.DataFrame:
        """Institutions with too few positives to train on at all."""
        return self.per_institution[self.per_institution["positives"] < MIN_POSITIVES_TO_TRAIN]


#: Below this many positive rows in its pooled training matrix, an
#: institution cannot be said to have trained on the positive class at all.
#: Not a frozen research definition — a reporting threshold for stage 4.0,
#: and the number stage 4.1 has to decide what to *do* about.
MIN_POSITIVES_TO_TRAIN = 10


def view_load_report(
    training_windows,
    assignment: VisibilityAssignment,
    lookback_h: int,
) -> ViewLoadReport:
    """How many training rows and positives each institution has.

    ``training_windows`` must already be the rule-3-valid training windows
    for ``lookback_h`` — this function does not choose them, so it cannot
    quietly disagree with stage 3.3 about which windows training uses.

    Counts are **pooled over windows**, which is what a model actually sees
    (stage 3.3 pools every training window into one matrix). Windows at a
    6-hour stride with a 24h or 48h lookback overlap heavily, so a
    transaction is counted once per window it falls in; ``distinct_rows``
    reports the deduplicated figure beside it.
    """
    rows = []
    central_rows = 0
    central_pos = 0
    for window in training_windows:
        central_rows += len(window.transactions)
        central_pos += int(window.transactions["Is Laundering"].sum())

    for inst in range(assignment.k):
        pooled_rows = 0
        pooled_pos = 0
        distinct: set = set()
        per_window = []
        for window in training_windows:
            view = institution_view(window, inst, assignment)
            txns = view.transactions
            n = len(txns)
            pos = int(txns["Is Laundering"].sum())
            pooled_rows += n
            pooled_pos += pos
            distinct.update(txns.index.to_numpy().tolist())
            per_window.append(pos)
        rows.append(
            {
                "institution": inst,
                "rows": pooled_rows,
                "positives": pooled_pos,
                "positive_rate": (pooled_pos / pooled_rows) if pooled_rows else 0.0,
                "class_weight": ((pooled_rows - pooled_pos) / pooled_pos) if pooled_pos else float("nan"),
                "distinct_rows": len(distinct),
                "share_of_central_rows": (pooled_rows / central_rows) if central_rows else 0.0,
                "share_of_central_positives": (pooled_pos / central_pos) if central_pos else 0.0,
                "windows_with_no_positive": sum(1 for p in per_window if p == 0),
            }
        )

    return ViewLoadReport(
        seed=assignment.seed,
        target=assignment.target,
        lookback_h=lookback_h,
        n_windows=len(training_windows),
        per_institution=pd.DataFrame(rows),
        central={
            "rows": central_rows,
            "positives": central_pos,
            "positive_rate": (central_pos / central_rows) if central_rows else 0.0,
            "class_weight": ((central_rows - central_pos) / central_pos) if central_pos else float("nan"),
        },
    )


def natural_assignment(result, k: int = constants.K_INSTITUTIONS) -> VisibilityAssignment:
    """The untouched, bank-derived split: no account reassigned at all.

    This is the control the visibility sweep is read against, and it is what
    the ``100`` bin approximates. Recorded with ``target = 1.0`` and
    ``seed = -1`` so it can never be confused with an experiment run.
    """
    return VisibilityAssignment(
        seed=-1,
        target=1.0,
        global_assignment={},
        bank2inst=result.bank2inst,
        k=k,
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
