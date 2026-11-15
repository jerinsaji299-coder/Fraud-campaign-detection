"""Stage 4.3: boundary-account evidence.

**What is shared (D3).** A *boundary account* is one with at least one
counterparty in another institution. For its boundary accounts only, an
institution receives the **window node features** computed on the union of the
views that contain that account - degree in and out, counterparty count,
currency count, and the two amount aggregates, i.e. exactly the node features
the spec already has. No labels. No raw transactions. No features for
non-boundary accounts. The feature spec is refitted per institution on its
augmented view, as in 4.1.

**The union of views containing an account is the whole window, for that
account.** This is worth stating because it makes the implementation small
and because it is not obvious. Visibility is an endpoint rule, so an account's
*owner* sees every transaction the account is an endpoint of. The union of the
views that contain account ``a`` therefore includes the owner's view, which
already contains all of ``a``'s incident edges. Every node feature in the spec
is an aggregate over incident edges alone - see
``features._raw_node_features``, where all six columns are per-account
groupbys. So the node features of ``a`` over the union equal the node features
of ``a`` over the full window, exactly, and the shared evidence is computable
in one pass over the window instead of one pass per subset of institutions.
:func:`union_node_features_match_global` is the test of that claim.

A second consequence: for an account an institution **owns**, the augmentation
is a no-op, because the institution already saw all of that account's edges.
The evidence that actually moves is the profile of a boundary *counterparty* -
what institution ``i`` could not see about an account held by someone else.

**What this condition cannot do, by construction.** It shares node features,
not rows. An institution's view holds exactly the transactions it held before,
so the per-institution oracle ceiling is **unchanged** - the ceiling is the
oracle through the same visibility mask, and the mask did not move. 4.3
therefore cannot raise the ceiling; it can only recover more of it. That is a
property of the definition D3 fixes, and it is what keeps every comparison
paired against an identical denominator. A form of exchange that moved
*transactions* across institutions would raise the ceiling, and it is not this
one.

**Two evaluations, by D3.**

* **4.3a**, ``evidence``: each institution scores its own view with its own
  augmented booster. Against 4.1 this isolates the effect of the evidence.
* **4.3b**, ``fedavg_evidence``: each institution scores its own view with all
  K augmented boosters, averaged, under the 4.2 rule. Against 4.2 this
  isolates the effect of the evidence *under federation*.

Both are evaluated by :func:`isolated.run_cell`, so tau selection, the seven
operating points, the grid-edge policy, the ceiling, the pooled means and the
union are the same machinery as every other condition.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from .. import constants, features, windows
from ..models import xgb_baseline
from . import fedavg, isolated, views as fv

#: 4.3a: isolated scoring on augmented features.
CONDITION_EVIDENCE = "evidence"
#: 4.3b: score-averaged scoring on augmented features.
CONDITION_FEDAVG_EVIDENCE = "fedavg_evidence"


# ---------------------------------------------------------------------------
# the evidence itself
# ---------------------------------------------------------------------------


class BoundaryEvidence:
    """The shared node features for one ``(target, seed, lookback)`` cell.

    Holds the boundary-account set and, per detection window, the node
    features computed on the full window. Both are properties of the cell
    rather than of an institution, so they are computed once and every
    institution's override reads the same numbers - which is also what makes
    "the same evidence was shared with everyone" true by construction rather
    than by convention.

    ``max_cached_windows`` bounds the memory. A cached window costs
    ``n_accounts x 6`` float64, which is tens of megabytes on the full data;
    the cache is shared across the eight institutions, which is where its
    value is, so a cap below the number of windows in a split costs
    recomputation. ``None`` means unbounded.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        assignment: fv.VisibilityAssignment,
        lookback_h: int = constants.PHASE4_LOOKBACK_H,
        max_cached_windows: int | None = None,
    ):
        self._full_df = full_df
        self._assignment = assignment
        self._lookback_h = int(lookback_h)
        self._max_cached = max_cached_windows
        self._cache: OrderedDict = OrderedDict()
        self._boundary: frozenset[str] | None = None
        self.hits = 0
        self.misses = 0
        self.seconds = 0.0

    # --- the boundary set ---------------------------------------------
    @property
    def boundary_accounts(self) -> frozenset[str]:
        """Accounts with at least one counterparty in another institution.

        D3's definition, computed once over the whole frame.
        ``views.boundary_accounts`` returns the accounts visible to more than
        one institution, which is the same set: an account is visible to a
        second institution exactly when one of its counterparties is held
        there.
        """
        if self._boundary is None:
            self._boundary = frozenset(
                fv.boundary_accounts(self._full_df, self._assignment)
            )
        return self._boundary

    # --- the shared node features -------------------------------------
    def global_nodes(self, t, lookback_h: int | None = None):
        """``(index, matrix, names)`` of node features over the full window."""
        lookback_h = self._lookback_h if lookback_h is None else int(lookback_h)
        key = (pd.Timestamp(t), lookback_h)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return cached

        started = time.perf_counter()
        txns = windows.window_slice(self._full_df, key[0], lookback_h)
        accounts, matrix, names = features._raw_node_features(txns)
        # float64, NOT float32. The substitution must be an exact no-op for
        # an account whose owner already saw every one of its edges, and
        # narrowing here would shift `log_in_amount` and `log_out_amount` by
        # ~1e-7 on EVERY account - so 4.3a would differ from 4.1 by float
        # noise in every node column instead of only where the evidence is
        # genuinely new. `_apply_scaler` narrows to float32 at the end anyway,
        # after the arithmetic, exactly as it does for the un-augmented path.
        value = (pd.Index(accounts), matrix, names)
        self.misses += 1
        self.seconds += time.perf_counter() - started

        self._cache[key] = value
        if self._max_cached is not None:
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        return value

    # --- the hook features.py takes -----------------------------------
    def override_for(self, institution: int):
        """A ``node_override(window, accounts, matrix, names)`` for one institution.

        Substitutes the shared node features for the boundary accounts present
        in this window's view and leaves every other row untouched. Returns a
        copy, because the caller's matrix is the un-augmented one and other
        code may still be holding it.
        """

        def override(window, accounts, matrix, names):
            boundary = self.boundary_accounts
            rows = [k for k, account in enumerate(accounts) if account in boundary]
            if not rows:
                return matrix
            index, shared, shared_names = self.global_nodes(
                window.t, window.lookback_h
            )
            if list(shared_names) != list(names):
                raise ValueError(
                    "shared node features have different columns than the "
                    f"view's: {shared_names} vs {names}"
                )
            wanted = [accounts[k] for k in rows]
            positions = index.get_indexer(wanted)
            if (positions < 0).any():
                missing = [a for a, p in zip(wanted, positions) if p < 0]
                raise ValueError(
                    "boundary accounts present in a view but absent from the "
                    f"full window at {window.t}: {missing[:5]}"
                )
            out = np.array(matrix, dtype=matrix.dtype, copy=True)
            out[rows] = shared[positions].astype(matrix.dtype, copy=False)
            return out

        return override

    def stats(self) -> dict:
        return {
            "n_boundary_accounts": len(self.boundary_accounts),
            "windows_cached": len(self._cache),
            "global_node_hits": self.hits,
            "global_node_misses": self.misses,
            "global_node_seconds": round(self.seconds, 2),
            "max_cached_windows": self._max_cached,
        }


def union_node_features_match_global(
    full_df: pd.DataFrame,
    assignment: fv.VisibilityAssignment,
    window,
    account: str,
) -> tuple[np.ndarray, np.ndarray]:
    """The claim in this module's docstring, computed both ways.

    Returns ``(over_the_union_of_views, over_the_full_window)`` for one
    account in one window. The first is built the long way - take every
    institution that can see the account, union their views, featurise that -
    so a test can assert the two are identical rather than taking the
    reasoning on trust.
    """
    txns = window.transactions
    src, dst = fv.endpoint_institutions(txns, assignment)
    incident = (txns["src"].to_numpy() == account) | (txns["dst"].to_numpy() == account)
    holders = sorted(set(src[incident].tolist()) | set(dst[incident].tolist()))

    mask = np.zeros(len(txns), dtype=bool)
    for institution in holders:
        mask |= (src == institution) | (dst == institution)
    union_txns = txns[mask]

    def row_for(frame):
        accounts, matrix, _ = features._raw_node_features(frame)
        return matrix[accounts.index(account)]

    return row_for(union_txns), row_for(txns)


# ---------------------------------------------------------------------------
# training on an augmented view
# ---------------------------------------------------------------------------


def train_evidence(
    full_df: pd.DataFrame,
    institution: int,
    assignment: fv.VisibilityAssignment,
    evidence: BoundaryEvidence,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    params: xgb_baseline.XGBParams | None = None,
) -> isolated.IsolatedModel:
    """One institution's booster, trained on its **augmented** view.

    The same function as :func:`isolated.train_isolated` in every respect
    except the node override, which is threaded into both the feature spec and
    the pooled matrix - so the spec is fitted on augmented values and the model
    is trained on the same ones it will be scored on. The rows are still only
    this institution's: evidence changes what the model knows about a row, not
    which rows it has.
    """
    started = time.perf_counter()
    view = fv.visible_transactions(full_df, institution, assignment)
    override = evidence.override_for(institution)

    if len(view) == 0:
        return isolated.IsolatedModel(
            institution=institution, lookback_h=lookback_h,
            tier=isolated.TIER_UNTRAINED, n_rows=0, n_positive=0,
            training={"reason": "the institution sees no transactions at all"},
            seconds=time.perf_counter() - started,
        )

    try:
        data = xgb_baseline.build_training_data(
            view, lookback_h, node_override=override
        )
    except ValueError as exc:
        return isolated.IsolatedModel(
            institution=institution, lookback_h=lookback_h,
            tier=isolated.TIER_UNTRAINED, n_rows=0, n_positive=0,
            training={"reason": str(exc)},
            seconds=time.perf_counter() - started,
        )

    tier = isolated.training_tier(data.n_positive)
    weights = [w["class_weight"] for w in data.per_window]
    summary = {
        "n_windows": len(data.times),
        "n_rows": data.n_rows,
        "n_positive": data.n_positive,
        "positive_rate": round(data.positive_rate, 6),
        "class_weight_min": min(weights) if weights else None,
        "class_weight_max": max(weights) if weights else None,
        "per_window": data.per_window,
        "evidence": evidence.stats(),
    }

    if tier == isolated.TIER_UNTRAINED:
        summary["reason"] = "no positive rows in the pooled training matrix"
        return isolated.IsolatedModel(
            institution=institution, lookback_h=lookback_h, tier=tier,
            n_rows=data.n_rows, n_positive=data.n_positive, training=summary,
            seconds=time.perf_counter() - started,
        )

    model = xgb_baseline.train_baseline(
        view, lookback_h=lookback_h, params=params, data=data,
        variant="xgboost_centralized",
    )
    summary["xgboost"] = {
        k: v for k, v in model.training.items() if k != "per_window"
    }
    return isolated.IsolatedModel(
        institution=institution, lookback_h=lookback_h, tier=tier,
        n_rows=data.n_rows, n_positive=data.n_positive, model=model,
        training=summary, seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------


def evidence_scorer(
    model: isolated.IsolatedModel,
    institution: int,
    assignment: fv.VisibilityAssignment,
    evidence: BoundaryEvidence,
):
    """4.3a: this institution's own augmented booster on its own view."""
    if not model.trained:
        return isolated.zero_scorer()
    override = evidence.override_for(institution)

    def inner(visible: pd.DataFrame, t) -> np.ndarray:
        return model.model.predict_window(visible, t, node_override=override)

    return isolated.masked_scorer(inner, institution, assignment)


def averaged_evidence_scorer(
    models: dict[int, isolated.IsolatedModel],
    institution: int,
    assignment: fv.VisibilityAssignment,
    evidence: BoundaryEvidence,
):
    """4.3b: every augmented booster on this institution's view, averaged.

    The 4.2 rule from D2 - equal weights, untrained institutions left out of
    the mean rather than entered at 0 - applied to the augmented boosters, and
    with the augmented featurisation. The override belongs to the institution
    whose view is being scored, not to the model doing the scoring: the
    evidence is what *this* institution received.
    """
    boosters = [models[i].model for i in fedavg.participating(models)]
    if not boosters:
        return isolated.zero_scorer()
    override = evidence.override_for(institution)

    def inner(visible: pd.DataFrame, t) -> np.ndarray:
        per_model = xgb_baseline.predict_window_multi(
            boosters, visible, t, node_override=override
        )
        return per_model.mean(axis=0)

    return isolated.masked_scorer(inner, institution, assignment)


# ---------------------------------------------------------------------------
# the runner: one provider shared by training and both evaluations
# ---------------------------------------------------------------------------


class EvidenceRunner:
    """Holds the one :class:`BoundaryEvidence` a cell shares.

    The provider has to be the same object for training, for 4.3a and for
    4.3b: it caches the shared node features across institutions, which is
    where nearly all of its value is, and it guarantees that all three used
    identical evidence rather than three independently computed copies of it.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        lookback_h: int = constants.PHASE4_LOOKBACK_H,
        max_cached_windows: int | None = None,
    ):
        self.full_df = full_df
        self.lookback_h = int(lookback_h)
        self.max_cached_windows = max_cached_windows
        self._provider: BoundaryEvidence | None = None

    def provider(self, assignment: fv.VisibilityAssignment) -> BoundaryEvidence:
        if self._provider is None:
            self._provider = BoundaryEvidence(
                self.full_df, assignment, self.lookback_h, self.max_cached_windows
            )
        elif self._provider._assignment is not assignment:
            # One runner is one cell. A different assignment means a different
            # (target, seed), which has a different boundary set and different
            # shared features, so reusing this cache would score one cell with
            # another cell's evidence.
            raise RuntimeError(
                "this EvidenceRunner already holds evidence for "
                f"(target={self._provider._assignment.target}, "
                f"seed={self._provider._assignment.seed}); build a new runner "
                "per cell"
            )
        return self._provider

    # --- hooks isolated.run_cell takes --------------------------------
    def trainer(self, full_df, assignment, lookback_h, params, progress=None):
        evidence = self.provider(assignment)
        say = progress or (lambda *_: None)
        say("    boundary accounts: %d" % len(evidence.boundary_accounts))
        models: dict[int, isolated.IsolatedModel] = {}
        for institution in range(assignment.k):
            model = train_evidence(
                full_df, institution, assignment, evidence, lookback_h, params
            )
            models[institution] = model
            say(
                f"    institution {institution}: {model.n_rows:,} rows, "
                f"{model.n_positive:,} positives, {model.tier}, "
                f"{model.seconds:.1f}s"
            )
        return models

    def scorer_for_a(self, institution, models, assignment):
        return evidence_scorer(
            models[institution], institution, assignment, self.provider(assignment)
        )

    def scorer_for_b(self, institution, models, assignment):
        return averaged_evidence_scorer(
            models, institution, assignment, self.provider(assignment)
        )


def run_cell(
    result,
    target: float,
    seed: int,
    condition: str,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    params: xgb_baseline.XGBParams | None = None,
    progress=None,
    models: dict[int, isolated.IsolatedModel] | None = None,
    runner: "EvidenceRunner | None" = None,
) -> dict:
    """Stage 4.3a or 4.3b for one ``(target, seed, lookback)``.

    ``condition`` picks the scoring rule: :data:`CONDITION_EVIDENCE` for
    4.3a, :data:`CONDITION_FEDAVG_EVIDENCE` for 4.3b. Pass the same
    ``runner`` and the same ``models`` to both, so the two differ in the
    scoring rule and in nothing else - which is what makes 4.3b - 4.3a and
    4.3a - 4.1 separable.
    """
    if condition not in (CONDITION_EVIDENCE, CONDITION_FEDAVG_EVIDENCE):
        raise ValueError(f"unknown evidence condition {condition!r}")
    runner = runner or EvidenceRunner(result.full_df, lookback_h)
    scorer_for = (
        runner.scorer_for_a
        if condition == CONDITION_EVIDENCE
        else runner.scorer_for_b
    )
    return isolated.run_cell(
        result,
        target=target,
        seed=seed,
        lookback_h=lookback_h,
        splits=splits,
        params=params,
        progress=progress,
        models=models,
        condition=condition,
        scorer_for=scorer_for,
        trainer=runner.trainer,
    )


__all__ = [
    "CONDITION_EVIDENCE",
    "CONDITION_FEDAVG_EVIDENCE",
    "BoundaryEvidence",
    "EvidenceRunner",
    "averaged_evidence_scorer",
    "evidence_scorer",
    "run_cell",
    "train_evidence",
    "union_node_features_match_global",
]
