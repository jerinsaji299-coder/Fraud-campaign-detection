"""Stage 4.1: the ``isolated`` condition — each institution alone.

Trained on its own visible transactions, scoring its own visible
transactions, forming clusters inside its own view. No model averaging, no
boundary-account sharing, no coordination. This is the floor ``fedavg`` and
``fedavg_exchange`` have to beat.

**An institution is restricted by its SCORER, not by its data.** The
evaluation keeps the global transactions, campaigns, deadlines, detection
grid and horizons — exactly the objects Phase 3 passed — and institution
*i*'s scorer returns 0 outside :func:`views.visible_mask` and its booster's
probability inside, with features built from :func:`views.institution_view`
alone. Since the smallest model tau is 0.05 > 0, an invisible edge can never
be kept, so a cluster can only form inside the view, which is the definition
of the condition.

The rejected alternative was to pass the institution's view as ``full_df``.
It restricts the right rows for the wrong reason: ``evaluate_split`` derives
the unassigned-laundering account set — the set that makes a cluster
``ambiguous`` rather than a ``false_alarm`` — from that argument, so
filtering it would stop laundering the institution cannot see from counting
as laundering, silently reclassifying ambiguous clusters as false alarms and
moving every tau chosen against a budget.

*Visibility restricts what can be scored. It never restricts what is true.*

Training is different, and the asymmetry is deliberate: there the view **is**
passed as the frame, because training labels are per-row ``Is Laundering``
and no cluster, purity or ambiguity logic is involved. Filtering rows commutes
with slicing a window by time, so pooling the training windows of the filtered
frame gives exactly the per-window views, and the feature spec is fitted on
them — an institution standardises with its own statistics, which is the
honest reading of "alone".

Two pooled numbers come out of a cell, and they are never averaged together:

* the **per-institution mean**, reported twice — over all K institutions and
  over trained institutions only, because one mean conflates "cannot see the
  campaign" with "had no labels to learn from";
* the **union**, where a campaign counts as discovered if any institution
  discovers it alone and the detection time is the earliest across
  institutions. The union is a **supervisory** number — a regulator or FIU
  receiving all K alert streams — not a bank's: no single bank can compute
  it, see the other streams, or benefit from the deduplication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import constants, evaluation, windows
from ..evaluation import clusters as cluster_tools
from ..evaluation import evaluate as evaluate_mod
from ..models import xgb_baseline
from . import views as fv

#: Training tiers. ``MIN_POSITIVES_TO_TRAIN`` is a reporting threshold and
#: never a gate: an institution with 1-9 positives still trains and is
#: flagged, because refusing at 9 and training at 10 would delete a real
#: outcome on an arbitrary boundary. Only 0 positives cannot train, since
#: XGBoost needs both classes present.
TIER_UNTRAINED = "untrained"
TIER_THIN = "thin"
TIER_NORMAL = "normal"
TIERS = (TIER_UNTRAINED, TIER_THIN, TIER_NORMAL)

#: The condition's name, as `constants.CONDITIONS` and the frontend's
#: condition keys already spell it.
CONDITION = "isolated"


def training_tier(n_positive: int) -> str:
    """Which tier a pooled training matrix with this many positives is in."""
    if n_positive <= 0:
        return TIER_UNTRAINED
    if n_positive < fv.MIN_POSITIVES_TO_TRAIN:
        return TIER_THIN
    return TIER_NORMAL


#: Why an operating point has no tau. Both are **undefined** — excluded from
#: the pooled means and listed — and never recall 0, because a detector that
#: cannot be operated at a point was not measured at that point.
NO_AFFORDABLE_TAU = "no tau on the grid meets this budget"
TAU_BEYOND_GRID = "the chosen tau is at the top of the grid: the optimum lies beyond it"


def tau_grid_edge(tau: float | None, tau_grid=None) -> str | None:
    """``"low"``, ``"high"`` or ``None`` for a tau at an edge of the grid.

    The direction matters because the two mean opposite things. **High** is
    the stage 3.3 phenomenon: per-window class weights of 495-3,461 crush
    scores against 1, and a chosen tau at the top says the optimum lies
    beyond the grid. **Low** is the reverse — the most sensitive threshold on
    the grid is still the best one, so the optimum may lie below it. On the 1%
    sample a low edge is expected, because the thinned background leaves a
    ~3.3% positive rate against the full data's 0.102%; on the full data it is
    not, and the runner stops rather than recording it.
    """
    if tau is None:
        return None
    grid = tuple(tau_grid if tau_grid is not None else evaluation.MODEL_TAU_GRID)
    if not grid:
        return None
    if float(tau) <= min(grid):
        return "low"
    if float(tau) >= max(grid):
        return "high"
    return None


# ---------------------------------------------------------------------------
# scorers: the one place an institution's visibility is applied
# ---------------------------------------------------------------------------


def masked_scorer(inner, institution: int, assignment: fv.VisibilityAssignment):
    """Wrap ``inner`` so it only ever sees one institution's rows.

    ``inner`` is any ``scorer(window_transactions, t)``. The wrapper hands it
    the visible subset — nothing else — and scatters the result back into a
    zero-filled array in the **global window's own row order**, which is the
    order the cluster extraction pairs with ``src``/``dst``. The mask is a
    positional boolean over those rows, so the scatter needs no join and
    cannot misalign.
    """

    def score(window_transactions: pd.DataFrame, t) -> np.ndarray:
        n = len(window_transactions)
        scores = np.zeros(n, dtype=np.float64)
        if n == 0:
            return scores
        mask = fv.visible_mask(window_transactions, institution, assignment)
        if not mask.any():
            return scores
        visible = window_transactions[mask]
        inner_scores = np.asarray(inner(visible, t), dtype=np.float64)
        if len(inner_scores) != int(mask.sum()):
            raise ValueError(
                f"inner scorer returned {len(inner_scores)} scores for "
                f"{int(mask.sum())} visible rows at {t}"
            )
        scores[mask] = inner_scores
        return scores

    return score


def zero_scorer():
    """Scores nothing above any tau: what an untrained institution does.

    Used only in tests to show that skipping an untrained institution's
    evaluation is equivalent to running it, never in a run.
    """

    def score(window_transactions: pd.DataFrame, t) -> np.ndarray:
        return np.zeros(len(window_transactions), dtype=np.float64)

    return score


def oracle_view_scorer(institution: int, assignment: fv.VisibilityAssignment):
    """The oracle restricted to one institution's view.

    This is how a per-institution **ceiling** is computed: the same oracle
    the centralized ceiling uses, through the same mask the model goes
    through, so the ceiling and the model differ in the scores and in nothing
    else. The centralized ceiling (0.821 at L = 24, 0.893 at L = 48) is a
    property of the centralized view and is never reused for an
    institution's number.
    """

    def score(window_transactions: pd.DataFrame, t) -> np.ndarray:
        n = len(window_transactions)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        labels = evaluation.oracle_scores(
            window_transactions["Is Laundering"].to_numpy()
        )
        mask = fv.visible_mask(window_transactions, institution, assignment)
        return np.where(mask, labels, 0.0)

    return score


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


@dataclass
class IsolatedModel:
    """One institution's booster, or the record of why there isn't one."""

    institution: int
    lookback_h: int
    tier: str
    n_rows: int
    n_positive: int
    model: xgb_baseline.TrainedBaseline | None = None
    training: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def trained(self) -> bool:
        return self.model is not None

    def scorer(self, assignment: fv.VisibilityAssignment):
        """This institution's masked scorer, or an all-zero one if untrained."""
        if self.model is None:
            return zero_scorer()
        return masked_scorer(self.model.scorer(), self.institution, assignment)

    def as_row(self) -> dict:
        return {
            "institution": self.institution,
            "lookback_h": self.lookback_h,
            "tier": self.tier,
            "trained": self.trained,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "positive_rate": (self.n_positive / self.n_rows) if self.n_rows else 0.0,
            "class_weight_min": self.training.get("class_weight_min"),
            "class_weight_max": self.training.get("class_weight_max"),
            "n_windows": self.training.get("n_windows"),
            "train_seconds": round(self.seconds, 2),
        }


def train_isolated(
    full_df: pd.DataFrame,
    institution: int,
    assignment: fv.VisibilityAssignment,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    params: xgb_baseline.XGBParams | None = None,
) -> IsolatedModel:
    """Train one institution on its own view of the training windows.

    The view is passed as the frame, which is safe here and not at evaluation
    time (see the module docstring): filtering rows commutes with slicing by
    time, so ``build_training_data`` pools exactly this institution's views,
    and ``fit_feature_spec`` fits on those same views — an institution
    standardises with its own statistics.

    An institution with no positives cannot train and is returned as
    ``TIER_UNTRAINED`` rather than raising, because that is a measurable
    outcome of the condition and the run must carry it, not stop on it.
    """
    started = time.perf_counter()
    view = fv.visible_transactions(full_df, institution, assignment)

    if len(view) == 0:
        return IsolatedModel(
            institution=institution, lookback_h=lookback_h, tier=TIER_UNTRAINED,
            n_rows=0, n_positive=0,
            training={"reason": "the institution sees no transactions at all"},
            seconds=time.perf_counter() - started,
        )

    try:
        data = xgb_baseline.build_training_data(view, lookback_h)
    except ValueError as exc:
        return IsolatedModel(
            institution=institution, lookback_h=lookback_h, tier=TIER_UNTRAINED,
            n_rows=0, n_positive=0, training={"reason": str(exc)},
            seconds=time.perf_counter() - started,
        )

    tier = training_tier(data.n_positive)
    weights = [w["class_weight"] for w in data.per_window]
    summary = {
        "n_windows": len(data.times),
        "n_rows": data.n_rows,
        "n_positive": data.n_positive,
        "positive_rate": round(data.positive_rate, 6),
        "class_weight_min": min(weights) if weights else None,
        "class_weight_max": max(weights) if weights else None,
        "per_window": data.per_window,
    }

    if tier == TIER_UNTRAINED:
        summary["reason"] = "no positive rows in the pooled training matrix"
        return IsolatedModel(
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
    return IsolatedModel(
        institution=institution, lookback_h=lookback_h, tier=tier,
        n_rows=data.n_rows, n_positive=data.n_positive, model=model,
        training=summary, seconds=time.perf_counter() - started,
    )


def train_all_isolated(
    full_df: pd.DataFrame,
    assignment: fv.VisibilityAssignment,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    params: xgb_baseline.XGBParams | None = None,
    progress=None,
) -> dict[int, IsolatedModel]:
    """One model per institution, in institution order. Never pooled."""
    models: dict[int, IsolatedModel] = {}
    for institution in range(assignment.k):
        model = train_isolated(full_df, institution, assignment, lookback_h, params)
        models[institution] = model
        if progress:
            progress(
                f"    institution {institution}: {model.n_rows:,} rows, "
                f"{model.n_positive:,} positives, {model.tier}, "
                f"{model.seconds:.1f}s"
            )
    return models


# ---------------------------------------------------------------------------
# the union over institutions
# ---------------------------------------------------------------------------


def undetected_outcomes(camp: pd.DataFrame, split: str) -> list[evaluate_mod.CampaignOutcome]:
    """The outcome list of a detector that detects nothing on this split.

    What an untrained institution produces, built directly instead of by
    running the detection loop with an all-zero scorer. A test pins that the
    two agree, so this is a shortcut and not a second definition.
    """
    eligible = camp[(camp["eval_ok"]) & (camp["split"] == split)]
    return [
        evaluate_mod.CampaignOutcome(
            campaign_id=int(row.campaign_id), split=row.split,
            base_type=row.base_type, group=row.group, detected=False,
            detection_time=None, lead_time_h=None, normalized_lead_time=None,
            frac_observed=None, crosses_cutoff=bool(row.crosses_cutoff),
            shares_train_account=bool(row.shares_train_account),
            deadline=pd.Timestamp(row.deadline), end=pd.Timestamp(row.end),
            duration_h=float(row.duration_h), n_txn=int(row.n_txn),
        )
        for row in eligible.itertuples()
    ]


def union_outcomes(
    per_institution: dict[int, list[evaluate_mod.CampaignOutcome]],
) -> tuple[list[evaluate_mod.CampaignOutcome], dict[int, int]]:
    """Combine per-institution outcomes into the union detector's outcomes.

    A campaign is detected if **any** institution detected it, at the
    **earliest** such detection time; ties go to the lowest institution index
    so the result does not depend on dict order. The earliest detector's own
    outcome row is carried over whole, so ``lead_time_h``,
    ``normalized_lead_time`` and ``frac_observed`` stay consistent with the
    detection time they were measured at rather than being recomputed here.

    Returns the outcomes and, for the report, which institution each
    detection came from.
    """
    if not per_institution:
        return [], {}

    campaign_ids = {o.campaign_id for outcomes in per_institution.values() for o in outcomes}
    by_campaign: dict[int, list[tuple[int, evaluate_mod.CampaignOutcome]]] = {
        cid: [] for cid in campaign_ids
    }
    for institution in sorted(per_institution):
        for outcome in per_institution[institution]:
            by_campaign[outcome.campaign_id].append((institution, outcome))

    combined: list[evaluate_mod.CampaignOutcome] = []
    credited: dict[int, int] = {}
    for cid in sorted(campaign_ids):
        candidates = by_campaign[cid]
        detected = [(i, o) for i, o in candidates if o.detected]
        if detected:
            institution, outcome = min(
                detected, key=lambda pair: (pair[1].detection_time, pair[0])
            )
            credited[cid] = institution
            combined.append(outcome)
        else:
            combined.append(candidates[0][1])
    return combined, credited


def union_cluster_accounting(
    full_df: pd.DataFrame,
    camp: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    split: str,
    scorers: dict[int, object],
    taus: dict[int, float],
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
) -> dict:
    """Cluster counts for the union, deduplicated by account set.

    Two institutions raising the same account set at the same detection time
    is **one** case on a desk, so the union counts that cluster identity once.
    Deduplication is also why the union is a supervisory number: only an
    entity that receives every stream can do it.

    This is a second loop over the same **frozen** functions the evaluation
    uses — ``extract_clusters`` and ``classify_cluster`` — and not a second
    definition of either. It also returns each institution's own count, which
    a test requires to equal ``evaluate_split``'s ``n_false_alarms`` for that
    institution and tau; if the two ever disagree, this pass is wrong.
    """
    detection_times = windows.evaluation_detection_times(split)
    campaign_splits = dict(zip(camp["campaign_id"], camp["split"]))
    account_index = cluster_tools.build_account_index(accounts_by_campaign)
    unassigned = full_df[(full_df["Is Laundering"] == 1) & (full_df["campaign_id"] == -1)]
    unassigned_accounts = set(unassigned["src"]) | set(unassigned["dst"])

    institutions = sorted(scorers)
    per_institution = {i: {kind: 0 for kind in constants.HIT_KINDS} for i in institutions}
    union_kinds = {kind: 0 for kind in constants.HIT_KINDS}
    duplicate_count = 0
    union_clusters = 0

    horizon_days = 0.0
    if detection_times:
        span = detection_times[-1] - detection_times[0]
        horizon_days = span.total_seconds() / 86400.0 or 1.0

    for t in detection_times:
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue
        src = window["src"].tolist()
        dst = window["dst"].tolist()
        # cluster identity -> kind. A cluster is the same alert whichever
        # institution raised it, and the frozen classification depends only
        # on the account set, so the kind cannot disagree between them.
        seen_at_t: dict[frozenset[str], str] = {}
        raised_at_t = 0
        for institution in institutions:
            scores = np.asarray(scorers[institution](window, t))
            found = cluster_tools.extract_clusters(src, dst, scores, taus[institution])
            for cluster in found:
                kind, _ = cluster_tools.classify_cluster(
                    cluster, split, accounts_by_campaign, campaign_splits,
                    unassigned_accounts, account_index,
                )
                per_institution[institution][kind] += 1
                raised_at_t += 1
                seen_at_t[cluster] = kind
        for kind in seen_at_t.values():
            union_kinds[kind] += 1
        union_clusters += len(seen_at_t)
        duplicate_count += raised_at_t - len(seen_at_t)

    summed_false_alarms = sum(c["false_alarm"] for c in per_institution.values())
    return {
        "split": split,
        "lookback_h": lookback_h,
        "horizon_days": horizon_days,
        "per_institution": per_institution,
        "per_institution_false_alarms": {
            i: per_institution[i]["false_alarm"] for i in institutions
        },
        "union_kinds": union_kinds,
        "union_clusters": union_clusters,
        "summed_false_alarms": summed_false_alarms,
        "union_false_alarms": union_kinds["false_alarm"],
        "duplicate_clusters": duplicate_count,
        "summed_per_day": (
            round(summed_false_alarms / horizon_days, 2) if horizon_days else 0.0
        ),
        "union_per_day": (
            round(union_kinds["false_alarm"] / horizon_days, 2) if horizon_days else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# pooling, with denominators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PooledMean:
    """A per-institution mean, with everything needed to read it honestly."""

    over_all: float | None
    n_all: int
    over_trained: float | None
    n_trained: int
    untrained: tuple[int, ...]
    undefined: tuple[int, ...]
    #: dispersion over the SAME institutions as ``over_all``, because the
    #: eight institutions are not exchangeable -- institution 0 always holds
    #: the dataset's largest bank (see the README, "The institutions are not
    #: exchangeable"). A mean alone invites the reading "what a typical bank
    #: achieves", which this panel cannot support.
    minimum: float | None = None
    argmin: int | None = None
    maximum: float | None = None
    argmax: int | None = None
    median: float | None = None

    def as_row(self) -> dict:
        return {
            "mean_over_all_institutions": self.over_all,
            "n_institutions_in_mean": self.n_all,
            "mean_over_trained_institutions": self.over_trained,
            "n_trained_in_mean": self.n_trained,
            "median_over_institutions": self.median,
            "min_over_institutions": self.minimum,
            "min_institution": self.argmin,
            "max_over_institutions": self.maximum,
            "max_institution": self.argmax,
            "untrained_institutions": ",".join(str(i) for i in self.untrained),
            "undefined_institutions": ",".join(str(i) for i in self.undefined),
        }


def pooled_mean(values: dict[int, float | None], tiers: dict[int, str]) -> PooledMean:
    """The per-institution mean, computed twice.

    * **over all institutions** — an ``untrained`` institution enters at
      recall 0, because "this bank has nothing to learn from and therefore
      catches nothing" is an outcome of the condition, not missing data;
    * **over trained institutions only** — because one mean conflates
      "cannot see the campaign" with "had no labels to learn from", and those
      are different claims about federation.

    A value of ``None`` is **undefined** and is excluded from both means and
    listed: that is a cell with no affordable tau on the grid, or no campaign
    in the subset. It is never silently read as 0.

    Both means carry the **spread** they average over -- median, min and max
    with the institutions that produced them -- because the eight
    institutions are **not exchangeable**: institution 0 deterministically
    holds the dataset's largest bank, and so sees more of every campaign than
    the other seven. The mean is the mean over this fixed panel, not an
    estimate of a typical bank, and the spread is what stops it being read as
    one.
    """
    institutions = sorted(values)
    untrained = tuple(i for i in institutions if tiers.get(i) == TIER_UNTRAINED)
    undefined = tuple(i for i in institutions if values[i] is None)

    usable = [i for i in institutions if values[i] is not None]
    trained = [i for i in usable if tiers.get(i) != TIER_UNTRAINED]

    # dispersion over `usable`, i.e. the same set `over_all` averages, so the
    # spread and the mean always have the same denominator. Ties go to the
    # lowest institution id, so the choice is deterministic.
    lo = min(usable, key=lambda i: (values[i], i)) if usable else None
    hi = max(usable, key=lambda i: (values[i], -i)) if usable else None

    return PooledMean(
        over_all=(
            float(np.mean([values[i] for i in usable])) if usable else None
        ),
        n_all=len(usable),
        over_trained=(
            float(np.mean([values[i] for i in trained])) if trained else None
        ),
        n_trained=len(trained),
        untrained=untrained,
        undefined=undefined,
        minimum=float(values[lo]) if lo is not None else None,
        argmin=lo,
        maximum=float(values[hi]) if hi is not None else None,
        argmax=hi,
        median=(
            float(np.median([values[i] for i in usable])) if usable else None
        ),
    )


#: The threshold below which a chosen F1 tau is treated as having been
#: picked out of a flat curve rather than at a maximum. **Fixed 2026-11-16,
#: before any cell ran under the recording policy**, and quoted wherever the
#: flag appears, so it can never be chosen after seeing which cells it would
#: catch.
#:
#: Why the flag is needed at all: campaign F1 needs a cluster reaching >= 3 of
#: a campaign's accounts at >= 50% purity. A lone institution at visibility
#: target 1.0 has its own oracle ceiling at 0.139 on validation, so at any
#: selective threshold its components never reach three campaign accounts and
#: campaign F1 is ~0 at *every* tau on the grid. The argmax is then arbitrary
#: - the tie-break in `evaluation.best_tau` settles it, not the data - and the
#: tau it lands on is not an optimum. The number is still recorded (the
#: recording policy, 2026-11-16), but a reader must be told which cells are in
#: that regime, which is what this flag is for.
#:
#: 0.01 rather than exactly 0: a curve flat at 0.004 is in the same regime as
#: one flat at 0.0, and demanding an exact zero would let a single lucky
#: cluster in one window hide it.
F1_DEGENERATE_EPSILON = 0.01


def f1_is_degenerate(best_value: float | None) -> bool | None:
    """Was the F1 curve flat enough that its argmax carries no information?

    ``None`` when there is no curve at all - an untrained institution - which
    is not the same as a curve that is flat.
    """
    if best_value is None:
        return None
    return bool(best_value <= F1_DEGENERATE_EPSILON)


def degenerate_summary(flags: dict[int, bool | None]) -> dict:
    """How many of a pooled number's cells had a **degenerate** F1 curve.

    Mirrors `binding_summary` deliberately, and travels with every pooled
    figure for the same reason: a mean taken over cells whose threshold was
    chosen out of a flat curve is not the same quantity as one taken over
    cells with a real optimum, and hiding the mixture inside a single number
    would make them look alike.

    At a **budget** point the threshold comes from the alarm budget and not
    from the argmax, so the flag there describes the institution's detector
    rather than how that operating point was chosen. It is still counted,
    because "this institution detects almost nothing at any threshold" is
    context the budget rows need too.

    The flag informs interpretation and must never select the comparison
    point: `f1` and `total_10` stay primary as designated, or the choice
    becomes post hoc.

    ``None`` entries are cells with no curve - an untrained institution - and
    are out of both the numerator and the denominator.
    """
    with_curve = {i: value for i, value in flags.items() if value is not None}
    n_degenerate = sum(1 for value in with_curve.values() if value)
    return {
        "n_cells_with_an_f1_curve": len(with_curve),
        "n_f1_degenerate": n_degenerate,
        "f1_degenerate_fraction": (
            n_degenerate / len(with_curve) if with_curve else None
        ),
        "f1_degenerate_epsilon": F1_DEGENERATE_EPSILON,
    }


def binding_summary(flags: dict[int, bool | None]) -> dict:
    """How many of a pooled budget number's cells had a **binding** budget.

    A budget binds when it is what set the threshold. It does **not** bind
    when the most sensitive tau on the grid already fits inside it - the
    budget is then looser than anything the detector could do, and "recall at
    B a day" is not really an operating point at B.

    **Whether that happens is a property of the data density, not of being an
    isolated institution.** On the 1% sample no per-institution budget bound
    at any target, because an institution saw too little traffic to raise even
    10 alarms a day; on the full data, ~100x denser, session B measured every
    per-institution budget binding in every cell. So this count is not a
    standing caveat about isolated institutions - it is a per-cell measurement,
    and it reversed between the sample and the full data. Pooling a binding
    cell with a non-binding one into a single "recall at B a day" would average
    a budgeted operating point with an unbudgeted one, so every pooled budget
    number carries this count and the README reports it rather than assuming
    which way it fell.

    ``None`` entries are cells with no budget at all - the F1 point - and are
    excluded from both the numerator and the denominator, so an F1 row
    reports no binding fraction rather than a misleading 0.
    """
    with_budget = {i: value for i, value in flags.items() if value is not None}
    n_binding = sum(1 for value in with_budget.values() if value)
    return {
        "n_cells_with_a_budget": len(with_budget),
        "n_binding_budgets": n_binding,
        "binding_fraction": (
            n_binding / len(with_budget) if with_budget else None
        ),
        "non_binding_institutions": ",".join(
            str(i) for i, value in sorted(with_budget.items()) if not value
        ),
    }


def budget_share(budget: float, k: int = constants.K_INSTITUTIONS) -> float:
    """Each institution's slice of a matched **total** alert budget.

    Equal split, and equal **because it is untuned**: allocating the budget
    across institutions by anything data-dependent would be a tuning step
    that the equal-effort constraint does not cover, and it would hand the
    floor an advantage the federated conditions were never given.
    """
    return float(budget) / int(k)


#: The operating points a cell is evaluated at. ``f1`` and ``budget_*`` are
#: the Phase 3 points, at the institution's **own** full budget: a bank alone
#: has its own analysts. ``total_*`` give each institution ``B / K`` so the
#: union's summed volume meets ``B``, which is the only way to compare the
#: union against a centralized model at ``B`` without letting it outspend it
#: eight to one.
#: The two operating points Phase 4 **compares conditions at**, designated on
#: 2026-11-14 before any federated arm existed, so that the comparison point
#: cannot be said to have been picked after seeing which arm it favoured.
#:
#: ``f1`` is set by the frozen selection rule on validation. ``total_10``
#: is the only budget point at which the **union's summed** alarm volume is
#: held to the same B a centralized model is held to, so it is the only one
#: where a condition comparison is not also a spending comparison.
#:
#: That is the whole justification and it is **structural**, so it did not
#: move when the sample's binding result did. An earlier draft of this note
#: justified the pair differently - "the only two points where the threshold
#: is determined by something", `total_10` being "set by a budget that
#: actually binds" while the per-institution points were set by neither. On
#: the full data the per-institution budgets bind too (see the naming note in
#: `operating_points`), so that phrasing would now pick out four points
#: rather than two. The designation stands on the matched-volume reading,
#: which never depended on which budgets happened to bind.
PRIMARY_POINTS = ("f1", "total_10")


def operating_points(k: int = constants.K_INSTITUTIONS) -> list[dict]:
    """The seven points a cell is evaluated at.

    **Naming (D6, resolved 2026-11-15): ``per_inst_10/50/100``.** These
    points give each institution the *full* budget B, so the union spends up
    to ``K x B``. The name has been through all three candidates and the
    reasoning is worth keeping, because two of them were wrong for opposite
    reasons:

    * ``budget_*``, the original, invites a reader to compare it against a
      centralized model's genuinely budgeted B a day. That comparison lets the
      union outspend the centralized model eight to one, and it is the error
      ``budget_binding`` and ``total_*`` exist to prevent. **Rejected, and
      still rejected** - the point comparable to a centralized ``budget_10``
      is ``total_10``, which is why that is PRIMARY.
    * ``uncapped_*``, used 2026-11-14 to 2026-11-15, said these were not
      budgeted operating points at all. That was measured, on the 1% sample:
      the budget bound in **0 of 8** institutions at every target, because an
      institution seeing a fraction of the traffic could not raise even 10
      alarms a day at the grid's most sensitive threshold, so the grid floor
      set the threshold instead. **On the full data it reverses** - every
      per-institution budget point binds, ``binding_fraction`` 1.0 - because
      the full background is ~100x denser and the grid floor now raises far
      more than 100 alarms a day. The sample misled because *whether a budget
      binds is a property of data density, not of being isolated*, and a 1%
      sample changes exactly that property.
    * ``per_inst_*`` carries the **structure** instead: each institution at
      its own per-institution budget. Structure does not move when a
      measurement does, which is the whole lesson - a label must not encode a
      finding, because a finding can reverse and a column name cannot be
      renamed retroactively in reports already written.

    The per-cell ``budget_binding`` flag remains the authoritative answer to
    whether any particular cell was budgeted, and it is what to read; the name
    now says only which budget was applied to whom.

    ``total_*`` give each institution ``B / K`` so the **union's** summed
    volume meets B, which is the only reading in which a budget constrains
    anything here.
    """
    points = [{
        "operating_point": "f1", "kind": "f1", "budget": None,
        "per_institution_budget": None, "primary": True,
    }]
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        points.append({
            "operating_point": f"per_inst_{budget}",
            "kind": "per_institution_budget",
            "budget": float(budget),
            "per_institution_budget": float(budget),
            "primary": f"per_inst_{budget}" in PRIMARY_POINTS,
        })
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        points.append({
            "operating_point": f"total_{budget}",
            "kind": "matched_total_budget",
            "budget": float(budget),
            "per_institution_budget": budget_share(budget, k),
            "primary": f"total_{budget}" in PRIMARY_POINTS,
        })
    return points


# ---------------------------------------------------------------------------
# one institution, end to end
# ---------------------------------------------------------------------------


@dataclass
class InstitutionEvaluation:
    """Everything one institution produced in one cell."""

    institution: int
    tier: str
    tau_f1: float | None = None
    tau_at_grid_edge: bool = False
    #: the best campaign F1 the validation sweep actually reached, i.e. the
    #: value AT ``tau_f1``. Recorded beside the tau because the tau alone
    #: cannot say whether it was chosen at a maximum or out of a flat curve.
    f1_best_value: float | None = None
    #: was that best value at or below ``F1_DEGENERATE_EPSILON``?
    f1_degenerate: bool | None = None
    tau_curve: list[dict] = field(default_factory=list)
    #: operating point -> tau, or ``None`` when the point has no operating
    #: point at all: either no tau on the grid meets that budget, or the
    #: chosen tau sat at the **top** of the grid, which means the optimum lies
    #: beyond it. ``None`` means the detector cannot be *operated* there,
    #: which is not the same as detecting nothing.
    taus: dict[str, float | None] = field(default_factory=dict)
    #: operating point -> why it has no tau, for the report
    undefined_reason: dict[str, str] = field(default_factory=dict)
    #: operating point -> the tau that was rejected for sitting at the top of
    #: the grid, kept so the report shows the value and not just its absence
    rejected_taus: dict[str, float] = field(default_factory=dict)
    #: operating point -> "low" or "high" for a tau at a grid edge
    tau_edges: dict[str, str] = field(default_factory=dict)
    #: operating point -> was the budget BINDING? False when the budget tau
    #: sits at the bottom of the grid, i.e. the most sensitive threshold
    #: already fits the budget and the budget is therefore not what set the
    #: operating point. ``None`` for the F1 point, which has no budget.
    budget_binding: dict[str, bool] = field(default_factory=dict)
    #: ``(operating_point, split) -> outcomes``. Missing for an undefined point.
    outcomes: dict[tuple[str, str], list] = field(default_factory=dict)
    results: dict[tuple[str, str], object] = field(default_factory=dict)
    oracle_results: dict[str, object] = field(default_factory=dict)
    oracle_reachable: dict[str, list[int]] = field(default_factory=dict)
    #: ``(split, subset) -> this institution's own oracle recall``
    ceiling: dict[tuple[str, str], float | None] = field(default_factory=dict)
    seconds: float = 0.0


def evaluate_institution(
    full_df: pd.DataFrame,
    camp: pd.DataFrame,
    pattern_df: pd.DataFrame,
    accounts_by_campaign: dict[int, tuple[str, ...]],
    model: IsolatedModel,
    assignment: fv.VisibilityAssignment,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    seed: int = 0,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    points: list[dict] | None = None,
    progress=None,
    scorer=None,
    condition: str = CONDITION,
) -> InstitutionEvaluation:
    """Tau, ceiling and every operating point for one institution.

    ``scorer`` lets the caller own the institution's ``CachingScorer`` and
    keep it warm. That matters more than it looks: the union's cluster pass
    walks the same windows again, and with a fresh scorer it would
    re-featurise every window for every institution at every operating point
    — on the full data the featurisation is the expensive half, so sharing
    one cache per institution is the difference between clustering again and
    computing everything again.

    The ceiling is computed for **every** institution including an untrained
    one, because a ceiling is a property of the institution's view and not of
    its training data. An untrained institution is not scored at all: its
    outcomes are the all-missed list, which a test pins as identical to
    running the detection loop with an all-zero scorer.
    """
    started = time.perf_counter()
    points = points or operating_points(assignment.k)
    out = InstitutionEvaluation(institution=model.institution, tier=model.tier)

    # --- the ceiling: the oracle through the same mask ---
    oracle = oracle_view_scorer(model.institution, assignment)
    for split in splits:
        result = evaluation.evaluate_split(
            full_df, camp, pattern_df, accounts_by_campaign, split=split,
            scorer=oracle, tau=evaluation.SANITY_TAU["oracle"],
            lookback_h=lookback_h, seed=seed,
            scorer_name="oracle_isolated_i%d" % model.institution,
        )
        reachable = [o.campaign_id for o in result.outcomes if o.detected]
        out.oracle_results[split] = result
        out.oracle_reachable[split] = reachable
        for subset in constants.TEST_SUBSETS:
            out.ceiling[(split, subset)] = evaluation.summarize(
                result, subset, bootstrap_seed=seed, n_resamples=1,
                oracle_reachable=reachable,
            )["campaign_recall"]

    if not model.trained:
        for point in points:
            out.taus[point["operating_point"]] = None
            for split in splits:
                out.outcomes[(point["operating_point"], split)] = undetected_outcomes(
                    camp, split
                )
        out.seconds = time.perf_counter() - started
        return out

    # --- tau on validation only, by the frozen rule ---
    owns_scorer = scorer is None
    if owns_scorer:
        scorer = evaluation.CachingScorer(model.scorer(assignment), lookback_h)
    tau_f1, curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts_by_campaign, scorer=scorer,
        tau_grid=evaluation.MODEL_TAU_GRID, lookback_h=lookback_h, seed=seed,
        scorer_name="%s_i%d" % (condition, model.institution), cache_scores=False,
    )
    out.tau_f1 = tau_f1
    out.tau_curve = curve
    out.tau_at_grid_edge = evaluation.tau_is_at_grid_edge(
        tau_f1, evaluation.MODEL_TAU_GRID
    )
    # The value AT the argmax, not just where the argmax was. `best_tau`
    # maximises `campaign_f1` over this same curve, so the max over the curve
    # IS the value at `tau_f1`; taking the max rather than looking the tau up
    # avoids depending on the tie-break to find the row again.
    out.f1_best_value = max(row["campaign_f1"] for row in curve) if curve else None
    out.f1_degenerate = f1_is_degenerate(out.f1_best_value)

    for point in points:
        name = point["operating_point"]
        if point["kind"] == "f1":
            chosen = tau_f1
        else:
            chosen = evaluation.select_tau_for_budget(
                curve, point["per_institution_budget"]
            )
        if chosen is None:
            out.taus[name] = None
            out.undefined_reason[name] = NO_AFFORDABLE_TAU
            continue

        edge = tau_grid_edge(chosen)
        # The two edges mean opposite things, and only one of them invalidates
        # the point.
        #
        # **High** — the chosen tau is the most extreme on the grid, so the
        # optimum lies beyond it and this operating point was not found. It is
        # recorded as undefined and excluded from the pooled means, never as
        # recall 0, for the same reason an unaffordable budget is: a detector
        # that cannot be operated here was not measured here.
        #
        # **Low** for a *budget* point is not a problem at all — it says the
        # most sensitive threshold on the grid already fits the budget, which
        # is a loose budget and not a missing optimum. **Low for the F1
        # point** does say the optimum may lie below the grid, so it is
        # flagged for the caller to act on; at the full data's 0.102%
        # prevalence it is not expected to happen.
        if edge == "high":
            out.tau_edges[name] = "high"
            out.rejected_taus[name] = chosen
            out.taus[name] = None
            out.undefined_reason[name] = TAU_BEYOND_GRID
            continue
        if edge == "low" and point["kind"] == "f1":
            out.tau_edges[name] = "low"
        if point["kind"] != "f1":
            # A budget tau at the bottom of the grid means the most sensitive
            # threshold already fits the budget: the budget did not bind, so
            # this is not really an operating point *at* B alarms a day. An
            # isolated institution sees a fraction of the traffic and will
            # often be here, while the centralized model at the same nominal
            # budget will not - and pooling the two would compare a budgeted
            # operating point against an unbudgeted one.
            out.budget_binding[name] = edge != "low"
        out.taus[name] = chosen

    # --- every defined operating point, on every split ---
    for point in points:
        name = point["operating_point"]
        tau = out.taus[name]
        if tau is None:
            continue
        for split in splits:
            result = evaluation.evaluate_split(
                full_df, camp, pattern_df, accounts_by_campaign, split=split,
                scorer=scorer, tau=float(tau), lookback_h=lookback_h, seed=seed,
                scorer_name="%s_i%d" % (condition, model.institution),
            )
            out.results[(name, split)] = result
            out.outcomes[(name, split)] = result.outcomes

    if owns_scorer:
        scorer.clear()
    out.seconds = time.perf_counter() - started
    if progress:
        undefined = sum(1 for v in out.taus.values() if v is None)
        progress(
            "    institution %d: tau_f1 = %s, %d point(s) with no affordable "
            "tau, %.1fs" % (model.institution, tau_f1, undefined, out.seconds)
        )
    return out


# ---------------------------------------------------------------------------
# one cell: (target, seed, lookback)
# ---------------------------------------------------------------------------


EMPTY_ACCOUNTING = {
    "union_kinds": {kind: 0 for kind in constants.HIT_KINDS},
    "union_clusters": 0,
    "horizon_days": 0.0,
}


def _as_result(
    split: str,
    outcomes: list,
    accounting: dict,
    lookback_h: int,
    seed: int,
    scorer_name: str,
):
    """An ``EvaluationResult`` assembled from counts rather than from a run.

    Used for the union, whose cluster counts come from the deduplicating
    pass, and for an untrained institution, which raises nothing. ``tau`` is
    NaN on purpose for the union: it has K thresholds, not one, and they are
    carried per institution rather than collapsed into a number that would
    read like a single detector's threshold.
    """
    kinds = accounting["union_kinds"]
    result = evaluate_mod.EvaluationResult(
        split=split, scorer=scorer_name, tau=float("nan"),
        lookback_h=lookback_h, seed=seed, outcomes=outcomes,
        n_hits_clusters=kinds["hit"], n_other_split_hits=kinds["other_split_hit"],
        n_ambiguous=kinds["ambiguous"], n_false_alarms=kinds["false_alarm"],
        n_clusters=accounting["union_clusters"],
    )
    result.horizon_days = accounting["horizon_days"]
    return result


def cell_name(target: float, seed: int, institution: int | None = None) -> str:
    """The on-disk name of a cell, shared by the two Kaggle subcommands.

    One name built in one place, so ``train-isolated`` and
    ``evaluate-isolated`` cannot disagree about where a booster lives — which
    is what makes a resumed session pick up the work it already did.
    """
    stem = "t%03d_s%d" % (round(target * 100), seed)
    return stem if institution is None else "%s_i%d" % (stem, institution)


def loaded_model(
    directory,
    institution: int,
    lookback_h: int,
    n_rows: int,
    n_positive: int,
) -> IsolatedModel:
    """Rebuild an :class:`IsolatedModel` from a saved booster.

    The tier is derived from the positive count rather than stored, so a
    saved model and a freshly trained one are classified by the same rule.
    """
    trained = xgb_baseline.TrainedBaseline.load(directory)
    return IsolatedModel(
        institution=institution, lookback_h=lookback_h,
        tier=training_tier(n_positive), n_rows=int(n_rows),
        n_positive=int(n_positive), model=trained,
        training={"loaded_from": str(directory)},
    )


def run_cell(
    result,
    target: float,
    seed: int,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    params: xgb_baseline.XGBParams | None = None,
    progress=None,
    models: dict[int, IsolatedModel] | None = None,
    condition: str = CONDITION,
    scorer_for=None,
    trainer=None,
) -> dict:
    """One ``(target, seed, lookback)`` cell of a per-institution condition.

    Returns row lists ready for CSV, so the local development script and the
    Kaggle runner assemble the same report from the same code and cannot
    drift apart. ``models`` skips training and uses boosters trained (and
    saved) earlier, which is how the Kaggle run splits a cell across two
    sessions.

    **Why this is parameterised rather than copied.** Stages 4.2 and 4.3
    differ from 4.1 in exactly one place: the scorer applied to an
    institution's view. Tau selection, the seven operating points, the
    grid-edge policy, the per-institution ceiling, both pooled means and the
    union are the same machinery, and they have to *be* the same machinery -
    rule 1 pairs conditions per ``(target, seed, institution)``, which is only
    meaningful if the two sides were measured identically. Two copies of this
    function would be two chances to diverge silently.

    * ``condition`` is the label written into every row.
    * ``scorer_for(institution, models, assignment)`` returns that
      institution's **inner** scorer. The default is the isolated one: the
      institution's own booster behind its own visibility mask. 4.2 returns an
      average over all K boosters; 4.3 returns a scorer that featurises with
      shared boundary-account evidence.
    * ``trainer(full_df, assignment, lookback_h, params, progress)`` returns
      the models when none are supplied. The default trains isolated boosters.

    The scorer factory returns the inner scorer and this function owns the
    ``CachingScorer`` around it, because the union's cluster pass walks the
    same windows again and a fresh scorer there would re-featurise everything.
    """
    from .. import visibility

    started = time.perf_counter()
    say = progress or (lambda *_: None)
    assignment, achieved = fv.assignment_from_pipeline(result, target=target, seed=seed)
    points = operating_points(assignment.k)
    bin_counts = (
        pd.Series([visibility.bin_of(v) for v in achieved.values()])
        .value_counts()
        .to_dict()
    )

    common = {
        "condition": condition,
        "model": "xgboost",
        "variant": "xgboost_centralized",
        "visibility_target": target,
        "lookback_h": lookback_h,
        "seed": seed,
    }

    if models is None:
        say("  training %d institutions ..." % assignment.k)
        models = (trainer or train_all_isolated)(
            result.full_df, assignment, lookback_h, params, progress=say
        )
    else:
        say("  using %d boosters trained earlier" % sum(
            1 for m in models.values() if m.trained
        ))
        missing = set(range(assignment.k)) - set(models)
        if missing:
            raise ValueError(
                f"no model supplied for institution(s) {sorted(missing)}; a cell "
                "is evaluated over every institution or not at all"
            )
    tiers = {i: m.tier for i, m in models.items()}

    say("  evaluating ...")
    # one warm score cache per institution, owned here because the union's
    # cluster pass walks the same windows again
    inner_scorer = scorer_for or (
        lambda institution, models, assignment: models[institution].scorer(assignment)
    )
    cached_scorers = {
        institution: evaluation.CachingScorer(
            inner_scorer(institution, models, assignment), lookback_h
        )
        for institution in sorted(models) if models[institution].trained
    }
    evaluations: dict[int, InstitutionEvaluation] = {}
    for institution in sorted(models):
        evaluations[institution] = evaluate_institution(
            result.full_df, result.camp, result.pattern_df,
            result.accounts_by_campaign, models[institution], assignment,
            lookback_h=lookback_h, seed=seed, splits=splits, points=points,
            progress=say, scorer=cached_scorers.get(institution),
            condition=condition,
        )

    institution_rows = []
    for institution in sorted(models):
        row = dict(common)
        row.update(models[institution].as_row())
        row["tau_f1"] = evaluations[institution].tau_f1
        row["tau_at_grid_edge"] = evaluations[institution].tau_at_grid_edge
        row["f1_best_value"] = evaluations[institution].f1_best_value
        row["f1_degenerate"] = evaluations[institution].f1_degenerate
        row["f1_degenerate_epsilon"] = F1_DEGENERATE_EPSILON
        row["eval_seconds"] = round(evaluations[institution].seconds, 2)
        institution_rows.append(row)

    tau_rows = []
    for institution in sorted(models):
        for point in points:
            name = point["operating_point"]
            tau_rows.append({
                **common, "institution": institution, "operating_point": name,
                "point_kind": point["kind"], "budget": point["budget"],
                "per_institution_budget": point["per_institution_budget"],
                "primary": point["primary"],
                "tau": evaluations[institution].taus.get(name),
                "defined": evaluations[institution].taus.get(name) is not None,
                "rejected_tau": evaluations[institution].rejected_taus.get(name),
                "tau_edge": evaluations[institution].tau_edges.get(name),
                "budget_binding": evaluations[institution].budget_binding.get(name),
                "undefined_reason": evaluations[institution].undefined_reason.get(name),
                "f1_best_value": evaluations[institution].f1_best_value,
                "f1_degenerate": evaluations[institution].f1_degenerate,
                "f1_degenerate_epsilon": F1_DEGENERATE_EPSILON,
                "tier": tiers[institution],
            })

    ceiling_rows = []
    for institution in sorted(models):
        for key, value in sorted(evaluations[institution].ceiling.items()):
            split, subset = key
            ceiling_rows.append({
                **common, "institution": institution, "eval_split": split,
                "test_subset": subset, "oracle_recall": value,
                "n_oracle_reachable": len(
                    evaluations[institution].oracle_reachable.get(split, [])
                ),
                "tier": tiers[institution],
            })

    summary_rows: list[dict] = []
    outcome_rows: list[dict] = []
    pooled_rows: list[dict] = []
    union_rows: list[dict] = []

    for point in points:
        name = point["operating_point"]
        for split in splits:
            recall_by_institution: dict[int, float | None] = {}
            relative_by_institution: dict[int, float | None] = {}
            binding_by_institution: dict[int, bool | None] = {
                i: ev.budget_binding.get(name) for i, ev in evaluations.items()
            }
            degenerate_by_institution: dict[int, bool | None] = {
                i: ev.f1_degenerate for i, ev in evaluations.items()
            }

            for institution in sorted(models):
                ev = evaluations[institution]
                res = ev.results.get((name, split))

                if res is None and ev.tier != TIER_UNTRAINED:
                    # no operating point here - either no affordable tau, or a
                    # chosen tau at the top of the grid with the optimum beyond
                    # it. Undefined, never zero.
                    recall_by_institution[institution] = None
                    relative_by_institution[institution] = None
                    for subset in constants.TEST_SUBSETS:
                        summary_rows.append({
                            **common, "institution": institution,
                            "operating_point": name, "eval_split": split,
                            "test_subset": subset, "tier": ev.tier,
                            "tau": None, "defined": False,
                            "rejected_tau": ev.rejected_taus.get(name),
                            "budget_binding": ev.budget_binding.get(name),
                            "f1_best_value": ev.f1_best_value,
                            "f1_degenerate": ev.f1_degenerate,
                            "undefined_reason": ev.undefined_reason.get(
                                name, NO_AFFORDABLE_TAU
                            ),
                        })
                    continue

                if res is None:
                    # untrained: operable in principle, detects nothing
                    res = _as_result(
                        split, ev.outcomes[(name, split)], EMPTY_ACCOUNTING,
                        lookback_h, seed,
                        "%s_i%d_untrained" % (condition, institution),
                    )

                for subset in constants.TEST_SUBSETS:
                    row = evaluation.summarize(
                        res, subset, bootstrap_seed=seed,
                        oracle_reachable=ev.oracle_reachable[split],
                        oracle_recall=ev.ceiling[(split, subset)],
                    )
                    summary_rows.append({
                        **common, "institution": institution,
                        "operating_point": name, "eval_split": split,
                        "tier": ev.tier, "defined": True,
                        "budget_binding": ev.budget_binding.get(name),
                        "f1_best_value": ev.f1_best_value,
                        "f1_degenerate": ev.f1_degenerate,
                        "undefined_reason": None, **row,
                    })
                    if subset == "all":
                        recall_by_institution[institution] = row["campaign_recall"]
                        relative_by_institution[institution] = row[
                            "recall_relative_to_ceiling"
                        ]

                outcome_rows.extend(evaluation.outcome_records(
                    res, **common, institution=institution,
                    operating_point=name, tier=ev.tier,
                ))

            pooled_recall = pooled_mean(recall_by_institution, tiers)
            pooled_relative = pooled_mean(relative_by_institution, tiers)
            # How many of the cells this mean is taken over actually had a
            # BINDING budget. Which way this falls is a property of the data
            # density, not of the condition: on the 1% sample no
            # per-institution budget bound at all, and on the full data every
            # one of them does. A pooled "recall at B a day" that hides the
            # mixture averages a budgeted operating point with an unbudgeted
            # one, so the count travels with every pooled budget number.
            for metric, pooled, values in (
                ("campaign_recall", pooled_recall, recall_by_institution),
                (
                    "recall_relative_to_own_ceiling",
                    pooled_relative,
                    relative_by_institution,
                ),
            ):
                # counted over the cells THIS mean is taken over, which is not
                # always the same set: a cell with no operating point is out of
                # both, and one whose own ceiling is 0 is out of the relative
                # mean only
                in_mean = [i for i, value in values.items() if value is not None]
                pooled_rows.append({
                    **common, "operating_point": name, "eval_split": split,
                    "test_subset": "all", "metric": metric,
                    **pooled.as_row(),
                    **binding_summary(
                        {i: binding_by_institution.get(i) for i in in_mean}
                    ),
                    **degenerate_summary(
                        {i: degenerate_by_institution.get(i) for i in in_mean}
                    ),
                })

            # --- the union: a supervisory reader, not a bank ---
            operating = {
                i: ev for i, ev in evaluations.items()
                if ev.taus.get(name) is not None and models[i].trained
            }
            union_campaign_outcomes, credited = union_outcomes({
                i: ev.outcomes[(name, split)]
                for i, ev in evaluations.items()
                if (name, split) in ev.outcomes
            })
            if not union_campaign_outcomes:
                continue

            accounting = union_cluster_accounting(
                result.full_df, result.camp, result.accounts_by_campaign, split,
                scorers={i: cached_scorers[i] for i in operating},
                taus={i: operating[i].taus[name] for i in operating},
                lookback_h=lookback_h,
            ) if operating else dict(EMPTY_ACCOUNTING, **{
                "per_institution_false_alarms": {}, "summed_false_alarms": 0,
                "union_false_alarms": 0, "duplicate_clusters": 0,
                "summed_per_day": 0.0, "union_per_day": 0.0,
            })

            union_res = _as_result(
                split, union_campaign_outcomes, accounting, lookback_h, seed,
                "%s_union" % condition,
            )
            union_oracle_outcomes, _ = union_outcomes({
                i: ev.oracle_results[split].outcomes for i, ev in evaluations.items()
            })
            union_reachable = [
                o.campaign_id for o in union_oracle_outcomes if o.detected
            ]
            union_ceiling = {}
            for subset in constants.TEST_SUBSETS:
                keep = evaluation.subset_filter(subset, union_reachable)
                kept = [o for o in union_oracle_outcomes if keep(o)]
                union_ceiling[subset] = (
                    sum(1 for o in kept if o.detected) / len(kept) if kept else None
                )

            for subset in constants.TEST_SUBSETS:
                row = evaluation.summarize(
                    union_res, subset, bootstrap_seed=seed,
                    oracle_reachable=union_reachable,
                    oracle_recall=union_ceiling[subset],
                )
                union_rows.append({
                    **common, "institution": "union", "operating_point": name,
                    "eval_split": split, "point_kind": point["kind"],
                    "primary": point["primary"],
                    "budget": point["budget"],
                    "per_institution_budget": point["per_institution_budget"],
                    "n_institutions_operating": len(operating),
                    "n_binding_budgets": sum(
                        1 for i in operating
                        if evaluations[i].budget_binding.get(name) is True
                    ),
                    **degenerate_summary(
                        {i: evaluations[i].f1_degenerate for i in operating}
                    ),
                    "summed_false_alarms": accounting["summed_false_alarms"],
                    "summed_false_alarms_per_day": accounting["summed_per_day"],
                    "duplicate_clusters": accounting["duplicate_clusters"],
                    "taus": ";".join(
                        "%d:%s" % (i, operating[i].taus[name])
                        for i in sorted(operating)
                    ),
                    **row,
                })

            outcome_rows.extend(evaluation.outcome_records(
                union_res, **common, institution="union",
                operating_point=name, tier="union",
            ))
            union_rows[-1]["credited_institutions"] = ";".join(
                "%d:%d" % (cid, inst) for cid, inst in sorted(credited.items())
            )

    flags = {
        "untrained": [i for i, tier in tiers.items() if tier == TIER_UNTRAINED],
        "thin": [i for i, tier in tiers.items() if tier == TIER_THIN],
        "tau_at_grid_edge": [
            i for i, ev in evaluations.items() if ev.tau_at_grid_edge
        ],
        # (institution, operating point) pairs, by why they have no tau. A
        # HIGH edge is an operating point that was not found and is excluded
        # like an unaffordable budget; a LOW edge on the F1 point is not
        # excluded but must not occur on the full data, and the runner stops
        # on it rather than recording it.
        "tau_at_high_edge": sorted(
            (i, name)
            for i, ev in evaluations.items()
            for name, edge in ev.tau_edges.items() if edge == "high"
        ),
        "tau_at_low_edge": sorted(
            (i, name)
            for i, ev in evaluations.items()
            for name, edge in ev.tau_edges.items() if edge == "low"
        ),
        # Institutions whose validation F1 curve was flat at or below
        # F1_DEGENERATE_EPSILON, so the tau they operate at was settled by the
        # tie-break and not by the data. Recorded, NOT excluded - see
        # `degenerate_summary`.
        "f1_degenerate": sorted(
            i for i, ev in evaluations.items() if ev.f1_degenerate
        ),
        "no_affordable_tau": sorted(
            (i, name)
            for i, ev in evaluations.items()
            for name, tau in ev.taus.items()
            if tau is None and ev.tier != TIER_UNTRAINED
            and ev.undefined_reason.get(name) == NO_AFFORDABLE_TAU
        ),
    }

    for scorer in cached_scorers.values():
        scorer.clear()

    return {
        "target": target,
        "seed": seed,
        "lookback_h": lookback_h,
        "n_reassigned": assignment.n_reassigned,
        "visibility_bins": bin_counts,
        "institution_rows": institution_rows,
        "tau_rows": tau_rows,
        "ceiling_rows": ceiling_rows,
        "summary_rows": summary_rows,
        "pooled_rows": pooled_rows,
        "union_rows": union_rows,
        "outcome_rows": outcome_rows,
        "tau_curves": {i: evaluations[i].tau_curve for i in sorted(evaluations)},
        "flags": flags,
        "seconds": round(time.perf_counter() - started, 1),
    }
