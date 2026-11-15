"""Stage 4.2: the FedAvg analogue for a gradient-boosted tree ensemble.

**What this condition is.** Every institution keeps its own data. The eight
boosters trained in stage 4.1 for the same ``(target, seed)`` are shared, and
each institution scores **its own view** with all eight of them, averaging the
scores per edge with equal weights. Nothing is retrained. The FedAvg
invariant is preserved exactly: *models move, data does not.*

**Why not parameter averaging.** FedAvg averages model *parameters*, which
requires the parameters to live in a fixed vector space where a coordinate-wise
mean means something. A gradient-boosted ensemble does not:

* a booster is a *sequence* of trees, and each tree's content is discrete
  structure - which feature a node splits on, at which threshold, in which
  order. There is no coordinate system in which "the mean of two split
  features" is defined;
* the trees are fitted *sequentially on residuals*, so tree ``j`` only means
  anything given trees ``1..j-1``. Tree 5 of institution A and tree 5 of
  institution B correct different errors of different predecessors, so pairing
  them by index pairs unrelated things;
* two institutions trained on different views choose different split features
  at the root, so even the leaf values are not comparable - they are indexed by
  paths that do not exist in the other tree.

Averaging leaf values across structurally different trees is not a smoothed
model; it is arithmetic on incommensurable numbers.

**Why score averaging is the standard analogue.** Where parameters cannot be
averaged, federated learning averages at the *output*: each client's model is
applied and the predictions combined. For tree ensembles this is the usual
substitution, and it keeps the three properties that make FedAvg a meaningful
baseline here - no raw data leaves an institution, every institution
contributes equally, and the combination is a single unweighted mean with
nothing fitted. The alternative in the literature, jointly *growing* trees
across parties (as in federated XGBoost), requires gradient or histogram
exchange per split and is a different condition with a different privacy
cost, not an analogue of FedAvg.

**Equal weights, and no weighted variant.** The mean is unweighted because
unweighted is *untuned*. Any weighting - by rows, by positives, by validation
score - is a fitted quantity, and fitting it would hand this arm an advantage
that the isolated floor was never given, breaking the equal-effort constraint
the phase is built on. D2 fixes this and no weighted variant is reported.

**What stays identical to 4.1, deliberately.** The visibility mask, the
per-institution oracle ceiling, tau selection on validation, the seven
operating points, the grid-edge policy, both pooled means and the union all
come from :mod:`fraudcamp.federation.isolated` unchanged - this module
supplies a scorer and nothing else. In particular **the ceilings are
identical**, because averaging scores cannot change what an institution can
see. That is what makes 4.2 - 4.1 a clean measurement of scoring: paired per
``(target, seed, institution)``, against the same denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import constants
from ..models import xgb_baseline
from . import isolated, views as fv

#: The condition's name, written into every row of every report.
CONDITION = "fedavg"


def participating(models: dict[int, isolated.IsolatedModel]) -> list[int]:
    """The institutions whose booster takes part in the average.

    An institution with **no** booster contributes nothing and is left out of
    the mean rather than entered at score 0. Entering a 0 would be a claim
    that the institution predicts "not laundering" with confidence, when in
    fact it has not predicted at all, and it would drag every averaged score
    towards the floor by ``1/K`` per missing model - changing the scores of
    the seven institutions that *did* train.

    On the full data this never fires: all 96 boosters trained. It is here
    because the 1% sample can starve an institution, and the development run
    must not silently measure something else.
    """
    return [i for i in sorted(models) if models[i].trained]


def averaged_scorer(
    models: dict[int, isolated.IsolatedModel],
    institution: int,
    assignment: fv.VisibilityAssignment,
):
    """Institution ``institution``'s view, scored by every booster, averaged.

    The mask is applied first and the models score only what this institution
    can see, which is the whole condition: the models travelled, the rows did
    not. Masking is :func:`isolated.masked_scorer`, so the scatter back into
    the global row order is the same code the isolated condition uses and
    cannot misalign differently.
    """
    boosters = [models[i].model for i in participating(models)]
    if not boosters:
        return isolated.zero_scorer()

    def inner(visible: pd.DataFrame, t) -> np.ndarray:
        per_model = xgb_baseline.predict_window_multi(boosters, visible, t)
        return per_model.mean(axis=0)

    return isolated.masked_scorer(inner, institution, assignment)


def naive_averaged_scorer(
    models: dict[int, isolated.IsolatedModel],
    institution: int,
    assignment: fv.VisibilityAssignment,
):
    """The same thing, featurising once per booster instead of once per view.

    Kept only so a test can pin that the shared-featurisation path in
    :func:`averaged_scorer` returns bit-identical scores. If the two ever
    disagree, the fast path is wrong and this is the definition.
    """
    boosters = [models[i].model for i in participating(models)]
    if not boosters:
        return isolated.zero_scorer()

    def inner(visible: pd.DataFrame, t) -> np.ndarray:
        stacked = np.vstack([
            booster.predict_window(visible, t).reshape(1, -1)
            for booster in boosters
        ])
        return stacked.mean(axis=0)

    return isolated.masked_scorer(inner, institution, assignment)


def scorer_for(
    institution: int,
    models: dict[int, isolated.IsolatedModel],
    assignment: fv.VisibilityAssignment,
):
    """The factory :func:`isolated.run_cell` takes."""
    return averaged_scorer(models, institution, assignment)


def run_cell(
    result,
    target: float,
    seed: int,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    params: xgb_baseline.XGBParams | None = None,
    progress=None,
    models: dict[int, isolated.IsolatedModel] | None = None,
) -> dict:
    """Stage 4.2 for one ``(target, seed, lookback)``.

    Identical to :func:`isolated.run_cell` except for the scorer and the
    condition label. ``models`` are stage 4.1's boosters: this condition
    trains nothing of its own, and passing them is the normal path. If they
    are not supplied they are trained here, isolated, exactly as 4.1 trains
    them - the same boosters, so 4.1 and 4.2 remain paired.
    """
    return isolated.run_cell(
        result,
        target=target,
        seed=seed,
        lookback_h=lookback_h,
        splits=splits,
        params=params,
        progress=progress,
        models=models,
        condition=CONDITION,
        scorer_for=scorer_for,
    )


__all__ = [
    "CONDITION",
    "averaged_scorer",
    "naive_averaged_scorer",
    "participating",
    "run_cell",
    "scorer_for",
]
