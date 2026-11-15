"""Stage 4.4: the centralized model, re-reported inside the Phase 4 cell grid.

**Why this stage exists.** Rule 1 pairs conditions per
``(target, seed, institution)`` or per ``(target, seed, union)`` over
*identical* cells. The Phase 3 centralized result is not in that shape - it is
indexed by model seed, not by visibility cell - so it cannot be pasted into a
Phase 4 comparison table. This stage re-reports it in the Phase 4 shape: the
frozen 22-point grid, L = 48, the same seven operating points, the same
subsets, the same summariser.

**How it reuses everything.** A centralized model is the ``K = 1`` case of a
per-institution condition, so it is evaluated by
:func:`isolated.evaluate_institution` through a degenerate assignment in which
every account belongs to institution 0. Then:

* ``visible_mask`` is all-true, so the model sees everything;
* ``oracle_view_scorer`` is the unmasked oracle, so the ceiling is the
  centralized ceiling;
* ``operating_points(k=1)`` makes ``total_B`` and ``per_inst_B`` carry the same
  budget ``B / 1 = B``. D4's "for centralized, ``total_B = per_inst_B = B``"
  therefore falls out of the existing definition rather than being a special
  case written for this stage.

Nothing about tau selection, the grid-edge policy, the ceiling or the
summariser is re-implemented here, which is the only way the arms stay
comparable.

**The seed question is NOT settled in this module.** Phase 3's seeds are
*model* seeds; Phase 4's are *visibility* seeds, and all 96 Phase 4 boosters
are trained at model seed 0. The three readings of "the centralized arm at
three seeds" give materially different numbers, so
:func:`model_seed_policy` requires the caller to name one and has no default.
See the README's D4 note.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import constants, evaluation
from ..models import xgb_baseline
from . import isolated, views as fv

#: The condition's name, written into every row.
CONDITION = "centralized"

#: Phase 3's recorded centralized result at L = 48, test, subset ``all``, from
#: ``docs/reference/evaluate_baseline.csv``. Per MODEL seed - which is the
#: whole difficulty, because the headline 0.679 / 0.760 is their mean over
#: five seeds and Phase 4's scope is three.
PHASE3_L48_TEST: dict[int, dict[str, float]] = {
    0: {"campaign_recall": 0.6964285714285714, "relative": 0.78, "n_detected": 39},
    1: {"campaign_recall": 0.6607142857142857, "relative": 0.74, "n_detected": 37},
    2: {"campaign_recall": 0.6964285714285714, "relative": 0.78, "n_detected": 39},
    3: {"campaign_recall": 0.6607142857142857, "relative": 0.74, "n_detected": 37},
    4: {"campaign_recall": 0.6785714285714286, "relative": 0.76, "n_detected": 38},
}

#: The ceiling Phase 3 recorded at L = 48, identical at every seed. Stage 4.1
#: predicts the union's ceiling equals it exactly.
PHASE3_L48_CEILING = 0.8928571428571429

#: The headline in the README, and what D4 asks a retrain to reproduce. It is
#: the mean over the five seeds above.
PHASE3_HEADLINE = {"campaign_recall": 0.6785714285714286, "relative": 0.76}

#: The three defensible readings of "centralized under the Phase 4 scope".
#: None is a default: they differ by more than a tenth of the effect being
#: measured, so the caller states which one it means.
MODEL_SEED_POLICIES = {
    # Matched to the federated arms: every Phase 4 booster is model seed 0, so
    # this is the only reading in which the arms differ ONLY in the condition.
    # Constant across all 12 cells, because a centralized model cannot depend
    # on a visibility assignment.
    "matched_seed0": (0,),
    # Phase 4's three seeds read as model seeds. Fits "3 seeds" literally, but
    # makes the centralized arm the only one whose MODEL changes from cell to
    # cell, so its cell-to-cell spread is model noise where every other arm's
    # is visibility noise.
    "phase4_seeds_as_model_seeds": (0, 1, 2),
    # Phase 3's five seeds, which is what reproduces the 0.679 / 0.760
    # headline. Outside the Phase 4 scope of three seeds.
    "phase3_five_seeds": (0, 1, 2, 3, 4),
}


#: The policy Phase 4 uses, decided by the user as O1 on 2026-11-15.
#: Recorded here so every runner reads the decision from one place rather than
#: repeating a string. Phase 4 studies fragmentation, not model seed: all 96
#: federated boosters are model seed 0, so the centralized comparator must be
#: model seed 0 or the contrast carries a model-seed difference inside it.
#: `phase4_seeds_as_model_seeds` was rejected explicitly - it would make
#: centralized the only arm whose model varies per cell, so its spread would be
#: model noise against every other arm's visibility noise.
PHASE4_POLICY = "matched_seed0"


def model_seed_policy(name: str) -> tuple[int, ...]:
    """The model seeds a named policy uses. Raises on an unknown name.

    There is deliberately no default. Picking one silently would change a
    headline number - the centralized comparator moves between 0.760 and 0.780
    relative recall depending on the answer - and that is exactly the kind of
    choice this project requires to be stated.
    """
    if name not in MODEL_SEED_POLICIES:
        raise ValueError(
            f"unknown model-seed policy {name!r}; one of "
            f"{sorted(MODEL_SEED_POLICIES)} must be named explicitly, because "
            "they give different centralized numbers"
        )
    return MODEL_SEED_POLICIES[name]


def phase3_reference(seeds) -> dict:
    """Phase 3's recorded mean over the given model seeds.

    Used for the reproduction check D4 asks for. Reads the committed table
    above rather than recomputing, so the check compares this run against the
    record and not against itself.
    """
    seeds = list(seeds)
    missing = [s for s in seeds if s not in PHASE3_L48_TEST]
    if missing:
        raise ValueError(f"no Phase 3 record for model seed(s) {missing}")
    return {
        "model_seeds": seeds,
        "campaign_recall": float(
            np.mean([PHASE3_L48_TEST[s]["campaign_recall"] for s in seeds])
        ),
        "relative": float(np.mean([PHASE3_L48_TEST[s]["relative"] for s in seeds])),
        "oracle_recall": PHASE3_L48_CEILING,
    }


# ---------------------------------------------------------------------------
# the degenerate assignment: K = 1
# ---------------------------------------------------------------------------


def single_institution_assignment(result) -> fv.VisibilityAssignment:
    """Every account in institution 0, so the view is the whole dataset.

    ``seed = -1`` and ``target = 1.0`` follow ``views.natural_assignment``'s
    convention for an assignment that is not an experiment run, so this can
    never be mistaken for a visibility cell.
    """
    return fv.VisibilityAssignment(
        seed=-1,
        target=1.0,
        global_assignment={},
        bank2inst={bank: 0 for bank in result.bank2inst},
        k=1,
    )


def as_model(
    booster: xgb_baseline.TrainedBaseline,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    n_rows: int = 0,
    n_positive: int = 0,
) -> isolated.IsolatedModel:
    """Wrap a Phase 3 booster so the Phase 4 evaluator accepts it."""
    return isolated.IsolatedModel(
        institution=0,
        lookback_h=lookback_h,
        tier=isolated.TIER_NORMAL,
        n_rows=n_rows,
        n_positive=n_positive,
        model=booster,
        training={"source": "phase 3 centralized booster, not retrained"},
    )


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def evaluate_once(
    result,
    booster: xgb_baseline.TrainedBaseline,
    model_seed: int,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    bootstrap_seed: int = 0,
    progress=None,
) -> isolated.InstitutionEvaluation:
    """One centralized detection run: tau, ceiling, every operating point.

    Independent of ``(target, seed)`` by construction - a centralized model
    sees every transaction whatever the visibility assignment does, and the
    assignment moves accounts between institutions without touching a single
    row. That invariance is asserted by a test rather than assumed, and it is
    why :func:`run` evaluates once and replicates.
    """
    assignment = single_institution_assignment(result)
    model = as_model(booster, lookback_h)
    return isolated.evaluate_institution(
        result.full_df,
        result.camp,
        result.pattern_df,
        result.accounts_by_campaign,
        model,
        assignment,
        lookback_h=lookback_h,
        seed=bootstrap_seed,
        splits=splits,
        points=isolated.operating_points(1),
        progress=progress,
        condition="%s_s%d" % (CONDITION, model_seed),
    )


def run(
    result,
    boosters: dict[int, xgb_baseline.TrainedBaseline],
    policy: str,
    targets=None,
    seeds=None,
    lookback_h: int = constants.PHASE4_LOOKBACK_H,
    splits: tuple[str, ...] = ("val", "test", "stress"),
    progress=None,
) -> dict:
    """Stage 4.4 over the whole Phase 4 cell grid.

    ``boosters`` maps model seed -> the Phase 3 booster for that seed;
    ``policy`` names which of those seeds to use (see
    :data:`MODEL_SEED_POLICIES`). The detection run happens once per model
    seed and is then **replicated** across every ``(target, seed)`` cell, with
    the cell's own seed used as the bootstrap seed. Replication rather than
    re-running is not a shortcut: re-running would produce the same outcomes
    by construction, and claiming 12 measurements where there are
    ``len(seeds)`` would misstate the evidence. Every row carries
    ``replicated_from_model_seed`` and ``n_independent_runs`` so a reader
    cannot miss it.
    """
    started = time.perf_counter()
    say = progress or (lambda *_: None)
    model_seeds = model_seed_policy(policy)
    targets = list(targets if targets is not None else constants.VISIBILITY_TARGETS)
    seeds = list(seeds if seeds is not None else constants.PHASE4_SEEDS)

    missing = [s for s in model_seeds if s not in boosters]
    if missing:
        raise ValueError(
            f"policy {policy!r} needs model seed(s) {missing}, which were not "
            "supplied. Attach the Phase 3 booster output for them, or choose a "
            "policy whose seeds you have."
        )

    points = isolated.operating_points(1)
    evaluations: dict[int, isolated.InstitutionEvaluation] = {}
    for model_seed in model_seeds:
        say(f"  centralized, model seed {model_seed} ...")
        evaluations[model_seed] = evaluate_once(
            result, boosters[model_seed], model_seed,
            lookback_h=lookback_h, splits=splits,
            bootstrap_seed=model_seed, progress=say,
        )

    rows: list[dict] = []
    tau_rows: list[dict] = []
    ceiling_rows: list[dict] = []
    outcome_rows: list[dict] = []

    for target in targets:
        for seed in seeds:
            common = {
                "condition": CONDITION,
                "model": "xgboost",
                "variant": "xgboost_centralized",
                "visibility_target": target,
                "lookback_h": lookback_h,
                "seed": seed,
                "model_seed_policy": policy,
                "n_independent_runs": len(model_seeds),
            }
            for model_seed in model_seeds:
                ev = evaluations[model_seed]
                for point in points:
                    name = point["operating_point"]
                    tau_rows.append({
                        **common, "institution": CONDITION,
                        "operating_point": name,
                        "point_kind": point["kind"],
                        "budget": point["budget"],
                        "per_institution_budget": point["per_institution_budget"],
                        "primary": point["primary"],
                        "replicated_from_model_seed": model_seed,
                        "tau": ev.taus.get(name),
                        "defined": ev.taus.get(name) is not None,
                        "rejected_tau": ev.rejected_taus.get(name),
                        "tau_edge": ev.tau_edges.get(name),
                        "budget_binding": ev.budget_binding.get(name),
                        "undefined_reason": ev.undefined_reason.get(name),
                        "f1_best_value": ev.f1_best_value,
                        "f1_degenerate": ev.f1_degenerate,
                        "f1_degenerate_epsilon": isolated.F1_DEGENERATE_EPSILON,
                    })
                    for split in splits:
                        res = ev.results.get((name, split))
                        if res is None:
                            for subset in constants.TEST_SUBSETS:
                                rows.append({
                                    **common, "institution": CONDITION,
                                    "operating_point": name,
                                    "eval_split": split, "test_subset": subset,
                                    "replicated_from_model_seed": model_seed,
                                    "defined": False, "tau": None,
                                    "rejected_tau": ev.rejected_taus.get(name),
                                    "budget_binding": ev.budget_binding.get(name),
                                    "f1_best_value": ev.f1_best_value,
                                    "f1_degenerate": ev.f1_degenerate,
                                    "undefined_reason": ev.undefined_reason.get(
                                        name, isolated.NO_AFFORDABLE_TAU
                                    ),
                                })
                            continue
                        for subset in constants.TEST_SUBSETS:
                            summary = evaluation.summarize(
                                res, subset, bootstrap_seed=seed,
                                oracle_reachable=ev.oracle_reachable[split],
                                oracle_recall=ev.ceiling[(split, subset)],
                            )
                            rows.append({
                                **common, "institution": CONDITION,
                                "operating_point": name, "eval_split": split,
                                "replicated_from_model_seed": model_seed,
                                "defined": True,
                                "budget_binding": ev.budget_binding.get(name),
                                "f1_best_value": ev.f1_best_value,
                                "f1_degenerate": ev.f1_degenerate,
                                "undefined_reason": None,
                                "primary": point["primary"],
                                **summary,
                            })
                        outcome_rows.extend(evaluation.outcome_records(
                            res, **common, institution=CONDITION,
                            operating_point=name, tier="centralized",
                        ))

    for model_seed in model_seeds:
        ev = evaluations[model_seed]
        for (split, subset), value in sorted(ev.ceiling.items()):
            ceiling_rows.append({
                "condition": CONDITION, "lookback_h": lookback_h,
                "model_seed": model_seed, "eval_split": split,
                "test_subset": subset, "oracle_recall": value,
                "n_oracle_reachable": len(ev.oracle_reachable.get(split, [])),
            })

    return {
        "policy": policy,
        "model_seeds": list(model_seeds),
        "targets": targets,
        "seeds": seeds,
        "lookback_h": lookback_h,
        "rows": rows,
        "tau_rows": tau_rows,
        "ceiling_rows": ceiling_rows,
        "outcome_rows": outcome_rows,
        "reproduction": reproduction_check(evaluations, model_seeds),
        "seconds": round(time.perf_counter() - started, 1),
    }


def reproduction_check(
    evaluations: dict[int, isolated.InstitutionEvaluation],
    model_seeds,
) -> dict:
    """Does this run reproduce Phase 3's recorded centralized result?

    Compares the PRIMARY ``f1`` point on test, subset ``all``, against
    ``docs/reference/evaluate_baseline.csv`` as transcribed in
    :data:`PHASE3_L48_TEST`, per seed and in the mean. Reported rather than
    asserted, because the honest answer depends on which seeds the policy
    chose - and a mismatch against the 0.679 headline is *expected* for any
    policy that is not the five-seed one.
    """
    per_seed = []
    for model_seed in model_seeds:
        ev = evaluations[model_seed]
        res = ev.results.get(("f1", "test"))
        recorded = PHASE3_L48_TEST.get(model_seed)
        measured = None
        if res is not None:
            summary = evaluation.summarize(
                res, "all", bootstrap_seed=model_seed,
                oracle_reachable=ev.oracle_reachable["test"],
                oracle_recall=ev.ceiling[("test", "all")],
            )
            measured = {
                "campaign_recall": summary["campaign_recall"],
                "relative": summary["recall_relative_to_ceiling"],
                "oracle_recall": summary["oracle_recall"],
                "tau": ev.taus.get("f1"),
            }
        per_seed.append({
            "model_seed": model_seed,
            "measured": measured,
            "phase3_recorded": recorded,
            "recall_matches": (
                measured is not None and recorded is not None
                and abs(measured["campaign_recall"] - recorded["campaign_recall"]) < 5e-3
            ),
            "ceiling_matches": (
                measured is not None
                and abs(measured["oracle_recall"] - PHASE3_L48_CEILING) < 5e-4
            ),
        })
    measured_means = [
        s["measured"]["campaign_recall"] for s in per_seed if s["measured"]
    ]
    return {
        "per_seed": per_seed,
        "measured_mean_recall": (
            float(np.mean(measured_means)) if measured_means else None
        ),
        "phase3_mean_over_these_seeds": phase3_reference(model_seeds),
        "phase3_headline": PHASE3_HEADLINE,
        "headline_is_five_seed_mean": True,
    }


__all__ = [
    "CONDITION",
    "PHASE4_POLICY",
    "MODEL_SEED_POLICIES",
    "PHASE3_HEADLINE",
    "PHASE3_L48_CEILING",
    "PHASE3_L48_TEST",
    "as_model",
    "evaluate_once",
    "model_seed_policy",
    "phase3_reference",
    "reproduction_check",
    "run",
    "single_institution_assignment",
]
