"""The Phase 4 comparison: isolated vs fedavg vs +evidence vs centralized.

**Rule 1, enforced rather than assumed.** Every contrast is a **paired
difference over identical cells**: per ``(target, seed, institution,
operating_point, eval_split, test_subset)`` for a per-institution reading, and
per ``(target, seed, operating_point, eval_split, test_subset)`` for the union
and for anything compared against centralized. If the two sides do not cover
exactly the same cells, :func:`paired` raises and names the difference. That
is a stop condition for the phase, so it must stop the code too - a comparison
over nearly-identical cells is the failure mode that looks like a result.

**Three things the pairing has to get right, and does.**

1. **An undefined cell removes the pair, not just its own side.** Stage 4.1's
   grid-edge policy leaves some ``(institution, operating point)`` cells
   undefined. If one arm is undefined there and the other is not, keeping the
   defined side would compare arm A over 8 institutions against arm B over 7.
   Both sides are dropped together and the count is reported.
2. **The ceilings must agree between arms.** 4.2 and 4.3 change scoring, never
   visibility, so every per-institution oracle ceiling must be identical
   across the four per-institution conditions. This is the measurable form of
   the stage 4.3 result that evidence cannot raise a ceiling, so it is checked
   on real data and raises if it ever fails.
3. **The per-institution mean is never a standard error.** The eight
   institutions are not exchangeable, so a difference is summarised by its
   mean over institutions together with its median, range and the number of
   institutions in which it has each sign - never by a t-statistic across
   them. Intervals come from the per-seed bootstrap already in the summary
   rows.

**The seed-noise floor.** Phase 4's three seeds vary the visibility assignment
only: all boosters are trained at model seed 0. Model-seed noise is therefore
held fixed and invisible here, but Phase 3 measures it at centralized -
0.661-0.696 recall, 0.74-0.78 relative over five model seeds. Every difference
this module reports carries ``below_seed_noise_floor``, true when its absolute
value is under that spread, so a difference smaller than a quantity the design
cannot see is never presented as an effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import constants
from . import centralized as centralized_mod, evidence, fedavg, isolated

#: The four per-institution conditions, floor first.
PER_INSTITUTION_CONDITIONS = (
    isolated.CONDITION,
    fedavg.CONDITION,
    evidence.CONDITION_EVIDENCE,
    evidence.CONDITION_FEDAVG_EVIDENCE,
)

#: Measured at centralized across Phase 3's five MODEL seeds, which is the one
#: place this project measures model-seed noise at all. 0.035 recall is exactly
#: 2 campaigns of 56. A Phase 4 difference smaller than this is inside the
#: noise of a quantity Phase 4 holds fixed, so it is flagged and must not be
#: written up as an effect.
SEED_NOISE_FLOOR = {
    "campaign_recall": 0.0357142857142857,
    "recall_relative_to_ceiling": 0.04,
}

#: The contrasts, in the order the report presents them. Each one answers a
#: question that the other arms cannot, which is why all of them are reported
#: rather than only the headline.
CONTRASTS = (
    {
        "name": "fedavg - isolated",
        "baseline": isolated.CONDITION,
        "treatment": fedavg.CONDITION,
        "question": "what does sharing MODEL KNOWLEDGE buy, at unchanged visibility",
    },
    {
        "name": "evidence - isolated",
        "baseline": isolated.CONDITION,
        "treatment": evidence.CONDITION_EVIDENCE,
        "question": "what does sharing BOUNDARY EVIDENCE buy, own model (4.3a vs 4.1)",
    },
    {
        "name": "fedavg_evidence - fedavg",
        "baseline": fedavg.CONDITION,
        "treatment": evidence.CONDITION_FEDAVG_EVIDENCE,
        "question": "what does boundary evidence buy UNDER federation (4.3b vs 4.2)",
    },
    {
        "name": "fedavg_evidence - isolated",
        "baseline": isolated.CONDITION,
        "treatment": evidence.CONDITION_FEDAVG_EVIDENCE,
        "question": "both mechanisms together, against the floor",
    },
    {
        "name": "fedavg_evidence - evidence",
        "baseline": evidence.CONDITION_EVIDENCE,
        "treatment": evidence.CONDITION_FEDAVG_EVIDENCE,
        "question": "does score averaging still add anything once evidence is shared",
    },
)

#: Cell keys for the two pairing granularities.
PER_INSTITUTION_KEYS = (
    "visibility_target", "seed", "institution", "operating_point",
    "eval_split", "test_subset",
)
CELL_KEYS = (
    "visibility_target", "seed", "operating_point", "eval_split", "test_subset",
)

#: The metrics a contrast is taken on. Relative recall is the one the phase
#: argues about, because each arm is measured against its own ceiling; raw
#: recall is carried beside it so a reader can see both.
METRICS = ("campaign_recall", "recall_relative_to_ceiling")


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


class NonIdenticalCells(RuntimeError):
    """Raised when a contrast would compare over cells that do not match.

    A stop condition for the phase, and therefore for the code. The message
    names what is missing from which side, because the useful next action is
    always to find the run that did not finish rather than to relax the rule.
    """


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with a defined operating point and a value to compare.

    ``defined`` is false where the grid-edge policy or an unaffordable budget
    left a cell with no operating point. Such a cell was not measured, so it
    cannot enter a difference - and neither can its partner.
    """
    out = frame
    if "defined" in out.columns:
        out = out[out["defined"].astype(str).str.lower().isin({"true", "1"})]
    return out


def paired(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    keys,
    metrics=METRICS,
    label: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Treatment minus baseline, over exactly the cells both sides define.

    Returns the paired frame and a provenance dict recording how many cells
    each side had, how many pairs survived, and which cells were dropped
    because one side was undefined. Nothing is silently discarded.

    Raises :class:`NonIdenticalCells` when the two sides' cell *grids* differ
    - i.e. when a cell is absent altogether rather than present-but-undefined.
    An absent cell means a run did not cover it, which is a different problem
    from a cell the policy excluded, and it must not be papered over.
    """
    keys = list(keys)
    missing_cols = [
        c for c in keys if c not in baseline.columns or c not in treatment.columns
    ]
    if missing_cols:
        raise NonIdenticalCells(
            f"{label or 'contrast'}: missing key column(s) {missing_cols}"
        )

    # Duplicate keys would make the merge below a cross product and inflate
    # `n_pairs` - the one number a reader uses to judge how much the
    # difference rests on. Caught here rather than discovered from a
    # suspiciously round denominator.
    for side, frame in (("baseline", baseline), ("treatment", treatment)):
        counts = frame.groupby(keys, dropna=False).size()
        repeated = counts[counts > 1]
        if len(repeated):
            raise NonIdenticalCells(
                f"{label or 'contrast'}: the {side} has {len(repeated)} "
                f"duplicated cell key(s), e.g. {repeated.index[0]} appearing "
                f"{int(repeated.iloc[0])} times. A merge on a duplicated key "
                "is a cross product, so every pair count downstream would be "
                "inflated. Deduplicate the input or widen the key."
            )

    left_all = set(map(tuple, baseline[keys].to_numpy().tolist()))
    right_all = set(map(tuple, treatment[keys].to_numpy().tolist()))
    only_left = left_all - right_all
    only_right = right_all - left_all
    if only_left or only_right:
        raise NonIdenticalCells(
            f"{label or 'contrast'}: the two arms do not cover the same cells. "
            f"{len(only_left)} cell(s) only in the baseline "
            f"(e.g. {sorted(only_left)[:3]}), "
            f"{len(only_right)} only in the treatment "
            f"(e.g. {sorted(only_right)[:3]}). A paired comparison over "
            "non-identical cells is not a paired comparison; find the run that "
            "did not finish."
        )

    left = usable(baseline)
    right = usable(treatment)
    left_defined = set(map(tuple, left[keys].to_numpy().tolist()))
    right_defined = set(map(tuple, right[keys].to_numpy().tolist()))
    both = left_defined & right_defined
    dropped = sorted((left_defined | right_defined) - both)

    cols = keys + [m for m in metrics if m in left.columns and m in right.columns]
    merged = left[cols].merge(
        right[cols], on=keys, suffixes=("_baseline", "_treatment"), how="inner"
    )
    for metric in metrics:
        a, b = f"{metric}_baseline", f"{metric}_treatment"
        if a in merged.columns and b in merged.columns:
            merged[f"d_{metric}"] = (
                pd.to_numeric(merged[b], errors="coerce")
                - pd.to_numeric(merged[a], errors="coerce")
            )

    provenance = {
        "label": label,
        "cells_in_grid": len(left_all),
        "pairs_compared": int(len(merged)),
        "pairs_dropped_undefined_on_one_side": len(dropped),
        "dropped_cells": [list(c) for c in dropped[:20]],
    }
    return merged, provenance


# ---------------------------------------------------------------------------
# checks that must hold before any difference is reported
# ---------------------------------------------------------------------------


def check_ceilings_agree(
    frames: dict[str, pd.DataFrame], keys=PER_INSTITUTION_KEYS
) -> dict:
    """Every per-institution condition must share every oracle ceiling.

    The measurable form of the stage 4.3 result: 4.2 and 4.3 change what a
    model knows, never which rows it has, so the visibility mask - and hence
    the oracle through it - is identical across all four arms. If this ever
    fails, a condition has moved visibility and no contrast built on it means
    anything, so it raises.
    """
    keys = [k for k in keys if k != "operating_point"]
    reference = None
    report = {}
    for condition, frame in sorted(frames.items()):
        if "oracle_recall" not in frame.columns:
            continue
        ceiling = (
            usable(frame)[keys + ["oracle_recall"]]
            .drop_duplicates()
            .sort_values(keys)
            .reset_index(drop=True)
        )
        report[condition] = int(len(ceiling))
        if reference is None:
            reference, reference_name = ceiling, condition
            continue
        merged = reference.merge(
            ceiling, on=keys, suffixes=("_ref", "_this"), how="inner"
        )
        delta = (
            pd.to_numeric(merged["oracle_recall_this"], errors="coerce")
            - pd.to_numeric(merged["oracle_recall_ref"], errors="coerce")
        ).abs()
        worst = float(delta.max()) if len(delta) else 0.0
        if worst > 1e-9:
            offending = merged.loc[delta.idxmax(), keys].to_dict()
            raise NonIdenticalCells(
                f"condition {condition!r} has a different oracle ceiling from "
                f"{reference_name!r} (max difference {worst:.6g} at "
                f"{offending}). 4.2 and 4.3 must not change what an "
                "institution can see; a moved ceiling means the visibility "
                "mask moved and every contrast built on it is invalid."
            )
    return {"n_ceiling_cells": report, "max_disagreement": 0.0}


def check_union_ceiling_matches_centralized(
    union: pd.DataFrame, centralized: pd.DataFrame
) -> dict:
    """The stage 4.1 structural claim, re-checked wherever both are present.

    The union of the eight views must reach exactly the campaigns the
    centralized model can reach, so the two oracle recalls must be equal. A
    disagreement would mean the frozen visibility argument is wrong, which
    matters more than anything else in the report.
    """
    keys = ["visibility_target", "seed", "eval_split", "test_subset"]
    left = usable(union)[keys + ["oracle_recall"]].drop_duplicates()
    right = usable(centralized)[keys + ["oracle_recall"]].drop_duplicates()
    merged = left.merge(right, on=keys, suffixes=("_union", "_centralized"))
    if not len(merged):
        return {"compared": 0, "max_difference": None, "agrees": None}
    delta = (
        pd.to_numeric(merged["oracle_recall_union"], errors="coerce")
        - pd.to_numeric(merged["oracle_recall_centralized"], errors="coerce")
    ).abs()
    worst = float(delta.max())
    return {
        "compared": int(len(merged)),
        "max_difference": worst,
        "agrees": bool(worst <= 1e-9),
    }


# ---------------------------------------------------------------------------
# summarising a difference
# ---------------------------------------------------------------------------


def summarise_difference(
    frame: pd.DataFrame, metric: str, group, unit: str
) -> list[dict]:
    """Mean, median and range of a paired difference, with its denominator.

    No standard error is taken across institutions: they are not exchangeable
    (institution 0 deterministically holds the dataset's largest bank), so a
    spread across them is a description of this fixed panel and not an
    estimate. ``n_positive`` / ``n_negative`` say in how many cells the
    difference had each sign, which is the honest robustness statement
    available here.
    """
    column = f"d_{metric}"
    if column not in frame.columns:
        return []
    floor = SEED_NOISE_FLOOR.get(metric)
    rows = []
    for key, chunk in frame.groupby(list(group), dropna=False, sort=True):
        values = pd.to_numeric(chunk[column], errors="coerce").dropna()
        if not len(values):
            continue
        mean = float(values.mean())
        row = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        row.update({
            "unit": unit,
            "metric": metric,
            "n_pairs": int(len(values)),
            "mean_difference": mean,
            "median_difference": float(values.median()),
            "min_difference": float(values.min()),
            "max_difference": float(values.max()),
            "n_positive": int((values > 0).sum()),
            "n_zero": int((values == 0).sum()),
            "n_negative": int((values < 0).sum()),
            "baseline_mean": float(
                pd.to_numeric(chunk[f"{metric}_baseline"], errors="coerce").mean()
            ),
            "treatment_mean": float(
                pd.to_numeric(chunk[f"{metric}_treatment"], errors="coerce").mean()
            ),
            "seed_noise_floor": floor,
            "below_seed_noise_floor": (
                None if floor is None else bool(abs(mean) < floor)
            ),
        })
        rows.append(row)
    return rows


def compare(
    summaries: dict[str, pd.DataFrame],
    unions: dict[str, pd.DataFrame],
    centralized_rows: pd.DataFrame | None = None,
    progress=None,
) -> dict:
    """The whole Phase 4 comparison.

    ``summaries`` maps condition -> that condition's per-institution summary
    rows; ``unions`` maps condition -> its union rows; ``centralized_rows`` is
    stage 4.4's output. Any arm may be absent - the report then covers the
    contrasts it can form and lists the ones it skipped, which is what lets
    stage 4.3 be dropped without invalidating 4.1 / 4.2 / 4.4.
    """
    say = progress or (lambda *_: None)
    rows: list[dict] = []
    provenance: list[dict] = []
    skipped: list[dict] = []

    present = [c for c in PER_INSTITUTION_CONDITIONS if c in summaries]
    say("  conditions present: %s" % ", ".join(present) or "(none)")

    checks = {
        "ceilings_agree_across_conditions": check_ceilings_agree(
            {c: summaries[c] for c in present}
        ) if len(present) > 1 else {"n_ceiling_cells": {}, "max_disagreement": None},
    }

    # --- condition against condition ---------------------------------
    for contrast in CONTRASTS:
        base, treat = contrast["baseline"], contrast["treatment"]
        if base not in summaries or treat not in summaries:
            skipped.append({**contrast, "reason": "one arm not present"})
            continue
        for unit, keys, group, source in (
            ("per_institution", PER_INSTITUTION_KEYS,
             ("visibility_target", "seed", "operating_point", "eval_split",
              "test_subset"), summaries),
            ("union", CELL_KEYS,
             ("visibility_target", "seed", "operating_point", "eval_split",
              "test_subset"), unions),
        ):
            if base not in source or treat not in source:
                continue
            merged, prov = paired(
                source[base], source[treat], keys,
                label=f"{contrast['name']} [{unit}]",
            )
            provenance.append({**prov, "contrast": contrast["name"], "unit": unit})
            for metric in METRICS:
                for row in summarise_difference(merged, metric, group, unit):
                    rows.append({
                        "contrast": contrast["name"],
                        "baseline_condition": base,
                        "treatment_condition": treat,
                        "question": contrast["question"],
                        **row,
                    })

    # --- each arm against centralized --------------------------------
    if centralized_rows is not None and len(centralized_rows):
        checks["union_ceiling_equals_centralized"] = (
            check_union_ceiling_matches_centralized(
                pd.concat([unions[c] for c in unions], ignore_index=True),
                centralized_rows,
            ) if unions else {"compared": 0, "agrees": None}
        )
        for condition in present:
            for unit, source in (("union_vs_centralized", unions),):
                if condition not in source:
                    continue
                merged, prov = paired(
                    source[condition], centralized_rows, CELL_KEYS,
                    label=f"centralized - {condition} [{unit}]",
                )
                provenance.append({
                    **prov, "contrast": f"centralized - {condition}", "unit": unit,
                })
                for metric in METRICS:
                    for row in summarise_difference(
                        merged, metric,
                        ("visibility_target", "seed", "operating_point",
                         "eval_split", "test_subset"),
                        unit,
                    ):
                        rows.append({
                            "contrast": f"centralized - {condition}",
                            "baseline_condition": condition,
                            "treatment_condition": centralized_mod.CONDITION,
                            "question": (
                                "how much of the centralized result is still "
                                f"out of reach for {condition}"
                            ),
                            **row,
                        })
    else:
        skipped.append({
            "name": "centralized contrasts", "reason": "no centralized rows",
        })

    frame = pd.DataFrame(rows)
    if len(frame):
        frame["primary"] = frame["operating_point"].isin(isolated.PRIMARY_POINTS)
        frame = frame.sort_values(
            ["primary", "contrast", "visibility_target", "seed", "metric", "unit"],
            ascending=[False, True, False, True, True, True],
        ).reset_index(drop=True)

    return {
        "rows": frame.to_dict("records") if len(frame) else [],
        "checks": checks,
        "provenance": provenance,
        "skipped": skipped,
        "conditions_present": present,
        "centralized_present": bool(
            centralized_rows is not None and len(centralized_rows)
        ),
        "primary_points": list(isolated.PRIMARY_POINTS),
        "seed_noise_floor": dict(SEED_NOISE_FLOOR),
    }


def headline(rows, split: str = "test", subset: str = "all") -> pd.DataFrame:
    """The primary-point view: `f1` and `total_10`, test, subset `all`."""
    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    keep = (
        frame["operating_point"].isin(isolated.PRIMARY_POINTS)
        & (frame["eval_split"] == split)
        & (frame["test_subset"] == subset)
        & (frame["metric"] == "recall_relative_to_ceiling")
    )
    return frame[keep].sort_values(
        ["contrast", "unit", "visibility_target", "seed"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


__all__ = [
    "CELL_KEYS",
    "CONTRASTS",
    "METRICS",
    "NonIdenticalCells",
    "PER_INSTITUTION_CONDITIONS",
    "PER_INSTITUTION_KEYS",
    "SEED_NOISE_FLOOR",
    "check_ceilings_agree",
    "check_union_ceiling_matches_centralized",
    "compare",
    "headline",
    "paired",
    "summarise_difference",
    "usable",
]
