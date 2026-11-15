"""The Phase 4 comparison: rule 1, enforced rather than assumed.

This module is the one place a Phase 4 claim becomes a number, so the tests
are about the ways a difference could look fine and mean nothing:

1. **comparing over cells that are not the same cells** - a stop condition for
   the phase, so it must stop the code;
2. **a cell undefined on one side** - keeping the defined side would compare
   8 institutions against 7;
3. **a duplicated cell key** - the merge becomes a cross product and `n_pairs`,
   the one number a reader judges weight by, silently inflates;
4. **a ceiling that moved** - then a condition changed visibility and no
   contrast built on it means anything;
5. **a difference inside the model-seed noise** - reported, but flagged, since
   Phase 4 holds the model seed fixed and cannot see it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraudcamp.federation import compare as compare_mod


def row(condition, target, seed, institution, point, recall, relative,
        split="test", subset="all", defined=True, ceiling=0.8928571428571429):
    return {
        "condition": condition, "visibility_target": target, "seed": seed,
        "institution": institution, "operating_point": point,
        "eval_split": split, "test_subset": subset, "defined": defined,
        "campaign_recall": recall, "recall_relative_to_ceiling": relative,
        "oracle_recall": ceiling,
    }


def frame(condition, *, n_institutions=2, recall=0.4, relative=0.5,
          point="total_10", defined=True, ceiling=0.8928571428571429):
    return pd.DataFrame([
        row(condition, 1.0, 0, i, point, recall, relative, defined=defined,
            ceiling=ceiling)
        for i in range(n_institutions)
    ])


# ---------------------------------------------------------------------------
# 1. non-identical cells stop the code
# ---------------------------------------------------------------------------


def test_cells_that_do_not_match_raise_and_name_the_difference():
    baseline = frame("isolated", n_institutions=3)
    treatment = frame("fedavg", n_institutions=2)
    with pytest.raises(compare_mod.NonIdenticalCells) as caught:
        compare_mod.paired(
            baseline, treatment, compare_mod.PER_INSTITUTION_KEYS,
            label="fedavg - isolated",
        )
    message = str(caught.value)
    assert "fedavg - isolated" in message
    assert "do not cover the same cells" in message
    assert "1 cell(s) only in the baseline" in message
    assert "did not finish" in message, "the message must say what to do"


def test_a_missing_key_column_raises_rather_than_producing_a_partial_pairing():
    baseline = frame("isolated")
    treatment = frame("fedavg").drop(columns=["institution"])
    with pytest.raises(compare_mod.NonIdenticalCells, match="missing key column"):
        compare_mod.paired(
            baseline, treatment, compare_mod.PER_INSTITUTION_KEYS,
        )


# ---------------------------------------------------------------------------
# 2. an undefined cell removes the pair, not just its own side
# ---------------------------------------------------------------------------


def test_an_undefined_cell_on_one_side_removes_the_pair():
    """POLICY: otherwise arm A is averaged over 8 and arm B over 7."""
    baseline = pd.DataFrame([
        row("isolated", 1.0, 0, 0, "f1", 0.4, 0.5),
        row("isolated", 1.0, 0, 1, "f1", None, None, defined=False),
    ])
    treatment = pd.DataFrame([
        row("fedavg", 1.0, 0, 0, "f1", 0.6, 0.7),
        row("fedavg", 1.0, 0, 1, "f1", 0.5, 0.6),
    ])
    merged, provenance = compare_mod.paired(
        baseline, treatment, compare_mod.PER_INSTITUTION_KEYS, label="x",
    )
    assert provenance["cells_in_grid"] == 2
    assert provenance["pairs_compared"] == 1, (
        "the pair whose baseline was undefined must be dropped entirely"
    )
    assert provenance["pairs_dropped_undefined_on_one_side"] == 1
    assert len(merged) == 1
    assert merged["d_campaign_recall"].iloc[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 3. a duplicated key would inflate n_pairs
# ---------------------------------------------------------------------------


def test_a_duplicated_cell_key_raises_instead_of_inflating_n_pairs():
    """A merge on a duplicated key is a cross product.

    `n_pairs` is the number a reader uses to judge how much a difference
    rests on, so an inflated one is worse than a missing one.
    """
    baseline = frame("isolated")
    doubled = pd.concat([frame("fedavg"), frame("fedavg")], ignore_index=True)
    with pytest.raises(compare_mod.NonIdenticalCells) as caught:
        compare_mod.paired(
            baseline, doubled, compare_mod.PER_INSTITUTION_KEYS, label="x",
        )
    message = str(caught.value)
    assert "duplicated cell key" in message
    assert "cross product" in message


# ---------------------------------------------------------------------------
# 4. the ceilings must agree across conditions
# ---------------------------------------------------------------------------


def test_the_ceilings_must_agree_across_every_per_institution_condition():
    """The measurable form of the stage 4.3 result.

    4.2 and 4.3 change what a model knows, never which rows it has, so every
    per-institution oracle ceiling must be identical across the four arms. A
    moved ceiling means the visibility mask moved.
    """
    agreeing = {
        "isolated": frame("isolated"),
        "fedavg": frame("fedavg"),
        "evidence": frame("evidence"),
    }
    report = compare_mod.check_ceilings_agree(agreeing)
    assert report["max_disagreement"] == 0.0
    assert set(report["n_ceiling_cells"]) == set(agreeing)

    moved = dict(agreeing)
    moved["evidence"] = frame("evidence", ceiling=0.75)
    with pytest.raises(compare_mod.NonIdenticalCells) as caught:
        compare_mod.check_ceilings_agree(moved)
    message = str(caught.value)
    assert "different oracle ceiling" in message
    assert "visibility mask moved" in message


def test_the_union_ceiling_is_checked_against_the_centralized_one():
    union = pd.DataFrame([
        {"visibility_target": 1.0, "seed": 0, "eval_split": "test",
         "test_subset": "all", "oracle_recall": 50 / 56, "defined": True},
    ])
    centralized = pd.DataFrame([
        {"visibility_target": 1.0, "seed": 0, "eval_split": "test",
         "test_subset": "all", "oracle_recall": 50 / 56, "defined": True},
    ])
    agrees = compare_mod.check_union_ceiling_matches_centralized(
        union, centralized
    )
    assert agrees["compared"] == 1
    assert agrees["agrees"] is True
    assert agrees["max_difference"] == pytest.approx(0.0)

    centralized.loc[0, "oracle_recall"] = 0.95
    disagrees = compare_mod.check_union_ceiling_matches_centralized(
        union, centralized
    )
    assert disagrees["agrees"] is False
    assert disagrees["max_difference"] > 0


# ---------------------------------------------------------------------------
# 5. the seed-noise floor, and what is NOT reported across institutions
# ---------------------------------------------------------------------------


def test_a_difference_inside_the_model_seed_noise_is_flagged_not_hidden():
    """It is still reported - with `below_seed_noise_floor` true."""
    baseline = frame("isolated", relative=0.50)
    treatment = frame("fedavg", relative=0.52)      # +0.02, inside the floor
    merged, _ = compare_mod.paired(
        baseline, treatment, compare_mod.PER_INSTITUTION_KEYS,
    )
    rows = compare_mod.summarise_difference(
        merged, "recall_relative_to_ceiling",
        ("visibility_target", "seed", "operating_point", "eval_split",
         "test_subset"),
        "per_institution",
    )
    assert len(rows) == 1
    assert rows[0]["mean_difference"] == pytest.approx(0.02)
    assert rows[0]["seed_noise_floor"] == pytest.approx(0.04)
    assert rows[0]["below_seed_noise_floor"] is True

    big = frame("fedavg", relative=0.62)            # +0.12, above the floor
    merged_big, _ = compare_mod.paired(
        baseline, big, compare_mod.PER_INSTITUTION_KEYS,
    )
    rows_big = compare_mod.summarise_difference(
        merged_big, "recall_relative_to_ceiling",
        ("visibility_target", "seed", "operating_point", "eval_split",
         "test_subset"),
        "per_institution",
    )
    assert rows_big[0]["below_seed_noise_floor"] is False


def test_a_difference_carries_its_spread_and_sign_counts_not_a_standard_error():
    """The institutions are not exchangeable, so no interval is taken over them."""
    baseline = pd.DataFrame([
        row("isolated", 1.0, 0, 0, "total_10", 0.4, 0.5),
        row("isolated", 1.0, 0, 1, "total_10", 0.4, 0.5),
        row("isolated", 1.0, 0, 2, "total_10", 0.4, 0.5),
    ])
    treatment = pd.DataFrame([
        row("fedavg", 1.0, 0, 0, "total_10", 0.9, 0.9),   # +0.4
        row("fedavg", 1.0, 0, 1, "total_10", 0.4, 0.5),   #  0.0
        row("fedavg", 1.0, 0, 2, "total_10", 0.1, 0.2),   # -0.3
    ])
    merged, _ = compare_mod.paired(
        baseline, treatment, compare_mod.PER_INSTITUTION_KEYS,
    )
    out = compare_mod.summarise_difference(
        merged, "recall_relative_to_ceiling",
        ("visibility_target", "seed", "operating_point", "eval_split",
         "test_subset"),
        "per_institution",
    )[0]
    assert out["n_pairs"] == 3
    assert out["mean_difference"] == pytest.approx((0.4 + 0.0 - 0.3) / 3)
    assert out["median_difference"] == pytest.approx(0.0)
    assert out["min_difference"] == pytest.approx(-0.3)
    assert out["max_difference"] == pytest.approx(0.4)
    assert (out["n_positive"], out["n_zero"], out["n_negative"]) == (1, 1, 1)
    # No interval across institutions, under any name. Asserted as an exact
    # key set rather than a substring scan, so adding one later fails here.
    assert set(out) == {
        "visibility_target", "seed", "operating_point", "eval_split",
        "test_subset", "unit", "metric", "n_pairs", "mean_difference",
        "median_difference", "min_difference", "max_difference",
        "n_positive", "n_zero", "n_negative", "baseline_mean",
        "treatment_mean", "seed_noise_floor", "below_seed_noise_floor",
        "n_f1_degenerate_baseline", "n_f1_degenerate_treatment",
    }, (
        "the per-institution summary gained or lost a field. The eight "
        "institutions are not exchangeable, so nothing resembling a standard "
        "error, confidence interval or t-statistic across them may be added "
        "here - update this list deliberately if the change is intended."
    )


# ---------------------------------------------------------------------------
# dropping 4.3 leaves a complete result with the gaps NAMED
# ---------------------------------------------------------------------------


def test_dropping_the_evidence_arms_names_the_contrasts_it_cannot_form():
    summaries = {"isolated": frame("isolated"), "fedavg": frame("fedavg")}
    unions = {}
    out = compare_mod.compare(summaries, unions, None)
    assert out["conditions_present"] == ["isolated", "fedavg"]
    produced = {r["contrast"] for r in out["rows"]}
    assert "fedavg - isolated" in produced
    skipped = {s.get("name") for s in out["skipped"]}
    for missing in ("evidence - isolated", "fedavg_evidence - fedavg",
                    "fedavg_evidence - isolated", "fedavg_evidence - evidence"):
        assert missing in skipped, f"{missing} must be NAMED, not absent"
    assert all(
        s["reason"] for s in out["skipped"]
    ), "every skipped contrast must carry a reason"
    assert out["centralized_present"] is False


def test_the_primary_points_sort_first():
    summaries = {
        "isolated": pd.concat([
            frame("isolated", point="f1"),
            frame("isolated", point="per_inst_50"),
        ], ignore_index=True),
        "fedavg": pd.concat([
            frame("fedavg", point="f1", relative=0.6),
            frame("fedavg", point="per_inst_50", relative=0.6),
        ], ignore_index=True),
    }
    out = compare_mod.compare(summaries, {}, None)
    rows = out["rows"]
    assert rows, "expected some rows"
    assert rows[0]["primary"] is True
    primaries = [r["primary"] for r in rows]
    assert primaries == sorted(primaries, reverse=True), (
        "primary points must come first in the report"
    )


# ---------------------------------------------------------------------------
# the degenerate-F1 count travels with the difference
# ---------------------------------------------------------------------------


def test_the_degenerate_f1_count_travels_with_a_paired_difference():
    """A difference whose cells had flat F1 curves is not the same comparison.

    So the count is carried per side, counted over the pairs the mean is
    taken over - the same contract as the binding-budget count on a pooled
    budget number.
    """
    base = [row("isolated", 1.0, 0, i, "f1", 0.3, 0.3) for i in range(3)]
    treat = [row("fedavg", 1.0, 0, i, "f1", 0.4, 0.4) for i in range(3)]
    for i, flag in enumerate([True, True, False]):
        base[i]["f1_degenerate"] = flag
    for i, flag in enumerate([True, False, False]):
        treat[i]["f1_degenerate"] = flag

    merged, _ = compare_mod.paired(
        pd.DataFrame(base), pd.DataFrame(treat),
        compare_mod.PER_INSTITUTION_KEYS, label="t",
    )
    out = compare_mod.summarise_difference(
        merged, "recall_relative_to_ceiling",
        ("visibility_target", "seed", "operating_point", "eval_split",
         "test_subset"),
        "per_institution",
    )[0]
    assert out["n_pairs"] == 3
    assert out["n_f1_degenerate_baseline"] == 2
    assert out["n_f1_degenerate_treatment"] == 1


def test_a_missing_degenerate_column_is_none_and_never_zero():
    """Absent is not the same claim as "none were degenerate"."""
    merged, _ = compare_mod.paired(
        pd.DataFrame([row("isolated", 1.0, 0, 0, "f1", 0.3, 0.3)]),
        pd.DataFrame([row("fedavg", 1.0, 0, 0, "f1", 0.4, 0.4)]),
        compare_mod.PER_INSTITUTION_KEYS, label="t",
    )
    out = compare_mod.summarise_difference(
        merged, "recall_relative_to_ceiling",
        ("visibility_target", "seed", "operating_point", "eval_split",
         "test_subset"),
        "per_institution",
    )[0]
    assert out["n_f1_degenerate_baseline"] is None
    assert out["n_f1_degenerate_treatment"] is None
