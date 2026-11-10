"""Cross-model comparison: the saved records, the edge diagnostic, the shapes.

Three things here would quietly invalidate a stage 3.4.5 claim if they broke,
so each is pinned by a test that can fail:

1. A campaign-level record that survives a CSV round trip **unchanged**. The
   nasty case is `detected`: a CSV writes `False`, and `bool("False")` is
   `True`, so a careless parse marks every campaign detected and every recall
   1.0 - a number that looks plausible and is worthless.
2. A direct comparison that is identical whether it runs in process or from
   two files. If the file path changed the answer, rule 3 would hold only
   inside a single run.
3. Match keys that actually identify one detector at one operating point. If
   two variants collapsed into one key, the paired set would be built from a
   mixture of two models and nothing would say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudcamp import evaluation
from fraudcamp.evaluation import (
    CampaignOutcome,
    EvaluationResult,
    average_precision,
    cluster_size_distribution,
    compare_outcome_tables,
    coverage_rows,
    edge_metrics,
    outcome_records,
    outcomes_from_records,
    paired_lead_time_comparison,
    precision_at_n_positive,
)
from fraudcamp.evaluation.edge_metrics import edge_metrics_over_times

DEADLINE = pd.Timestamp("2022-09-07 00:00")


def outcome(campaign_id: int, lead_time_h: float | None, detected: bool = True):
    return CampaignOutcome(
        campaign_id=campaign_id,
        split="test",
        base_type="SCATTER-GATHER",
        group="fragmentable",
        detected=detected,
        detection_time=DEADLINE if detected else None,
        lead_time_h=lead_time_h,
        normalized_lead_time=(lead_time_h / 120.0) if lead_time_h else None,
        frac_observed=0.25 if detected else None,
        crosses_cutoff=False,
        shares_train_account=False,
        deadline=DEADLINE,
        end=DEADLINE,
        duration_h=120.0,
        n_txn=12,
    )


def result_of(outcomes, scorer="model", tau=0.9, split="test", seed=0):
    return EvaluationResult(
        split=split, scorer=scorer, tau=tau, lookback_h=24, seed=seed,
        outcomes=list(outcomes),
    )


# ---------------------------------------------------------------------------
# edge-level metrics
# ---------------------------------------------------------------------------


def test_average_precision_matches_a_hand_computed_value():
    # ranked 1, 0, 1, 0: precision 1/1 at the first positive and 2/3 at the
    # second, averaged over two positives
    ap = average_precision(np.array([0.9, 0.8, 0.1, 0.05]), np.array([1, 0, 1, 0]))
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)


def test_a_perfect_ranking_scores_one_and_a_reversed_one_scores_less():
    labels = np.array([1, 1, 0, 0, 0, 0])
    perfect = average_precision(np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.0]), labels)
    reversed_scores = average_precision(
        np.array([0.0, 0.1, 0.2, 0.3, 0.8, 0.9]), labels
    )
    assert perfect == pytest.approx(1.0)
    assert reversed_scores < 0.5


def test_no_positives_is_undefined_not_zero():
    """NaN, because 0.0 would read as a model that failed rather than a
    window with nothing to find."""
    assert np.isnan(average_precision(np.array([0.9, 0.1]), np.array([0, 0])))
    assert np.isnan(edge_metrics(np.array([0.9, 0.1]), np.array([0, 0]))["ap_lift"])


def test_ap_lift_is_against_the_positive_rate_which_is_a_random_ranker():
    rng = np.random.default_rng(0)
    labels = np.zeros(20_000)
    labels[:20] = 1.0
    scores = rng.random(20_000)
    metrics = edge_metrics(scores, labels)
    assert metrics["positive_rate"] == pytest.approx(0.001)
    # a random ranker's AP is the positive rate, so its lift is about 1
    assert 0.2 < metrics["ap_lift"] < 5.0


def test_precision_at_n_positive_is_the_top_p_hit_rate():
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert precision_at_n_positive(
        np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1]), labels
    ) == pytest.approx(1.0)
    assert precision_at_n_positive(
        np.array([0.9, 0.8, 0.1, 0.7, 0.1, 0.1]), labels
    ) == pytest.approx(2 / 3)


def test_the_gnn_and_the_diagnostic_share_one_average_precision():
    """The number the GNN stops on and the number XGBoost is diagnosed with
    must be produced by the same function, or a comparison of the two is a
    comparison of two implementations."""
    from fraudcamp.models import gnn

    assert gnn.average_precision is average_precision


def test_pooled_edge_metrics_are_recomputed_not_averaged():
    """A window with four positives must not weigh as much as one with four
    hundred, which averaging the per-window rows would do."""
    times = [pd.Timestamp("2022-09-05 06:00"), pd.Timestamp("2022-09-05 12:00")]
    frame = pd.DataFrame(
        {
            "ts": [
                pd.Timestamp("2022-09-05 05:00"), pd.Timestamp("2022-09-05 05:30"),
                pd.Timestamp("2022-09-05 11:00"), pd.Timestamp("2022-09-05 11:30"),
                pd.Timestamp("2022-09-05 11:40"), pd.Timestamp("2022-09-05 11:50"),
            ],
            "src": list("abcdef"),
            "dst": list("bcdefa"),
            "Is Laundering": [1, 0, 1, 1, 1, 0],
        }
    )

    def scorer(window, t):
        # perfect in the first window, inverted in the second
        if t == times[0]:
            return window["Is Laundering"].to_numpy().astype(float)
        return 1.0 - window["Is Laundering"].to_numpy().astype(float)

    rows = edge_metrics_over_times(frame, scorer, times, lookback_h=2)
    per_window = [r for r in rows if r["window"] == "one"]
    pooled = [r for r in rows if r["window"] == "pooled"]
    assert len(per_window) == 2 and len(pooled) == 1
    mean_of_windows = np.mean([r["average_precision"] for r in per_window])
    assert pooled[0]["n_positive"] == 4
    assert pooled[0]["average_precision"] != pytest.approx(mean_of_windows)


def test_a_scorer_returning_the_wrong_number_of_scores_raises():
    frame = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2022-09-05 05:00")],
            "src": ["a"], "dst": ["b"], "Is Laundering": [1],
        }
    )
    with pytest.raises(ValueError, match="scores for"):
        edge_metrics_over_times(
            frame, lambda w, t: np.array([0.5, 0.5]),
            [pd.Timestamp("2022-09-05 06:00")], lookback_h=2,
        )


# ---------------------------------------------------------------------------
# cluster shape
# ---------------------------------------------------------------------------


def test_cluster_size_distribution_counts_only_its_own_kind():
    result = result_of([])
    result.cluster_sizes = [
        {"t": DEADLINE, "kind": "false_alarm", "n_accounts": 3},
        {"t": DEADLINE, "kind": "false_alarm", "n_accounts": 3},
        {"t": DEADLINE, "kind": "false_alarm", "n_accounts": 40},
        {"t": DEADLINE, "kind": "hit", "n_accounts": 500},
    ]
    alarms = cluster_size_distribution(result, "false_alarm")
    assert alarms["n_clusters"] == 3
    assert alarms["max_accounts"] == 40
    assert alarms["median_accounts"] == 3
    assert alarms["n_3"] == 2 and alarms["n_11_50"] == 1
    assert cluster_size_distribution(result, "hit")["n_clusters"] == 1


def test_two_models_can_match_on_count_and_differ_entirely_on_shape():
    """Why the shape is reported at all: the same false-alarm count can be
    three-account components or one enormous one."""
    small = result_of([])
    small.cluster_sizes = [
        {"t": DEADLINE, "kind": "false_alarm", "n_accounts": 3} for _ in range(10)
    ]
    large = result_of([])
    large.cluster_sizes = [
        {"t": DEADLINE, "kind": "false_alarm", "n_accounts": 300} for _ in range(10)
    ]
    a, b = (
        cluster_size_distribution(small, "false_alarm"),
        cluster_size_distribution(large, "false_alarm"),
    )
    assert a["n_clusters"] == b["n_clusters"] == 10
    assert a["n_accounts_total"] == 30 and b["n_accounts_total"] == 3000
    assert a["n_3"] == 10 and b["n_201_plus"] == 10


def test_an_empty_distribution_reports_no_clusters_rather_than_failing():
    empty = cluster_size_distribution(result_of([]), "false_alarm")
    assert empty["n_clusters"] == 0
    assert empty["median_accounts"] is None
    assert empty["n_3"] == 0


# ---------------------------------------------------------------------------
# the records, and the round trip
# ---------------------------------------------------------------------------


def test_outcome_records_carry_every_field_the_subsets_need():
    rows = outcome_records(
        result_of([outcome(1, 50.0)]), model="gnn", operating_point="f1"
    )
    assert len(rows) == 1
    for field in evaluation.OUTCOME_FIELDS:
        assert field in rows[0]
    assert rows[0]["model"] == "gnn" and rows[0]["operating_point"] == "f1"
    assert rows[0]["eval_split"] == "test" and rows[0]["tau"] == 0.9


def test_a_csv_round_trip_preserves_which_campaigns_were_detected(tmp_path):
    """The whole round trip, the way `compare-models` reads it."""
    outcomes = [outcome(1, 50.0), outcome(2, None, detected=False)]
    path = tmp_path / "outcomes.csv"
    pd.DataFrame(outcome_records(result_of(outcomes))).to_csv(path, index=False)

    rebuilt = outcomes_from_records(pd.read_csv(path).to_dict("records"))
    assert [o.detected for o in rebuilt] == [True, False]
    assert [o.lead_time_h for o in rebuilt] == [50.0, None]
    assert rebuilt[0].detection_time == DEADLINE and rebuilt[1].detection_time is None
    assert all(o.crosses_cutoff is False for o in rebuilt)


def test_the_word_false_is_parsed_as_false_not_as_a_truthy_string():
    """`bool("False")` is True. pandas usually infers a real boolean column,
    so the round trip above would pass even with a naive coercion - which is
    exactly why this is pinned separately. A column read as text (dtype=str,
    an object column holding a blank, a hand-edited file) must not report
    every campaign detected and a recall of 1.0.
    """
    assert evaluation.evaluate._as_bool("False") is False
    assert evaluation.evaluate._as_bool("True") is True
    records = [
        {**row, "detected": str(row["detected"])}
        for row in outcome_records(
            result_of([outcome(1, 50.0), outcome(2, None, detected=False)])
        )
    ]
    rebuilt = outcomes_from_records(records)
    assert [o.detected for o in rebuilt] == [True, False]
    with pytest.raises(ValueError, match="not a boolean"):
        evaluation.evaluate._as_bool("probably")


def test_reading_the_file_as_text_gives_the_same_detections(tmp_path):
    """Belt and braces on the above, through a real file."""
    outcomes = [outcome(1, 50.0), outcome(2, None, detected=False)]
    path = tmp_path / "outcomes.csv"
    pd.DataFrame(outcome_records(result_of(outcomes))).to_csv(path, index=False)
    as_text = pd.read_csv(path, dtype=str).to_dict("records")
    rebuilt = outcomes_from_records(as_text)
    assert sum(o.detected for o in rebuilt) == 1
    assert rebuilt[0].lead_time_h == pytest.approx(50.0)


def test_the_round_trip_preserves_the_paired_comparison_exactly(tmp_path):
    """Rule 3 has to hold across sessions, not only inside one process."""
    a = [outcome(1, 60.0), outcome(2, 30.0), outcome(3, None, detected=False)]
    b = [outcome(1, 72.0), outcome(2, 30.0), outcome(3, 12.0)]
    in_process = paired_lead_time_comparison(a, b, "a", "b", split="test")

    paths = []
    for index, outcomes in enumerate((a, b)):
        path = tmp_path / f"{index}.csv"
        pd.DataFrame(
            outcome_records(
                result_of(outcomes), ablation="full", operating_point="f1"
            )
        ).to_csv(path, index=False)
        paths.append(path)

    rows, notes, unmatched = compare_outcome_tables(
        pd.read_csv(paths[0]).to_dict("records"),
        pd.read_csv(paths[1]).to_dict("records"),
        "a", "b",
    )
    assert not unmatched and len(rows) == 1
    assert rows[0]["median_difference_h"] == pytest.approx(
        in_process.median_difference_h
    )
    assert rows[0]["ci_low_h"] == pytest.approx(in_process.difference_ci_h[0])
    assert rows[0]["n_both"] == in_process.n_both
    assert rows[0]["n_only_b"] == in_process.n_only_b == 1
    assert "detects 1 that a misses" in notes[0]


def test_the_sign_convention_is_b_minus_a():
    """Positive means the second detector was earlier. A flipped sign would
    reverse every conclusion while every interval still looked reasonable."""
    a = outcome_records(result_of([outcome(1, 20.0)]), operating_point="f1")
    b = outcome_records(result_of([outcome(1, 50.0)]), operating_point="f1")
    rows, _, _ = compare_outcome_tables(a, b, "slow", "fast")
    assert rows[0]["median_difference_h"] == pytest.approx(30.0)
    assert rows[0]["n_b_earlier"] == 1 and rows[0]["n_a_earlier"] == 0


# ---------------------------------------------------------------------------
# matching: what is compared with what
# ---------------------------------------------------------------------------


def test_only_matched_keys_are_compared_and_the_rest_are_reported():
    a = outcome_records(result_of([outcome(1, 20.0)]), operating_point="f1")
    a += outcome_records(
        result_of([outcome(1, 20.0)]), operating_point="budget_10"
    )
    b = outcome_records(result_of([outcome(1, 50.0)]), operating_point="f1")
    rows, _, unmatched = compare_outcome_tables(a, b, "a", "b")
    assert len(rows) == 1 and rows[0]["operating_point"] == "f1"
    assert len(unmatched) == 1 and "budget_10" in unmatched[0]


def test_seeds_are_matched_so_model_variance_does_not_enter_the_difference():
    a = outcome_records(
        result_of([outcome(1, 20.0)], seed=0), operating_point="f1"
    )
    b = outcome_records(
        result_of([outcome(1, 50.0)], seed=1), operating_point="f1"
    )
    rows, _, unmatched = compare_outcome_tables(a, b, "a", "b")
    assert rows == [] and len(unmatched) == 2


def test_two_variants_under_one_key_are_refused_rather_than_mixed():
    """If the ablation arms collapsed into one group the paired set would be
    built from a mixture of two models, and the report would not say so."""
    a = outcome_records(
        result_of([outcome(1, 20.0)], scorer="full"),
        operating_point="f1", ablation="full",
    )
    # the same campaign again, as a second variant, but tagged identically
    a += outcome_records(
        result_of([outcome(1, 80.0)], scorer="no_format"),
        operating_point="f1", ablation="full",
    )
    b = outcome_records(
        result_of([outcome(1, 50.0)]), operating_point="f1", ablation="full"
    )
    with pytest.raises(ValueError, match="repeats campaign ids"):
        compare_outcome_tables(a, b, "a", "b")


def test_the_ablation_arms_are_compared_arm_by_arm():
    a, b = [], []
    for arm, lead_a, lead_b in (("full", 20.0, 50.0), ("no_format", 40.0, 10.0)):
        a += outcome_records(
            result_of([outcome(1, lead_a)]), operating_point="f1", ablation=arm
        )
        b += outcome_records(
            result_of([outcome(1, lead_b)]), operating_point="f1", ablation=arm
        )
    rows, _, unmatched = compare_outcome_tables(a, b, "a", "b")
    assert not unmatched
    by_arm = {row["ablation"]: row["median_difference_h"] for row in rows}
    assert by_arm["full"] == pytest.approx(30.0)
    assert by_arm["no_format"] == pytest.approx(-30.0)


def test_coverage_is_reported_by_campaign_id_each_way():
    a = outcome_records(
        result_of([outcome(1, 20.0), outcome(2, None, detected=False), outcome(3, 5.0)]),
        operating_point="f1",
    )
    b = outcome_records(
        result_of([outcome(1, 20.0), outcome(2, 9.0), outcome(3, None, detected=False)]),
        operating_point="f1",
    )
    rows = coverage_rows(a, b, "a", "b")
    assert len(rows) == 1
    assert rows[0]["n_detected_a"] == 2 and rows[0]["n_detected_b"] == 2
    assert rows[0]["n_both"] == 1
    assert rows[0]["only_a"] == "3" and rows[0]["only_b"] == "2"
    assert rows[0]["recall_a"] == pytest.approx(2 / 3)
