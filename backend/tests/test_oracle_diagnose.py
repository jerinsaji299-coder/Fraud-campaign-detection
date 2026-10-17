"""Why the oracle misses campaigns it has perfect information about.

Diagnosis only — nothing here changes the matching rule. Each case is built
by hand so the expected reason follows from the construction rather than
from whatever the code happens to say.
"""

import pandas as pd

from fraudcamp import evaluation
from fraudcamp.evaluation import scorers
from test_evaluation import _world  # the hand-built world builder


def _oracle(window, t):
    return scorers.oracle_scores(window["Is Laundering"].to_numpy())


def _unassigned_accounts(full_df):
    rows = full_df[(full_df["Is Laundering"] == 1) & (full_df["campaign_id"] == -1)]
    return set(rows["src"]) | set(rows["dst"])


SPARSE = [
    ("2022-09-06 02:00", "1_A", "1_B", 0),
    ("2022-09-07 18:00", "1_B", "1_C", 0),
    ("2022-09-09 10:00", "1_C", "1_D", 0),
]
TIGHT = [
    ("2022-09-06 07:00", "1_A", "1_B", 0),
    ("2022-09-06 08:00", "1_B", "1_C", 0),
    ("2022-09-06 09:00", "1_C", "1_D", 0),
]


def test_slow_when_transactions_are_further_apart_than_the_lookback():
    """40-hour gaps mean no 24h window ever holds 3 connected accounts.

    The evidence does exist: pooled over the whole campaign its own edges
    connect 3 accounts. That is what makes this `slow` rather than
    `structurally_disconnected` — a longer lookback could reach it.
    """
    full_df, camp, _, accounts = _world(SPARSE)
    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]

    assert row["reason"] == "slow"
    assert row["max_connected_own_accounts"] < 3
    assert row["max_connected_all_edges"] >= 3
    assert row["gap_exceeds_lookback"] is True
    assert row["max_txn_gap_h"] > 24
    assert "largest gap" in row["detail"]


def test_structurally_disconnected_when_the_evidence_never_connects():
    """Three disjoint pairs: six accounts, no two edges sharing an account.

    Pool every edge, ignore windows entirely, and the largest connected set
    is still 2. No lookback can ever make this matchable, which is a
    different statement from "too slow" and gets its own reason.
    """
    disjoint = [
        ("2022-09-06 02:00", "1_A", "1_B", 0),
        ("2022-09-06 03:00", "1_C", "1_D", 0),
        ("2022-09-06 04:00", "1_E", "1_F", 0),
    ]
    full_df, camp, _, accounts = _world(disjoint)
    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]

    assert row["reason"] == "structurally_disconnected"
    assert row["max_connected_all_edges"] == 2
    assert row["gap_exceeds_lookback"] is False  # nothing to do with timing
    assert "no lookback can" in row["detail"]


def test_the_two_sparsity_reasons_are_told_apart_by_pooled_connectivity():
    """The one thing that separates them, stated as a test: same miss, and
    the pooled connectivity alone decides which reason is reported."""
    disjoint = [
        ("2022-09-06 02:00", "1_A", "1_B", 0),
        ("2022-09-06 03:00", "1_C", "1_D", 0),
        ("2022-09-06 04:00", "1_E", "1_F", 0),
    ]
    rows = {}
    for name, campaign_rows in (("slow", SPARSE), ("disconnected", disjoint)):
        full_df, camp, _, accounts = _world(campaign_rows)
        rows[name] = evaluation.diagnose_misses(
            full_df, camp, accounts, [0], "test", 24
        )[0]

    # neither ever gets 3 connected accounts inside a window
    assert rows["slow"]["max_connected_own_accounts"] < 3
    assert rows["disconnected"]["max_connected_own_accounts"] < 3
    # pooled, one does and one does not
    assert rows["slow"]["max_connected_all_edges"] >= 3
    assert rows["disconnected"]["max_connected_all_edges"] < 3
    assert rows["slow"]["reason"] == "slow"
    assert rows["disconnected"]["reason"] == "structurally_disconnected"


def test_a_longer_lookback_can_rescue_a_sparse_campaign():
    """The same campaign at L=48 reaches more connected accounts, which is
    the mechanism behind the oracle doing better at L=48 than L=24."""
    full_df, camp, _, accounts = _world(SPARSE)
    at_24 = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]
    at_48 = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 48)[0]
    assert at_48["max_connected_own_accounts"] >= at_24["max_connected_own_accounts"]


def test_purity_failure_when_the_campaign_is_swallowed_by_a_bigger_cluster():
    """The 4 accounts do connect, but a chain of unassigned laundering drags
    them into a cluster where they are a minority, so the 50% rule declines
    it.

    The campaign is extended to Sept 7 so its deadline leaves a pre-deadline
    window that can actually see the fragment — otherwise the honest
    diagnosis would be `late_only`, not a purity failure.
    """
    full_df, camp, _, accounts = _world(TIGHT + [("2022-09-07 20:00", "1_D", "1_A", 0)])
    noise = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2022-09-06 07:30") + pd.Timedelta(minutes=i),
                "src": "1_D" if i == 0 else f"9_N{i}",
                "dst": f"9_N{i + 1}",
                "From Bank": 9, "To Bank": 9, "Amount Paid": 5.0,
                "Amount Received": 5.0, "Payment Currency": "US Dollar",
                "Receiving Currency": "US Dollar", "Payment Format": "ACH",
                "Is Laundering": 1, "campaign_id": -1,
            }
            for i in range(14)
        ]
    )
    full_df = pd.concat([full_df, noise], ignore_index=True)

    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]
    assert row["reason"] == "purity_failure"
    assert row["max_connected_own_accounts"] >= 3
    assert row["best_cluster_purity"] < 0.5
    assert row["merged_with_unassigned"] > 0
    assert "merged with" in row["detail"]


def test_late_only_when_the_evidence_arrives_after_the_deadline():
    """All three edges fall in one window, but the deadline passes before the
    first detection time that can see them."""
    full_df, camp, _, accounts = _world(TIGHT)
    assert camp.loc[0, "deadline"] == pd.Timestamp("2022-09-06 09:00")

    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]
    assert row["reason"] == "late_only"
    assert "after the deadline" in row["detail"]


def test_every_reported_field_is_present_and_the_reason_is_known():
    full_df, camp, _, accounts = _world(SPARSE)
    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]
    for field in (
        "campaign_id", "split", "lookback_h", "base_type", "group", "reason",
        "detail", "n_txn", "n_accounts", "duration_h", "deadline",
        "max_connected_own_accounts", "max_connected_all_edges",
        "max_txn_gap_h", "gap_exceeds_lookback",
        "best_cluster_purity", "best_cluster_size", "merged_with_campaigns",
        "merged_with_unassigned", "windows_checked",
    ):
        assert field in row, field
    assert row["reason"] in evaluation.REASONS


def test_a_real_miss_is_never_unexplained():
    """`unexplained` means the diagnosis and the evaluation disagree, which
    would mean one of them is wrong. Every genuine miss must get a reason."""
    for campaign_rows in (SPARSE, TIGHT):
        full_df, camp, pattern_df, accounts = _world(campaign_rows)
        for lookback in (24, 48):
            result = evaluation.evaluate_split(
                full_df, camp, pattern_df, accounts, "test", _oracle,
                tau=0.5, lookback_h=lookback, scorer_name="oracle",
            )
            missed = [o.campaign_id for o in result.outcomes if not o.detected]
            for row in evaluation.diagnose_misses(
                full_df, camp, accounts, missed, "test", lookback
            ):
                assert row["reason"] != "unexplained", row["detail"]


def test_a_detected_campaign_is_not_diagnosed_as_sparse():
    """Sanity in the other direction: a campaign the oracle finds must not
    look like one it cannot."""
    full_df, camp, pattern_df, accounts = _world(
        TIGHT + [("2022-09-07 20:00", "1_D", "1_A", 0)]
    )
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    assert result.outcomes[0].detected
    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]
    assert row["max_connected_own_accounts"] >= 3
    assert row["reason"] not in ("slow", "structurally_disconnected")


def test_false_alarm_diagnosis_characterises_the_clusters():
    """The oracle keeps only laundering edges, so every false alarm is real
    laundering the rule could not attribute to a scored campaign."""
    full_df, camp, _, accounts = _world(
        TIGHT + [("2022-09-07 20:00", "1_D", "1_A", 0)]
    )
    noise = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2022-09-06 10:00") + pd.Timedelta(minutes=i),
                "src": "1_A" if i == 0 else f"7_M{i}",
                "dst": f"7_M{i + 1}",
                "From Bank": 7, "To Bank": 7, "Amount Paid": 5.0,
                "Amount Received": 5.0, "Payment Currency": "US Dollar",
                "Receiving Currency": "US Dollar", "Payment Format": "ACH",
                "Is Laundering": 1, "campaign_id": -1,
            }
            for i in range(4)
        ]
    )
    full_df = pd.concat([full_df, noise], ignore_index=True)

    rows = evaluation.diagnose_false_alarms(
        full_df, accounts, _unassigned_accounts(full_df), "test", 24
    )
    for row in rows:
        assert row["cluster_size"] >= 3
        assert row["fails"] in evaluation.FALSE_ALARM_CONDITIONS
        assert {"t", "split", "lookback_h", "campaigns_touched"} <= set(row)
        # whichever condition is named, the rule really does decline it
        assert not (
            row["best_purity_fraction"] >= 0.5
            and row["best_coverage_accounts"] >= 3
        )


def test_a_false_alarm_can_fail_coverage_while_passing_purity():
    """The case the old summary line contradicted itself over.

    A cluster two thirds of which belongs to one campaign has passed the
    0.5 purity condition comfortably. What it failed is coverage: it holds
    only 2 of that campaign's accounts and the rule needs 3. Reporting its
    0.667 as "below the 0.5 the rule requires" was simply wrong, so each
    condition is now checked against its own threshold.
    """
    from fraudcamp.evaluation.diagnose import _failed_condition

    campaign = {0: ("1_A", "1_B", "1_C", "1_D")}
    row = _failed_condition({"1_A", "1_B", "7_X"}, {0}, campaign)

    assert row["best_purity_fraction"] > 0.5
    assert row["best_coverage_accounts"] == 2
    assert row["fails"] == "coverage"


def test_each_failing_condition_is_named_for_the_binding_bound():
    from fraudcamp.evaluation.diagnose import _failed_condition

    campaign = {0: ("1_A", "1_B", "1_C", "1_D")}

    # enough of the campaign's accounts, but swamped by other accounts
    diluted = _failed_condition(
        {"1_A", "1_B", "1_C", "7_X", "7_Y", "7_Z", "7_W"}, {0}, campaign
    )
    assert diluted["best_coverage_accounts"] == 3
    assert diluted["best_purity_fraction"] < 0.5
    assert diluted["fails"] == "purity"

    # neither condition met
    both = _failed_condition({"1_A", "7_X", "7_Y", "7_Z"}, {0}, campaign)
    assert both["fails"] == "both"

    # touches no campaign account at all: pure unassigned laundering
    none = _failed_condition({"7_X", "7_Y", "7_Z"}, set(), campaign)
    assert none["fails"] == "no_campaign_accounts"
    assert none["best_purity_fraction"] == 0.0
    assert none["best_coverage_accounts"] == 0


def test_the_named_condition_agrees_with_the_frozen_matching_rule():
    """The classification must use the rule's own thresholds, not a copy
    that could drift from them."""
    from fraudcamp.evaluation.clusters import cluster_hits_campaign
    from fraudcamp.evaluation.diagnose import _failed_condition

    campaign_accounts = ("1_A", "1_B", "1_C", "1_D")
    campaign = {0: campaign_accounts}
    clusters = [
        {"1_A", "1_B", "7_X"},
        {"1_A", "1_B", "1_C", "7_X", "7_Y", "7_Z", "7_W"},
        {"1_A", "7_X", "7_Y", "7_Z"},
        {"1_A", "1_B", "1_C"},
        {"1_A", "1_B", "1_C", "1_D", "7_X"},
    ]
    for cluster in clusters:
        row = _failed_condition(cluster, {0}, campaign)
        hits = cluster_hits_campaign(cluster, campaign_accounts)
        passes_both = (
            row["best_purity_fraction"] >= 0.5 and row["best_coverage_accounts"] >= 3
        )
        assert hits == passes_both, (cluster, row)


def test_diagnosis_does_not_change_the_matching_rule():
    """Guard against the obvious temptation: running a diagnosis must leave
    the evaluation's verdict untouched."""
    full_df, camp, pattern_df, accounts = _world(SPARSE)
    before = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)
    after = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    assert [o.detected for o in before.outcomes] == [o.detected for o in after.outcomes]
