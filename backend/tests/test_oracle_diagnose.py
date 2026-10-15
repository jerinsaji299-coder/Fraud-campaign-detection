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


def test_temporal_sparsity_when_transactions_are_further_apart_than_the_lookback():
    """40-hour gaps mean no 24h window ever holds 3 connected accounts."""
    full_df, camp, _, accounts = _world(SPARSE)
    row = evaluation.diagnose_misses(full_df, camp, accounts, [0], "test", 24)[0]

    assert row["reason"] == "temporal_sparsity"
    assert row["max_connected_own_accounts"] < 3
    assert row["gap_exceeds_lookback"] is True
    assert row["max_txn_gap_h"] > 24
    assert "largest gap" in row["detail"]


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
        "max_connected_own_accounts", "max_txn_gap_h", "gap_exceeds_lookback",
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
    assert row["reason"] != "temporal_sparsity"


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
        assert row["best_overlap_fraction"] < 0.5
        assert {"t", "split", "lookback_h", "campaigns_touched"} <= set(row)


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
