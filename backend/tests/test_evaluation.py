"""Cluster extraction, matching, outcomes and lead time.

The end-to-end cases are built by hand with timestamps chosen so the
expected detection time and lead time can be worked out on paper, rather
than asserted against whatever the code happens to produce.
"""

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, evaluation
from fraudcamp.evaluation import clusters as ct
from fraudcamp.evaluation import scorers

# --- extraction ------------------------------------------------------------


def test_extraction_keeps_only_edges_at_or_above_tau():
    src = ["A", "C", "E"]
    dst = ["B", "D", "F"]
    scores = np.array([0.9, 0.5, 0.1])
    found = ct.extract_clusters(src, dst, scores, tau=0.5, min_accounts=2)
    members = {frozenset(c) for c in found}
    assert members == {frozenset({"A", "B"}), frozenset({"C", "D"})}


def test_extraction_joins_transitively():
    found = ct.extract_clusters(["A", "B", "C"], ["B", "C", "D"], np.ones(3), tau=0.5)
    assert found == [frozenset({"A", "B", "C", "D"})]


def test_extraction_drops_components_below_the_minimum():
    """Two accounts is not a cluster: the frozen rule needs at least 3."""
    found = ct.extract_clusters(["A"], ["B"], np.ones(1), tau=0.5)
    assert found == []


def test_extraction_ignores_edge_direction():
    """The graph for clustering is undirected."""
    a = ct.extract_clusters(["A", "B"], ["B", "C"], np.ones(2), tau=0.5)
    b = ct.extract_clusters(["B", "C"], ["A", "B"], np.ones(2), tau=0.5)
    assert a == b == [frozenset({"A", "B", "C"})]


def test_extraction_with_nothing_above_tau():
    assert ct.extract_clusters(["A"], ["B"], np.array([0.1]), tau=0.5) == []


def test_extraction_is_deterministic():
    src, dst = ["A", "B", "C", "X"], ["B", "C", "A", "Y"]
    scores = np.ones(4)
    runs = [ct.extract_clusters(src, dst, scores, 0.5) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# --- the matching rule -----------------------------------------------------


def test_matching_needs_both_half_the_cluster_and_three_accounts():
    campaign = ["A", "B", "C", "D"]
    # 3 of 3 cluster accounts in the campaign, covering 3 -> hit
    assert ct.cluster_hits_campaign({"A", "B", "C"}, campaign)
    # covers only 2 campaign accounts -> no
    assert not ct.cluster_hits_campaign({"A", "B"}, campaign)
    # 3 covered but only 3 of 8 cluster accounts (37.5%) -> no
    assert not ct.cluster_hits_campaign(
        {"A", "B", "C", "W", "X", "Y", "Z", "V"}, campaign
    )


def test_matching_at_exactly_the_threshold():
    """50% and 3 covered is a hit: the rule is inclusive."""
    campaign = ["A", "B", "C"]
    assert ct.cluster_hits_campaign({"A", "B", "C", "X", "Y", "Z"}, campaign)


def test_empty_cluster_hits_nothing():
    assert not ct.cluster_hits_campaign(set(), ["A", "B", "C"])


# --- classification, including rule 1 --------------------------------------


@pytest.fixture
def small_world():
    accounts_by_campaign = {
        1: ("A", "B", "C", "D"),
        2: ("P", "Q", "R", "S"),
    }
    campaign_splits = {1: "test", 2: "train"}
    return accounts_by_campaign, campaign_splits


def test_hit_on_the_evaluated_split(small_world):
    accounts, splits = small_world
    kind, ids = ct.classify_cluster({"A", "B", "C"}, "test", accounts, splits, set())
    assert kind == "hit"
    assert ids == [1]


def test_hit_on_another_split_is_an_other_split_hit(small_world):
    """Rule 1: a correct find, but not of what we are scoring."""
    accounts, splits = small_world
    kind, ids = ct.classify_cluster({"P", "Q", "R"}, "test", accounts, splits, set())
    assert kind == "other_split_hit"
    assert ids == [2]


def test_a_cluster_hitting_both_counts_for_the_evaluated_split(small_world):
    accounts, splits = small_world
    accounts = {**accounts, 3: ("A", "B", "C", "D")}
    splits = {**splits, 3: "train"}
    kind, ids = ct.classify_cluster({"A", "B", "C"}, "test", accounts, splits, set())
    assert kind == "hit"
    assert ids == [1]  # only the evaluated split's campaign is credited


def test_ambiguous_cluster_is_not_a_false_alarm(small_world):
    """A cluster made of accounts doing unassigned laundering has found real
    laundering with no campaign to match, so it is neither."""
    accounts, splits = small_world
    unassigned = {"U1", "U2", "U3"}
    kind, ids = ct.classify_cluster({"U1", "U2", "U3"}, "test", accounts, splits, unassigned)
    assert kind == "ambiguous"
    assert ids == []


def test_a_cluster_of_ordinary_accounts_is_a_false_alarm(small_world):
    accounts, splits = small_world
    kind, ids = ct.classify_cluster({"X", "Y", "Z"}, "test", accounts, splits, set())
    assert kind == "false_alarm"
    assert ids == []


def test_ambiguity_needs_a_majority():
    """One unassigned-laundering account out of four is not "mostly"."""
    assert not ct.is_ambiguous({"U1", "X", "Y", "Z"}, {"U1"})
    assert ct.is_ambiguous({"U1", "U2", "X", "Y"}, {"U1", "U2"})  # exactly 50%


# --- end to end, with hand-computed expectations ---------------------------

CSV_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]


def _world(campaign_rows, background_rows=(), split="test"):
    """Build the (full_df, camp, pattern_df, accounts) a split evaluation
    needs, straight from explicit rows."""
    rows = []
    for ts, src, dst, campaign_id in list(campaign_rows) + list(background_rows):
        rows.append(
            {
                "ts": pd.Timestamp(ts),
                "src": src,
                "dst": dst,
                "From Bank": int(src.split("_")[0]),
                "To Bank": int(dst.split("_")[0]),
                "Amount Paid": 100.0,
                "Amount Received": 100.0,
                "Payment Currency": "US Dollar",
                "Receiving Currency": "US Dollar",
                "Payment Format": "ACH",
                "Is Laundering": 1 if campaign_id is not None else 0,
                "campaign_id": campaign_id if campaign_id is not None else -1,
            }
        )
    full_df = pd.DataFrame(rows)

    pattern_df = full_df[full_df["campaign_id"] != -1].copy()
    camp_rows = []
    for cid, group in pattern_df.groupby("campaign_id"):
        accounts = sorted(set(group["src"]) | set(group["dst"]))
        start, end = group["ts"].min(), group["ts"].max()
        camp_rows.append(
            {
                "campaign_id": int(cid),
                "base_type": "CYCLE",
                "n_txn": len(group),
                "n_accounts": len(accounts),
                "n_banks": 2,
                "start": start,
                "end": end,
                "duration_h": (end - start).total_seconds() / 3600,
                "crosses_cutoff": False,
                "deadline": end,
                "eval_ok": True,
                "split": split,
                "group": "fragmentable",
                "shares_train_account": False,
                "reassignable": True,
            }
        )
    camp = pd.DataFrame(camp_rows)
    accounts_by_campaign = {
        int(cid): tuple(sorted(set(g["src"]) | set(g["dst"])))
        for cid, g in pattern_df.groupby("campaign_id")
    }
    return full_df, camp, pattern_df, accounts_by_campaign


def _oracle(window, t):
    return scorers.oracle_scores(window["Is Laundering"].to_numpy())


def test_campaign_detected_at_the_expected_time_with_the_expected_lead_time():
    """Three edges on Sept 6 at 07:00/08:00/09:00 give four accounts, plus a
    fourth edge on Sept 7 20:00 so the campaign is still open.

    The first test detection time whose 24h window contains three edges is
    Sept 6 12:00. end = Sept 7 20:00, so:
        lead_time_h = 32.0
        duration    = Sept 6 07:00 -> Sept 7 20:00 = 37.0h
        normalized  = 32/37 = 0.8649
        frac_observed = 3 of 4 = 0.75
    """
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
        ]
    )
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    outcome = result.outcomes[0]
    assert outcome.detected
    assert outcome.detection_time == pd.Timestamp("2022-09-06 12:00")
    assert outcome.lead_time_h == pytest.approx(32.0)
    assert outcome.normalized_lead_time == pytest.approx(32.0 / 37.0, rel=1e-6)
    assert outcome.frac_observed == pytest.approx(0.75)


def test_a_campaign_whose_deadline_passes_first_is_missed():
    """The same three edges, but nothing after them: end = Sept 6 09:00, so
    the deadline is before the first detection time that could see three
    edges (Sept 6 12:00). Detected too late means missed."""
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
        ]
    )
    assert camp.loc[0, "deadline"] == pd.Timestamp("2022-09-06 09:00")
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    outcome = result.outcomes[0]
    assert not outcome.detected
    assert outcome.detection_time is None
    assert outcome.lead_time_h is None


def test_lead_time_is_never_negative():
    """deadline = min(end, CUTOFF) <= end, and detection requires
    t <= deadline, so a detected campaign always has lead time >= 0."""
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-09 20:00", "1_D", "1_A", 0),
        ]
    )
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    for outcome in result.outcomes:
        if outcome.detected:
            assert outcome.lead_time_h >= 0


def test_ambiguous_clusters_are_counted_and_not_charged_as_false_alarms():
    """Unassigned laundering (label 1, no campaign) forms its own cluster.
    It must land in `ambiguous`, leaving false alarms at zero."""
    full_df, camp, pattern_df, accounts = _world(
        campaign_rows=[
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
        ],
    )
    # three unassigned-laundering edges among their own accounts
    extra = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp(ts), "src": src, "dst": dst,
                "From Bank": 9, "To Bank": 9, "Amount Paid": 10.0,
                "Amount Received": 10.0, "Payment Currency": "US Dollar",
                "Receiving Currency": "US Dollar", "Payment Format": "ACH",
                "Is Laundering": 1, "campaign_id": -1,
            }
            for ts, src, dst in [
                ("2022-09-06 07:30", "9_U1", "9_U2"),
                ("2022-09-06 07:40", "9_U2", "9_U3"),
                ("2022-09-06 07:50", "9_U3", "9_U4"),
            ]
        ]
    )
    full_df = pd.concat([full_df, extra], ignore_index=True)

    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    assert result.n_ambiguous >= 1
    assert result.n_false_alarms == 0
    assert result.outcomes[0].detected


def test_other_split_hits_are_separated_from_hits_and_false_alarms():
    """A train campaign present in a test window must not be scored as a
    test hit, nor charged as a false alarm (rule 1)."""
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
            ("2022-09-06 07:10", "2_P", "2_Q", 1),
            ("2022-09-06 08:10", "2_Q", "2_R", 1),
            ("2022-09-06 09:10", "2_R", "2_S", 1),
        ]
    )
    camp.loc[camp["campaign_id"] == 1, "split"] = "train"

    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    assert result.n_hits_clusters >= 1
    assert result.n_other_split_hits >= 1
    assert result.n_false_alarms == 0
    # only the test campaign is in the outcome list
    assert [o.campaign_id for o in result.outcomes] == [0]


def test_random_scores_find_nothing_and_raise_false_alarms():
    """The other half of the sanity pair, on a hand-made world."""
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
        ],
        background_rows=[
            (f"2022-09-06 {7 + (i % 5):02d}:{i % 60:02d}", f"5_X{i}", f"5_X{i + 1}", None)
            for i in range(60)
        ],
    )

    def random_scorer(window, t):
        return scorers.random_scores(len(window), 123)

    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", random_scorer,
        tau=0.5, lookback_h=24, scorer_name="random",
    )
    assert not result.outcomes[0].detected
    assert result.n_false_alarms > 0


# --- metrics ---------------------------------------------------------------


def test_summary_subsets_and_precision_exclude_other_and_ambiguous():
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
        ]
    )
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    for subset in constants.TEST_SUBSETS:
        summary = evaluation.summarize(result, subset=subset)
        assert summary["campaign_recall"] == 1.0
        assert summary["campaign_precision"] == 1.0
        assert summary["false_alarms"] == 0
        assert summary["test_subset"] == subset


def test_bootstrap_ci_is_seeded_and_brackets_the_statistic():
    values = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    a = evaluation.bootstrap_ci(values, seed=1)
    b = evaluation.bootstrap_ci(values, seed=1)
    assert a == b
    low, high = a
    assert low <= float(np.median(values)) <= high


def test_bootstrap_ci_needs_at_least_two_values():
    assert evaluation.bootstrap_ci([]) == (None, None)
    assert evaluation.bootstrap_ci([5.0]) == (None, None)


def test_bootstrap_ci_ignores_missing_values():
    assert evaluation.bootstrap_ci([1.0, None, 3.0, None])[0] is not None


def test_summarize_by_group_and_base_type():
    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-06 07:00", "1_A", "1_B", 0),
            ("2022-09-06 08:00", "1_B", "1_C", 0),
            ("2022-09-06 09:00", "1_C", "1_D", 0),
            ("2022-09-07 20:00", "1_D", "1_A", 0),
        ]
    )
    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "test", _oracle,
        tau=0.5, lookback_h=24, scorer_name="oracle",
    )
    for attribute in ("group", "base_type"):
        rows = evaluation.summarize_by(result, attribute)
        assert len(rows) == 1
        assert rows[0]["campaign_recall"] == 1.0


def test_scipy_and_python_component_labellings_agree():
    """`extract_clusters` uses scipy when available. The two implementations
    must partition the graph identically, or results would depend on which
    machine ran them."""
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 4000
    accounts = [f"acct{i}" for i in range(1200)]
    src = [accounts[i] for i in rng.integers(0, len(accounts), n)]
    dst = [accounts[i] for i in rng.integers(0, len(accounts), n)]
    scores = rng.random(n)

    keep = np.flatnonzero(scores >= 0.5)
    kept_src = np.asarray(src, dtype=object)[keep]
    kept_dst = np.asarray(dst, dtype=object)[keep]
    codes, uniques = pd.factorize(np.concatenate([kept_src, kept_dst]), sort=True)
    half = keep.size

    def partition(labels):
        groups: dict[int, set[str]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), set()).add(uniques[index])
        return {frozenset(members) for members in groups.values()}

    try:
        fast = ct._components_scipy(codes[:half], codes[half:], len(uniques))
    except ImportError:
        pytest.skip("scipy not installed")
    slow = ct._components_python(codes[:half], codes[half:], len(uniques))
    assert partition(fast) == partition(slow)


# --- tau selection ---------------------------------------------------------


def _val_world():
    """A validation campaign plus background, for tau selection."""
    return _world(
        [
            ("2022-09-05 07:00", "1_A", "1_B", 0),
            ("2022-09-05 08:00", "1_B", "1_C", 0),
            ("2022-09-05 09:00", "1_C", "1_D", 0),
            ("2022-09-06 20:00", "1_D", "1_A", 0),
        ],
        background_rows=[
            (f"2022-09-05 {7 + (i % 10):02d}:{i % 60:02d}", f"5_X{i}", f"5_X{i + 1}", None)
            for i in range(40)
        ],
        split="val",
    )


def test_select_tau_uses_validation_only_and_returns_the_curve():
    full_df, camp, pattern_df, accounts = _val_world()
    tau, curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts, _oracle,
        tau_grid=(0.25, 0.5, 0.75), lookback_h=24, scorer_name="oracle",
    )
    assert [row["tau"] for row in curve] == [0.25, 0.5, 0.75]
    assert all("campaign_f1" in row for row in curve)
    # the oracle's scores are 0/1, so every tau in (0, 1] is equivalent and
    # the tie-break picks the lowest
    assert tau == 0.25


def test_select_tau_prefers_the_higher_f1():
    """A scorer that only separates above 0.6 must select a tau above it."""
    full_df, camp, pattern_df, accounts = _val_world()

    def stepped(window, t):
        # laundering edges score 0.8; everything else 0.7, so tau <= 0.7
        # drags in the whole background and destroys precision
        labels = window["Is Laundering"].to_numpy()
        return np.where(labels == 1, 0.8, 0.7)

    tau, curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts, stepped,
        tau_grid=(0.5, 0.75), lookback_h=24, scorer_name="stepped",
    )
    by_tau = {row["tau"]: row for row in curve}
    assert by_tau[0.75]["campaign_f1"] >= by_tau[0.5]["campaign_f1"]
    assert tau == 0.75


def test_campaign_f1_is_zero_when_nothing_is_found():
    full_df, camp, pattern_df, accounts = _val_world()

    def nothing(window, t):
        return np.zeros(len(window))

    result = evaluation.evaluate_split(
        full_df, camp, pattern_df, accounts, "val", nothing,
        tau=0.5, lookback_h=24, scorer_name="none",
    )
    metrics = evaluation.campaign_f1(result)
    assert metrics["campaign_recall"] == 0.0
    assert metrics["campaign_f1"] == 0.0


def test_default_tau_grid_is_ordered_and_within_range():
    grid = evaluation.DEFAULT_TAU_GRID
    assert list(grid) == sorted(grid)
    assert all(0 < tau <= 1 for tau in grid)


def test_scorers_reject_unknown_names():
    with pytest.raises(ValueError, match="unknown sanity scorer"):
        scorers.score_window("magic", np.array([0, 1]))


def test_random_scorer_is_seeded():
    a = scorers.random_scores(50, 4)
    b = scorers.random_scores(50, 4)
    c = scorers.random_scores(50, 5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
