import random

from fraudcamp import visibility


def test_visibility_chain_is_full():
    edges = [("A", "B"), ("B", "C")]
    assignment = {"A": 0, "B": 1, "C": 2}
    assert visibility.visibility_of(edges, assignment, k=3) == 1.0


def test_visibility_fanout_is_full():
    edges = [("A", "B1"), ("A", "B2"), ("A", "B3")]
    assignment = {"A": 0, "B1": 1, "B2": 2, "B3": 3}
    assert visibility.visibility_of(edges, assignment, k=4) == 1.0


def test_visibility_no_overlap_is_low():
    # each institution holds both endpoints of exactly one edge, so no
    # institution sees more than 1 of the 3 edges
    edges = [("A", "B"), ("C", "D"), ("E", "F")]
    assignment = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2}
    assert visibility.visibility_of(edges, assignment, k=3) == 1 / 3


def test_target_one_puts_whole_campaign_in_one_institution():
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    free = ["A", "B", "C", "D"]
    rng = random.Random(0)
    out = visibility.assign_campaign(edges, free, {}, target=1.0, rng=rng, k=8)
    assert len(set(out.values())) == 1


def test_bin_of_thresholds():
    assert visibility.bin_of(0.9) == "100"
    assert visibility.bin_of(0.95) == "100"
    assert visibility.bin_of(0.89) == "75"
    assert visibility.bin_of(0.65) == "75"
    assert visibility.bin_of(0.64) == "50"
    assert visibility.bin_of(0.45) == "50"
    assert visibility.bin_of(0.44) == "low"
    assert visibility.bin_of(0.0) == "low"


def test_assign_campaign_never_reassigns_locked_accounts():
    edges = [("A", "B"), ("B", "C")]
    free = ["A", "C"]
    locked = {"B": 5}
    out = visibility.assign_campaign(edges, free, locked, target=0.5, rng=random.Random(0), k=8)
    assert set(out.keys()) == {"A", "C"}


def test_conflict_resolution_earliest_campaign_wins():
    # campaign 0 starts first and shares account "shared" with campaign 1.
    # campaign 0 is processed first (caller sorts by start time), so campaign
    # 1 must receive "shared" as a locked account, not reassign it.
    edges0 = [("shared", "x0"), ("x0", "y0"), ("y0", "z0")]
    accounts0 = {"shared", "x0", "y0", "z0"}
    edges1 = [("shared", "x1")]
    accounts1 = {"shared", "x1"}

    ordered = [
        (0, edges0, accounts0),
        (1, edges1, accounts1),
    ]
    expected_campaign0_only = visibility.build_global_assignment([(0, edges0, accounts0)], target=0.5, seed=0, k=8)

    global_assignment = visibility.build_global_assignment(ordered, target=0.5, seed=0, k=8)
    assert global_assignment["shared"] == expected_campaign0_only["shared"]


def test_recompute_achieved_uses_final_assignment():
    edges0 = [("shared", "x0")]
    accounts0 = {"shared", "x0"}
    edges1 = [("shared", "x1")]
    accounts1 = {"shared", "x1"}
    ordered = [(0, edges0, accounts0), (1, edges1, accounts1)]
    global_assignment = visibility.build_global_assignment(ordered, target=1.0, seed=1, k=8)
    bank2inst = {}  # no natural fallback needed; all accounts were reassigned
    achieved = visibility.recompute_achieved([(0, edges0), (1, edges1)], global_assignment, bank2inst, k=8)
    # target 1.0 always reaches full visibility for a single-edge campaign
    assert achieved[0] == 1.0
    assert achieved[1] == 1.0
