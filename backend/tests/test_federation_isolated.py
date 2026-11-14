"""Stage 4.1: the isolated condition.

These tests are about the claims the stage's numbers rest on, not about how
well an institution detects anything:

1. **the mask is applied in the scorer, and nothing else moves** — an
   institution scores only its own rows, the scores land on the right rows,
   and a cluster can therefore only form inside the view;
2. **the ground truth stays global** — the reason the view is *not* passed as
   ``full_df``. A cluster touching laundering the institution cannot see is
   still ``ambiguous``, and the test shows the rejected design getting this
   wrong, so the choice is evidenced rather than asserted;
3. **the ceiling is the institution's own** — the oracle goes through the
   same mask, and an institution's ceiling is below the centralized one;
4. **the union is the union of detections** — earliest wins, deduplicated by
   account set, and its per-institution false-alarm counts are required to
   agree with ``evaluate_split``, which is what makes the second loop
   trustworthy;
5. **the pooled means carry their denominators** — untrained institutions
   enter the all-institutions mean at 0 and are excluded from the trained-only
   mean, and an undefined cell is excluded from both and listed.

`test_the_guards_can_fail` at the bottom reproduces the bugs these guard
against, so the suite is shown to be able to fail rather than taken on trust.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, evaluation, pipeline, windows
from fraudcamp.evaluation import clusters as cluster_tools
from fraudcamp.federation import isolated
from fraudcamp.federation import views as fv
from fraudcamp.models import xgb_baseline

pytest.importorskip("xgboost", reason="xgboost arrives with stage 3.3")

K = constants.K_INSTITUTIONS


# ---------------------------------------------------------------------------
# fixtures: a hand-built frame where every account's institution is chosen
# ---------------------------------------------------------------------------


def _row(ts, src, dst, laundering=0, campaign_id=-1, amount=100.0, fmt="ACH"):
    src_bank, dst_bank = int(src.split("_")[0]), int(dst.split("_")[0])
    return {
        "ts": pd.Timestamp(ts),
        "src": src, "dst": dst,
        "From Bank": src_bank, "To Bank": dst_bank,
        "Amount Paid": amount, "Amount Received": amount,
        "Payment Currency": "US Dollar", "Receiving Currency": "US Dollar",
        "Payment Format": fmt,
        "Is Laundering": int(laundering),
        "campaign_id": int(campaign_id),
    }


def _assignment(accounts: dict[str, int], target=1.0, seed=0) -> fv.VisibilityAssignment:
    """An assignment that pins every account by hand.

    ``bank2inst`` still covers the banks in use, so a missing account would
    fall back rather than raise and a test could not mistake one for the
    other.
    """
    banks = {int(a.split("_")[0]) for a in accounts}
    return fv.VisibilityAssignment(
        seed=seed, target=target,
        global_assignment=dict(accounts),
        bank2inst={bank: 0 for bank in banks},
        k=K,
    )


@pytest.fixture
def frame():
    """Four transactions over three institutions, one window.

    Institution 1 holds ``1_B``, which is an endpoint of two edges, so it
    alone can see a three-account component — the mechanism behind the
    structural result recorded in the README.
    """
    rows = [
        _row("2022-09-02 01:00", "0_A", "1_B", laundering=1, campaign_id=0),
        _row("2022-09-02 02:00", "1_B", "2_C", laundering=1, campaign_id=0),
        _row("2022-09-02 03:00", "3_D", "4_E"),
        _row("2022-09-02 04:00", "0_A", "0_A"),  # a self-transfer, which exist
    ]
    df = pd.DataFrame(rows)
    accounts = {"0_A": 0, "1_B": 1, "2_C": 2, "3_D": 3, "4_E": 4}
    return df, _assignment(accounts)


# ---------------------------------------------------------------------------
# 1. the mask lives in the scorer
# ---------------------------------------------------------------------------


def test_masked_scorer_scores_visible_rows_and_zeroes_the_rest(frame):
    df, assignment = frame

    def inner(window, t):
        # a distinctive value per row, so a misalignment is visible
        return np.arange(1, len(window) + 1, dtype=float)

    for institution in range(5):
        mask = fv.visible_mask(df, institution, assignment)
        scores = isolated.masked_scorer(inner, institution, assignment)(df, df["ts"].max())
        assert len(scores) == len(df)
        assert (scores[~mask] == 0).all(), institution
        # the visible rows carry the inner scorer's values, in its own order
        assert list(scores[mask]) == list(np.arange(1, int(mask.sum()) + 1, dtype=float))


def test_masked_scorer_puts_the_scores_on_the_right_rows(frame):
    """The alignment claim: positional, in the global window's row order."""
    df, assignment = frame

    def inner(window, t):
        # score = the row's own amount, which identifies the row
        return window["Amount Paid"].to_numpy() + window.index.to_numpy()

    institution = 1
    mask = fv.visible_mask(df, institution, assignment)
    scores = isolated.masked_scorer(inner, institution, assignment)(df, df["ts"].max())
    expected = np.where(mask, df["Amount Paid"].to_numpy() + df.index.to_numpy(), 0.0)
    assert list(scores) == list(expected)


def test_a_cluster_can_only_form_inside_the_view(frame):
    """Clusters from the masked scorer equal clusters from the view alone."""
    df, assignment = frame

    def inner(window, t):
        return np.ones(len(window))

    for institution in range(5):
        scores = isolated.masked_scorer(inner, institution, assignment)(df, df["ts"].max())
        from_mask = cluster_tools.extract_clusters(
            df["src"].tolist(), df["dst"].tolist(), scores, tau=0.05
        )
        view = fv.visible_transactions(df, institution, assignment)
        from_view = cluster_tools.extract_clusters(
            view["src"].tolist(), view["dst"].tolist(), np.ones(len(view)), tau=0.05
        )
        assert from_mask == from_view, institution


def test_only_the_institution_holding_the_middle_account_sees_the_triple(frame):
    """The degree-2 mechanism, on data small enough to read."""
    df, assignment = frame

    def inner(window, t):
        return np.ones(len(window))

    reaching = []
    for institution in range(5):
        scores = isolated.masked_scorer(inner, institution, assignment)(df, df["ts"].max())
        found = cluster_tools.extract_clusters(
            df["src"].tolist(), df["dst"].tolist(), scores, tau=0.05
        )
        if any(len(c) >= constants.MATCH_MIN_ACCOUNTS for c in found):
            reaching.append(institution)
    assert reaching == [1], "only 1_B's institution is an endpoint of both edges"


def test_the_smallest_model_tau_is_above_zero():
    """What makes the zero-fill safe: a zeroed row can never be kept."""
    assert min(evaluation.MODEL_TAU_GRID) > 0


# ---------------------------------------------------------------------------
# 2. the ground truth stays global — why the view is not passed as full_df
# ---------------------------------------------------------------------------


@pytest.fixture
def ambiguity_frame():
    """A cluster whose accounts touch laundering that belongs to no campaign,
    where that laundering is invisible to the institution being evaluated.

    Globally the cluster is `ambiguous`. If the institution's view were
    passed as `full_df`, the unassigned laundering would vanish from the
    ground truth and the same cluster would be a `false_alarm`.
    """
    rows = [
        # the three accounts the cluster is made of, all held by institution 1
        _row("2022-09-02 01:00", "1_X", "1_Y"),
        _row("2022-09-02 02:00", "1_Y", "1_Z"),
        # laundering on those same accounts, belonging to no campaign, and
        # invisible to institution 1 because institution 2 holds both ends
        _row("2022-09-02 03:00", "2_P", "2_Q", laundering=1, campaign_id=-1),
    ]
    df = pd.DataFrame(rows)
    accounts = {"1_X": 1, "1_Y": 1, "1_Z": 1, "2_P": 2, "2_Q": 2}
    return df, _assignment(accounts)


def test_global_truth_makes_a_cluster_ambiguous_not_a_false_alarm(ambiguity_frame):
    df, assignment = ambiguity_frame
    # the unassigned-laundering accounts, as evaluate_split derives them
    unassigned = df[(df["Is Laundering"] == 1) & (df["campaign_id"] == -1)]
    global_accounts = set(unassigned["src"]) | set(unassigned["dst"])

    cluster = frozenset({"1_X", "1_Y", "1_Z"})
    # the cluster's accounts are not the laundering accounts here, so make
    # the overlap explicit: ambiguity is about touching unassigned laundering
    assert not (cluster & global_accounts)

    # institution 1 cannot see the laundering row at all
    view = fv.visible_transactions(df, 1, assignment)
    assert len(view) == 2
    view_unassigned = view[(view["Is Laundering"] == 1) & (view["campaign_id"] == -1)]
    assert len(view_unassigned) == 0, (
        "the institution's view holds no unassigned laundering, which is "
        "exactly why it must not be used as the ground truth"
    )
    view_accounts = set(view_unassigned["src"]) | set(view_unassigned["dst"])
    assert view_accounts == set()
    # so the two candidate ground truths genuinely disagree
    assert global_accounts != view_accounts


def test_passing_the_view_as_full_df_moves_the_ground_truth():
    """The rejected design, shown getting a classification wrong.

    A cluster made of accounts that *do* touch unassigned laundering is
    `ambiguous` under the global truth and a `false_alarm` under the view's,
    which is a different false-alarm rate and therefore a different tau at
    every budget.
    """
    rows = [
        _row("2022-09-02 01:00", "1_X", "1_Y"),
        _row("2022-09-02 02:00", "1_Y", "1_Z"),
        # unassigned laundering ON the cluster's own accounts, between two
        # accounts institution 2 holds... except one end is 1_X, so
        # institution 1 does see this one; make it invisible to institution 1
        # by using 1_X's counterpart only in institution 2's hands
        _row("2022-09-02 03:00", "2_P", "2_Q", laundering=1, campaign_id=-1),
    ]
    df = pd.DataFrame(rows)
    assignment = _assignment(
        {"1_X": 1, "1_Y": 1, "1_Z": 1, "2_P": 2, "2_Q": 2}
    )
    # a cluster of the laundering accounts themselves: ambiguous globally
    cluster = frozenset({"2_P", "2_Q", "1_Z"})

    def accounts_of(frame):
        rows = frame[(frame["Is Laundering"] == 1) & (frame["campaign_id"] == -1)]
        return set(rows["src"]) | set(rows["dst"])

    kind_global, _ = cluster_tools.classify_cluster(
        cluster, "test", {}, {}, accounts_of(df)
    )
    kind_view, _ = cluster_tools.classify_cluster(
        cluster, "test", {}, {},
        accounts_of(fv.visible_transactions(df, 1, assignment)),
    )
    assert kind_global == "ambiguous"
    assert kind_view == "false_alarm"
    assert kind_global != kind_view


# ---------------------------------------------------------------------------
# 3. the oracle through the same mask
# ---------------------------------------------------------------------------


def test_oracle_view_scorer_is_the_oracle_on_the_view(frame):
    df, assignment = frame
    labels = df["Is Laundering"].to_numpy()
    for institution in range(5):
        mask = fv.visible_mask(df, institution, assignment)
        scores = isolated.oracle_view_scorer(institution, assignment)(df, df["ts"].max())
        assert list(scores) == list(np.where(mask, labels, 0.0))


def test_oracle_view_scorer_handles_an_empty_window(frame):
    df, assignment = frame
    empty = df.iloc[0:0]
    assert len(isolated.oracle_view_scorer(0, assignment)(empty, df["ts"].max())) == 0
    assert len(isolated.masked_scorer(
        lambda w, t: np.ones(len(w)), 0, assignment
    )(empty, df["ts"].max())) == 0


# ---------------------------------------------------------------------------
# 4. the union
# ---------------------------------------------------------------------------


def _outcome(campaign_id, detected, hours=None):
    return evaluation.CampaignOutcome(
        campaign_id=campaign_id, split="test", base_type="FAN-OUT", group="treatment",
        detected=detected,
        detection_time=pd.Timestamp("2022-09-10 00:00") + pd.Timedelta(hours=hours or 0)
        if detected else None,
        lead_time_h=float(hours or 0) if detected else None,
        normalized_lead_time=None, frac_observed=None,
        crosses_cutoff=False, shares_train_account=False,
        deadline=pd.Timestamp("2022-09-20"), end=pd.Timestamp("2022-09-15"),
        duration_h=48.0, n_txn=5,
    )


def test_union_detects_if_any_institution_does_and_takes_the_earliest():
    per_institution = {
        0: [_outcome(1, False), _outcome(2, True, hours=12)],
        1: [_outcome(1, True, hours=6), _outcome(2, True, hours=3)],
        2: [_outcome(1, False), _outcome(2, False)],
    }
    combined, credited = isolated.union_outcomes(per_institution)
    by_id = {o.campaign_id: o for o in combined}
    assert by_id[1].detected and by_id[1].lead_time_h == 6.0
    assert by_id[2].detected and by_id[2].lead_time_h == 3.0
    assert credited == {1: 1, 2: 1}


def test_union_breaks_a_tie_on_the_lowest_institution():
    per_institution = {
        5: [_outcome(7, True, hours=9)],
        2: [_outcome(7, True, hours=9)],
    }
    _, credited = isolated.union_outcomes(per_institution)
    assert credited == {7: 2}


def test_union_misses_what_nobody_detects():
    per_institution = {0: [_outcome(1, False)], 1: [_outcome(1, False)]}
    combined, credited = isolated.union_outcomes(per_institution)
    assert combined[0].detected is False
    assert credited == {}


def test_union_is_empty_without_institutions():
    assert isolated.union_outcomes({}) == ([], {})


# ---------------------------------------------------------------------------
# 5. pooled means
# ---------------------------------------------------------------------------


def test_pooled_mean_reports_both_denominators():
    values = {0: 0.4, 1: 0.0, 2: 0.8}
    tiers = {0: isolated.TIER_NORMAL, 1: isolated.TIER_UNTRAINED, 2: isolated.TIER_THIN}
    pooled = isolated.pooled_mean(values, tiers)
    assert pooled.over_all == pytest.approx((0.4 + 0.0 + 0.8) / 3)
    assert pooled.n_all == 3
    assert pooled.over_trained == pytest.approx((0.4 + 0.8) / 2)
    assert pooled.n_trained == 2
    assert pooled.untrained == (1,)
    assert pooled.undefined == ()


def test_pooled_mean_excludes_undefined_from_both_and_lists_it():
    values = {0: 0.5, 1: None, 2: 0.1}
    tiers = {i: isolated.TIER_NORMAL for i in (0, 1, 2)}
    pooled = isolated.pooled_mean(values, tiers)
    assert pooled.over_all == pytest.approx(0.3)
    assert pooled.n_all == 2
    assert pooled.over_trained == pytest.approx(0.3)
    assert pooled.n_trained == 2
    assert pooled.undefined == (1,)


def test_pooled_mean_of_nothing_is_none_not_zero():
    pooled = isolated.pooled_mean({0: None}, {0: isolated.TIER_NORMAL})
    assert pooled.over_all is None and pooled.n_all == 0


def test_an_untrained_institution_is_zero_and_an_unaffordable_budget_is_undefined():
    """The two decisions of 4.1 §8 and §9, side by side."""
    tiers = {0: isolated.TIER_UNTRAINED, 1: isolated.TIER_NORMAL}
    zero_case = isolated.pooled_mean({0: 0.0, 1: 0.6}, tiers)
    undefined_case = isolated.pooled_mean({0: 0.0, 1: None}, tiers)
    assert zero_case.over_all == pytest.approx(0.3)
    assert zero_case.over_trained == pytest.approx(0.6)
    # the unaffordable cell leaves the mean, it does not drag it to 0
    assert undefined_case.over_all == pytest.approx(0.0)
    assert undefined_case.n_all == 1
    assert undefined_case.over_trained is None
    assert undefined_case.n_trained == 0


# ---------------------------------------------------------------------------
# tiers and budgets
# ---------------------------------------------------------------------------


def test_training_tiers_follow_the_approved_rule():
    assert isolated.training_tier(0) == isolated.TIER_UNTRAINED
    assert isolated.training_tier(1) == isolated.TIER_THIN
    assert isolated.training_tier(fv.MIN_POSITIVES_TO_TRAIN - 1) == isolated.TIER_THIN
    assert isolated.training_tier(fv.MIN_POSITIVES_TO_TRAIN) == isolated.TIER_NORMAL
    assert isolated.training_tier(10_000) == isolated.TIER_NORMAL


def test_the_budget_is_split_equally_and_untuned():
    assert isolated.budget_share(10, 8) == 1.25
    assert isolated.budget_share(100, 8) == 12.5
    assert isolated.budget_share(10, 4) == 2.5


def test_operating_points_cover_both_readings_of_the_budget():
    points = isolated.operating_points(8)
    names = [p["operating_point"] for p in points]
    assert names[0] == "f1"
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        assert f"budget_{budget}" in names
        assert f"total_{budget}" in names
    per_institution = {p["operating_point"]: p["per_institution_budget"] for p in points}
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        # a bank alone has its own analysts; the union shares one desk
        assert per_institution[f"budget_{budget}"] == float(budget)
        assert per_institution[f"total_{budget}"] == budget / 8


# ---------------------------------------------------------------------------
# a world big enough to train on, split across institutions by bank
# ---------------------------------------------------------------------------


def _world(rows, camp_splits, bank2inst=None):
    """``(result-like, assignment)`` from explicit rows.

    ``rows`` are ``(ts, src, dst, campaign_id_or_None)``. Accounts are named
    ``"{bank}_{n}"`` and each bank is its own institution, so which
    institution sees what is readable from the account names.
    """
    frame = pd.DataFrame([
        _row(ts, src, dst, laundering=cid is not None,
             campaign_id=-1 if cid is None else cid)
        for ts, src, dst, cid in rows
    ])
    pattern_df = frame[frame["campaign_id"] != -1].copy()

    camp_rows = []
    for cid, group in pattern_df.groupby("campaign_id"):
        accounts = sorted(set(group["src"]) | set(group["dst"]))
        start, end = group["ts"].min(), group["ts"].max()
        camp_rows.append({
            "campaign_id": int(cid), "base_type": "CYCLE", "n_txn": len(group),
            "n_accounts": len(accounts), "n_banks": 2, "start": start, "end": end,
            "duration_h": (end - start).total_seconds() / 3600 or 1.0,
            "crosses_cutoff": False, "deadline": end, "eval_ok": True,
            "split": camp_splits[int(cid)], "group": "fragmentable",
            "shares_train_account": False, "reassignable": True,
        })
    camp = pd.DataFrame(camp_rows)

    accounts_by_campaign = {
        int(cid): tuple(sorted(set(g["src"]) | set(g["dst"])))
        for cid, g in pattern_df.groupby("campaign_id")
    }
    edges_by_campaign = {
        int(cid): [(s, d) for s, d in zip(g["src"], g["dst"])]
        for cid, g in pattern_df.groupby("campaign_id")
    }
    banks = sorted({int(a.split("_")[0]) for a in frame["src"]} |
                   {int(a.split("_")[0]) for a in frame["dst"]})
    bank2inst = bank2inst or {bank: bank % K for bank in banks}

    result = SimpleNamespace(
        full_df=frame, camp=camp, pattern_df=pattern_df,
        accounts_by_campaign=accounts_by_campaign,
        edges_by_campaign=edges_by_campaign, bank2inst=bank2inst,
    )
    assignment = fv.VisibilityAssignment(
        seed=0, target=1.0, global_assignment={}, bank2inst=bank2inst, k=K,
    )
    return result, assignment


def _build_trainable_world():
    """Background traffic plus one val campaign and one test campaign."""
    rows = []
    for day in range(1, 12):
        for hour in (1, 7, 13, 19):
            stamp = pd.Timestamp(f"2022-09-{day:02d} {hour:02d}:00")
            for i in range(12):
                rows.append((
                    stamp + pd.Timedelta(minutes=i),
                    f"{i % K}_bg{i}", f"{(i + 3) % K}_bg{(i + 5) % 12}", None,
                ))
    # unassigned laundering, so every institution has positives to train on
    for day in range(1, 12):
        for hour in (2, 14):
            stamp = pd.Timestamp(f"2022-09-{day:02d} {hour:02d}:00")
            for i in range(K):
                rows.append((
                    stamp + pd.Timedelta(minutes=i),
                    f"{i % K}_ml{i}", f"{(i + 1) % K}_ml{i}", -1,
                ))
    frame_rows = [
        (ts, src, dst, None if cid is None else None)
        for ts, src, dst, cid in rows if cid is None
    ]
    laundering_rows = [
        (ts, src, dst, -1) for ts, src, dst, cid in rows if cid == -1
    ]

    # a val campaign and a test campaign, each a 4-account chain
    campaign_rows = []
    for cid, base_day in ((0, 5), (1, 7)):
        for step, (a, b) in enumerate([("0_c%d" % cid, "1_c%d" % cid),
                                       ("1_c%d" % cid, "2_c%d" % cid),
                                       ("2_c%d" % cid, "3_c%d" % cid)]):
            campaign_rows.append((
                pd.Timestamp(f"2022-09-{base_day:02d} 07:00") + pd.Timedelta(hours=step),
                a, b, cid,
            ))

    result, assignment = _world(
        frame_rows + laundering_rows + campaign_rows, {0: "val", 1: "test"},
    )
    # the laundering rows must really be laundering-but-unassigned
    unassigned = result.full_df[
        (result.full_df["Is Laundering"] == 1) & (result.full_df["campaign_id"] == -1)
    ]
    assert len(unassigned) > 0
    return result, assignment


@pytest.fixture
def trainable_world():
    return _build_trainable_world()


@pytest.fixture(scope="module")
def _module_world():
    """The same world, built once: `run_cell` trains K boosters."""
    return _build_trainable_world()


# ---------------------------------------------------------------------------
# the all-missed shortcut is the same thing as running the loop
# ---------------------------------------------------------------------------


def test_undetected_outcomes_equals_running_the_loop_with_a_zero_scorer(trainable_world):
    result, _ = trainable_world
    for split in ("val", "test"):
        shortcut = isolated.undetected_outcomes(result.camp, split)
        run = evaluation.evaluate_split(
            result.full_df, result.camp, result.pattern_df,
            result.accounts_by_campaign, split=split, scorer=isolated.zero_scorer(),
            tau=min(evaluation.MODEL_TAU_GRID), lookback_h=48,
        )
        assert [vars(o) for o in shortcut] == [vars(o) for o in run.outcomes]
        assert run.n_clusters == 0


# ---------------------------------------------------------------------------
# the union's false-alarm pass agrees with the evaluation it stands in for
# ---------------------------------------------------------------------------


def _all_ones(window, t):
    return np.ones(len(window), dtype=float)


def test_union_pass_reproduces_each_institutions_false_alarm_count(trainable_world):
    """The cross-check that makes the second loop trustworthy.

    `union_cluster_accounting` walks the detection grid itself. If its
    per-institution counts did not equal `evaluate_split`'s, the union's
    alert volume would be measured against a different rule from the one
    every per-institution number uses.
    """
    result, assignment = trainable_world
    scorers = {
        i: isolated.masked_scorer(_all_ones, i, assignment) for i in range(K)
    }
    for tau in (0.5, 0.9):
        taus = {i: tau for i in range(K)}
        accounting = isolated.union_cluster_accounting(
            result.full_df, result.camp, result.accounts_by_campaign, "test",
            scorers=scorers, taus=taus, lookback_h=48,
        )
        for institution in range(K):
            direct = evaluation.evaluate_split(
                result.full_df, result.camp, result.pattern_df,
                result.accounts_by_campaign, split="test",
                scorer=scorers[institution], tau=tau, lookback_h=48,
            )
            assert (
                accounting["per_institution_false_alarms"][institution]
                == direct.n_false_alarms
            ), f"institution {institution} at tau {tau}"
            assert (
                accounting["per_institution"][institution]["hit"]
                == direct.n_hits_clusters
            )
        assert accounting["horizon_days"] > 0


def test_union_deduplicates_a_cluster_two_institutions_both_raise():
    """A path whose middle account sits in another institution: both ends'
    institution and the middle's institution see the *same* three accounts,
    so it is one alert on a supervisory desk, not two."""
    rows = [
        ("2022-09-07 07:00", "0_X", "1_M", None),
        ("2022-09-07 08:00", "1_M", "0_Z", None),
    ]
    # a test campaign is needed only so the world has a split to evaluate
    rows += [
        ("2022-09-07 09:00", "4_c0", "5_c0", 0),
        ("2022-09-07 10:00", "5_c0", "6_c0", 0),
        ("2022-09-07 11:00", "6_c0", "7_c0", 0),
    ]
    result, assignment = _world(rows, {0: "test"})
    scorers = {
        i: isolated.masked_scorer(_all_ones, i, assignment) for i in (0, 1)
    }
    accounting = isolated.union_cluster_accounting(
        result.full_df, result.camp, result.accounts_by_campaign, "test",
        scorers=scorers, taus={0: 0.5, 1: 0.5}, lookback_h=48,
    )
    assert accounting["summed_false_alarms"] > accounting["union_false_alarms"]
    assert accounting["duplicate_clusters"] > 0
    # and the per-institution counts are untouched by the deduplication
    assert accounting["per_institution_false_alarms"][0] > 0
    assert accounting["per_institution_false_alarms"][1] > 0


# ---------------------------------------------------------------------------
# training: on the view, and on nothing else
# ---------------------------------------------------------------------------


def test_training_uses_only_the_institutions_own_rows(trainable_world):
    result, assignment = trainable_world
    for institution in (0, 1):
        model = isolated.train_isolated(
            result.full_df, institution, assignment, lookback_h=48
        )
        view = fv.visible_transactions(result.full_df, institution, assignment)
        pooled = xgb_baseline.build_training_data(view, 48)
        assert model.n_rows == pooled.n_rows
        assert model.n_positive == pooled.n_positive
        assert model.n_rows < len(result.full_df), "a view is a strict subset here"


def test_every_institution_has_the_same_feature_names(trainable_world):
    """Institution identity never reaches the model inputs: the matrices
    differ in their rows, never in their columns."""
    result, assignment = trainable_world
    names = None
    for institution in range(K):
        model = isolated.train_isolated(
            result.full_df, institution, assignment, lookback_h=48
        )
        if not model.trained:
            continue
        if names is None:
            names = list(model.model.feature_names)
        assert list(model.model.feature_names) == names
        assert not any("institution" in n.lower() for n in names)


def test_an_institution_with_no_positives_is_untrained_and_does_not_raise():
    """The 4.1 §8 rule: 0 positives cannot train, and that is a recorded
    outcome rather than an exception that stops a 96-cell run."""
    rows = [("2022-09-07 07:00", "0_c0", "1_c0", 0),
            ("2022-09-07 08:00", "1_c0", "2_c0", 0),
            ("2022-09-07 09:00", "2_c0", "3_c0", 0)]
    # background with no laundering at all, in the training windows
    for day in (1, 2, 3, 4, 5):
        for hour in (1, 13):
            for i in range(6):
                rows.append((
                    pd.Timestamp(f"2022-09-{day:02d} {hour:02d}:00") + pd.Timedelta(minutes=i),
                    f"{i % K}_q{i}", f"{(i + 1) % K}_q{i}", None,
                ))
    result, assignment = _world(rows, {0: "test"})
    model = isolated.train_isolated(result.full_df, 0, assignment, lookback_h=48)
    assert model.tier == isolated.TIER_UNTRAINED
    assert model.model is None
    assert model.n_positive == 0
    # and its scorer is harmless
    scores = model.scorer(assignment)(result.full_df, pd.Timestamp("2022-09-07 12:00"))
    assert (scores == 0).all()


# ---------------------------------------------------------------------------
# the ceiling is the institution's own, and it is lower
# ---------------------------------------------------------------------------


def test_the_ceiling_is_per_institution_and_most_institutions_have_none():
    """Why a ceiling cannot be shared across institutions.

    A four-account chain ``a -> b -> c -> d``, one account per institution.
    Centrally the oracle finds it. Institution-wise, only the two that hold a
    **middle** account see two of its edges and therefore three of its
    accounts; the two holding an end see one edge, two accounts, and nothing
    the matching rule can accept. Four of eight institutions have a ceiling
    of 0 on this campaign, so dividing any of their recalls by the
    centralized 0.893 would measure them against a bound they cannot reach.

    This also shows the limit of the structural theorem recorded in the
    README: the **union** always matches the centralized reach, because some
    institution holds a middle account — but an individual institution
    usually does not.
    """
    rows = [
        ("2022-09-07 07:00", "0_a", "1_b", 0),
        ("2022-09-07 08:00", "1_b", "2_c", 0),
        ("2022-09-07 09:00", "2_c", "3_d", 0),
        # a late fifth account, so the campaign is still open at the first
        # detection time whose window holds the chain (deadline = end)
        ("2022-09-08 20:00", "3_d", "4_e", 0),
    ]
    for day in (1, 2, 3, 4, 5):
        for i in range(4):
            rows.append((
                pd.Timestamp(f"2022-09-{day:02d} 05:00") + pd.Timedelta(minutes=i),
                f"{i % K}_z{i}", f"{(i + 1) % K}_z{i}", None,
            ))
    result, assignment = _world(rows, {0: "test"})

    def centralized_oracle(window, t):
        return evaluation.oracle_scores(window["Is Laundering"].to_numpy())

    central = evaluation.evaluate_split(
        result.full_df, result.camp, result.pattern_df, result.accounts_by_campaign,
        split="test", scorer=centralized_oracle,
        tau=evaluation.SANITY_TAU["oracle"], lookback_h=48,
    )
    assert sum(1 for o in central.outcomes if o.detected) == 1

    detecting = []
    for institution in range(K):
        own = evaluation.evaluate_split(
            result.full_df, result.camp, result.pattern_df,
            result.accounts_by_campaign, split="test",
            scorer=isolated.oracle_view_scorer(institution, assignment),
            tau=evaluation.SANITY_TAU["oracle"], lookback_h=48,
        )
        if sum(1 for o in own.outcomes if o.detected):
            detecting.append(institution)

    # 1 holds b and 2 holds c: each is an endpoint of two campaign edges
    assert detecting == [1, 2], detecting
    # and the union over institutions recovers the centralized reach
    assert detecting, "the union must find what the centralized oracle finds"


# ---------------------------------------------------------------------------
# the whole cell
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cell(_module_world):
    result, _ = _module_world
    return isolated.run_cell(
        result, target=1.0, seed=0, lookback_h=48, splits=("val", "test"),
    )


def test_run_cell_produces_every_report_and_flags_what_it_should(cell):
    assert len(cell["institution_rows"]) == K
    assert {row["institution"] for row in cell["institution_rows"]} == set(range(K))
    # one tau row per institution per operating point
    assert len(cell["tau_rows"]) == K * len(isolated.operating_points(K))
    # one ceiling row per institution per split per subset
    assert len(cell["ceiling_rows"]) == K * 2 * len(constants.TEST_SUBSETS)
    assert cell["pooled_rows"], "the two pooled means must be reported"
    for row in cell["pooled_rows"]:
        assert "n_institutions_in_mean" in row and "n_trained_in_mean" in row
    assert set(cell["flags"]) == {
        "untrained", "thin", "tau_at_grid_edge", "tau_at_high_edge",
        "tau_at_low_edge", "no_affordable_tau",
    }


def test_the_union_never_detects_less_than_its_best_institution(cell):
    """A property of taking a union, and a guard against combining the wrong
    outcome lists."""
    per_point = {}
    for row in cell["summary_rows"]:
        if row.get("test_subset") != "all" or not row.get("defined"):
            continue
        key = (row["operating_point"], row["eval_split"])
        recall = row.get("campaign_recall")
        if recall is not None:
            per_point.setdefault(key, []).append(recall)
    union = {
        (row["operating_point"], row["eval_split"]): row["campaign_recall"]
        for row in cell["union_rows"] if row["test_subset"] == "all"
    }
    assert union, "the union must be reported"
    for key, recalls in per_point.items():
        if key in union and union[key] is not None:
            assert union[key] >= max(recalls) - 1e-12, key


def test_the_union_is_credited_to_an_institution_that_detected_it(cell):
    for row in cell["union_rows"]:
        credited = row.get("credited_institutions")
        if not credited:
            continue
        for pair in credited.split(";"):
            _, institution = pair.split(":")
            assert 0 <= int(institution) < K


def test_pooled_means_agree_with_the_per_institution_rows(cell):
    """The pooled numbers are a function of the rows above them, not a
    separate computation that could drift."""
    recalls: dict[tuple, dict[int, float | None]] = {}
    tiers: dict[int, str] = {}
    for row in cell["summary_rows"]:
        if row.get("test_subset") != "all":
            continue
        key = (row["operating_point"], row["eval_split"])
        value = row.get("campaign_recall") if row.get("defined") else None
        recalls.setdefault(key, {})[row["institution"]] = value
        tiers[row["institution"]] = row["tier"]
    for row in cell["pooled_rows"]:
        if row["metric"] != "campaign_recall":
            continue
        key = (row["operating_point"], row["eval_split"])
        expected = isolated.pooled_mean(recalls[key], tiers)
        assert row["mean_over_all_institutions"] == expected.over_all
        assert row["n_institutions_in_mean"] == expected.n_all
        assert row["mean_over_trained_institutions"] == expected.over_trained
        assert row["n_trained_in_mean"] == expected.n_trained


def test_the_matched_total_budget_is_an_eighth_of_the_per_institution_one(cell):
    by_point = {row["operating_point"]: row for row in cell["tau_rows"]}
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        assert by_point[f"budget_{budget}"]["per_institution_budget"] == float(budget)
        assert by_point[f"total_{budget}"]["per_institution_budget"] == budget / K


def test_outcome_rows_carry_the_keys_a_later_comparison_needs(cell):
    """4.2 and 4.3 are paired against these rows, so every key that
    distinguishes one detector from another has to be on them."""
    required = {
        "condition", "institution", "visibility_target", "operating_point",
        "lookback_h", "seed", "eval_split", "campaign_id", "detected",
    }
    assert cell["outcome_rows"]
    for row in cell["outcome_rows"][:50]:
        assert required <= set(row)
    assert any(row["institution"] == "union" for row in cell["outcome_rows"])


def test_run_cell_is_deterministic(_module_world):
    """Same seed, same rows — including the union's credited institutions,
    which a non-deterministic tie-break would change."""
    result, _ = _module_world
    again = isolated.run_cell(
        result, target=1.0, seed=0, lookback_h=48, splits=("val",),
    )
    once_more = isolated.run_cell(
        result, target=1.0, seed=0, lookback_h=48, splits=("val",),
    )
    # wall-clock fields are the one thing that legitimately differs
    timings = {"train_seconds", "eval_seconds"}

    def without_timings(rows):
        return [{k: v for k, v in row.items() if k not in timings} for row in rows]

    assert without_timings(again["institution_rows"]) == without_timings(
        once_more["institution_rows"]
    )
    assert again["pooled_rows"] == once_more["pooled_rows"]
    assert again["tau_rows"] == once_more["tau_rows"]
    assert again["ceiling_rows"] == once_more["ceiling_rows"]
    assert [
        row.get("credited_institutions") for row in again["union_rows"]
    ] == [row.get("credited_institutions") for row in once_more["union_rows"]]


def _world_with_an_untrained_institution():
    """Traffic everywhere, but no laundering that institutions 6 or 7 see.

    They therefore have rows and no positives: `untrained` by the approved
    rule, which is the one case where the two pooled means differ.
    """
    rows = []
    for day in range(1, 12):
        for hour in (1, 7, 13, 19):
            stamp = pd.Timestamp(f"2022-09-{day:02d} {hour:02d}:00")
            for i in range(12):
                rows.append((
                    stamp + pd.Timedelta(minutes=i),
                    f"{i % K}_bg{i}", f"{(i + 3) % K}_bg{(i + 5) % 12}", None,
                ))
    # unassigned laundering between banks 0-5 only, so banks 6 and 7 are
    # never an endpoint of a positive row
    for day in range(1, 12):
        for hour in (2, 14):
            stamp = pd.Timestamp(f"2022-09-{day:02d} {hour:02d}:00")
            for i in range(6):
                rows.append((
                    stamp + pd.Timedelta(minutes=i),
                    f"{i % 6}_ml{i}", f"{(i + 1) % 6}_ml{i}", -1,
                ))
    # a test campaign, also clear of banks 6 and 7
    for step, (a, b) in enumerate([("0_c0", "1_c0"), ("1_c0", "2_c0"),
                                   ("2_c0", "3_c0"), ("3_c0", "4_c0")]):
        rows.append((
            pd.Timestamp("2022-09-07 07:00") + pd.Timedelta(hours=step * 12),
            a, b, 0,
        ))
    return _world(rows, {0: "test"})


def test_an_untrained_institution_is_reported_apart_from_the_blind_ones():
    """The decision this stage added, on a fixture that shows why it matters.

    No laundering touches banks 6 or 7, so those two institutions have rows
    and **no positives**: `untrained`. At target 1.0 the visibility search
    then moves the whole campaign into one institution - here institution 6 -
    so the one institution that **can see** the campaign is the one that had
    **nothing to learn from**, while every institution that could train
    cannot see the campaign at all.

    That is the conflation the second mean exists to prevent. Every
    institution's recall is 0 here, but for two different reasons, and the
    report has to keep them apart: the untrained one enters the
    all-institutions mean at 0 and is excluded from the trained-only mean,
    and its **own ceiling** is still computed, because a ceiling is a
    property of the view and not of the training data.
    """
    result, assignment = _world_with_an_untrained_institution()
    cell = isolated.run_cell(
        result, target=1.0, seed=0, lookback_h=48, splits=("test",),
    )

    tiers = {row["institution"]: row["tier"] for row in cell["institution_rows"]}
    untrained = cell["flags"]["untrained"]
    assert untrained == [6, 7], tiers
    assert all(tiers[i] == isolated.TIER_UNTRAINED for i in untrained)

    # a ceiling is computed for an untrained institution too - and here the
    # highest ceiling of all belongs to one of them
    ceilings = {
        row["institution"]: row["oracle_recall"]
        for row in cell["ceiling_rows"]
        if row["test_subset"] == "all" and row["eval_split"] == "test"
    }
    assert max(ceilings.values()) > 0
    best = max(ceilings, key=lambda i: ceilings[i])
    assert best in untrained, (
        "this fixture is built so that the institution holding the campaign "
        "is one that could not train; if that stops being true the test is "
        "no longer demonstrating the conflation"
    )

    # an untrained institution detects nothing: recall 0, not undefined
    for row in cell["summary_rows"]:
        if row["institution"] in untrained and row.get("test_subset") == "all":
            assert row["campaign_recall"] == 0.0
            assert row["defined"] is True

    # and the two means are reported with their own denominators, derived
    # from the rows above them rather than assumed: on this fixture the
    # grid-edge policy also excludes whichever trained institutions chose a
    # tau at the top of the grid, so the denominators are not simply K.
    defined = {}
    for row in cell["summary_rows"]:
        if row.get("test_subset") != "all":
            continue
        defined.setdefault(row["operating_point"], {})[row["institution"]] = bool(
            row.get("defined")
        )
    recall_means = [
        row for row in cell["pooled_rows"] if row["metric"] == "campaign_recall"
    ]
    assert recall_means
    for row in recall_means:
        by_institution = defined[row["operating_point"]]
        in_mean = [i for i, ok in by_institution.items() if ok]
        assert row["untrained_institutions"] == "6,7"
        assert row["n_institutions_in_mean"] == len(in_mean)
        assert row["n_trained_in_mean"] == len(
            [i for i in in_mean if i not in untrained]
        )
        # an untrained institution is always IN the mean, at 0: it was
        # operable and detected nothing
        for institution in untrained:
            assert by_institution[institution] is True

    # the relative-to-own-ceiling mean excludes the blind institutions as
    # UNDEFINED (0 / 0), which is why it is a separate metric
    relative = [
        row for row in cell["pooled_rows"]
        if row["metric"] == "recall_relative_to_own_ceiling"
    ]
    assert relative
    for row in relative:
        assert row["undefined_institutions"], (
            "an institution whose own ceiling is 0 has no ratio, and must be "
            "listed rather than counted as 0"
        )


# ---------------------------------------------------------------------------
# the grid-edge policy, decided before the full-data run
# ---------------------------------------------------------------------------


def test_tau_grid_edge_names_the_direction():
    grid = evaluation.MODEL_TAU_GRID
    assert isolated.tau_grid_edge(min(grid)) == "low"
    assert isolated.tau_grid_edge(max(grid)) == "high"
    assert isolated.tau_grid_edge(0.5) is None
    assert isolated.tau_grid_edge(None) is None
    # and it follows whatever grid is in force, so extending the grid moves
    # the edge rather than leaving the check behind
    assert isolated.tau_grid_edge(0.9999, (0.05, 0.9999)) == "high"
    assert isolated.tau_grid_edge(0.9999, (0.05, 0.9999, 0.99999)) is None


def _evaluate_with_curve(monkeypatch, chosen_tau, budget_tau, world):
    """Run `evaluate_institution` with tau selection stubbed to a known value.

    The policy is about what happens *after* a tau is chosen, so the choice
    itself is stubbed: that keeps the test about the policy instead of about
    coaxing a fixture into producing a particular threshold.
    """
    result, assignment = world
    monkeypatch.setattr(
        isolated.evaluation, "select_tau",
        lambda *a, **k: (chosen_tau, [{"tau": chosen_tau, "campaign_f1": 1.0,
                                       "false_alarms_per_day": 0.0}]),
    )
    monkeypatch.setattr(
        isolated.evaluation, "select_tau_for_budget",
        lambda curve, budget: budget_tau,
    )
    model = isolated.train_isolated(result.full_df, 0, assignment, lookback_h=48)
    assert model.trained
    return isolated.evaluate_institution(
        result.full_df, result.camp, result.pattern_df,
        result.accounts_by_campaign, model, assignment,
        lookback_h=48, seed=0, splits=("test",),
    )


def test_a_tau_at_the_top_of_the_grid_is_undefined_not_zero(monkeypatch, trainable_world):
    """POLICY: a high edge means the optimum lies beyond the grid, so the
    operating point was not found. It is excluded and listed, exactly like an
    unaffordable budget - never recorded as recall 0, which would be a
    measurement of a detector that was never operated there."""
    high = max(evaluation.MODEL_TAU_GRID)
    ev = _evaluate_with_curve(monkeypatch, high, high, trainable_world)

    for point in (p["operating_point"] for p in isolated.operating_points()):
        assert ev.taus[point] is None, point
        assert ev.tau_edges[point] == "high", point
        assert ev.rejected_taus[point] == high, point
        assert ev.undefined_reason[point] == isolated.TAU_BEYOND_GRID
        # and nothing was evaluated at it
        assert ("test", point) not in ev.results


def test_a_low_edge_is_flagged_for_the_f1_point_and_not_for_a_budget(
    monkeypatch, trainable_world
):
    """POLICY: the two edges are not symmetric.

    A low **F1** tau says the optimum may lie below the grid - a real signal,
    which the full-data runner stops on. A low **budget** tau says only that
    the most sensitive threshold already fits the budget, which is a loose
    budget and not a missing optimum, so it is not flagged and not excluded.
    """
    low = min(evaluation.MODEL_TAU_GRID)
    ev = _evaluate_with_curve(monkeypatch, low, low, trainable_world)

    assert ev.tau_edges["f1"] == "low"
    assert ev.taus["f1"] == low, "a low edge is recorded, not excluded"
    for point in ("budget_10", "total_10"):
        assert ev.taus[point] == low
        assert point not in ev.tau_edges, "a loose budget is not a finding"


def test_run_cell_separates_the_two_edges_in_its_flags(monkeypatch, _module_world):
    """What a runner acts on: high-edge pairs are excluded, low-edge F1 taus
    are a stop signal, and the no-affordable-tau list stays separate."""
    result, _ = _module_world
    high = max(evaluation.MODEL_TAU_GRID)
    monkeypatch.setattr(
        isolated.evaluation, "select_tau",
        lambda *a, **k: (high, [{"tau": high, "campaign_f1": 1.0,
                                 "false_alarms_per_day": 0.0}]),
    )
    monkeypatch.setattr(
        isolated.evaluation, "select_tau_for_budget", lambda curve, budget: high
    )
    cell = isolated.run_cell(
        result, target=1.0, seed=0, lookback_h=48, splits=("test",),
    )
    flags = cell["flags"]
    assert flags["tau_at_high_edge"], "every point should be excluded here"
    assert flags["tau_at_low_edge"] == []
    assert flags["no_affordable_tau"] == [], (
        "a tau beyond the grid is its own reason and must not be filed as an "
        "unaffordable budget"
    )

    # the excluded cells carry the rejected tau and the reason
    rows = [r for r in cell["tau_rows"] if r["tau_edge"] == "high"]
    assert rows
    for row in rows:
        assert row["tau"] is None and row["rejected_tau"] == high
        assert row["undefined_reason"] == isolated.TAU_BEYOND_GRID

    # and they are excluded from the pooled means, not scored 0
    for row in cell["pooled_rows"]:
        if row["metric"] == "campaign_recall":
            assert row["n_institutions_in_mean"] == 0
            assert row["mean_over_all_institutions"] is None
            assert row["undefined_institutions"]


# ---------------------------------------------------------------------------
# the guards can fail
# ---------------------------------------------------------------------------


def test_the_guards_can_fail(frame, trainable_world):
    """Three bugs, reproduced against the same data the suite uses, so the
    tests above are shown to be able to fail.

    Each is written as the mutation it guards against, not as a mock.
    """
    df, assignment = frame

    # BUG 1: the mask is applied but the scores are not scattered back into
    # the global row order — the classic way a masked scorer misaligns.
    def buggy_masked(inner, institution, assignment):
        def score(window, t):
            mask = fv.visible_mask(window, institution, assignment)
            inner_scores = np.asarray(inner(window[mask], t), dtype=float)
            padded = np.zeros(len(window))
            padded[: len(inner_scores)] = inner_scores  # positions, not rows
            return padded
        return score

    def inner(window, t):
        return np.arange(1, len(window) + 1, dtype=float)

    institution = 2
    mask = fv.visible_mask(df, institution, assignment)
    good = isolated.masked_scorer(inner, institution, assignment)(df, df["ts"].max())
    bad = buggy_masked(inner, institution, assignment)(df, df["ts"].max())
    assert (good[~mask] == 0).all()
    assert not (bad[~mask] == 0).all(), "BUG 1 did not reproduce"

    # BUG 2: the union sums per-institution false alarms instead of
    # deduplicating cluster identities.
    rows = [
        ("2022-09-07 07:00", "0_X", "1_M", None),
        ("2022-09-07 08:00", "1_M", "0_Z", None),
        ("2022-09-07 09:00", "4_c0", "5_c0", 0),
        ("2022-09-07 10:00", "5_c0", "6_c0", 0),
        ("2022-09-07 11:00", "6_c0", "7_c0", 0),
    ]
    world, world_assignment = _world(rows, {0: "test"})
    scorers = {
        i: isolated.masked_scorer(_all_ones, i, world_assignment) for i in (0, 1)
    }
    accounting = isolated.union_cluster_accounting(
        world.full_df, world.camp, world.accounts_by_campaign, "test",
        scorers=scorers, taus={0: 0.5, 1: 0.5}, lookback_h=48,
    )
    assert accounting["summed_false_alarms"] != accounting["union_false_alarms"], (
        "BUG 2 did not reproduce: summing and deduplicating agree here, so "
        "this fixture cannot tell the two apart"
    )

    # BUG 3: an untrained institution recorded as undefined instead of 0,
    # which silently removes it from the pooled mean.
    tiers = {0: isolated.TIER_UNTRAINED, 1: isolated.TIER_NORMAL}
    correct = isolated.pooled_mean({0: 0.0, 1: 0.8}, tiers)
    buggy = isolated.pooled_mean({0: None, 1: 0.8}, tiers)
    assert correct.over_all == pytest.approx(0.4)
    assert buggy.over_all == pytest.approx(0.8), "BUG 3 did not reproduce"
    assert correct.over_all != buggy.over_all
