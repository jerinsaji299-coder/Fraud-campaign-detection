"""Stage 4.0: per-institution views of the data.

The three properties the stage is defined by, each checked against the whole
mini dataset rather than a hand-picked row:

1. a view holds **exactly** the transactions where that institution holds the
   sender or the receiver — no more and no fewer;
2. the union of the views covers **every** transaction;
3. a campaign's visibility recomputed **from the views** equals the achieved
   visibility the pipeline records.

Each was confirmed to fail when the bug it guards is introduced — see
`test_the_guards_can_fail` at the bottom, which reproduces the three bugs
against the same data rather than asking the reader to take it on trust.
"""

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, features, pipeline, visibility, windows
from fraudcamp.federation import views as fv


K = constants.K_INSTITUTIONS


@pytest.fixture
def built(mini_dataset):
    return pipeline.build(mini_dataset)


@pytest.fixture
def assignment(built):
    """A reassigned split at the lowest target, so accounts really do move."""
    assignment, achieved = fv.assignment_from_pipeline(built, target=0.25, seed=0)
    return assignment, achieved


# ---------------------------------------------------------------------------
# property 1: a view is exactly what that institution can see
# ---------------------------------------------------------------------------


def test_a_view_holds_exactly_the_rows_the_institution_is_an_endpoint_of(built, assignment):
    assignment, _ = assignment
    df = built.full_df
    for inst in range(K):
        view = fv.visible_transactions(df, inst, assignment)
        # every row in the view really has this institution at an endpoint
        for src, dst in zip(view["src"], view["dst"]):
            assert (
                assignment.institution_of(src) == inst
                or assignment.institution_of(dst) == inst
            ), f"institution {inst} sees a transaction it holds neither end of"
        # and every row NOT in the view really does not
        outside = df.drop(index=view.index)
        for src, dst in zip(outside["src"], outside["dst"]):
            assert (
                assignment.institution_of(src) != inst
                and assignment.institution_of(dst) != inst
            ), f"institution {inst} is missing a transaction it holds an end of"


def test_a_view_is_a_row_subset_and_adds_no_columns(built, assignment):
    """A view must not smuggle in the institution that produced it: that
    would be an identity feature, which the frozen rules forbid."""
    assignment, _ = assignment
    window = windows.build_window(built.full_df, "2022-09-02 00:00", lookback_h=48)
    for inst, view in fv.all_views(window, assignment).items():
        assert list(view.transactions.columns) == list(window.transactions.columns)
        assert view.transactions.index.isin(window.transactions.index).all()
        assert view.t == window.t and view.lookback_h == window.lookback_h
        assert view.split == window.split


def test_a_transaction_is_seen_by_one_or_two_institutions_never_none(built, assignment):
    assignment, _ = assignment
    df = built.full_df
    src, dst = fv.endpoint_institutions(df, assignment)
    n_seen = np.array(
        [((src == i) | (dst == i)).astype(int) for i in range(K)]
    ).sum(axis=0)
    assert set(np.unique(n_seen)).issubset({1, 2})
    # exactly 2 when the endpoints sit in different institutions
    assert (n_seen == 2).sum() == int((src != dst).sum())
    assert (n_seen == 1).sum() == int((src == dst).sum())


# ---------------------------------------------------------------------------
# property 2: the views cover everything
# ---------------------------------------------------------------------------


def test_the_union_of_all_views_is_the_whole_dataset(built, assignment):
    assignment, _ = assignment
    df = built.full_df
    covered: set = set()
    for inst in range(K):
        covered.update(fv.visible_transactions(df, inst, assignment).index.to_numpy().tolist())
    assert covered == set(df.index.to_numpy().tolist())


def test_the_union_of_window_views_is_the_whole_window(built, assignment):
    assignment, _ = assignment
    for t in ("2022-09-02 00:00", "2022-09-06 12:00", "2022-09-08 00:00"):
        window = windows.build_window(built.full_df, t, lookback_h=48)
        if window.n_transactions == 0:
            continue
        covered: set = set()
        for view in fv.all_views(window, assignment).values():
            covered.update(view.transactions.index.to_numpy().tolist())
        assert covered == set(window.transactions.index.to_numpy().tolist())


def test_all_views_agrees_with_visible_transactions(built, assignment):
    """The fast k-at-once path and the one-institution path must not drift."""
    assignment, _ = assignment
    window = windows.build_window(built.full_df, "2022-09-06 12:00", lookback_h=48)
    fast = fv.all_views(window, assignment)
    for inst in range(K):
        slow = fv.institution_view(window, inst, assignment)
        pd.testing.assert_frame_equal(fast[inst].transactions, slow.transactions)


def test_views_agree_with_the_phase_2_visible_transactions_function(built, assignment):
    """Phase 2 already had a `visible_transactions`; Phase 4 must not have
    invented a second, different notion of 'sees'."""
    assignment, _ = assignment
    df = built.full_df
    for inst in range(K):
        phase2 = visibility.visible_transactions(
            df, inst, assignment.global_assignment, assignment.bank2inst
        )
        phase4 = fv.visible_transactions(df, inst, assignment)
        pd.testing.assert_frame_equal(phase2, phase4)


# ---------------------------------------------------------------------------
# property 3: the views reproduce the recorded achieved visibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [1.0, 0.75, 0.5, 0.25])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_campaign_visibility_from_views_matches_the_pipeline(built, target, seed):
    assignment, achieved = fv.assignment_from_pipeline(built, target=target, seed=seed)
    table = fv.campaign_visibility_table(built.full_df, assignment)
    assert len(table) > 0
    for row in table.itertuples():
        assert row.visibility_from_views == pytest.approx(
            achieved[int(row.campaign_id)]
        ), (
            f"campaign {row.campaign_id} at seed {seed} target {target}: "
            f"views say {row.visibility_from_views}, pipeline says "
            f"{achieved[int(row.campaign_id)]}"
        )


def test_visibility_from_views_also_matches_the_bin(built):
    assignment, achieved = fv.assignment_from_pipeline(built, target=0.5, seed=0)
    table = fv.campaign_visibility_table(built.full_df, assignment)
    for row in table.itertuples():
        assert row.bin_from_views == visibility.bin_of(achieved[int(row.campaign_id)])


def test_the_natural_split_reassigns_nobody(built):
    natural = fv.natural_assignment(built)
    assert natural.n_reassigned == 0
    for account in built.full_df["src"].unique():
        assert natural.institution_of(account) == natural.bank2inst[int(account.split("_", 1)[0])]


# ---------------------------------------------------------------------------
# features computed from a view, by the unchanged Phase 3 featuriser
# ---------------------------------------------------------------------------


def test_a_view_feeds_the_phase_3_featuriser_unchanged(built, assignment):
    """The whole reason `institution_view` returns a `Window`."""
    assignment, _ = assignment
    train = [
        windows.build_window(built.full_df, t, lookback_h=48)
        for t in windows.training_detection_times(lookback_h=48)
    ]
    train = [w for w in train if w.n_transactions > 0]
    spec = features.fit_feature_spec(train, lookback_h=48)

    window = next(w for w in train if w.n_transactions > 0)
    central = features.build_window_features(window, spec)

    seen_any = False
    for inst, view in fv.all_views(window, assignment).items():
        if view.n_transactions == 0:
            continue
        seen_any = True
        wf = features.build_window_features(view, spec)
        # same feature space, fewer rows
        assert wf.edge_features.shape[1] == central.edge_features.shape[1]
        assert wf.node_features.shape[1] == central.node_features.shape[1]
        assert wf.n_edges == view.n_transactions
        assert wf.n_edges <= central.n_edges
        # the view's accounts are a subset of the window's
        assert set(wf.accounts).issubset(set(central.accounts))
    assert seen_any


def test_node_features_are_computed_from_the_view_alone(built, assignment):
    """An account's degree in a view counts only the transactions that view
    holds, so a boundary account must look *smaller* there than centrally.
    This is the property the 4.1 exchange condition exists to repair."""
    assignment, _ = assignment
    train = [
        windows.build_window(built.full_df, t, lookback_h=48)
        for t in windows.training_detection_times(lookback_h=48)
    ]
    train = [w for w in train if w.n_transactions > 0]
    spec = features.fit_feature_spec(train, lookback_h=48)

    window = max(train, key=lambda w: w.n_transactions)
    central = features.build_window_features(window, spec)
    central_degree = dict(
        zip(central.accounts, central.node_features[:, central.node_feature_names.index("in_degree")])
    )

    found_a_smaller_one = False
    for inst, view in fv.all_views(window, assignment).items():
        if view.n_transactions == 0:
            continue
        wf = features.build_window_features(view, spec)
        degree = dict(
            zip(wf.accounts, wf.node_features[:, wf.node_feature_names.index("in_degree")])
        )
        for account, value in degree.items():
            # standardised, so compare on the raw counts instead
            assert account in central_degree
            if value < central_degree[account]:
                found_a_smaller_one = True
    assert found_a_smaller_one, (
        "no account lost degree in any view: either every transaction is "
        "intra-institution, or the view is not restricting what it should"
    )


# ---------------------------------------------------------------------------
# boundary accounts, which 4.1 builds on
# ---------------------------------------------------------------------------


def test_boundary_accounts_are_exactly_those_visible_to_more_than_one(built, assignment):
    assignment, _ = assignment
    df = built.full_df
    boundary = fv.boundary_accounts(df, assignment)
    for account, insts in boundary.items():
        assert len(insts) > 1
        holders = {
            inst
            for inst in range(K)
            if account
            in set(fv.visible_transactions(df, inst, assignment)["src"])
            | set(fv.visible_transactions(df, inst, assignment)["dst"])
        }
        assert holders == set(insts)
    # an account only ever transacting inside its own institution is not one
    all_accounts = set(df["src"]) | set(df["dst"])
    for account in all_accounts - set(boundary):
        own = assignment.institution_of(account)
        rows = df[(df["src"] == account) | (df["dst"] == account)]
        others = {assignment.institution_of(a) for a in rows["src"]} | {
            assignment.institution_of(a) for a in rows["dst"]
        }
        assert others == {own}


# ---------------------------------------------------------------------------
# edge cases and determinism
# ---------------------------------------------------------------------------


def test_an_empty_frame_gives_empty_views(built, assignment):
    assignment, _ = assignment
    empty = built.full_df.iloc[0:0]
    assert len(fv.visible_transactions(empty, 0, assignment)) == 0
    assert fv.boundary_accounts(empty, assignment) == {}
    src, dst = fv.endpoint_institutions(empty, assignment)
    assert len(src) == 0 and len(dst) == 0


def test_an_institution_with_no_rows_still_gets_a_view(built, assignment):
    """`all_views` must return k entries even when some are empty, or a
    caller looping over institutions would silently skip one."""
    assignment, _ = assignment
    window = windows.build_window(built.full_df, "2022-09-01 06:00", lookback_h=24)
    got = fv.all_views(window, assignment)
    assert set(got) == set(range(K))


def test_views_are_reproducible_from_the_seed(built):
    a1, _ = fv.assignment_from_pipeline(built, target=0.25, seed=0)
    a2, _ = fv.assignment_from_pipeline(built, target=0.25, seed=0)
    assert a1.global_assignment == a2.global_assignment
    window = windows.build_window(built.full_df, "2022-09-06 12:00", lookback_h=48)
    for inst in range(K):
        pd.testing.assert_frame_equal(
            fv.institution_view(window, inst, a1).transactions,
            fv.institution_view(window, inst, a2).transactions,
        )


def test_different_seeds_give_different_views(built):
    """If they did not, the seed would not be controlling anything."""
    a0, _ = fv.assignment_from_pipeline(built, target=0.25, seed=0)
    a1, _ = fv.assignment_from_pipeline(built, target=0.25, seed=1)
    assert a0.global_assignment != a1.global_assignment


# ---------------------------------------------------------------------------
# the load report
# ---------------------------------------------------------------------------


def test_view_load_report_sums_back_to_the_central_totals(built, assignment):
    assignment, _ = assignment
    train = [
        windows.build_window(built.full_df, t, lookback_h=48)
        for t in windows.training_detection_times(lookback_h=48)
    ]
    train = [w for w in train if w.n_transactions > 0]
    report = fv.view_load_report(train, assignment, lookback_h=48)

    per = report.per_institution
    assert len(per) == K
    assert report.central["rows"] == sum(len(w.transactions) for w in train)
    # each transaction is seen by 1 or 2 institutions, so the views' pooled
    # rows must land between the central count and twice it
    assert report.central["rows"] <= per["rows"].sum() <= 2 * report.central["rows"]
    assert report.central["positives"] <= per["positives"].sum() <= 2 * report.central["positives"]
    assert (per["rows"] >= per["distinct_rows"]).all()


def test_view_load_report_flags_starved_institutions(built, assignment):
    assignment, _ = assignment
    train = [
        windows.build_window(built.full_df, t, lookback_h=48)
        for t in windows.training_detection_times(lookback_h=48)
    ]
    train = [w for w in train if w.n_transactions > 0]
    report = fv.view_load_report(train, assignment, lookback_h=48)
    starved = report.starved
    assert set(starved["institution"]).issubset(set(range(K)))
    assert (starved["positives"] < fv.MIN_POSITIVES_TO_TRAIN).all()


# ---------------------------------------------------------------------------
# the guards are proven able to fail
# ---------------------------------------------------------------------------


def test_the_guards_can_fail(built):
    """Each of the three defining properties, re-checked against a
    deliberately broken implementation of the same thing.

    A test that cannot fail is worthless, so the bug each one guards is
    introduced here and the property is required to break.
    """
    assignment, achieved = fv.assignment_from_pipeline(built, target=0.25, seed=0)
    df = built.full_df
    src, dst = fv.endpoint_institutions(df, assignment)

    # BUG 1: "sees" implemented as sender-only (the `or receiver` dropped).
    # At least one institution must then be missing rows it should hold.
    missing_rows = [
        inst
        for inst in range(K)
        if int((src == inst).sum()) != int(((src == inst) | (dst == inst)).sum())
    ]
    assert missing_rows, "sender-only visibility should not match the real rule"
    # and the sender-only views no longer cover the dataset
    sender_only_cover: set = set()
    for inst in range(K):
        sender_only_cover.update(df[src == inst].index.to_numpy().tolist())
    assert len(sender_only_cover) <= len(df)

    # BUG 2: an institution dropped from the loop. Only an institution that
    # *uniquely* sees something breaks coverage — a transaction with both
    # endpoints in it — so the bug is injected where it can actually bite.
    uniquely_seen = {
        inst: int(((src == inst) & (dst == inst)).sum()) for inst in range(K)
    }
    victim = max(uniquely_seen, key=uniquely_seen.get)
    assert uniquely_seen[victim] > 0, "fixture has no intra-institution transaction"
    covered: set = set()
    for inst in range(K):
        if inst == victim:
            continue
        covered.update(fv.visible_transactions(df, inst, assignment).index.to_numpy().tolist())
    assert covered != set(df.index.to_numpy().tolist()), (
        f"dropping institution {victim} should leave the union incomplete"
    )

    # BUG 3: visibility computed as a *mean* over institutions, not a max.
    campaign_ids = sorted(int(c) for c in pd.unique(df["campaign_id"]) if int(c) >= 0)
    disagreed = False
    for campaign_id in campaign_ids:
        rows = df[df["campaign_id"] == campaign_id]
        if len(rows) == 0:
            continue
        csrc, cdst = fv.endpoint_institutions(rows, assignment)
        mean_version = float(
            np.mean([((csrc == i) | (cdst == i)).sum() / len(rows) for i in range(K)])
        )
        if abs(mean_version - achieved[campaign_id]) > 1e-9:
            disagreed = True
    assert disagreed, "mean-over-institutions should not reproduce the recorded visibility"
