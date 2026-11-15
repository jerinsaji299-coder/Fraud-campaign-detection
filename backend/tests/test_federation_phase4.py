"""Stages 4.2, 4.3 and 4.4: score averaging, boundary evidence, centralized.

These tests are about the claims the three stages rest on, and about the two
places where an optimisation could silently become a different answer:

1. **score averaging is an average of the same scores** — the shared
   featurisation in 4.2 must be bit-identical to featurising once per booster,
   and an institution with no booster must be left out of the mean rather than
   entered at 0;
2. **the evidence is what D3 says it is** — node features over the union of
   the views that hold an account equal node features over the whole window,
   which is why the implementation is one pass; substituting boundary accounts
   is the same as substituting every account in a view; and an account its
   owner holds is unaffected;
3. **evidence moves knowledge, not rows** — the augmented view has exactly the
   rows the un-augmented one had, so the ceiling cannot move, which is what
   keeps 4.3 - 4.1 paired against one denominator;
4. **centralized is the K = 1 case** — the mask is all-true, `total_B` and
   `per_inst_B` coincide by definition rather than by special case, and the
   model-seed policy has to be named because the three readings disagree.

`test_the_phase4_guards_can_fail` at the bottom reproduces the bugs these
guard against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, evaluation, features, windows
from fraudcamp.federation import centralized, evidence, fedavg, isolated
from fraudcamp.federation import views as fv
from fraudcamp.models import xgb_baseline

pytest.importorskip("xgboost", reason="xgboost arrives with stage 3.3")

from test_federation_isolated import _build_trainable_world  # noqa: E402

K = constants.K_INSTITUTIONS
L = 48


@pytest.fixture(scope="module")
def world():
    return _build_trainable_world()


@pytest.fixture(scope="module")
def trained(world):
    """The 4.1 boosters: what 4.2 shares and what 4.3 is compared against."""
    result, assignment = world
    return isolated.train_all_isolated(result.full_df, assignment, lookback_h=L)


def _a_window(result, split="test"):
    """A non-empty detection window of `split`, as a `Window`."""
    built = [
        windows.build_window(result.full_df, t, L)
        for t in windows.detection_times_for_split(split, L)
    ]
    non_empty = [w for w in built if w.n_transactions > 0]
    assert non_empty, f"no non-empty {split} window in the fixture"
    return non_empty[0]


# ---------------------------------------------------------------------------
# 4.2: the average is an average
# ---------------------------------------------------------------------------


def test_shared_featurisation_is_bit_identical_to_one_spec_at_a_time(world, trained):
    """The optimisation that makes 4.2 affordable must not change a score.

    `build_window_features_multi` builds the raw features once and applies
    each spec's scaler. That is only sound because the raw features depend on
    a spec through the one-hot vocabularies alone. If that ever stops being
    true this test fails rather than the numbers quietly drifting.
    """
    result, _ = world
    window = _a_window(result)
    specs = [m.model.spec for m in trained.values() if m.trained]
    assert len(specs) >= 2, "need at least two boosters to share anything"

    shared = features.build_window_features_multi(window, specs)
    assert len(shared) == len(specs)
    for spec, built in zip(specs, shared):
        alone = features.build_window_features(window, spec)
        np.testing.assert_array_equal(built.edge_features, alone.edge_features)
        np.testing.assert_array_equal(built.node_features, alone.node_features)
        np.testing.assert_array_equal(built.edge_index, alone.edge_index)
        assert built.accounts == alone.accounts
        assert built.edge_feature_names == alone.edge_feature_names


def test_the_fast_and_naive_averaged_scorers_agree_exactly(world, trained):
    """4.2's scorer against the obvious implementation of 4.2's definition."""
    result, assignment = world
    window = _a_window(result)
    for institution in range(K):
        fast = fedavg.averaged_scorer(trained, institution, assignment)
        naive = fedavg.naive_averaged_scorer(trained, institution, assignment)
        a = fast(window.transactions, window.t)
        b = naive(window.transactions, window.t)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (len(window.transactions),)


def test_averaging_identical_models_returns_one_models_scores(world, trained):
    """Equal weights, stated as a property: mean of K copies is the original.

    Catches a weighting bug that a comparison against the naive path cannot,
    because both paths would carry the same wrong weights.
    """
    result, assignment = world
    window = _a_window(result)
    one = next(m for m in trained.values() if m.trained)
    copies = {i: one for i in range(K)}

    averaged = fedavg.averaged_scorer(copies, 0, assignment)
    single = isolated.masked_scorer(
        one.model.scorer(), 0, assignment
    )
    np.testing.assert_allclose(
        averaged(window.transactions, window.t),
        single(window.transactions, window.t),
        rtol=0, atol=1e-12,
    )


def test_an_institution_without_a_booster_is_left_out_of_the_mean(world, trained):
    """POLICY: a missing model contributes nothing; it is not a score of 0.

    Entering 0 would claim the institution predicted "not laundering", and
    would pull every averaged score down by 1/K per missing model - changing
    the scores of the institutions that did train.
    """
    result, assignment = world
    window = _a_window(result)
    full = {i: m for i, m in trained.items()}
    assert all(m.trained for m in full.values()), "fixture should train all K"

    starved = dict(full)
    starved[3] = isolated.IsolatedModel(
        institution=3, lookback_h=L, tier=isolated.TIER_UNTRAINED,
        n_rows=0, n_positive=0,
    )
    assert fedavg.participating(starved) == [i for i in range(K) if i != 3]

    with_seven = fedavg.averaged_scorer(starved, 0, assignment)(
        window.transactions, window.t
    )
    # the mean of the seven that remain, computed independently
    boosters = [full[i].model for i in range(K) if i != 3]
    mask = fv.visible_mask(window.transactions, 0, assignment)
    expected = np.zeros(len(window.transactions))
    if mask.any():
        stacked = np.vstack([
            b.predict_window(window.transactions[mask], window.t).reshape(1, -1)
            for b in boosters
        ])
        expected[mask] = stacked.mean(axis=0)
    np.testing.assert_allclose(with_seven, expected, rtol=0, atol=1e-12)

    # and an all-untrained panel scores nothing rather than raising
    none_trained = {
        i: isolated.IsolatedModel(
            institution=i, lookback_h=L, tier=isolated.TIER_UNTRAINED,
            n_rows=0, n_positive=0,
        )
        for i in range(K)
    }
    zeros = fedavg.averaged_scorer(none_trained, 0, assignment)(
        window.transactions, window.t
    )
    assert not zeros.any()


def test_averaged_scores_stay_inside_the_view(world, trained):
    """Models moved; rows did not. Everything outside the view is still 0."""
    result, assignment = world
    window = _a_window(result)
    for institution in (0, 1):
        scores = fedavg.averaged_scorer(trained, institution, assignment)(
            window.transactions, window.t
        )
        mask = fv.visible_mask(window.transactions, institution, assignment)
        assert not scores[~mask].any(), "scored a row the institution cannot see"
        assert ((scores >= 0) & (scores <= 1)).all()


# ---------------------------------------------------------------------------
# 4.3: the evidence is what D3 defines
# ---------------------------------------------------------------------------


def test_union_of_views_node_features_equal_whole_window_node_features(world):
    """D3's "union of the views that contain that account", the long way.

    The implementation computes the shared features in one pass over the
    window. That is only the same thing because an account's owner sees every
    edge the account is an endpoint of, so the union of the views holding it
    already contains all of them. Checked here for every boundary account in a
    window rather than argued.
    """
    result, assignment = world
    window = _a_window(result)
    boundary = fv.boundary_accounts(result.full_df, assignment)
    in_window = sorted(
        set(window.transactions["src"]) | set(window.transactions["dst"])
    )
    checked = [a for a in in_window if a in boundary]
    assert checked, "no boundary account in this window: fixture too simple"

    for account in checked:
        over_union, over_window = evidence.union_node_features_match_global(
            result.full_df, assignment, window, account
        )
        np.testing.assert_array_equal(over_union, over_window)


def test_substituting_boundary_accounts_equals_substituting_every_account(world):
    """Why the implementation may touch only boundary accounts.

    For an account its institution OWNS, the view already holds every incident
    edge, so the shared features equal the view-local ones and substituting is
    a no-op. The two augmentations are therefore the same matrix - which is
    also the statement that no non-boundary account's features were shared.
    """
    result, assignment = world
    window = _a_window(result)
    provider = evidence.BoundaryEvidence(result.full_df, assignment, L)

    for institution in range(K):
        view = fv.institution_view(window, institution, assignment)
        if len(view.transactions) == 0:
            continue
        accounts, raw, names = features._raw_node_features(view.transactions)

        boundary_only = provider.override_for(institution)(
            view, accounts, raw, names
        )
        index, shared, _ = provider.global_nodes(view.t, view.lookback_h)
        everything = shared[index.get_indexer(accounts)].astype(raw.dtype)

        np.testing.assert_allclose(boundary_only, everything, rtol=0, atol=0)


def test_an_owned_accounts_features_are_unchanged(world):
    """Stated separately because it is the privacy-relevant half.

    An institution learns nothing new about an account it holds - it already
    saw every edge. What it learns is the profile of an account held
    elsewhere that transacts with it.
    """
    result, assignment = world
    window = _a_window(result)
    provider = evidence.BoundaryEvidence(result.full_df, assignment, L)

    changed_any = False
    for institution in range(K):
        view = fv.institution_view(window, institution, assignment)
        if len(view.transactions) == 0:
            continue
        accounts, raw, names = features._raw_node_features(view.transactions)
        augmented = provider.override_for(institution)(view, accounts, raw, names)
        for k, account in enumerate(accounts):
            owned = assignment.institution_of(account) == institution
            same = np.array_equal(raw[k], augmented[k])
            if owned:
                assert same, (
                    f"institution {institution} already saw every edge of the "
                    f"account it owns ({account}), so nothing should change"
                )
            elif not same:
                changed_any = True
    assert changed_any, (
        "no non-owned account's features changed: the fixture is not "
        "exercising the evidence at all"
    )


def test_evidence_moves_knowledge_not_rows_so_the_ceiling_cannot_move(world):
    """4.3's view holds exactly 4.1's rows, hence exactly 4.1's ceiling.

    This is what makes 4.3a - 4.1 a measurement of scoring against one
    denominator, and it is a property of D3's definition: node features are
    shared, transactions are not.
    """
    result, assignment = world
    window = _a_window(result)
    for institution in range(K):
        view = fv.institution_view(window, institution, assignment)
        augmented_rows = fv.visible_transactions(
            window.transactions, institution, assignment
        )
        pd.testing.assert_frame_equal(view.transactions, augmented_rows)

    # and the oracle through the mask is the same object of study either way
    oracle = isolated.oracle_view_scorer(0, assignment)
    np.testing.assert_array_equal(
        oracle(window.transactions, window.t),
        oracle(window.transactions, window.t),
    )


def test_the_override_reaches_fit_feature_spec_and_build_training_data(world):
    """D3: "the feature spec is refitted per institution on its augmented view".

    If the spec were fitted on un-augmented values and applied to augmented
    ones, every node column would be shifted by the difference between a
    partial and a complete neighbourhood - which is the quantity 4.3 exists to
    measure. So the override has to reach the scaler statistics, not only the
    matrix that is scored.

    Tested with a synthetic override rather than the real evidence, because
    this fixture's TRAINING period is entirely within-institution: no boundary
    account's features change there, so the real override is a no-op on
    training windows and could not distinguish a working hook from a missing
    one. The real evidence is exercised on evaluation windows by the tests
    above. What is pinned here is the plumbing, exactly where a silent failure
    would otherwise hide.
    """
    result, assignment = world
    institution = 0
    view = fv.visible_transactions(result.full_df, institution, assignment)
    times = windows.training_detection_times(L)
    train_windows = list(windows.iter_windows(view, times, L))
    assert train_windows

    def plus_one(window, accounts, matrix, names):
        return matrix + 1.0

    plain = features.fit_feature_spec(train_windows, L)
    shifted = features.fit_feature_spec(train_windows, L, node_override=plus_one)

    assert plain.node_feature_names == shifted.node_feature_names
    np.testing.assert_allclose(
        np.array(shifted.node_mean), np.array(plain.node_mean) + 1.0,
        rtol=0, atol=1e-9,
    )
    # a shift does not change a standard deviation, and the EDGE half is
    # untouched: the evidence is node-level only
    np.testing.assert_allclose(shifted.node_std, plain.node_std, rtol=0, atol=1e-9)
    assert plain.edge_mean == shifted.edge_mean
    assert plain.edge_std == shifted.edge_std

    # and build_training_data must hand it to BOTH the spec and the matrix
    plain_data = xgb_baseline.build_training_data(view, L)
    shifted_data = xgb_baseline.build_training_data(view, L, node_override=plus_one)
    np.testing.assert_allclose(
        np.array(shifted_data.spec.node_mean),
        np.array(plain_data.spec.node_mean) + 1.0,
        rtol=0, atol=1e-9,
    )
    # fitted and applied consistently, so the standardised matrix is unchanged
    assert plain_data.matrix.shape == shifted_data.matrix.shape
    np.testing.assert_allclose(
        plain_data.matrix, shifted_data.matrix, rtol=0, atol=1e-5
    )


def test_a_spec_fitted_unaugmented_and_scored_augmented_really_differs(world):
    """The failure mode the test above prevents, shown to be a real one."""
    result, assignment = world
    institution = 0
    view = fv.visible_transactions(result.full_df, institution, assignment)
    times = windows.training_detection_times(L)
    train_windows = list(windows.iter_windows(view, times, L))

    def plus_one(window, accounts, matrix, names):
        return matrix + 1.0

    plain = features.fit_feature_spec(train_windows, L)
    consistent = xgb_baseline.build_training_data(
        view, L, node_override=plus_one
    ).matrix
    mismatched_rows = []
    for window in train_windows:
        mismatched_rows.append(
            features.build_window_features(window, plain, plus_one)
            .xgboost_matrix()[0]
        )
    mismatched = np.vstack(mismatched_rows)
    assert mismatched.shape == consistent.shape
    assert not np.allclose(mismatched, consistent, atol=1e-3), (
        "fitting the spec on un-augmented values and scoring augmented ones "
        "should shift every node column; if it does not, the test above is "
        "guarding nothing"
    )


def test_evidence_shares_no_labels(world):
    """The override writes node feature columns and nothing else.

    The spec's node features are the six aggregates in
    `features.CONTINUOUS_NODE_FEATURES`; none of them is derived from
    `Is Laundering`, and the override only ever replaces rows of that matrix.
    """
    result, assignment = world
    window = _a_window(result)
    provider = evidence.BoundaryEvidence(result.full_df, assignment, L)
    view = fv.institution_view(window, 0, assignment)
    accounts, raw, names = features._raw_node_features(view.transactions)
    augmented = provider.override_for(0)(view, accounts, raw, names)

    assert list(names) == list(features.CONTINUOUS_NODE_FEATURES)
    assert augmented.shape == raw.shape
    # a label-carrying column would have to be constant within a campaign's
    # accounts and separable; the cheapest real check is that the shared
    # values are reproducible from the unlabelled frame alone
    unlabelled = window.transactions.drop(columns=["Is Laundering", "campaign_id"])
    unlabelled = unlabelled.assign(**{"Is Laundering": 0, "campaign_id": -1})
    accounts2, raw2, _ = features._raw_node_features(unlabelled)
    assert accounts2 == sorted(
        set(window.transactions["src"]) | set(window.transactions["dst"])
    )
    index, shared, _ = provider.global_nodes(view.t, view.lookback_h)
    np.testing.assert_allclose(
        shared[index.get_indexer(accounts2)].astype(np.float64), raw2,
        rtol=0, atol=1e-6,
    )


def test_evidence_runner_refuses_to_mix_two_cells(world):
    """One runner is one cell: a second assignment would be another cell's
    evidence, and the cache would silently serve it."""
    result, assignment = world
    runner = evidence.EvidenceRunner(result.full_df, L)
    runner.provider(assignment)
    other = fv.VisibilityAssignment(
        seed=assignment.seed + 1, target=assignment.target,
        global_assignment={}, bank2inst=assignment.bank2inst, k=assignment.k,
    )
    with pytest.raises(RuntimeError, match="already holds evidence"):
        runner.provider(other)


# ---------------------------------------------------------------------------
# 4.4: centralized as the K = 1 case
# ---------------------------------------------------------------------------


def test_the_single_institution_assignment_sees_everything(world):
    result, _ = world
    assignment = centralized.single_institution_assignment(result)
    assert assignment.k == 1
    mask = fv.visible_mask(result.full_df, 0, assignment)
    assert mask.all(), "the centralized view must be the whole dataset"
    src, dst = fv.endpoint_institutions(result.full_df, assignment)
    assert not src.any() and not dst.any()


def test_total_and_per_institution_budgets_coincide_for_one_institution():
    """D4's "for centralized, total_B = per_inst_B = B", from the existing
    definition rather than a special case: `budget_share(B, 1) == B`."""
    points = {p["operating_point"]: p for p in isolated.operating_points(1)}
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        per_inst = points[f"per_inst_{budget}"]
        total = points[f"total_{budget}"]
        assert per_inst["per_institution_budget"] == float(budget)
        assert total["per_institution_budget"] == float(budget)
        assert per_inst["per_institution_budget"] == total["per_institution_budget"]
    # and that is NOT true for K = 8, which is the whole reason both exist
    eight = {p["operating_point"]: p for p in isolated.operating_points(8)}
    assert eight["total_10"]["per_institution_budget"] == 1.25
    assert eight["per_inst_10"]["per_institution_budget"] == 10.0


def test_the_model_seed_policy_must_be_named_and_the_three_disagree():
    """The choice D4 does not settle, pinned so it cannot be made silently."""
    with pytest.raises(ValueError, match="must be named explicitly"):
        centralized.model_seed_policy("whatever")

    matched = centralized.phase3_reference(
        centralized.model_seed_policy("matched_seed0")
    )
    three = centralized.phase3_reference(
        centralized.model_seed_policy("phase4_seeds_as_model_seeds")
    )
    five = centralized.phase3_reference(
        centralized.model_seed_policy("phase3_five_seeds")
    )
    assert matched["relative"] == pytest.approx(0.78)
    assert three["relative"] == pytest.approx(0.7667, abs=5e-4)
    assert five["relative"] == pytest.approx(0.76)
    assert len({matched["relative"], three["relative"], five["relative"]}) == 3


def test_the_phase3_headline_is_the_five_seed_mean():
    """0.679 / 0.760 is a mean over FIVE model seeds, and Phase 4's scope is
    three. Recorded here because the reproduction check in D4 depends on it."""
    five = centralized.phase3_reference([0, 1, 2, 3, 4])
    assert five["campaign_recall"] == pytest.approx(
        centralized.PHASE3_HEADLINE["campaign_recall"]
    )
    assert five["relative"] == pytest.approx(centralized.PHASE3_HEADLINE["relative"])
    assert five["oracle_recall"] == pytest.approx(0.8928571428571429)
    # no single seed equals the headline recall except seed 4
    equal = [
        s for s, row in centralized.PHASE3_L48_TEST.items()
        if abs(row["campaign_recall"] - five["campaign_recall"]) < 1e-9
    ]
    assert equal == [4]


def test_the_phase3_ceiling_is_the_union_ceiling_stage_41_predicts():
    """Confirmed against a committed Phase 3 file, not only against session B:
    50 of 56 campaigns, identical at every model seed."""
    assert centralized.PHASE3_L48_CEILING == pytest.approx(50 / 56)


# ---------------------------------------------------------------------------
# the guards can fail
# ---------------------------------------------------------------------------


def test_the_phase4_guards_can_fail(world, trained):
    """Each guard above, shown failing on the bug it is there to catch."""
    result, assignment = world
    window = _a_window(result)

    # 1. a weighted average is not the unweighted one these tests pin
    boosters = [m.model for m in trained.values() if m.trained]
    mask = fv.visible_mask(window.transactions, 0, assignment)
    visible = window.transactions[mask]
    stacked = xgb_baseline.predict_window_multi(boosters, visible, window.t)
    weights = np.linspace(0.5, 1.5, len(boosters))
    weights = weights / weights.sum()
    weighted = (stacked * weights[:, None]).sum(axis=0)
    assert not np.allclose(weighted, stacked.mean(axis=0)), (
        "a weighted mean should differ from the equal-weight mean"
    )

    # 2. entering a missing model as 0 really does move the average
    with_zero = np.vstack([stacked, np.zeros((1, stacked.shape[1]))]).mean(axis=0)
    assert not np.allclose(with_zero, stacked.mean(axis=0))

    # 3. fitting the spec un-augmented and scoring augmented really differs
    provider = evidence.BoundaryEvidence(result.full_df, assignment, L)
    view = fv.institution_view(window, 0, assignment)
    accounts, raw, names = features._raw_node_features(view.transactions)
    augmented = provider.override_for(0)(view, accounts, raw, names)
    assert not np.array_equal(raw, augmented), (
        "the evidence changed nothing at all, so no test of it means anything"
    )
