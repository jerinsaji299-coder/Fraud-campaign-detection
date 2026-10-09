"""Features, and the leakage properties they must satisfy.

The leakage tests are written as *properties* rather than spot checks: a
feature matrix must be unchanged when the future is deleted, when accounts
and banks are relabelled, and when the labels are flipped. Any one of those
changing means information is reaching the model that must not.
"""

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, features, pipeline, windows


@pytest.fixture(scope="module")
def full_df(tmp_path_factory):
    from conftest import write_mini_dataset

    config = write_mini_dataset(tmp_path_factory.mktemp("features"))
    return pipeline.build(config).full_df


@pytest.fixture(scope="module")
def train_windows(full_df):
    times = windows.detection_times_for_split("train")
    return list(windows.iter_windows(full_df, times, lookback_h=24))


@pytest.fixture(scope="module")
def spec(train_windows):
    return features.fit_feature_spec(train_windows, lookback_h=24)


# --- the spec --------------------------------------------------------------


def test_spec_is_fitted_only_on_training_windows(spec, train_windows):
    assert spec.fitted_on["n_windows"] == len(train_windows)
    # nothing the spec saw may be at or after the training boundary
    assert pd.Timestamp(spec.fitted_on["max_timestamp_seen"]) < pd.Timestamp(
        constants.DETECTION_TRAIN_END
    )


def test_fitting_on_validation_or_test_windows_is_refused(full_df):
    val = list(windows.iter_windows(full_df, windows.detection_times_for_split("val"), 24))
    test = list(windows.iter_windows(full_df, windows.detection_times_for_split("test"), 24))
    offending = val + test
    if not offending:
        pytest.skip("fixture produced no val/test windows")
    with pytest.raises(ValueError, match="only training windows"):
        features.fit_feature_spec(offending)


def test_fitting_requires_windows():
    with pytest.raises(ValueError, match="no training windows"):
        features.fit_feature_spec([])


def test_spec_round_trips_through_json(spec, tmp_path):
    path = spec.save(tmp_path / "spec.json")
    loaded = features.FeatureSpec.load(path)
    assert loaded.edge_feature_names == spec.edge_feature_names
    assert loaded.node_feature_names == spec.node_feature_names
    assert loaded.edge_mean == spec.edge_mean
    assert loaded.node_std == spec.node_std
    assert loaded.payment_currencies == spec.payment_currencies


def test_scaler_only_touches_continuous_columns(spec):
    for i, name in enumerate(spec.edge_feature_names):
        if name not in features.CONTINUOUS_EDGE_FEATURES:
            assert spec.edge_mean[i] == 0.0 and spec.edge_std[i] == 1.0, name
    for i, name in enumerate(spec.node_feature_names):
        if name not in features.CONTINUOUS_NODE_FEATURES:
            assert spec.node_mean[i] == 0.0 and spec.node_std[i] == 1.0, name


def test_scaler_is_unchanged_by_the_existence_of_later_data(full_df, train_windows):
    """Fitting must depend only on the training windows, not on what else
    happens to be in the dataframe."""
    truncated = full_df[full_df["ts"] < pd.Timestamp(constants.DETECTION_TRAIN_END)]
    rebuilt = list(
        windows.iter_windows(truncated, windows.detection_times_for_split("train"), 24)
    )
    a = features.fit_feature_spec(train_windows)
    b = features.fit_feature_spec(rebuilt)
    assert a.edge_mean == b.edge_mean
    assert a.node_mean == b.node_mean
    assert a.payment_currencies == b.payment_currencies


# --- feature content -------------------------------------------------------


def _any_window(full_df, spec, t="2022-09-02 00:00", lookback=24):
    window = windows.build_window(full_df, pd.Timestamp(t), lookback_h=lookback)
    return features.build_window_features(window, spec)


def test_shapes_line_up(full_df, spec):
    wf = _any_window(full_df, spec)
    assert wf.edge_features.shape == (wf.n_edges, len(wf.edge_feature_names))
    assert wf.node_features.shape == (wf.n_nodes, len(wf.node_feature_names))
    assert wf.edge_index.shape == (2, wf.n_edges)
    assert wf.labels.shape == (wf.n_edges,)
    assert wf.edge_index.max() < wf.n_nodes


def test_accounts_are_sorted(full_df, spec):
    wf = _any_window(full_df, spec)
    assert wf.accounts == sorted(wf.accounts)


def test_no_identity_feature_names(spec):
    """No feature may be named after an account, a bank or an institution."""
    banned = ("account", "bank_id", "institution", "src_id", "dst_id", "campaign")
    for name in spec.edge_feature_names + spec.node_feature_names:
        lowered = name.lower()
        assert not any(b in lowered for b in banned), name
    # the only permitted bank-derived feature is the equality flag
    assert "cross_bank" in spec.edge_feature_names


def test_features_are_finite(full_df, spec):
    wf = _any_window(full_df, spec)
    assert np.isfinite(wf.edge_features).all()
    assert np.isfinite(wf.node_features).all()


def test_xgboost_matrix_is_edge_plus_both_endpoints(full_df, spec):
    wf = _any_window(full_df, spec)
    matrix, names = wf.xgboost_matrix()
    assert matrix.shape == (
        wf.n_edges,
        len(wf.edge_feature_names) + 2 * len(wf.node_feature_names),
    )
    assert len(names) == matrix.shape[1]
    # the src block really is the src node's row
    first_src = wf.edge_index[0][0]
    offset = len(wf.edge_feature_names)
    assert np.allclose(
        matrix[0, offset : offset + len(wf.node_feature_names)],
        wf.node_features[first_src],
    )


def test_class_weight_balances_the_classes(full_df, spec):
    wf = _any_window(full_df, spec)
    positives = int(wf.labels.sum())
    if positives == 0:
        assert wf.class_weight() == 1.0
    else:
        assert wf.class_weight() == pytest.approx((wf.n_edges - positives) / positives)


# --- LEAKAGE PROPERTIES ----------------------------------------------------


def test_no_feature_uses_a_transaction_at_or_after_the_detection_time(full_df, spec):
    """The central property. Deleting everything from `t` onward must not
    change a single number in the feature matrices."""
    t = pd.Timestamp("2022-09-02 00:00")

    with_future = _any_window(full_df, spec, t=t)
    past_only = full_df[full_df["ts"] < t]
    without_future = features.build_window_features(
        windows.build_window(past_only, t, lookback_h=24), spec
    )

    assert with_future.accounts == without_future.accounts
    np.testing.assert_array_equal(with_future.edge_features, without_future.edge_features)
    np.testing.assert_array_equal(with_future.node_features, without_future.node_features)
    np.testing.assert_array_equal(with_future.edge_index, without_future.edge_index)


def test_future_rows_cannot_change_features_even_when_extreme(full_df, spec):
    """A sharper version: give the future wildly different values. If any
    aggregate reached forward, the node features would move."""
    t = pd.Timestamp("2022-09-02 00:00")
    baseline = _any_window(full_df, spec, t=t)

    poisoned = full_df.copy()
    future = poisoned["ts"] >= t
    assert future.any(), "fixture must have post-t rows for this test to mean anything"
    poisoned.loc[future, "Amount Paid"] = 1e12
    poisoned.loc[future, "Amount Received"] = 1e12
    poisoned.loc[future, "Payment Currency"] = "Dogecoin"
    poisoned.loc[future, "Is Laundering"] = 1

    after = features.build_window_features(
        windows.build_window(poisoned, t, lookback_h=24), spec
    )
    np.testing.assert_array_equal(baseline.edge_features, after.edge_features)
    np.testing.assert_array_equal(baseline.node_features, after.node_features)


def test_labels_do_not_leak_into_features(full_df, spec):
    """Flipping every label must leave the features identical."""
    t = pd.Timestamp("2022-09-02 00:00")
    baseline = _any_window(full_df, spec, t=t)

    flipped = full_df.copy()
    flipped["Is Laundering"] = 1 - flipped["Is Laundering"]
    after = features.build_window_features(
        windows.build_window(flipped, t, lookback_h=24), spec
    )

    np.testing.assert_array_equal(baseline.edge_features, after.edge_features)
    np.testing.assert_array_equal(baseline.node_features, after.node_features)
    # the labels themselves must of course differ
    assert not np.array_equal(baseline.labels, after.labels)


def test_account_identity_does_not_leak(full_df, spec):
    """Relabelling accounts, preserving structure, must not change any
    feature value. Only the account *names* and their sort order may move,
    so compare multisets of node rows rather than row order."""
    t = pd.Timestamp("2022-09-02 00:00")
    baseline = _any_window(full_df, spec, t=t)

    renamed = full_df.copy()
    mapping = {a: f"zz_{i}" for i, a in enumerate(sorted(set(renamed["src"]) | set(renamed["dst"])))}
    renamed["src"] = renamed["src"].map(mapping)
    renamed["dst"] = renamed["dst"].map(mapping)
    after = features.build_window_features(
        windows.build_window(renamed, t, lookback_h=24), spec
    )

    # edge features are per-row and keep their order
    np.testing.assert_array_equal(baseline.edge_features, after.edge_features)
    # node features: same rows, possibly permuted
    before_rows = sorted(map(tuple, np.round(baseline.node_features, 10)))
    after_rows = sorted(map(tuple, np.round(after.node_features, 10)))
    assert before_rows == after_rows


def test_bank_identity_does_not_leak_beyond_the_equality_flag(full_df, spec):
    """Renumbering banks while preserving which pairs differ must not change
    anything, because only `From Bank != To Bank` is allowed to be used."""
    t = pd.Timestamp("2022-09-02 00:00")
    baseline = _any_window(full_df, spec, t=t)

    shifted = full_df.copy()
    shifted["From Bank"] = shifted["From Bank"] * 10 + 7
    shifted["To Bank"] = shifted["To Bank"] * 10 + 7  # equality structure preserved
    after = features.build_window_features(
        windows.build_window(shifted, t, lookback_h=24), spec
    )
    np.testing.assert_array_equal(baseline.edge_features, after.edge_features)


def test_training_windows_contain_no_transaction_from_sept_5_onward(train_windows):
    boundary = pd.Timestamp(constants.DETECTION_TRAIN_END)
    for window in train_windows:
        assert (window.transactions["ts"] < boundary).all(), (
            f"window at {window.t} reaches into {boundary}"
        )


@pytest.mark.parametrize("lookback", constants.LOOKBACKS_H)
def test_no_lookback_reaches_past_the_detection_time(full_df, spec, lookback):
    t = pd.Timestamp("2022-09-07 00:00")
    window = windows.build_window(full_df, t, lookback_h=lookback)
    if window.n_transactions == 0:
        pytest.skip(f"no transactions in the {lookback}h window at {t}")
    assert (window.transactions["ts"] < t).all()
    assert (window.transactions["ts"] >= t - pd.Timedelta(hours=lookback)).all()


def test_unseen_categories_fall_into_the_other_column(full_df, spec):
    """A currency never seen in training must not shift the known columns."""
    t = pd.Timestamp("2022-09-02 00:00")
    exotic = full_df.copy()
    in_window = (exotic["ts"] >= t - pd.Timedelta(hours=24)) & (exotic["ts"] < t)
    exotic.loc[in_window, "Payment Currency"] = "Quatloo"

    wf = features.build_window_features(
        windows.build_window(exotic, t, lookback_h=24), spec
    )
    other_col = wf.edge_feature_names.index(f"pay_cur_{features.OTHER}")
    assert (wf.edge_features[:, other_col] == 1.0).all()


def test_featurising_an_empty_window_is_refused(full_df, spec):
    empty = full_df[full_df["ts"] < pd.Timestamp("2000-01-01")]
    window = windows.build_window(empty, pd.Timestamp("2022-09-02 00:00"), 24)
    with pytest.raises(ValueError, match="empty"):
        features.build_window_features(window, spec)


def test_features_are_deterministic_in_process(full_df, spec):
    a = _any_window(full_df, spec)
    b = _any_window(full_df, spec)
    np.testing.assert_array_equal(a.edge_features, b.edge_features)
    np.testing.assert_array_equal(a.node_features, b.node_features)
    assert a.accounts == b.accounts
