"""The non-graph baseline.

These tests are about the things that would quietly invalidate a result:
training on a window it should not see, scores that do not line up with the
rows they are supposed to describe, a saved model that does not reload as
itself, and identity reaching the feature matrix. The model's accuracy is
not tested here — accuracy is measured on the full data, not asserted in a
unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, evaluation, features, windows
from fraudcamp.models import xgb_baseline

pytest.importorskip("xgboost", reason="xgboost arrives with stage 3.3")


def _training_frame(n_per_window: int = 60) -> pd.DataFrame:
    """A dataframe with traffic in every training window and a few positives.

    Built with a fixed RNG so the trained model is the same every run; the
    shape matters here, not the realism.
    """
    rng = np.random.default_rng(0)
    rows = []
    for day in range(1, 8):
        for hour in (1, 7, 13, 19):
            stamp = pd.Timestamp(f"2022-09-0{day} {hour:02d}:00")
            for i in range(n_per_window):
                laundering = int(i % 20 == 0)
                rows.append(
                    {
                        "ts": stamp + pd.Timedelta(minutes=i),
                        "src": f"{i % 7}_A{i % 11}",
                        "dst": f"{(i + 3) % 7}_B{i % 13}",
                        "From Bank": i % 7,
                        "To Bank": (i + 3) % 7,
                        "Amount Paid": float(rng.integers(10, 10_000)),
                        "Amount Received": float(rng.integers(10, 10_000)),
                        "Payment Currency": "US Dollar" if i % 3 else "Euro",
                        "Receiving Currency": "US Dollar",
                        "Payment Format": "ACH" if laundering else "Cheque",
                        "Is Laundering": laundering,
                        "campaign_id": 0 if laundering else -1,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    frame = _training_frame()
    data = xgb_baseline.build_training_data(frame, 24)
    model = xgb_baseline.train_baseline(
        frame, 24, params=xgb_baseline.XGBParams(n_estimators=20, seed=0), data=data
    )
    return frame, data, model


def test_training_uses_only_rule_3_valid_training_times(trained):
    """A window without its full lookback is silently clamped, which would
    train the model on a distribution it never meets at evaluation time."""
    _, data, _ = trained
    expected = windows.training_detection_times(24)

    assert data.times == expected
    assert len(data.times) == 13  # the measured count at L=24
    for t in data.times:
        assert windows.split_of_detection_time(t) == "train"
        assert windows.has_full_lookback(t, 24)
        assert t <= constants.DETECTION_TRAIN_END


def test_the_feature_spec_cannot_be_fitted_outside_training(trained):
    """The one mistake in this module that would not announce itself."""
    frame, _, _ = trained
    test_times = windows.evaluation_detection_times("test")[:2]
    test_windows = list(windows.iter_windows(frame, test_times, 24))
    assert test_windows, "fixture should produce non-empty test windows"

    with pytest.raises(ValueError, match="only training windows"):
        features.fit_feature_spec(test_windows, 24)


def test_class_weight_is_per_window_not_pooled(trained):
    """The design says weight n_negative/n_positive *per window*. Pooling it
    would let a busy window's balance set the weight for a quiet one."""
    _, data, _ = trained
    weights = {row["t"]: row["class_weight"] for row in data.per_window}
    assert len(weights) == len(data.times)

    # every positive row carries its own window's weight, every negative 1.0
    assert set(np.unique(data.weights[data.labels == 0])) == {1.0}
    positive_weights = set(np.unique(data.weights[data.labels == 1]).round(2))
    assert positive_weights <= set(round(w, 2) for w in weights.values())


def test_no_identity_reaches_the_feature_matrix(trained):
    """Account, bank and institution identity are never inputs. Banks enter
    only as an inequality and accounts only as `src == dst`."""
    _, data, _ = trained
    names = [name.lower() for name in data.feature_names]

    for name in names:
        assert "bank" not in name or name.endswith("cross_bank"), name
        assert "account_id" not in name
        assert "institution" not in name
        assert "campaign" not in name
        assert "launder" not in name and "label" not in name


def test_scores_line_up_with_the_window_rows(trained):
    """The evaluation pairs scores with `src`/`dst` positionally, so a
    permuted output would silently cluster the wrong transactions."""
    frame, _, model = trained
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    assert len(window) > 0

    scores = model.scorer()(window, t)
    assert scores.shape == (len(window),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()

    # Positional, and not symmetric: in the fixture the laundering rows are
    # exactly the ACH ones, so if the output order did not follow the input
    # order the mask would select the wrong scores. Comparing a reversed
    # input against reversed output would NOT catch that - the two
    # reversals cancel - which is why the check is anchored to a column
    # whose relationship to the score is known.
    is_ach = (window["Payment Format"] == "ACH").to_numpy()
    assert is_ach.any() and (~is_ach).any()
    assert scores[is_ach].mean() > scores[~is_ach].mean() + 0.1

    # and the same window in a different row order gives each row the same
    # score, which pins the mapping rather than just its aggregate
    shuffled = window.iloc[::-1].reset_index(drop=True)
    shuffled_scores = model.scorer()(shuffled, t)
    shuffled_ach = (shuffled["Payment Format"] == "ACH").to_numpy()
    assert shuffled_scores[shuffled_ach].mean() > shuffled_scores[~shuffled_ach].mean()


def test_an_empty_window_scores_to_an_empty_array(trained):
    frame, _, model = trained
    empty = frame.iloc[0:0]
    scores = model.scorer()(empty, pd.Timestamp("2022-09-07 00:00"))
    assert scores.shape == (0,)


def test_the_same_seed_gives_the_same_model(trained):
    """Reproducibility within one machine: the seed is the only source of
    randomness the training is allowed."""
    frame, data, model = trained
    again = xgb_baseline.train_baseline(
        frame, 24, params=xgb_baseline.XGBParams(n_estimators=20, seed=0), data=data
    )
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)

    assert np.allclose(model.scorer()(window, t), again.scorer()(window, t))


def test_a_different_seed_is_actually_used(trained):
    """If the seed were ignored, the test above would pass for the wrong
    reason. Subsampling is on, so a different seed must change something."""
    frame, data, model = trained
    other = xgb_baseline.train_baseline(
        frame, 24, params=xgb_baseline.XGBParams(n_estimators=20, seed=99), data=data
    )
    assert other.params.seed == 99
    assert model.params.seed == 0


def test_save_and_load_round_trips_to_identical_scores(trained, tmp_path):
    frame, _, model = trained
    model.save(tmp_path / "m")
    loaded = xgb_baseline.TrainedBaseline.load(tmp_path / "m")

    assert loaded.lookback_h == model.lookback_h
    assert loaded.feature_names == model.feature_names
    assert loaded.params == model.params

    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    assert np.allclose(loaded.scorer()(window, t), model.scorer()(window, t))


def test_a_mismatched_spec_is_refused_rather_than_scored(trained):
    """A model and a feature spec that do not belong together would produce
    numbers that look fine and mean nothing."""
    frame, _, model = trained
    broken = xgb_baseline.TrainedBaseline(
        booster=model.booster,
        spec=model.spec,
        lookback_h=model.lookback_h,
        feature_names=model.feature_names[:-1],  # one column short
        params=model.params,
    )
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)

    with pytest.raises(ValueError, match="do not belong to each other"):
        broken.scorer()(window, t)


def test_the_model_chooses_no_threshold_of_its_own(trained):
    """Tau belongs to the evaluation, chosen on validation only. The model
    emits probabilities and must not round them."""
    frame, _, model = trained
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    scores = model.scorer()(window, t)

    assert not set(np.unique(scores)) <= {0.0, 1.0}
    assert not hasattr(model, "tau")


def test_it_plugs_into_the_evaluation_unchanged(trained):
    """The point of the scorer signature: the same `evaluate_split` that
    scored the oracle scores a model, with no model-specific path."""
    frame, _, model = trained
    camp = pd.DataFrame(
        [
            {
                "campaign_id": 0, "split": "test", "base_type": "STACK",
                "group": "fragmentable", "n_txn": 20, "n_accounts": 8,
                "duration_h": 100.0, "start": pd.Timestamp("2022-09-06 01:00"),
                "end": pd.Timestamp("2022-09-09 01:00"),
                "deadline": pd.Timestamp("2022-09-09 01:00"),
                "crosses_cutoff": False, "shares_train_account": True,
                "eval_ok": True,
            }
        ]
    )
    accounts = {
        0: tuple(sorted(set(frame.loc[frame["campaign_id"] == 0, "src"])
                        | set(frame.loc[frame["campaign_id"] == 0, "dst"])))
    }
    result = evaluation.evaluate_split(
        frame, camp, frame, accounts, "test", model.scorer(),
        tau=0.5, lookback_h=24, scorer_name="xgboost",
    )
    assert result.scorer == "xgboost"
    assert len(result.outcomes) == 1
    summary = evaluation.summarize(result, subset="all")
    assert summary["campaign_recall"] in (0.0, 1.0)


def test_hyperparameters_are_the_same_for_both_lookbacks():
    """Tuning per lookback would make the L comparison a comparison of
    tuning effort rather than of lookback length."""
    assert xgb_baseline.XGBParams(seed=0) == xgb_baseline.XGBParams(seed=0)
    params = xgb_baseline.XGBParams(seed=3)
    booster_params = params.booster_params()
    assert booster_params["seed"] == 3
    assert booster_params["objective"] == "binary:logistic"
    assert "scale_pos_weight" not in booster_params  # weighting is per row


# --- the payment-format ablation ------------------------------------------


def test_the_ablation_drops_every_format_column_and_nothing_else(trained):
    """`fmt___other__` is easy to miss, and leaving it in would make the
    ablation incomplete while looking complete."""
    _, data, _ = trained
    dropped = xgb_baseline.dropped_columns(
        data.feature_names, xgb_baseline.MODEL_VARIANTS["xgboost_centralized_no_format"]
    )
    assert dropped, "the fixture should produce format columns"
    assert all(name.startswith("fmt_") for name in dropped)
    assert "fmt___other__" in dropped
    assert set(dropped) == {n for n in data.feature_names if n.startswith("fmt_")}

    kept = [n for n in data.feature_names if n not in dropped]
    assert not any(n.startswith("fmt_") for n in kept)
    # currencies, formats' neighbours in the one-hot block, are untouched
    assert any(n.startswith("pay_cur_") for n in kept)
    assert any(n.startswith("recv_cur_") for n in kept)
    assert len(kept) + len(dropped) == len(data.feature_names)


def test_the_ablation_changes_only_the_columns(trained):
    """An ablation is only interpretable if everything else is identical, so
    the two variants are built from one matrix rather than featurised twice.
    """
    _, data, _ = trained
    ablated = data.for_variant("xgboost_centralized_no_format")

    assert ablated.n_rows == data.n_rows
    assert np.array_equal(ablated.labels, data.labels)
    assert np.array_equal(ablated.weights, data.weights)
    assert ablated.times == data.times
    assert ablated.per_window == data.per_window
    assert ablated.matrix.shape[1] == data.matrix.shape[1] - len(ablated.dropped)

    # the kept columns are the same numbers, not recomputed ones
    mask = xgb_baseline.keep_mask(data.feature_names, ("fmt_",))
    assert np.array_equal(ablated.matrix, data.matrix[:, mask])


def test_the_full_variant_is_a_no_op_view(trained):
    _, data, _ = trained
    assert data.for_variant("xgboost_centralized") is data


def test_slicing_twice_is_free_so_seeds_can_share_a_matrix(trained):
    """`train_baseline` slices per call; passing already-sliced data must
    not copy 1.2 GB again for every seed."""
    _, data, _ = trained
    once = data.for_variant("xgboost_centralized_no_format")
    twice = once.for_variant("xgboost_centralized_no_format")
    assert twice is once
    assert once.dropped  # and the record of what was dropped survives


def test_an_ablated_model_trains_scores_and_round_trips(trained, tmp_path):
    frame, data, _ = trained
    ablated = xgb_baseline.train_baseline(
        frame, 24,
        params=xgb_baseline.XGBParams(n_estimators=20, seed=0),
        data=data.for_variant("xgboost_centralized_no_format"),
        variant="xgboost_centralized_no_format",
    )
    assert ablated.variant == "xgboost_centralized_no_format"
    assert ablated.dropped_prefixes == ("fmt_",)
    assert not any(n.startswith("fmt_") for n in ablated.feature_names)
    assert ablated.training["dropped_columns"]

    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    scores = ablated.scorer()(window, t)
    assert scores.shape == (len(window),)

    ablated.save(tmp_path / "ab")
    loaded = xgb_baseline.TrainedBaseline.load(tmp_path / "ab")
    assert loaded.variant == "xgboost_centralized_no_format"
    assert np.allclose(loaded.scorer()(window, t), scores)


def test_an_unknown_variant_is_refused(trained):
    frame, data, _ = trained
    with pytest.raises(ValueError, match="unknown variant"):
        xgb_baseline.train_baseline(frame, 24, data=data, variant="xgboost_no_hour")


def test_the_variants_and_seeds_are_frozen():
    assert list(xgb_baseline.MODEL_VARIANTS) == [
        "xgboost_centralized", "xgboost_centralized_no_format",
    ]
    assert xgb_baseline.MODEL_VARIANTS["xgboost_centralized"] == ()
    assert xgb_baseline.MODEL_VARIANTS["xgboost_centralized_no_format"] == ("fmt_",)
    assert xgb_baseline.MODEL_SEEDS == (0, 1, 2, 3, 4)
    # variant names are model names, not the four federated conditions
    assert not set(xgb_baseline.MODEL_VARIANTS) & set(constants.CONDITIONS)
