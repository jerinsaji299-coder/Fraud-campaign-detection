"""The centralized GNN.

As with the XGBoost tests, these are about the things that would quietly
invalidate a result rather than about accuracy: an edge from the future
entering the graph, identity reaching the inputs, scores that do not line
up with the rows they describe, an ablation that does not ablate, and a
model that cannot be resumed or reloaded as itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudcamp import constants, evaluation, features, windows
from fraudcamp.models import gnn

pytest.importorskip("torch", reason="torch arrives with stage 3.4")
pytest.importorskip("torch_geometric", reason="torch_geometric arrives with stage 3.4")

import torch  # noqa: E402


def _frame(n_per_window: int = 40) -> pd.DataFrame:
    """Traffic in every training and validation window, with a few positives
    and one deliberate hub account so degrees are heavy-tailed."""
    rng = np.random.default_rng(0)
    rows = []
    for day in range(1, 8):
        for hour in (1, 7, 13, 19):
            stamp = pd.Timestamp(f"2022-09-0{day} {hour:02d}:00")
            for i in range(n_per_window):
                laundering = int(i % 20 == 0)
                # every fifth transaction touches the hub, so one account
                # has a degree an order of magnitude above the rest
                src = "9_HUB" if i % 5 == 0 else f"{i % 7}_A{i % 11}"
                rows.append(
                    {
                        "ts": stamp + pd.Timedelta(minutes=i),
                        "src": src,
                        "dst": f"{(i + 3) % 7}_B{i % 13}",
                        "From Bank": 9 if src == "9_HUB" else i % 7,
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
def world():
    frame = _frame()
    times = windows.training_detection_times(24)
    train_windows = list(windows.iter_windows(frame, times, 24))
    spec = features.fit_feature_spec(train_windows, 24)
    node_pre = gnn.fit_node_preprocessor(train_windows)
    return frame, spec, node_pre, train_windows


@pytest.fixture(scope="module")
def trained(world):
    frame, spec, node_pre, _ = world
    model = gnn.train_gnn(
        frame, 24, "gnn_centralized",
        params=gnn.GNNParams(max_epochs=2, patience=2, seed=0),
        spec=spec, node_pre=node_pre, device=torch.device("cpu"),
    )
    return frame, model


# --- causality ------------------------------------------------------------


def test_the_window_graph_contains_no_edge_at_or_after_t(world):
    """The upper bound is strict: a transaction stamped exactly `t` is
    already the future, and a graph that included it would be scoring on
    information the detector cannot have."""
    frame, spec, node_pre, _ = world
    t = pd.Timestamp("2022-09-06 12:00")

    # a transaction exactly at t, and one after it
    extra = frame.iloc[:2].copy()
    extra["ts"] = [t, t + pd.Timedelta(minutes=30)]
    extra["src"] = ["1_FUTURE", "1_FUTURE"]
    poisoned = pd.concat([frame, extra], ignore_index=True)

    clean_window = windows.build_window(frame, t, 24)
    poisoned_window = windows.build_window(poisoned, t, 24)

    clean = gnn.build_window_graph(clean_window, spec, node_pre)
    poisoned_graph = gnn.build_window_graph(poisoned_window, spec, node_pre)

    assert clean.n_edges == poisoned_graph.n_edges
    assert clean.n_accounts == poisoned_graph.n_accounts
    assert poisoned_window.transactions["ts"].max() < t
    assert "1_FUTURE" not in set(poisoned_window.transactions["src"])


def test_the_graph_covers_exactly_the_causal_window(world):
    frame, spec, node_pre, _ = world
    t = pd.Timestamp("2022-09-06 12:00")
    window = windows.build_window(frame, t, 24)
    graph = gnn.build_window_graph(window, spec, node_pre)

    assert graph.n_edges == len(window.transactions)
    # message passing doubles the edges and nothing else
    assert graph.edge_index.shape[1] == 2 * graph.n_edges
    assert graph.target_index.shape[1] == graph.n_edges
    lower = t - pd.Timedelta(hours=24)
    assert window.transactions["ts"].min() >= lower


def test_early_stopping_never_sees_a_test_detection_time():
    """The leakage trap in this sub-stage. Edge-level average precision
    reads every label in a window, so selecting epochs on the validation
    *evaluation horizon* - which runs to Sept 8 - would be selecting on
    test-period edges. Only the validation detection split is allowed."""
    for lookback in (24, 48):
        chosen = gnn.validation_detection_times(lookback)
        assert chosen, "early stopping needs at least one validation window"
        for t in chosen:
            assert windows.split_of_detection_time(t) == "val"
            assert t <= constants.DETECTION_VAL_END
            assert windows.has_full_lookback(t, lookback)

        horizon = windows.evaluation_detection_times("val")
        assert len(horizon) > len(chosen)
        assert any(
            windows.split_of_detection_time(t) == "test" for t in horizon
        ), "the horizon should reach into the test detection split"


# --- identity -------------------------------------------------------------


def test_no_identity_reaches_the_model_inputs(world):
    _, spec, node_pre, _ = world

    for name in node_pre.feature_names:
        lowered = name.lower()
        assert "account" not in lowered
        assert "bank" not in lowered
        assert "campaign" not in lowered
        assert "launder" not in lowered and "label" not in lowered
    assert len(node_pre.feature_names) == 6

    for name in spec.edge_feature_names:
        lowered = name.lower()
        assert "bank" not in lowered or lowered.endswith("cross_bank")
        assert "account" not in lowered
        assert "campaign" not in lowered


def test_the_model_has_no_per_account_embedding(trained):
    """An embedding table indexed by account would learn who an account is,
    which the frozen design forbids. The only input-sized parameter should
    be the projection of the six node features."""
    _, model = trained
    for name, parameter in model.model.named_parameters():
        assert "embedding" not in name.lower()
        # no parameter should be as large as an account vocabulary
        assert parameter.shape[0] < 10_000, (name, tuple(parameter.shape))
    first = model.model.input_projection[0]
    assert first.in_features == 6


# --- node preprocessing (the approved addition) ---------------------------


def test_log1p_is_applied_to_the_heavy_tailed_counts_only(world):
    _, _, node_pre, _ = world
    assert set(node_pre.log1p_columns) == {
        "in_degree", "out_degree", "n_counterparties", "n_currencies"
    }
    # the amounts are already log1p in the shared features; transforming
    # them twice would be a different feature, not a fix
    assert "log_in_amount" not in node_pre.log1p_columns
    assert "log_out_amount" not in node_pre.log1p_columns


def test_log1p_shrinks_a_heavy_tailed_count(world):
    """The mechanism of the approved addition, on a column with a real tail.

    Standardising does not bound a heavy tail: one account with degree
    10,000 among thousands of degree-1 accounts lands hundreds of standard
    deviations out, and three layers of sum aggregation compound whatever
    it is handed. `log1p` first turns that into a few.

    Tested on a constructed column rather than the fixture's windows,
    because standardisation normalises away any monotone transform of a
    column with only two distinct values - which is what the fixture's
    `n_currencies` is, and why comparing window maxima there shows no
    difference. The real before-and-after, on the sample's largest windows,
    is reported in the README.
    """
    # a power-law-ish degree distribution, which is what account degrees
    # actually look like
    rng = np.random.default_rng(0)
    counts = np.floor(1.0 + rng.pareto(1.2, size=20_000) * 3.0)

    def max_abs_z(values):
        standard = values.std() or 1.0
        return float(np.abs((values - values.mean()) / standard).max())

    raw_z = max_abs_z(counts)
    logged_z = max_abs_z(np.log1p(counts))

    assert raw_z > 50, raw_z
    assert logged_z < raw_z / 5, (logged_z, raw_z)

    # It reduces the extreme, it does not bound it: a distribution whose
    # bulk is tightly concentrated still produces large z-scores, because
    # standardising divides by a small standard deviation. That is why
    # LayerNorm is kept as well rather than instead.
    assert logged_z > 3


def test_the_transform_applies_log1p_then_standardises(world):
    """The order matters: the shared spec standardises raw counts, so its
    statistics do not describe a transformed column and the GNN fits its
    own."""
    _, spec, node_pre, train_windows = world
    _, raw, names = gnn.raw_node_features(train_windows[0])
    transformed = node_pre.transform(raw)

    assert transformed.shape == raw.shape
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()

    # recomputing the documented pipeline by hand must give the same thing
    expected = np.asarray(raw, dtype=np.float64).copy()
    for index, name in enumerate(names):
        if name in node_pre.log1p_columns:
            expected[:, index] = np.log1p(expected[:, index])
    expected = (expected - np.asarray(node_pre.mean)) / np.asarray(node_pre.std)
    assert np.allclose(transformed, expected.astype(np.float32), atol=1e-6)

    # and it is not the shared spec's standardisation
    assert not np.allclose(node_pre.mean, spec.node_mean)


def test_the_node_preprocessor_cannot_be_fitted_outside_training(world):
    frame, _, _, _ = world
    test_times = windows.evaluation_detection_times("test")[:2]
    test_windows = list(windows.iter_windows(frame, test_times, 24))
    assert test_windows

    with pytest.raises(ValueError, match="only training windows"):
        gnn.fit_node_preprocessor(test_windows)


def test_the_node_preprocessor_round_trips(world, tmp_path):
    frame, spec, node_pre, train_windows = world
    path = node_pre.save(tmp_path / "node.json")
    loaded = gnn.NodePreprocessor.load(path)

    _, raw, _ = gnn.raw_node_features(train_windows[0])
    assert np.allclose(loaded.transform(raw), node_pre.transform(raw))


# --- the loss -------------------------------------------------------------


def test_the_balanced_loss_starts_at_ln2_whatever_the_window():
    """Window-invariance by construction, which is why this loss was
    adopted: a first step that is not near 0.69 is then a bug."""
    for n_edges, n_positive in ((100, 1), (100_000, 2), (1_000, 500)):
        labels = torch.zeros(n_edges)
        labels[:n_positive] = 1.0
        logits = torch.zeros(n_edges)
        assert float(gnn.balanced_bce(logits, labels)) == pytest.approx(
            np.log(2), abs=1e-6
        )


def test_the_class_weight_ratio_is_n_neg_over_n_pos():
    """The frozen class-weighting principle, stated as a test: a positive
    carries 1/(2 n_pos) and a negative 1/(2 n_neg), so the ratio of
    per-sample weights is exactly n_neg / n_pos."""
    n_edges, n_positive = 1_000, 10
    labels = torch.zeros(n_edges)
    labels[:n_positive] = 1.0
    logits = torch.zeros(n_edges, requires_grad=True)

    gnn.balanced_bce(logits, labels).backward()
    gradients = logits.grad.abs()
    per_positive = float(gradients[:n_positive].mean())
    per_negative = float(gradients[n_positive:].mean())

    n_negative = n_edges - n_positive
    assert per_positive / per_negative == pytest.approx(
        n_negative / n_positive, rel=1e-5
    )


def test_a_window_with_no_positives_still_has_a_loss():
    labels = torch.zeros(50)
    loss = gnn.balanced_bce(torch.zeros(50), labels)
    assert float(loss) == pytest.approx(np.log(2), abs=1e-6)
    assert torch.isfinite(loss)


# --- average precision ----------------------------------------------------


def test_average_precision_matches_hand_computed_values():
    # ranked 1,0,1,0: precision 1.0 at the first hit, 2/3 at the second
    assert gnn.average_precision(
        np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 0, 1, 0])
    ) == pytest.approx((1.0 + 2 / 3) / 2)
    # a perfect ranking is 1.0
    assert gnn.average_precision(
        np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])
    ) == pytest.approx(1.0)
    # the worst ranking: hits last
    assert gnn.average_precision(
        np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])
    ) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_average_precision_is_nan_without_positives():
    assert np.isnan(
        gnn.average_precision(np.array([0.5, 0.4]), np.array([0, 0]))
    )


# --- the scorer -----------------------------------------------------------


def test_scores_line_up_with_the_window_rows(trained):
    """The evaluation pairs scores with `src`/`dst` positionally, so a
    permuted or doubled output would cluster the wrong transactions."""
    frame, model = trained
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    assert len(window) > 0

    scores = model.scorer()(window, t)
    assert scores.shape == (len(window),)  # not 2E, despite the doubled graph
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()

    # anchored to a column whose relationship to the score is known in the
    # fixture, so a reversal cannot cancel out as it would if a reversed
    # input were compared with a reversed output
    is_ach = (window["Payment Format"] == "ACH").to_numpy()
    assert is_ach.any() and (~is_ach).any()
    shuffled = window.iloc[::-1].reset_index(drop=True)
    shuffled_scores = model.scorer()(shuffled, t)
    shuffled_ach = (shuffled["Payment Format"] == "ACH").to_numpy()
    assert np.allclose(
        np.sort(scores[is_ach]), np.sort(shuffled_scores[shuffled_ach]), atol=1e-5
    )


def test_an_empty_window_scores_to_an_empty_array(trained):
    frame, model = trained
    scores = model.scorer()(frame.iloc[0:0], pd.Timestamp("2022-09-07 00:00"))
    assert scores.shape == (0,)


def test_the_model_chooses_no_threshold(trained):
    frame, model = trained
    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    scores = model.scorer()(window, t)
    assert not set(np.unique(scores)) <= {0.0, 1.0}
    assert not hasattr(model, "tau")


def test_it_plugs_into_the_shared_evaluation(trained):
    """The point of matching the XGBoost scorer's interface: the same
    `evaluate_split` scores both, with no model-specific path."""
    frame, model = trained
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
        tau=0.5, lookback_h=24, scorer_name="gnn_centralized",
    )
    assert result.scorer == "gnn_centralized"
    assert len(result.outcomes) == 1


# --- the ablation ---------------------------------------------------------


def test_the_no_format_variant_drops_exactly_the_format_columns(world):
    frame, spec, node_pre, train_windows = world
    window = train_windows[0]

    full = gnn.build_window_graph(window, spec, node_pre, "gnn_centralized")
    ablated = gnn.build_window_graph(
        window, spec, node_pre, "gnn_centralized_no_format"
    )

    format_columns = [n for n in spec.edge_feature_names if n.startswith("fmt_")]
    assert format_columns, "the fixture should produce format columns"
    assert ablated.target_attr.shape[1] == full.target_attr.shape[1] - len(
        format_columns
    )
    # nothing else moves: same rows, same labels, same graph
    assert torch.equal(ablated.labels, full.labels)
    assert torch.equal(ablated.edge_index, full.edge_index)
    assert torch.equal(ablated.x, full.x)

    # the kept columns are the same numbers, not recomputed ones
    from fraudcamp.models.xgb_baseline import keep_mask

    mask = keep_mask(spec.edge_feature_names, ("fmt_",))
    assert torch.equal(ablated.target_attr, full.target_attr[:, mask])
    assert "fmt___other__" in format_columns


def test_the_variants_and_seeds_are_frozen():
    assert list(gnn.GNN_VARIANTS) == ["gnn_centralized", "gnn_centralized_no_format"]
    assert gnn.GNN_VARIANTS["gnn_centralized"] == ()
    assert gnn.GNN_VARIANTS["gnn_centralized_no_format"] == ("fmt_",)
    assert gnn.GNN_SEEDS == (0, 1, 2, 3, 4)
    # model variant names, not the four federated conditions
    assert not set(gnn.GNN_VARIANTS) & set(constants.CONDITIONS)


# --- resumability and reload ---------------------------------------------


def test_a_saved_model_reloads_to_identical_scores(trained, tmp_path):
    frame, model = trained
    directory = tmp_path / "gnn_centralized_L24_s0"
    assert not gnn.is_trained(directory)
    model.save(directory)
    assert gnn.is_trained(directory)

    loaded = gnn.TrainedGNN.load(directory)
    assert loaded.variant == model.variant
    assert loaded.lookback_h == model.lookback_h
    assert loaded.params == model.params

    t = windows.evaluation_detection_times("test")[1]
    window = windows.window_slice(frame, t, 24)
    assert np.allclose(
        loaded.scorer()(window, t), model.scorer()(window, t), atol=1e-6
    )


def test_a_half_written_model_is_not_counted_as_finished(trained, tmp_path):
    """Resumability turns on this: a run interrupted mid-save must be
    retrained, not skipped and then loaded as a broken model."""
    frame, model = trained
    directory = tmp_path / "partial"
    model.save(directory)
    assert gnn.is_trained(directory)

    (directory / "node_preprocessor.json").unlink()
    assert not gnn.is_trained(directory)


def test_training_records_what_it_did(trained):
    _, model = trained
    training = model.training
    assert training["variant"] == "gnn_centralized"
    assert training["n_training_windows"] == len(
        windows.training_detection_times(24)
    )
    assert training["n_validation_windows"] >= 1
    assert training["epochs_run"] >= 1
    assert training["best_epoch"] >= 0
    assert "seeding" in training and "deterministic_algorithms" in training["seeding"]
    assert training["params"]["learning_rate"] == 1e-3
    assert training["params"]["weight_decay"] == 0.0
    assert training["params"]["max_epochs"] == 2  # as passed by the fixture
    for row in training["history"]:
        assert np.isfinite(row["train_loss"])
        assert "max_abs_logit" in row


def test_layer_normalisation_keeps_the_logits_sane(trained):
    """The reason LayerNorm and the log1p were added: the probe's initial
    losses implied logits of order 1e4-1e5."""
    _, model = trained
    for row in model.training["history"]:
        assert row["max_abs_logit"] < 100.0, row
        assert row["train_loss"] < 10.0, row
