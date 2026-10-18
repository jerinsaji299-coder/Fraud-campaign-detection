"""The deterministic tau tie-break and the alert-budget metrics.

Both frozen 2026-10-15. The tie-break exists because ties are common and
must not be settled by grid order; the budgets exist because recall alone
can be inflated by alerting more — the random scorer reached 0.27 recall at
L = 48 while raising about 47,000 alarms a day.
"""

import pytest

from fraudcamp import evaluation


def _row(tau, f1, precision, false_alarms, fa_per_day=0.0, recall=None):
    return {
        "tau": tau,
        "lookback_h": 24,
        "campaign_f1": f1,
        "campaign_precision": precision,
        "campaign_recall": recall if recall is not None else f1,
        "false_alarms": false_alarms,
        "false_alarms_per_day": fa_per_day,
        "hits": 1,
    }


# --- the tie-break ---------------------------------------------------------


def test_highest_f1_outranks_everything():
    curve = [_row(0.5, 0.90, 1.0, 0), _row(0.9, 0.95, 0.1, 999)]
    assert evaluation.best_tau(curve) == 0.9


def test_then_higher_precision():
    curve = [_row(0.5, 0.8, 0.6, 10), _row(0.9, 0.8, 0.9, 10)]
    assert evaluation.best_tau(curve) == 0.9


def test_then_fewer_false_alarms():
    """Equal F1 and equal precision, so the absolute alarm count decides."""
    curve = [_row(0.5, 0.8, 0.7, 500), _row(0.9, 0.8, 0.7, 5)]
    assert evaluation.best_tau(curve) == 0.9


def test_finally_the_higher_tau():
    curve = [_row(0.5, 0.8, 0.7, 10), _row(0.9, 0.8, 0.7, 10)]
    assert evaluation.best_tau(curve) == 0.9


def test_precision_outranks_the_tau_preference():
    """A lower tau with better precision beats a higher tau: the tau
    preference is the last resort, not a thumb on the scale."""
    curve = [_row(0.5, 0.8, 0.95, 2), _row(0.99, 0.8, 0.5, 50)]
    assert evaluation.best_tau(curve) == 0.5


def test_result_does_not_depend_on_grid_order():
    """The whole point of fixing the tie-break."""
    rows = [_row(0.5, 0.8, 0.7, 10), _row(0.9, 0.8, 0.7, 10), _row(0.75, 0.8, 0.7, 10)]
    orderings = [
        rows,
        list(reversed(rows)),
        [rows[1], rows[0], rows[2]],
        [rows[2], rows[1], rows[0]],
    ]
    assert {evaluation.best_tau(order) for order in orderings} == {0.9}


def test_single_candidate_is_returned():
    assert evaluation.best_tau([_row(0.42, 0.1, 0.1, 1)]) == 0.42


# --- alert budgets ---------------------------------------------------------


BUDGET_CURVE = [
    _row(0.50, 0.4, 0.20, 9999, fa_per_day=400.0, recall=0.90),
    _row(0.75, 0.5, 0.40, 900, fa_per_day=90.0, recall=0.70),
    _row(0.90, 0.5, 0.60, 300, fa_per_day=30.0, recall=0.55),
    _row(0.99, 0.3, 0.90, 50, fa_per_day=5.0, recall=0.20),
]


@pytest.mark.parametrize(
    "budget,expected", [(100, 0.75), (50, 0.90), (10, 0.99)]
)
def test_budget_takes_the_lowest_affordable_tau(budget, expected):
    """Within a fixed alert budget the most sensitive detector wins."""
    assert evaluation.select_tau_for_budget(BUDGET_CURVE, budget) == expected


def test_unreachable_budget_returns_none_rather_than_approximating():
    curve = [_row(0.5, 0.4, 0.2, 9999, fa_per_day=400.0)]
    assert evaluation.select_tau_for_budget(curve, 10) is None


def test_budget_boundary_is_inclusive():
    curve = [_row(0.9, 0.5, 0.6, 100, fa_per_day=10.0)]
    assert evaluation.select_tau_for_budget(curve, 10) == 0.9


def test_budgets_are_frozen():
    assert evaluation.FALSE_ALARM_BUDGETS == (10, 50, 100)


def test_every_budget_is_satisfiable_on_the_example_curve():
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        assert evaluation.select_tau_for_budget(BUDGET_CURVE, budget) is not None


def test_recall_vs_false_alarms_is_sorted_for_plotting():
    points = evaluation.recall_vs_false_alarms(BUDGET_CURVE)
    assert [p["false_alarms_per_day"] for p in points] == [5.0, 30.0, 90.0, 400.0]
    assert [p["campaign_recall"] for p in points] == [0.20, 0.55, 0.70, 0.90]
    assert all({"tau", "campaign_precision", "campaign_f1"} <= set(p) for p in points)


def test_the_motivation_as_a_test():
    """Recall alone cannot separate these two detectors; the budget view
    can. The profligate one has the higher recall and is unusable."""
    curve = [
        _row(0.50, 0.02, 0.01, 999_999, fa_per_day=47_000.0, recall=0.27),
        _row(0.99, 0.30, 0.90, 20, fa_per_day=2.0, recall=0.20),
    ]
    assert max(curve, key=lambda r: r["campaign_recall"])["tau"] == 0.50
    assert evaluation.select_tau_for_budget(curve, 10) == 0.99
    assert evaluation.select_tau_for_budget(curve, 100) == 0.99


def test_budget_selection_is_independent_of_curve_order():
    shuffled = [BUDGET_CURVE[2], BUDGET_CURVE[0], BUDGET_CURVE[3], BUDGET_CURVE[1]]
    for budget in evaluation.FALSE_ALARM_BUDGETS:
        assert evaluation.select_tau_for_budget(
            shuffled, budget
        ) == evaluation.select_tau_for_budget(BUDGET_CURVE, budget)


# --- the model tau grid (frozen 2026-10-17) --------------------------------


def test_the_model_grid_is_frozen_and_reaches_both_ends():
    """One grid for every model, so a comparison between models or
    conditions is never a comparison of how far each was allowed to search.
    """
    assert evaluation.MODEL_TAU_GRID == (
        0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
        0.995, 0.999, 0.9995, 0.9999,
    )
    # the sanity scorers keep theirs: their scores are 0/1, so neither end
    # would change anything and the low end would only cost clustering time
    assert evaluation.DEFAULT_TAU_GRID == (0.5, 0.75, 0.9, 0.95, 0.99)
    assert min(evaluation.MODEL_TAU_GRID) < min(evaluation.DEFAULT_TAU_GRID)
    assert max(evaluation.MODEL_TAU_GRID) > max(evaluation.DEFAULT_TAU_GRID)


def test_the_near_one_tail_resolves_where_the_linear_grid_could_not():
    """Class-weighted scores pile up against 1. On the full data F1 rose
    from 0.128 at tau 0.95 to 0.543 at 0.99 and the sweep chose 0.99, the
    old top: a jump that large across the last interval means the optimum
    is beyond it, and a linear grid cannot resolve that region.
    """
    grid = evaluation.MODEL_TAU_GRID
    tail = [tau for tau in grid if tau > 0.99]
    assert tail == [0.995, 0.999, 0.9995, 0.9999]

    # each step takes a bite out of what is left above it, which is the
    # point: the resolution is in the remaining distance to 1, not in tau
    remaining = [1 - tau for tau in grid if tau >= 0.99]
    assert remaining == sorted(remaining, reverse=True)
    for finer, coarser in zip(remaining[1:], remaining):
        assert finer <= coarser / 2

    # 0.99 is no longer a boundary, so the full-data choice can be read as a
    # maximum rather than an edge
    assert not evaluation.tau_is_at_grid_edge(0.99, grid)
    assert evaluation.tau_is_at_grid_edge(0.9999, grid)


def test_a_tau_at_either_end_of_the_grid_is_reported_as_an_edge():
    """An edge tau is a boundary, not a maximum, and the full-data run stops
    rather than extending the grid silently."""
    grid = evaluation.MODEL_TAU_GRID
    assert evaluation.tau_is_at_grid_edge(0.05, grid)
    assert evaluation.tau_is_at_grid_edge(0.9999, grid)
    assert not evaluation.tau_is_at_grid_edge(0.3, grid)
    assert not evaluation.tau_is_at_grid_edge(0.99, grid)
    # a single-point grid is both ends at once
    assert evaluation.tau_is_at_grid_edge(0.5, (0.5,))


# --- the score cache -------------------------------------------------------


def test_caching_a_scorer_cannot_change_the_curve():
    """The whole justification for the cache: a scorer's output does not
    depend on tau, so reusing it across the sweep must be invisible in the
    result. Only the cost changes."""
    import numpy as np

    from test_evaluation import _world

    rows = [
        ("2022-09-05 02:00", "1_A", "1_B", 0),
        ("2022-09-05 03:00", "1_B", "1_C", 0),
        ("2022-09-05 04:00", "1_C", "1_D", 0),
        ("2022-09-06 02:00", "1_D", "1_E", 0),
    ]
    full_df, camp, pattern_df, accounts = _world(rows, split="val")

    calls = {"n": 0}

    def scorer(window, t):
        calls["n"] += 1
        # deterministic in the window, as a model is
        return np.linspace(0.1, 0.95, len(window))

    uncached_tau, uncached_curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts, scorer,
        tau_grid=evaluation.MODEL_TAU_GRID, lookback_h=24,
        scorer_name="fake", cache_scores=False,
    )
    uncached_calls = calls["n"]

    calls["n"] = 0
    cached_tau, cached_curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts, scorer,
        tau_grid=evaluation.MODEL_TAU_GRID, lookback_h=24,
        scorer_name="fake", cache_scores=True,
    )
    cached_calls = calls["n"]

    assert cached_tau == uncached_tau
    # identical in everything except the timing column
    strip = lambda curve: [  # noqa: E731
        {k: v for k, v in row.items() if k != "seconds"} for row in curve
    ]
    assert strip(cached_curve) == strip(uncached_curve)
    # and it did less work: one scoring call per window, not per (window, tau)
    assert cached_calls < uncached_calls
    assert cached_calls * len(evaluation.MODEL_TAU_GRID) == uncached_calls


def test_the_cache_refuses_a_window_that_changed_under_it():
    """Returning the wrong scores would be silent and wrong, so a key whose
    row count no longer matches raises instead."""
    import numpy as np
    import pandas as pd

    scorer = evaluation.CachingScorer(
        lambda window, t: np.zeros(len(window)), lookback_h=24
    )
    t = pd.Timestamp("2022-09-06 12:00")
    first = scorer(pd.DataFrame({"a": [1, 2, 3]}), t)
    assert len(first) == 3
    assert len(scorer(pd.DataFrame({"a": [1, 2, 3]}), t)) == 3  # a cache hit

    with pytest.raises(ValueError, match="only valid"):
        scorer(pd.DataFrame({"a": [1, 2]}), t)


def test_every_curve_row_records_what_that_tau_cost():
    """Low thresholds keep most of a window and clustering scales with the
    edges kept, so the cost of searching the low end is on the record."""
    from test_evaluation import _world

    full_df, camp, pattern_df, accounts = _world(
        [
            ("2022-09-05 02:00", "1_A", "1_B", 0),
            ("2022-09-05 03:00", "1_B", "1_C", 0),
            ("2022-09-05 04:00", "1_C", "1_D", 0),
        ],
        split="val",
    )
    _, curve = evaluation.select_tau(
        full_df, camp, pattern_df, accounts,
        lambda window, t: [0.5] * len(window),
        tau_grid=(0.2, 0.5), lookback_h=24, scorer_name="fake",
    )
    assert len(curve) == 2
    for row in curve:
        assert "seconds" in row
        assert row["seconds"] >= 0
