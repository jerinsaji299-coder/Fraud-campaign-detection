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
