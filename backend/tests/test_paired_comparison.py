"""The paired lead-time comparison, which is how two detectors' earliness is
compared anywhere in this project.

The point of these tests is the first one: on data where the unpaired medians
say the better detector got worse, the paired comparison says the truth. The
rest pin the sign convention, the coverage counts, and the seeding, so a
reported interval cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudcamp.evaluation import (
    CampaignOutcome,
    bootstrap_ci,
    describe_paired_comparison,
    paired_lead_time_comparison,
)

DEADLINE = pd.Timestamp("2022-09-07 00:00")


def outcome(
    campaign_id: int,
    lead_time_h: float | None,
    *,
    detected: bool = True,
    normalized: float | None = None,
    frac_observed: float | None = None,
    group: str = "fragmentable",
) -> CampaignOutcome:
    """A CampaignOutcome carrying only the fields the comparison reads."""
    return CampaignOutcome(
        campaign_id=campaign_id,
        split="test",
        base_type="SCATTER-GATHER",
        group=group,
        detected=detected,
        detection_time=DEADLINE if detected else None,
        lead_time_h=lead_time_h,
        normalized_lead_time=normalized,
        frac_observed=frac_observed,
        crosses_cutoff=False,
        shares_train_account=False,
        deadline=DEADLINE,
        end=DEADLINE,
        duration_h=120.0,
        n_txn=12,
    )


def test_unpaired_medians_mislead_where_the_paired_difference_does_not():
    """The case the paired method exists for.

    B detects everything A detects, at exactly the same time, plus one extra
    campaign that is only detectable late. Comparing medians makes B look
    worse by 45 hours. B did not get later on any campaign: it got one more.
    """
    a = [outcome(1, 100.0), outcome(2, None, detected=False)]
    b = [outcome(1, 100.0), outcome(2, 10.0)]

    a_detected = [o.lead_time_h for o in a if o.detected]
    b_detected = [o.lead_time_h for o in b if o.detected]
    assert np.median(a_detected) == 100.0
    assert np.median(b_detected) == 55.0  # the misleading comparison

    paired = paired_lead_time_comparison(a, b, "L=24h", "L=48h")
    assert paired.n_both == 1
    assert paired.median_difference_h == 0.0  # nothing got later
    assert (paired.n_only_b, paired.n_only_a) == (1, 0)
    assert paired.only_b_campaigns == [2]


def test_positive_difference_means_b_detected_earlier():
    a = [outcome(1, 10.0), outcome(2, 20.0)]
    b = [outcome(1, 16.0), outcome(2, 32.0)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.median_difference_h == pytest.approx(9.0)
    assert (paired.n_b_earlier, paired.n_a_earlier, paired.n_tied) == (2, 0, 0)

    flipped = paired_lead_time_comparison(b, a, "b", "a")
    assert flipped.median_difference_h == pytest.approx(-9.0)
    assert (flipped.n_b_earlier, flipped.n_a_earlier, flipped.n_tied) == (0, 2, 0)


def test_coverage_is_counted_separately_from_earliness():
    a = [outcome(1, 50.0), outcome(2, 30.0), outcome(3, None, detected=False)]
    b = [outcome(1, 50.0), outcome(2, None, detected=False), outcome(3, 40.0)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.n_both == 1
    assert (paired.n_only_a, paired.only_a_campaigns) == (1, [2])
    assert (paired.n_only_b, paired.only_b_campaigns) == (1, [3])
    assert paired.n_neither == 0


def test_campaigns_neither_detects_are_counted_and_excluded():
    a = [outcome(1, 50.0), outcome(2, None, detected=False)]
    b = [outcome(1, 60.0), outcome(2, None, detected=False)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.n_neither == 1
    assert paired.n_both == 1
    assert paired.median_difference_h == pytest.approx(10.0)


def test_campaigns_absent_from_one_side_are_ignored():
    """Only campaigns present in both evaluations can be paired."""
    a = [outcome(1, 50.0), outcome(2, 50.0)]
    b = [outcome(1, 70.0)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.n_both == 1
    assert paired.n_only_a == 0  # campaign 2 was not evaluated under b at all
    assert paired.median_difference_h == pytest.approx(20.0)


def test_no_shared_detections_yields_no_difference():
    a = [outcome(1, 50.0), outcome(2, None, detected=False)]
    b = [outcome(1, None, detected=False), outcome(2, 50.0)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.n_both == 0
    assert paired.median_difference_h is None
    assert paired.difference_ci_h == (None, None)
    assert "cannot be compared" in describe_paired_comparison(paired)


def test_normalized_and_frac_observed_are_paired_too():
    a = [
        outcome(1, 50.0, normalized=0.4, frac_observed=0.6),
        outcome(2, 20.0, normalized=0.2, frac_observed=0.8),
    ]
    b = [
        outcome(1, 60.0, normalized=0.5, frac_observed=0.5),
        outcome(2, 30.0, normalized=0.3, frac_observed=0.7),
    ]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.median_normalized_difference == pytest.approx(0.1)
    assert paired.median_frac_observed_difference == pytest.approx(-0.1)


def test_missing_normalized_values_do_not_sink_the_hours_comparison():
    a = [outcome(1, 50.0, normalized=None), outcome(2, 20.0, normalized=0.2)]
    b = [outcome(1, 55.0, normalized=0.5), outcome(2, 25.0, normalized=None)]

    paired = paired_lead_time_comparison(a, b, "a", "b")
    assert paired.median_difference_h == pytest.approx(5.0)
    assert paired.median_normalized_difference is None


def test_identical_detectors_give_a_zero_interval():
    outcomes = [outcome(i, 10.0 * i) for i in range(1, 8)]
    paired = paired_lead_time_comparison(outcomes, list(outcomes), "a", "b")

    assert paired.median_difference_h == 0.0
    assert paired.difference_ci_h == (0.0, 0.0)
    assert paired.n_tied == 7


def test_interval_is_seeded_and_reproducible():
    """Same seed, same interval — and the seed really reaches the bootstrap.

    Two different seeds are *not* asserted to disagree: the statistic is a
    median over a handful of distinct differences, so the percentiles of 1000
    resamples land on the same grid points either way. What matters is that
    the interval is a function of the seed and nothing else, so the number
    written into the README can be recomputed.
    """
    a = [outcome(i, float(i)) for i in range(1, 25)]
    b = [outcome(i, float(i) + (i * 7 % 11) - 5.0) for i in range(1, 25)]

    first = paired_lead_time_comparison(a, b, "a", "b", bootstrap_seed=7)
    again = paired_lead_time_comparison(a, b, "a", "b", bootstrap_seed=7)
    assert first.difference_ci_h == again.difference_ci_h

    differences = [y.lead_time_h - x.lead_time_h for x, y in zip(a, b)]
    assert first.difference_ci_h == bootstrap_ci(differences, seed=7)
    assert first.difference_ci_h != bootstrap_ci(differences, seed=7, n_resamples=3)

    low, high = first.difference_ci_h
    assert low <= first.median_difference_h <= high


def test_description_refuses_to_claim_a_difference_when_zero_is_inside():
    """A noisy wash must read as a wash, not as a direction."""
    a = [outcome(i, float(i * 7 % 13)) for i in range(1, 21)]
    b = [outcome(i, float(i * 5 % 13)) for i in range(1, 21)]

    paired = paired_lead_time_comparison(a, b, "L=24h", "L=48h", bootstrap_seed=0)
    low, high = paired.difference_ci_h
    assert low <= 0 <= high
    assert "no earliness difference is demonstrated" in describe_paired_comparison(paired)


def test_description_states_a_direction_when_the_interval_excludes_zero():
    a = [outcome(i, 10.0) for i in range(1, 21)]
    b = [outcome(i, 40.0) for i in range(1, 21)]

    paired = paired_lead_time_comparison(a, b, "L=24h", "L=48h", bootstrap_seed=0)
    low, high = paired.difference_ci_h
    assert low > 0
    text = describe_paired_comparison(paired)
    assert "L=48h is earlier" in text
    assert "20 campaigns both detect" in text


def test_as_dict_is_flat_enough_for_a_csv_row():
    a = [outcome(1, 10.0), outcome(2, 20.0), outcome(3, None, detected=False)]
    b = [outcome(1, 20.0), outcome(2, 30.0), outcome(3, 5.0)]

    row = paired_lead_time_comparison(
        a, b, "L=24h", "L=48h", split="test"
    ).as_dict()

    assert row["label_a"] == "L=24h" and row["label_b"] == "L=48h"
    assert row["split"] == "test"
    assert row["only_b_campaigns"] == "3"
    assert row["only_a_campaigns"] == ""
    assert "ci_low_h" in row and "ci_high_h" in row
    assert "difference_ci_h" not in row
    assert all(not isinstance(v, (list, tuple, dict)) for v in row.values())


def test_against_real_oracle_outcomes_for_two_lookbacks():
    """End to end on a built world, since that is how it will be called.

    Campaign 0 is tight enough for either lookback, and runs long enough
    that a detection time falls before its deadline. Campaign 1 has 30-hour
    gaps, so no 24h window ever holds three of its accounts at once, while a
    48h window does. The paired difference on campaign 0 must be zero and
    campaign 1 must show up as coverage, not as a lead-time change.
    """
    from fraudcamp import evaluation
    from fraudcamp.evaluation import scorers
    from test_evaluation import _world

    rows = [
        ("2022-09-06 01:00", "1_A", "1_B", 0),
        ("2022-09-06 02:00", "1_B", "1_C", 0),
        ("2022-09-06 03:00", "1_C", "1_D", 0),
        ("2022-09-09 10:00", "1_D", "1_E", 0),  # keeps the deadline later
        ("2022-09-06 02:00", "2_A", "2_B", 1),
        ("2022-09-07 08:00", "2_B", "2_C", 1),  # 30h gaps: 24h never spans two
        ("2022-09-08 14:00", "2_C", "2_D", 1),
    ]
    full_df, camp, pattern_df, accounts = _world(rows)

    def oracle(window, t):
        return scorers.oracle_scores(window["Is Laundering"].to_numpy())

    evaluated = {
        lookback: evaluation.evaluate_split(
            full_df, camp, pattern_df, accounts, "test", oracle,
            tau=scorers.SANITY_TAU["oracle"], lookback_h=lookback,
        )
        for lookback in (24, 48)
    }

    detected = {
        lookback: {o.campaign_id for o in result.outcomes if o.detected}
        for lookback, result in evaluated.items()
    }
    assert detected[24] == {0}
    assert detected[48] == {0, 1}

    paired = paired_lead_time_comparison(
        evaluated[24].outcomes, evaluated[48].outcomes, "L=24h", "L=48h", split="test"
    )
    assert paired.n_both == 1
    assert paired.median_difference_h == 0.0
    assert paired.only_b_campaigns == [1]
    assert paired.n_only_a == 0
