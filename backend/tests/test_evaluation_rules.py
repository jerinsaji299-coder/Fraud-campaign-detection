"""The three Phase 3 evaluation rules added 2026-10-10.

Rule 1  other-split hits are neither hits nor false alarms
Rule 2  validation is evaluated over (Sept 5, Sept 8]; test is unchanged
Rule 3  training detection times must have their full lookback

The clustering and matching that *feed* rule 1 arrive in stage 3.2; what is
testable today is the rule itself, which is a pure function of two splits.
"""

import pandas as pd
import pytest

from fraudcamp import constants, windows


# --- Rule 1: other-split hits ---------------------------------------------


@pytest.mark.parametrize("split", ["val", "test", "stress"])
def test_a_hit_on_the_evaluated_split_is_a_hit(split):
    assert windows.hit_kind(split, split) == "hit"


@pytest.mark.parametrize(
    "evaluated,hit",
    [
        ("test", "train"),
        ("test", "val"),
        ("test", "stress"),
        ("val", "train"),
        ("val", "test"),
        ("val", "stress"),
        ("stress", "test"),
    ],
)
def test_a_hit_on_a_different_split_is_an_other_split_hit(evaluated, hit):
    """Rule 1: correct detection of a campaign we are not currently scoring.
    Not a hit, and emphatically not a false alarm."""
    assert windows.hit_kind(evaluated, hit) == "other_split_hit"


def test_a_hit_on_a_campaign_outside_every_split_is_an_other_split_hit():
    """Campaigns starting on/after the cutoff have split None. A cluster
    finding one has still found something real."""
    assert windows.hit_kind("test", None) == "other_split_hit"


def test_the_outcome_vocabulary_is_fixed():
    assert constants.HIT_KINDS == ["hit", "other_split_hit", "ambiguous", "false_alarm"]
    # the two kinds this rule can produce are both in it
    assert windows.hit_kind("test", "test") in constants.HIT_KINDS
    assert windows.hit_kind("test", "val") in constants.HIT_KINDS


# --- Rule 2: evaluation horizons ------------------------------------------


def test_validation_horizon_runs_to_sept_8():
    start, end = windows.evaluation_horizon("val")
    assert start == pd.Timestamp("2022-09-05 00:00")
    assert end == pd.Timestamp("2022-09-08 00:00")


def test_test_horizon_is_unchanged():
    start, end = windows.evaluation_horizon("test")
    assert start == pd.Timestamp("2022-09-06 00:00")
    assert end == pd.Timestamp(constants.CUTOFF)


def test_validation_horizon_boundaries_are_exclusive_start_inclusive_end():
    times = windows.evaluation_detection_times("val")
    assert pd.Timestamp("2022-09-05 00:00") not in times  # exclusive start
    assert pd.Timestamp("2022-09-05 06:00") == times[0]
    assert pd.Timestamp("2022-09-08 00:00") == times[-1]  # inclusive end
    assert pd.Timestamp("2022-09-08 06:00") not in times


def test_validation_horizon_extends_past_the_validation_data_split():
    """The whole point of rule 2: validation campaigns start on Sept 5 but
    need until Sept 8 to run their course, so the horizon is longer than the
    Sept 5-6 training-data split."""
    times = windows.evaluation_detection_times("val")
    beyond = [t for t in times if t > pd.Timestamp(constants.DETECTION_VAL_END)]
    assert beyond, "the validation horizon must reach past Sept 6"
    assert max(beyond) == pd.Timestamp("2022-09-08 00:00")


def test_validation_and_test_horizons_overlap_by_design():
    val = set(windows.evaluation_detection_times("val"))
    test = set(windows.evaluation_detection_times("test"))
    overlap = val & test
    assert overlap, "the horizons are expected to overlap"
    assert min(overlap) == pd.Timestamp("2022-09-06 06:00")
    assert max(overlap) == pd.Timestamp("2022-09-08 00:00")


def test_no_evaluation_time_exceeds_the_cutoff():
    for split in constants.EVALUATION_HORIZONS:
        times = windows.evaluation_detection_times(split)
        assert max(times) <= pd.Timestamp(constants.CUTOFF)


def test_stress_horizon_runs_from_sept_8_to_the_cutoff():
    """Frozen 2026-10-10. Stress campaigns start Sept 8-10, so the horizon
    opens at Sept 8 and runs to the cutoff, capped per campaign by its own
    deadline."""
    start, end = windows.evaluation_horizon("stress")
    assert start == pd.Timestamp("2022-09-08 00:00")
    assert end == pd.Timestamp(constants.CUTOFF)


def test_stress_horizon_boundaries_are_exclusive_start_inclusive_end():
    times = windows.evaluation_detection_times("stress")
    assert pd.Timestamp("2022-09-08 00:00") not in times  # exclusive start
    assert times[0] == pd.Timestamp("2022-09-08 06:00")
    assert times[-1] == pd.Timestamp(constants.CUTOFF)  # inclusive end
    assert len(times) == 12


def test_stress_horizon_starts_where_the_validation_horizon_ends():
    """The validation horizon closes at Sept 8 00:00 and the stress horizon
    opens there, so the two abut without overlapping."""
    _, val_end = windows.evaluation_horizon("val")
    stress_start, _ = windows.evaluation_horizon("stress")
    assert val_end == stress_start
    val = set(windows.evaluation_detection_times("val"))
    stress = set(windows.evaluation_detection_times("stress"))
    assert not (val & stress)


def test_stress_horizon_is_contained_in_the_test_horizon():
    """Stress runs inside the test window in time, which is why rule 1
    matters here: a cluster at a shared detection time can hit a test
    campaign while stress is being scored, or vice versa."""
    stress = set(windows.evaluation_detection_times("stress"))
    test = set(windows.evaluation_detection_times("test"))
    assert stress < test


def test_every_split_with_campaigns_to_score_has_a_horizon():
    assert set(constants.EVALUATION_HORIZONS) == {"val", "test", "stress"}


def test_training_has_no_evaluation_horizon():
    """Training is scored on validation, so asking for its horizon is a
    mistake rather than a missing definition."""
    with pytest.raises(ValueError, match="no evaluation horizon defined"):
        windows.evaluation_horizon("train")


# --- Rule 3: full-history training windows --------------------------------


def test_full_lookback_boundary_is_inclusive():
    # t - L == DATA_START is exactly enough history
    assert windows.has_full_lookback(pd.Timestamp("2022-09-02 00:00"), 24)
    # six hours short is not
    assert not windows.has_full_lookback(pd.Timestamp("2022-09-01 18:00"), 24)


@pytest.mark.parametrize(
    "lookback,expected_first",
    [(24, "2022-09-02 00:00"), (48, "2022-09-03 00:00"), (72, "2022-09-04 00:00")],
)
def test_training_times_start_one_full_lookback_after_the_data(lookback, expected_first):
    times = windows.training_detection_times(lookback)
    assert times[0] == pd.Timestamp(expected_first)
    assert times[-1] == pd.Timestamp(constants.DETECTION_TRAIN_END)


def test_usable_training_time_counts():
    """Recorded explicitly: a longer lookback costs training windows, and at
    L=72 only five remain out of the sixteen on the grid. That count is one
    of the three reasons L=72 was dropped as a candidate."""
    assert windows.usable_training_times_per_lookback() == {24: 13, 48: 9, 72: 5}
    assert len(windows.detection_times_for_split("train")) == 16


# --- lookback candidates (frozen 2026-10-10) -------------------------------


def test_lookback_candidates_are_24_and_48():
    assert constants.LOOKBACKS_H == (24, 48)
    assert constants.DEFAULT_LOOKBACK_H in constants.LOOKBACKS_H


def test_72_is_excluded_but_still_measured():
    """L=72 is dropped as a candidate yet stays in the measured set, so
    `features-smoke` keeps reporting it and the reason for dropping it stays
    on the record instead of living only in prose."""
    assert constants.EXCLUDED_LOOKBACKS_H == (72,)
    assert 72 not in constants.LOOKBACKS_H
    assert 72 in constants.MEASURED_LOOKBACKS_H


def test_measured_lookbacks_are_candidates_plus_excluded():
    assert set(constants.MEASURED_LOOKBACKS_H) == set(constants.LOOKBACKS_H) | set(
        constants.EXCLUDED_LOOKBACKS_H
    )
    # no lookback may be both a candidate and excluded
    assert not set(constants.LOOKBACKS_H) & set(constants.EXCLUDED_LOOKBACKS_H)


def test_every_candidate_lookback_has_a_workable_number_of_training_windows():
    """The guard behind the decision: a candidate must leave enough
    full-history training detection times to train on. L=72's five is what
    disqualified it."""
    usable = windows.usable_training_times_per_lookback()
    for lookback in constants.LOOKBACKS_H:
        assert usable[lookback] >= 9, (
            f"L={lookback} leaves only {usable[lookback]} training detection times"
        )
    for lookback in constants.EXCLUDED_LOOKBACKS_H:
        assert usable[lookback] < 9


@pytest.mark.parametrize("lookback", constants.MEASURED_LOOKBACKS_H)
def test_every_training_window_has_its_full_lookback(lookback):
    """The property rule 3 exists to guarantee."""
    for t in windows.training_detection_times(lookback):
        assert t - pd.Timedelta(hours=lookback) >= pd.Timestamp(constants.DATA_START), (
            f"training window at {t} with L={lookback} reaches before the data starts"
        )


@pytest.mark.parametrize("lookback", constants.MEASURED_LOOKBACKS_H)
def test_the_unfiltered_grid_would_violate_rule_3(lookback):
    """The complement: without the filter, early training times give short
    windows. This is what rule 3 removes, and it demonstrates the filter is
    doing work rather than being a no-op."""
    unfiltered = windows.detection_times_for_split("train")
    short = [t for t in unfiltered if not windows.has_full_lookback(t, lookback)]
    assert short, f"expected some short windows at L={lookback}"
    assert set(windows.training_detection_times(lookback)) == set(unfiltered) - set(short)


@pytest.mark.parametrize("lookback", constants.MEASURED_LOOKBACKS_H)
def test_evaluation_times_are_unaffected_by_rule_3(lookback):
    """Rule 3 is a training-only filter; every evaluation time is days past
    the start of the data and so always has its full lookback."""
    for split in constants.EVALUATION_HORIZONS:
        for t in windows.evaluation_detection_times(split):
            assert windows.has_full_lookback(t, lookback), (
                f"{split} evaluation time {t} lacks a full {lookback}h lookback"
            )


def test_longer_lookbacks_give_a_subset_of_training_times():
    times = {lookback: set(windows.training_detection_times(lookback)) for lookback in (24, 48, 72)}
    assert times[72] < times[48] < times[24]


def test_training_times_never_reach_into_validation_data():
    """Rule 3 must not weaken the existing guarantee that training sees no
    transaction from Sept 5 onward."""
    for lookback in constants.LOOKBACKS_H:
        for t in windows.training_detection_times(lookback):
            assert t <= pd.Timestamp(constants.DETECTION_TRAIN_END)
