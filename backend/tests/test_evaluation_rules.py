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


def test_stress_horizon_is_not_silently_invented():
    """Rule 1 mentions stress evaluation, but no stress horizon has been
    specified. Guessing one would freeze a research definition by accident,
    so asking for it raises with a pointer to the open question."""
    with pytest.raises(ValueError, match="deliberately unspecified"):
        windows.evaluation_horizon("stress")


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
    L=72 only five remain out of the sixteen on the grid."""
    assert windows.usable_training_times_per_lookback() == {24: 13, 48: 9, 72: 5}
    assert len(windows.detection_times_for_split("train")) == 16


@pytest.mark.parametrize("lookback", constants.LOOKBACKS_H)
def test_every_training_window_has_its_full_lookback(lookback):
    """The property rule 3 exists to guarantee."""
    for t in windows.training_detection_times(lookback):
        assert t - pd.Timedelta(hours=lookback) >= pd.Timestamp(constants.DATA_START), (
            f"training window at {t} with L={lookback} reaches before the data starts"
        )


@pytest.mark.parametrize("lookback", constants.LOOKBACKS_H)
def test_the_unfiltered_grid_would_violate_rule_3(lookback):
    """The complement: without the filter, early training times give short
    windows. This is what rule 3 removes, and it demonstrates the filter is
    doing work rather than being a no-op."""
    unfiltered = windows.detection_times_for_split("train")
    short = [t for t in unfiltered if not windows.has_full_lookback(t, lookback)]
    assert short, f"expected some short windows at L={lookback}"
    assert set(windows.training_detection_times(lookback)) == set(unfiltered) - set(short)


@pytest.mark.parametrize("lookback", constants.LOOKBACKS_H)
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
