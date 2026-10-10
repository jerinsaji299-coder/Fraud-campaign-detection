"""Detection times and causal window slicing."""

import pandas as pd
import pytest

from fraudcamp import constants, pipeline, windows


def test_grid_is_aligned_to_six_hourly_boundaries():
    times = windows.detection_times()
    assert all(t.hour in (0, 6, 12, 18) for t in times)
    assert all(t.minute == 0 and t.second == 0 for t in times)


def test_grid_spans_the_study_period_and_excludes_the_empty_first_window():
    times = windows.detection_times()
    assert times[0] == pd.Timestamp("2022-09-01 06:00")
    assert times[-1] == pd.Timestamp(constants.CUTOFF)
    # Sept 1 00:00 would have nothing behind it
    assert pd.Timestamp(constants.DATA_START) not in times
    # 10 days of 6-hourly steps
    assert len(times) == 40


def test_grid_steps_are_exactly_six_hours():
    times = windows.detection_times()
    gaps = {(b - a) for a, b in zip(times, times[1:])}
    assert gaps == {pd.Timedelta(hours=6)}


@pytest.mark.parametrize(
    "t,expected",
    [
        ("2022-09-01 06:00", "train"),
        ("2022-09-04 18:00", "train"),
        ("2022-09-05 00:00", "train"),  # inclusive upper bound
        ("2022-09-05 06:00", "val"),
        ("2022-09-06 00:00", "val"),  # inclusive upper bound
        ("2022-09-06 06:00", "test"),
        ("2022-09-11 00:00", "test"),  # the cutoff is the last test time
        ("2022-09-11 06:00", None),  # past the study period
    ],
)
def test_detection_time_splits(t, expected):
    assert windows.split_of_detection_time(pd.Timestamp(t)) == expected


def test_splits_partition_the_grid():
    times = set(windows.detection_times())
    per_split = {s: set(windows.detection_times_for_split(s)) for s in constants.DETECTION_SPLIT_ORDER}
    union = set().union(*per_split.values())
    assert union == times
    # and no overlap
    assert sum(len(v) for v in per_split.values()) == len(times)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split must be one of"):
        windows.detection_times_for_split("stress")


def test_window_excludes_its_own_detection_time(mini_dataset):
    """The strict upper bound: a transaction stamped exactly t is future."""
    df = pipeline.build(mini_dataset).full_df
    t = pd.Timestamp("2022-09-02 00:00")
    # the fixture has a transaction at exactly 2022-09-02 01:00; shift one to t
    df = df.copy()
    df.loc[df.index[0], "ts"] = t

    window = windows.window_slice(df, t, lookback_h=24)
    assert (window["ts"] < t).all()
    assert t not in set(window["ts"])


def test_window_respects_the_lookback(mini_dataset):
    df = pipeline.build(mini_dataset).full_df
    t = pd.Timestamp("2022-09-03 00:00")
    for lookback in constants.MEASURED_LOOKBACKS_H:
        window = windows.window_slice(df, t, lookback_h=lookback)
        lower = t - pd.Timedelta(hours=lookback)
        assert (window["ts"] >= lower).all()
        assert (window["ts"] < t).all()


def test_longer_lookbacks_are_supersets(mini_dataset):
    df = pipeline.build(mini_dataset).full_df
    t = pd.Timestamp("2022-09-07 00:00")
    seen = [set(windows.window_slice(df, t, lookback_h=lb).index) for lb in (24, 48, 72)]
    assert seen[0] <= seen[1] <= seen[2]


def test_build_window_reports_its_split_and_contents(mini_dataset):
    df = pipeline.build(mini_dataset).full_df
    window = windows.build_window(df, pd.Timestamp("2022-09-02 00:00"), lookback_h=24)
    assert window.split == "train"
    assert window.start == pd.Timestamp("2022-09-01 00:00")
    assert window.n_transactions == len(window.transactions)
    assert window.n_laundering <= window.n_transactions


def test_iter_windows_skips_empty_ones(mini_dataset):
    df = pipeline.build(mini_dataset).full_df
    times = windows.detection_times()
    built = list(windows.iter_windows(df, times, lookback_h=24))
    assert built, "the fixture should produce at least one non-empty window"
    assert all(w.n_transactions > 0 for w in built)
    assert len(built) < len(times)  # the sparse fixture leaves gaps


def test_scan_agrees_with_the_real_window_slicer(mini_dataset):
    """`features-smoke`'s cheap scan locates windows with binary search
    instead of the real slicer, so it must agree with it exactly — otherwise
    it would misreport the sizes the GPU estimate is built from."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "kaggle"))
    from kaggle_runner import scan_window_sizes

    df = pipeline.build(mini_dataset).full_df
    rows = scan_window_sizes(df, constants.MEASURED_LOOKBACKS_H)
    assert rows, "the scan produced nothing"

    for row in rows:
        window = windows.window_slice(df, row["t"], row["lookback_h"])
        assert len(window) == row["n_edges"], f"edge count differs at {row['t']}"
        assert int(window["Is Laundering"].sum()) == row["n_laundering"]
        expected_accounts = len(set(window["src"]) | set(window["dst"]))
        assert expected_accounts == row["n_accounts"], f"account count differs at {row['t']}"


def test_scan_only_covers_rule_3_valid_training_times(mini_dataset):
    """The scan must not include training times whose lookback runs off the
    start of the data — the bug that made the first full-data run report a
    clamped window as the largest."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "kaggle"))
    from kaggle_runner import scan_window_sizes

    df = pipeline.build(mini_dataset).full_df
    for row in scan_window_sizes(df, constants.MEASURED_LOOKBACKS_H):
        if row["split"] == "train":
            assert windows.has_full_lookback(
                pd.Timestamp(row["t"]), row["lookback_h"]
            ), f"scan included an invalid training time {row['t']} at L={row['lookback_h']}"


def test_window_summary_shape(mini_dataset):
    df = pipeline.build(mini_dataset).full_df
    built = list(windows.iter_windows(df, windows.detection_times(), lookback_h=24))
    summary = windows.window_summary(built)
    assert set(summary.columns) >= {
        "t", "split", "n_transactions", "n_laundering", "laundering_rate", "n_accounts"
    }
    assert (summary["laundering_rate"] <= 1.0).all()
