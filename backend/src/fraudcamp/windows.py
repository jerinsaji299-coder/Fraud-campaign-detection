"""Detection times and the causal windows they look back over.

A detection time `t` is a moment at which a model is asked whether a
campaign is forming. At `t` it may see **only** transactions with
`t - L <= timestamp < t`: never a transaction at `t` itself, and never one
after it. Everything downstream (features, training, evaluation) goes
through this module so that the causality rule is enforced in exactly one
place rather than re-implemented per model.

The grid is aligned to 00:00 / 06:00 / 12:00 / 18:00 and runs in
`(DATA_START, DETECTION_TEST_END]`. `DATA_START` itself is excluded because
its window is empty by construction.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import pandas as pd

from . import constants


def detection_times(
    start: _dt.datetime = constants.DATA_START,
    end: _dt.datetime = constants.DETECTION_TEST_END,
    step_h: int = constants.DETECTION_WINDOW_STEP_H,
) -> list[pd.Timestamp]:
    """Every aligned detection time in `(start, end]`, earliest first."""
    step = pd.Timedelta(hours=step_h)

    # round `start` up onto the aligned grid
    first = pd.Timestamp(start).ceil(f"{step_h}h")
    if first <= pd.Timestamp(start):
        first = first + step

    times: list[pd.Timestamp] = []
    current = first
    while current <= pd.Timestamp(end):
        times.append(current)
        current = current + step
    return times


def split_of_detection_time(t: pd.Timestamp | _dt.datetime) -> str | None:
    """train / val / test, or None for a time outside the study period."""
    t = pd.Timestamp(t)
    if t <= pd.Timestamp(constants.DETECTION_TRAIN_END):
        return "train"
    if t <= pd.Timestamp(constants.DETECTION_VAL_END):
        return "val"
    if t <= pd.Timestamp(constants.DETECTION_TEST_END):
        return "test"
    return None


def detection_times_for_split(
    split: str,
    start: _dt.datetime = constants.DATA_START,
    end: _dt.datetime = constants.DETECTION_TEST_END,
    step_h: int = constants.DETECTION_WINDOW_STEP_H,
) -> list[pd.Timestamp]:
    if split not in constants.DETECTION_SPLIT_ORDER:
        raise ValueError(
            f"split must be one of {constants.DETECTION_SPLIT_ORDER}, got {split!r}"
        )
    return [t for t in detection_times(start, end, step_h) if split_of_detection_time(t) == split]


def has_full_lookback(
    t: pd.Timestamp | _dt.datetime,
    lookback_h: int,
    data_start: _dt.datetime = constants.DATA_START,
) -> bool:
    """Whether `[t - L, t)` lies entirely within the data (rule 3).

    A detection time close to the start of the data would otherwise give a
    silently short window — a 24h feature computed over 6 hours — which
    would train the model on a different distribution than it sees later.
    """
    return pd.Timestamp(t) - pd.Timedelta(hours=lookback_h) >= pd.Timestamp(data_start)


def training_detection_times(
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    start: _dt.datetime = constants.DATA_START,
    end: _dt.datetime = constants.DETECTION_TEST_END,
    step_h: int = constants.DETECTION_WINDOW_STEP_H,
) -> list[pd.Timestamp]:
    """Training detection times that have their full lookback (rule 3).

    This is what training must iterate over; `detection_times_for_split`
    ("train") is the unfiltered grid and would include short windows.
    """
    return [
        t
        for t in detection_times_for_split("train", start, end, step_h)
        if has_full_lookback(t, lookback_h, start)
    ]


def usable_training_times_per_lookback(
    lookbacks: tuple[int, ...] = constants.LOOKBACKS_H,
) -> dict[int, int]:
    """How many training detection times survive rule 3, per lookback.

    Longer lookbacks cost training windows: the data starts on Sept 1 and
    training ends on Sept 5, so a 72h lookback leaves only the last few.
    """
    return {lookback: len(training_detection_times(lookback)) for lookback in lookbacks}


def evaluation_horizon(split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The (exclusive start, inclusive end] detection window a split's
    campaigns are evaluated over (rule 2)."""
    if split not in constants.EVALUATION_HORIZONS:
        known = sorted(constants.EVALUATION_HORIZONS)
        raise ValueError(
            f"no evaluation horizon defined for split {split!r}; defined: {known}. "
            "The stress horizon is deliberately unspecified - see README "
            "'Known limitations and open questions'."
        )
    start, end = constants.EVALUATION_HORIZONS[split]
    return pd.Timestamp(start), pd.Timestamp(end)


def evaluation_detection_times(
    split: str, step_h: int = constants.DETECTION_WINDOW_STEP_H
) -> list[pd.Timestamp]:
    """Detection times a split's campaigns are evaluated at (rule 2).

    Note the validation and test horizons overlap by design: validation
    campaigns start on Sept 5 but need until Sept 8 to finish. Only
    validation-campaign labels inform tau, so nothing leaks.
    """
    start, end = evaluation_horizon(split)
    return [t for t in detection_times(constants.DATA_START, end, step_h) if t > start]


def hit_kind(evaluated_split: str, hit_campaign_split: str | None) -> str:
    """Classify a cluster that matched a campaign (rule 1).

    A cluster can legitimately hit a real campaign that belongs to a
    different split than the one being scored. That is neither a success for
    this split nor a false alarm — it is a correct detection of something
    we are not currently measuring — so it is counted separately and
    excluded from both precision terms.
    """
    if hit_campaign_split == evaluated_split:
        return "hit"
    return "other_split_hit"


def window_slice(
    df: pd.DataFrame,
    t: pd.Timestamp | _dt.datetime,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    ts_column: str = "ts",
) -> pd.DataFrame:
    """Transactions visible at `t`: `t - lookback_h <= ts < t`.

    The upper bound is strict. A transaction stamped exactly `t` is in the
    future as far as a detector standing at `t` is concerned.
    """
    t = pd.Timestamp(t)
    lower = t - pd.Timedelta(hours=lookback_h)
    mask = (df[ts_column] >= lower) & (df[ts_column] < t)
    return df[mask]


@dataclass(frozen=True)
class Window:
    t: pd.Timestamp
    lookback_h: int
    split: str | None
    transactions: pd.DataFrame

    @property
    def start(self) -> pd.Timestamp:
        return self.t - pd.Timedelta(hours=self.lookback_h)

    @property
    def n_transactions(self) -> int:
        return len(self.transactions)

    @property
    def n_laundering(self) -> int:
        return int(self.transactions["Is Laundering"].sum())


def build_window(
    df: pd.DataFrame,
    t: pd.Timestamp | _dt.datetime,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
) -> Window:
    t = pd.Timestamp(t)
    return Window(
        t=t,
        lookback_h=lookback_h,
        split=split_of_detection_time(t),
        transactions=window_slice(df, t, lookback_h),
    )


def iter_windows(
    df: pd.DataFrame,
    times: list[pd.Timestamp],
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    skip_empty: bool = True,
):
    """Yield a Window per detection time. Empty windows are skipped by
    default: there is nothing to score, and they would otherwise contribute
    degenerate rows to training."""
    for t in times:
        window = build_window(df, t, lookback_h)
        if skip_empty and window.n_transactions == 0:
            continue
        yield window


def window_summary(windows) -> pd.DataFrame:
    """One row per window: size, label balance, time span. Used to sanity
    check a configuration before spending GPU time on it."""
    rows = []
    for w in windows:
        rows.append(
            {
                "t": w.t,
                "split": w.split,
                "lookback_h": w.lookback_h,
                "n_transactions": w.n_transactions,
                "n_laundering": w.n_laundering,
                "laundering_rate": (w.n_laundering / w.n_transactions) if w.n_transactions else 0.0,
                "n_accounts": len(set(w.transactions["src"]) | set(w.transactions["dst"]))
                if w.n_transactions
                else 0,
            }
        )
    return pd.DataFrame(rows)
