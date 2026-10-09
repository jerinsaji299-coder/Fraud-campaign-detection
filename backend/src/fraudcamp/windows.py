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
