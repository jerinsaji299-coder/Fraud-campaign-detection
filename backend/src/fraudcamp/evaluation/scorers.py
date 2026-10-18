"""Sanity scorers, built before any model so they can validate the
evaluation code rather than the reverse.

ORACLE scores each edge with its true label. It should discover almost
every evaluable campaign, at about the moment the campaign's third account
appears. If it does not, the evaluation code is wrong, not the detector.

RANDOM scores uniformly at random. It should find almost nothing and raise
many false alarms. If it scores well, the matching rule is too generous.
"""

from __future__ import annotations

import numpy as np


def oracle_scores(labels: np.ndarray) -> np.ndarray:
    """score = true label. The best any detector could possibly do."""
    return np.asarray(labels, dtype=np.float64)


def random_scores(n: int, seed: int) -> np.ndarray:
    """Uniform [0, 1). Seeded per (detection time, seed) by the caller so a
    run is reproducible."""
    return np.random.default_rng(seed).random(n)


def score_window(scorer: str, labels: np.ndarray, seed: int = 0) -> np.ndarray:
    if scorer == "oracle":
        return oracle_scores(labels)
    if scorer == "random":
        return random_scores(len(labels), seed)
    raise ValueError(f"unknown sanity scorer {scorer!r}; expected 'oracle' or 'random'")


#: Thresholds that make each sanity scorer behave as intended. The oracle's
#: scores are 0/1, so any tau in (0, 1] keeps exactly the laundering edges.
SANITY_TAU = {"oracle": 0.5, "random": 0.5}


class CachingScorer:
    """Memoise a scorer's output per detection time.

    A tau sweep calls the scorer once per tau per detection time with the
    same window every time, and a scorer's output does not depend on tau.
    For a model that means re-featurising a million-edge window twelve
    times to get twelve identical answers. This keeps the first answer.

    Only safe for a **deterministic** scorer at a **fixed lookback**, which
    is why the lookback is part of the construction and the row count is
    checked: if a window of a different size arrives at a time already
    cached, the assumption is broken and it raises rather than returning
    the wrong scores. The random scorer is deterministic in this sense too
    — it is seeded per detection time — but it is cheap enough not to need
    this.

    Scores are one float per edge, so a whole split's cache is tens of MB
    where the features would be gigabytes. Call `clear()` between models.
    """

    def __init__(self, scorer, lookback_h: int):
        self._scorer = scorer
        self._lookback_h = int(lookback_h)
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, window, t):
        key = (pd_timestamp(t), self._lookback_h)
        cached = self._cache.get(key)
        if cached is not None:
            if len(cached) != len(window):
                raise ValueError(
                    f"cached scores for {key} cover {len(cached)} rows but this "
                    f"window has {len(window)}; a CachingScorer is only valid "
                    "for one lookback over one dataframe"
                )
            self.hits += 1
            return cached
        scores = self._scorer(window, t)
        self._cache[key] = scores
        self.misses += 1
        return scores

    def clear(self) -> None:
        self._cache.clear()

    @property
    def cached_windows(self) -> int:
        return len(self._cache)


def pd_timestamp(t):
    """Normalise a detection time to a hashable, comparable key."""
    import pandas as pd

    return pd.Timestamp(t)
