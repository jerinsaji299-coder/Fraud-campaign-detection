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
