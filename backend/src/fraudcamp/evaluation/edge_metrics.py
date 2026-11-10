"""Edge-level scoring quality: the diagnostic layer under the campaign layer.

Every result this project reports is campaign-level, because the research
question is about discovering campaigns. But a campaign-level number is the
product of two stages - score the edges, then cluster the survivors and
match them - and when two models differ it does not say which stage differs.
These metrics read the scores alone, before any threshold or clustering, so
a difference here is a difference in ranking and a difference that appears
only after clustering is a difference in how the scores are *shaped* for
connected-component extraction.

Nothing here selects anything. No tau is chosen on it, no model is stopped
on it except where the GNN's design says so (which uses `average_precision`
from this module on the validation **detection** split only), and no
headline number comes from it. It is evidence about a mechanism.

The label is `Is Laundering`, the same one both models were trained on,
including laundering that belongs to no campaign - so the two models are
scored against exactly the target they fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import constants, windows


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Edge-level average precision: the area under precision-recall.

    The GNN's early-stopping metric, and the diagnostic used to compare two
    models' raw ranking. Threshold-free, so it does not depend on a tau that
    has not been chosen yet, and it is the right summary for rare positives
    - a 0.1% positive rate makes ROC-AUC look good for almost any model.
    Campaign-level F1 is *not* used for stopping: it needs cluster
    extraction at several thresholds on every epoch, and over 26 validation
    campaigns it moves in steps of 1/26 and would stop on noise.

    Equivalent to scikit-learn's `average_precision_score`, implemented here
    because scikit-learn is not a dependency of this project.

    Returns NaN when a window holds no positive edge, because average
    precision is undefined there - not 0, which would read as a failure.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = labels.sum()
    if positives == 0 or len(scores) == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    true_positives = np.cumsum(hits)
    precision = true_positives / np.arange(1, len(hits) + 1)
    # sum of precision at each positive, over the number of positives: the
    # step-function area, which is what average_precision_score computes
    return float((precision * hits).sum() / positives)


def precision_at_n_positive(scores: np.ndarray, labels: np.ndarray) -> float:
    """Precision in the top-`P` scores, where `P` is the number of positives.

    The readable companion to average precision: "of the P edges it ranked
    highest, this fraction really is laundering". It equals recall at the
    same cut, since the cut is P wide, so one number says both.

    Ties are broken by `argsort`'s stable order, which is window row order -
    arbitrary but not biased towards either label, and identical for every
    model, which is what matters for a comparison.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0 or len(scores) == 0:
        return float("nan")
    top = np.argsort(-scores, kind="stable")[:positives]
    return float(labels[top].sum() / positives)


def edge_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """One window's edge-level scoring quality.

    `ap_lift` is average precision over the positive rate, because a random
    ranker's average precision *is* the positive rate: at 0.1% positives an
    AP of 0.4 is a 400x lift, and the raw 0.4 on its own reads as poor.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_edges = int(len(labels))
    n_positive = int(labels.sum())
    rate = n_positive / n_edges if n_edges else float("nan")
    ap = average_precision(scores, labels)
    return {
        "n_edges": n_edges,
        "n_positive": n_positive,
        "positive_rate": rate,
        "average_precision": ap,
        "ap_lift": (ap / rate) if (rate and not np.isnan(ap)) else float("nan"),
        "precision_at_n_positive": precision_at_n_positive(scores, labels),
    }


def edge_metrics_over_times(
    full_df: pd.DataFrame,
    scorer,
    times,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    label_column: str = "Is Laundering",
    **extra,
) -> list[dict]:
    """`edge_metrics` per detection time, plus a `pooled` row over all of them.

    The pooled row is not the mean of the per-window rows: average precision
    is not linear, and a window with four positives would otherwise weigh as
    much as one with four hundred. It is recomputed over the concatenated
    scores, which is also how the GNN's early stopping computes it.

    `scorer` has the usual signature `scorer(window_transactions, t)` and so
    cannot see outside its window.
    """
    rows: list[dict] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for t in times:
        window = windows.window_slice(full_df, t, lookback_h)
        if len(window) == 0:
            continue
        scores = np.asarray(scorer(window, t), dtype=np.float64)
        labels = window[label_column].to_numpy().astype(np.float64)
        if len(scores) != len(labels):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(labels)} rows "
                f"at {t}"
            )
        all_scores.append(scores)
        all_labels.append(labels)
        rows.append(
            {**extra, "t": pd.Timestamp(t), "lookback_h": lookback_h,
             "window": "one", **edge_metrics(scores, labels)}
        )

    if all_scores:
        rows.append(
            {
                **extra, "t": None, "lookback_h": lookback_h, "window": "pooled",
                **edge_metrics(
                    np.concatenate(all_scores), np.concatenate(all_labels)
                ),
            }
        )
    return rows
