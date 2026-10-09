"""Edge and node features for one detection window.

Three rules govern everything here, and each has a test:

1. **Nothing from the future.** Every feature for detection time `t` is
   computed from the window `[t - L, t)` alone, which `windows.py` is the
   only thing allowed to slice.
2. **No identity.** No account id, bank id or institution is ever a
   feature. Banks appear only as the equality `From Bank != To Bank`, and
   accounts only as the equality `src == dst`; nothing distinguishes *which*
   account or bank is involved.
3. **No labels.** `Is Laundering` is carried alongside the features as the
   target, and never enters them.

Vocabularies (currencies, payment formats) and the scaler are fitted on
training windows only and saved, so validation and test see exactly the
transformation training saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants

OTHER = "__other__"

#: Columns standardised by the scaler. One-hot, cyclical and binary features
#: are already on a comparable scale and are left untouched.
CONTINUOUS_EDGE_FEATURES = ("log_amount_paid", "log_amount_received")
CONTINUOUS_NODE_FEATURES = (
    "in_degree",
    "out_degree",
    "log_in_amount",
    "log_out_amount",
    "n_counterparties",
    "n_currencies",
)

DOW_NAMES = tuple(f"dow_{i}" for i in range(7))


# ---------------------------------------------------------------------------
# vocabularies + scaler, fitted on training windows only
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Everything learned from the training windows. Saved to JSON so the
    exact transformation can be replayed for validation, test, and any later
    condition."""

    payment_currencies: list[str] = field(default_factory=list)
    receiving_currencies: list[str] = field(default_factory=list)
    payment_formats: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)
    node_feature_names: list[str] = field(default_factory=list)
    edge_mean: list[float] = field(default_factory=list)
    edge_std: list[float] = field(default_factory=list)
    node_mean: list[float] = field(default_factory=list)
    node_std: list[float] = field(default_factory=list)
    lookback_h: int = constants.DEFAULT_LOOKBACK_H
    fitted_on: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _vocabulary(values: pd.Series) -> list[str]:
    """Sorted unique values. Sorted so the column order is reproducible."""
    return sorted(str(v) for v in pd.unique(values.dropna()))


def fit_feature_spec(
    train_windows: list,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
) -> FeatureSpec:
    """Fit vocabularies and the scaler on training windows ONLY.

    Passing anything other than training windows here is the single most
    dangerous mistake available in this module, so it is checked rather than
    trusted.
    """
    if not train_windows:
        raise ValueError("no training windows: cannot fit a feature spec")

    offenders = sorted({w.split for w in train_windows if w.split != "train"} - {None})
    if offenders:
        raise ValueError(
            "fit_feature_spec must see only training windows, got splits: "
            f"{offenders}. Fitting on validation or test data would leak."
        )

    frames = [w.transactions for w in train_windows]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    spec = FeatureSpec(
        payment_currencies=_vocabulary(combined["Payment Currency"]),
        receiving_currencies=_vocabulary(combined["Receiving Currency"]),
        payment_formats=_vocabulary(combined["Payment Format"]),
        lookback_h=lookback_h,
        fitted_on={
            "n_windows": len(train_windows),
            "n_transactions": int(len(combined)),
            "first_t": str(min(w.t for w in train_windows)),
            "last_t": str(max(w.t for w in train_windows)),
            "max_timestamp_seen": str(combined["ts"].max()) if len(combined) else None,
        },
    )

    # Column names are fixed by the vocabularies, so build one window's worth
    # of raw (unscaled) features to learn the names and the scaler stats.
    edge_blocks, node_blocks = [], []
    for window in train_windows:
        if window.n_transactions == 0:
            continue
        edges, edge_names = _raw_edge_features(window.transactions, spec)
        accounts, nodes, node_names = _raw_node_features(window.transactions)
        edge_blocks.append(edges)
        node_blocks.append(nodes)
        spec.edge_feature_names = edge_names
        spec.node_feature_names = node_names

    if not edge_blocks:
        raise ValueError("every training window was empty: cannot fit a scaler")

    all_edges = np.vstack(edge_blocks)
    all_nodes = np.vstack(node_blocks)

    spec.edge_mean, spec.edge_std = _scaler_stats(
        all_edges, spec.edge_feature_names, CONTINUOUS_EDGE_FEATURES
    )
    spec.node_mean, spec.node_std = _scaler_stats(
        all_nodes, spec.node_feature_names, CONTINUOUS_NODE_FEATURES
    )
    return spec


def _scaler_stats(
    matrix: np.ndarray, names: list[str], continuous: tuple[str, ...]
) -> tuple[list[float], list[float]]:
    """Mean/std per column, but identity (0, 1) for columns that must not be
    scaled, so the transform can be applied uniformly."""
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    std = np.ones(matrix.shape[1], dtype=np.float64)
    for i, name in enumerate(names):
        if name in continuous:
            mean[i] = float(matrix[:, i].mean())
            column_std = float(matrix[:, i].std())
            # a constant column would divide by zero; leave it centred only
            std[i] = column_std if column_std > 1e-12 else 1.0
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# raw features
# ---------------------------------------------------------------------------


def _one_hot(values: pd.Series, vocab: list[str], prefix: str) -> tuple[np.ndarray, list[str]]:
    """One-hot against a fixed vocabulary, with a trailing OTHER column for
    anything unseen in training."""
    # A dict map rather than pd.Categorical: passing values outside the
    # categories is deprecated there, and unseen values are exactly the case
    # the OTHER column exists for.
    lookup = {value: index for index, value in enumerate(vocab)}
    codes = values.astype(str).map(lookup).fillna(-1).astype(np.int64).to_numpy()
    out = np.zeros((len(values), len(vocab) + 1), dtype=np.float64)
    known = codes >= 0
    rows = np.arange(len(values))
    out[rows[known], codes[known]] = 1.0
    out[rows[~known], len(vocab)] = 1.0
    names = [f"{prefix}_{v}" for v in vocab] + [f"{prefix}_{OTHER}"]
    return out, names


def _raw_edge_features(
    txns: pd.DataFrame, spec: FeatureSpec
) -> tuple[np.ndarray, list[str]]:
    """One row per transaction. Nothing here looks beyond the row itself."""
    hour = txns["ts"].dt.hour.to_numpy() + txns["ts"].dt.minute.to_numpy() / 60.0
    angle = 2 * np.pi * hour / 24.0

    columns: list[np.ndarray] = [
        np.log1p(txns["Amount Paid"].to_numpy(dtype=np.float64)),
        np.log1p(txns["Amount Received"].to_numpy(dtype=np.float64)),
        (txns["Payment Currency"].to_numpy() == txns["Receiving Currency"].to_numpy()).astype(np.float64),
        np.sin(angle),
        np.cos(angle),
        (txns["src"].to_numpy() == txns["dst"].to_numpy()).astype(np.float64),
        (txns["From Bank"].to_numpy() != txns["To Bank"].to_numpy()).astype(np.float64),
    ]
    names = [
        "log_amount_paid",
        "log_amount_received",
        "same_currency",
        "hour_sin",
        "hour_cos",
        "self_transfer",
        "cross_bank",
    ]

    dow = txns["ts"].dt.dayofweek.to_numpy()
    dow_matrix = np.zeros((len(txns), 7), dtype=np.float64)
    dow_matrix[np.arange(len(txns)), dow] = 1.0
    columns.append(dow_matrix)
    names.extend(DOW_NAMES)

    for series, vocab, prefix in (
        (txns["Payment Currency"], spec.payment_currencies, "pay_cur"),
        (txns["Receiving Currency"], spec.receiving_currencies, "recv_cur"),
        (txns["Payment Format"], spec.payment_formats, "fmt"),
    ):
        block, block_names = _one_hot(series, vocab, prefix)
        columns.append(block)
        names.extend(block_names)

    matrix = np.column_stack([c if c.ndim == 2 else c.reshape(-1, 1) for c in columns])
    return matrix, names


def _raw_node_features(txns: pd.DataFrame) -> tuple[list[str], np.ndarray, list[str]]:
    """Per-account features from the window alone.

    Accounts are returned sorted, so node ordering is reproducible across
    processes for the same reason the visibility search sorts.
    """
    accounts = sorted(set(txns["src"]) | set(txns["dst"]))
    index = pd.Index(accounts)

    out_group = txns.groupby("src", sort=False)
    in_group = txns.groupby("dst", sort=False)

    out_degree = out_group.size().reindex(index, fill_value=0).to_numpy(dtype=np.float64)
    in_degree = in_group.size().reindex(index, fill_value=0).to_numpy(dtype=np.float64)
    out_amount = (
        out_group["Amount Paid"].sum().reindex(index, fill_value=0.0).to_numpy(dtype=np.float64)
    )
    in_amount = (
        in_group["Amount Received"].sum().reindex(index, fill_value=0.0).to_numpy(dtype=np.float64)
    )

    # counterparties and currencies need both directions pooled per account
    pooled = pd.concat(
        [
            pd.DataFrame(
                {
                    "account": txns["src"].to_numpy(),
                    "counterparty": txns["dst"].to_numpy(),
                    "currency": txns["Payment Currency"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "account": txns["dst"].to_numpy(),
                    "counterparty": txns["src"].to_numpy(),
                    "currency": txns["Receiving Currency"].to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    pooled_group = pooled.groupby("account", sort=False)
    n_counterparties = (
        pooled_group["counterparty"].nunique().reindex(index, fill_value=0).to_numpy(dtype=np.float64)
    )
    n_currencies = (
        pooled_group["currency"].nunique().reindex(index, fill_value=0).to_numpy(dtype=np.float64)
    )

    matrix = np.column_stack(
        [
            in_degree,
            out_degree,
            np.log1p(in_amount),
            np.log1p(out_amount),
            n_counterparties,
            n_currencies,
        ]
    )
    names = list(CONTINUOUS_NODE_FEATURES)
    return accounts, matrix, names


# ---------------------------------------------------------------------------
# the public result
# ---------------------------------------------------------------------------


@dataclass
class WindowFeatures:
    """Everything a model needs for one detection window, and nothing a
    model is allowed to peek at beyond it.

    `labels`, `campaign_ids` and `timestamps` are carried for training
    targets and for evaluation; they are never inputs.
    """

    t: pd.Timestamp
    lookback_h: int
    split: str | None
    accounts: list[str]
    node_features: np.ndarray  # (n_accounts, d_node)
    edge_features: np.ndarray  # (n_edges, d_edge)
    edge_index: np.ndarray  # (2, n_edges) into `accounts`
    labels: np.ndarray  # (n_edges,)
    campaign_ids: np.ndarray  # (n_edges,) -1 when not part of a campaign
    timestamps: np.ndarray
    edge_feature_names: list[str]
    node_feature_names: list[str]

    @property
    def n_edges(self) -> int:
        return int(self.edge_features.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if self.n_edges else 0.0

    def class_weight(self) -> float:
        """Weight for the positive class under extreme imbalance:
        n_negative / n_positive, so the two classes contribute equally.
        Returns 1.0 when a window has no positives."""
        positives = float(self.labels.sum())
        if positives == 0:
            return 1.0
        return float(len(self.labels) - positives) / positives

    def xgboost_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Flat per-transaction rows for the non-graph baseline: the edge's
        own features plus the window features of both endpoints."""
        src_nodes = self.node_features[self.edge_index[0]]
        dst_nodes = self.node_features[self.edge_index[1]]
        matrix = np.hstack([self.edge_features, src_nodes, dst_nodes])
        names = (
            list(self.edge_feature_names)
            + [f"src_{n}" for n in self.node_feature_names]
            + [f"dst_{n}" for n in self.node_feature_names]
        )
        return matrix, names


def build_window_features(window, spec: FeatureSpec) -> WindowFeatures:
    """Features for one window, using a spec fitted on training windows."""
    txns = window.transactions
    if len(txns) == 0:
        raise ValueError(f"window at {window.t} is empty: nothing to featurise")

    edge_matrix, edge_names = _raw_edge_features(txns, spec)
    accounts, node_matrix, node_names = _raw_node_features(txns)

    edge_matrix = _apply_scaler(edge_matrix, spec.edge_mean, spec.edge_std)
    node_matrix = _apply_scaler(node_matrix, spec.node_mean, spec.node_std)

    position = {account: i for i, account in enumerate(accounts)}
    edge_index = np.vstack(
        [
            np.fromiter((position[a] for a in txns["src"]), dtype=np.int64, count=len(txns)),
            np.fromiter((position[a] for a in txns["dst"]), dtype=np.int64, count=len(txns)),
        ]
    )

    return WindowFeatures(
        t=window.t,
        lookback_h=window.lookback_h,
        split=window.split,
        accounts=accounts,
        node_features=node_matrix,
        edge_features=edge_matrix,
        edge_index=edge_index,
        labels=txns["Is Laundering"].to_numpy(dtype=np.int64),
        campaign_ids=(
            txns["campaign_id"].to_numpy(dtype=np.int64)
            if "campaign_id" in txns
            else np.full(len(txns), -1, dtype=np.int64)
        ),
        timestamps=txns["ts"].to_numpy(),
        edge_feature_names=edge_names,
        node_feature_names=node_names,
    )


def _apply_scaler(matrix: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """Standardise, then narrow to float32.

    Arithmetic happens in float64 so the result is bit-identical run to run,
    but the stored matrix is float32: a busy 24h window holds ~1M
    transactions, where the difference is hundreds of megabytes per window.
    """
    if not mean or not std:
        return matrix.astype(np.float32)
    if len(mean) != matrix.shape[1]:
        raise ValueError(
            f"scaler expects {len(mean)} columns, got {matrix.shape[1]}. The spec "
            "was fitted with different vocabularies than these features use."
        )
    scaled = (matrix - np.asarray(mean)) / np.asarray(std)
    return scaled.astype(np.float32)
