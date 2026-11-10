"""The centralized GNN: GINEConv edge classification over one window.

Stage 3.4, under the design approved on 2026-11-10. What matters most here
is what this module does *not* own: the windows, the features, the labels,
the matching rule, the tau grid and the paired comparisons all come from
the same code that scored XGBoost, so a GNN-versus-XGBoost difference is a
difference between models.

The architecture: a linear projection of the six node features to hidden
64, then three GINEConv layers at 64 with layer normalisation after each,
then an edge readout MLP on `[h_src, h_dst, edge_features]`. Message
passing runs **both ways** - every edge is also inserted reversed, with a
flag column saying which direction it is - so a node sees what it sent and
what it received. Only the real transactions are scored; the reversed
copies exist to carry messages.

Two things are model-side only, and deliberately do not touch the shared
feature spec:

* **`log1p` on the heavy-tailed node counts** before standardisation. The
  shared spec standardises raw degrees, and standardising a heavy-tailed
  column does not bound it: on the 1% sample's largest windows the
  standardised node features still reach |x| = 110 and 159. Sum
  aggregation then compounds that over three layers, which is what produced
  the probe's initial losses of 10,615 and 356,394. XGBoost is indifferent
  to monotone transforms of its inputs - a tree splits the same either way
  - so this is applied here and nowhere else, and the stage 3.3 results
  stand untouched.
* **Layer normalisation** after each convolution, which bounds the scale
  per node per layer whatever the degree.

No account identity and no learned per-account embedding: a node's only
inputs are its six window features. The graph is built from `[t - L, t)`
and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants, features, windows
from ..evaluation.edge_metrics import average_precision
from .xgb_baseline import dropped_columns, keep_mask

#: Model variants, as the edge-feature prefixes each one drops. The same
#: ablation as stage 3.3, so "every model both ways" holds across models.
#: Separate names from the XGBoost variants because a results table has to
#: say which model it is talking about.
GNN_VARIANTS: dict[str, tuple[str, ...]] = {
    "gnn_centralized": (),
    "gnn_centralized_no_format": ("fmt_",),
}

#: Seeds every configuration is trained at, matching stage 3.3.
GNN_SEEDS = (0, 1, 2, 3, 4)

#: The node features that are counts, and so heavy-tailed: a hub account's
#: degree is orders of magnitude above the median. `log_in_amount` and
#: `log_out_amount` are already `log1p` in the shared features and are left
#: alone.
HEAVY_TAILED_NODE_FEATURES = (
    "in_degree",
    "out_degree",
    "n_counterparties",
    "n_currencies",
)


@dataclass(frozen=True)
class GNNParams:
    """The single frozen configuration.

    Fixed before any full-data training and never tuned on validation or
    test, which is the condition that makes the GNN-versus-XGBoost
    comparison a comparison of models rather than of tuning effort -
    XGBoost got one configuration too.
    """

    layers: int = 3
    hidden: int = 64
    dropout: float = 0.0
    learning_rate: float = 1e-3
    #: Zero: the model is ~50k parameters, regularisation comes from early
    #: stopping, and a decay term would be one more knob nobody tuned.
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 10
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# node preprocessing: model-side only
# --------------------------------------------------------------------------


@dataclass
class NodePreprocessor:
    """`log1p` the heavy-tailed counts, then standardise.

    Fitted on **training windows only**, like the shared feature spec, and
    saved with the model so scoring applies exactly what training applied.
    The statistics are the GNN's own: the shared spec's node statistics were
    computed on raw counts and do not describe the transformed column.
    """

    feature_names: list[str]
    log1p_columns: list[str]
    mean: list[float]
    std: list[float]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Raw node features in, model inputs out."""
        if raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} node features, "
                f"got {raw.shape[1]}"
            )
        matrix = np.asarray(raw, dtype=np.float64).copy()
        for index, name in enumerate(self.feature_names):
            if name in self.log1p_columns:
                matrix[:, index] = np.log1p(matrix[:, index])
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        return ((matrix - mean) / std).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NodePreprocessor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def raw_node_features(window) -> tuple[list[str], np.ndarray, list[str]]:
    """The window's node features *before* the shared spec standardises them.

    `features.build_window_features` returns them already standardised with
    statistics fitted on raw counts, which is the wrong starting point for a
    `log1p`. Recomputing from the transactions is a second pass over the
    window's groupbys, which costs a few seconds on the largest window and
    is paid once per window per run; the alternative - inverting the shared
    scaler - would depend on its internals and on no column having zero
    variance.

    The account order is the same by construction, because this is the same
    function `build_window_features` calls.
    """
    return features._raw_node_features(window.transactions)


def fit_node_preprocessor(train_windows: list) -> NodePreprocessor:
    """Fit the `log1p` + standardise statistics on training windows ONLY."""
    if not train_windows:
        raise ValueError("no training windows: cannot fit a node preprocessor")
    offenders = sorted({w.split for w in train_windows if w.split != "train"} - {None})
    if offenders:
        raise ValueError(
            "fit_node_preprocessor must see only training windows, got splits: "
            f"{offenders}. Fitting on validation or test data would leak."
        )

    matrices, names = [], None
    for window in train_windows:
        _, raw, node_names = raw_node_features(window)
        names = node_names
        matrices.append(np.asarray(raw, dtype=np.float64))
    pooled = np.vstack(matrices)

    log1p_columns = [n for n in names if n in HEAVY_TAILED_NODE_FEATURES]
    for index, name in enumerate(names):
        if name in log1p_columns:
            pooled[:, index] = np.log1p(pooled[:, index])

    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    # A constant column would otherwise divide by zero; 1.0 leaves it as a
    # centred constant, which carries no information either way.
    std[std == 0] = 1.0
    return NodePreprocessor(
        feature_names=list(names),
        log1p_columns=log1p_columns,
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
    )


# --------------------------------------------------------------------------
# the window graph
# --------------------------------------------------------------------------


@dataclass
class WindowGraph:
    """One window as the tensors the model consumes, on the CPU.

    `target_*` are the real transactions, in the window's own row order, so
    the scores line up with `src`/`dst` for cluster extraction. The
    bidirectional `edge_*` carry messages and are never scored.
    """

    x: object  # torch.Tensor (n_accounts, node_dims)
    edge_index: object  # (2, 2E)
    edge_attr: object  # (2E, edge_dims + 1)
    target_index: object  # (2, E)
    target_attr: object  # (E, edge_dims)
    labels: object  # (E,)
    n_edges: int
    n_accounts: int

    def to(self, device):
        import torch  # noqa: F401 - imported for its side effect on dtypes

        return WindowGraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            target_index=self.target_index.to(device),
            target_attr=self.target_attr.to(device),
            labels=self.labels.to(device),
            n_edges=self.n_edges,
            n_accounts=self.n_accounts,
        )

    def memory_mb(self) -> float:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.x, self.edge_index, self.edge_attr,
                self.target_index, self.target_attr, self.labels,
            )
        ) / 1e6


def build_window_graph(
    window,
    spec: features.FeatureSpec,
    node_pre: NodePreprocessor,
    variant: str = "gnn_centralized",
) -> WindowGraph:
    """Features for one window, as a graph, with the variant's columns gone."""
    import torch

    built = features.build_window_features(window, spec)
    _, raw_nodes, _ = raw_node_features(window)
    if raw_nodes.shape[0] != built.n_nodes:
        raise ValueError(
            f"node feature rows ({raw_nodes.shape[0]}) do not match the "
            f"window's accounts ({built.n_nodes}); the account orders differ"
        )

    edge_attr = built.edge_features
    prefixes = GNN_VARIANTS[variant]
    if prefixes:
        mask = keep_mask(built.edge_feature_names, prefixes)
        edge_attr = edge_attr[:, mask]

    x = torch.from_numpy(node_pre.transform(raw_nodes))
    target_index = torch.from_numpy(built.edge_index.astype("int64"))
    target_attr = torch.from_numpy(np.ascontiguousarray(edge_attr))
    labels = torch.from_numpy(built.labels.astype("float32"))

    n_edges = target_attr.shape[0]
    edge_index = torch.cat([target_index, target_index.flip(0)], dim=1)
    flag = torch.cat([torch.zeros(n_edges, 1), torch.ones(n_edges, 1)], dim=0)
    both_attr = torch.cat(
        [torch.cat([target_attr, target_attr], dim=0), flag], dim=1
    )

    return WindowGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=both_attr,
        target_index=target_index,
        target_attr=target_attr,
        labels=labels,
        n_edges=n_edges,
        n_accounts=int(x.shape[0]),
    )


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def build_model(edge_dims: int, node_dims: int, params: GNNParams):
    """GINE 3x64 with layer normalisation, plus the edge readout.

    The six node features are projected to `hidden` **before** the first
    convolution. PyG's `GINEConv(edge_dim=...)` projects edge features down
    to the node width to add them to the messages, so a first layer at the
    raw node width would squeeze 55 edge features into 6 dimensions and
    throw most of the edge information away before it was ever used.
    """
    import torch
    from torch import nn
    from torch_geometric.nn import GINEConv

    edge_dim = edge_dims + 1  # the direction flag

    class CampaignGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.Linear(node_dims, params.hidden),
                nn.LayerNorm(params.hidden),
                nn.ReLU(),
            )
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(params.layers):
                self.convs.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(params.hidden, params.hidden),
                            nn.ReLU(),
                            nn.Linear(params.hidden, params.hidden),
                        ),
                        edge_dim=edge_dim,
                    )
                )
                self.norms.append(nn.LayerNorm(params.hidden))
            self.dropout = nn.Dropout(params.dropout) if params.dropout else None
            self.readout = nn.Sequential(
                nn.Linear(2 * params.hidden + edge_dims, params.hidden),
                nn.ReLU(),
                nn.Linear(params.hidden, 1),
            )

        def forward(self, x, edge_index, edge_attr, target_index, target_attr):
            h = self.input_projection(x)
            for conv, norm in zip(self.convs, self.norms):
                h = torch.relu(norm(conv(h, edge_index, edge_attr)))
                if self.dropout is not None:
                    h = self.dropout(h)
            src, dst = target_index
            return self.readout(
                torch.cat([h[src], h[dst], target_attr], dim=1)
            ).squeeze(-1)

    return CampaignGNN()


# --------------------------------------------------------------------------
# loss and metric
# --------------------------------------------------------------------------


def balanced_bce(logits, labels):
    """Half the mean over positives plus half the mean over negatives.

    The approved loss. Every window contributes comparably **by
    construction**: the value does not depend on the window's size or on its
    positive rate, and at initialisation it is `ln 2` for any window, so a
    first step that is not near 0.69 is a bug rather than a mystery.

    The class weight enters as the per-sample weights `1 / (2 n_pos)` and
    `1 / (2 n_neg)`, so the ratio of per-positive to per-negative weight is
    exactly `n_neg / n_pos` - the frozen class-weighting principle. This is
    the same objective as `pos_weight=n_neg/n_pos` under mean reduction, up
    to the factor `n_neg / N` (measured at 0.9996-0.9997 on the full data's
    largest windows) and a constant 2 the learning rate absorbs.

    A window with no positives falls back to the negative mean alone, since
    half of nothing is not a number.
    """
    import torch

    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any()):
        return losses[negative].mean()
    if not bool(negative.any()):
        return losses[positive].mean()
    return 0.5 * losses[positive].mean() + 0.5 * losses[negative].mean()


#: Edge-level average precision, the early-stopping metric. Imported from
#: `evaluation.edge_metrics` rather than defined here, so that the number the
#: GNN stops on and the number XGBoost is diagnosed with are produced by one
#: function and cannot drift apart.
_ = average_precision


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class TrainedGNN:
    """A trained model plus everything needed to score a window with it."""

    model: object
    spec: features.FeatureSpec
    node_pre: NodePreprocessor
    params: GNNParams
    variant: str
    lookback_h: int
    edge_dims: int
    node_dims: int
    training: dict = field(default_factory=dict)

    def score_graph(self, graph: WindowGraph, device=None) -> np.ndarray:
        """Score an already-built window graph.

        Separate from `scorer` so a caller evaluating several models can
        build each window once and reuse it: featurising a window is far
        more expensive than a forward pass, and five seeds of one variant
        consume byte-identical graphs. The graph must have been built for
        this model's variant - the edge columns have to match.
        """
        import torch

        if graph.target_attr.shape[1] != self.edge_dims:
            raise ValueError(
                f"graph has {graph.target_attr.shape[1]} edge features, this "
                f"model expects {self.edge_dims}; it was built for a "
                "different variant"
            )
        target = device or next(self.model.parameters()).device
        on_device = graph if graph.x.device == target else graph.to(target)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                on_device.x, on_device.edge_index, on_device.edge_attr,
                on_device.target_index, on_device.target_attr,
            )
            probabilities = torch.sigmoid(logits)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def build_graph(self, window_transactions: pd.DataFrame, t) -> WindowGraph:
        """The graph for one window, for this model's variant."""
        window = windows.Window(
            t=pd.Timestamp(t),
            lookback_h=self.lookback_h,
            split=windows.split_of_detection_time(t),
            transactions=window_transactions,
        )
        return build_window_graph(window, self.spec, self.node_pre, self.variant)

    def scorer(self, device=None):
        """A `scorer(window_transactions, t)` - the XGBoost scorer's interface.

        Returns one probability per transaction in the window's own row
        order, so `evaluation.evaluate_split` accepts it with no
        model-specific path. The window is all it is given, so it cannot see
        past `t`. Builds the graph per call; use `score_graph` when several
        models share a window.
        """
        target = device or next(self.model.parameters()).device

        def score(window_transactions: pd.DataFrame, t) -> np.ndarray:
            if len(window_transactions) == 0:
                return np.zeros(0, dtype=np.float64)
            return self.score_graph(
                self.build_graph(window_transactions, t), device=target
            )

        return score

    def save(self, directory: str | Path) -> Path:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "gnn_model.pt")
        self.spec.save(directory / "feature_spec.json")
        self.node_pre.save(directory / "node_preprocessor.json")
        (directory / "model_meta.json").write_text(
            json.dumps(
                {
                    "variant": self.variant,
                    "lookback_h": self.lookback_h,
                    "edge_dims": self.edge_dims,
                    "node_dims": self.node_dims,
                    "params": self.params.as_dict(),
                    "training": self.training,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, device=None) -> "TrainedGNN":
        import torch

        directory = Path(directory)
        meta = json.loads((directory / "model_meta.json").read_text(encoding="utf-8"))
        params = GNNParams(**meta["params"])
        model = build_model(
            int(meta["edge_dims"]), int(meta["node_dims"]), params
        )
        state = torch.load(
            directory / "gnn_model.pt",
            map_location=device or "cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(device or "cpu")
        model.eval()
        return cls(
            model=model,
            spec=features.FeatureSpec.load(directory / "feature_spec.json"),
            node_pre=NodePreprocessor.load(directory / "node_preprocessor.json"),
            params=params,
            variant=meta["variant"],
            lookback_h=int(meta["lookback_h"]),
            edge_dims=int(meta["edge_dims"]),
            node_dims=int(meta["node_dims"]),
            training=meta.get("training", {}),
        )


def is_trained(directory: str | Path) -> bool:
    """Has a finished model been saved here? Used to skip on restart."""
    directory = Path(directory)
    return all(
        (directory / name).exists()
        for name in (
            "gnn_model.pt", "feature_spec.json",
            "node_preprocessor.json", "model_meta.json",
        )
    )


def set_seeds(seed: int) -> dict:
    """Seed every generator, and report whether determinism was obtainable.

    GPU scatter accumulates in nondeterministic order and floating-point
    addition is not associative, so bit-identical training is not promised:
    `use_deterministic_algorithms` is *attempted* and the outcome recorded
    rather than assumed. Two things make that acceptable - the spread over
    five seeds is reported and will dwarf last-bit drift, and evaluation is
    exactly reproducible either way because it runs from saved weights.
    """
    import os
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = False
    error = None
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"seed": seed, "deterministic_algorithms": deterministic, "error": error}


def validation_detection_times(lookback_h: int) -> list:
    """The detection times early stopping is allowed to see.

    The **validation detection split**, `(Sept 5, Sept 6]` - four times -
    and deliberately not the longer validation *evaluation horizon* that tau
    selection uses. That horizon runs to Sept 8 and so contains detection
    times inside the test detection split; edge-level average precision
    reads every label in a window, so selecting epochs on it would be
    selecting on test-period edges. Tau selection can use the horizon
    because only validation campaigns' outcomes enter it (rule 2); this
    cannot.
    """
    return windows.full_lookback_detection_times("val", lookback_h)


def train_gnn(
    full_df: pd.DataFrame,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    variant: str = "gnn_centralized",
    params: GNNParams | None = None,
    spec: features.FeatureSpec | None = None,
    node_pre: NodePreprocessor | None = None,
    device=None,
    progress=None,
) -> TrainedGNN:
    """Full-batch training: one whole window per step.

    Windows are featurised **once** and cached on the host, then moved to
    the device one at a time. Re-featurising every epoch would cost ~70s
    against ~4s of compute and would dominate training by twenty-fold.

    Window order is shuffled every epoch from `seed * 1000 + epoch`, which
    is the only stochasticity besides initialisation now that batching is
    gone. Early stopping watches validation edge-level average precision;
    the returned model is the best epoch's, not the last.
    """
    import torch

    if variant not in GNN_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(GNN_VARIANTS)}"
        )
    params = params or GNNParams()
    seed_report = set_seeds(params.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_times = windows.training_detection_times(lookback_h)
    if not training_times:
        raise ValueError(
            f"no training detection times with a full {lookback_h}h lookback"
        )
    train_windows = list(windows.iter_windows(full_df, training_times, lookback_h))
    spec = spec or features.fit_feature_spec(train_windows, lookback_h)
    node_pre = node_pre or fit_node_preprocessor(train_windows)

    train_graphs = [
        build_window_graph(w, spec, node_pre, variant) for w in train_windows
    ]
    validation_graphs = [
        build_window_graph(w, spec, node_pre, variant)
        for w in windows.iter_windows(
            full_df, validation_detection_times(lookback_h), lookback_h
        )
    ]
    if not validation_graphs:
        raise ValueError("no validation windows: early stopping has nothing to watch")

    edge_dims = train_graphs[0].target_attr.shape[1]
    node_dims = train_graphs[0].x.shape[1]
    model = build_model(edge_dims, node_dims, params).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    history: list[dict] = []
    best = {"epoch": -1, "average_precision": -1.0, "state": None}
    rng_root = params.seed * 1000

    for epoch in range(params.max_epochs):
        model.train()
        order = np.random.default_rng(rng_root + epoch).permutation(len(train_graphs))
        losses, max_logit = [], 0.0
        for position in order:
            graph = train_graphs[int(position)].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                graph.x, graph.edge_index, graph.edge_attr,
                graph.target_index, graph.target_attr,
            )
            loss = balanced_bce(logits, graph.labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            max_logit = max(max_logit, float(logits.detach().abs().max()))
            del graph, logits, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for graph in validation_graphs:
                on_device = graph.to(device)
                logits = model(
                    on_device.x, on_device.edge_index, on_device.edge_attr,
                    on_device.target_index, on_device.target_attr,
                )
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
                labels.append(on_device.labels.detach().cpu().numpy())
                del on_device, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        validation_ap = average_precision(
            np.concatenate(scores), np.concatenate(labels)
        )

        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "val_average_precision": round(float(validation_ap), 6),
            "max_abs_logit": round(max_logit, 2),
        }
        if device.type == "cuda":
            row["peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 1
            )
            row["peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1e6, 1
            )
            torch.cuda.reset_peak_memory_stats()
        history.append(row)
        if progress:
            progress(row)

        if validation_ap > best["average_precision"]:
            best = {
                "epoch": epoch,
                "average_precision": float(validation_ap),
                "state": {k: v.detach().cpu().clone() for k, v in
                          model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= params.patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    return TrainedGNN(
        model=model,
        spec=spec,
        node_pre=node_pre,
        params=params,
        variant=variant,
        lookback_h=lookback_h,
        edge_dims=edge_dims,
        node_dims=node_dims,
        training={
            "variant": variant,
            "lookback_h": lookback_h,
            "dropped_prefixes": list(GNN_VARIANTS[variant]),
            "dropped_columns": dropped_columns(
                list(spec.edge_feature_names), GNN_VARIANTS[variant]
            ),
            "n_training_windows": len(train_graphs),
            "n_validation_windows": len(validation_graphs),
            "training_times": [str(t) for t in training_times],
            "validation_times": [
                str(t) for t in validation_detection_times(lookback_h)
            ],
            "cached_graph_memory_mb": round(
                sum(g.memory_mb() for g in train_graphs + validation_graphs), 1
            ),
            "epochs_run": len(history),
            "best_epoch": best["epoch"],
            "best_val_average_precision": round(best["average_precision"], 6),
            "history": history,
            "seeding": seed_report,
            "device": str(device),
            "params": params.as_dict(),
        },
    )
