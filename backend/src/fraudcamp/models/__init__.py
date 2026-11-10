"""Models: the centralized baselines of Phase 3.

`xgb_baseline` is the non-graph control (stage 3.3) and `gnn` is the graph
model (stage 3.4). Neither defines features or thresholds of its own —
features come from `features.py` and tau from `evaluation.select_tau` — so
a model comparison is a comparison of models. Both are trained at five
seeds, in a full and a no-format variant, at both lookbacks.
"""

from .gnn import (
    GNN_SEEDS,
    GNN_VARIANTS,
    HEAVY_TAILED_NODE_FEATURES,
    GNNParams,
    NodePreprocessor,
    TrainedGNN,
    WindowGraph,
    average_precision,
    balanced_bce,
    build_model,
    build_window_graph,
    fit_node_preprocessor,
    is_trained,
    raw_node_features,
    set_seeds,
    train_gnn,
    validation_detection_times,
)
from .xgb_baseline import (
    MODEL_SEEDS,
    MODEL_VARIANTS,
    TrainedBaseline,
    TrainingData,
    XGBParams,
    build_training_data,
    dropped_columns,
    feature_importance,
    keep_mask,
    train_baseline,
)

__all__ = [
    "GNN_SEEDS",
    "GNN_VARIANTS",
    "HEAVY_TAILED_NODE_FEATURES",
    "GNNParams",
    "NodePreprocessor",
    "TrainedGNN",
    "WindowGraph",
    "average_precision",
    "balanced_bce",
    "build_model",
    "build_window_graph",
    "fit_node_preprocessor",
    "is_trained",
    "raw_node_features",
    "set_seeds",
    "train_gnn",
    "validation_detection_times",
    "MODEL_SEEDS",
    "MODEL_VARIANTS",
    "TrainedBaseline",
    "TrainingData",
    "XGBParams",
    "build_training_data",
    "dropped_columns",
    "feature_importance",
    "keep_mask",
    "train_baseline",
]
