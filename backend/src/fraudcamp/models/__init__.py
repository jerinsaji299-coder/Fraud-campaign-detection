"""Models: the centralized baselines of Phase 3.

`xgb_baseline` is the non-graph control (stage 3.3); the GNN lands in 3.4.
Neither defines features or thresholds of its own — features come from
`features.py` and tau from `evaluation.select_tau` — so a model comparison
is a comparison of models.
"""

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
