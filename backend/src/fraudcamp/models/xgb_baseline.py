"""The non-graph baseline: XGBoost over per-transaction features.

This is the control the GNN has to beat. It sees exactly the same windows,
the same causal bounds and the same features as the graph model will, with
one difference that is the whole point of the comparison: each transaction
is an independent row. The edge's own features plus both endpoints' window
features go in; no message passing, no neighbourhood beyond the two
endpoints. If the GNN cannot beat this, the graph structure is not where
the signal is.

What is deliberately *not* here:

* **No tau.** The model emits a probability per transaction. Turning
  probabilities into detections is the evaluation's job, and tau is chosen
  on validation only, by the same `evaluation.select_tau` the sanity
  scorers used. The model cannot see a threshold.
* **No campaign knowledge.** Training labels are per-transaction
  `Is Laundering`, including laundering that belongs to no campaign.
  Campaign identity never reaches the model, in training or scoring.
* **No feature of its own.** Features come from `features.py` unchanged, so
  the leakage properties already tested there apply here without a second
  implementation to keep honest.

Training pools every rule-3-valid training detection time into one matrix
and fits one model per lookback. Pooling rather than one-model-per-window is
what makes a single scorer usable at every detection time, and rule 3 keeps
a silently-clamped short window out of the training distribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants, features, windows

#: The model variants, as feature prefixes each one drops. Everything else
#: — windows, labels, weighting, hyperparameters, seeds — is identical, so a
#: difference between them is attributable to the dropped columns and
#: nothing else.
#:
#: **Why the format ablation exists.** In the IBM synthetic data laundering
#: is strongly linked to payment format: on the 1% sample `fmt_ACH` alone
#: carries 46% of the baseline's gain and the format one-hots together carry
#: ~87%. Payment format is an allowed input under the frozen design — it is
#: a property of the transaction, not an identity or a label — but a feature
#: that strong in *synthetic* data is a shortcut a model can take instead of
#: learning transactional structure, and a result resting on it would say
#: more about the generator than about campaign discovery. So every model
#: from here on is reported **with and without** it, and the gap between the
#: two is part of the result.
#:
#: These are model variant names. They are not the four Phase 4-5
#: `constants.CONDITIONS`, which are about federation.
MODEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "xgboost_centralized": (),
    "xgboost_centralized_no_format": ("fmt_",),
}

#: Seeds every model is trained at, so a reported difference can be read
#: against the spread across seeds rather than taken from one run.
MODEL_SEEDS = (0, 1, 2, 3, 4)


def dropped_columns(feature_names: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """The columns a variant removes, in the matrix's own order."""
    return [
        name for name in feature_names
        if any(name.startswith(prefix) for prefix in prefixes)
    ]


def keep_mask(feature_names: list[str], prefixes: tuple[str, ...]):
    """Boolean mask over columns: True for the ones a variant keeps."""
    return np.array(
        [
            not any(name.startswith(prefix) for prefix in prefixes)
            for name in feature_names
        ],
        dtype=bool,
    )


#: Frozen hyperparameters for stage 3.3. Chosen once, documented, and the
#: same for both lookbacks so the L comparison is not confounded by a
#: different model. Deliberately unremarkable: this is a baseline, and
#: tuning it per lookback or per condition would make later comparisons a
#: comparison of tuning effort.
@dataclass(frozen=True)
class XGBParams:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    max_bin: int = 256
    seed: int = 0

    def booster_params(self) -> dict:
        """Parameters for `xgboost.train`, excluding the round count."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "max_bin": self.max_bin,
            "seed": self.seed,
        }


@dataclass
class TrainingData:
    """The pooled training matrix, with the provenance of every row."""

    matrix: np.ndarray
    labels: np.ndarray
    #: Per-row weight: a positive carries its own window's
    #: n_negative / n_positive, a negative carries 1.0. This is the design's
    #: "class-weighted loss with weight n_negative/n_positive per window",
    #: applied per window rather than pooled, so a quiet window's positives
    #: are not drowned by a busy one's.
    weights: np.ndarray
    feature_names: list[str]
    spec: features.FeatureSpec
    times: list[pd.Timestamp]
    per_window: list[dict]
    #: Columns a variant removed, kept so the record survives slicing once
    #: and reusing the result across seeds.
    dropped: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if self.n_rows else 0.0

    def memory_mb(self) -> float:
        return float(
            self.matrix.nbytes + self.labels.nbytes + self.weights.nbytes
        ) / 1e6

    def for_variant(self, variant: str) -> "TrainingData":
        """The same pooled windows with a variant's columns removed.

        Built as a view onto one matrix rather than by featurising twice:
        the windows, labels and weights are identical by construction, which
        is exactly what the ablation needs to hold fixed. On the full data
        this also saves re-reading 6.7M rows of features per variant.
        """
        prefixes = MODEL_VARIANTS[variant]
        if not prefixes:
            return self
        mask = keep_mask(self.feature_names, prefixes)
        if mask.all():
            return self  # already sliced: reused across seeds without a copy
        return TrainingData(
            matrix=self.matrix[:, mask],
            labels=self.labels,
            weights=self.weights,
            feature_names=[
                name for name, keep in zip(self.feature_names, mask) if keep
            ],
            spec=self.spec,
            times=self.times,
            per_window=self.per_window,
            dropped=dropped_columns(self.feature_names, prefixes),
        )


def build_training_data(
    full_df: pd.DataFrame,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    spec: features.FeatureSpec | None = None,
) -> TrainingData:
    """Pool every rule-3-valid training window into one matrix.

    The feature spec is fitted on these same windows unless one is passed
    in, and `fit_feature_spec` refuses anything that is not a training
    window, so a spec can never be fitted on validation or test data.
    """
    times = windows.training_detection_times(lookback_h)
    if not times:
        raise ValueError(
            f"no training detection times with a full {lookback_h}h lookback"
        )
    train_windows = list(windows.iter_windows(full_df, times, lookback_h))
    if not train_windows:
        raise ValueError("every training window is empty: nothing to train on")

    spec = spec or features.fit_feature_spec(train_windows, lookback_h)

    matrices, labels, weights, per_window = [], [], [], []
    feature_names: list[str] = []
    for window in train_windows:
        built = features.build_window_features(window, spec)
        matrix, names = built.xgboost_matrix()
        feature_names = names
        weight = built.class_weight()
        row_weights = np.where(built.labels == 1, weight, 1.0)

        matrices.append(matrix)
        labels.append(built.labels)
        weights.append(row_weights)
        per_window.append(
            {
                "t": str(window.t),
                "n_edges": built.n_edges,
                "n_positive": int(built.labels.sum()),
                "positive_rate": round(built.positive_rate, 6),
                "class_weight": round(weight, 2),
            }
        )

    return TrainingData(
        matrix=np.vstack(matrices).astype(np.float32, copy=False),
        labels=np.concatenate(labels),
        weights=np.concatenate(weights).astype(np.float32, copy=False),
        feature_names=feature_names,
        spec=spec,
        times=times,
        per_window=per_window,
    )


@dataclass
class TrainedBaseline:
    """A trained booster plus everything needed to score a window with it."""

    booster: object  # xgboost.Booster
    spec: features.FeatureSpec
    lookback_h: int
    feature_names: list[str]
    params: XGBParams
    variant: str = "xgboost_centralized"
    training: dict = field(default_factory=dict)

    @property
    def dropped_prefixes(self) -> tuple[str, ...]:
        return MODEL_VARIANTS[self.variant]

    def scorer(self):
        """A `scorer(window_transactions, t)` for `evaluation.evaluate_split`.

        Returns one probability per transaction, in the window's own row
        order, because that is the order the cluster extraction pairs with
        `src`/`dst`. The scorer is handed only the window, so it cannot see
        past `t` even by accident.
        """
        import xgboost as xgb

        def score(window_transactions: pd.DataFrame, t) -> np.ndarray:
            if len(window_transactions) == 0:
                return np.zeros(0, dtype=np.float64)
            window = windows.Window(
                t=pd.Timestamp(t),
                lookback_h=self.lookback_h,
                split=windows.split_of_detection_time(t),
                transactions=window_transactions,
            )
            built = features.build_window_features(window, self.spec)
            matrix, names = built.xgboost_matrix()
            if self.dropped_prefixes:
                mask = keep_mask(names, self.dropped_prefixes)
                matrix = matrix[:, mask]
                names = [n for n, keep in zip(names, mask) if keep]
            if names != self.feature_names:
                raise ValueError(
                    "feature names at scoring time differ from training; the "
                    "spec and the model do not belong to each other"
                )
            scores = self.booster.predict(
                xgb.DMatrix(matrix, feature_names=self.feature_names)
            )
            return np.asarray(scores, dtype=np.float64)

        return score

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "xgb_model.json"))
        self.spec.save(directory / "feature_spec.json")
        (directory / "model_meta.json").write_text(
            json.dumps(
                {
                    "variant": self.variant,
                    "lookback_h": self.lookback_h,
                    "feature_names": self.feature_names,
                    "params": asdict(self.params),
                    "training": self.training,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "TrainedBaseline":
        import xgboost as xgb

        directory = Path(directory)
        booster = xgb.Booster()
        booster.load_model(str(directory / "xgb_model.json"))
        meta = json.loads((directory / "model_meta.json").read_text(encoding="utf-8"))
        return cls(
            booster=booster,
            spec=features.FeatureSpec.load(directory / "feature_spec.json"),
            lookback_h=int(meta["lookback_h"]),
            feature_names=list(meta["feature_names"]),
            params=XGBParams(**meta["params"]),
            variant=meta.get("variant", "xgboost_centralized"),
            training=meta.get("training", {}),
        )


def train_baseline(
    full_df: pd.DataFrame,
    lookback_h: int = constants.DEFAULT_LOOKBACK_H,
    params: XGBParams | None = None,
    spec: features.FeatureSpec | None = None,
    data: TrainingData | None = None,
    variant: str = "xgboost_centralized",
) -> TrainedBaseline:
    """Train one booster, for one lookback, one variant and one seed."""
    import xgboost as xgb

    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(MODEL_VARIANTS)}"
        )
    params = params or XGBParams()
    data = data or build_training_data(full_df, lookback_h, spec)
    full_names = list(data.feature_names)
    previously_dropped = list(data.dropped)
    data = data.for_variant(variant)

    dtrain = xgb.DMatrix(
        data.matrix,
        label=data.labels,
        weight=data.weights,
        feature_names=data.feature_names,
    )
    booster = xgb.train(
        params.booster_params(),
        dtrain,
        num_boost_round=params.n_estimators,
    )
    return TrainedBaseline(
        booster=booster,
        spec=data.spec,
        lookback_h=lookback_h,
        feature_names=data.feature_names,
        params=params,
        variant=variant,
        training={
            "variant": variant,
            "dropped_columns": (
                data.dropped
                or previously_dropped
                or dropped_columns(full_names, MODEL_VARIANTS[variant])
            ),
            "n_features": len(data.feature_names),
            "n_windows": len(data.times),
            "detection_times": [str(t) for t in data.times],
            "n_rows": data.n_rows,
            "n_positive": data.n_positive,
            "positive_rate": round(data.positive_rate, 6),
            "matrix_memory_mb": round(data.memory_mb(), 1),
            "per_window": data.per_window,
            "xgboost_version": xgb.__version__,
        },
    )


def feature_importance(model: TrainedBaseline, top: int = 15) -> list[dict]:
    """Gain-based importance, for the report only.

    Never used to select features: doing that on anything but training
    windows would leak, and doing it at all would make the baseline's
    feature set differ from the GNN's, which is what the comparison holds
    fixed.
    """
    scores = model.booster.get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(scores.values()) or 1.0
    return [
        {"feature": name, "gain": round(gain, 2), "share": round(gain / total, 4)}
        for name, gain in ranked[:top]
    ]
