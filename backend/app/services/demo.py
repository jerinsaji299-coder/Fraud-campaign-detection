"""The live demo panel: input -> output, computed on the spot.

**What is live and what is cached.** The featurised windows are cached, so a
toggle flip runs only the **scaler, the predict calls and the connected
components** - which is the whole detection path the research code uses, not a
reimplementation of it. Nothing is replayed: flipping the toggle really does
re-score the window with a different set of boosters and re-extract the
clusters.

**The API's no-raw-data rule still holds.** This service reads only the small
bundle `scripts/make_demo_bundle.py` wrote - the offered windows'
transactions, the ground truth the frozen matching rule needs, the account ->
institution map, and the eight boosters. It never opens the dataset, and it
computes nothing the research code does not already define: clusters come from
`evaluation.clusters.extract_clusters`, the match decision and the purity and
coverage from `evaluation.clusters.cluster_match_detail`, the classification
from `classify_cluster`. The panel displays those numbers; it does not invent
a second definition of any of them.

If the bundle is absent every endpoint says so and the panel shows an empty
state, which is what the frontend rule requires instead of placeholder data.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from fraudcamp import constants, features, windows
from fraudcamp.evaluation import clusters as cluster_tools
from fraudcamp.models import xgb_baseline

MANIFEST = "demo_manifest.json"
WINDOWS = "demo_windows.parquet"
TRUTH = "demo_truth.json"
INSTITUTIONS = "demo_institutions.json"

#: How many (detection_time, institution) featurisations to keep. Each holds
#: one raw edge matrix and one raw node matrix for one view, which is a few MB
#: on the sample and tens on the full data. Four covers a user flipping
#: between two institutions on two windows without recomputing.
MAX_CACHED_VIEWS = 4

MODE_OWN = "own"
MODE_AVERAGED = "averaged"
MODES = (MODE_OWN, MODE_AVERAGED)


class DemoBundle:
    """The bundle on disk, loaded once, plus the featurisation cache."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.available = False
        self.reason: str | None = None
        self.manifest: dict = {}
        self.windows: pd.DataFrame = pd.DataFrame()
        self.truth: dict = {}
        self.account_institution: dict[str, int] = {}
        self.k = constants.K_INSTITUTIONS
        self.boosters: dict[int, xgb_baseline.TrainedBaseline] = {}
        self._views: OrderedDict = OrderedDict()
        self._account_index: dict[str, list[int]] = {}
        self._accounts_by_campaign: dict[int, tuple[str, ...]] = {}

    def load(self) -> "DemoBundle":
        missing = [
            name for name in (MANIFEST, WINDOWS, TRUTH, INSTITUTIONS)
            if not (self.directory / name).exists()
        ]
        if missing:
            self.reason = (
                f"no demo bundle in {self.directory}: missing {missing}. "
                "Build it with `python scripts/make_demo_bundle.py "
                "--config configs/local.yaml`."
            )
            return self
        self.manifest = json.loads(
            (self.directory / MANIFEST).read_text(encoding="utf-8")
        )
        self.windows = pd.read_parquet(self.directory / WINDOWS)
        self.truth = json.loads(
            (self.directory / TRUTH).read_text(encoding="utf-8")
        )
        institutions = json.loads(
            (self.directory / INSTITUTIONS).read_text(encoding="utf-8")
        )
        self.account_institution = institutions["account_institution"]
        self.k = int(institutions.get("k", constants.K_INSTITUTIONS))

        self._accounts_by_campaign = {
            int(cid): tuple(accounts)
            for cid, accounts in self.truth["campaign_accounts"].items()
        }
        self._account_index = cluster_tools.build_account_index(
            self._accounts_by_campaign
        )

        models_dir = self.directory / "models"
        for institution in range(self.k):
            path = models_dir / f"i{institution}"
            if (path / "xgb_model.json").exists():
                self.boosters[institution] = xgb_baseline.TrainedBaseline.load(path)
        if not self.boosters:
            self.reason = f"the bundle at {self.directory} holds no boosters"
            return self
        self.available = True
        return self

    # --- the view, featurised once ------------------------------------
    def detection_times(self) -> list[pd.Timestamp]:
        return [pd.Timestamp(t) for t in sorted(
            pd.unique(self.windows["detection_time"])
        )]

    def _window_at(self, t: pd.Timestamp) -> pd.DataFrame:
        return self.windows[self.windows["detection_time"] == pd.Timestamp(t)]

    def _visible_mask(self, frame: pd.DataFrame, institution: int) -> np.ndarray:
        """The frozen endpoint rule, applied through the exported map.

        The map was written by `views.visible_mask`'s own assignment, so this
        is the same rule and not a second one; it is re-expressed here only
        because the server does not hold a `VisibilityAssignment`.
        """
        lookup = self.account_institution
        src = frame["src"].map(lookup).to_numpy()
        dst = frame["dst"].map(lookup).to_numpy()
        return (src == institution) | (dst == institution)

    def _view_features(self, t: pd.Timestamp, institution: int):
        """Raw features for one institution's view of one window, cached.

        Raw rather than scaled: the raw parts are what every booster shares,
        and applying the eight scalers is cheap. Caching these makes a toggle
        flip cost the scalers, the predicts and the components, and nothing
        else.
        """
        key = (pd.Timestamp(t), int(institution))
        cached = self._views.get(key)
        if cached is not None:
            self._views.move_to_end(key)
            return cached
        frame = self._window_at(t)
        mask = self._visible_mask(frame, institution)
        visible = frame[mask]
        if len(visible) == 0:
            value = (visible, mask, None)
        else:
            window = xgb_baseline.as_window(
                visible, t, self.manifest.get("lookback_h", constants.PHASE4_LOOKBACK_H)
            )
            reference = self.boosters[sorted(self.boosters)[0]].spec
            value = (visible, mask, features._raw_parts(window, reference))
        self._views[key] = value
        while len(self._views) > MAX_CACHED_VIEWS:
            self._views.popitem(last=False)
        return value

    # --- the live path -------------------------------------------------
    def run(
        self,
        t: pd.Timestamp,
        institution: int,
        mode: str,
        tau: float,
    ) -> dict:
        """Score, cluster, classify. The whole panel output for one setting."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if institution not in self.boosters:
            raise ValueError(f"no booster for institution {institution}")

        t = pd.Timestamp(t)
        started = time.perf_counter()
        # Asked BEFORE the call, because `_view_features` inserts the key: a
        # flag read afterwards is always True and reports nothing.
        was_cached = (pd.Timestamp(t), int(institution)) in self._views
        visible, mask, parts = self._view_features(t, institution)
        t_after_features = time.perf_counter()

        if parts is None:
            return {
                "detection_time": str(t), "institution": institution,
                "mode": mode, "tau": tau,
                "n_edges_in_window": int(len(self._window_at(t))),
                "n_edges_in_view": 0,
                "flagged_edges": [], "n_flagged_edges": 0,
                "clusters": [], "n_clusters": 0, "alert_count": 0,
                "empty_reason": "this institution sees nothing in this window",
                "latency_ms": {},
            }

        used = (
            [institution] if mode == MODE_OWN else sorted(self.boosters)
        )
        scores_per_model = []
        for index in used:
            booster = self.boosters[index]
            built = features._scaled(
                xgb_baseline.as_window(
                    visible, t,
                    self.manifest.get("lookback_h", constants.PHASE4_LOOKBACK_H),
                ),
                booster.spec, parts,
            )
            scores_per_model.append(booster.predict_built(built))
        stacked = np.vstack([s.reshape(1, -1) for s in scores_per_model])
        scores = stacked.mean(axis=0)
        t_after_predict = time.perf_counter()

        found = cluster_tools.extract_clusters(
            visible["src"].tolist(), visible["dst"].tolist(), scores, tau
        )
        t_after_clusters = time.perf_counter()

        demo_campaign = self.manifest.get("demo_campaign_id")
        campaign_splits = {
            int(cid): split for cid, split in self.truth["campaign_split"].items()
        }
        unassigned = set(self.truth.get("unassigned_laundering_accounts", []))
        split = self.manifest.get("split", "test")

        clusters = []
        for cluster in found:
            kind, hit_ids = cluster_tools.classify_cluster(
                cluster, split, self._accounts_by_campaign, campaign_splits,
                unassigned, self._account_index,
            )
            detail = None
            if demo_campaign is not None and int(demo_campaign) in self._accounts_by_campaign:
                detail = cluster_tools.cluster_match_detail(
                    cluster, self._accounts_by_campaign[int(demo_campaign)]
                )
            clusters.append({
                "accounts": sorted(cluster),
                "n_accounts": len(cluster),
                "kind": kind,
                "hit_campaigns": hit_ids,
                "matches_demo_campaign": bool(
                    demo_campaign is not None and int(demo_campaign) in hit_ids
                ),
                "vs_demo_campaign": detail,
                "institutions": sorted({
                    int(self.account_institution.get(a, -1)) for a in cluster
                }),
            })

        flagged = np.flatnonzero(scores >= tau)
        flagged_edges = [
            {
                "src": visible.iloc[int(i)]["src"],
                "dst": visible.iloc[int(i)]["dst"],
                "score": float(scores[int(i)]),
                "ts": str(visible.iloc[int(i)]["ts"]),
                "is_laundering": int(visible.iloc[int(i)]["Is Laundering"]),
                "campaign_id": int(visible.iloc[int(i)]["campaign_id"]),
            }
            for i in flagged[:400]
        ]

        lead = self._lead_time(t, demo_campaign, clusters)
        return {
            "detection_time": str(t),
            "institution": institution,
            "mode": mode,
            "models_used": used,
            "n_models_used": len(used),
            "tau": tau,
            "n_edges_in_window": int(len(self._window_at(t))),
            "n_edges_in_view": int(len(visible)),
            "n_flagged_edges": int(len(flagged)),
            "flagged_edges": flagged_edges,
            "flagged_edges_truncated": bool(len(flagged) > 400),
            "clusters": clusters,
            "n_clusters": len(found),
            "alert_count": len(found),
            "cluster_kinds": {
                kind: sum(1 for c in clusters if c["kind"] == kind)
                for kind in constants.HIT_KINDS
            },
            "demo_campaign_id": demo_campaign,
            "demo_campaign_detected": any(
                c["matches_demo_campaign"] for c in clusters
            ),
            "best_purity_vs_demo_campaign": max(
                (c["vs_demo_campaign"]["purity"] for c in clusters
                 if c["vs_demo_campaign"]), default=None
            ),
            "best_coverage_vs_demo_campaign": max(
                (c["vs_demo_campaign"]["coverage_fraction"] for c in clusters
                 if c["vs_demo_campaign"]), default=None
            ),
            **lead,
            "latency_ms": {
                "featurise": round((t_after_features - started) * 1000, 1),
                "featurise_was_cached": bool(was_cached),
                "predict": round((t_after_predict - t_after_features) * 1000, 1),
                "clusters": round((t_after_clusters - t_after_predict) * 1000, 1),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
        }

    def _lead_time(self, t, demo_campaign, clusters) -> dict:
        """Hours between this detection time and the campaign's deadline/end.

        Reported only when a cluster actually matched the demo campaign, and
        only against the recorded deadline and end - the same two timestamps
        the research lead time uses. A detection after the deadline is
        reported with a negative margin rather than hidden.
        """
        if demo_campaign is None:
            return {"lead_time": None}
        detected = any(c["matches_demo_campaign"] for c in clusters)
        deadline = self.truth.get("deadlines", {}).get(str(demo_campaign))
        end = self.truth.get("campaign_end", {}).get(str(demo_campaign))
        if not detected or deadline is None:
            return {
                "lead_time": {
                    "detected": detected,
                    "deadline": deadline,
                    "campaign_end": end,
                    "hours_before_deadline": None,
                    "hours_before_end": None,
                    "beat_the_deadline": None,
                }
            }
        t = pd.Timestamp(t)
        before_deadline = (pd.Timestamp(deadline) - t).total_seconds() / 3600.0
        before_end = (
            (pd.Timestamp(end) - t).total_seconds() / 3600.0 if end else None
        )
        return {
            "lead_time": {
                "detected": True,
                "deadline": deadline,
                "campaign_end": end,
                "hours_before_deadline": round(before_deadline, 2),
                "hours_before_end": (
                    round(before_end, 2) if before_end is not None else None
                ),
                "beat_the_deadline": bool(before_deadline >= 0),
            }
        }

    def tau_for(self, institution: int, operating_point: str) -> float | None:
        taus = self.manifest.get("taus", {}) or {}
        value = taus.get(operating_point, {}).get(str(institution))
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None


def load_bundle(directory: str | Path) -> DemoBundle:
    return DemoBundle(directory).load()


def aggregate_table(compare_csv: str | Path) -> dict:
    """The aggregate figures that must sit beside the panel.

    Read from the Phase 4 comparison artifact rather than recomputed. Returns
    an explicit empty state when the file is absent, because an invented
    aggregate next to a live panel is exactly the thing the frontend rule
    forbids.
    """
    path = Path(compare_csv)
    if not path.exists():
        return {
            "available": False,
            "reason": (
                f"no comparison artifact at {path}. The panel shows the "
                "mechanism on one campaign; the aggregate says how often it "
                "holds, and it must come from an evaluation run."
            ),
            "rows": [],
        }
    frame = pd.read_csv(path)
    keep = frame[
        (frame.get("eval_split") == "test")
        & (frame.get("test_subset") == "all")
        & (frame.get("metric") == "recall_relative_to_ceiling")
    ] if len(frame) else frame
    return {
        "available": bool(len(keep)),
        "reason": None if len(keep) else "the comparison artifact has no test rows",
        "rows": keep.to_dict("records") if len(keep) else [],
        "source": str(path),
    }


__all__ = [
    "MAX_CACHED_VIEWS",
    "MODES",
    "MODE_AVERAGED",
    "MODE_OWN",
    "DemoBundle",
    "aggregate_table",
    "load_bundle",
]
