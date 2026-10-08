"""Loads the exported artifacts once at startup and caches them.

The server never touches the raw dataset — everything it serves comes from
the small files written by `fraudcamp.export`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, Request

from fraudcamp import constants

SUMMARY = "summary.json"
CAMPAIGNS = "campaigns.parquet"
CAMPAIGN_TRANSACTIONS = "campaign_transactions.parquet"
INSTITUTIONS = "institutions.json"
RESULTS_RUNS = "results/runs.parquet"
RESULTS_DETECTIONS = "results/detections.parquet"

CORE_ARTIFACTS = [SUMMARY, CAMPAIGNS, CAMPAIGN_TRANSACTIONS, INSTITUTIONS]


def default_artifacts_dir() -> Path:
    """FRAUDCAMP_ARTIFACTS_DIR if set, else backend/artifacts."""
    env = os.environ.get("FRAUDCAMP_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / "artifacts"


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe dicts (numpy scalars unwrapped, NaN/NaT -> None)."""
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if value is None or value is pd.NaT:
                clean[key] = None
            elif isinstance(value, pd.Timestamp):
                clean[key] = value.to_pydatetime()
            elif isinstance(value, float) and math.isnan(value):
                clean[key] = None
            elif hasattr(value, "item"):  # numpy scalar
                clean[key] = value.item()
            else:
                clean[key] = value
        out.append(clean)
    return out


class ArtifactStore:
    """Holds every artifact in memory. Missing files are tolerated so that
    /health can report what is present and each endpoint can fail with a
    clear message."""

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.summary: dict | None = None
        self.campaigns: pd.DataFrame | None = None
        self.campaign_transactions: pd.DataFrame | None = None
        self.institutions: dict | None = None
        self.visibility: dict[int, pd.DataFrame] = {}
        self.results_runs: pd.DataFrame | None = None
        self.results_detections: pd.DataFrame | None = None

    def load(self) -> "ArtifactStore":
        d = self.artifacts_dir
        if (d / SUMMARY).exists():
            self.summary = json.loads((d / SUMMARY).read_text(encoding="utf-8"))
        if (d / CAMPAIGNS).exists():
            self.campaigns = pd.read_parquet(d / CAMPAIGNS)
        if (d / CAMPAIGN_TRANSACTIONS).exists():
            self.campaign_transactions = pd.read_parquet(d / CAMPAIGN_TRANSACTIONS)
        if (d / INSTITUTIONS).exists():
            self.institutions = json.loads((d / INSTITUTIONS).read_text(encoding="utf-8"))
        for seed in constants.VISIBILITY_SEEDS:
            path = d / "visibility" / f"seed_{seed}.parquet"
            if path.exists():
                self.visibility[seed] = pd.read_parquet(path)
        if (d / RESULTS_RUNS).exists():
            self.results_runs = pd.read_parquet(d / RESULTS_RUNS)
        if (d / RESULTS_DETECTIONS).exists():
            self.results_detections = pd.read_parquet(d / RESULTS_DETECTIONS)
        return self

    # -- availability ------------------------------------------------------

    def present(self) -> dict[str, bool]:
        d = self.artifacts_dir
        found = {name: (d / name).exists() for name in CORE_ARTIFACTS}
        for seed in constants.VISIBILITY_SEEDS:
            found[f"visibility/seed_{seed}.parquet"] = (
                d / "visibility" / f"seed_{seed}.parquet"
            ).exists()
        found[RESULTS_RUNS] = (d / RESULTS_RUNS).exists()
        found[RESULTS_DETECTIONS] = (d / RESULTS_DETECTIONS).exists()
        return found

    @property
    def results_available(self) -> bool:
        return self.results_runs is not None or self.results_detections is not None

    # -- guarded accessors -------------------------------------------------

    def require_summary(self) -> dict:
        if self.summary is None:
            raise _not_exported(SUMMARY)
        return self.summary

    def require_campaigns(self) -> pd.DataFrame:
        if self.campaigns is None:
            raise _not_exported(CAMPAIGNS)
        return self.campaigns

    def require_campaign_transactions(self) -> pd.DataFrame:
        if self.campaign_transactions is None:
            raise _not_exported(CAMPAIGN_TRANSACTIONS)
        return self.campaign_transactions

    def require_institutions(self) -> dict:
        if self.institutions is None:
            raise _not_exported(INSTITUTIONS)
        return self.institutions

    def require_visibility(self, seed: int) -> pd.DataFrame:
        if seed not in self.visibility:
            raise _not_exported(f"visibility/seed_{seed}.parquet")
        return self.visibility[seed]

    def require_campaign(self, campaign_id: int) -> dict:
        campaigns = self.require_campaigns()
        row = campaigns[campaigns["campaign_id"] == campaign_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found.")
        return to_records(row)[0]


def _not_exported(name: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            f"Artifact '{name}' has not been exported yet. Run "
            "`python scripts/run_pipeline.py --config configs/local.yaml --export`."
        ),
    )


def get_store(request: Request) -> ArtifactStore:
    return request.app.state.store
