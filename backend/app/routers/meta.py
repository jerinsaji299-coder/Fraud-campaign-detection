"""/health, /summary and /methodology."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas import HealthResponse, MethodologyResponse, SummaryResponse
from app.services import methodology
from app.services.artifacts import CORE_ARTIFACTS, ArtifactStore, get_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(store: ArtifactStore = Depends(get_store)) -> HealthResponse:
    present = store.present()
    return HealthResponse(
        status="ok",
        artifacts_dir=str(store.artifacts_dir),
        artifacts=present,
        core_artifacts_available=all(present[name] for name in CORE_ARTIFACTS),
        results_available=store.results_available,
    )


@router.get("/summary", response_model=SummaryResponse, tags=["dataset"])
def summary(store: ArtifactStore = Depends(get_store)) -> SummaryResponse:
    return SummaryResponse(**store.require_summary())


@router.get("/methodology", response_model=MethodologyResponse, tags=["meta"])
def get_methodology() -> MethodologyResponse:
    """The frozen research definitions, with the reason for each. Values
    come from the same constants module the pipeline uses."""
    return MethodologyResponse(**methodology.build_methodology())
