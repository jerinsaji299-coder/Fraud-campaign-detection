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
    missing = store.missing_core()

    hint: str | None = None
    if missing and store.nested_dir is not None:
        hint = (
            f"The artifacts are one directory too deep, in {store.nested_dir}. "
            f"Move its contents up into {store.artifacts_dir} so that "
            "summary.json sits directly there, then restart the server."
        )
    elif missing:
        hint = (
            f"{len(missing)} core artifact(s) missing from {store.artifacts_dir}: "
            f"{', '.join(missing)}. Run the pipeline with --export, or unpack "
            "artifacts.zip from a Kaggle run into that folder, then restart "
            "the server (artifacts are read only at startup)."
        )

    return HealthResponse(
        status="ok",
        artifacts_dir=str(store.artifacts_dir),
        artifacts=present,
        core_artifacts_available=not missing,
        results_available=store.results_available,
        missing_artifacts=missing,
        nested_artifacts_dir=str(store.nested_dir) if store.nested_dir else None,
        hint=hint,
    )


@router.get("/summary", response_model=SummaryResponse, tags=["dataset"])
def summary(store: ArtifactStore = Depends(get_store)) -> SummaryResponse:
    return SummaryResponse(**store.require_summary())


@router.get("/methodology", response_model=MethodologyResponse, tags=["meta"])
def get_methodology() -> MethodologyResponse:
    """The frozen research definitions, with the reason for each. Values
    come from the same constants module the pipeline uses."""
    return MethodologyResponse(**methodology.build_methodology())
