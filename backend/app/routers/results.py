"""/results/* — 404 until Phase 5 writes them."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.artifacts import ArtifactStore, get_store, to_records

router = APIRouter()

NOT_YET = (
    "Experiment results are not available yet. They are written by Phase 5 "
    "into artifacts/results/; until then this endpoint has nothing to serve."
)


@router.get("/results/summary", tags=["results"])
def results_summary(store: ArtifactStore = Depends(get_store)) -> dict:
    if store.results_runs is None:
        raise HTTPException(status_code=404, detail=NOT_YET)
    return {"items": to_records(store.results_runs), "total": len(store.results_runs)}


@router.get("/results/detections", tags=["results"])
def results_detections(
    campaign_id: int | None = Query(None), store: ArtifactStore = Depends(get_store)
) -> dict:
    if store.results_detections is None:
        raise HTTPException(status_code=404, detail=NOT_YET)
    df = store.results_detections
    if campaign_id is not None:
        df = df[df["campaign_id"] == campaign_id]
    return {"items": to_records(df), "total": len(df)}
