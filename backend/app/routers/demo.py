"""/demo/* — the live panel: input -> output, recomputed on every request.

The one part of this API that computes rather than serves. It is kept separate
and labelled for that reason, and it still never reads the raw dataset: it
reads the bundle `scripts/make_demo_bundle.py` wrote. Everything it computes
comes from `fraudcamp`'s own functions - clusters, the frozen matching rule,
purity and coverage - so the panel cannot show a number the research code
would not agree with.

404 with an explanation when the bundle is absent, so the frontend shows an
empty state rather than placeholder data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.services import demo as demo_service

router = APIRouter()

NO_BUNDLE = (
    "The live demo bundle is not installed. Build it with "
    "`python scripts/make_demo_bundle.py --config configs/local.yaml`, which "
    "writes artifacts/demo_panel/."
)


def get_demo(request: Request) -> demo_service.DemoBundle:
    bundle = getattr(request.app.state, "demo", None)
    if bundle is None or not bundle.available:
        detail = NO_BUNDLE if bundle is None else (bundle.reason or NO_BUNDLE)
        raise HTTPException(status_code=404, detail=detail)
    return bundle


@router.get("/demo/manifest", tags=["demo"])
def manifest(bundle: demo_service.DemoBundle = Depends(get_demo)) -> dict:
    """What the panel can show, and the provenance a viewer needs.

    Includes the campaign-selection rule verbatim and its disclaimer, because
    the panel illustrates a mechanism on one campaign and must never be read
    as a representative result.
    """
    m = bundle.manifest
    return {
        "visibility_target": m.get("visibility_target"),
        "seed": m.get("seed"),
        "lookback_h": m.get("lookback_h"),
        "split": m.get("split"),
        "k_institutions": bundle.k,
        "institutions_with_a_model": sorted(bundle.boosters),
        "demo_campaign": {
            "campaign_id": m.get("demo_campaign_id"),
            "accounts": m.get("demo_campaign_accounts", []),
            "institutions_holding": m.get("demo_campaign_institutions", []),
            "achieved_visibility": m.get("achieved_visibility"),
        },
        "selection_rule": m.get("campaign_selection", {}).get("rule", []),
        "selection_disclaimer": m.get("campaign_selection", {}).get("DISCLAIMER"),
        "selection_audit": {
            "rank_chosen": (
                m.get("campaign_selection", {}).get("chosen", {}) or {}
            ).get("rank_in_ranking"),
            "n_candidates": m.get("campaign_selection", {}).get("n_candidates"),
            "n_skipped": len(m.get("campaign_selection", {}).get("skipped", [])),
        },
        "windows": m.get("windows", []),
        "detection_times": [str(t) for t in bundle.detection_times()],
        "primary_points": m.get("primary_points", []),
        "taus": m.get("taus", {}),
        "match_rule": m.get("match_rule", {}),
        "models_mb": m.get("models_mb"),
        "modes": list(demo_service.MODES),
        "development_only": m.get("DEVELOPMENT_ONLY_IF_SAMPLE"),
        "generated": m.get("generated"),
    }


@router.get("/demo/run", tags=["demo"])
def run(
    detection_time: str = Query(..., description="one of /demo/manifest's detection_times"),
    institution: int = Query(..., ge=0, description="whose view is being scored"),
    mode: str = Query(
        demo_service.MODE_OWN,
        description="'own' = this institution's model; 'averaged' = all K "
                    "models applied to this view and averaged (the 4.2 rule)",
    ),
    operating_point: str = Query(
        "total_10", description="which recorded tau to use"
    ),
    tau: float | None = Query(None, description="override the recorded tau"),
    bundle: demo_service.DemoBundle = Depends(get_demo),
) -> dict:
    """Score the view, extract clusters, classify them. Live on every call."""
    if mode not in demo_service.MODES:
        raise HTTPException(
            status_code=422,
            detail=f"mode must be one of {list(demo_service.MODES)}",
        )
    if institution not in bundle.boosters:
        raise HTTPException(
            status_code=404,
            detail=f"no booster for institution {institution}; the bundle has "
                   f"{sorted(bundle.boosters)}",
        )
    available = {str(t) for t in bundle.detection_times()}
    if detection_time not in available:
        raise HTTPException(
            status_code=404,
            detail=f"detection_time {detection_time!r} is not in the bundle; "
                   f"it has {sorted(available)}",
        )
    chosen_tau = tau
    tau_source = "override"
    if chosen_tau is None:
        chosen_tau = bundle.tau_for(institution, operating_point)
        tau_source = f"recorded {operating_point}"
    if chosen_tau is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no recorded tau for institution {institution} at "
                f"{operating_point!r}, and none supplied. Run the evaluation "
                "that writes phase4_isolated_taus.csv, or pass ?tau="
            ),
        )
    try:
        out = bundle.run(detection_time, institution, mode, float(chosen_tau))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["tau_source"] = tau_source
    out["operating_point"] = operating_point
    return out


@router.get("/demo/aggregate", tags=["demo"])
def aggregate(request: Request) -> dict:
    """The aggregate figures that must accompany the panel.

    The panel shows the mechanism on one campaign; this says how often it
    holds. Read from the comparison artifact, never recomputed here, and an
    explicit empty state when there is none.
    """
    path = getattr(request.app.state, "demo_compare_csv", None)
    if path is None:
        raise HTTPException(status_code=404, detail=NO_BUNDLE)
    return demo_service.aggregate_table(path)
