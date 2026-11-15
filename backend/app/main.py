"""FastAPI app serving the exported artifacts.

Never loads the raw dataset — everything comes from backend/artifacts/, read
once at startup and cached. See README "API reference".

**One router computes rather than serves:** `/demo/*`, the live panel, scores a
cached demo window with the stage 4.1 boosters and extracts clusters on every
request. It still never opens the dataset — it reads the small bundle
`scripts/make_demo_bundle.py` wrote — and every quantity it returns comes from
`fraudcamp`'s own functions, so it cannot show a number the research code
would not agree with. It 404s with an explanation when the bundle is absent.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import campaigns, demo, meta, results, visibility
from app.services.artifacts import ArtifactStore, default_artifacts_dir
from app.services.demo import load_bundle

FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

DESCRIPTION = """
Read-only API over the precomputed research artifacts for *Early Discovery
of Emerging Fraud Campaigns from Fragmented Cross-Institution Evidence*.

The research pipeline is the single source of truth: every number here was
computed by `fraudcamp` and exported to disk. This server does no analysis
of its own and never reads the raw transaction dataset.
"""


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="Fraud Campaign Discovery API",
        description=DESCRIPTION,
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    root = Path(artifacts_dir or default_artifacts_dir())
    app.state.store = ArtifactStore(root).load()
    # The live demo bundle is optional: without it `/demo/*` 404s with an
    # explanation and the panel shows an empty state, which is what the
    # frontend rule requires instead of placeholder numbers.
    app.state.demo = load_bundle(root / "demo_panel")
    app.state.demo_compare_csv = root / "phase4_dev" / "phase4_compare.csv"

    app.include_router(meta.router, prefix="/api")
    app.include_router(campaigns.router, prefix="/api")
    app.include_router(visibility.router, prefix="/api")
    app.include_router(results.router, prefix="/api")
    app.include_router(demo.router, prefix="/api")
    return app


app = create_app()
