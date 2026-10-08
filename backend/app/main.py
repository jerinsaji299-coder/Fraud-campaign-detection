"""FastAPI app serving the exported artifacts.

Never loads the raw dataset — everything comes from backend/artifacts/,
read once at startup and cached. See README "API reference".
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import campaigns, meta, results, visibility
from app.services.artifacts import ArtifactStore, default_artifacts_dir

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
    app.state.store = ArtifactStore(artifacts_dir or default_artifacts_dir()).load()

    app.include_router(meta.router, prefix="/api")
    app.include_router(campaigns.router, prefix="/api")
    app.include_router(visibility.router, prefix="/api")
    app.include_router(results.router, prefix="/api")
    return app


app = create_app()
