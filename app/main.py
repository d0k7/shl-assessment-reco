from __future__ import annotations

import os
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.index.search_index import load_artifacts, recommend_from_query
from app.schemas.api import RecommendRequest, RecommendResponse, HealthResponse

app = FastAPI(
    title="SHL Assessment Recommender",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for assignment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Lazy-loaded artifacts (Render-safe) --------
_artifacts: Optional[Any] = None
_artifacts_lock = Lock()


def get_artifacts() -> Any:
    """
    Lazy-load heavy artifacts (FAISS index, metadata, embedding model, BM25, etc.)
    so the server can bind to $PORT immediately on Render.
    """
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    with _artifacts_lock:
        if _artifacts is None:
            _artifacts = load_artifacts()
    return _artifacts


@app.get("/")
def read_root():
    return {"message": "Welcome to the recommendation API!"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Keep this endpoint super-lightweight (no loading artifacts here)
    return HealthResponse(status="healthy")


# Optional: many platforms use /healthz
@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse)
def recommend_items(payload: RecommendRequest) -> RecommendResponse:
    try:
        artifacts = get_artifacts()
        recs = recommend_from_query(
            query=payload.query,
            top_k=payload.top_k,
            artifacts=artifacts,
        )
        return RecommendResponse(recommended_assessments=recs)
    except Exception as e:
        # return 500 but with a helpful message in logs
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {type(e).__name__}: {e}")
