from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.index.bm25_light import BM25Artifacts, load_bm25_artifacts, recommend_bm25
from app.schemas.api import RecommendRequest, RecommendResponse, HealthResponse

app = FastAPI(title="SHL Assessment Recommender", version="0.2.0-bm25-render")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifacts: Optional[BM25Artifacts] = None
_lock = Lock()


def get_artifacts() -> BM25Artifacts:
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    with _lock:
        if _artifacts is None:
            _artifacts = load_bm25_artifacts()
    return _artifacts


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"message": "SHL Recommender API (BM25 lightweight backend for Render)"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse, tags=["reco"])
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        artifacts = get_artifacts()
        recs = recommend_bm25(payload.query, artifacts, top_k=payload.top_k)
        # Your schema likely expects "recommended_assessments" list of items with name/url/description
        # We will map exactly:
        return RecommendResponse(recommended_assessments=recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {type(e).__name__}: {e}")
