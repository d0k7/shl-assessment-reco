from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.index.bm25_runtime import build_bm25_from_catalog, recommend_bm25
from app.schemas.api import RecommendRequest, RecommendResponse, HealthResponse

app = FastAPI(title="SHL Assessment Recommender", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup() -> None:
    # Render-safe: builds from data/catalog.jsonl in-memory
    app.state.artifacts = build_bm25_from_catalog()

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")

@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty")

    try:
        recs = recommend_bm25(query=query, top_k=payload.top_k, artifacts=app.state.artifacts)
        return RecommendResponse(recommended_assessments=recs)
    except Exception as e:
        # make the error visible to Streamlit
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
