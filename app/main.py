from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.index.search_index import load_artifacts, recommend_from_query
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
    app.state.artifacts = load_artifacts()


@app.get("/")
def root():
    return {"message": "SHL Assessment Recommender API is running"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse)
def recommend_items(payload: RecommendRequest) -> RecommendResponse:
    recs = recommend_from_query(payload.query, payload.top_k, app.state.artifacts)
    return RecommendResponse(recommended_assessments=recs)
