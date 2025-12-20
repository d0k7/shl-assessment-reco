from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.index.search_index import load_artifacts, recommend_from_query
from app.schemas.api import RecommendRequest, RecommendResponse, HealthResponse

app = FastAPI(
    title="SHL Assessment Recommender",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for assignment; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Load model/index/meta once
    app.state.artifacts = load_artifacts()


@app.get("/")
def read_root():
    return {"message": "Welcome to the recommendation API!"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse)
def recommend_items(payload: RecommendRequest) -> RecommendResponse:
    artifacts = app.state.artifacts
    recs = recommend_from_query(query=payload.query, top_k=payload.top_k, artifacts=artifacts)
    return RecommendResponse(recommended_assessments=recs)
