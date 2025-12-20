from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, HttpUrl, field_validator


class HealthResponse(BaseModel):
    status: str


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query OR JD text OR a URL pointing to a JD")
    top_k: int = Field(default=10, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class RecommendedAssessment(BaseModel):
    name: str
    url: HttpUrl
    description: str = ""
    duration: int = 0
    remote_support: str = "No"
    adaptive_support: str = "No"
    test_type: List[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]
