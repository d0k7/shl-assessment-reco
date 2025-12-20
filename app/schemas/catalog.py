from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    name: str
    url: str
    description: Optional[str] = None

    # SHL catalog has tags like ["K"], ["P"], etc.
    test_type: Optional[List[str]] = Field(default_factory=list)

    # sometimes present
    job_levels: Optional[List[str]] = Field(default_factory=list)
    languages: Optional[List[str]] = Field(default_factory=list)
    assessment_length: Optional[str] = None
