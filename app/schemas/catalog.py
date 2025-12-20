from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, HttpUrl


class CatalogItem(BaseModel):
    url: HttpUrl
    name: str = Field(..., min_length=1)

    adaptive_support: str = Field(default="No")
    remote_support: str = Field(default="No")
    duration: int = Field(default=0, ge=0)
    description: str = Field(default="")
    test_type: List[str] = Field(default_factory=list)

    source: str = Field(default="shl")
