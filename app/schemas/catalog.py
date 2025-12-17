from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, HttpUrl


class CatalogItem(BaseModel):
    """
    One SHL "Individual Test Solution" item.

    We store the fields that map cleanly to the required /recommend output schema
    (Appendix 2 in the assignment PDF).
    """
    url: HttpUrl
    name: str = Field(..., min_length=1)

    # Fields expected in the example response format
    adaptive_support: str = Field(default="No", description='Either "Yes" or "No"')
    remote_support: str = Field(default="No", description='Either "Yes" or "No"')
    duration: int = Field(default=0, ge=0, description="Duration in minutes (0 if unknown)")
    description: str = Field(default="")
    test_type: List[str] = Field(default_factory=list, description="Array of strings")

    # Helpful metadata
    source: str = Field(default="shl")
