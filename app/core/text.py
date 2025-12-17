from __future__ import annotations

import re
from typing import Optional

import trafilatura


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_main_text(html: str) -> str:
    """
    Extract readable main content from HTML.
    """
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    return normalize_space(text or "")


def to_yes_no(value: str) -> Optional[str]:
    """
    Normalizes various forms into "Yes"/"No". Returns None if uncertain.
    """
    v = normalize_space(value).lower()
    if not v:
        return None
    if v in {"yes", "y", "true"}:
        return "Yes"
    if v in {"no", "n", "false"}:
        return "No"
    if "yes" in v:
        return "Yes"
    if "no" in v:
        return "No"
    return None


def parse_duration_minutes(value: str) -> Optional[int]:
    """
    Extract minutes from e.g. "45 mins", "60 minutes", "Approx. 30 min".
    """
    v = normalize_space(value).lower()
    if not v:
        return None

    m = re.search(r"(\d{1,4})\s*(min|mins|minutes)\b", v)
    if m:
        return int(m.group(1))

    if re.fullmatch(r"\d{1,4}", v):
        return int(v)

    return None
