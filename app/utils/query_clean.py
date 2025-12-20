from __future__ import annotations

import re

_STOP_PATTERNS = [
    r"equal opportunity employer.*",
    r"we are an equal opportunity.*",
    r"all qualified applicants.*",
    r"accommodation.*disability.*",
    r"privacy policy.*",
]


def clean_query(text: str) -> str:
    """
    Light cleaning for JDs:
    - remove excessive whitespace
    - remove common boilerplate fragments
    - keep content length manageable
    """
    if not text:
        return ""

    t = text.strip()

    # collapse whitespace
    t = re.sub(r"\s+", " ", t)

    # drop boilerplate
    lower = t.lower()
    for pat in _STOP_PATTERNS:
        m = re.search(pat, lower, flags=re.IGNORECASE)
        if m:
            # remove that segment onwards
            t = t[: m.start()].strip()
            lower = t.lower()

    # clamp to avoid super long embeddings
    if len(t) > 2000:
        t = t[:2000]

    return t


def extract_keywords(text: str) -> str:
    """
    Simple keyword signal for embeddings (no LLM needed).
    Pulls out:
      - technologies (java/python/sql/etc.)
      - role words (manager, analyst, coo, marketing, sales)
    """
    if not text:
        return ""

    t = text.lower()

    tech = re.findall(
        r"\b(java|python|sql|javascript|node\.js|react|aws|azure|gcp|docker|kubernetes|spark|etl|ml|ai)\b",
        t,
    )
    roles = re.findall(
        r"\b(coo|cto|ceo|marketing manager|product manager|consultant|analyst|sales|customer service|engineer|developer)\b",
        t,
    )

    kws = list(dict.fromkeys(tech + roles))  # unique, keep order
    return "Keywords: " + ", ".join(kws) if kws else ""
