from __future__ import annotations

import re
from functools import lru_cache

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@lru_cache(maxsize=64)
def maybe_fetch_jd_text(query_or_url: str) -> str:
    q = (query_or_url or "").strip()
    if not q or not _URL_RE.match(q):
        return q

    # Best-effort: if deps aren't installed or request fails, just return URL string.
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return q

    try:
        resp = requests.get(q, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        # light cleanup
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines[:400])  # cap size
    except Exception:
        return q
