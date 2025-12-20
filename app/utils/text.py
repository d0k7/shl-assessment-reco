from __future__ import annotations

import re
import httpx
from bs4 import BeautifulSoup

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def looks_like_url(s: str) -> bool:
    return bool(_URL_RE.match(s.strip()))


def normalize_whitespace(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fetch_url_text(url: str, *, timeout_s: float = 12.0, max_chars: int = 8000) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SHL-RecoBot/1.0)"}
    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = normalize_whitespace(text)

    if len(text) > max_chars:
        text = text[:max_chars]
    return text
