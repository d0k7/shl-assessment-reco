from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """
    Normalize SHL URLs so duplicates don't break matching.
    - force https
    - strip fragments and query
    - strip trailing slash
    """
    url = (url or "").strip()
    if not url:
        return url

    p = urlparse(url)
    scheme = "https"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", "", ""))
