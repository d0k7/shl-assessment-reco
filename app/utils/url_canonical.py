from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    """
    Normalize URLs so we don't treat the same catalog item as multiple entries.

    - lowercases scheme + host
    - strips fragments
    - removes trailing slash (except root)
    - keeps query (SHL catalog URLs typically don't need it, but safe to keep)
    """
    url = (url or "").strip()
    if not url:
        return url

    p = urlparse(url)

    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()

    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # strip fragment
    fragment = ""

    return urlunparse((scheme, netloc, path, p.params, p.query, fragment))
